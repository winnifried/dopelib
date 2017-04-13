/**
 *
 * Copyright (C) 2012-2014 by the DOpElib authors
 *
 * This file is part of DOpElib
 *
 * DOpElib is free software: you can redistribute it
 * and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later
 * version.
 *
 * DOpElib is distributed in the hope that it will be
 * useful, but WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 * PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * Please refer to the file LICENSE.TXT included in this distribution
 * for further information on this license.
 *
 **/
//c++ includes
#include <iostream>
#include <fstream>

//deal.ii includes
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_generator.h>

//DOpE includes
#include <include/parameterreader.h>
#include <templates/directlinearsolver.h>
#include <templates/integrator.h>
#include <basic/mol_spacetimehandler.h>
#include <problemdata/simpledirichletdata.h>
#include <container/integratordatacontainer.h>
#include <interfaces/functionalinterface.h>
#include <problemdata/noconstraints.h>

#include <templates/integratormixeddims.h> // for mixed dim opt. control
#include <templates/newtonsolvermixeddims.h> // for mixed dim opt. control

//DOpE includes for instationary problems
#include <reducedproblems/instatreducedproblem.h>
#include <templates/instat_step_newtonsolver.h>
#include <opt_algorithms/reducednewtonalgorithm.h>
#include <container/instatoptproblemcontainer.h>

//various timestepping schemes
#include <tsschemes/backward_euler_problem.h>

#include "localpde.h"
#include "localfunctional.h"
#include "functionals.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

// Define dimensions for control- and state problem
const static int DIM = 2;
const static int CDIM = 0;

#define DOFHANDLER DoFHandler
#define FE FESystem
#define CDC ElementDataContainer
#define FDC FaceDataContainer

typedef QGauss<DIM> QUADRATURE;
typedef QGauss<DIM - 1> FACEQUADRATURE;
typedef BlockSparseMatrix<double> MATRIX;
typedef BlockSparsityPattern SPARSITYPATTERN;
typedef BlockVector<double> VECTOR;

typedef FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> FUNC;

typedef OptProblemContainer<FUNC,
        LocalFunctional<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM>,
        LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
        SimpleDirichletData<VECTOR, DIM>,
        NoConstraints<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM>, SPARSITYPATTERN,
        VECTOR, CDIM, DIM> OP_BASE;

typedef StateProblem<OP_BASE, LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
        SimpleDirichletData<VECTOR, DIM>, SPARSITYPATTERN, VECTOR, DIM> PROB;

// Typedefs for timestep problem
#define TSP BackwardEulerProblem
//FIXME: This should be a reasonable dual timestepping scheme
#define DTSP BackwardEulerProblem

//typedef InstatOptProblemContainer<TSP,DTSP,FUNC,FUNC,PDE,DD,CONS,SPARSITYPATTERN, VECTOR, CDIM,DIM> OP;
typedef InstatOptProblemContainer<TSP, DTSP, FUNC,
        LocalFunctional<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM>,
        LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
        SimpleDirichletData<VECTOR, DIM>,
        NoConstraints<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM>, SPARSITYPATTERN,
        VECTOR, CDIM, DIM> OP;
#undef TSP
#undef DTSP

typedef IntegratorDataContainer<DOFHANDLER, QUADRATURE, FACEQUADRATURE, VECTOR,
        DIM> IDC;
typedef Integrator<IDC, VECTOR, double, DIM> INTEGRATOR;
//special integrator for the mixed dims
typedef IntegratorMixedDimensions<IDC, VECTOR, double, CDIM, DIM> INTEGRATORM;

typedef DirectLinearSolverWithMatrix<SPARSITYPATTERN, MATRIX, VECTOR> LINEARSOLVER;
//dummy solver for the 0d control
typedef VoidLinearSolver<VECTOR> VOIDLS;
//special newtonsolver for the mixed dims
typedef NewtonSolverMixedDimensions<INTEGRATORM, VOIDLS, VECTOR> CNLS;
typedef InstatStepNewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR> NLS;
typedef ReducedNewtonAlgorithm<OP, VECTOR> RNA;
typedef InstatReducedProblem<CNLS, NLS, INTEGRATORM, INTEGRATOR, OP, VECTOR, CDIM,
        DIM> RP;

int
main(int argc, char **argv)
{
  /**
   * In this example we show the control of the
   *  heat equation via a time-dependent right hand side
   */

  string paramfile = "dope.prm";

  if (argc == 2)
    {
      paramfile = argv[1];
    }
  else if (argc > 2)
    {
      std::cout << "Usage: " << argv[0] << " [ paramfile ] " << std::endl;
      return -1;
    }
  const int niter = 3;

  //First, declare the parameters and read them in.
  ParameterReader pr;
  RP::declare_params(pr);
  RNA::declare_params(pr);
  pr.read_parameters(paramfile);

  //Create the triangulation.
  Triangulation<DIM> triangulation;
  GridGenerator::hyper_cube(triangulation, 0., PI);
  triangulation.refine_global(2);

  //Define the Finite Elements and quadrature formulas for control and state.
  FE<DIM> control_fe(FE_Nothing<DIM>(1), 1);
  FESystem<DIM> state_fe(FE_Q<DIM>(1), 1); //Q1

  QGauss<DIM> quadrature_formula(3);
  QGauss<DIM - 1> face_quadrature_formula(3);
  IDC idc(quadrature_formula, face_quadrature_formula);

  //Define the localPDE and the functionals we are interested in.
  LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM> LPDE;
  LocalFunctional<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> LFunc;
  LocalPointFunctional<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> LPF;
  StateErrorFunctional<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> SEF;
  ControlErrorFunctional<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> CEF;

  //Time grid of [0,1] with 20 subintervalls as initial discretization.
  dealii::Triangulation<1> times;
  dealii::GridGenerator::subdivided_hyper_cube(times, 10);


  //Note that we give DOpEtypes::initial as the type of control.
  MethodOfLines_SpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR, CDIM,
                                 DIM> DOFH(triangulation, control_fe, state_fe, times,
                                           DOpEtypes::ControlType::nonstationary);

  NoConstraints<ElementDataContainer, FaceDataContainer, DOFHANDLER, VECTOR, CDIM,
                DIM> Constraints;
  OP P(LFunc, LPDE, Constraints, DOFH);

  P.AddFunctional(&LPF);
  P.AddFunctional(&SEF);
  P.AddFunctional(&CEF);

  std::vector<bool> comp_mask(1);
  comp_mask[0] = true;

  //Here we use zero boundary values
  DOpEWrapper::ZeroFunction<DIM> zf;
  SimpleDirichletData<VECTOR, DIM> DD1(zf);

  P.SetDirichletBoundaryColors(0, comp_mask, &DD1);
  P.SetInitialValues(&zf);

  RP solver(&P, DOpEtypes::VectorStorageType::fullmem, pr, idc);

  RNA Alg(&P, &solver, pr);
  ControlVector<VECTOR> q(&DOFH, DOpEtypes::VectorStorageType::fullmem);
  Alg.ReInit();

  for (int i = 0; i < niter; i++)
    {
      try
        {
          Alg.Solve(q);
        }
      catch (DOpEException &e)
        {
          std::cout
              << "Warning: During execution of `" + e.GetThrowingInstance()
              + "` the following Problem occurred!" << std::endl;
          std::cout << e.GetErrorMessage() << std::endl;
        }
      if (i != niter - 1)
        {
          //global mesh refinement
          DOFH.RefineSpaceTime();
          Alg.ReInit();
        }
    }

  return 0;
}

#undef FDC
#undef CDC
#undef FE
#undef DOFHANDLER
