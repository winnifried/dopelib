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

#include <iostream>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/point.h>

#include <opt_algorithms/reducednewtonalgorithm.h>
#include <container/optproblemcontainer.h>
#include <interfaces/functionalinterface.h>
#include <reducedproblems/statreducedproblem.h>
#include <templates/directlinearsolver.h>
#include <templates/voidlinearsolver.h>
#include <templates/integrator.h>
#include <templates/newtonsolver.h>
#include <templates/integratormixeddims.h>
#include <templates/newtonsolvermixeddims.h>
#include <include/parameterreader.h>
#include <basic/mol_spacetimehandler.h>
#include <problemdata/noconstraints.h>
#include <container/integratordatacontainer.h>

//Here we incorporate the control in the boundary
#include "localdirichletdata.h"
#include "localpde.h"
#include "localfunctional.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

#define DOFHANDLER DoFHandler
#define FE FESystem

const static int DIM = 2;
const static int CDIM = 0;

typedef QGauss<DIM> QUADRATURE;
typedef QGauss<DIM - 1> FACEQUADRATURE;

typedef BlockSparseMatrix<double> MATRIX;
typedef BlockSparsityPattern SPARSITYPATTERN;
typedef BlockVector<double> VECTOR;

#define CDC ElementDataContainer
#define FDC FaceDataContainer

typedef LocalFunctional<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> COSTFUNCTIONAL;
typedef FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> FUNCTIONALINTERFACE;

//Note that we use LocalDirichletData instead of SimpleDirichletData
typedef OptProblemContainer<FUNCTIONALINTERFACE, COSTFUNCTIONAL,
        LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
        LocalDirichletData<VECTOR, DIM>,
        NoConstraints<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM>, SPARSITYPATTERN,
        VECTOR, CDIM, DIM> OP;
typedef IntegratorDataContainer<DOFHANDLER, QUADRATURE, FACEQUADRATURE, VECTOR,
        DIM> IDC;
typedef Integrator<IDC, VECTOR, double, DIM> INTEGRATOR;
typedef IntegratorMixedDimensions<IDC, VECTOR, double, CDIM, DIM> INTEGRATORM;
typedef DirectLinearSolverWithMatrix<SPARSITYPATTERN, MATRIX, VECTOR> LINEARSOLVER;
typedef VoidLinearSolver<VECTOR> VOIDLS;
typedef NewtonSolverMixedDimensions<INTEGRATORM, VOIDLS, VECTOR> NLSM;
typedef NewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR> NLS;
typedef ReducedNewtonAlgorithm<OP, VECTOR> RNA;
typedef StatReducedProblem<NLSM, NLS, INTEGRATORM, INTEGRATOR, OP, VECTOR, CDIM,
        DIM> RP;
typedef MethodOfLines_SpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR,
        CDIM, DIM> STH;

int
main(int argc, char **argv)
{
  /**
   * Solves an optimization problem with an PDE constraint
   * and control int the dirichlet boundary values of the state.
   * The state solves the laplacian.
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

  ParameterReader pr;
  RP::declare_params(pr);
  RNA::declare_params(pr);
  pr.read_parameters(paramfile);

  const  int niter = 1;

  //Create triangulation
  Triangulation<DIM> triangulation;
  const Point<DIM> p1(0, 0), p2(1, 1);
  GridGenerator::hyper_rectangle(triangulation, p1, p2, true);
  triangulation.refine_global(5);

  //Define FE
  FE<DIM> control_fe(FE_Nothing<DIM>(1), 5); //5 Parameter
  FE<DIM> state_fe(FE_Q<DIM>(1), 2);

  //Define quad rules
  QUADRATURE quadrature_formula(2);
  FACEQUADRATURE face_quadrature_formula(2);
  IDC idc(quadrature_formula, face_quadrature_formula);

  //define the PDE and the co
  LocalPDE<CDC, FDC, DOFHANDLER, VECTOR,  DIM> LPDE;
  COSTFUNCTIONAL LFunc;

  STH DOFH(triangulation, control_fe, state_fe, DOpEtypes::stationary);

  NoConstraints<CDC, FDC, DOFHANDLER, VECTOR, CDIM,
                DIM> Constraints;

  OP P(LFunc, LPDE, Constraints, DOFH);

  std::vector<bool> comp_mask(2, true);

  LocalDirichletData<VECTOR, DIM> DD;
  P.SetDirichletBoundaryColors(0, comp_mask, &DD);
  P.SetDirichletBoundaryColors(1, comp_mask, &DD);
  P.SetDirichletBoundaryColors(2, comp_mask, &DD);
  P.SetDirichletBoundaryColors(3, comp_mask, &DD);

  RP solver(&P, DOpEtypes::VectorStorageType::fullmem, pr, idc);
  RNA Alg(&P, &solver, pr);

  //Set the initial control values:
  Vector<double> qinit(5);
  {
    qinit(0) = 0.;
    qinit(1) = 0.;
    qinit(2) = 0.;
    qinit(3) = 0.;
    qinit(4) = 1.;
  }
  Alg.ReInit();
  ControlVector<VECTOR> q(&DOFH, DOpEtypes::VectorStorageType::fullmem);
  q.GetSpacialVector() = qinit;

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
          DOFH.RefineSpace();
          Alg.ReInit();
        }
    }
  return 0;
}


#undef FDC
#undef CDC
#undef FE
#undef DOFHANDLER
