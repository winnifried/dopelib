/**
 *
 * Copyright (C) 2012 by the DOpElib authors
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
#include <base/quadrature_lib.h>
#include <base/function.h>
#include <dofs/dof_handler.h>
#include <fe/fe_q.h>
#include <fe/fe_dgp.h> //for discont. finite elements
#include <fe/fe_nothing.h>
#include <grid/tria.h>
#include <grid/grid_in.h>
#include <grid/tria_boundary_lib.h>
#include <grid/grid_generator.h>

//DOpE includes
#include "parameterreader.h"
#include "directlinearsolver.h"
#include "integrator.h"
#include "mol_spacetimehandler.h"
#include "simpledirichletdata.h"
#include "integratordatacontainer.h"
#include "newtonsolver.h"
#include "fractional_step_theta_step_newtonsolver.h"
#include "functionalinterface.h"
#include "noconstraints.h"

#include "instatreducedproblem.h"
#include "instat_step_newtonsolver.h"
#include "reducednewtonalgorithm.h"
#include "instatoptproblemcontainer.h"

#include "backward_euler_problem.h"

//Problem specific includes
#include "localpde.h"
#include "localfunctional.h"
#include "functionals.h"
#include "my_functions.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

// Define dimensions for control- and state problem
const static int DIM = 2;

#define DOFHANDLER DoFHandler
#define FE FESystem
#define CDC CellDataContainer
#define FDC FaceDataContainer

typedef QGauss<DIM> QUADRATURE;
typedef QGauss<DIM - 1> FACEQUADRATURE;
typedef BlockSparseMatrix<double> MATRIX;
typedef BlockSparsityPattern SPARSITYPATTERN;
typedef BlockVector<double> VECTOR;

typedef FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM> FUNC;

typedef OptProblemContainer<
    LocalFunctional<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM>, FUNC,
    LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
    SimpleDirichletData<VECTOR, DIM>,
    NoConstraints<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM>, SPARSITYPATTERN,
    VECTOR, DIM, DIM> OP_BASE;

typedef StateProblem<OP_BASE, LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
    SimpleDirichletData<VECTOR, DIM>, SPARSITYPATTERN, VECTOR, DIM> PROB;

#define TSP BackwardEulerProblem
//FIXME: This should be a reasonable dual timestepping scheme
#define DTSP BackwardEulerProblem

typedef InstatOptProblemContainer<TSP, DTSP, FUNC,
    LocalFunctional<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM>,
    LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
    SimpleDirichletData<VECTOR, DIM>,
    NoConstraints<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM>, SPARSITYPATTERN,
    VECTOR, DIM, DIM> OP;
#undef TSP
#undef DTSP

typedef IntegratorDataContainer<DOFHANDLER, QUADRATURE,
    FACEQUADRATURE, VECTOR, DIM> IDC;
typedef Integrator<IDC, VECTOR, double, DIM> INTEGRATOR;
typedef DirectLinearSolverWithMatrix<SPARSITYPATTERN, MATRIX, VECTOR> LINEARSOLVER;
typedef NewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR> CNLS;
typedef InstatStepNewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR> NLS;
typedef ReducedNewtonAlgorithm<OP, VECTOR> RNA;
typedef InstatReducedProblem<CNLS, NLS, INTEGRATOR, INTEGRATOR, OP, VECTOR, DIM,
    DIM> SSolver;

int
main(int argc, char **argv)
{
  /**
   * In this example we  solve a nonlinear, timedependent PDE,
   * see the documentation for more information.
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

  //First, declare the parameters and read them in.
  ParameterReader pr;
  SSolver::declare_params(pr);
  RNA::declare_params(pr);
  pr.read_parameters(paramfile);

  std::string cases = "solve";

  //Create the triangulation.
  Triangulation<DIM> triangulation;
  GridGenerator::hyper_cube(triangulation, 0., PI);

  //Define the Finite Elements and quadrature formulas for control and state.
  FESystem<DIM> control_fe(FE_Nothing<DIM>(), 1);
  FESystem<DIM> state_fe(FE_Q<DIM>(1), 1);

  QGauss<DIM> quadrature_formula(3);
  QGauss<DIM - 1> face_quadrature_formula(3);
  IDC idc(quadrature_formula, face_quadrature_formula);

  //Define the localPDE and the functionals we are interested in. Here, LFunc is a dummy necessary for the control,
  //LPF is a SpaceTimePointevaluation
  LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM> LPDE;
  LocalFunctional<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM> LFunc;
  LocalPointFunctional<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM> LPF;
  LocalPointFunctional2<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM> LPF2;

  //Time grid of [0,1]
  Triangulation<1> times;
  GridGenerator::subdivided_hyper_cube(times, 50);

  triangulation.refine_global(4);
  MethodOfLines_SpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR, DIM,
      DIM> DOFH(triangulation, control_fe, state_fe, times,
      DOpEtypes::undefined);

  NoConstraints<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM> Constraints;
  OP P(LFunc, LPDE, Constraints, DOFH);

  P.AddFunctional(&LPF);
  P.AddFunctional(&LPF2);

  std::vector<bool> comp_mask(1);
  comp_mask[0] = true;

  //Here we use zero boundary values
  DOpEWrapper::ZeroFunction<DIM> zf;
  SimpleDirichletData<VECTOR, DIM> DD1(zf);

  P.SetDirichletBoundaryColors(0, comp_mask, &DD1);

  //prepare the initial data
  InitialData initial_data;
  P.SetInitialValues(&initial_data);

  SSolver solver(&P, "fullmem", pr, idc);

  RNA Alg(&P, &solver, pr);

  try
  {

    Alg.ReInit();

    Vector<double> solution;
    ControlVector<VECTOR> q(&DOFH, "fullmem");

    Alg.SolveForward(q);

    SolutionExtractor<SSolver, VECTOR> a(solver);
    const StateVector<VECTOR> &statevec = a.GetU();

    stringstream out;
    double product = statevec * statevec;
    out << "Backward euler: u * u = " << product << std::endl;

    P.GetOutputHandler()->Write(out, 0);

  }
  catch (DOpEException &e)
  {
    std::cout
        << "Warning: During execution of `" + e.GetThrowingInstance()
            + "` the following Problem occurred!" << std::endl;
    std::cout << e.GetErrorMessage() << std::endl;
  }

  return 0;
}

#undef FDC
#undef CDC
#undef FE
#undef DOFHANDLER
