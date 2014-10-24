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
#include <base/quadrature_lib.h>
#include <base/function.h>
#include <dofs/dof_handler.h>
#include <fe/fe_q.h>
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
#include "functionalinterface.h"
// note that we solve a pure pde problem here,
// but use the framework of an optimization problem.
// Thus, the optimization ingredients are only
// dummys, i.e. no functional, no constraints, etc.
#include "noconstraints.h"

//DOpE includes for instationary problems
#include "instatreducedproblem.h"
#include "instat_step_newtonsolver.h"
#include "reducednewtonalgorithm.h"
#include "instatoptproblemcontainer.h"

//various timestepping schemes
//#include "forward_euler_problem.h"
//#include "backward_euler_problem.h"
//#include "crank_nicolson_problem.h"
#include "shifted_crank_nicolson_problem.h"

//Problem specific includes
#include "localpde.h"
#include "localfunctional.h"
#include "functionals.h"
#include "my_functions.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

const static int DIM = 2;

#define DOFHANDLER DoFHandler
#define FE FESystem
#define CDC ElementDataContainer
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

// Typedefs for timestep problem
#define TSP ShiftedCrankNicolsonProblem
//FIXME: This should be a reasonable dual timestepping scheme
#define DTSP ShiftedCrankNicolsonProblem
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
    DIM> RP;

int
main(int argc, char **argv)
{
  /**
   *  In this example we solve the instationary Navier Stokes' equations.
   *  We use the well-known Taylor-Hood element
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
  LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>::declare_params(pr);
  BoundaryParabel::declare_params(pr);
  LocalBoundaryFunctionalDrag<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM>::declare_params(
      pr);
  LocalBoundaryFunctionalLift<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM>::declare_params(
      pr);
  pr.read_parameters(paramfile);

  /*** Create the triangulation*********************************/
  Triangulation<DIM> triangulation;
  GridIn<DIM> grid_in;
  grid_in.attach_triangulation(triangulation);
  //std::ifstream input_file("channel.inp");
  std::ifstream input_file("nsbench4_original.inp");
  grid_in.read_ucd(input_file);
  /**********************************************/

  Point<DIM> p(0.2, 0.2);
  double radius = 0.05;
  static const HyperBallBoundary<DIM> boundary(p, radius);
  triangulation.set_boundary(80, boundary);
  triangulation.refine_global(2);

  // The control_fe is only a dummy finite element,
  // as we solve a pure PDE problem
  FE<DIM> control_fe(FE_Nothing<DIM>(), 1);
  FE<DIM> state_fe(FE_Q<DIM>(2), 2, FE_Q<DIM>(1), 1);

  QUADRATURE quadrature_formula(3);
  FACEQUADRATURE face_quadrature_formula(3);
  IDC idc(quadrature_formula, face_quadrature_formula);

  LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM> LPDE(pr);

  // Define a dummy functional for the optimization framework.
  LocalFunctional<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM> LFunc;

  LocalPointFunctionalPressure<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM> LPFP;
  LocalBoundaryFunctionalDrag<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM> LBFD(pr);
  LocalBoundaryFunctionalLift<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM> LBFL(pr);

  // Create a time grid of [0,8] with
  // 80 subintervalls for the timediscretization.
  Triangulation<1> times;
  GridGenerator::subdivided_hyper_cube(times, 80, 0, 8);

  // We give the spatial and time triangulation as well as the state/control finite
  // elements to the MOL-space time handler. DOpEtypes::undefined marks
  // the type of the control, see dopetypes.h for more information.
  MethodOfLines_SpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR, DIM,
      DIM> DOFH(triangulation, control_fe, state_fe, times,
      DOpEtypes::ControlType::stationary);

  // As we solve a pure PDE problem, we have no constraints
  NoConstraints<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM> Constraints;
  OP P(LFunc, LPDE, Constraints, DOFH);

  P.AddFunctional(&LPFP);
  P.AddFunctional(&LBFD);
  P.AddFunctional(&LBFL);

  // We want to evaluate drag and lift at the enclosed cylinder,
  // which has the boundary color 80.
  P.SetBoundaryFunctionalColors(80);

  std::vector<bool> comp_mask(3);

  comp_mask[0] = true;
  comp_mask[1] = true;
  comp_mask[2] = false;

  // Define the noslip boundary conditions...
  DOpEWrapper::ZeroFunction<DIM> zf(3);
  SimpleDirichletData<VECTOR, DIM> DD1(zf);

  //... and the inflow values.
  BoundaryParabel boundary_parabel(pr);
  SimpleDirichletData<VECTOR, DIM> DD2(boundary_parabel);

  P.SetDirichletBoundaryColors(0, comp_mask, &DD2);
  P.SetDirichletBoundaryColors(2, comp_mask, &DD1);
  P.SetDirichletBoundaryColors(80, comp_mask, &DD1);

  P.SetBoundaryEquationColors(1);

  //We use zero initial values.
  P.SetInitialValues(&zf);
//  BoundaryParabelExact boundary_parabel_ex;
//  P.SetInitialValues(&boundary_parabel_ex);

  RP solver(&P, DOpEtypes::VectorStorageType::fullmem, pr, idc);
  RNA Alg(&P, &solver, pr);

  try
  {
    Alg.ReInit();
    ControlVector<VECTOR> q(&DOFH, DOpEtypes::VectorStorageType::fullmem);
    Alg.SolveForward(q);
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

