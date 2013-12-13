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
#include "functionalinterface.h"
#include "noconstraints.h"

#include "instatreducedproblem.h"
#include "instat_step_newtonsolver.h"
#include "reducednewtonalgorithm.h"
#include "instatoptproblemcontainer.h"

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
   * Nonstationary FSI problem in an ALE framework
   * with biharmonic mesh motion
   * Fluid: NSE
   * Structure: INH or STVK
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
  LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, 2>::declare_params(pr);
  BoundaryParabel::declare_params(pr);
  LocalBoundaryFaceFunctionalDrag<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM>::declare_params(
      pr);
  LocalBoundaryFaceFunctionalLift<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM>::declare_params(
      pr);
  pr.read_parameters(paramfile);

  /**********************************************************/
  Triangulation<DIM> triangulation;

  GridIn<DIM> grid_in;
  grid_in.attach_triangulation(triangulation);

  // Grid for Benchmark with flag
  std::ifstream input_file("bench_fs_t0100_tw.inp");

  grid_in.read_ucd(input_file);

  Point<DIM> p(0.2, 0.2);
  double radius = 0.05;
  static const HyperBallBoundary<DIM> boundary(p, radius);

  // cylinder boundary
  triangulation.set_boundary(80, boundary);
  // cylinder boundary attached to the flag
  triangulation.set_boundary(81, boundary);
  triangulation.refine_global(1);
  /**************************************************************/

  FESystem<DIM> control_fe(FE_Nothing<DIM>(), 1);

  FESystem<DIM> state_fe(FE_Q<DIM>(2), 2,   // v
			 FE_Q<DIM>(2), 2,   // u
			 FE_DGP<DIM>(1), 1, // p
			 FE_Q<DIM>(2), 2);  // w (add. displacement)

  QGauss<DIM> quadrature_formula(3);
  QGauss<DIM - 1> face_quadrature_formula(3);
  IDC idc(quadrature_formula, face_quadrature_formula);

  LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM> LPDE(pr);
  LocalFunctional<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM> LFunc;

  LocalPointFunctionalPressure<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM> LPFP;
  LocalPointFunctionalDeflectionX<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM> LPFDX;
  LocalPointFunctionalDeflectionY<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM> LPFDY;

  LocalBoundaryFaceFunctionalDrag<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM> LBFD(
      pr);
  LocalBoundaryFaceFunctionalLift<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM> LBFL(
      pr);

  // Specification of time step size and time interval 
  // for different FSI benchmark tests

  // FSI 1: Time grid of [0,25] with
  // 25 subintervalls for the time discretization.
  // Below: times, 25, 0, 25 

  // FSI 2: k = 1.0e-2 until T=10
  // Below: times, 1000, 0, 10

  // FSI 3: k = 1.0e-2 until T=10
  // Below: times, 10000, 0, 10
  Triangulation<1> times;
  GridGenerator::subdivided_hyper_cube(times, 25, 0, 25);


  MethodOfLines_SpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR, DIM,
      DIM> DOFH(triangulation, control_fe, state_fe, times,
      DOpEtypes::undefined);

  NoConstraints<ElementDataContainer, FaceDataContainer, DOFHANDLER, VECTOR, DIM,
      DIM> Constraints;

  OP P(LFunc, LPDE, Constraints, DOFH);

  P.AddFunctional(&LPFP); // pressure difference
  P.AddFunctional(&LPFDX); // deflection of x
  P.AddFunctional(&LPFDY); // deflection of y

  P.AddFunctional(&LBFD); // drag at cylinder and interface
  P.AddFunctional(&LBFL); // lift at cylinder and interface

  // We have functionals acting on the boundaries with
  // the colors 80 and 81
  P.SetBoundaryFunctionalColors(80);
  P.SetBoundaryFunctionalColors(81);

  std::vector<bool> comp_mask(7);

  comp_mask[0] = true; // vx
  comp_mask[1] = true; // vy
  comp_mask[2] = true; // ux
  comp_mask[3] = true; // uy
  comp_mask[4] = false; // pressure
  comp_mask[5] = false; // wx
  comp_mask[6] = false; // wy

  DOpEWrapper::ZeroFunction<DIM> zf(7);
  SimpleDirichletData<VECTOR, DIM> DD1(zf);

  BoundaryParabel boundary_parabel(pr);
  SimpleDirichletData<VECTOR, DIM> DD2(boundary_parabel);
  P.SetDirichletBoundaryColors(0, comp_mask, &DD2); // inflow boundary
  P.SetDirichletBoundaryColors(2, comp_mask, &DD1); // rigid walls

  comp_mask[5] = true; // wx
  comp_mask[6] = true; // wy
  P.SetDirichletBoundaryColors(80, comp_mask, &DD1); // cylinder
  P.SetDirichletBoundaryColors(81, comp_mask, &DD1); // cylinder attached to flag

  P.SetBoundaryEquationColors(1); // outflow boundary

  P.SetInitialValues(&zf);

  RP solver(&P, "fullmem", pr, idc);
  RNA Alg(&P, &solver, pr);

  // Mesh-refinement cycles
  int niter = 1;
  Alg.ReInit();
  ControlVector<VECTOR> q(&DOFH, "fullmem");

  for (int i = 0; i < niter; i++)
  {
    try
    {
      Alg.SolveForward(q);
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

