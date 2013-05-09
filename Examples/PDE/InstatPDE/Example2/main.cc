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

#include "reducednewtonalgorithm.h"
#include "instatoptproblemcontainer.h"
#include "forward_euler_problem.h"
#include "backward_euler_problem.h"
#include "crank_nicolson_problem.h"
#include "shifted_crank_nicolson_problem.h"
#include "fractional_step_theta_problem.h"
#include "functionalinterface.h"
#include "pdeinterface.h"
#include "instatreducedproblem.h"
#include "instat_step_newtonsolver.h"
#include "fractional_step_theta_step_newtonsolver.h"
#include "newtonsolver.h"
#include "gmreslinearsolver.h"
#include "cglinearsolver.h"
#include "directlinearsolver.h"
#include "integrator.h"
#include "parameterreader.h"
#include "mol_spacetimehandler.h"
#include "simpledirichletdata.h"
#include "noconstraints.h"
#include "sparsitymaker.h"
#include "userdefineddofconstraints.h"
#include "integratordatacontainer.h"

#include <iostream>
#include <fstream>

#include <grid/tria.h>
#include <grid/grid_in.h>
#include <grid/tria_boundary_lib.h>
#include <dofs/dof_handler.h>
#include <grid/grid_generator.h>
#include <fe/fe_q.h>
#include <fe/fe_dgp.h>
#include <dofs/dof_tools.h>
#include <base/quadrature_lib.h>
#include <base/function.h>

#include "localpde.h"
#include "localfunctional.h"
#include "functionals.h"

#include "my_functions.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

// Define dimensions for control- and state problem
#define LOCALDOPEDIM 2
#define LOCALDEALDIM 2
#define VECTOR BlockVector<double>
#define SPARSITYPATTERN BlockSparsityPattern
#define MATRIX BlockSparseMatrix<double>
#define DOFHANDLER DoFHandler
#define FE FESystem
#define FUNC FunctionalInterface<CellDataContainer,FaceDataContainer,DOFHANDLER,VECTOR,LOCALDOPEDIM,LOCALDEALDIM>
#define PDE PDEInterface<CellDataContainer,FaceDataContainer,DOFHANDLER,VECTOR,LOCALDEALDIM>
#define DD DirichletDataInterface<VECTOR,LOCALDOPEDIM,LOCALDEALDIM>
#define CONS ConstraintInterface<CellDataContainer,FaceDataContainer,DOFHANDLER,VECTOR,LOCALDOPEDIM,LOCALDEALDIM>

typedef OptProblemContainer<FUNC, FUNC, PDE, DD, CONS, SPARSITYPATTERN, VECTOR,
    LOCALDOPEDIM, LOCALDEALDIM> OP_BASE;
#define PROB StateProblem<OP_BASE,PDE,DD,SPARSITYPATTERN,VECTOR,LOCALDOPEDIM,LOCALDEALDIM>

// Typedefs for timestep problem
#define TSP BackwardEulerProblem
//FIXME: This should be a reasonable dual timestepping scheme
#define DTSP BackwardEulerProblem

typedef InstatOptProblemContainer<TSP, DTSP,FUNC, FUNC, PDE, DD, CONS,
    SPARSITYPATTERN, VECTOR, LOCALDOPEDIM, LOCALDEALDIM> OP;

#undef TSP
#undef DTSP


typedef IntegratorDataContainer<DOFHANDLER, Quadrature<LOCALDEALDIM>,
    Quadrature<LOCALDEALDIM - 1>, VECTOR, LOCALDEALDIM> IDC;

typedef Integrator<IDC, VECTOR, double, 2> INTEGRATOR;

typedef DirectLinearSolverWithMatrix<SPARSITYPATTERN, MATRIX, VECTOR> LINEARSOLVER;

typedef NewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR>
    CNLS;
typedef InstatStepNewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR> NLS;
typedef ReducedNewtonAlgorithm<OP, VECTOR> RNA;
typedef InstatReducedProblem<CNLS, NLS, INTEGRATOR, INTEGRATOR, OP,
    VECTOR, 2, 2> SSolver;

int
main(int argc, char **argv)
{
  /**
   * Instationary FSI problem in an ALE framework
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
  SSolver::declare_params(pr);
  RNA::declare_params(pr);
  LocalPDE<DOFHANDLER, VECTOR,  2>::declare_params(pr);
  BoundaryParabel::declare_params(pr);
  LocalBoundaryFaceFunctionalDrag<DOFHANDLER,VECTOR, LOCALDOPEDIM, LOCALDEALDIM>::declare_params(
      pr);
  LocalBoundaryFaceFunctionalLift<DOFHANDLER,VECTOR, LOCALDOPEDIM, LOCALDEALDIM>::declare_params(
      pr);
  pr.read_parameters(paramfile);

  std::string cases = "solve";

  Triangulation<2> triangulation;

  GridIn<2> grid_in;
  grid_in.attach_triangulation(triangulation);

  // Grid for Benchmark with flag
  std::ifstream input_file("bench_fs_t0100_tw.inp");

  grid_in.read_ucd(input_file);

  Point<2> p(0.2, 0.2);
  double radius = 0.05;
  static const HyperBallBoundary<2> boundary(p, radius);

  // cylinder boundary
  triangulation.set_boundary(80, boundary);
  // cylinder boundary attached to the flag
  triangulation.set_boundary(81, boundary);

  FESystem<2> control_fe(FE_Q<2>(1), 1);

  // FE for the state equation: v,u,p
  FESystem<2> state_fe(FE_Q<2>(2), 2, 
		 FE_Q<2>(2), 2,
		 FE_DGP<2>(1), 1);

  QGauss<2> quadrature_formula(3);
  QGauss<1> face_quadrature_formula(3);
  IDC idc(quadrature_formula, face_quadrature_formula);

  LocalPDE<DOFHANDLER,VECTOR,  2> LPDE(pr);
  LocalFunctional<DOFHANDLER,VECTOR, 2, 2> LFunc;

  LocalPointFunctionalPressure<DOFHANDLER,VECTOR, 2, 2> LPFP;
  LocalPointFunctionalDeflectionX<DOFHANDLER,VECTOR, 2, 2> LPFDX;
  LocalPointFunctionalDeflectionY<DOFHANDLER,VECTOR, 2, 2> LPFDY;

  LocalBoundaryFaceFunctionalDrag<DOFHANDLER,VECTOR, LOCALDOPEDIM, LOCALDEALDIM> LBFD(pr);
  LocalBoundaryFaceFunctionalLift<DOFHANDLER,VECTOR, LOCALDOPEDIM, LOCALDEALDIM> LBFL(pr);

  //Time grid of [0,25]
  Triangulation<1> times;
  GridGenerator::subdivided_hyper_cube(times, 25, 0, 25);

  triangulation.refine_global(1);
  MethodOfLines_SpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR,
      LOCALDOPEDIM, LOCALDEALDIM> DOFH(triangulation, control_fe, state_fe,
      times, DOpEtypes::undefined);

  NoConstraints<CellDataContainer, FaceDataContainer, DOFHANDLER, VECTOR,
      LOCALDOPEDIM, LOCALDEALDIM> Constraints;

  OP P(LFunc, LPDE, Constraints, DOFH);

  //P.HasFaces();
  P.AddFunctional(&LPFP); // pressure difference
  P.AddFunctional(&LPFDX); // deflection of x
  P.AddFunctional(&LPFDY); // deflection of y

  P.AddFunctional(&LBFD); // drag at cylinder and interface
  P.AddFunctional(&LBFL); // lift at cylinder and interface

  // fuer Drag und Lift Auswertung am Zylinder
  P.SetBoundaryFunctionalColors(80);
  P.SetBoundaryFunctionalColors(81);

  std::vector<bool> comp_mask(5);

  comp_mask[0] = true; // vx
  comp_mask[1] = true; // vy
  comp_mask[2] = true; // ux
  comp_mask[3] = true; // uy
  comp_mask[4] = false; // pressure

  DOpEWrapper::ZeroFunction<2> zf(5);
  SimpleDirichletData<VECTOR, LOCALDOPEDIM, LOCALDEALDIM> DD1(zf);

  BoundaryParabel boundary_parabel(pr);
  SimpleDirichletData<VECTOR, LOCALDOPEDIM, LOCALDEALDIM> DD2(boundary_parabel);
  P.SetDirichletBoundaryColors(0, comp_mask, &DD2); // inflow boundary
  P.SetDirichletBoundaryColors(2, comp_mask, &DD1); // rigid walls
  P.SetDirichletBoundaryColors(80, comp_mask, &DD1); // cylinder
  P.SetDirichletBoundaryColors(81, comp_mask, &DD1); // cylinder attached to flag

  P.SetBoundaryEquationColors(1); // outflow boundary

  BoundaryParabelExact boundary_parabel_ex;
  P.SetInitialValues(&zf);
  //P.SetInitialValues(&boundary_parabel_ex);

  SSolver solver(&P, "fullmem", pr, idc);
  RNA Alg(&P, &solver, pr);

  // Mesh-refinement cycles
  int niter = 1;
  Alg.ReInit();
  ControlVector<VECTOR> q(&DOFH, "fullmem");

  for (int i = 0; i < niter; i++)
    {
      try
        {
          if (cases == "check")
            {
              ControlVector<VECTOR> dq(q);
              Alg.CheckGrads(1., q, dq, 2);
              Alg.CheckHessian(1., q, dq, 2);
            }
          else
            {
              Alg.SolveForward(q);
            }
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
          //triangulation.refine_global (1);
          DOFH.RefineSpace();
          Alg.ReInit();
        }
    }

  return 0;
}
#undef LOCALDOPEDIM
#undef LOCALDEALDIM

