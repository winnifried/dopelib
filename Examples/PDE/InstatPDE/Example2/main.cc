/**
 *
 * Copyright (C) 2012-2018 by the DOpElib authors
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
#include <deal.II/fe/fe_dgp.h> //for discont. finite elements
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_in.h>
#if DEAL_II_VERSION_GTE(9,1,1)
#else
#include <deal.II/grid/tria_boundary_lib.h>
#endif
#include <deal.II/grid/grid_generator.h>
#if DEAL_II_VERSION_GTE(9,0,0)
#include <deal.II/grid/manifold_lib.h>
#endif

//DOpE includes
#include <include/parameterreader.h>
#include <templates/directlinearsolver.h>
#include <templates/integrator.h>
#include <basic/mol_statespacetimehandler.h>
#include <problemdata/simpledirichletdata.h>
#include <container/integratordatacontainer.h>
#include <templates/newtonsolver.h>
#include <interfaces/functionalinterface.h>

#include <reducedproblems/instatpdeproblem.h>
#include <templates/instat_step_newtonsolver.h>
#include <container/instatpdeproblemcontainer.h>

#include <tsschemes/shifted_crank_nicolson_problem.h>

//Problem specific includes
#include "localpde.h"
#include "functionals.h"
#include "my_functions.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

const static int DIM = 2;

#if DEAL_II_VERSION_GTE(9,3,0)
#define DOFHANDLER false
#else
#define DOFHANDLER DoFHandler
#endif

#define FE FESystem
#define EDC ElementDataContainer
#define FDC FaceDataContainer

typedef QGauss<DIM> QUADRATURE;
typedef QGauss<DIM - 1> FACEQUADRATURE;
typedef BlockSparseMatrix<double> MATRIX;
typedef BlockSparsityPattern SPARSITYPATTERN;
typedef BlockVector<double> VECTOR;


typedef PDEProblemContainer<
LocalPDE<EDC, FDC, DOFHANDLER, VECTOR, DIM>,
         SimpleDirichletData<VECTOR, DIM>,
         SPARSITYPATTERN,
         VECTOR, DIM> OP_BASE;

typedef StateProblem<OP_BASE, LocalPDE<EDC, FDC, DOFHANDLER, VECTOR, DIM>,
        SimpleDirichletData<VECTOR, DIM>, SPARSITYPATTERN, VECTOR, DIM> PROB;

// Typedefs for timestep problem
#define TSP ShiftedCrankNicolsonProblem
//FIXME: This should be a reasonable dual timestepping scheme
#define DTSP ShiftedCrankNicolsonProblem
typedef InstatPDEProblemContainer<TSP, DTSP,
        LocalPDE<EDC, FDC, DOFHANDLER, VECTOR, DIM>,
        SimpleDirichletData<VECTOR, DIM>,
        SPARSITYPATTERN,
        VECTOR, DIM> OP;
#undef TSP
#undef DTSP

typedef IntegratorDataContainer<DOFHANDLER, QUADRATURE,
        FACEQUADRATURE, VECTOR, DIM> IDC;
typedef Integrator<IDC, VECTOR, double, DIM> INTEGRATOR;
typedef DirectLinearSolverWithMatrix<SPARSITYPATTERN, MATRIX, VECTOR> LINEARSOLVER;
typedef NewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR> CNLS;
typedef InstatStepNewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR> NLS;
typedef InstatPDEProblem<NLS, INTEGRATOR, OP, VECTOR, DIM> RP;

int
main(int argc, char **argv)
{
  /**
   * Nonstationary FSI problem in an ALE framework
   * with biharmonic mesh motion
   * Fluid: NSE
   * Structure: INH or STVK
   */

  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv);

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
  DOpEOutputHandler<VECTOR>::declare_params(pr);
  LocalPDE<EDC, FDC, DOFHANDLER, VECTOR, 2>::declare_params(pr);
  BoundaryParabel::declare_params(pr);
  LocalBoundaryFaceFunctionalDrag<EDC, FDC, DOFHANDLER, VECTOR, DIM, DIM>::declare_params(
    pr);
  LocalBoundaryFaceFunctionalLift<EDC, FDC, DOFHANDLER, VECTOR, DIM, DIM>::declare_params(
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
#if DEAL_II_VERSION_GTE(9,0,0)
  static const SphericalManifold<DIM> boundary(p);
  triangulation.set_all_manifold_ids_on_boundary(80,80);
  triangulation.set_all_manifold_ids_on_boundary(81,81);
  triangulation.set_manifold(80,boundary);
  triangulation.set_manifold(81,boundary);
#else
  double radius = 0.05;
  static const HyperBallBoundary<DIM> boundary(p, radius);

  // cylinder boundary
  triangulation.set_boundary(80, boundary);
  // cylinder boundary attached to the flag
  triangulation.set_boundary(81, boundary);
#endif
  triangulation.refine_global(1);
  /**************************************************************/

  FESystem<DIM> state_fe(FE_Q<DIM>(2), 2,   // v
                         FE_Q<DIM>(2), 2,   // u
                         FE_DGP<DIM>(1), 1, // p
                         FE_Q<DIM>(2), 2);  // w (add. displacement)

  QGauss<DIM> quadrature_formula(3);
  QGauss<DIM - 1> face_quadrature_formula(3);
  IDC idc(quadrature_formula, face_quadrature_formula);

  LocalPDE<EDC, FDC, DOFHANDLER, VECTOR, DIM> LPDE(pr);

  LocalPointFunctionalPressure<EDC, FDC, DOFHANDLER, VECTOR, DIM, DIM> LPFP;
  LocalPointFunctionalDeflectionX<EDC, FDC, DOFHANDLER, VECTOR, DIM, DIM> LPFDX;
  LocalPointFunctionalDeflectionY<EDC, FDC, DOFHANDLER, VECTOR, DIM, DIM> LPFDY;

  LocalBoundaryFaceFunctionalDrag<EDC, FDC, DOFHANDLER, VECTOR, DIM, DIM> LBFD(
    pr);
  LocalBoundaryFaceFunctionalLift<EDC, FDC, DOFHANDLER, VECTOR, DIM, DIM> LBFL(
    pr);

  // Specification of time step size and time interval
  // for different FSI benchmark tests

  // FSI 1: Time grid of [0,25] with
  // 25 subintervalls for the time discretization.
  // Below: times, 25, 0, 25

  // FSI 2: k = 1.0e-2 until T=10
  // Below: times, 1000, 0, 10

  // FSI 3: k = 1.0e-3 until T=10
  // Below: times, 10000, 0, 10
  Triangulation<1> times;
  GridGenerator::subdivided_hyper_cube(times, 25, 0, 25);


  MethodOfLines_StateSpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR,
                                      DIM> DOFH(triangulation, state_fe, times);

  OP P(LPDE, DOFH);

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

  try
    {
      RP solver(&P, DOpEtypes::VectorStorageType::only_recent, pr, idc);
      DOpEOutputHandler<VECTOR> out(&solver, pr);
      DOpEExceptionHandler<VECTOR> ex(&out);
      P.RegisterOutputHandler(&out);
      P.RegisterExceptionHandler(&ex);
      solver.RegisterOutputHandler(&out);
      solver.RegisterExceptionHandler(&ex);

      // Mesh-refinement cycles
      int niter = 1;

      for (int i = 0; i < niter; i++)
        {
          try
            {
              //Before solving we have to reinitialize the stateproblem and outputhandler.
              solver.ReInit();
              out.ReInit();

              stringstream outp;
              outp << "**************************************************\n";
              outp << "*             Starting Forward Solve             *\n";
              outp << "*   Solving : " << P.GetName() << "\t*\n";
              outp << "*   SDoFs   : ";
              solver.StateSizeInfo(outp);
              outp << "**************************************************";
              //We print this header with priority 1 and 1 empty line in front and after.
              out.Write(outp, 1, 1, 1);

              //We compute the value of the functionals. To this end, we have to solve
              //the PDE at hand.
              solver.ComputeReducedFunctionals();
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
            }
        }
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
#undef EDC
#undef FE
#undef DOFHANDLER

