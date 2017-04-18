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
#include <fstream>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/base/quadrature_lib.h>
// for grid local refinement
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/grid/grid_refinement.h>

#include <container/pdeproblemcontainer.h>
#include <reducedproblems/statpdeproblem.h>
#include <templates/newtonsolver.h>
#include <templates/trilinosdirectlinearsolver.h>
#include <templates/directlinearsolver.h>
#include <templates/integrator.h>
#include <include/parameterreader.h>
#include <basic/mol_statespacetimehandler.h>
#include <problemdata/simpledirichletdata.h>
#include <include/solutionextractor.h>
#include <container/integratordatacontainer.h>

#include "localpde.h"
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
typedef SparseMatrix<double> MATRIX;
typedef SparsityPattern SPARSITYPATTERN;
typedef Vector<double> VECTOR;

typedef PDEProblemContainer<LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
        SimpleDirichletData<VECTOR, DIM>, SPARSITYPATTERN, VECTOR, DIM, FE,
        DOFHANDLER> OP;
typedef IntegratorDataContainer<DOFHANDLER, QUADRATURE, FACEQUADRATURE, VECTOR,
        DIM> IDC;
typedef Integrator<IDC, VECTOR, double, DIM> INTEGRATOR;
typedef TrilinosDirectLinearSolverWithMatrix<SPARSITYPATTERN, MATRIX, VECTOR> LINEARSOLVER;
typedef NewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR> NLS;
typedef StatPDEProblem<NLS, INTEGRATOR, OP, VECTOR, DIM> RP;
typedef MethodOfLines_StateSpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN,
        VECTOR, DIM> STH;

int
main(int argc, char **argv)
{
#ifdef DOPELIB_WITH_MPI
  //If deal.II has been configured with MPI we need to initialize it
  //athough we don't use MPI in the code explicitely.
  //However, if not initialized the trilinos solver will fail.
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);
#endif

  /**
   * Stationary FSI problem in an ALE framework
   * Fluid: Navier-Stokes equations.
   * Structure: Incompressible INH model and compressible STVK material
   * We use the Q2^c-Q1^c element for discretization.
   * Computation of PointFunctionals, i.e. pressure, x-and y-deflections
   * Main innovation is the usage of the Kelly error estimator to compute
   * solutions on adaptively refined meshes.
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
  DOpEOutputHandler<VECTOR>::declare_params(pr);
  LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>::declare_params(pr);
  BoundaryParabel::declare_params(pr);
  LocalBoundaryFaceFunctionalDrag<CDC, FDC, DOFHANDLER, VECTOR, DIM>::declare_params(
    pr);
  LocalBoundaryFaceFunctionalLift<CDC, FDC, DOFHANDLER, VECTOR, DIM>::declare_params(
    pr);
  LINEARSOLVER::declare_params(pr);
  pr.read_parameters(paramfile);

  //Create the triangulation
  Triangulation<DIM> triangulation;

  GridIn<DIM> grid_in;
  grid_in.attach_triangulation(triangulation);
  // Grid for "normal" fluid Benchmark
  //std::ifstream input_file("nsbench4_original.inp");

  // Grid for Benchmark with flag (FSI)
  std::ifstream input_file("bench_fs_t0100_tw.inp");

  grid_in.read_ucd(input_file);

  Point<DIM> p(0.2, 0.2);
  double radius = 0.05;
  static const HyperBallBoundary<DIM> boundary(p, radius);
  triangulation.set_boundary(80, boundary);
  triangulation.set_boundary(81, boundary);
  triangulation.refine_global(1);

  FESystem<DIM> state_fe(FE_Q<DIM>(2), 2, FE_Q<DIM>(1), 1, FE_Q<DIM>(2), 2);

  QUADRATURE quadrature_formula(3);
  FACEQUADRATURE face_quadrature_formula(3);
  IDC idc(quadrature_formula, face_quadrature_formula);

  LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM> LPDE(pr);

  LocalPointFunctionalPressure<CDC, FDC, DOFHANDLER, VECTOR, DIM> LPFP;
  LocalPointFunctionalDeflectionX<CDC, FDC, DOFHANDLER, VECTOR, DIM> LPFDX;
  LocalPointFunctionalDeflectionY<CDC, FDC, DOFHANDLER, VECTOR, DIM> LPFDY;
  LocalBoundaryFaceFunctionalDrag<CDC, FDC, DOFHANDLER, VECTOR, DIM> LBFD(pr);
  LocalBoundaryFaceFunctionalLift<CDC, FDC, DOFHANDLER, VECTOR, DIM> LBFL(pr);

  STH DOFH(triangulation, state_fe);

  OP P(LPDE, DOFH);

  P.AddFunctional(&LPFP);
  P.AddFunctional(&LPFDX);
  P.AddFunctional(&LPFDY);
  P.AddFunctional(&LBFD);
  P.AddFunctional(&LBFL);

  //For drag and lift evaluation
  P.SetBoundaryFunctionalColors(80);
  P.SetBoundaryFunctionalColors(81);

  std::vector<bool> comp_mask(5, true);
  comp_mask[2] = false;

  DOpEWrapper::ZeroFunction<DIM> zf(5);
  SimpleDirichletData<VECTOR, DIM> DD1(zf);

  BoundaryParabel boundary_parabel(pr);
  SimpleDirichletData<VECTOR, DIM> DD2(boundary_parabel);
  P.SetDirichletBoundaryColors(0, comp_mask, &DD2);
  P.SetDirichletBoundaryColors(2, comp_mask, &DD1);

  P.SetDirichletBoundaryColors(80, comp_mask, &DD1);
  P.SetDirichletBoundaryColors(81, comp_mask, &DD1);

  P.SetBoundaryEquationColors(1);

  RP solver(&P, DOpEtypes::VectorStorageType::fullmem, pr, idc);

  //Only needed for pure PDE Problems
  DOpEOutputHandler<VECTOR> out(&solver, pr);
  DOpEExceptionHandler<VECTOR> ex(&out);
  P.RegisterOutputHandler(&out);
  P.RegisterExceptionHandler(&ex);
  solver.RegisterOutputHandler(&out);
  solver.RegisterExceptionHandler(&ex);

  // Mesh-refinement cycles
  int niter = 3;

  Vector<double> solution;
  solver.ReInit();
  out.ReInit();

  for (int i = 0; i < niter; i++)
    {
      try
        {
          stringstream outp;

          outp << "**************************************************\n";
          outp << "*             Starting Forward Solve             *\n";
          outp << "*   Solving : " << P.GetName() << "\t*\n";
          outp << "*   SDoFs   : ";
          solver.StateSizeInfo(outp);
          outp << "**************************************************";
          out.Write(outp, 1, 1, 1);

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
          SolutionExtractor<RP, VECTOR> a(solver);
          const StateVector<VECTOR> &gu = a.GetU();
          solution = 0;
          solution = gu.GetSpacialVector();
          Vector<float> estimated_error_per_element(triangulation.n_active_cells());

          std::vector<bool> component_mask(5, false);
          component_mask[2] = true;

          KellyErrorEstimator<DIM>::estimate(
            static_cast<const DoFHandler<DIM>&>(DOFH.GetStateDoFHandler()),
            QGauss<1>(2), FunctionMap<DIM>::type(), solution,
            estimated_error_per_element, component_mask);
          DOFH.RefineSpace(RefineFixedNumber(estimated_error_per_element, 0.3, 0.0));
          solver.ReInit();
          out.ReInit();
        }
    }

  return 0;
}

#undef FDC
#undef CDC
#undef FE
#undef DOFHANDLER
