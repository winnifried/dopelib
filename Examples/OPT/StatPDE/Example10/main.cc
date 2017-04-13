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
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/grid/grid_refinement.h>

#include <opt_algorithms/reducednewtonalgorithm.h>
#include <container/optproblemcontainer.h>
#include <interfaces/functionalinterface.h>
#include <reducedproblems/statreducedproblem.h>
#include <templates/newtonsolver.h>
#include <templates/directlinearsolver.h>
#include <templates/voidlinearsolver.h>
#include <templates/integrator.h>
#include <problemdata/noconstraints.h>
#include <include/solutionextractor.h>
#include <templates/integratormixeddims.h>
#include <templates/newtonsolvermixeddims.h>
#include <include/parameterreader.h>
#include <basic/mol_spacetimehandler.h>
#include <problemdata/simpledirichletdata.h>
#include <container/integratordatacontainer.h>

#include "localpde.h"
#include "localfunctional.h"
#include "functionals.h"
#include "my_functions.h"

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

typedef OptProblemContainer<FUNCTIONALINTERFACE, COSTFUNCTIONAL,
        LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
        SimpleDirichletData<VECTOR, DIM>,
        NoConstraints<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM>, SPARSITYPATTERN,
        VECTOR, CDIM, DIM> OP;

typedef IntegratorDataContainer<DOFHANDLER, QUADRATURE, FACEQUADRATURE, VECTOR,
        DIM> IDC;
typedef Integrator<IDC, VECTOR, double, DIM> INTEGRATOR;
//special newtonsolver for the mixed dims
typedef IntegratorMixedDimensions<IDC, VECTOR, double, CDIM, DIM> INTEGRATORM;

typedef DirectLinearSolverWithMatrix<SPARSITYPATTERN, MATRIX, VECTOR> LINEARSOLVER;

//dummy solver for the 0d control
typedef VoidLinearSolver<VECTOR> VOIDLS;

//special newtonsolver for the mixed dims
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
   * In this example we study
   * stationary flow control. The configuration
   * comes from the original fluid benchmark problem
   * (Schaefer/Turek; 1996)
   * and has been modified (as R. Becker; 2000) to reduce drag around the
   * cylinder. To gain the solvability of
   * the optimization problem we add a quadratic
   * regularization term to the cost functional.
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
  COSTFUNCTIONAL::declare_params(pr);
  BoundaryParabel::declare_params(pr);
  LocalBoundaryFaceFunctionalDrag<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM>::declare_params(
    pr);
  LocalBoundaryFaceFunctionalLift<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM>::declare_params(
    pr);
  pr.read_parameters(paramfile);

  // Mesh-refinement cycles
  const int niter = 1;

  //create triangulation
  Triangulation<DIM> triangulation;
  GridIn<DIM> grid_in;
  grid_in.attach_triangulation(triangulation);
  // Grid for "normal" fluid Benchmark
  std::ifstream input_file("nsbench4_modified.inp");

  grid_in.read_ucd(input_file);

  Point<DIM> p(0.2, 0.2);
  const double radius = 0.05;
  static const HyperBallBoundary<DIM> boundary(p, radius);
  triangulation.set_boundary(80, boundary);
  triangulation.refine_global(2);


  FE<DIM> control_fe(FE_Nothing<DIM>(1), 2); //2 Parameter, thus 2 components
  FE<DIM> state_fe(FE_Q<DIM>(2), 2, // velocities
                   FE_Q<DIM>(1), 1); // pressure with CG(1)

  QUADRATURE quadrature_formula(3);
  FACEQUADRATURE face_quadrature_formula(3);
  IDC idc(quadrature_formula, face_quadrature_formula);

  LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM> LPDE(pr);
  COSTFUNCTIONAL LFunc(pr);

  LocalPointFunctionalPressure<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> LPFP;
  LocalPointFunctionalDeflectionX<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> LPFDX;
  LocalPointFunctionalDeflectionY<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> LPFDY;
  LocalBoundaryFaceFunctionalDrag<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> LBFD(
    pr);
  LocalBoundaryFaceFunctionalLift<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> LBFL(
    pr);


  STH DOFH(triangulation, control_fe, state_fe, DOpEtypes::stationary);

  NoConstraints<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> Constraints;

  OP P(LFunc, LPDE, Constraints, DOFH);

  P.AddFunctional(&LPFP);
  P.AddFunctional(&LBFD);
  P.AddFunctional(&LBFL);

  //Due to drag and lift evaluation at the boundary
  P.SetBoundaryFunctionalColors(80);

  // Due to regularization
  P.SetBoundaryFunctionalColors(50);
  P.SetBoundaryFunctionalColors(51);

  std::vector<bool> comp_mask(3, true);
  comp_mask[2] = false;

  DOpEWrapper::ZeroFunction<DIM> zf(3);
  SimpleDirichletData<VECTOR, DIM> DD1(zf);

  BoundaryParabel boundary_parabel(pr);
  SimpleDirichletData<VECTOR, DIM> DD2(boundary_parabel);

  P.SetDirichletBoundaryColors(0, comp_mask, &DD2); // flow by Dirichlet data
  P.SetDirichletBoundaryColors(2, comp_mask, &DD1);
  P.SetDirichletBoundaryColors(80, comp_mask, &DD1);

  P.SetBoundaryEquationColors(1); // do-nothing at outflow boundary
  P.SetBoundaryEquationColors(50); // upper control bc \Gamma_q1
  P.SetBoundaryEquationColors(51); // lower control bc \Gamma_q2

  // We need these functions to evaluate
  // BoundaryEquation_Q, etc.
  P.SetControlBoundaryEquationColors(50); // upper control bc \Gamma_q1
  P.SetControlBoundaryEquationColors(51); // lower control bc \Gamma_q2

  RP solver(&P, DOpEtypes::VectorStorageType::fullmem, pr, idc);
  RNA Alg(&P, &solver, pr);

  std::string cases = "solve";

  Vector<double> solution;
  Alg.ReInit();
  ControlVector<VECTOR> q(&DOFH, DOpEtypes::VectorStorageType::fullmem);
  q = 0.1;
  for (int i = 0; i < niter; i++)
    {
      try
        {
          if (cases == "check")
            {

              ControlVector<VECTOR> dq(q);
              // eps: step size for difference quotient
              // choose: 1.0, 0.1, 0.01, etc.
              const double eps_diff = 1.0e-2;
              Alg.CheckGrads(eps_diff, q, dq, 2);
              Alg.CheckHessian(eps_diff, q, dq, 2);
            }
          else
            {
              //Alg.SolveForward(q);  // just solves the forward problem
              Alg.Solve(q);
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
          SolutionExtractor<RP, VECTOR> a(solver);
          const StateVector<VECTOR> &gu = a.GetU();
          solution = 0;
          solution = gu.GetSpacialVector();
          Vector<float> estimated_error_per_element(triangulation.n_active_cells());

          std::vector<bool> component_mask(3, false);
          component_mask[2] = true;

          KellyErrorEstimator<DIM>::estimate(
            static_cast<const DoFHandler<DIM>&>(DOFH.GetStateDoFHandler()),
            QGauss<1>(2), FunctionMap<DIM>::type(), solution,
            estimated_error_per_element, component_mask);

          DOFH.RefineSpace(RefineFixedNumber(estimated_error_per_element, 0.5, 0.0));
          Alg.ReInit();
        }

    }

  return 0;
}


#undef FDC
#undef CDC
#undef FE
#undef DOFHANDLER
