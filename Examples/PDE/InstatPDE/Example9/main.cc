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

//DOpE includes
#include <include/parameterreader.h>
#include <templates/directlinearsolver.h>
#include <templates/integrator.h>
#include <basic/rothe_statespacetimehandler.h>
#include <problemdata/simpledirichletdata.h>
#include <container/integratordatacontainer.h>
#include <templates/newtonsolver.h>
#include <templates/fractional_step_theta_step_newtonsolver.h>

#include <reducedproblems/instatpdeproblem.h>
#include <templates/instat_step_newtonsolver.h>
#include <container/instatpdeproblemcontainer.h>
#include <container/residualestimator.h>

#include <tsschemes/forward_euler_problem.h>
#include <tsschemes/backward_euler_problem.h>
#include <tsschemes/crank_nicolson_problem.h>
#include <tsschemes/shifted_crank_nicolson_problem.h>
#include <tsschemes/fractional_step_theta_problem.h>

//Problem specific includes
#include "localpde.h"
#include "functionals.h"
#include "my_functions.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

// Define dimensions for control- and state problem
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

//typedef FunctionalInterface<EDC, FDC, DOFHANDLER, VECTOR, DIM, DIM> FUNC;

typedef PDEProblemContainer<
LocalPDE<EDC, FDC, DOFHANDLER, VECTOR, DIM>,
         SimpleDirichletData<VECTOR, DIM>,
         SPARSITYPATTERN,
         VECTOR, DIM> OP_BASE;

typedef StateProblem<OP_BASE, LocalPDE<EDC, FDC, DOFHANDLER, VECTOR, DIM>,
        SimpleDirichletData<VECTOR, DIM>, SPARSITYPATTERN, VECTOR, DIM> PROB;
// Typedefs for timestep problem

#define TSP BackwardEulerProblem
//FIXME: This should be a reasonable dual timestepping scheme
#define DTSP BackwardEulerProblem

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

typedef InstatPDEProblem<NLS, INTEGRATOR, OP, VECTOR,
        DIM> RP;

typedef H1ResidualErrorContainer<Rothe_StateSpaceTimeHandler<FE,
        DOFHANDLER,
        SPARSITYPATTERN,
        VECTOR,
        DIM>,
        VECTOR, DIM> H1_RESC;

int
main(int argc, char **argv)
{
  /**
   * In this example we solve the two dimensional heat equation on varying meshes in time.
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

  //First, declare the parameters and read them in.
  ParameterReader pr;
  RP::declare_params(pr);
  DOpEOutputHandler<VECTOR>::declare_params(pr);
  pr.read_parameters(paramfile);

  std::string cases = "solve";

  //Create the triangulation.
  Triangulation<DIM> triangulation;
  GridGenerator::hyper_cube(triangulation, 0., 1.,true);

  //Define the Finite Elements and quadrature formulas for the state.
  FESystem<DIM> state_fe(FE_Q<DIM>(1), 1);

  QGauss<DIM> quadrature_formula(3);
  QGauss<DIM - 1> face_quadrature_formula(3);
  IDC idc(quadrature_formula, face_quadrature_formula);

  //Define the localPDE and the functionals we are interested in. Here, LFunc is a dummy necessary for the control,
  //LPF is a SpaceTimePointevaluation
  LocalPDE<EDC, FDC, DOFHANDLER, VECTOR, DIM> LPDE;
  LocalPointFunctional<EDC, FDC, DOFHANDLER, VECTOR, DIM, DIM> LPF;

  //Time grid of [0,1]
  Triangulation<1> times;
  unsigned int n_time_steps=50;
  GridGenerator::subdivided_hyper_cube(times,n_time_steps);
  triangulation.refine_global(4);
  std::vector<unsigned int> Rothe_time_to_dof(n_time_steps+1,0);
  for (unsigned int i = 0; i < Rothe_time_to_dof.size(); i++)
    Rothe_time_to_dof[i]=i%2;
  Rothe_StateSpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR,
                              DIM> DOFH(triangulation, state_fe, times, Rothe_time_to_dof);

  OP P(LPDE, DOFH);

  P.AddFunctional(&LPF);

  std::vector<bool> comp_mask(1);
  comp_mask[0] = true;

  //Here we use zero boundary values
  DOpEWrapper::ZeroFunction<DIM> zf;
  SimpleDirichletData<VECTOR, DIM> DD1(zf);
  DirichletValues dirichlet;
  SimpleDirichletData<VECTOR, DIM> DD2(dirichlet);
  P.SetDirichletBoundaryColors(0, comp_mask, &DD1);
  P.SetDirichletBoundaryColors(1, comp_mask, &DD2);
  P.SetDirichletBoundaryColors(2, comp_mask, &DD1);
  P.SetDirichletBoundaryColors(3, comp_mask, &DD1);

  //prepare the initial data
  InitialData initial_data;
  P.SetInitialValues(&initial_data);

  RP solver(&P, DOpEtypes::VectorStorageType::fullmem, pr, idc);

  //Use one outputhandler for all problems
  DOpEOutputHandler<VECTOR> output(&solver, pr);
  DOpEExceptionHandler<VECTOR> ex(&output);

  P.RegisterOutputHandler(&output);
  P.RegisterExceptionHandler(&ex);
  solver.RegisterOutputHandler(&output);
  solver.RegisterExceptionHandler(&ex);

  H1_RESC h1resc(DOFH, DOpEtypes::VectorStorageType::fullmem, pr, DOpEtypes::primal_only);
  unsigned int n_iter=4;
  for (unsigned int j = 0; j < n_iter; j++)
    {
      try
        {

          solver.ReInit();
          output.ReInit();

          stringstream outp;
          {
            outp << "**************************************************\n";
            outp << "*             Starting Forward Solve             *\n";
            outp << "*   Solving : " << P.GetName() << "\t*\n";
            outp << "*   SDoFs   : ";
            solver.StateSizeInfo(outp);
            outp << "**************************************************";
            //We print this header with priority 1 and 1 empty line in front and after.
            output.Write(outp, 1, 1, 1);

            //We compute the value of the functionals. To this end, we have to solve
            //the PDE at hand.
            solver.ComputeReducedFunctionals();
            solver.ComputeRefinementIndicators(h1resc, LPDE);
            outp << "H1-Error estimator: " << sqrt(h1resc.GetError()) << std::endl;
            output.Write(outp, 1, 1, 1);
          }

          // The soluction extractor class allows us
          // to get the solution vector from the solver
          SolutionExtractor<RP, VECTOR> a(solver);
          const StateVector<VECTOR> &statevec = a.GetU();

          double product = statevec * statevec;
          outp << "Backward euler: u * u = " << product << std::endl;
          output.Write(outp, 0);
        }

      catch (DOpEException &e)
        {
          std::cout
              << "Warning: During execution of `" + e.GetThrowingInstance()
              + "` the following Problem occurred!" << std::endl;
          std::cout << e.GetErrorMessage() << std::endl;
        }
      //No refinement after the last iteration
      if (j != n_iter-1)
        {
          //For global mesh refinement, uncomment the next line
          // DOFH.RefineSpace(DOpEtypes::RefinementType::global); //or just DOFH.RefineSpace()

          const std::vector<dealii::Vector<float> > error_ind(h1resc.GetErrorIndicators());
          DOFH.RefineSpace(SpaceTimeRefineOptimized(error_ind));
          //There are other mesh refinement strategies implemented, for example
          //DOFH.RefineSpace(RefineFixedNumber(error_ind, 0.4));
          //DOFH.RefineSpace(RefineFixedFraction(error_ind, 0.8));
        }

    }
  return 0;
}

#undef FDC
#undef EDC
#undef FE
#undef DOFHANDLER


