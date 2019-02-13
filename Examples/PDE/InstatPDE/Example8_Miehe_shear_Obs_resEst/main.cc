/**
 *
 * Copyright (C) 2012-2017 by the DOpElib authors
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
//#include <basic/mol_statespacetimehandler.h>
#include <basic/rothe_statespacetimehandler.h> // new!
#include <problemdata/simpledirichletdata.h>
#include <container/integratordatacontainer.h>
#include <templates/newtonsolver.h>
#include <interfaces/functionalinterface.h>


//DOpE includes for instationary problems
#include <reducedproblems/instatpdeproblem.h>
#include <container/instatpdeproblemcontainer.h>
#include <container/dwrdatacontainer.h>

// Timestepping scheme
#include <tsschemes/backward_euler_problem.h>
#include <tsschemes/crank_nicolson_problem.h>
#include <tsschemes/shifted_crank_nicolson_problem.h>
#include <tsschemes/fractional_step_theta_problem.h>

// A new heuristic Newton solver
#include "instat_step_modified_newtonsolver.h"

//Problem specific includes
// Here are two implementations for the local pde.
// Either of them can be activated by uncommenting the desired
// one and by commenting the other one.

// PDE 1
// A quasi-monolithic formulation based
// on ideas outlined in Heister/Wheeler/Wick; CMAME, 2015
#include "localpde_quasi_monolithic.h"

// PDE 2
// A fully implicit formulation without any time-lagging of
// the phase-field variable. For the fully
// implicit setting, the Newton solver needs to be changed
// to a modified heuristic version that temporarily allows for
// an increase of the residual.
// #include "localpde_fully_implicit.h"


// Finally, as in the other DOpE examples, we
// have goal functional evaluations and problem-specific data.
#include "functionals.h"
#include "problem_data.h"
#include "functions.h"
#include "obstacleestimator.h"


using namespace std;
using namespace dealii;
using namespace DOpE;

// Ggf. in cmake (CMAKELists.txt) die Dimension ebenfalls aendern 
// und neues Makefile erzeugen
const static int DIM = 2;

#define DOFHANDLER DoFHandler
#define FE FESystem
#define CDC ElementDataContainer
#define FDC FaceDataContainer


/*********************************************************************************/
//Use LobattoFormulas, as obstacle multiplier is located in vertices
typedef QGaussLobatto<DIM> QUADRATURE;
typedef QGaussLobatto<DIM - 1> FACEQUADRATURE;
//typedef QGauss<DIM - 1> FACEQUADRATURE;
typedef BlockSparseMatrix<double> MATRIX;
typedef BlockSparsityPattern SPARSITYPATTERN;
typedef BlockVector<double> VECTOR;

typedef PDEProblemContainer<
LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
         SimpleDirichletData<VECTOR, DIM>,
         SPARSITYPATTERN,
         VECTOR, DIM> OP_BASE;

typedef StateProblem<OP_BASE, LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
        SimpleDirichletData<VECTOR, DIM>, SPARSITYPATTERN, VECTOR, DIM> PROB;

#define TSP BackwardEulerProblem
#define DTSP BackwardEulerProblem
typedef InstatPDEProblemContainer<TSP, DTSP,
        LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
        SimpleDirichletData<VECTOR, DIM>, SPARSITYPATTERN,
        VECTOR, DIM> OP;
#undef TSP
#undef DTSP

typedef IntegratorDataContainer<DOFHANDLER, QUADRATURE,
        FACEQUADRATURE, VECTOR, DIM> IDC;
typedef Integrator<IDC, VECTOR, double, DIM> INTEGRATOR;
typedef DirectLinearSolverWithMatrix<SPARSITYPATTERN, MATRIX, VECTOR> LINEARSOLVER;

typedef InstatStepModifiedNewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR> NLS;
typedef InstatPDEProblem<NLS, INTEGRATOR, OP, VECTOR, DIM> RP;
typedef Rothe_StateSpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR, DIM> STH;
typedef ObstacleResidualErrorContainer<STH, VECTOR, DIM> OBSTACLE_RESC;

/*********************************************************************************/
void
declare_params(ParameterReader &param_reader)
{
  param_reader.SetSubsection("main parameters");
  param_reader.declare_entry("max_iter", "1", Patterns::Integer(0),
                             "How many iterations?");
  param_reader.declare_entry("prerefine", "1", Patterns::Integer(1),
                             "How often should we refine the coarse grid?");
}

/*********************************************************************************/

int
main(int argc, char **argv)
{
  /**
   *  We solve a quasi-static phase-field brittle fracture
   *  propagation problem. The crack irreversibility
   *  constraint is imposed with the help of a Lagrange multiplier
   *  The configuration is the single edge notched shear test.
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

  /*********************************************************************************/
  // Parameter data
  ParameterReader pr;
  RP::declare_params(pr);
  DOpEOutputHandler<VECTOR>::declare_params(pr);
  LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>::declare_params(pr);
  declare_params(pr);
  NonHomoDirichletData::declare_params(pr);
  LocalBoundaryFunctionalStressX<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM>::declare_params(pr);
  pr.read_parameters(paramfile);

  //************************************************
  //define some constants
  pr.SetSubsection("main parameters");
  int prerefine = pr.get_integer("prerefine");
  int max_iter = pr.get_integer("max_iter");

  /*********************************************************************************/
  // Reading mesh and creating triangulation
  Triangulation<DIM> triangulation(
    Triangulation<DIM>::MeshSmoothing::patch_level_1);
  GridIn<DIM> grid_in;
  grid_in.attach_triangulation(triangulation);
  std::ifstream input_file("unit_slit.inp");
  grid_in.read_ucd(input_file);
  triangulation.refine_global(prerefine);


  /*********************************************************************************/
  // Assigning finite elements
  // (1) = polynomial degree - please test (2) for u and phi
  // zweite Zahl: Anzahl der Komponenten
  FE<DIM> state_fe(FE_Q<DIM>(1), 2, // vector-valued (here dim=2): displacements 
		   FE_Q<DIM>(1), 1, // scalar-valued phase-field
    		   //FE_Q<DIM>(1), 1, // scalar-valued pressure
    		   FE_Q<DIM>(1), 1);  // scalar-valued multiplier for irreversibility

  QUADRATURE quadrature_formula(3);
  FACEQUADRATURE face_quadrature_formula(3);
  IDC idc(quadrature_formula, face_quadrature_formula);

  /*********************************************************************************/
  // Defining the specific PDE
  LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM> LPDE(pr);

  /*********************************************************************************/
  // Defining goal functional
  LocalBoundaryFunctionalStressX<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM> LBFSX(pr);


  /*********************************************************************************/
  // Create a time grid of [0,0.02] with
  // 80 subintervalls for the timediscretization.
  // timestep size -> 10e-3
  Triangulation<1> times;
  unsigned int num_intervals = 20;
  double initial_time = 0.0;
  double end_time = 0.02;
  GridGenerator::subdivided_hyper_cube(times, num_intervals, initial_time, end_time);
  
  // we want to log the refinement history
  //ofstream history ("mesh.history");
    //std::cout << "Number of active cells beginning: "
            //<< times.n_active_cells()
            //<< std::endl;

  //Triangulation<1>::active_cell_iterator cell,endc;
  	//cell = times.begin_active(),
        //endc = times.end();
  //int i = 1;
  //for (; cell != endc; ++cell)
    //{
	//if(i == 3 || i == 10 || i == 15)
	  //{
		//for (int n=0; n<3; ++n)
    		   //{
		 	//cell->set_coarsen_flag();
		   //}
	  //}
        //i++;
	//times.save_coarsen_flags(history);
        //times.execute_coarsening_and_refinement ();
    //}

  //std::cout << "Number of active cells after refinement/coarsening: "
            //<< times.n_active_cells()
            //<< std::endl;
    //abort();

  /*********************************************************************************/
  // We give the spatial and time triangulation as well as the state finite
  // elements to the MOL-space time handler. DOpEtypes::undefined marks
  // the type of the control, see dopetypes.h for more information.
  // MethodOfLines_StateSpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR, DIM>
  //STH DOFH(triangulation, state_fe, times);
  std::vector<unsigned int> Rothe_time_to_dof(21,0);

  for (int k=1.0; k<21.0; k++)
	{
	  Rothe_time_to_dof[k]=k;
	}


  Rothe_StateSpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR,
			      DIM> DOFH(triangulation, state_fe, times, Rothe_time_to_dof);



  // Finales Problem, was alles handled
  OP P(LPDE, DOFH);


  /*********************************************************************************/
  // Add quantity of interest to the problem
  P.AddFunctional(&LBFSX);

  // We want to evaluate the stress on the top boundary with id = 3
  P.SetBoundaryFunctionalColors(3);


  /*********************************************************************************/
  // Prescribing boundary values
  // We have 3 components (2D displacements and scalar-valued phase-field)
  // 4 components with u(x), u(y), phi(x), p(x): pressure ist new component!
  std::vector<bool> comp_mask(4);
  comp_mask[2] = false; // phase-field component (always hom. Neumann data)
  
  //comp_mask[3] = false; // 
  // Fixed boundaries
  DOpEWrapper::ZeroFunction<DIM> zf(4);
  SimpleDirichletData<VECTOR, DIM> DD1(zf);

  // Non-homogeneous boundary (on top where we tear)
  NonHomoDirichletData dirichlet_data(pr);
  SimpleDirichletData<VECTOR, DIM> DD2(dirichlet_data);

  comp_mask[0] = false;
  comp_mask[1] = true;
  P.SetDirichletBoundaryColors(0, comp_mask, &DD1);

  comp_mask[0] = false;
  comp_mask[1] = true;
  P.SetDirichletBoundaryColors(1, comp_mask, &DD1);

  comp_mask[0] = true;
  comp_mask[1] = true;
  P.SetDirichletBoundaryColors(2, comp_mask, &DD1);

  comp_mask[0] = true;
  comp_mask[1] = true;
  P.SetDirichletBoundaryColors(3, comp_mask, &DD2);

  comp_mask[0] = false;
  comp_mask[1] = true;
  P.SetDirichletBoundaryColors(4, comp_mask, &DD1);


  /*********************************************************************************/
  // Initial data
  InitialData initial_data;
  P.SetInitialValues(&initial_data);


  /*********************************************************************************/
  RP solver(&P, DOpEtypes::VectorStorageType::fullmem, pr, idc);

  //Only needed for pure PDE Problems: We define and register
  //the output- and exception handler. The first handels the
  //output on the screen as well as the output of files. The
  //amount of the output can be steered by the paramfile.
  DOpEOutputHandler<VECTOR> out(&solver, pr);
  DOpEExceptionHandler<VECTOR> ex(&out);
  P.RegisterOutputHandler(&out);
  P.RegisterExceptionHandler(&ex);
  solver.RegisterOutputHandler(&out);
  solver.RegisterExceptionHandler(&ex);

  /**********************************************************************/
  
  OBSTACLE_RESC resc(DOFH, DOpEtypes::VectorStorageType::fullmem, pr, DOpEtypes::primal_only);
  
  //**************************************************************************************************
 
  for (int i = 0; i < max_iter; i++)
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
      solver.ComputeRefinementIndicators(resc, LPDE);
      resc.GetErrorIndicators();
    }
  catch (DOpEException &e)
    {
      std::cout
          << "Warning: During execution of `" + e.GetThrowingInstance()
          + "` the following Problem occurred!" << std::endl;
      std::cout << e.GetErrorMessage() << std::endl;
    }
      if (i != max_iter - 1)
        {
          const std::vector<dealii::Vector<float> > error_ind(resc.GetErrorIndicators());
	  //std::cout << "main.cc: after error ind1 " << error_ind[5](5) << std::endl;
	  
          DOFH.RefineSpace(SpaceTimeRefineOptimized(error_ind));
	  
        }
  }
  return 0;
}

#undef FDC
#undef CDC
#undef FE
#undef DOFHANDLER

