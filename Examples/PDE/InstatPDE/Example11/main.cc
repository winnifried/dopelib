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
#include <templates/instat_step_newtonsolver.h>
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

//Problem specific includes
// Here are two implementations for the local pde.
// Either of them can be activated by uncommenting the desired
// one and by commenting the other one.

// PDE 1
// A quasi-monolithic formulation based
// on ideas outlined in Heister/Wheeler/Wick; CMAME, 2015
#include "localpde.h"


// Finally, as in the other DOpE examples, we
// have goal functional evaluations and problem-specific data.
#include "functionals.h"
#include "problem_data.h"
#include "sneddon_geometricrefinement.h"
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
SneddonPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
         SimpleDirichletData<VECTOR, DIM>,
         SPARSITYPATTERN,
         VECTOR, DIM> OP_BASE;

typedef StateProblem<OP_BASE, SneddonPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
        SimpleDirichletData<VECTOR, DIM>, SPARSITYPATTERN, VECTOR, DIM> PROB;

#define TSP BackwardEulerProblem
#define DTSP BackwardEulerProblem
typedef InstatPDEProblemContainer<TSP, DTSP,
        SneddonPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
        SimpleDirichletData<VECTOR, DIM>, SPARSITYPATTERN,
        VECTOR, DIM> OP;
#undef TSP
#undef DTSP

typedef IntegratorDataContainer<DOFHANDLER, QUADRATURE,
        FACEQUADRATURE, VECTOR, DIM> IDC;
typedef Integrator<IDC, VECTOR, double, DIM> INTEGRATOR;
typedef DirectLinearSolverWithMatrix<SPARSITYPATTERN, MATRIX, VECTOR> LINEARSOLVER;

typedef InstatStepNewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR> NLS;
typedef InstatPDEProblem<NLS, INTEGRATOR, OP, VECTOR, DIM> RP;

typedef Rothe_StateSpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR, DIM> STH;
typedef ObstacleResidualErrorContainer<STH, VECTOR, DIM> OBSTACLE_RESC;

/*********************************************************************************/
void
declare_params(ParameterReader &param_reader)
{
  param_reader.SetSubsection("main parameters");
  param_reader.declare_entry("prerefine", "1", Patterns::Integer(1),
                             "How often should we refine the coarse grid?");
  param_reader.declare_entry("local_prerefine", "1", Patterns::Integer(1),
                             "How often should we refine the coarse grid locally near the crack?");
  param_reader.declare_entry("num_intervals", "1", Patterns::Integer(1),
                               "How many quasi-timesteps?");
  param_reader.declare_entry("interpolate_initial", "true", Patterns::Bool(),
			     "How many quasi-timesteps?");
  param_reader.declare_entry("niter", "1", Patterns::Integer(1),
                             "How many different mesh levels do you want a solution on?");
  param_reader.declare_entry("adjust_params_to_mesh", "false", Patterns::Bool(),
                             "Should the parameters (eps,phi^0) be adjusted during the iterations?");
  param_reader.declare_entry("ref_type", "1", Patterns::Integer(0),
                             "Which refinement type? (0 = global, 1 = geometric, 2 = adaptive)");
  
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
  SneddonPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>::declare_params(pr);
  declare_params(pr);
  LocalFunctionalTCV<CDC, FDC, DOFHANDLER, VECTOR, DIM>::declare_params(pr);
  pr.read_parameters(paramfile);

  //************************************************
  //define some constants
  pr.SetSubsection("main parameters");
  int prerefine = pr.get_integer("prerefine");
  int local_prerefine = pr.get_integer("local_prerefine");
  int num_intervals = pr.get_integer("num_intervals");
  bool interpolate_initial = pr.get_bool("interpolate_initial");
  int niter = pr.get_integer("niter");
  bool adjust_params_to_mesh = pr.get_bool("adjust_params_to_mesh");
  int ref_type = pr.get_integer("ref_type");
  /*********************************************************************************/
  // Reading mesh and creating triangulation
  Triangulation<DIM> triangulation(Triangulation<DIM>::MeshSmoothing::patch_level_1);
  GridGenerator::subdivided_hyper_cube(triangulation, 20, -10, 10);

  for (int k = 0; k < local_prerefine; k++)
  {
  Triangulation<DIM>::active_cell_iterator
      cell = triangulation.begin_active(),
      endc = triangulation.end();
      for (; cell!=endc; ++cell)
       {
  
        for (unsigned int vertex=0;vertex < GeometryInfo<DIM>::vertices_per_cell;++vertex)
         {
            Tensor<1,DIM> cell_vertex = (cell->vertex(vertex));
            if (cell_vertex[0] <= 2.5 && cell_vertex[0] >= -2.5
                    && cell_vertex[1] <= 1.25 && cell_vertex[1] >= -1.25)
            {
              cell->set_refine_flag();
              break;
            }
         }
        }
  triangulation.execute_coarsening_and_refinement();
  }

  triangulation.refine_global(prerefine);

  double meshsize=1./pow(2.,prerefine+local_prerefine);
  double eps = 4*meshsize*sqrt(2);
  /*********************************************************************************/
  // Assigning finite elements
  // (1) = polynomial degree - please test (2) for u and phi
  // zweite Zahl: Anzahl der Komponenten
  FE<DIM> state_fe(FE_Q<DIM>(1), 2, // vector-valued (here dim=2): displacements 
		   FE_Q<DIM>(1), 1, // scalar-valued phase-field
    		   FE_Q<DIM>(1), 1);  // scalar-valued multiplier for irreversibility

  QUADRATURE quadrature_formula(4);
  FACEQUADRATURE face_quadrature_formula(3);
  IDC idc(quadrature_formula, face_quadrature_formula);

  /*********************************************************************************/
  // Defining the specific PDE
  SneddonPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM> LPDE(pr,eps,meshsize);

  /*********************************************************************************/
  // Defining goal functional
  LocalFunctionalTCV<CDC, FDC, DOFHANDLER, VECTOR, DIM> TCV(pr);
  LocalFunctionalBulk<CDC, FDC, DOFHANDLER, VECTOR, DIM> LFB(pr);
  LocalFunctionalCrack<CDC, FDC, DOFHANDLER, VECTOR, DIM> LFC(pr,eps);
  
  /*********************************************************************************/
  // Create a time grid of [0,0.02] with
  // 80 subintervalls for the timediscretization.
  Triangulation<1> times;
  double initial_time = 0.0;
  double end_time = 5.; //1e-4
  GridGenerator::subdivided_hyper_cube(times, num_intervals, initial_time, end_time);

  /*********************************************************************************/
  // We give the spatial and time triangulation as well as the state finite
  // elements to the Rothe-space time handler. We only use one common mesh in space
  std::vector<unsigned int> Rothe_time_to_dof(num_intervals+1,0);
  STH DOFH(triangulation, state_fe, times, Rothe_time_to_dof);

  OP P(LPDE, DOFH);


  /*********************************************************************************/
  // Add quantity of interest to the problem
  P.AddFunctional(&TCV);
  P.AddFunctional(&LFB);
  P.AddFunctional(&LFC);

   /*********************************************************************************/
  // Prescribing boundary values
  // We have 3 components (2D displacements and scalar-valued phase-field)
  // 4 components with u(x), u(y), phi(x), p(x): pressure ist new component!
  std::vector<bool> comp_mask(4);
  comp_mask[2] = false; // phase-field component (always hom. Neumann data)
  comp_mask[3] = false; // hydro-static pressure

  // Fixed boundaries
  DOpEWrapper::ZeroFunction<DIM> zf(4);
  SimpleDirichletData<VECTOR, DIM> DD1(zf);

  //All Boundaries no displacement
  comp_mask[0] = true;
  comp_mask[1] = true;
  P.SetDirichletBoundaryColors(0, comp_mask, &DD1);
  P.SetDirichletBoundaryColors(1, comp_mask, &DD1);
  P.SetDirichletBoundaryColors(2, comp_mask, &DD1);
  P.SetDirichletBoundaryColors(3, comp_mask, &DD1);


  /*********************************************************************************/
  // Initial data
  InitialData initial_data(meshsize,interpolate_initial);
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

  OBSTACLE_RESC resc(DOFH, DOpEtypes::VectorStorageType::fullmem, pr, DOpEtypes::primal_only);
  
  //**************************************************************************************************
  std::vector<double> TCV_val(niter,0.);
  std::vector<double> LFB_val(niter,0.);
  std::vector<double> LFC_val(niter,0.);
  std::vector<double> d_val(niter,0.);
  std::vector<double> eps_val(niter,0.);
  std::vector<bool> errors(niter,false);
  
  for(int i = 0; i < niter; i++)
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
      outp << "*  d = "<<meshsize<<" h = "<<meshsize*sqrt(2)<<"\t\t*\n";
      outp << "**************************************************";
      //We print this header with priority 1 and 1 empty line in front and after.
      out.Write(outp, 1, 1, 1);

      //We compute the value of the functionals. To this end, we have to solve
      //the PDE at hand.
      solver.ComputeReducedFunctionals();

      if(ref_type == 2)
      {
	solver.ComputeRefinementIndicators(resc, LPDE);
      }
    }
    catch (DOpEException &e)
    {
      std::cout
	<< "Warning: During execution of `" + e.GetThrowingInstance()
	+ "` the following Problem occurred!" << std::endl;
      std::cout << e.GetErrorMessage() << std::endl;
      errors[i] = true;
    }
    //Store Results
    {
      TCV_val[i] = solver.GetTimeFunctionalValue(TCV.GetName())[num_intervals];
      LFB_val[i] = solver.GetTimeFunctionalValue(LFB.GetName())[num_intervals];
      LFC_val[i] = solver.GetTimeFunctionalValue(LFC.GetName())[num_intervals];
      d_val[i] = meshsize;
      eps_val[i] = eps;
    }
    //Refine and adjust
    if( i != niter-1 )
    {
      if(ref_type == 0)
      {
	DOFH.RefineSpace();
      }
      else if (ref_type == 1)
      {
      SneddonGeometricVolumeRefinement<DIM> ref_cont;
      DOFH.RefineSpace(ref_cont);
      }
      else if (ref_type == 2)
      {
	std::vector<dealii::Vector<float> > error_ind(resc.GetErrorIndicators());
	DOFH.RefineSpace(SpaceTimeRefineOptimized(error_ind));
      }
      else
      {
	std::cout<<"Unknown ref_type == "<<ref_type<<std::endl;
	abort();
      }
      if(adjust_params_to_mesh)
      {
	meshsize *= 0.5;
	eps *= 0.5;
	LPDE.SetParams(meshsize,eps);
	initial_data.SetParams(meshsize);
	LFC.SetParams(eps);
      }
    }
  }//Endof niter loop
  { // Print final table 
    pr.SetSubsection("Local PDE parameters");
      double E = pr.get_double("Young_modulus");
      double nu  = pr.get_double("Poisson_ratio");
      bool comp_f = pr.get_bool("compressible_in_fracture");
      
      stringstream outp;
      outp << "*******************************************************************************************************************************************************************************************************\n";
      outp << "*******************************************************************************************************************************************************************************************************\n";
      outp << "*  d_0 = "<<d_val[0]<<" h_0 = "<<d_val[0]*sqrt(2)<<" with ("<<local_prerefine<<") geometric prerefines\n";
      outp << "* epsilon_0 = "<<eps_val[0]<<"\n";
      outp << "* E = "<<E<<" nu = "<<nu<<"\n";
      if(comp_f)
      {
	outp << "* Compressible material in Fracture" <<std::endl;
      }
      outp<<std::endl;
      outp<<"Iter |       d      |      eps     |     TCV      |        LFB   |      LFC     |"<<std::endl;
      outp<<"-----+--------------+--------------+--------------+--------------+--------------+"<<std::endl;
      for( int i = 0; i < niter ; i++)
      {
	if(errors[i])
	{
	  outp<<i<<"    | Error in computation "<<std::endl; 
	}
	else
	{
	  outp<<i<<"    | ";
	  outp<<std::setfill(' ') << std::setw(12)<<d_val[i]<<" | ";
	  outp<<std::setfill(' ') << std::setw(12)<<eps_val[i]<<" | ";
	  outp<<std::setfill(' ') << std::setw(12)<<TCV_val[i]<<" | ";
	  outp<<std::setfill(' ') << std::setw(12)<<LFB_val[i]<<" | ";
	  outp<<std::setfill(' ') << std::setw(12)<<LFC_val[i]<<std::endl;
	}
      }
      out.Write(outp, 1, 1, 1);
    }

 
  return 0;
}

#undef FDC
#undef CDC
#undef FE
#undef DOFHANDLER

