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


#include <iostream>
#include <fstream>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_dgp.h>

#include <deal.II/base/quadrature_lib.h>

#include <reducedproblems/statpdeproblem.h>

#include <templates/newtonsolver.h>
#include <templates/directlinearsolver.h>
#include <templates/integrator.h>

#include <include/parameterreader.h>

#include <basic/mol_statespacetimehandler.h>

#include <problemdata/simpledirichletdata.h>

#include <container/interpolatedintegratordatacontainer.h>

#include "localpde.h"
#include "my_functions.h"

//H div spaces where we interpolate to
#include <deal.II/fe/fe_raviart_thomas.h>

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

typedef QGauss<DIM> QUADRATURE;
typedef QGauss<DIM - 1> FACEQUADRATURE;

typedef BlockSparseMatrix<double> MATRIX;
typedef BlockSparsityPattern SPARSITYPATTERN;
typedef BlockVector<double> VECTOR;

//FIXME: Why use the standard data containers if the interpolated are needed in the PDE?
#define CDC ElementDataContainer
#define FDC FaceDataContainer

typedef PDEProblemContainer<LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
        SimpleDirichletData<VECTOR, DIM>, SPARSITYPATTERN, VECTOR, DIM, FE,
        DOFHANDLER> OP;

typedef InterpolatedIntegratorDataContainer<DOFHANDLER, QUADRATURE, FACEQUADRATURE, VECTOR,
        DIM> IDC;

typedef Integrator<IDC, VECTOR, double, DIM> INTEGRATOR;

typedef DirectLinearSolverWithMatrix<SPARSITYPATTERN, MATRIX, VECTOR> LINEARSOLVER;

typedef NewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR> NLS;

typedef StatPDEProblem<NLS, INTEGRATOR, OP, VECTOR, DIM> RP;

typedef MethodOfLines_StateSpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN,
        VECTOR, DIM> STH;

int
main(int argc, char **argv)
{
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

   pr.read_parameters(paramfile); 

   Triangulation<DIM> triangulation;

   GridGenerator::hyper_cube(triangulation, 0, 1);
   triangulation.refine_global(4);

   
   // Raviart Thomas interpolation //
   FE<DIM> state_fe(FE_Q<DIM>(2), DIM, FE_DGQ<DIM>(1), 1); //Q2Q1
   FE_RaviartThomasNodal<DIM> fe_interpolate(1);
   

   QUADRATURE quadrature_formula(4);
   FACEQUADRATURE face_quadrature_formula(4);
   
   //Components needed for interpolatedintegratordatacontainer //
   FEValuesExtractors::Vector velocity_component(0);
   FEValuesExtractors::Scalar pressure_component(2);
   MappingQGeneric<DIM> map(1);   

   IDC idc(velocity_component, map, fe_interpolate, quadrature_formula, face_quadrature_formula);
   LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM> LPDE;

   STH DOFH(triangulation, state_fe);

   OP P(LPDE, DOFH);
   
   //Specification of the dirichlet values
   // We define the boundary values in myfunctions.h and pass it to
   // SimpleDirichletData< >
   BoundaryValues<DIM> boundary_values;
   SimpleDirichletData<VECTOR, DIM> DD2(boundary_values);

   //Next, we define on which boundaries (identified via
   //boundary-colors) and which components (specified via an component mask)
   //we want to impose the dirichlet conditions and give all
   //these informations to the problemcontainer P. Note that we
   //do not impose any boundary condition on the outflow boundary (number 1).
   std::vector<bool> comp_mask(3, true);
   comp_mask[DIM] = false;
 
   P.SetDirichletBoundaryColors(0, comp_mask, &DD2);

   //We define the stateproblem, which steers the solution process.
   
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
 
      // To calculate error mostly taken from deal.II example 20.

      DoFHandler<DIM> dof_handler(triangulation);
      dof_handler.distribute_dofs(state_fe);

      DoFRenumbering::component_wise(dof_handler);

      vector<types::global_dof_index> dofs_per_component(DIM + 1);
      DoFTools::count_dofs_per_component(dof_handler, dofs_per_component);

      const unsigned int n_u = dofs_per_component[0] * DIM;
      const unsigned int n_p = dofs_per_component[DIM];

      VECTOR solution;

      solution.reinit(2);
      solution.block(0).reinit(n_u);
      solution.block(1).reinit(n_p);
      solution.collect_sizes();

      const ComponentSelectFunction<DIM> velocity_mask(make_pair(0, DIM), DIM + 1);
      const ComponentSelectFunction<DIM> pressure_mask(DIM, DIM + 1);

      ExactSolution<DIM>	exact_solution;
      Vector<double>		cellwise_errors(triangulation.n_active_cells());

      QGauss<DIM>		qgauss_error(4);
      SolutionExtractor<RP, VECTOR> 	a1(solver);
      const StateVector<VECTOR>		&gu = a1.GetU();
      solution		= gu.GetSpacialVector();

      VectorTools::integrate_difference(dof_handler, solution, exact_solution,
		cellwise_errors, qgauss_error, VectorTools::L2_norm, &velocity_mask);
    
      const double v_l2_error = VectorTools::compute_global_error(triangulation,
			cellwise_errors, VectorTools::L2_norm);

      VectorTools::integrate_difference(dof_handler, solution, exact_solution,
		cellwise_errors, qgauss_error, VectorTools::H1_norm, &velocity_mask);
    
      const double v_h1_error = VectorTools::compute_global_error(triangulation,
			cellwise_errors, VectorTools::H1_norm);

      VectorTools::integrate_difference(dof_handler, solution, exact_solution,
		cellwise_errors, qgauss_error, VectorTools::L2_norm, &pressure_mask);
    
      const double p_l2_error = VectorTools::compute_global_error(triangulation,
			cellwise_errors, VectorTools::L2_norm);
    
      cout << " Errors : ||e_v||_l2 : " << v_l2_error <<  
		",||v||_h1 : " << v_h1_error << ", ||e_p||_l2 : " << 
		p_l2_error << endl;


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
