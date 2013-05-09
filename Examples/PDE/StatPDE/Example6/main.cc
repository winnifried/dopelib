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

#include "pdeproblemcontainer.h"
#include "functionalinterface.h"
#include "pdeinterface.h"
#include "statpdeproblem.h"
#include "newtonsolver.h"
#include "directlinearsolver.h"
#include "userdefineddofconstraints.h"
#include "myconstraintsmaker.h"
#include "sparsitymaker.h"
#include "integratordatacontainer.h"

#include "integrator.h"
#include "parameterreader.h"

#include "mol_statespacetimehandler.h"
#include "simpledirichletdata.h"
#include "active_fe_index_setter_interface.h"

#include <iostream>
#include <fstream>

#include <grid/tria.h>
#include <grid/grid_in.h>
#include <grid/tria_boundary_lib.h>
#include <dofs/dof_handler.h>
#include <grid/grid_generator.h>
#include <fe/fe_q.h>
#include <fe/fe_nothing.h>
#include <dofs/dof_tools.h>
#include <base/quadrature_lib.h>
#include <base/function.h>

#include "localpde.h"
#include "functionals.h"

using namespace std;
using namespace dealii;
using namespace DOpE;



#define VECTOR Vector<double>
#define MATRIX SparseMatrix<double>
#define SPARSITYPATTERN SparsityPattern
#define DOFHANDLER DoFHandler
#define FE FESystem
#define CDC CellDataContainer
#define FDC FaceDataContainer

typedef PDEProblemContainer<LocalPDELaplace<CDC,FDC,DOFHANDLER, VECTOR, 2> , DirichletDataInterface<VECTOR, 2> ,
			    SPARSITYPATTERN, VECTOR, 2> OP;
typedef IntegratorDataContainer<DOFHANDLER, Quadrature<2>, Quadrature<1>, VECTOR, 2 > IDC;
typedef Integrator<IDC, VECTOR, double, 2> INTEGRATOR;
//********************Linearsolver**********************************
typedef DirectLinearSolverWithMatrix<SPARSITYPATTERN, MATRIX,
    VECTOR> LINEARSOLVER;
//********************Linearsolver**********************************

typedef NewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR> NLS;
typedef StatPDEProblem<NLS, INTEGRATOR, OP, VECTOR, 2> SSolver;

void declare_params(ParameterReader &param_reader)
{
	param_reader.SetSubsection("main parameters");
	param_reader.declare_entry("max_iter", "1", Patterns::Integer(0),
	                           "How many iterations?");
	param_reader.declare_entry("quad order", "2", Patterns::Integer(1),
	                           "Order of the quad formula?");
	param_reader.declare_entry("facequad order", "2", Patterns::Integer(1),
	                           "Order of the face quad formula?");
	param_reader.declare_entry("order fe", "2", Patterns::Integer(1),
	                           "Order of the finite element?");
	param_reader.declare_entry("prerefine", "1", Patterns::Integer(1),
	                           "How often should we refine the coarse grid?");
}

int main(int argc, char **argv)
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

	SSolver::declare_params(pr);
	DOpEOutputHandler<VECTOR>::declare_params(pr);
 	declare_params(pr);

	pr.read_parameters(paramfile);

	//************************************************
	//define some constants
	pr.SetSubsection("main parameters");
	int max_iter = pr.get_integer("max_iter");
	int prerefine = pr.get_integer("prerefine");

	//*************************************************

	//Make triangulation *************************************************
	const Point<2> center(0, 0);
	const HyperShellBoundary<2> boundary_description(center);
	Triangulation<2> triangulation;
	GridGenerator::hyper_cube_with_cylindrical_hole(triangulation, 0.1,
	                                                        1., 1, 1, true);
	triangulation.set_boundary(4, boundary_description);
	if (prerefine > 0)
		triangulation.refine_global(prerefine);
	//*************************************************

	//FiniteElemente*************************************************
	pr.SetSubsection("main parameters");
	FE<2> state_fe(FE_Q<2>(pr.get_integer("order fe")), 2);

	//Quadrature formulas*************************************************
	pr.SetSubsection("main parameters");
	QGauss<2> quadrature_formula(pr.get_integer("quad order"));
	QGauss<1> face_quadrature_formula(pr.get_integer("facequad order"));
	IDC idc(quadrature_formula, face_quadrature_formula);
	//**************************************************************************************************

	//Functionals*************************************************
	LocalBoundaryFunctionalMassFlux<CDC,FDC,DOFHANDLER,VECTOR, 2> LBFMF;
	LocalPDELaplace<CDC,FDC,DOFHANDLER,VECTOR, 2> LPDE;
	//*************************************************

	//pseudo time*************************************************
	std::vector<double> times(1, 0.);
	//*************************************************

	/***********************************/
	DOpE::PeriodicityConstraints<DOFHANDLER,2> constraints_mkr;
	MethodOfLines_StateSpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR, 2> DOFH(triangulation,  state_fe);
	//Add the periodicity constraints through the following:
	DOFH.SetUserDefinedDoFConstraints(constraints_mkr);
	/***********************************/

	OP P(LPDE, DOFH);
	P.AddFunctional(&LBFMF);
	P.SetBoundaryFunctionalColors(1);
	//Boundary conditions************************************************
	std::vector<bool> comp_mask(2);
	comp_mask[0] = true;
	comp_mask[1] = true;

	DOpEWrapper::ZeroFunction < 2 > zf(2);
	SimpleDirichletData<VECTOR, 2> DD1(zf);
	//Set zero dirichlet at the hole in the middle of the domain
	P.SetDirichletBoundaryColors(4, comp_mask, &DD1);
	/************************************************/
	SSolver solver(&P, "fullmem", pr,idc);


        //Only needed for pure PDE Problems
        DOpEOutputHandler<VECTOR> out(&solver,pr);
        DOpEExceptionHandler<VECTOR> ex(&out);
        P.RegisterOutputHandler(&out);
        P.RegisterExceptionHandler(&ex);
        solver.RegisterOutputHandler(&out);
        solver.RegisterExceptionHandler(&ex); 
	/**********************************************************************/
	for (int i = 0; i < max_iter; i++)
	{
		try
		{  
		  solver.ReInit();
		  out.ReInit();
		  stringstream outp;
		  
		  outp << "**************************************************\n";
		  outp << "*             Starting Forward Solve             *\n";
		  outp << "*   Solving : "<<P.GetName()<<"\t*\n";
		  outp << "*   SDoFs   : ";
		  solver.StateSizeInfo(outp);
		  outp << "**************************************************";
		  out.Write(outp,1,1,1);
		  
		  solver.ComputeReducedFunctionals();
		} catch (DOpEException &e)
		{
			std::cout
			    << "Warning: During execution of `" + e.GetThrowingInstance()
			        + "` the following Problem occurred!" << std::endl;
			std::cout << e.GetErrorMessage() << std::endl;
		}
		if (i != max_iter - 1)
		{
		  DOFH.RefineSpace();
		}
	}
	return 0;
}
