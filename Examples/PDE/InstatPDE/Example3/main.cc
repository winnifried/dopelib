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
#include "voidlinearsolver.h"
#include "integrator.h"
#include "integratormixeddims.h"
#include "parameterreader.h"
#include "mol_spacetimehandler.h"
#include "simpledirichletdata.h"
#include "noconstraints.h"
#include "solutionextractor.h"
#include "sparsitymaker.h"
#include "userdefineddofconstraints.h"
#include "instatoptproblemcontainer.h"
#include "integratordatacontainer.h"

#include <iostream>
#include <fstream>

#include <base/quadrature_lib.h>
#include <base/function.h>

#include <grid/tria.h>
#include <grid/grid_in.h>
#include <grid/grid_generator.h>
#include <grid/tria_boundary_lib.h>

#include <dofs/dof_handler.h>
#include <dofs/dof_tools.h>

#include <fe/fe_q.h>

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

typedef OptProblemContainer<FUNC,FUNC,PDE,DD,CONS,SPARSITYPATTERN, VECTOR, LOCALDOPEDIM,LOCALDEALDIM> OP_BASE;

#define PROB StateProblem<OP_BASE,PDE,DD,SPARSITYPATTERN,VECTOR,LOCALDOPEDIM,LOCALDEALDIM>

// Typedefs for timestep problem

#define TSP ShiftedCrankNicolsonProblem
//FIXME: This should be a reasonable dual timestepping scheme
#define DTSP ShiftedCrankNicolsonProblem

typedef InstatOptProblemContainer<TSP,DTSP,FUNC,FUNC,PDE,DD,CONS,SPARSITYPATTERN, VECTOR, LOCALDOPEDIM,LOCALDEALDIM> OP;
#undef TSP
#undef DTSP


typedef IntegratorDataContainer<DOFHANDLER, Quadrature<LOCALDEALDIM>, Quadrature<LOCALDEALDIM-1>, VECTOR, LOCALDEALDIM > IDC;

typedef Integrator<IDC , VECTOR , double, LOCALDEALDIM> INTEGRATOR;

typedef DirectLinearSolverWithMatrix<SPARSITYPATTERN, MATRIX , VECTOR> LINEARSOLVER;

typedef NewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR> CNLS;
typedef InstatStepNewtonSolver<INTEGRATOR, LINEARSOLVER,VECTOR> NLS;

typedef ReducedNewtonAlgorithm<OP, VECTOR> RNA;
typedef InstatReducedProblem<CNLS, NLS, INTEGRATOR, INTEGRATOR, OP, VECTOR, LOCALDOPEDIM, LOCALDEALDIM>
SSolver;

void ColorizeTriangulation(Triangulation<2> &coarse_grid, double upper_bound)
{
	Triangulation<2>::cell_iterator cell = coarse_grid.begin();
	Triangulation<2>::cell_iterator endc = coarse_grid.end();
	for (; cell != endc; ++cell)
		for (unsigned int face = 0; face < GeometryInfo<2>::faces_per_cell; ++face)
		{
			if (std::fabs(cell->face(face)->center()(1) - (0)) < 1e-12)
			{
				cell->face(face)->set_boundary_indicator(1);
			}
			else if (std::fabs(cell->face(face)->center()(0) - (upper_bound)) < 1e-12)
			{
				cell->face(face)->set_boundary_indicator(0);
			}
			else if (std::fabs(cell->face(face)->center()(1) - (upper_bound)) < 1e-12)
			{
				cell->face(face)->set_boundary_indicator(0);
			}
			else if (std::fabs(cell->face(face)->center()(0) - (0)) < 1e-12)
			{
				cell->face(face)->set_boundary_indicator(1);
			}
		}
}

int main(int argc, char **argv)
{
	/**
	 * In this example we will solve the two dimensional Black-Scholes equation.
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

	//First, declare the parameters and read them in.
	ParameterReader pr;
	SSolver::declare_params(pr);
	RNA::declare_params(pr);
	LocalPDE<DOFHANDLER, VECTOR, LOCALDEALDIM>::declare_params(pr);
	InitialData::declare_params(pr);
	pr.SetSubsection("Discretization parameters");
	pr.declare_entry("upper bound", "0.0", Patterns::Double(0));
	pr.read_parameters(paramfile);

	std::string cases = "solve";

	//Create the triangulation.
	double upper_bound = pr.get_double("upper bound");
	Triangulation<LOCALDEALDIM> triangulation;
	GridGenerator::hyper_cube(triangulation, 0., upper_bound);
	ColorizeTriangulation(triangulation, upper_bound);

	//Define the Finite Elements and quadrature formulas for control and state.
	FESystem<LOCALDEALDIM> control_fe(FE_Q<LOCALDOPEDIM> (1), 1); //Q1 geht auch P0?
	FESystem<LOCALDEALDIM> state_fe(FE_Q<LOCALDEALDIM> (1), 1);//Q1

  /******hp******************/
//  DOpEWrapper::FECollection<2>  control_fe_collection(control_fe);
//  DOpEWrapper::FECollection<2>  state_fe_collection(state_fe);
//  /******hp******************/

	QGauss<LOCALDEALDIM> quadrature_formula(3);
	QGauss<LOCALDEALDIM - 1> face_quadrature_formula(3);
	IDC idc(quadrature_formula, face_quadrature_formula);

  /******hp******************/
//  hp::QCollection<2> q_coll(quadrature_formula);
//  hp::QCollection<1> face_q_coll(face_quadrature_formula);

  /******hp******************/

	//Define the localPDE and the functionals we are interested in. Here, LFunc is a dummy necessary for the control,
	//LPF is a SpaceTimePointevaluation
	LocalPDE<DOFHANDLER, VECTOR, LOCALDEALDIM> LPDE(pr);
	LocalFunctional<DOFHANDLER, VECTOR, LOCALDOPEDIM, LOCALDEALDIM> LFunc;
	LocalPointFunctional<DOFHANDLER, VECTOR, LOCALDOPEDIM, LOCALDEALDIM> LPF;

	//Time grid of [0,1]
  Triangulation<1> times;
  GridGenerator::subdivided_hyper_cube(times, 20);

		triangulation.refine_global(5);
		MethodOfLines_SpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR,LOCALDOPEDIM, LOCALDEALDIM> DOFH(triangulation, control_fe, state_fe, times, DOpEtypes::undefined);

	  /***************hp********************/
//	  MethodOfLines_SpaceTimeHandler<FE, DOFHANDLERSPARSITYPATTERN, VECTOR,LOCALDOPEDIM, LOCALDEALDIM> DOFH(triangulation,control_fe_collection, state_fe_collection,times);
	  /***************hp********************/

		NoConstraints<CellDataContainer,FaceDataContainer,DOFHANDLER,VECTOR, LOCALDOPEDIM, LOCALDEALDIM> Constraints;
		OP P(LFunc, LPDE, Constraints, DOFH);

		P.AddFunctional(&LPF);

		std::vector<bool> comp_mask(1);
		comp_mask[0] = true;

		//Here we use zero boundary values
		DOpEWrapper::ZeroFunction<LOCALDEALDIM> zf(1);
		SimpleDirichletData<VECTOR, LOCALDOPEDIM, LOCALDEALDIM> DD1(zf);

		P.SetDirichletBoundaryColors(0, comp_mask, &DD1);

		//prepare the initial data
		InitialData initial_data(pr);
		P.SetInitialValues(&initial_data);

		SSolver solver(&P, "store_on_disc", pr, idc);
		/***************hp********************/
//		  SSolver solver(&P,"fullmem",pr,
//		                 q_coll,
//		                 face_q_coll);
		/***************hp********************/

		RNA Alg(&P, &solver, pr);

		try
		{

			Alg.ReInit();
			ControlVector<VECTOR> q(&DOFH, "fullmem");
			if (cases == "check")
			{
				ControlVector<VECTOR> dq(q);
				Alg.CheckGrads(1., q, dq, 2);
				Alg.CheckHessian(1., q, dq, 2);
			}
			else
			{
				Alg.SolveForward(q);

				SolutionExtractor<SSolver, VECTOR> a(solver);
				const StateVector<VECTOR> &statevec = a.GetU();

				stringstream out;
				double product = statevec * statevec;
				out << " u * u = " << product << std::endl;
				P.GetOutputHandler()->Write(out, 0);

			}
		} catch (DOpEException &e)
		{
			std::cout << "Warning: During execution of `" + e.GetThrowingInstance()
			+ "` the following Problem occurred!" << std::endl;
			std::cout << e.GetErrorMessage() << std::endl;
		}

		return 0;
}

#undef LOCALDOPEDIM
#undef LOCALDEALDIM
