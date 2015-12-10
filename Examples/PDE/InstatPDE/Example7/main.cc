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
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>

#include "functionalinterface.h"
#include "pdeinterface.h"
#include "newtonsolver.h"
#include "richardsonlinearsolver.h"
#include "preconditioner_wrapper.h"
#include "sparsitymaker.h"
#include "integratordatacontainer.h"
#include "functionalinterface.h"
#include "noconstraints.h"

#include "integrator.h"
#include "parameterreader.h"

#include "mol_spacetimehandler.h"
#include "simpledirichletdata.h"
#include "active_fe_index_setter_interface.h"

#include "instatreducedproblem.h"
#include "instat_step_newtonsolver.h"
#include "reducednewtonalgorithm.h"
#include "instatoptproblemcontainer.h"
#include "shifted_crank_nicolson_problem.h"
#include "backward_euler_problem.h"

#include "localpde.h"
#include "functionals.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

const static int DIM = 1;

#define DOFHANDLER DoFHandler
#define FE FESystem
#define CDC ElementDataContainer
#define FDC FaceDataContainer

typedef QGauss<DIM> QUADRATURE;
typedef QGauss<DIM - 1> FACEQUADRATURE;
typedef SparseMatrix<double> MATRIX;
typedef SparsityPattern SPARSITYPATTERN;
typedef Vector<double> VECTOR;

//Second number denotes the number of unknowns per element (the blocksize we want to use)
typedef DOpEWrapper::PreconditionBlockSSOR_Wrapper<MATRIX,2> PRECONDITIONERSSOR; 

typedef FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM> FUNC;

typedef OptProblemContainer<
    LocalFunctional<CDC, FDC, DOFHANDLER, VECTOR, DIM>, FUNC,
    LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
    SimpleDirichletData<VECTOR, DIM>,
    NoConstraints<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM>, SPARSITYPATTERN,
    VECTOR, DIM, DIM> OP_BASE;
typedef StateProblem<OP_BASE, LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
    SimpleDirichletData<VECTOR, DIM>, SPARSITYPATTERN, VECTOR, DIM> PROB;

// Typedefs for timestep problem
#define TSP ShiftedCrankNicolsonProblem
//#define TSP BackwardEulerProblem
//FIXME: This should be a reasonable dual timestepping scheme
#define DTSP ShiftedCrankNicolsonProblem
typedef InstatOptProblemContainer<TSP, DTSP, FUNC,
    LocalFunctional<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
    LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
    SimpleDirichletData<VECTOR, DIM>,
    NoConstraints<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM>, SPARSITYPATTERN,
    VECTOR, DIM, DIM> OP;
#undef TSP
#undef DTSP

typedef IntegratorDataContainer<DOFHANDLER, QUADRATURE, FACEQUADRATURE,
    VECTOR, DIM> IDC;
typedef Integrator<IDC, VECTOR, double, DIM> INTEGRATOR;
typedef RichardsonLinearSolverWithMatrix<PRECONDITIONERSSOR, SPARSITYPATTERN, MATRIX, VECTOR> LINEARSOLVER;

typedef NewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR> CNLS;
typedef InstatStepNewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR> NLS;
typedef ReducedNewtonAlgorithm<OP, VECTOR> RNA;
typedef InstatReducedProblem<CNLS, NLS, INTEGRATOR, INTEGRATOR, OP, VECTOR, DIM,
    DIM> RP;

typedef MethodOfLines_SpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN,
				       VECTOR, DIM, DIM> STH;

void
declare_params(ParameterReader &param_reader)
{
  param_reader.SetSubsection("main parameters");
  param_reader.declare_entry("max_iter", "1", Patterns::Integer(0),
      "How many iterations?");
  param_reader.declare_entry("prerefine", "1", Patterns::Integer(1),
      "How often should we refine the coarse grid?");
}

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
  RNA::declare_params(pr);
  LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>::declare_params(pr);
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
  Triangulation<DIM> triangulation;
  GridGenerator::hyper_cube(triangulation, 0, 2);
  triangulation.refine_global(prerefine);
  //*************************************************************

  //FiniteElemente*************************************************
  FESystem<DIM> control_fe(FE_Nothing<DIM>(), 1);
  FE<DIM> state_fe(FE_DGQ<DIM>(0), 2);

  //Quadrature formulas*************************************************
  pr.SetSubsection("main parameters");
  QGauss<DIM> quadrature_formula(1);
  QGauss<DIM-1> face_quadrature_formula(1);
  IDC idc(quadrature_formula, face_quadrature_formula);
  //**************************************************************************

  //Functionals*************************************************
  LocalFunctional<CDC, FDC, DOFHANDLER, VECTOR, DIM> MVF;
  //*************************************************
  
  //pde*************************************************
  LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM> LPDE(pr);
  //*************************************************


  //Time grid of [0,1]
  Triangulation<1> times;
  GridGenerator::hyper_cube(times,0, 1);
  times.refine_global(prerefine);

  //space time handler***********************************/
  STH DOFH(triangulation, control_fe, state_fe, times, DOpEtypes::ControlType::stationary, true);
  /***********************************/
  NoConstraints<ElementDataContainer, FaceDataContainer, DOFHANDLER, VECTOR, DIM,
		DIM> Constraints;

  OP P(MVF, LPDE, Constraints, DOFH);
  //Boundary conditions************************************************
  P.SetBoundaryEquationColors(0);
  P.SetBoundaryEquationColors(1);
  /************************************************/
  //prepare the initial data
  InitialData initial_data;
  P.SetInitialValues(&initial_data);


  RP solver(&P, DOpEtypes::VectorStorageType::fullmem, pr, idc);

  RNA Alg(&P, &solver, pr);

  
  //**************************************************************************************************

  for (int i = 0; i < max_iter; i++)
  {
    try
    {
      Alg.ReInit();
      ControlVector<VECTOR> q(&DOFH, DOpEtypes::VectorStorageType::fullmem);
      
      Alg.SolveForward(q);
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
      //For global mesh refinement, uncomment the next line
      DOFH.RefineSpaceTime(DOpEtypes::RefinementType::global); //or just DOFH.RefineSpace()
    }
  }
  return 0;
}
#undef FDC
#undef CDC
#undef FE
#undef DOFHANDLER
