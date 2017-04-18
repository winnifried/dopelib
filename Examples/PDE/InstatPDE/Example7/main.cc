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

#include <interfaces/functionalinterface.h>
#include <interfaces/pdeinterface.h>
#include <templates/newtonsolver.h>
#include <templates/richardsonlinearsolver.h>
#include <wrapper/preconditioner_wrapper.h>
#include <include/sparsitymaker.h>
#include <container/integratordatacontainer.h>

#include <templates/integrator.h>
#include <include/parameterreader.h>

#include <basic/mol_statespacetimehandler.h>
#include <problemdata/simpledirichletdata.h>
#include <interfaces/active_fe_index_setter_interface.h>

#include <reducedproblems/instatpdeproblem.h>
#include <templates/instat_step_newtonsolver.h>
#include <container/instatpdeproblemcontainer.h>
#include <tsschemes/shifted_crank_nicolson_problem.h>
#include <tsschemes/backward_euler_problem.h>

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

typedef PDEProblemContainer<
LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
         SimpleDirichletData<VECTOR, DIM>,
         SPARSITYPATTERN,
         VECTOR, DIM> OP_BASE;
typedef StateProblem<OP_BASE, LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
        SimpleDirichletData<VECTOR, DIM>, SPARSITYPATTERN, VECTOR, DIM> PROB;

// Typedefs for timestep problem
#define TSP ShiftedCrankNicolsonProblem
//#define TSP BackwardEulerProblem
//FIXME: This should be a reasonable dual timestepping scheme
#define DTSP ShiftedCrankNicolsonProblem
typedef InstatPDEProblemContainer<TSP, DTSP,
        LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
        SimpleDirichletData<VECTOR, DIM>,
        SPARSITYPATTERN,
        VECTOR, DIM> OP;
#undef TSP
#undef DTSP

typedef IntegratorDataContainer<DOFHANDLER, QUADRATURE, FACEQUADRATURE,
        VECTOR, DIM> IDC;
typedef Integrator<IDC, VECTOR, double, DIM> INTEGRATOR;
typedef RichardsonLinearSolverWithMatrix<PRECONDITIONERSSOR, SPARSITYPATTERN, MATRIX, VECTOR> LINEARSOLVER;

typedef NewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR> CNLS;
typedef InstatStepNewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR> NLS;
typedef InstatPDEProblem<NLS, INTEGRATOR, OP, VECTOR, DIM> RP;

typedef MethodOfLines_StateSpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN,
        VECTOR, DIM> STH;

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
  STH DOFH(triangulation, state_fe, times, true);
  /***********************************/

  OP P(LPDE, DOFH);
  P.AddFunctional(&MVF);
  //Boundary conditions************************************************
  P.SetBoundaryEquationColors(0);
  P.SetBoundaryEquationColors(1);
  /************************************************/
  //prepare the initial data
  InitialData initial_data;
  P.SetInitialValues(&initial_data);


  RP solver(&P, DOpEtypes::VectorStorageType::fullmem, pr, idc);

  DOpEOutputHandler<VECTOR> out(&solver, pr);
  DOpEExceptionHandler<VECTOR> ex(&out);
  P.RegisterOutputHandler(&out);
  P.RegisterExceptionHandler(&ex);
  solver.RegisterOutputHandler(&out);
  solver.RegisterExceptionHandler(&ex);


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
