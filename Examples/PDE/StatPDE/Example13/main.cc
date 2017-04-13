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
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>

#include <container/pdeproblemcontainer.h>
#include <interfaces/functionalinterface.h>
#include <interfaces/pdeinterface.h>
#include <reducedproblems/statpdeproblem.h>
#include <templates/newtonsolver.h>
#include <templates/directlinearsolver.h>
#include <templates/richardsonlinearsolver.h>
#include <wrapper/preconditioner_wrapper.h>
#include <include/userdefineddofconstraints.h>
#include <include/sparsitymaker.h>
#include <container/integratordatacontainer.h>

#include <templates/integrator.h>
#include <include/parameterreader.h>

#include <basic/mol_statespacetimehandler.h>
#include <problemdata/simpledirichletdata.h>
#include <interfaces/active_fe_index_setter_interface.h>

#include "localpde.h"
#include "functionals.h"

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

//Second number denotes the number of unknowns per element (the blocksize we want to use)
typedef DOpEWrapper::PreconditionBlockSSOR_Wrapper<MATRIX,4> PRECONDITIONERSSOR;

typedef PDEProblemContainer<LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
        SimpleDirichletData<VECTOR, DIM>, SPARSITYPATTERN, VECTOR, DIM> OP;
typedef IntegratorDataContainer<DOFHANDLER, QUADRATURE, FACEQUADRATURE,
        VECTOR, DIM> IDC;
typedef Integrator<IDC, VECTOR, double, DIM> INTEGRATOR;
//typedef DirectLinearSolverWithMatrix<SPARSITYPATTERN, MATRIX, VECTOR> LINEARSOLVER;
typedef RichardsonLinearSolverWithMatrix<PRECONDITIONERSSOR, SPARSITYPATTERN, MATRIX, VECTOR> LINEARSOLVER;

typedef NewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR> NLS;
typedef StatPDEProblem<NLS, INTEGRATOR, OP, VECTOR, DIM> RP;
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
  /**
   * We solve the standard laplace equation in 2d. The
   * main feature is the use of the DWR method for error
   * estimation and grid refinement.
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
  //GridGenerator::hyper_cube(triangulation, 0, 1,true);
  GridGenerator::hyper_rectangle(triangulation, Point<2>(0,0), Point<2>(1,1),true);
  triangulation.refine_global(prerefine);
  //*************************************************************

  //FiniteElemente*************************************************
  FE<DIM> state_fe(FE_DGQ<DIM>(1), 1);

  //Quadrature formulas*************************************************
  pr.SetSubsection("main parameters");
  QGauss<DIM> quadrature_formula(2);
  QGauss<1> face_quadrature_formula(2);
  IDC idc(quadrature_formula, face_quadrature_formula);
  //**************************************************************************

  //Functionals*************************************************
  MeanValueFunctional<CDC, FDC, DOFHANDLER, VECTOR, DIM> MVF;
  //*************************************************

  //pde*************************************************
  LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM> LPDE;
  //*************************************************

  //space time handler***********************************/
  STH DOFH(triangulation, state_fe, true);
  /***********************************/

  OP P(LPDE, DOFH);
  P.AddFunctional(&MVF);
  //Boundary conditions************************************************
  P.SetBoundaryEquationColors(0);
  P.SetBoundaryEquationColors(1);
  P.SetBoundaryEquationColors(2);
  P.SetBoundaryEquationColors(3);
  /************************************************/
  RP solver(&P, DOpEtypes::VectorStorageType::fullmem, pr, idc);

  //Only needed for pure PDE Problems
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
          solver.ReInit();
          out.ReInit();
          stringstream outp;

          outp << "**************************************************\n";
          outp << "*             Starting Forward Solve             *\n";
          outp << "*   Solving : " << P.GetName() << "\t*\n";
          outp << "*   SDoFs   : ";
          solver.StateSizeInfo(outp);
          outp << "**************************************************";
          out.Write(outp, 1, 1, 1);

          solver.ComputeReducedFunctionals();

          const double exact_value = 0.25*0.25*M_PI;

          double error = exact_value - solver.GetFunctionalValue(MVF.GetName());
          outp << "Mean value: " <<solver.GetFunctionalValue(MVF.GetName()) << " - Mean value error: " << error << std::endl;
          out.Write(outp, 1, 1, 1);
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
          DOFH.RefineSpace(DOpEtypes::RefinementType::global); //or just DOFH.RefineSpace()
        }
    }
  return 0;
}
#undef FDC
#undef CDC
#undef FE
#undef DOFHANDLER
