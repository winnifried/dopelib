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
#include "statpdeproblem.h"
#include "newtonsolver.h"
#include "integrator.h"
#include "parameterreader.h"
#include "mol_statespacetimehandler.h"
#include "simpledirichletdata.h"
#include "sparsitymaker.h"
#include "integratordatacontainer.h"
#include "preconditioner_wrapper.h"

#include <iostream>
#include <fstream>

#include <grid/tria.h>
#include <grid/grid_in.h>
#include <dofs/dof_handler.h>
#include <grid/grid_generator.h>
#include <grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_out.h>
#include <fe/fe_dgq.h>
#include <dofs/dof_tools.h>
#include <base/quadrature_lib.h>
#include <base/function.h>
#include <numerics/vector_tools.h>
#include <fe/fe_raviart_thomas.h>

#include "localpde.h"
#include "localfunctional.h"
#include "schurlinearsolver.h"

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
typedef BlockSparseMatrix<double> MATRIX;
typedef BlockSparsityPattern SPARSITYPATTERN;
typedef BlockVector<double> VECTOR;

typedef PDEProblemContainer<LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
    SimpleDirichletData<VECTOR, DIM>, SPARSITYPATTERN, VECTOR, DIM, FE,
    DOFHANDLER> OP;
typedef IntegratorDataContainer<DOFHANDLER, QUADRATURE, FACEQUADRATURE, VECTOR,
    DIM> IDC;
typedef Integrator<IDC, VECTOR, double, DIM> INTEGRATOR;
typedef SchurLinearSolverWithMatrix LINEARSOLVER;
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
}

int
main(int argc, char **argv)
{
  /**
   *  In this example solve the vector valued Laplace equation
   *  in mixed formulation. The novelty is the use of 
   *  Raviart-Thomas elements together with a schur complement solver.
   *  This is the DOpE implementation of the deal.ii example 
   *  step-20.
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

  //define some constants
  pr.SetSubsection("main parameters");
  const int max_iter = pr.get_integer("max_iter");
 
  Triangulation<DIM> triangulation;
 
  GridGenerator::hyper_cube (triangulation, -1, 1);

  FE<DIM> state_fe(FE_RaviartThomasNodal<DIM>(0),1,FE_DGQ<DIM>(0),1);

  QUADRATURE quadrature_formula(2);
  FACEQUADRATURE face_quadrature_formula(1);
  IDC idc(quadrature_formula, face_quadrature_formula);

  LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM> LPDE;
  
  //We need to state the mapping explicitly, otherwise the divergence of 
  //the RT-elements provided by dealii will be NaN.
  DOpEWrapper::Mapping<DIM,DOFHANDLER> mapping(1,true);

  LocalFunctional<CDC,FDC,DOFHANDLER, VECTOR, DIM> LF;

  STH DOFH(triangulation, mapping, state_fe);

  OP P(LPDE, DOFH);

  P.SetBoundaryEquationColors(0);
  P.AddFunctional(&LF);

  RP solver(&P, DOpEtypes::VectorStorageType::fullmem, pr, idc);

  //Only needed for pure PDE Problems
  DOpEOutputHandler<VECTOR> out(&solver, pr);
  DOpEExceptionHandler<VECTOR> ex(&out);

  P.RegisterOutputHandler(&out);
  P.RegisterExceptionHandler(&ex);

  solver.RegisterOutputHandler(&out);
  solver.RegisterExceptionHandler(&ex);

  VECTOR solution;

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
      outp << "**************************************************\n";
      out.Write(outp, 1, 1, 1);

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
      DOFH.RefineSpace();
    }
  }

  return 0;
}
#undef FDC
#undef CDC
#undef FE
#undef DOFHANDLER
