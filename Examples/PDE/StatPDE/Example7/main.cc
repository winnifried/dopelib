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
#include "integrator.h"
#include "parameterreader.h"
#include "mol_statespacetimehandler.h"
#include "simpledirichletdata.h"
#include "celldatacontainer.h"
#include "facedatacontainer.h"
#include "sparsitymaker.h"
#include "userdefineddofconstraints.h"
#include "integratordatacontainer.h"

#include <iostream>
#include <fstream>

#include <grid/tria.h>
#include <grid/grid_in.h>
#include <dofs/dof_handler.h>
#include <grid/grid_generator.h>
#include <fe/fe_q.h>
#include <dofs/dof_tools.h>
#include <base/quadrature_lib.h>
#include <base/function.h>

#include "localpde.h"
#include "functionals.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

#define MATRIX BlockSparseMatrix<double>
#define SPARSITYPATTERN BlockSparsityPattern
#define VECTOR BlockVector<double>
#define DOFHANDLER DoFHandler<2>
#define FE FESystem<2>

//#define MATRIX SparseMatrix<double>
//#define SPARSITYPATTERN SparsityPattern
//#define VECTOR Vector<double>

typedef PDEProblemContainer<
    PDEInterface<CellDataContainer, FaceDataContainer, DOFHANDLER, VECTOR, 2>,
    DirichletDataInterface<VECTOR, 2>, SPARSITYPATTERN, VECTOR, 2> OP;
typedef IntegratorDataContainer<DOFHANDLER, Quadrature<2>,
    Quadrature<1>, VECTOR, 2> IDC;

typedef Integrator<IDC, VECTOR, double, 2> INTEGRATOR;

typedef DirectLinearSolverWithMatrix<SPARSITYPATTERN, MATRIX, VECTOR, 2> LINEARSOLVER;

typedef NewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR, 2> NLS;
typedef StatPDEProblem<NLS, INTEGRATOR, OP, VECTOR, 2> SSolver;

int
main(int argc, char **argv)
{
  /**
   *  Solving the standard Laplace equation
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
  SSolver::declare_params(pr);
  DOpEOutputHandler<VECTOR>::declare_params(pr);
  pr.read_parameters(paramfile);

  Triangulation<2> triangulation;

  FESystem<2> state_fe(FE_Q<2>(1), 2);

  QGauss<2> quadrature_formula(3);
  QGauss<1> face_quadrature_formula(3);
  IDC idc(quadrature_formula, face_quadrature_formula);

  LocalPDE<VECTOR, 2> LPDE;

  LocalPointFunctionalX<VECTOR, 2> LPFX;

  // Pseudo time
  std::vector<double> times(1, 0.);

  // Spatial grid
  GridGenerator::hyper_cube(triangulation, 0, 1);
  triangulation.refine_global(3);

  MethodOfLines_StateSpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR, 2> DOFH(
      triangulation, state_fe);

  OP P(LPDE, DOFH);

  P.AddFunctional(&LPFX);

  std::vector<bool> comp_mask(2);

  comp_mask[0] = true;
  comp_mask[1] = true;

  DOpEWrapper::ZeroFunction<2> zf(2);
  SimpleDirichletData<VECTOR, 2> DD1(zf);

  P.SetDirichletBoundaryColors(0, comp_mask, &DD1);

  SSolver solver(&P, "fullmem", pr, idc);
  //Only needed for pure PDE Problems
  DOpEOutputHandler<VECTOR> out(&solver, pr);
  DOpEExceptionHandler<VECTOR> ex(&out);
  P.RegisterOutputHandler(&out);
  P.RegisterExceptionHandler(&ex);
  solver.RegisterOutputHandler(&out);
  solver.RegisterExceptionHandler(&ex);

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
