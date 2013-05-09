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
#include "indexsetter.h"
#include "my_functions.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

#define MATRIX BlockSparseMatrix<double>
#define SPARSITYPATTERN BlockSparsityPattern
#define VECTOR BlockVector<double>
#define DOFHANDLER hp::DoFHandler
#define FE hp::FECollection
#define QUADRATURE hp::QCollection<2>
#define FACEQUADRATURE hp::QCollection<1>


typedef PDEProblemContainer<LocalPDE<DOFHANDLER, VECTOR, 2> ,
DirichletDataInterface<VECTOR, 2> ,SPARSITYPATTERN,VECTOR, 2,FE,DOFHANDLER> OP;
typedef IntegratorDataContainer<DOFHANDLER, QUADRATURE, FACEQUADRATURE, VECTOR,
    2> IDC;

typedef Integrator<IDC, VECTOR, double, 2> INTEGRATOR;

typedef DirectLinearSolverWithMatrix<SPARSITYPATTERN, MATRIX, VECTOR, 2> LINEARSOLVER;

typedef NewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR, 2> NLS;
typedef StatPDEProblem<NLS, INTEGRATOR, OP, VECTOR, 2> SSolver;

int
main(int argc, char **argv)
{
  /**
   *  In this example we solve stationary (linear) Stokes' equations
   *  with symmetric stress tensor and do-nothing condition on
   *  the outflow boundary. In this case we employ an additional
   *  term on the outflow boundary due the symmetry of the stress tensor.
   *
   *  To make this run with hp, just replace the indicated parts.
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

  GridIn<2> grid_in;
  grid_in.attach_triangulation(triangulation);
  std::ifstream input_file("channel.inp");
  grid_in.read_ucd(input_file);

  FESystem<2> state_fe(FE_Q<2>(2), 2, FE_Q<2>(1), 1); //Q2Q1

  /******hp******************/
  FESystem<2> state_fe_2(FE_Q<2>(3), 2, FE_Q<2>(2), 1); //Q3Q2
  hp::FECollection < 2 > state_fe_collection(state_fe);
  state_fe_collection.push_back(state_fe_2);
  /******hp******************/

  QGauss<2> quadrature_formula(3);
  QGauss<1> face_quadrature_formula(3);

  /******hp******************/
  QGauss<2> quadrature_formula_2(4);
  QGauss<1> face_quadrature_formula_2(4);
  hp::QCollection<2> q_coll(quadrature_formula);
  q_coll.push_back(quadrature_formula_2);
  hp::QCollection<1> face_q_coll(face_quadrature_formula);
  face_q_coll.push_back(face_quadrature_formula_2);
  IDC idc(q_coll, face_q_coll);
  /******hp******************/

  LocalPDE<DOFHANDLER, VECTOR, 2> LPDE;

  LocalPointFunctionalX<DOFHANDLER, VECTOR, 2> LPFX;
  LocalBoundaryFluxFunctional<DOFHANDLER, VECTOR, 2> LBFF;

  triangulation.refine_global(3);

  
  /***************hp********************/
  ActiveFEIndexSetter<2> indexsetter(pr);
  MethodOfLines_StateSpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR, 2> DOFH(
      triangulation, state_fe_collection, indexsetter);
  /***************hp********************/

  OP P(LPDE, DOFH);

  P.AddFunctional(&LPFX);
  P.AddFunctional(&LBFF);

  // fuer Flux Auswertung am Ausflussrand
  P.SetBoundaryFunctionalColors(1);

  std::vector<bool> comp_mask(3);

  comp_mask[0] = true;
  comp_mask[1] = true;
  comp_mask[2] = false;

  DOpEWrapper::ZeroFunction < 2 > zf(3);
  SimpleDirichletData<VECTOR, 2> DD1(zf);

  BoundaryParabel boundary_parabel;
  SimpleDirichletData<VECTOR, 2> DD2(boundary_parabel);

  P.SetDirichletBoundaryColors(0, comp_mask, &DD2);

  P.SetDirichletBoundaryColors(2, comp_mask, &DD1);
  P.SetDirichletBoundaryColors(3, comp_mask, &DD1);

  P.SetBoundaryEquationColors(1);

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
