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
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/base/quadrature_lib.h>

#include <container/pdeproblemcontainer.h>
#include <interfaces/pdeinterface.h>
#include <reducedproblems/statpdeproblem.h>
#include <templates/newtonsolver.h>
#include <templates/directlinearsolver.h>
#include <templates/integrator.h>
#include <include/parameterreader.h>
#include <basic/mol_statespacetimehandler.h>
#include <problemdata/simpledirichletdata.h>
#include <container/integratordatacontainer.h>

// Two header files localpde* in which two different material
// laws are described
#include "localpde.h" // INH with pressure filtering (normalization)
//#include "localpde_stvk_material.h" // simplified STVK (compressible)

#include "functionals.h"
#include "my_functions.h"

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
typedef DirectLinearSolverWithMatrix<SPARSITYPATTERN, MATRIX, VECTOR> LINEARSOLVER;
typedef NewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR> NLS;
typedef StatPDEProblem<NLS, INTEGRATOR, OP, VECTOR, DIM> RP;
typedef MethodOfLines_StateSpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN,
        VECTOR, DIM> STH;

int
main(int argc, char **argv)
{
  /**
   * Stationary FSI problem in an ALE framework
   * Fluid: Stokes equ.
   * Structure: Incompressible neo Hookean (INH) model or
   * in a different header file, a simplified compressible SVTK solid.
   * We use the Q2^c-Q1^c element for discretization of Stokes.
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

  pr.read_parameters(paramfile);

  //Create Triangulation
  Triangulation<DIM> triangulation;
  GridIn<DIM> grid_in;
  grid_in.attach_triangulation(triangulation);
  std::ifstream input_file("channel.inp");
  grid_in.read_ucd(input_file);
  triangulation.refine_global(2);

  //Define FE and Quadrature
  //(v_1, v_2, p, u_1, u_2) with velocity v, pressure p and displacement u
  FE<DIM> state_fe(FE_Q<DIM>(2), 2, FE_Q<DIM>(1), 1, FE_Q<DIM>(2), 2);


  QUADRATURE quadrature_formula(3);
  FACEQUADRATURE face_quadrature_formula(3);
  IDC idc(quadrature_formula, face_quadrature_formula);

  //Define PDE and Functionals
  LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM> LPDE;
  LocalPointFunctionalX<CDC, FDC, DOFHANDLER, VECTOR, DIM> LPFX;
  LocalBoundaryFluxFunctional<CDC, FDC, DOFHANDLER, VECTOR, DIM> LBFF;

  //Define SpaceTimeHandler and problemcontainer and add the functionals
  STH DOFH(triangulation, state_fe);
  OP P(LPDE, DOFH);
  P.AddFunctional(&LPFX);
  P.AddFunctional(&LBFF);

  //For Flux evaluation on the outflow boundary
  P.SetBoundaryFunctionalColors(1);

  //Describe Dirichletvalues
  DOpEWrapper::ZeroFunction<DIM> zf(5);
  SimpleDirichletData<VECTOR, DIM> DD1(zf);
  BoundaryParabel boundary_parabel;
  SimpleDirichletData<VECTOR, DIM> DD2(boundary_parabel);

  std::vector<bool> comp_mask(5, true);
  comp_mask[2] = false;

  // Inflow boundary (at left)
  P.SetDirichletBoundaryColors(0, comp_mask, &DD2);
  P.SetDirichletBoundaryColors(2, comp_mask, &DD1);
  P.SetDirichletBoundaryColors(3, comp_mask, &DD1);

  // Substract the non-symmetric part from the outflow boundary
  // (see do-nothing condition in the literature)
  P.SetBoundaryEquationColors(1);



  RP solver(&P, DOpEtypes::VectorStorageType::fullmem, pr, idc);
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
#undef FDC
#undef CDC
#undef FE
#undef DOFHANDLER
