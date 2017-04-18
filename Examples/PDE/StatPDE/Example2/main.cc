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
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>

#include <container/pdeproblemcontainer.h>
#include <reducedproblems/statpdeproblem.h>
#include <templates/newtonsolver.h>
#include <templates/directlinearsolver.h>
#include <include/userdefineddofconstraints.h>

#include <include/sparsitymaker.h>
#include <container/integratordatacontainer.h>
#include <templates/integrator.h>
#include <include/parameterreader.h>
#include <basic/mol_statespacetimehandler.h>
#include <problemdata/simpledirichletdata.h>

#include "localpde.h"
#include "functionals.h"
#include "myconstraintsmaker.h"

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

void
declare_params(ParameterReader &param_reader)
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

int
main(int argc, char **argv)
{
  /**
   * We solve the Laplace equation with periodic boundary conditions.
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
  const int max_iter = pr.get_integer("max_iter");
  const int prerefine = pr.get_integer("prerefine");

  //*************************************************

  //Make triangulation *************************************************
  const Point<DIM> center(0, 0);
  const HyperShellBoundary<DIM> boundary_description(center);
  Triangulation<DIM> triangulation;
  GridGenerator::hyper_cube_with_cylindrical_hole(triangulation, 0.1, 1., 1, 1,
                                                  true);
  triangulation.set_boundary(4, boundary_description);
  if (prerefine > 0)
    triangulation.refine_global(prerefine);
  //*************************************************

  //FiniteElemente*************************************************
  pr.SetSubsection("main parameters");
  FE<DIM> state_fe(FE_Q<DIM>(pr.get_integer("order fe")), 2);

  //Quadrature formulas*************************************************
  pr.SetSubsection("main parameters");
  QUADRATURE quadrature_formula(pr.get_integer("quad order"));
  FACEQUADRATURE face_quadrature_formula(pr.get_integer("facequad order"));
  IDC idc(quadrature_formula, face_quadrature_formula);
  //**************************************************************************************************

  //Functional and PDE*************************************************
  LocalBoundaryFunctionalMassFlux<CDC, FDC, DOFHANDLER, VECTOR, DIM> LBFMF;
  LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM> LPDE;
  //*************************************************

  /***********************************/
  //Here we set declare the periodicityconstraints and give them to the SpaceTimeHandler.
  PeriodicityConstraints<DOFHANDLER, DIM> constraints_mkr;
  STH DOFH(triangulation, state_fe);
  //Add the periodicity constraints through the following:
  DOFH.SetUserDefinedDoFConstraints(constraints_mkr);
  /***********************************/

  OP P(LPDE, DOFH);
  P.AddFunctional(&LBFMF);
  P.SetBoundaryFunctionalColors(1);
  //Boundary conditions************************************************
  std::vector<bool> comp_mask(2, true);
  DOpEWrapper::ZeroFunction<DIM> zf(2);
  SimpleDirichletData<VECTOR, DIM> DD1(zf);
  //Set zero dirichlet at the hole in the middle of the domain
  P.SetDirichletBoundaryColors(4, comp_mask, &DD1);
  /************************************************/
  RP solver(&P, DOpEtypes::VectorStorageType::fullmem, pr, idc);

  //Only needed for pure PDE Problems
  DOpEOutputHandler<VECTOR> out(&solver, pr);
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
