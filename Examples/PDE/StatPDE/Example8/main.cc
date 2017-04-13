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
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/base/quadrature_lib.h>

#include <container/pdeproblemcontainer.h>
#include <interfaces/functionalinterface.h>
#include <interfaces/pdeinterface.h>
#include <reducedproblems/statpdeproblem.h>
#include <templates/newtonsolver.h>
#include <templates/directlinearsolver.h>
#include <templates/integrator.h>
#include <include/parameterreader.h>
#include <basic/mol_statespacetimehandler.h>
#include <problemdata/simpledirichletdata.h>
#include <container/integratordatacontainer.h>

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

typedef QSimpson<DIM> QUADRATURE;
typedef QSimpson<DIM - 1> FACEQUADRATURE;
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
   *  In this example we solve an stationary plasticity benchmark, see
   *  E. Stein (editor), Error-controlled Adaptive Finite Elements in Solid Mechanics,
   Wiley (2003), pp. 386 - 387
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

  unsigned int niter = 2;

  Triangulation<DIM> triangulation;

  const Point<DIM> center(100, 0);
  const double inner_radius = 10.0;

  GridIn<DIM> grid_in;
  grid_in.attach_triangulation(triangulation);
  std::ifstream input_file("benchmark_2.inp");
  grid_in.read_ucd(input_file);

  static const HyperBallBoundary<DIM> boundary(center, inner_radius);
  triangulation.set_boundary(1, boundary);
  triangulation.refine_global(4);
  {
    std::ofstream out("grid.eps");
    GridOut grid_out;
    grid_out.write_eps(triangulation, out);
  }

  FE<DIM> state_fe(FE_Q<DIM>(1), 2); //Q1

  QUADRATURE quadrature_formula;
  FACEQUADRATURE face_quadrature_formula;
  IDC idc(quadrature_formula, face_quadrature_formula);

  LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM> LPDE;

  LocalPointFunctionalDisp_1<CDC, FDC, DOFHANDLER, VECTOR, DIM> LPFD_1;
  LocalPointFunctionalDisp_2<CDC, FDC, DOFHANDLER, VECTOR, DIM> LPFD_2;
  LocalPointFunctionalDisp_3<CDC, FDC, DOFHANDLER, VECTOR, DIM> LPFD_3;
  LocalDomainFunctionalStress<CDC, FDC, DOFHANDLER, VECTOR, DIM> LDFS;

  LocalBoundaryFaceFunctionalUpBd<CDC, FDC, DOFHANDLER, VECTOR, DIM> LBFUB;

  STH DOFH(triangulation, state_fe);

  OP P(LPDE, DOFH);

  P.AddFunctional(&LPFD_1); // x-displacement in (45,0)
  P.AddFunctional(&LPFD_2); // y-displacement in (50,50)
  P.AddFunctional(&LPFD_3); // x-displacement in (0,50)
  P.AddFunctional(&LDFS); // yy-stress in (45,0)
  P.AddFunctional(&LBFUB); // y-displacement-integral on upper boundary

  P.SetBoundaryFunctionalColors(3);

  DOpEWrapper::ZeroFunction<DIM> zf_1(2);
  SimpleDirichletData<VECTOR, DIM> DD1(zf_1);
  std::vector<bool> comp_mask(2);
  comp_mask[0] = true;
  comp_mask[1] = false;
  P.SetDirichletBoundaryColors(2, comp_mask, &DD1);

  comp_mask[0] = false;
  comp_mask[1] = true;
  P.SetDirichletBoundaryColors(0, comp_mask, &DD1);
  P.SetBoundaryEquationColors(3);

  RP solver(&P, DOpEtypes::VectorStorageType::fullmem, pr, idc);

  //Only needed for pure PDE Problems
  DOpEOutputHandler<VECTOR> out(&solver, pr);
  DOpEExceptionHandler<VECTOR> ex(&out);
  P.RegisterOutputHandler(&out);
  P.RegisterExceptionHandler(&ex);
  solver.RegisterOutputHandler(&out);
  solver.RegisterExceptionHandler(&ex);

  solver.ReInit();
  out.ReInit();

  for (unsigned int iter = 0; iter < niter; iter++)
    {
      try
        {
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
      if (iter != niter - 1)
        {
          DOFH.RefineSpace();
          solver.ReInit();
          out.ReInit();
        }
    }

  return 0;
}
#undef FDC
#undef CDC
#undef FE
#undef DOFHANDLER
