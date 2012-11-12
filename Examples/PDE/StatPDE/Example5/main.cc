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
#include "sparsitymaker.h"
#include "userdefineddofconstraints.h"
#include "integratordatacontainer.h"

#include <iostream>
#include <fstream>

#include <grid/tria.h>
#include <grid/grid_in.h>
#include <grid/tria_accessor.h>
#include <grid/tria_iterator.h>
#include <dofs/dof_handler.h>
#include <grid/grid_generator.h>
#include <grid/tria_boundary_lib.h>
#include <grid/grid_out.h>
#include <fe/fe_q.h>
#include <fe/fe_dgp.h>
#include <fe/fe_nothing.h>
#include <dofs/dof_tools.h>
#include <base/quadrature_lib.h>
#include <base/function.h>

#include "localpde.h"
#include "functionals.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

#define VECTOR BlockVector<double>
#define DOFHANDLER DoFHandler<2>
#define FE FESystem<2>

typedef PDEProblemContainer<
    PDEInterface<CellDataContainer, FaceDataContainer, DOFHANDLER, VECTOR, 2>,
    DirichletDataInterface<VECTOR, 2>, BlockSparsityPattern, VECTOR, 2> OP;
typedef IntegratorDataContainer<DOFHANDLER, Quadrature<2>,
    Quadrature<1>, VECTOR, 2> IDC;
typedef Integrator<IDC, VECTOR, double, 2> INTEGRATOR;

typedef DirectLinearSolverWithMatrix<BlockSparsityPattern,
    BlockSparseMatrix<double>, VECTOR, 2> LINEARSOLVER;
typedef NewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR, 2> NLS;

typedef StatPDEProblem<NLS, INTEGRATOR, OP, VECTOR, 2> SSolver;

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
  SSolver::declare_params(pr);
  DOpEOutputHandler<VECTOR>::declare_params(pr);
  pr.read_parameters(paramfile);

  Triangulation<2> triangulation;

  const Point<2> center(100, 0);
  const double inner_radius = 10.0;

  GridIn<2> grid_in;
  grid_in.attach_triangulation(triangulation);
  std::ifstream input_file("benchmark_2.inp");
  grid_in.read_ucd(input_file);

  static const HyperBallBoundary<2> boundary(center, inner_radius);
  triangulation.set_boundary(1, boundary);
  triangulation.refine_global(4);
    {
      std::ofstream out("grid.eps");
      GridOut grid_out;
      grid_out.write_eps(triangulation, out);
    }

  FESystem<2> state_fe(FE_Q<2>(1), 2); //Q1

  QSimpson<2> quadrature_formula;
  QSimpson<1> face_quadrature_formula;
  IDC idc(quadrature_formula, face_quadrature_formula);

  LocalPDE<VECTOR, 2> LPDE;

  LocalPointFunctionalDisp_1<VECTOR, 2> LPFD_1;
  LocalPointFunctionalDisp_2<VECTOR, 2> LPFD_2;
  LocalPointFunctionalDisp_3<VECTOR, 2> LPFD_3;
  LocalDomainFunctionalStress<VECTOR, 2> LDFS;

  LocalBoundaryFaceFunctionalUpBd<VECTOR, 2> LBFUB;

  //pseudo time
  std::vector<double> times(1, 0.);

  MethodOfLines_StateSpaceTimeHandler<FE, DOFHANDLER, BlockSparsityPattern,
      VECTOR, 2> DOFH(triangulation, state_fe);

  OP P(LPDE, DOFH);

  P.AddFunctional(&LPFD_1); // x-displacement in (45,0)
  P.AddFunctional(&LPFD_2); // y-displacement in (50,50)
  P.AddFunctional(&LPFD_3); // x-displacement in (0,50)
  P.AddFunctional(&LDFS); // yy-stress in (45,0)

  P.AddFunctional(&LBFUB); // y-displacement-integral on upper boundary

  P.SetBoundaryFunctionalColors(3);

  std::vector<bool> comp_mask(2);

  comp_mask[0] = true;
  comp_mask[1] = false;

  DOpEWrapper::ZeroFunction<2> zf_1(2);
  SimpleDirichletData<VECTOR, 2> DD1(zf_1);

  P.SetDirichletBoundaryColors(2, comp_mask, &DD1);

  comp_mask[0] = false;
  comp_mask[1] = true;

  SimpleDirichletData<VECTOR, 2> DD2(zf_1);

  P.SetDirichletBoundaryColors(0, comp_mask, &DD2);
  P.SetBoundaryEquationColors(3);

  SSolver solver(&P, "fullmem", pr, idc);
  //Only needed for pure PDE Problems
  DOpEOutputHandler<VECTOR> out(&solver, pr);
  DOpEExceptionHandler<VECTOR> ex(&out);
  P.RegisterOutputHandler(&out);
  P.RegisterExceptionHandler(&ex);
  solver.RegisterOutputHandler(&out);
  solver.RegisterExceptionHandler(&ex);

  solver.ReInit();
  out.ReInit();

  unsigned int niter = 2;
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
          DOFH.RefineSpace("global");
          solver.ReInit();
          out.ReInit();
        }
    }

  return 0;
}
