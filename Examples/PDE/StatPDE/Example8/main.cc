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
#include "gmreslinearsolver.h"
#include "integrator.h"
#include "parameterreader.h"
#include "mol_statespacetimehandler.h"
#include "simpledirichletdata.h"
#include "celldatacontainer.h"
#include "facedatacontainer.h"
#include "sparsitymaker.h"
#include "userdefineddofconstraints.h"
#include "preconditioner_wrapper.h"
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

// for grid local refinement
#include <numerics/error_estimator.h>
#include <grid/grid_refinement.h>

#include "localpde.h"
#include "functionals.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

// Define block issues
#define MATRIXBLOCK BlockSparseMatrix<double>
#define SPARSITYPATTERNBLOCK BlockSparsityPattern
#define VECTORBLOCK BlockVector<double>

// Define "normal" issues
#define MATRIX SparseMatrix<double>
#define SPARSITYPATTERN SparsityPattern
#define VECTOR Vector<double>

#define DOFHANDLER DoFHandler
#define FE FESystem
#define CDC CellDataContainer
#define FDC FaceDataContainer

#define PRECONDITIONERIDENTITYBLOCK DOpEWrapper::PreconditionIdentity_Wrapper<MATRIXBLOCK>
#define PRECONDITIONERIDENTITY DOpEWrapper::PreconditionIdentity_Wrapper<MATRIX>
#define PRECONDITIONERSSOR DOpEWrapper::PreconditionSSOR_Wrapper<MATRIX>

typedef PDEProblemContainer<
  LocalPDE<CDC,FDC, DOFHANDLER, VECTORBLOCK,
	   3>, DirichletDataInterface<VECTORBLOCK, 3>, SPARSITYPATTERNBLOCK,
  VECTORBLOCK, 3> OP1;

typedef PDEProblemContainer<
  LocalPDE<CDC,FDC, DOFHANDLER, VECTOR, 3>,
    DirichletDataInterface<VECTOR, 3>, SPARSITYPATTERN, VECTOR, 3> OP2;

typedef PDEProblemContainer<
  LocalPDE<CDC,FDC, DOFHANDLER, VECTOR, 3>,
  DirichletDataInterface<VECTOR, 3>, SPARSITYPATTERN, VECTOR, 3> OP3;

typedef IntegratorDataContainer<DOFHANDLER, Quadrature<3>,
    Quadrature<2>, VECTOR, 3> IDC;
typedef IntegratorDataContainer<DOFHANDLER, Quadrature<3>,
    Quadrature<2>, VECTORBLOCK, 3> IDCBLOCK;

typedef Integrator<IDCBLOCK, VECTORBLOCK, double, 3> INTEGRATOR1;
typedef Integrator<IDC, VECTOR, double, 3> INTEGRATOR2;
typedef Integrator<IDC, VECTOR, double, 3> INTEGRATOR3;

typedef GMRESLinearSolverWithMatrix<PRECONDITIONERIDENTITYBLOCK,
    SPARSITYPATTERNBLOCK, MATRIXBLOCK, VECTORBLOCK> LINEARSOLVER1;

typedef GMRESLinearSolverWithMatrix<PRECONDITIONERIDENTITY, SPARSITYPATTERN,
    MATRIX, VECTOR> LINEARSOLVER2;

typedef GMRESLinearSolverWithMatrix<PRECONDITIONERSSOR, SPARSITYPATTERN, MATRIX,
    VECTOR> LINEARSOLVER3;

typedef NewtonSolver<INTEGRATOR1, LINEARSOLVER1, VECTORBLOCK> NLS1;
typedef NewtonSolver<INTEGRATOR2, LINEARSOLVER2, VECTOR> NLS2;
typedef NewtonSolver<INTEGRATOR3, LINEARSOLVER3, VECTOR> NLS3;

typedef StatPDEProblem<NLS1, INTEGRATOR1, OP1, VECTORBLOCK, 3> SSolver1;
typedef StatPDEProblem<NLS2, INTEGRATOR2, OP2, VECTOR, 3> SSolver2;
typedef StatPDEProblem<NLS3, INTEGRATOR3, OP3, VECTOR, 3> SSolver3;

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
  SSolver1::declare_params(pr);
  SSolver2::declare_params(pr);
  SSolver3::declare_params(pr);
  DOpEOutputHandler<VECTOR>::declare_params(pr);

  pr.read_parameters(paramfile);

  Triangulation<3> triangulation;

  FE<3> state_fe(FE_Q<3>(1), 3);

  QGauss<3> quadrature_formula(3);
  QGauss<2> face_quadrature_formula(3);
  IDC idc(quadrature_formula, face_quadrature_formula);
  IDCBLOCK idcblock(quadrature_formula, face_quadrature_formula);
  LocalPDE<CDC,FDC,DOFHANDLER,VECTORBLOCK, 3> LPDE1;
  LocalPointFunctionalX<CDC,FDC,DOFHANDLER,VECTORBLOCK, 3> LPFX1;

  LocalPDE<CDC,FDC,DOFHANDLER,VECTOR, 3> LPDE2;
  LocalPointFunctionalX<CDC,FDC,DOFHANDLER,VECTOR, 3> LPFX2;

  // Pseudo time
  std::vector<double> times(1, 0.);

  // Spatial grid
  GridGenerator::hyper_cube(triangulation, 0, 1);
  triangulation.refine_global(3);

  MethodOfLines_StateSpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERNBLOCK,
      VECTORBLOCK, 3> DOFH1(triangulation, state_fe);

  MethodOfLines_StateSpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR, 3> DOFH2(
      triangulation, state_fe);

  OP1 P1(LPDE1, DOFH1);

  OP2 P2(LPDE2, DOFH2);

  OP3 P3(LPDE2, DOFH2);

  P1.AddFunctional(&LPFX1);
  P2.AddFunctional(&LPFX2);
  P3.AddFunctional(&LPFX2);

  std::vector<bool> comp_mask(3);

  comp_mask[0] = true;
  comp_mask[1] = true;
  comp_mask[2] = true;

  DOpEWrapper::ZeroFunction<3> zf(3);
  SimpleDirichletData<VECTORBLOCK, 3> DD1(zf);
  SimpleDirichletData<VECTOR, 3> DD2(zf);

  P1.SetDirichletBoundaryColors(0, comp_mask, &DD1);
  P2.SetDirichletBoundaryColors(0, comp_mask, &DD2);
  P3.SetDirichletBoundaryColors(0, comp_mask, &DD2);

    {
      SSolver1 solver1(&P1, "fullmem", pr, idcblock);

      DOpEOutputHandler<VECTORBLOCK> out(&solver1, pr);
      DOpEExceptionHandler<VECTORBLOCK> ex(&out);
      P1.RegisterOutputHandler(&out);
      P1.RegisterExceptionHandler(&ex);
      solver1.RegisterOutputHandler(&out);
      solver1.RegisterExceptionHandler(&ex);
      // Mesh-refinement cycles
      int niter = 3;

      Vector<double> solution;

      for (int i = 0; i < niter; i++)
        {
          try
            {
              solver1.ReInit();
              out.ReInit();
              stringstream outp;

              outp << "**************************************************\n";
              outp << "*             Starting Forward Solve - 1         *\n";
              outp << "*   Solving : " << P1.GetName() << "\t*\n";
              outp << "*   SDoFs   : ";
              solver1.StateSizeInfo(outp);
              outp << "**************************************************";
              out.Write(outp, 1, 1, 1);

              solver1.ComputeReducedFunctionals();
            }
          catch (DOpEException &e)
            {
              std::cout
                  << "Warning: During execution of `" + e.GetThrowingInstance()
                      + "` the following Problem occurred!" << std::endl;
              std::cout << e.GetErrorMessage() << std::endl;
            }
          if (i != niter - 1)
            {
              SolutionExtractor<SSolver1, VECTORBLOCK > a1(solver1);
              const StateVector<VECTORBLOCK > &gu1 = a1.GetU();
              solution = 0;
              solution = gu1.GetSpacialVector();
              Vector<float> estimated_error_per_cell(
                  triangulation.n_active_cells());

              std::vector<bool> component_mask(3, true);

              KellyErrorEstimator<3>::estimate(
                  static_cast<const DoFHandler<3>&>(DOFH1.GetStateDoFHandler()),
                  QGauss<2>(3), FunctionMap<3>::type(), solution,
                  estimated_error_per_cell, component_mask);

              DOFH1.RefineSpace(RefineFixedNumber(estimated_error_per_cell, 0.2,
                  0.0));
            }
        }
    }
    {
      SSolver2 solver2(&P2, "fullmem", pr, idc);
      SSolver3 solver3(&P3, "fullmem", pr, idc);

      DOpEOutputHandler<VECTOR> out(&solver2, pr);
      DOpEExceptionHandler<VECTOR> ex(&out);
      P2.RegisterOutputHandler(&out);
      P2.RegisterExceptionHandler(&ex);
      solver2.RegisterOutputHandler(&out);
      solver2.RegisterExceptionHandler(&ex);
      P3.RegisterOutputHandler(&out);
      P3.RegisterExceptionHandler(&ex);
      solver3.RegisterOutputHandler(&out);
      solver3.RegisterExceptionHandler(&ex);
      // Mesh-refinement cycles
      int niter = 3;

      Vector<double> solution;

      for (int i = 0; i < niter; i++)
        {
          try
            {
              solver2.ReInit();
              solver3.ReInit();
              out.ReInit();
              stringstream outp;
              outp << "**************************************************\n";
              outp << "*             Starting Forward Solve - 2         *\n";
              outp << "*   Solving : " << P2.GetName() << "\t*\n";
              outp << "*   SDoFs   : ";
              solver2.StateSizeInfo(outp);
              outp << "**************************************************";
              out.Write(outp, 1, 1, 1);

              solver2.ComputeReducedFunctionals();

              outp << "**************************************************\n";
              outp << "*             Starting Forward Solve - 3         *\n";
              outp << "*   Solving : " << P3.GetName() << "\t*\n";
              outp << "*   SDoFs   : ";
              solver3.StateSizeInfo(outp);
              outp << "**************************************************";
              out.Write(outp, 1, 1, 1);

              solver3.ComputeReducedFunctionals();
            }
          catch (DOpEException &e)
            {
              std::cout
                  << "Warning: During execution of `" + e.GetThrowingInstance()
                      + "` the following Problem occurred!" << std::endl;
              std::cout << e.GetErrorMessage() << std::endl;
            }
          if (i != niter - 1)
            {
              SolutionExtractor<SSolver2, VECTOR > a1(solver2);
              const StateVector<VECTOR > &gu1 = a1.GetU();
              solution = 0;
              solution = gu1.GetSpacialVector();
              Vector<float> estimated_error_per_cell(
                  triangulation.n_active_cells());

              std::vector<bool> component_mask(3, true);

              KellyErrorEstimator<3>::estimate(
                  static_cast<const DoFHandler<3>&>(DOFH2.GetStateDoFHandler()),
                  QGauss<2>(3), FunctionMap<3>::type(), solution,
                  estimated_error_per_cell, component_mask);
              DOFH2.RefineSpace(RefineFixedNumber(estimated_error_per_cell, 0.2,
                  0.0));
            }
        }

    }

  return 0;
}
