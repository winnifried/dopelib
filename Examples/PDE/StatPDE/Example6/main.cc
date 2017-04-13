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
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/numerics/error_estimator.h>

#include <container/pdeproblemcontainer.h>
#include <interfaces/functionalinterface.h>
#include <interfaces/pdeinterface.h>
#include <reducedproblems/statpdeproblem.h>
#include <templates/newtonsolver.h>
#include <templates/gmreslinearsolver.h>
#include <templates/integrator.h>
#include <include/parameterreader.h>
#include <basic/mol_statespacetimehandler.h>
#include <problemdata/simpledirichletdata.h>
#include <wrapper/preconditioner_wrapper.h>
#include <container/integratordatacontainer.h>

#include "localpde.h"
#include "functionals.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

const static int DIM = 3;

#define DOFHANDLER DoFHandler
#define FE FESystem
#define CDC ElementDataContainer
#define FDC FaceDataContainer

typedef QGauss<DIM> QUADRATURE;
typedef QGauss<DIM - 1> FACEQUADRATURE;

// Define block issues
typedef BlockSparseMatrix<double> MATRIXBLOCK;
typedef BlockSparsityPattern SPARSITYPATTERNBLOCK;
typedef BlockVector<double> VECTORBLOCK;

// Define "normal" issues
typedef SparseMatrix<double> MATRIX;
typedef SparsityPattern SPARSITYPATTERN;
typedef Vector<double> VECTOR;

//Define different preconditioners
typedef DOpEWrapper::PreconditionIdentity_Wrapper<MATRIXBLOCK> PRECONDITIONERIDENTITYBLOCK;
typedef DOpEWrapper::PreconditionIdentity_Wrapper<MATRIX> PRECONDITIONERIDENTITY;
typedef DOpEWrapper::PreconditionSSOR_Wrapper<MATRIX> PRECONDITIONERSSOR;

//Define problemcontainer for block and non block
typedef PDEProblemContainer<LocalPDE<CDC, FDC, DOFHANDLER, VECTORBLOCK, DIM>,
        SimpleDirichletData<VECTORBLOCK, DIM>, SPARSITYPATTERNBLOCK, VECTORBLOCK,
        DIM> OPBLOCK;
typedef PDEProblemContainer<LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
        SimpleDirichletData<VECTOR, DIM>, SPARSITYPATTERN, VECTOR, DIM> OP;

//Define integratordatacontainer for block and non block vectors
typedef IntegratorDataContainer<DOFHANDLER, QUADRATURE, FACEQUADRATURE, VECTOR,
        DIM> IDC;
typedef IntegratorDataContainer<DOFHANDLER, QUADRATURE, FACEQUADRATURE,
        VECTORBLOCK, DIM> IDCBLOCK;

//Define block and nonblock integrators
typedef Integrator<IDCBLOCK, VECTORBLOCK, double, DIM> BLOCKINTEGRATOR;
typedef Integrator<IDC, VECTOR, double, DIM> INTEGRATOR;

//We set up three different linear solvers: Block and nonblock GMRES without
//a preconditioner and an SSor preconditioned non-block GMRES
typedef GMRESLinearSolverWithMatrix<PRECONDITIONERIDENTITYBLOCK,
        SPARSITYPATTERNBLOCK, MATRIXBLOCK, VECTORBLOCK> GMRESIDENTITYBLOCK;
typedef GMRESLinearSolverWithMatrix<PRECONDITIONERIDENTITY, SPARSITYPATTERN,
        MATRIX, VECTOR> GMRESIDENTITY;
typedef GMRESLinearSolverWithMatrix<PRECONDITIONERSSOR, SPARSITYPATTERN, MATRIX,
        VECTOR> GMRESSSOR;

//Define three newtonsolver fitting the three linear solvers
typedef NewtonSolver<BLOCKINTEGRATOR, GMRESIDENTITYBLOCK, VECTORBLOCK> NLS1;
typedef NewtonSolver<INTEGRATOR, GMRESIDENTITY, VECTOR> NLS2;
typedef NewtonSolver<INTEGRATOR, GMRESSSOR, VECTOR> NLS3;

//Define the three ssolver fitting the three linear solvers.
typedef StatPDEProblem<NLS1, BLOCKINTEGRATOR, OPBLOCK, VECTORBLOCK, DIM> RP1;
typedef StatPDEProblem<NLS2, INTEGRATOR, OP, VECTOR, DIM> RP2;
typedef StatPDEProblem<NLS3, INTEGRATOR, OP, VECTOR, DIM> RP3;

//Define the spacetimehandler for block and non block vectors
typedef MethodOfLines_StateSpaceTimeHandler<FE, DOFHANDLER,
        SPARSITYPATTERNBLOCK, VECTORBLOCK, DIM> STHBLOCK;
typedef MethodOfLines_StateSpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN,
        VECTOR, DIM> STH;

int
main(int argc, char **argv)
{
  /**
   *  Solving the standard Laplace equation in 3d with a locally refined grid
   *  and three different iterative linear solvers.
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

  //Declare parameters
  ParameterReader pr;
  RP1::declare_params(pr);
  RP2::declare_params(pr);
  RP3::declare_params(pr);
  DOpEOutputHandler<VECTOR>::declare_params(pr);
  pr.read_parameters(paramfile);

  // Mesh-refinement cycles
  const int niter = 3;

  //Create triangulation
  Triangulation<DIM> triangulation;
  GridGenerator::hyper_cube(triangulation, 0, 1);
  triangulation.refine_global(3);

  //Define the finite element as well as the quadrature rules
  FE<DIM> state_fe(FE_Q<DIM>(1), 3);
  QUADRATURE quadrature_formula(3);
  FACEQUADRATURE face_quadrature_formula(3);
  IDC idc(quadrature_formula, face_quadrature_formula);
  IDCBLOCK idcblock(quadrature_formula, face_quadrature_formula);

  //Set up the pde and the pointfunctional for block-vectors and non block vectors
  LocalPDE<CDC, FDC, DOFHANDLER, VECTORBLOCK, DIM> LPDE1;
  LocalPointFunctionalX<CDC, FDC, DOFHANDLER, VECTORBLOCK, DIM> LPFX1;

  LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM> LPDE2;
  LocalPointFunctionalX<CDC, FDC, DOFHANDLER, VECTOR, DIM> LPFX2;

  //Define the different STH and OP objects
  STHBLOCK DOFH1(triangulation, state_fe);
  STH DOFH2(triangulation, state_fe);

  OPBLOCK Pblock(LPDE1, DOFH1);
  OP P(LPDE2, DOFH2);

  //Add the functionals to the problemcontainer
  Pblock.AddFunctional(&LPFX1);
  P.AddFunctional(&LPFX2);

  //Set the dirichlet values
  DOpEWrapper::ZeroFunction<DIM> zf(3);
  SimpleDirichletData<VECTORBLOCK, DIM> DD1(zf);
  SimpleDirichletData<VECTOR, DIM> DD2(zf);
  std::vector<bool> comp_mask(3, true);
  Pblock.SetDirichletBoundaryColors(0, comp_mask, &DD1);
  P.SetDirichletBoundaryColors(0, comp_mask, &DD2);

  //We solve first with the nonpreconditioned GMRES with blockstructure.
  {
    RP1 solver1(&Pblock, DOpEtypes::VectorStorageType::fullmem, pr, idcblock);

    DOpEOutputHandler<VECTORBLOCK> out(&solver1, pr);
    DOpEExceptionHandler<VECTORBLOCK> ex(&out);
    Pblock.RegisterOutputHandler(&out);
    Pblock.RegisterExceptionHandler(&ex);
    solver1.RegisterOutputHandler(&out);
    solver1.RegisterExceptionHandler(&ex);


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
            outp << "*   Solving : " << Pblock.GetName() << "\t*\n";
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
            //We extract the solution out of the statproblem..
            SolutionExtractor<RP1, VECTORBLOCK> a1(solver1);
            const StateVector<VECTORBLOCK> &gu1 = a1.GetU();
            solution = gu1.GetSpacialVector();
            Vector<float> estimated_error_per_element(triangulation.n_active_cells());

            std::vector<bool> component_mask(3, true);

            //..and estimate the error with the help of the KellyErrorEstimator
            KellyErrorEstimator<DIM>::estimate(
              static_cast<const DoFHandler<DIM>&>(DOFH1.GetStateDoFHandler()),
              QGauss<2>(3), FunctionMap<DIM>::type(), solution,
              estimated_error_per_element, component_mask);

            //We choose a refinement strategy (here fixednumber) and
            //refine the spatial esh accordingly.
            DOFH1.RefineSpace(
              RefineFixedNumber(estimated_error_per_element, 0.2, 0.0));
          }
      }
  }
  //Here we solve with nonpreconditioned GMRES without blockstructure as
  //well as with the SSOR preconditioned GMRES.
  {
    RP2 solver2(&P, DOpEtypes::VectorStorageType::fullmem, pr, idc);
    RP3 solver3(&P, DOpEtypes::VectorStorageType::fullmem, pr, idc);

    DOpEOutputHandler<VECTOR> out(&solver2, pr);
    DOpEExceptionHandler<VECTOR> ex(&out);
    P.RegisterOutputHandler(&out);
    P.RegisterExceptionHandler(&ex);
    solver2.RegisterOutputHandler(&out);
    solver2.RegisterExceptionHandler(&ex);
    P.RegisterOutputHandler(&out);
    P.RegisterExceptionHandler(&ex);
    solver3.RegisterOutputHandler(&out);
    solver3.RegisterExceptionHandler(&ex);

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
            outp << "*   Solving : " << P.GetName() << "\t*\n";
            outp << "*   SDoFs   : ";
            solver2.StateSizeInfo(outp);
            outp << "**************************************************";
            out.Write(outp, 1, 1, 1);

            solver2.ComputeReducedFunctionals();

            outp << "**************************************************\n";
            outp << "*             Starting Forward Solve - 3         *\n";
            outp << "*   Solving : " << P.GetName() << "\t*\n";
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
            SolutionExtractor<RP2, VECTOR> a1(solver2);
            const StateVector<VECTOR> &gu1 = a1.GetU();
            solution = gu1.GetSpacialVector();
            Vector<float> estimated_error_per_element(triangulation.n_active_cells());

            std::vector<bool> component_mask(3, true);

            KellyErrorEstimator<DIM>::estimate(
              static_cast<const DoFHandler<DIM>&>(DOFH2.GetStateDoFHandler()),
              QGauss<2>(3), FunctionMap<DIM>::type(), solution,
              estimated_error_per_element, component_mask);
            DOFH2.RefineSpace(
              RefineFixedNumber(estimated_error_per_element, 0.2, 0.0));
          }
      }

  }

  return 0;
}
#undef FDC
#undef CDC
#undef FE
#undef DOFHANDLER
