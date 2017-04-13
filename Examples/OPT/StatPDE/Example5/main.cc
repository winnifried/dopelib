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

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/numerics/error_estimator.h>

#include <opt_algorithms/reducednewtonalgorithm.h>
#include <container/optproblemcontainer.h>
#include <interfaces/functionalinterface.h>
#include <reducedproblems/statreducedproblem.h>
#include <templates/newtonsolver.h>
#include <templates/cglinearsolver.h>
#include <templates/directlinearsolver.h>
#include <templates/integrator_multimesh.h> //for multimesh purposes
#include <include/parameterreader.h>
#include <basic/mol_multimesh_spacetimehandler.h> //for multimesh purposes
#include <problemdata/simpledirichletdata.h>
#include <problemdata/noconstraints.h>
#include <wrapper/preconditioner_wrapper.h>
#include <container/integratordatacontainer.h>

#include "localpde.h"
#include "localfunctional.h"
#include "functionals.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

const static int DIM = 2;
const static int CDIM = 2;

#define DOFHANDLER DoFHandler
#define FE FESystem

typedef QGauss<DIM> QUADRATURE;
typedef QGauss<DIM - 1> FACEQUADRATURE;

typedef BlockSparseMatrix<double> MATRIX;
typedef BlockSparsityPattern SPARSITYPATTERN;
typedef BlockVector<double> VECTOR;

//We use the multimesh variants of Element- and FaceDataContainer
#define CDC Multimesh_ElementDataContainer
#define FDC Multimesh_FaceDataContainer

typedef LocalFunctional<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> COSTFUNCTIONAL;
typedef FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> FUNCTIONALINTERFACE;

typedef OptProblemContainer<FUNCTIONALINTERFACE, COSTFUNCTIONAL,
        LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
        SimpleDirichletData<VECTOR, DIM>,
        NoConstraints<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM>, SPARSITYPATTERN,
        VECTOR, CDIM, DIM> OP;

typedef IntegratorDataContainer<DOFHANDLER, QUADRATURE, FACEQUADRATURE, VECTOR,
        DIM> IDC;

//We use the multimesh-integrator
typedef IntegratorMultiMesh<IDC, VECTOR, double, DIM> INTEGRATOR;

//Uncomment to use a CG-Method with Identity Preconditioner
typedef CGLinearSolverWithMatrix<
DOpEWrapper::PreconditionIdentity_Wrapper<MATRIX>, SPARSITYPATTERN, MATRIX,
            VECTOR> LINEARSOLVER;
//Uncomment to use UMFPACK
//typedef DirectLinearSolverWithMatrix<SPARSITYPATTERN,MATRIX,VECTOR> LINEARSOLVER;

typedef NewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR> NLS;
typedef ReducedNewtonAlgorithm<OP, VECTOR> RNA;
typedef StatReducedProblem<NLS, NLS, INTEGRATOR, INTEGRATOR, OP, VECTOR, CDIM,
        DIM> RP;
typedef MethodOfLines_MultiMesh_SpaceTimeHandler<FE, DOFHANDLER,
        SPARSITYPATTERN, VECTOR, DIM> STH;

int
main(int argc, char **argv)
{
  /**
   * We solve a PDE constrained optimization problem with a distributed
   * control on the right hand side. In this example, we use different
   * meshes for the control and the state variable.
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
  RNA::declare_params(pr);

  pr.read_parameters(paramfile);

  const int niter = 3;

  Triangulation<DIM> triangulation;
  GridGenerator::hyper_cube(triangulation, 0, 1);
  triangulation.refine_global(3);

  FE<DIM> control_fe(FE_Q<DIM>(1), 1);
  FE<DIM> state_fe(FE_Q<DIM>(1), 1);

  QUADRATURE quadrature_formula(2);
  FACEQUADRATURE face_quadrature_formula(2);
  IDC idc(quadrature_formula, face_quadrature_formula);

  LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM> LPDE;
  COSTFUNCTIONAL LFunc;
  //AuxFunctionals
  LocalPointFunctional<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> LPF;
  LocalMeanValueFunctional<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> LMF;
  QErrorFunctional<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> QEF;
  UErrorFunctional<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> UEF;

  STH DOFH(triangulation, control_fe, state_fe, DOpEtypes::stationary);

  NoConstraints<Multimesh_ElementDataContainer, Multimesh_FaceDataContainer,
                DOFHANDLER, VECTOR, CDIM, DIM> Constraints;

  OP P(LFunc, LPDE, Constraints, DOFH);

  P.AddFunctional(&LPF);
  P.AddFunctional(&LMF);
  P.AddFunctional(&QEF);
  P.AddFunctional(&UEF);

  DOpEWrapper::ZeroFunction<DIM> zf(1);
  SimpleDirichletData<VECTOR, DIM> DD(zf);
  std::vector<bool> comp_mask(1, true);
  P.SetDirichletBoundaryColors(0, comp_mask, &DD);

  RP solver(&P, DOpEtypes::VectorStorageType::fullmem, pr, idc);

  RNA Alg(&P, &solver, pr);

  Alg.ReInit();
  ControlVector<VECTOR> q(&DOFH, DOpEtypes::VectorStorageType::fullmem);
  DOFH.RefineStateSpace();
  Alg.ReInit();

  //First we test the multimesh setting with global refinement
  for (int i = 0; i < niter; i++)
    {
      try
        {
          q = 0.;
          Alg.Solve(q);
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
          //DOFH.RefineSpace();
          DOFH.RefineControlSpace();
          Alg.ReInit();
        }
    }
  DOFH.RefineStateSpace();
  Alg.ReInit();

  //Secondly, we use local refinement
  for (int i = 0; i < niter; i++)
    {
      try
        {
          q = 0.;
          Alg.Solve(q);
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
          if (i % 2 == 0)
            {
              SolutionExtractor<RP, VECTOR> a(solver);
              const StateVector<VECTOR> &gu = a.GetU();
              Vector<double> solution;
              solution = 0;
              solution = gu.GetSpacialVector();
              Vector<float> estimated_error_per_element(triangulation.n_active_cells());

              std::vector<bool> component_mask(1, true);

              KellyErrorEstimator<DIM>::estimate(
                static_cast<const DoFHandler<DIM>&>(DOFH.GetStateDoFHandler()),
                QGauss<1>(2), FunctionMap<DIM>::type(), solution,
                estimated_error_per_element, component_mask);
              DOFH.RefineStateSpace(
                RefineFixedNumber(estimated_error_per_element, 0.1, 0.0));
              Alg.ReInit();
            }
          if (i % 2 == 1)
            {
              Vector<double> solution;
              solution = 0;
              solution = q.GetSpacialVector();
              Vector<float> estimated_error_per_element(triangulation.n_active_cells());

              std::vector<bool> component_mask(1, true);

              KellyErrorEstimator<DIM>::estimate(
                static_cast<const DoFHandler<DIM>&>(DOFH.GetControlDoFHandler()),
                QGauss<1>(2), FunctionMap<DIM>::type(), solution,
                estimated_error_per_element, component_mask);
              DOFH.RefineControlSpace(
                RefineFixedNumber(estimated_error_per_element, 0.1, 0.0));
              Alg.ReInit();
            }
        }
    }
  return 0;
}

#undef FDC
#undef CDC
#undef FE
#undef DOFHANDLER
