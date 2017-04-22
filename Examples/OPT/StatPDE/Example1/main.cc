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

#include <opt_algorithms/reducednewtonalgorithm.h>
#include <opt_algorithms/reducedtrustregionnewton.h>
#include <container/optproblemcontainer.h>
#include <interfaces/functionalinterface.h>
#include <reducedproblems/statreducedproblem.h>
#include <templates/newtonsolver.h>
#include <templates/cglinearsolver.h>
#include <templates/directlinearsolver.h>
#include <templates/integrator.h>
#include <include/parameterreader.h>
#include <basic/mol_spacetimehandler.h>
#include <problemdata/simpledirichletdata.h>
#include <problemdata/noconstraints.h>
#include <wrapper/preconditioner_wrapper.h>
#include <container/integratordatacontainer.h>
#include <container/higher_order_dwrc_control.h>

#include "localpde.h"
#include "localfunctional.h"
#include "functionals.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

//Some abbreviations for better readability
const static int DIM = 2;
const static int CDIM = 2;
//stands for the dimension of the controlvariable;

#define DOFHANDLER DoFHandler
#define FE FESystem

typedef QGauss<DIM> QUADRATURE;
typedef QGauss<DIM - 1> FACEQUADRATURE;

//Fix for a bug in deal.ii 8.5.0
#if DEAL_II_VERSION_GTE(8,5,0)
#if DEAL_II_VERSION_GTE(9,0,0)//post deal 8.5.0
typedef BlockSparseMatrix<double> MATRIX;
typedef BlockSparsityPattern SPARSITYPATTERN;
typedef BlockVector<double> VECTOR;
#else //dealii 8.5.0
typedef SparseMatrix<double> MATRIX;
typedef SparsityPattern SPARSITYPATTERN;
typedef Vector<double> VECTOR;
#endif
#else //pre deal 8.5.0
typedef BlockSparseMatrix<double> MATRIX;
typedef BlockSparsityPattern SPARSITYPATTERN;
typedef BlockVector<double> VECTOR;
#endif

#define CDC ElementDataContainer
#define FDC FaceDataContainer

typedef LocalFunctional<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> COSTFUNCTIONAL;
typedef FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> FUNCTIONALINTERFACE;

//We use an optproblemcontainer instead of a pdeproblemcontainer, as we solve an optimization
//problem. The optproblemcontainer holds all the information regarding the opt-problem.
typedef OptProblemContainer<FUNCTIONALINTERFACE, COSTFUNCTIONAL,
        LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
        SimpleDirichletData<VECTOR, DIM>,
        NoConstraints<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM>, SPARSITYPATTERN,
        VECTOR, CDIM, DIM> OP;

typedef IntegratorDataContainer<DOFHANDLER, QUADRATURE, FACEQUADRATURE, VECTOR,
        DIM> IDC;

typedef Integrator<IDC, VECTOR, double, DIM> INTEGRATOR;

//Uncomment to use a CG-Method with Identity Preconditioner
typedef CGLinearSolverWithMatrix<
DOpEWrapper::PreconditionIdentity_Wrapper<MATRIX>, SPARSITYPATTERN, MATRIX,
            VECTOR> LINEARSOLVER;
//Uncomment to use UMFPACK
//typedef DirectLinearSolverWithMatrix<SPARSITYPATTERN,MATRIX,VECTOR> LINEARSOLVER;

typedef NewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR> NLS;

//For different optimization algorithms
typedef ReducedNewtonAlgorithm<OP, VECTOR> RNA;
typedef ReducedTrustregion_NewtonAlgorithm<OP, VECTOR> RNA2;

//This class represents the reduced problem and steers the solution process.
typedef StatReducedProblem<NLS, NLS, INTEGRATOR, INTEGRATOR, OP, VECTOR, CDIM,
        DIM> RP;

//The spacetimehandler manages all the things related to the degrees of
//freedom in space and time (for the optimization as well as the state variable!)
typedef MethodOfLines_SpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR,
        CDIM, DIM> STH;

typedef HigherOrderDWRContainerControl<STH, IDC, CDC<DOFHANDLER, VECTOR, DIM>,
        FDC<DOFHANDLER, VECTOR, DIM>, VECTOR> HO_DWRC;

int
main(int argc, char **argv)
{
  /**
   * This class solves a distributed minimization problem with
   * the laplacian as PDE constraint and the control on the right
   * hand side of the PDE. The target functional is
   * $\frac{1}{2} \|u-u^d\|^2 + \frac{\alpha}{2}\|q\|^2$.
   *
   * Additionally we estimate the error in the cost functional.
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
  const unsigned int c_fe_order = 1;
  const unsigned int s_fe_order = 2;
  const unsigned int q_order = std::max(c_fe_order, s_fe_order) + 1;
  const int niter = 2;

  ParameterReader pr;
  RP::declare_params(pr);
  RNA::declare_params(pr);
  RNA2::declare_params(pr);
  DOpEOutputHandler<VECTOR>::declare_params(pr);

  pr.read_parameters(paramfile);

  std::string cases = "solve";

  Triangulation<DIM> triangulation(
    Triangulation<DIM>::MeshSmoothing::patch_level_1);
  GridGenerator::hyper_cube(triangulation, 0, 1);

  // New in comparison to pure PDE problems
  FE<CDIM> control_fe(FE_Q<CDIM>(c_fe_order), 1);

  // This is the same as in pure PDE problems
  FE<DIM> state_fe(FE_Q<DIM>(s_fe_order), 1);

  QUADRATURE quadrature_formula(q_order);
  FACEQUADRATURE face_quadrature_formula(q_order);
  IDC idc(quadrature_formula, face_quadrature_formula);
  const double alpha = 1.e-3;

  LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM> LPDE(alpha);

  // New in comparison to pure PDE problems
  COSTFUNCTIONAL LFunc(alpha);

  //AuxFunctionals
  LocalPointFunctional<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> LPF;
  LocalMeanValueFunctional<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> LMF;

  triangulation.refine_global(5);

  STH DOFH(triangulation, control_fe, state_fe, DOpEtypes::stationary);

  NoConstraints<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> Constraints;

  OP P(LFunc, LPDE, Constraints, DOFH);

  P.AddFunctional(&LPF);
  P.AddFunctional(&LMF);

  std::vector<bool> comp_mask(1);
  comp_mask[0] = true;
  DOpEWrapper::ZeroFunction<2> zf(1);

  SimpleDirichletData<VECTOR, DIM> DD(zf);
  P.SetDirichletBoundaryColors(0, comp_mask, &DD);

  RP solver(&P, DOpEtypes::VectorStorageType::fullmem, pr, idc, 2);
  //Make sure we use the same outputhandler
  DOpEOutputHandler<VECTOR> out(&solver, pr);
  DOpEExceptionHandler<VECTOR> ex(&out);
  RNA Alg(&P, &solver, pr, &ex, &out);
  RNA2 Alg2(&P, &solver, pr, &ex, &out);

  /**********************For DWR***************************************/
  P.SetFunctionalForErrorEstimation(LFunc.GetName());
  FESystem<2> control_fe_high(FE_Q<2>(2 * c_fe_order), 1);
  FESystem<2> state_fe_high(FE_Q<2>(2 * s_fe_order), 1);
  QUADRATURE quadrature_formula_high(2 * q_order);
  FACEQUADRATURE face_quadrature_formula_high(2 * q_order);
  IDC idc_high(quadrature_formula_high, face_quadrature_formula_high);
  STH DOFH_higher_order(triangulation, control_fe_high, state_fe_high,
                        DOpEtypes::stationary);
  DOFH_higher_order.SetDoFHandlerOrdering(1, 0);
  HO_DWRC dwrc(DOFH_higher_order, idc_high, DOpEtypes::VectorStorageType::fullmem, DOpEtypes::VectorStorageType::fullmem, pr,
               DOpEtypes::mixed_control);
  P.InitializeDWRC(dwrc);
  /********************************************************************/

  Alg.ReInit();
  out.ReInit();
  ControlVector<VECTOR> q(&DOFH, DOpEtypes::VectorStorageType::fullmem);

  double ex_value = 1. / 8. * (25 * M_PI * M_PI * M_PI * M_PI + 1. / alpha);

  for (int i = 0; i < niter; i++)
    {
      try
        {
          if (cases == "check")
            {
              ControlVector<VECTOR> dq(q);
              Alg.CheckGrads(1., q, dq, 2);
              Alg.CheckHessian(1., q, dq, 2);
            }
          else
            {
              //We test ReducedNewtonAlgorithm as well as ReducedTrustedNewtonAlgorithm
              Alg2.Solve(q);
              q = 0.;
              Alg.Solve(q);

              solver.ComputeRefinementIndicators(q, dwrc, LPDE);
              const double value = solver.GetFunctionalValue(LFunc.GetName());
              const double est_error = dwrc.GetError();
              const double error = ex_value - value;
              stringstream outp;
              outp << "Exact Value: " << ex_value << "\t Computed: " << value
                   << std::endl;
              //Case distinction to make shure that the output is independent of
              //compiler induced floating point cancellation in almost zero values
              if (fabs(dwrc.GetPrimalError()) < 0.01 * fabs(dwrc.GetControlError()))
                {
                  outp << "Primal Err: " << std::setprecision(3) << " < " << 0.01 * fabs(dwrc.GetControlError());
                }
              else
                {
                  outp << "Primal Err: " << std::setprecision(3) << dwrc.GetPrimalError();
                }
              if (fabs(dwrc.GetDualError()) < 0.01 * fabs(dwrc.GetControlError()))
                {
                  outp << "\t Dual Err: " <<  std::setprecision(3) << " < " << 0.01 * fabs(dwrc.GetControlError());
                }
              else
                {
                  outp << "\t Dual Err: " <<  std::setprecision(3) <<dwrc.GetDualError();
                }
              outp << "\t Control Err: " <<  std::setprecision(3) <<dwrc.GetControlError() << std::endl;
              outp << "Est Total Error: " <<  std::setprecision(2) << est_error << " \tError: " << std::setprecision(2) << error;
              outp << "  Ieff (eh/e)= " <<  std::setprecision(2) << est_error / error << std::endl;
              out.Write(outp, 1, 1, 1);
            }
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
          //global mesh refinement
          DOFH.RefineSpace();
          Alg.ReInit();
          out.ReInit();
        }
    }

  return 0;
}

#undef FDC
#undef CDC
#undef FE
#undef DOFHANDLER
