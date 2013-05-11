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
#include <iostream>

#include <grid/tria.h>
//#include <dofs/dof_handler.h>
#include <grid/grid_generator.h>
#include <fe/fe_q.h>
//#include <dofs/dof_tools.h>
#include <base/quadrature_lib.h>
//#include <base/function.h>
#include <lac/precondition_block.h>

#include "reducednewtonalgorithm.h"
#include "reducedtrustregionnewton.h"
#include "optproblemcontainer.h"
#include "functionalinterface.h"
//#include "pdeinterface.h"
#include "statreducedproblem.h"
#include "newtonsolver.h"
#include "cglinearsolver.h"
#include "directlinearsolver.h"
#include "integrator.h"
#include "parameterreader.h"
#include "mol_spacetimehandler.h"
#include "simpledirichletdata.h"
#include "noconstraints.h"
//#include "sparsitymaker.h"
//#include "userdefineddofconstraints.h"
#include "preconditioner_wrapper.h"
#include "integratordatacontainer.h"
#include "higher_order_dwrc_control.h"

#include "localpde.h"
#include "localfunctional.h"
#include "functionals.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

//Some abbreviations for better readability
const static int DIM = 2;
const static int CDIM = 2
//stands for the dimension of the controlvariable;

#define DOFHANDLER DoFHandler
#define FE FESystem

typedef QGauss<DIM> QUADRATURE;
typedef QGauss<DIM - 1> FACEQUADRATURE;

typedef BlockSparseMatrix<double> MATRIX;
typedef BlockSparsityPattern SPARSITYPATTERN;
typedef BlockVector<double> VECTOR;

#define CDC CellDataContainer
#define FDC FaceDataContainer

typedef LocalFunctional<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> COSTFUNCTIONAL;
typedef FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> FUNCTIONALINTERFACE;

//We use an optproblemcontainer instead of a pdeproblemcontainer, as we solve an optimazation
//problem. The optproblemcontainer holds all the information regarding the opt-problem.
typedef OptProblemContainer<FUNCTIONALINTERFACE, COSTFUNCTIONAL,
    LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM>,
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

//For different optimization algorightms
typedef ReducedNewtonAlgorithm<OP, VECTOR> RNA;
typedef ReducedTrustregion_NewtonAlgorithm<OP, VECTOR> RNA2;

//This class represents the reduced problem and steers the solution process.
typedef StatReducedProblem<NLS, NLS, INTEGRATOR, INTEGRATOR, OP, VECTOR, CDIM,
    DIM> SSolver;

//The spacetimehandler manages all the things related to the degrees of
//freedom in space and time (for the optimization as wella s the state variable!)
typedef MethodOfLines_SpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR,
    CDIM, DIM> STH;

typedef HigherOrderDWRContainerControl<STH, IDC, CDC<DOFHANDLER, VECTOR, DIM>,
    FDC<DOFHANDLER, VECTOR, DIM>, VECTOR> HO_DWRC;

int
main(int argc, char **argv)
{
  /**
   *
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
  unsigned int c_fe_order = 1;
  unsigned int s_fe_order = 2;
  unsigned int q_order = std::max(c_fe_order, s_fe_order) + 1;

  ParameterReader pr;
  SSolver::declare_params(pr);
  RNA::declare_params(pr);
  RNA2::declare_params(pr);
  DOpEOutputHandler<VECTOR>::declare_params(pr);

  pr.read_parameters(paramfile);

  std::string cases = "solve";

  Triangulation<DIM> triangulation(
      Triangulation<DIM>::MeshSmoothing::patch_level_1);
  GridGenerator::hyper_cube(triangulation, 0, 1);

  FESystem<CDIM> control_fe(FE_Q<CDIM>(c_fe_order), 1);
  FESystem<DIM> state_fe(FE_Q<DIM>(s_fe_order), 1);

  QUADRATURE quadrature_formula(q_order);
  FACEQUADRATURE face_quadrature_formula(q_order);
  IDC idc(quadrature_formula, face_quadrature_formula);
  double alpha = 1.e-3;

  LocalPDE<VECTOR, CDIM, DIM> LPDE(alpha);
  LocalFunctional<VECTOR, CDIM, DIM> LFunc(alpha);

  //AuxFunctionals
  LocalPointFunctional<VECTOR, CDIM, DIM> LPF;
  LocalMeanValueFunctional<VECTOR, CDIM, DIM> LMF;

//  std::vector<double> times(1,0.);
  Triangulation<1> times;
  GridGenerator::hyper_cube(times);
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


  SSolver solver(&P, "fullmem", pr, idc, 2);
  //Make sure we use the same outputhandler
  DOpEOutputHandler<VECTOR> out(&solver, pr);
  DOpEExceptionHandler<VECTOR> ex(&out);
  RNA Alg(&P, &solver, pr, &ex, &out);
  RNA2 Alg2(&P, &solver, pr, &ex, &out);

  /**********************For DWR***************************************/
  P.SetFunctionalForErrorEstimation(LFunc.GetName());
  FESystem<2> control_fe_high(FE_Q<2>(2 * c_fe_order), 1);
  FESystem<2> state_fe_high(FE_Q<2>(2 * s_fe_order), 1);
  QGauss<2> quadrature_formula_high(2 * q_order);
  QGauss<1> face_quadrature_formula_high(2 * q_order);
  IDC idc_high(quadrature_formula_high, face_quadrature_formula_high);
  STH DOFH_higher_order(triangulation, control_fe_high, state_fe_high,
      DOpEtypes::stationary);
  DOFH_higher_order.SetDoFHandlerOrdering(1, 0);
  HO_DWRC dwrc(DOFH_higher_order, idc_high, "fullmem", "fullmem", pr,
      DOpEtypes::mixed_control);
  solver.InitializeDWRC(dwrc);
  /********************************************************************/
  int niter = 2;
  Alg.ReInit();
  out.ReInit();
  ControlVector<VECTOR> q(&DOFH, "fullmem");

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

        Alg2.Solve(q);
        q = 0.;
        Alg.Solve(q);

        solver.ComputeRefinementIndicators(q, dwrc, LPDE);
        double value = solver.GetFunctionalValue(LFunc.GetName());
        double est_error = dwrc.GetError();
        double error = ex_value - value;
        stringstream outp;
        outp << "Exact Value: " << ex_value << "\t Computed: " << value
            << std::endl;
        outp << "Primal Err: " << dwrc.GetPrimalError() << "\t Dual Err: "
            << dwrc.GetDualError();
        outp << "\t Control Err: " << dwrc.GetControlError() << std::endl;
        outp << "Est Total Error: " << est_error << " \tError: " << error;
        outp << "  Ieff (eh/e)= " << est_error / error << std::endl;
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
      //triangulation.refine_global (1);
      DOFH.RefineSpace();
      Alg.ReInit();
      out.ReInit();
    }
  }

  return 0;
}
