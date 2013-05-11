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
#include <dofs/dof_handler.h>
#include <grid/grid_generator.h>
#include <fe/fe_q.h>
#include <fe/fe_nothing.h>
#include <base/quadrature_lib.h>

#include "reducednewtonalgorithm.h"
#include "optproblemcontainer.h"
#include "functionalinterface.h"
#include "statreducedproblem.h"
#include "directlinearsolver.h"
#include "voidlinearsolver.h" // for mixed dim opt. control
#include "integrator.h"
#include "newtonsolver.h"
#include "integratormixeddims.h" // for mixed dim opt. control
#include "newtonsolvermixeddims.h" // for mixed dim opt. control
#include "parameterreader.h"
#include "mol_spacetimehandler.h"
#include "simpledirichletdata.h"
#include "noconstraints.h"
#include "integratordatacontainer.h"

#include "localpde.h"
#include "localfunctional.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

#define DOFHANDLER DoFHandler
#define FE FESystem


const static int DIM = 2;
const static int CDIM = 0;

typedef QGauss<DIM> QUADRATURE;
typedef QGauss<DIM - 1> FACEQUADRATURE;

typedef BlockSparseMatrix<double> MATRIX;
typedef BlockSparsityPattern SPARSITYPATTERN;
typedef BlockVector<double> VECTOR;

#define CDC CellDataContainer
#define FDC FaceDataContainer

typedef LocalFunctional<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> COSTFUNCTIONAL;
typedef FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> FUNCTIONALINTERFACE;

typedef OptProblemContainer<FUNCTIONALINTERFACE, COSTFUNCTIONAL,
    LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM>,
    SimpleDirichletData<VECTOR,  DIM>,
    NoConstraints<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM>, SPARSITYPATTERN,
    VECTOR, CDIM, DIM> OP;

typedef IntegratorDataContainer<DOFHANDLER, QUADRATURE, FACEQUADRATURE, VECTOR,
    DIM> IDC;
typedef Integrator<IDC, VECTOR, double, DIM> INTEGRATOR;
//special newtonsolver for the mixed dims
typedef IntegratorMixedDimensions<IDC, VECTOR, double, CDIM, DIM> INTEGRATORM;

typedef DirectLinearSolverWithMatrix<SPARSITYPATTERN, MATRIX, VECTOR> LINEARSOLVER;

//dummy solver for the 0d control
typedef VoidLinearSolver<VECTOR> VOIDLS;

//special newtonsolver for the mixed dims
typedef NewtonSolverMixedDimensions<INTEGRATORM, VOIDLS, VECTOR> NLSM;
typedef NewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR> NLS;
typedef ReducedNewtonAlgorithm<OP, VECTOR> RNA;

typedef StatReducedProblem<NLSM, NLS, INTEGRATORM, INTEGRATOR, OP, VECTOR, CDIM,
    DIM> SSolver;
typedef MethodOfLines_SpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR,
    CDIM, DIM> STH;

int
main(int argc, char **argv)
{
  /**
   * We solve an optimization problem of mixed Type, i.e. the
   * costfunctional has a point-part as well as a distributed
   * part. To handle this, we use IntegratorMixedDimensions.
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
  RNA::declare_params(pr);
  pr.read_parameters(paramfile);

  Triangulation<DIM> triangulation;
  GridGenerator::hyper_cube(triangulation, 0, 1);
  triangulation.refine_global(5);

  //We need no finite element for the control variable, so
  //we take the FE_Nothing element, which has 0 degrees of freedom.
  //The number of components of the finite element has to mathc
  //the number of parameters!
  FE<DIM> control_fe(FE_Nothing<DIM>(1), 3); //3 Parameter
  FE<DIM> state_fe(FE_Q<DIM>(1), 2);

  QUADRATURE quadrature_formula(2);
  FACEQUADRATURE face_quadrature_formula(2);
  IDC idc(quadrature_formula, face_quadrature_formula);

  LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> LPDE;
  COSTFUNCTIONAL LFunc;

  STH DOFH(triangulation, control_fe, state_fe, DOpEtypes::stationary);

  NoConstraints<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> Constraints;

  OP P(LFunc, LPDE, Constraints, DOFH);

  DOpEWrapper::ZeroFunction<DIM> zf(2);
  SimpleDirichletData<VECTOR, DIM> DD(zf);
  std::vector<bool> comp_mask(2, true);

  P.SetDirichletBoundaryColors(0, comp_mask, &DD);

  SSolver solver(&P, "fullmem", pr, idc);
  RNA Alg(&P, &solver, pr);

  try
  {
    Alg.ReInit();

    ControlVector<VECTOR> q(&DOFH, "fullmem");
    {
      //PreInitialization of q
      q = 2.;
    }

    Alg.Solve(q);
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
