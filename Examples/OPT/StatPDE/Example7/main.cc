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
#include <grid/grid_generator.h>
#include <fe/fe_q.h>
#include <dofs/dof_tools.h>
#include <base/quadrature_lib.h>
#include <lac/precondition_block.h>

#include "reduced_ipopt_algorithm.h"
#include "reduced_snopt_algorithm.h"
#include "optproblemcontainer.h"
#include "functionalinterface.h"
#include "statreducedproblem.h" 
#include "newtonsolver.h"
#include "cglinearsolver.h"
#include "integrator.h"
#include "parameterreader.h"
#include "mol_spacetimehandler.h"
#include "simpledirichletdata.h"
#include "noconstraints.h"
#include "preconditioner_wrapper.h"
#include "integratordatacontainer.h"

#include "localpde.h"
#include "localfunctional.h"
#include "functionals.h"
#include "localconstraints.h"
#include "localconstraintaccessor.h"

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

#define CDC CellDataContainer
#define FDC FaceDataContainer

typedef FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> FUNCTIONALINTERFACE;
typedef LocalFunctional<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> COSTFUNCTIONAL;
typedef LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM> PDE;
typedef SimpleDirichletData<VECTOR, DIM> DD;
typedef ConstraintInterface<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> CONS;

typedef OptProblemContainer<FUNCTIONALINTERFACE, COSTFUNCTIONAL, PDE, DD, CONS,
    SPARSITYPATTERN, VECTOR, CDIM, DIM> OP;

typedef IntegratorDataContainer<DOFHANDLER, QUADRATURE, FACEQUADRATURE, VECTOR,
    DIM> IDC;
typedef Integrator<IDC, VECTOR, double, DIM> INTEGRATOR;

typedef CGLinearSolverWithMatrix<
    DOpEWrapper::PreconditionIdentity_Wrapper<MATRIX>, SPARSITYPATTERN, MATRIX,
    VECTOR> LINEARSOLVER;

typedef NewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR> NLS;
typedef StatReducedProblem<NLS, NLS, INTEGRATOR, INTEGRATOR, OP, VECTOR, CDIM,
    DIM> SSolver;
//The two optimization algorightms using SNOPT/IPOPT
typedef Reduced_SnoptAlgorithm<OP, VECTOR> SNOPT_Alg;
typedef Reduced_IpoptAlgorithm<OP, VECTOR> IPOPT_Alg;

int
main(int argc, char **argv)
{
  /**
   * Distributed control in the right hand side of
   *  a linear elliptic PDE  plus some box constraints
   *  for the control. We use IPOPT/SNOPT for the optimization.
   *
   */
  std::string paramfile = "dope.prm";

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
  SNOPT_Alg::declare_params(pr);
  IPOPT_Alg::declare_params(pr);
  NLS::declare_params(pr);

  pr.read_parameters(paramfile);

  const int niter = 1;

  Triangulation<DIM> triangulation;
  GridGenerator::hyper_cube(triangulation, 0, 1);
  triangulation.refine_global(5);

  FE<DIM> control_fe(FE_Q<DIM>(1), 1);
  FE<DIM> state_fe(FE_Q<DIM>(2), 1);

  QUADRATURE quadrature_formula(2);
  FACEQUADRATURE face_quadrature_formula(2);
  IDC idc(quadrature_formula, face_quadrature_formula);

  PDE LPDE;
  COSTFUNCTIONAL LFunc;

  //AuxFunctionals
  LocalPointFunctional<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> LPF;
  LocalMeanValueFunctional<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> LMF;

  //Add Constrained description
  std::vector<std::vector<unsigned int> > lcc(1); //1 Control Block
  lcc[0].resize(2);
  lcc[0][0] = 1; //each component is constrained individualy
  lcc[0][1] = 2; // each two constraints (lower and upper bound)
  Constraints constraints(lcc, 0); //Second entry defines the numer of global constraints, here we have none

  MethodOfLines_SpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR, CDIM,
      DIM> DOFH(triangulation, control_fe, state_fe, constraints,
      DOpEtypes::stationary);
  LocalConstraintAccessor CA;
  LocalConstraint<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> LC(CA);

  OP P(LFunc, LPDE, LC, DOFH);

  P.AddFunctional(&LPF);
  P.AddFunctional(&LMF);
  //Dirichlet data
  std::vector<bool> comp_mask(1, true);
  DOpEWrapper::ZeroFunction<DIM> zf(1);
  DD DD_1(zf);
  P.SetDirichletBoundaryColors(0, comp_mask, &DD_1);

  SSolver solver(&P, "fullmem", pr, idc, 2);

  //uncomment to use SNOPT if installed
  //SNOPT_Alg Alg(&P,&solver,"fullmem",pr);
  IPOPT_Alg Alg(&P, &solver, "fullmem", pr);

  Alg.ReInit();
  ControlVector<BlockVector<double> > q(&DOFH, "fullmem");

  for (int i = 0; i < niter; i++)
  {
    try
    {
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
      DOFH.RefineSpace();
      Alg.ReInit();
    }
  }
  return 0;
}
