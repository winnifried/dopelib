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
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/precondition_block.h>

#include <opt_algorithms/reduced_ipopt_algorithm.h>
#include <opt_algorithms/reduced_snopt_algorithm.h>
#include <container/optproblemcontainer.h>
#include <interfaces/functionalinterface.h>
#include <reducedproblems/statreducedproblem.h>
#include <templates/newtonsolver.h>
#include <templates/cglinearsolver.h>
#include <templates/integrator.h>
#include <include/parameterreader.h>
#include <basic/mol_spacetimehandler.h>
#include <problemdata/simpledirichletdata.h>
#include <problemdata/noconstraints.h>
#include <wrapper/preconditioner_wrapper.h>
#include <container/integratordatacontainer.h>

#include "localpde.h"
#include "localfunctional.h"
#include "functionals.h"
#include "localconstraints.h"

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

#define CDC ElementDataContainer
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
        DIM> RP;
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
  RP::declare_params(pr);
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
  lcc[0][0] = 1; // each component is constrained individualy
  lcc[0][1] = 2; // number of constraints (lower and upper bound)
  Constraints constraints(lcc, 0); //Second entry defines the numer of global constraints, here we have none

  MethodOfLines_SpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR, CDIM,
                                 DIM> DOFH(triangulation, control_fe, state_fe, constraints,
                                           DOpEtypes::stationary);
  LocalConstraint<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> LC;

  OP P(LFunc, LPDE, LC, DOFH);

  P.AddFunctional(&LPF);
  P.AddFunctional(&LMF);
  //Dirichlet data
  std::vector<bool> comp_mask(1, true);
  DOpEWrapper::ZeroFunction<DIM> zf(1);
  DD DD_1(zf);
  P.SetDirichletBoundaryColors(0, comp_mask, &DD_1);

  RP solver(&P, DOpEtypes::VectorStorageType::fullmem, pr, idc, 2);

  //uncomment to use SNOPT if installed
  //SNOPT_Alg Alg(&P,&solver,DOpEtypes::VectorStorageType::fullmem,pr);
  IPOPT_Alg Alg(&P, &solver, DOpEtypes::VectorStorageType::fullmem, pr);

  Alg.ReInit();
  ControlVector<BlockVector<double> > q(&DOFH, DOpEtypes::VectorStorageType::fullmem);

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
#undef FDC
#undef CDC
#undef FE
#undef DOFHANDLER
