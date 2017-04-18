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

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <cmath>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/base/quadrature_lib.h>

#include <opt_algorithms/reduced_snopt_algorithm.h>
#include <container/optproblemcontainer.h>
#include <interfaces/functionalinterface.h>
#include <interfaces/pdeinterface.h>
#include <reducedproblems/statreducedproblem.h>
#include <templates/newtonsolver.h>
#include <templates/directlinearsolver.h>
#include <templates/integrator.h>
#include <include/parameterreader.h>
#include <basic/mol_spacetimehandler.h>
#include <problemdata/simpledirichletdata.h>
#include <basic/constraints.h>
#include <container/integratordatacontainer.h>
#include <include/pointconstraintsmaker.h>

#include "localconstraints.h"
#include "localpde.h"
#include "localfunctional.h"

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
typedef DirectLinearSolverWithMatrix<SPARSITYPATTERN, MATRIX, VECTOR> LINEARSOLVER;
typedef NewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR> NLS;
typedef StatReducedProblem<NLS, NLS, INTEGRATOR, INTEGRATOR, OP, VECTOR, CDIM,
        DIM> RP;
typedef Reduced_SnoptAlgorithm<OP, VECTOR> SNOPT_Alg;

int
main(int argc, char **argv)
{
  /**
   * This example implements the topology optimization of an MBB-Beam given in
   * OPT/StatPDE/Example6 using the SIMP method and uses SNOPT for the optimization.
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
  NLS::declare_params(pr);

  pr.read_parameters(paramfile);

  Triangulation<DIM> triangulation;
  std::vector<unsigned int> rep(2);
  rep[0] = 2;
  rep[1] = 1;
  GridGenerator::subdivided_hyper_rectangle(triangulation, rep,
                                            Point<DIM>(0, 0), Point<DIM>(2, 1), true);

  FE<DIM> control_fe(FE_DGP<DIM>(0), 1);
  FE<DIM> state_fe(FE_Q<DIM>(2), 2);

  QGauss<DIM> quadrature_formula(3);
  QGauss<1> face_quadrature_formula(2);
  IDC idc(quadrature_formula, face_quadrature_formula);

  LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, 2> LPDE;
  LocalFunctional<CDC, FDC, DOFHANDLER, VECTOR, 2, 2> LFunc;

  //triangulation.refine_global (5);
  triangulation.refine_global(3);

  //Add Constrained description
  std::vector<std::vector<unsigned int> > lcc(1); //1 Control Block
  lcc[0].resize(2);
  lcc[0][0] = 1; //each component is constrained individualy
  lcc[0][1] = 2; // each two constraints (lower and upper bound)
  Constraints constraints(lcc, 1);

  std::vector<Point<DIM> > c_points(1);
  std::vector<std::vector<bool> > c_comps(1, std::vector<bool>(2));
  c_points[0][0] = 2.0; //We want to constrain the displacement at the vertex (2.0, 0.0)
  c_points[0][1] = 0.0;
  c_comps[0][0] = false; //But we allow displacements in x-dir (comp = 0) to be free
  c_comps[0][1] = true; //Only the y-displacement is fixed.
  DOpE::PointConstraints<DOFHANDLER, 2, 2> constraints_mkr(c_points, c_comps);

  MethodOfLines_SpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR, 2, 2> DOFH(
    triangulation, control_fe, state_fe, constraints, DOpEtypes::stationary);

  DOFH.SetUserDefinedDoFConstraints(constraints_mkr);

  LocalConstraint<CDC, FDC, DOFHANDLER, VECTOR, 2, 2> LC;

  OP P(LFunc, LPDE, LC, DOFH);

  std::vector<bool> comp_mask(2);
  comp_mask[0] = true;
  comp_mask[1] = false;
  DOpEWrapper::ZeroFunction<DIM> zf(2);
  SimpleDirichletData<BlockVector<double>, 2> DD_1(zf);
  P.SetDirichletBoundaryColors(0, comp_mask, &DD_1);

  P.SetBoundaryFunctionalColors(3);
  P.SetBoundaryEquationColors(3);

  RP solver(&P, DOpEtypes::VectorStorageType::fullmem, pr, idc);

  SNOPT_Alg Alg(&P, &solver, DOpEtypes::VectorStorageType::fullmem, pr);

  int niter = 2;

  Alg.ReInit();
  ControlVector<BlockVector<double> > q(&DOFH, DOpEtypes::VectorStorageType::fullmem);
  //init q
  {
    q = 0.4;
  }
  for (int i = 0; i < niter; i++)
    {
      try
        {
          Alg.GetOutputHandler()->SetIterationNumber(0, "p_iter");
          LPDE.SetP(1);
          Alg.Solve(q);
          Alg.GetOutputHandler()->SetIterationNumber(1, "p_iter");
          LPDE.SetP(4);
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
