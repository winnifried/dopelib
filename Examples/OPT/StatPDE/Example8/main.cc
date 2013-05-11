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

#include "reduced_snopt_algorithm.h"
#include "optproblemcontainer.h"
#include "functionalinterface.h"
#include "pdeinterface.h"
#include "statreducedproblem.h" 
#include "newtonsolver.h"
#include "cglinearsolver.h"
#include "directlinearsolver.h"
#include "integrator.h"
#include "parameterreader.h"
#include "mol_spacetimehandler.h"
#include "simpledirichletdata.h"
#include "constraints.h"
#include "localconstraints.h"
#include "localconstraintaccessor.h"
#include "integratordatacontainer.h"
#include "pointconstraintsmaker.h"

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <cmath>

#include <grid/tria.h>
#include <dofs/dof_handler.h>
#include <grid/grid_generator.h>
#include <fe/fe_q.h>
#include <fe/fe_dgp.h>
#include <dofs/dof_tools.h>
#include <base/quadrature_lib.h>
#include <base/function.h>

#include "localpde.h"
#include "localfunctional.h"

using namespace dealii;
using namespace DOpE;

#define CDC CellDataContainer
#define FDC FaceDataContainer
#define VECTOR BlockVector<double>
#define SPARSITYPATTERN BlockSparsityPattern
#define MATRIX BlockSparseMatrix<double>
#define DOFHANDLER DoFHandler
#define FE FESystem
#define FUNC FunctionalInterface<CDC,FDC,DOFHANDLER,VECTOR,2,2>
#define PDE PDEInterface<CDC,FDC,DOFHANDLER,VECTOR,2>
#define DD DirichletDataInterface<VECTOR,2>
#define CONS ConstraintInterface<CDC,FDC,DOFHANDLER,VECTOR,2,2>

typedef SpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR, 2, 2> STH;

typedef OptProblemContainer<FUNC, FUNC, PDE, DD, CONS, SPARSITYPATTERN, VECTOR, 2, 2> OP;

typedef IntegratorDataContainer<DOFHANDLER, Quadrature<2>,
    Quadrature<1>, VECTOR, 2> IDC;
typedef Integrator<IDC, VECTOR, double, 2> INTEGRATOR;

//Uncomment to use a CG-Method with Identity Preconditioner
//typedef CGLinearSolverWithMatrix<PreconditionIdentity,OP,>SPARSITYPATTERN, MATRIX, VECTOR LINEARSOLVER;
//Uncomment to use UMFPACK
typedef DirectLinearSolverWithMatrix<SPARSITYPATTERN, MATRIX, VECTOR> LINEARSOLVER;

typedef NewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR> NLS;
typedef StatReducedProblem<NLS, NLS, INTEGRATOR, INTEGRATOR, OP, VECTOR, 2, 2> SSolver;

typedef Reduced_SnoptAlgorithm<OP, BlockVector<double>> MMA;

int
main(int argc, char **argv)
{

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
  MMA::declare_params(pr);
  NLS::declare_params(pr);

  pr.read_parameters(paramfile);

  Triangulation<2> triangulation;
  std::vector<unsigned int> rep(2);
  rep[0] = 2;
  rep[1] = 1;
  GridGenerator::subdivided_hyper_rectangle(triangulation, rep, Point<2>(0, 0),
      Point<2>(2, 1), true);

  FE<2> control_fe(FE_DGP<2>(0), 1);
  FE<2> state_fe(FE_Q<2>(2), 2);

  QGauss<2> quadrature_formula(3);
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


  std::vector<Point<2> > c_points(1);
  std::vector<std::vector<bool> > c_comps(1,std::vector<bool>(2));
  c_points[0][0] = 2.0;    //We want to constrain the displacement at the vertex (2.0, 0.0)
  c_points[0][1] = 0.0;
  c_comps[0][0] = false;   //But we allow displacements in x-dir (comp = 0) to be free
  c_comps[0][1] = true;    //Only the y-displacement is fixed.
  DOpE::PointConstraints<DOFHANDLER,2,2> constraints_mkr(c_points,c_comps);

  MethodOfLines_SpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR, 2, 2> DOFH(
      triangulation, control_fe, state_fe, constraints, DOpEtypes::stationary);
  
  DOFH.SetUserDefinedDoFConstraints(constraints_mkr);

  LocalConstraintAccessor CA;
  LocalConstraint<CDC, FDC, DOFHANDLER, VECTOR, 2, 2> LC(CA);

  OP P(LFunc, LPDE, LC, DOFH);

  std::vector<bool> comp_mask(2);
  comp_mask[0] = true;
  comp_mask[1] = false;
  DOpEWrapper::ZeroFunction<2> zf(2);
  SimpleDirichletData<BlockVector<double>, 2> DD_1(zf);
  P.SetDirichletBoundaryColors(0, comp_mask, &DD_1);

  P.SetBoundaryFunctionalColors(3);
  P.SetBoundaryEquationColors(3);

  SSolver solver(&P, "fullmem", pr, idc);

  MMA Alg(&P, &solver, "fullmem", pr);

  int niter = 2;

  Alg.ReInit();
  ControlVector<BlockVector<double> > q(&DOFH, "fullmem");
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
