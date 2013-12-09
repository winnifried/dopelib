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
#include <fe/fe_dgp.h>
#include <dofs/dof_tools.h>
#include <base/quadrature_lib.h>
#include <base/function.h>

#include "generalized_mma_algorithm.h"
#include "optproblemcontainer.h"
#include "functionalinterface.h"
#include "pdeinterface.h"
#include "statreducedproblem.h" 
#include "voidreducedproblem.h"
#include "newtonsolver.h"
#include "directlinearsolver.h"
#include "integrator.h"
#include "parameterreader.h"
#include "mol_spacetimehandler.h"
#include "simpledirichletdata.h"
#include "constraints.h"
#include "localconstraints.h"
#include "localconstraintaccessor.h"
#include "integratordatacontainer.h"

#include "localpde.h"
#include "localfunctional.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

#define DOFHANDLER DoFHandler
#define FE FESystem

const static int DIM = 2;
const static int CDIM = 2;

typedef QGauss<DIM> QUADRATURE;
typedef QGauss<DIM - 1> FACEQUADRATURE;

typedef BlockSparseMatrix<double> MATRIX;
typedef BlockSparsityPattern SPARSITYPATTERN;
typedef BlockVector<double> VECTOR;

#define CDC ElementDataContainer
#define FDC FaceDataContainer

typedef FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> FUNCTIONALINTERFACE;
typedef LocalFunctional<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> COSTFUNCTIONAL;

typedef SimpleDirichletData<VECTOR, DIM> DD;
typedef LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM> PDE;
typedef ConstraintInterface<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> CONS;

typedef SpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR, CDIM, DIM> STH;

typedef OptProblemContainer<FUNCTIONALINTERFACE, COSTFUNCTIONAL, PDE, DD, CONS,
    SPARSITYPATTERN, VECTOR, CDIM, DIM> OP;

typedef AugmentedLagrangianProblem<LocalConstraintAccessor, STH, OP, CDIM, DIM,
    1> ALagOP;
typedef IntegratorDataContainer<DOFHANDLER, Quadrature<DIM>, Quadrature<1>,
    VECTOR, DIM> IDC;
typedef Integrator<IDC, VECTOR, double, DIM> INTEGRATOR;
typedef DirectLinearSolverWithMatrix<SPARSITYPATTERN, MATRIX, VECTOR> LINEARSOLVER;
typedef NewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR> NLS;
typedef StatReducedProblem<NLS, NLS, INTEGRATOR, INTEGRATOR, OP, VECTOR, CDIM,
    DIM> SSolver;
typedef VoidReducedProblem<NLS, INTEGRATOR, ALagOP, VECTOR, CDIM, DIM> ALagSSolver;
typedef GeneralizedMMAAlgorithm<LocalConstraintAccessor, IDC, STH, OP, VECTOR,
    ALagSSolver, CDIM, DIM, 1> MMA;

int
main(int argc, char **argv)
{
  /**
   * 1This example implements the minimum compliance problem for
   * the thickness optimization of an MBB-Beam. Using the
   * MMA-Method of K. Svanberg together with an augmented
   * Lagrangian approach for the subproblems following M. Stingl.
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
  MMA::declare_params(pr);
  NLS::declare_params(pr);

  pr.read_parameters(paramfile);

  const int niter = 1;

  //Create triangulation
  Triangulation<DIM> triangulation;
  std::vector<unsigned int> rep(2);
  rep[0] = 2;
  rep[1] = 1;
  GridGenerator::subdivided_hyper_rectangle(triangulation, rep,
      Point<DIM>(0, 0), Point<DIM>(2, 1), true);
  triangulation.refine_global(3);

  FE<DIM> control_fe(FE_DGP<DIM>(0), 1);
  FE<DIM> state_fe(FE_Q<DIM>(2), 2);

  QUADRATURE quadrature_formula(3);
  FACEQUADRATURE face_quadrature_formula(2);
  IDC idc(quadrature_formula, face_quadrature_formula);

  PDE LPDE;
  COSTFUNCTIONAL LFunc;

  { //Set Dirichlet Boundary!
    for (Triangulation<DIM>::active_cell_iterator element =
        triangulation.begin_active(); element != triangulation.end(); ++element)
      for (unsigned int f = 0; f < GeometryInfo<DIM>::faces_per_cell; ++f)
      {
        if (element->face(f)->at_boundary())
        {
          if (element->face(f)->center()[1] == 0)
          {
            element->face(f)->set_all_boundary_indicators(5);
            if (fabs(element->face(f)->center()[0] - 2.)
                < std::max(0.25, element->face(f)->diameter()))
            {
              element->face(f)->set_all_boundary_indicators(2);
            }
          }
        }
      }
  }

  //Add Constrained description
  std::vector<std::vector<unsigned int> > lcc(1); //1 Control Block
  lcc[0].resize(2);
  lcc[0][0] = 1; //each component is constrained individualy
  lcc[0][1] = 2; // each two constraints (lower and upper bound)
  Constraints constraints(lcc, 1);

  MethodOfLines_SpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR, CDIM,
      DIM> DOFH(triangulation, control_fe, state_fe, constraints,
      DOpEtypes::stationary);

  LocalConstraintAccessor CA;
  LocalConstraint<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> LC(CA);

  OP P(LFunc, LPDE, LC, DOFH);

  std::vector<bool> comp_mask(2);
  comp_mask[0] = false;
  comp_mask[1] = true;
  std::vector<bool> comp_mask_2(2);
  comp_mask_2[0] = true;
  comp_mask_2[1] = false;
  DOpEWrapper::ZeroFunction<DIM> zf(2);
  DD DD_1(zf);
  P.SetDirichletBoundaryColors(2, comp_mask, &DD_1);
  P.SetDirichletBoundaryColors(0, comp_mask_2, &DD_1);

  P.SetBoundaryFunctionalColors(3);
  P.SetBoundaryEquationColors(3);

  SSolver solver(&P, "fullmem", pr, idc);

  MMA Alg(&P, &CA, &solver, "fullmem", pr, idc);

  Alg.ReInit();
  ControlVector<VECTOR> q(&DOFH, "fullmem");
  //init q
  {
    q = 0.4;
  }
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
      { //Set Dirichlet Boundary!
        for (Triangulation<DIM>::active_cell_iterator element =
            triangulation.begin_active(); element != triangulation.end(); ++element)
          for (unsigned int f = 0; f < GeometryInfo<DIM>::faces_per_cell; ++f)
          {
            if (element->face(f)->at_boundary())
            {
              if (element->face(f)->center()[1] == 0)
              {
                element->face(f)->set_all_boundary_indicators(5);
                if ((fabs(element->face(f)->center()[0] - 2.)
                    < std::max(0.25, element->face(f)->diameter())))
                {
                  element->face(f)->set_all_boundary_indicators(2);
                }
              }
            }
          }
      }
      Alg.ReInit();
    }
  }
  return 0;
}

#undef FDC
#undef CDC
#undef FE
#undef DOFHANDLER
