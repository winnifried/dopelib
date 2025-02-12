/**
 *
 * Copyright (C) 2012-2018 by the DOpElib authors
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
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>

#include "generalized_mma_algorithm.h"
#include <container/optproblemcontainer.h>
#include <interfaces/functionalinterface.h>
#include <interfaces/pdeinterface.h>
#include <reducedproblems/statreducedproblem.h>
#include "voidreducedproblem.h"
#include <templates/newtonsolver.h>
#include <templates/directlinearsolver.h>
#include <templates/integrator.h>
#include <include/parameterreader.h>
#include <basic/mol_spacetimehandler.h>
#include <problemdata/simpledirichletdata.h>
#include <basic/constraints.h>
#include "localconstraints.h"
#include "localconstraintaccessor.h"
#include <container/integratordatacontainer.h>

#include "localpde.h"
#include "localfunctional.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

#if DEAL_II_VERSION_GTE(9,3,0)
#define DOFHANDLER false
#else
#define DOFHANDLER DoFHandler
#endif

#define FE FESystem

const static int DIM = 2;
const static int CDIM = 2;

typedef QGauss<DIM> QUADRATURE;
typedef QGauss<DIM - 1> FACEQUADRATURE;

typedef BlockSparseMatrix<double> MATRIX;
typedef BlockSparsityPattern SPARSITYPATTERN;
typedef BlockVector<double> VECTOR;

#define EDC ElementDataContainer
#define FDC FaceDataContainer

typedef FunctionalInterface<EDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> FUNCTIONALINTERFACE;
typedef LocalFunctional<EDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> COSTFUNCTIONAL;

typedef SimpleDirichletData<VECTOR, DIM> DD;
typedef LocalPDE<EDC, FDC, DOFHANDLER, VECTOR, DIM> PDE;
typedef ConstraintInterface<EDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> CONS;

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
        DIM> RP;
typedef VoidReducedProblem<NLS, INTEGRATOR, ALagOP, VECTOR, CDIM, DIM> ALagRP;
typedef GeneralizedMMAAlgorithm<LocalConstraintAccessor, IDC, STH, OP, VECTOR,
        ALagRP, CDIM, DIM, 1> MMA;

int
main(int argc, char **argv)
{
  /**
   * This example implements the minimum compliance problem for
   * the thickness optimization of an MBB-Beam. Using the
   * MMA-Method of K. Svanberg together with an augmented
   * Lagrangian approach for the subproblems following M. Stingl.
   */
  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv);

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

  {
    //Set Dirichlet Boundary!
    for (Triangulation<DIM>::active_cell_iterator element =
           triangulation.begin_active(); element != triangulation.end(); ++element)
      for (unsigned int f = 0; f < element->n_faces(); ++f)
        {
          if (element->face(f)->at_boundary())
            {
              if (element->face(f)->center()[1] == 0)
                {
                  element->face(f)->set_all_boundary_ids(5);
                  if (fabs(element->face(f)->center()[0] - 2.)
                      < std::max(0.25, element->face(f)->diameter()))
                    {
                      element->face(f)->set_all_boundary_ids(2);
                    }
                }
            }
        }
  }

  //Add Constrained description
  std::vector<std::vector<unsigned int> > lcc(1); //1 Control Block
  lcc[0].resize(2);
  lcc[0][0] = 1; //each component is constrained individualy
  lcc[0][1] = 2; // number of constraints (lower and upper bound)
  Constraints constraints(lcc, 1); // here, we impose one global constraint

  MethodOfLines_SpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR, CDIM,
                                 DIM> DOFH(triangulation, control_fe, state_fe, constraints,
                                           DOpEtypes::stationary);

  LocalConstraintAccessor CA;
  LocalConstraint<EDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> LC(CA);

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

  RP solver(&P, DOpEtypes::VectorStorageType::fullmem, pr, idc);

  MMA Alg(&P, &CA, &solver, DOpEtypes::VectorStorageType::fullmem, pr, idc);

  Alg.ReInit();
  ControlVector<VECTOR> q(&DOFH, DOpEtypes::VectorStorageType::fullmem,pr);
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
          {
            //Set Dirichlet Boundary!
            for (Triangulation<DIM>::active_cell_iterator element =
                   triangulation.begin_active(); element != triangulation.end(); ++element)
              for (unsigned int f = 0; f < element->n_faces(); ++f)
                {
                  if (element->face(f)->at_boundary())
                    {
                      if (element->face(f)->center()[1] == 0)
                        {
                          element->face(f)->set_all_boundary_ids(5);
                          if ((fabs(element->face(f)->center()[0] - 2.)
                               < std::max(0.25, element->face(f)->diameter())))
                            {
                              element->face(f)->set_all_boundary_ids(2);
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
#undef EDC
#undef FE
#undef DOFHANDLER
