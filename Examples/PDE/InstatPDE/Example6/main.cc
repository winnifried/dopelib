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


//c++ includes
#include <iostream>
#include <fstream>

//deal.ii includes
#include <base/quadrature_lib.h>
#include <base/function.h>
#include <dofs/dof_handler.h>
#include <fe/fe_q.h>
#include <fe/fe_dgp.h> //for discont. finite elements
#include <fe/fe_nothing.h>
#include <grid/tria.h>
#include <grid/grid_in.h>
#include <grid/tria_boundary_lib.h>
#include <grid/grid_generator.h>

//DOpE includes
#include "parameterreader.h"
#include "directlinearsolver.h"
#include "integrator.h"
#include "mol_spacetimehandler.h"
#include "simpledirichletdata.h"
#include "integratordatacontainer.h"
#include "newtonsolver.h"
#include "functionalinterface.h"
#include "noconstraints.h"

#include "instatreducedproblem.h"
#include "instat_step_newtonsolver.h"
#include "reducednewtonalgorithm.h"
#include "instatoptproblemcontainer.h"

#include "backward_euler_problem.h"

//Problem specific includes
#include "localpde.h"
#include "localfunctional.h"
#include "functionals.h"
#include "indexsetter.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

// Define dimensions for control- and state problem
const static int DIM = 2;

#define DOFHANDLER hp::DoFHandler
#define FE hp::FECollection
#define CDC ElementDataContainer
#define FDC FaceDataContainer

typedef hp::QCollection<DIM> QUADRATURE;
typedef hp::QCollection<DIM - 1> FACEQUADRATURE;
typedef BlockSparseMatrix<double> MATRIX;
typedef BlockSparsityPattern SPARSITYPATTERN;
typedef BlockVector<double> VECTOR;

typedef FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM> FUNC;

typedef OptProblemContainer<LocalFunctional<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM> , FUNC,
LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
SimpleDirichletData<VECTOR, DIM>,
NoConstraints<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM>, SPARSITYPATTERN,
VECTOR, DIM, DIM, FE, DOFHANDLER> OP_BASE;

typedef StateProblem<OP_BASE, LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
SimpleDirichletData<VECTOR, DIM>, SPARSITYPATTERN, VECTOR, DIM> PROB;

// Typedefs for timestep problem
#define TSP BackwardEulerProblem
//FIXME: This should be a reasonable dual timestepping scheme
#define DTSP BackwardEulerProblem

typedef InstatOptProblemContainer<TSP, DTSP, FUNC,
    LocalFunctional<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM>,
    LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
    SimpleDirichletData<VECTOR, DIM>,
    NoConstraints<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM>, SPARSITYPATTERN,
    VECTOR, DIM, DIM,  FE, DOFHANDLER> OP;
#undef TSP
#undef DTSP

typedef IntegratorDataContainer<DOFHANDLER, QUADRATURE,
FACEQUADRATURE, VECTOR, DIM> IDC;
typedef Integrator<IDC, VECTOR, double, DIM> INTEGRATOR;
typedef DirectLinearSolverWithMatrix<SPARSITYPATTERN, MATRIX, VECTOR> LINEARSOLVER;
typedef NewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR> CNLS;
typedef InstatStepNewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR> NLS;
typedef ReducedNewtonAlgorithm<OP, VECTOR> RNA;
typedef InstatReducedProblem<CNLS, NLS, INTEGRATOR, INTEGRATOR, OP, VECTOR, DIM,
    DIM> RP;
int
main(int argc, char **argv)
{
  /**
   * First version of quasi-static Biot-Lame_Navier problem.
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
  LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>::declare_params(pr);
  pr.read_parameters(paramfile);

  Triangulation<DIM> triangulation;

  GridIn<DIM> grid_in;
  grid_in.attach_triangulation(triangulation);

  // Grid for Benchmark with flag
  std::ifstream input_file("rectangle_mandel_elasticity.inp");

  grid_in.read_ucd(input_file);

  FESystem<DIM> control_fe(FE_Nothing<DIM>(), 1);
  hp::FECollection<DIM> control_fe_collection(control_fe);
  control_fe_collection.push_back(control_fe); //Need same number of entries in FECollection as state

  // FE for the state equation: u,p
  FESystem<DIM> state_fe(FE_Q<DIM>(2), 2, FE_Q<DIM>(1), 1);
  FESystem<DIM> state_fe_2(FE_Q<DIM>(2), 2, FE_Nothing<DIM>(1), 1);
  hp::FECollection<DIM> state_fe_collection(state_fe);
  state_fe_collection.push_back(state_fe_2);

  QGauss<DIM> quadrature_formula(3);
  QGauss<DIM - 1> face_quadrature_formula(3);
  hp::QCollection<DIM> q_coll(quadrature_formula);
  q_coll.push_back(quadrature_formula);
  hp::QCollection<DIM - 1> face_q_coll(face_quadrature_formula);
  face_q_coll.push_back(face_quadrature_formula);

  IDC idc(q_coll, face_q_coll);

  LocalPDE<CDC, FDC, DOFHANDLER,VECTOR, DIM> LPDE(pr);
  LocalFunctional<CDC, FDC, DOFHANDLER,VECTOR, DIM, DIM> LFunc;

  LocalPointFunctionalP1<CDC, FDC, DOFHANDLER,VECTOR, DIM, DIM> LPFP1;
  LocalPointFunctionalP2<CDC, FDC, DOFHANDLER,VECTOR, DIM, DIM> LPFP2;

  //Time grid of [0,25]
  Triangulation<1> times;
  GridGenerator::subdivided_hyper_cube(times, 100, 0, 100000);

  triangulation.refine_global(3);
  ActiveFEIndexSetter<DIM> indexsetter;
  MethodOfLines_SpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR,
  DIM, DIM> DOFH(triangulation,
      control_fe_collection,
      state_fe_collection,
      times, DOpEtypes::undefined,
      indexsetter);

  NoConstraints<ElementDataContainer, FaceDataContainer, DOFHANDLER, VECTOR,
  DIM, DIM> Constraints;

  OP P(LFunc, LPDE, Constraints, DOFH);

  //  P.HasFaces();
  P.AddFunctional(&LPFP1); // p1
  P.AddFunctional(&LPFP2); // p2

  std::vector<bool> comp_mask(3);
  DOpEWrapper::ZeroFunction<DIM> zf(3);
  SimpleDirichletData<VECTOR, DIM> DD1(zf);

  comp_mask[0] = true; // ux
  comp_mask[1] = false; // uy
  comp_mask[2] = false; // p

  P.SetDirichletBoundaryColors(0, comp_mask, &DD1);

  comp_mask[0] = false; // ux
  comp_mask[1] = true; // uy
  comp_mask[2] = false; // p

  P.SetDirichletBoundaryColors(2, comp_mask, &DD1);

  comp_mask[0] = false; // ux
  comp_mask[1] = false; // uy
  comp_mask[2] = true; // p

  P.SetDirichletBoundaryColors(1, comp_mask, &DD1);

  comp_mask[0] = false; // ux
  comp_mask[1] = false; // uy
  comp_mask[2] = true; // p

  P.SetDirichletBoundaryColors(11, comp_mask, &DD1);

  comp_mask[0] = false; // ux
  comp_mask[1] = false; // uy
  comp_mask[2] = false; // p

  P.SetDirichletBoundaryColors(3, comp_mask, &DD1);

  P.SetBoundaryEquationColors(3); // top boundary

  P.SetInitialValues(&zf);

  RP solver(&P, "fullmem", pr, idc);
  RNA Alg(&P, &solver, pr);

  // Mesh-refinement cycles
  int niter = 1;
  Alg.ReInit();
  ControlVector<VECTOR> q(&DOFH, "fullmem");

  for (int i = 0; i < niter; i++)
  {
    try
    {

      Alg.SolveForward(q);

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
    }
  }

  return 0;
}

#undef FDC
#undef CDC
#undef FE
#undef DOFHANDLER

