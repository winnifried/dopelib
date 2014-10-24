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
#include "fractional_step_theta_step_newtonsolver.h"
#include "functionalinterface.h"
#include "noconstraints.h"

#include "instatreducedproblem.h"
#include "instat_step_newtonsolver.h"
#include "reducednewtonalgorithm.h"
#include "instatoptproblemcontainer.h"

#include "forward_euler_problem.h"
#include "backward_euler_problem.h"
#include "crank_nicolson_problem.h"
#include "shifted_crank_nicolson_problem.h"
#include "fractional_step_theta_problem.h"

//Problem specific includes
#include "localpde.h"
#include "localfunctional.h"
#include "functionals.h"
#include "my_functions.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

// Define dimensions for control- and state problem
const static int DIM = 1;

#define DOFHANDLER DoFHandler
#define FE FESystem
#define CDC ElementDataContainer
#define FDC FaceDataContainer

typedef QGauss<DIM> QUADRATURE;
typedef QGauss<DIM - 1> FACEQUADRATURE;
typedef BlockSparseMatrix<double> MATRIX;
typedef BlockSparsityPattern SPARSITYPATTERN;
typedef BlockVector<double> VECTOR;

//typedef FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM> FUNC;

typedef OptProblemContainer<
    LocalFunctional<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM>,
    LocalPointFunctional<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM>,
    LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
    SimpleDirichletData<VECTOR, DIM>,
    NoConstraints<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM>, SPARSITYPATTERN,
    VECTOR, DIM, DIM> OP_BASE;

typedef StateProblem<OP_BASE, LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
    SimpleDirichletData<VECTOR, DIM>, SPARSITYPATTERN, VECTOR, DIM> PROB;
// Typedefs for timestep problem

#define TSP1 ForwardEulerProblem
#define TSP2 BackwardEulerProblem
#define TSP3 CrankNicolsonProblem
#define TSP4 ShiftedCrankNicolsonProblem
#define TSP5 FractionalStepThetaProblem
//FIXME: This should be a reasonable dual timestepping scheme
#define DTSP1 ForwardEulerProblem
#define DTSP2 BackwardEulerProblem
#define DTSP3 CrankNicolsonProblem
#define DTSP4 ShiftedCrankNicolsonProblem
#define DTSP5 FractionalStepThetaProblem

typedef InstatOptProblemContainer<TSP1, DTSP1,
    LocalPointFunctional<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM>,
    LocalFunctional<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM>,
    LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
    SimpleDirichletData<VECTOR, DIM>,
    NoConstraints<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM>, SPARSITYPATTERN,
    VECTOR, DIM, DIM> OP1;
typedef InstatOptProblemContainer<TSP2, DTSP2,
    LocalPointFunctional<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM>,
    LocalFunctional<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM>,
    LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
    SimpleDirichletData<VECTOR, DIM>,
    NoConstraints<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM>, SPARSITYPATTERN,
    VECTOR, DIM, DIM> OP2;
typedef InstatOptProblemContainer<TSP3, DTSP3,
    LocalPointFunctional<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM>,
    LocalFunctional<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM>,
    LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
    SimpleDirichletData<VECTOR, DIM>,
    NoConstraints<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM>, SPARSITYPATTERN,
    VECTOR, DIM, DIM> OP3;
typedef InstatOptProblemContainer<TSP4, DTSP4,
    LocalPointFunctional<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM>,
    LocalFunctional<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM>,
    LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
    SimpleDirichletData<VECTOR, DIM>,
    NoConstraints<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM>, SPARSITYPATTERN,
    VECTOR, DIM, DIM> OP4;
typedef InstatOptProblemContainer<TSP5, DTSP5,
    LocalPointFunctional<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM>,
    LocalFunctional<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM>,
    LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
    SimpleDirichletData<VECTOR, DIM>,
    NoConstraints<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM>, SPARSITYPATTERN,
    VECTOR, DIM, DIM> OP5;

#undef TSP1  
#undef TSP2  
#undef TSP3  
#undef TSP4  
#undef TSP5  
#undef DTSP1 
#undef DTSP2 
#undef DTSP3 
#undef DTSP4 
#undef DTSP5 

typedef IntegratorDataContainer<DOFHANDLER, QUADRATURE,
    FACEQUADRATURE, VECTOR, DIM> IDC;

typedef Integrator<IDC, VECTOR, double, DIM> INTEGRATOR;

typedef DirectLinearSolverWithMatrix<SPARSITYPATTERN, MATRIX, VECTOR> LINEARSOLVER;

typedef NewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR> CNLS;
typedef InstatStepNewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR> NLS;

typedef FractionalStepThetaStepNewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR> NLS2;

typedef ReducedNewtonAlgorithm<OP1, VECTOR> RNA1;
typedef ReducedNewtonAlgorithm<OP2, VECTOR> RNA2;
typedef ReducedNewtonAlgorithm<OP3, VECTOR> RNA3;
typedef ReducedNewtonAlgorithm<OP4, VECTOR> RNA4;
typedef ReducedNewtonAlgorithm<OP5, VECTOR> RNA5;

typedef InstatReducedProblem<CNLS, NLS, INTEGRATOR, INTEGRATOR, OP1, VECTOR,
    DIM, DIM> RP1;
typedef InstatReducedProblem<CNLS, NLS, INTEGRATOR, INTEGRATOR, OP2, VECTOR,
    DIM, DIM> RP2;
typedef InstatReducedProblem<CNLS, NLS, INTEGRATOR, INTEGRATOR, OP3, VECTOR,
    DIM, DIM> RP3;
typedef InstatReducedProblem<CNLS, NLS, INTEGRATOR, INTEGRATOR, OP4, VECTOR,
    DIM, DIM> RP4;
typedef InstatReducedProblem<CNLS, NLS2, INTEGRATOR, INTEGRATOR, OP5, VECTOR,
    DIM, DIM> RP5;

int
main(int argc, char **argv)
{
  /**
   * In this example we solve the one dimensional heat equation.
   * It shows how DopE handels 1d equations.
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

  //First, declare the parameters and read them in.
  ParameterReader pr;
  RP1::declare_params(pr);
  RP2::declare_params(pr);
  RP3::declare_params(pr);
  RP4::declare_params(pr);
  RP5::declare_params(pr);
  RNA1::declare_params(pr);
  pr.read_parameters(paramfile);

  std::string cases = "solve";

  //Create the triangulation.
  Triangulation<DIM> triangulation;
  GridGenerator::hyper_cube(triangulation, 0., 1.);

  //Define the Finite Elements and quadrature formulas for control and state.
  FESystem<DIM> control_fe(FE_Nothing<DIM>(), 1);
  FESystem<DIM> state_fe(FE_Q<DIM>(1), 1);

  QGauss<DIM> quadrature_formula(3);
  QGauss<DIM - 1> face_quadrature_formula(3);
  IDC idc(quadrature_formula, face_quadrature_formula);

  //Define the localPDE and the functionals we are interested in. Here, LFunc is a dummy necessary for the control,
  //LPF is a SpaceTimePointevaluation
  LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM> LPDE;
  LocalFunctional<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM> LFunc;
  LocalPointFunctional<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM> LPF;

  //Time grid of [0,1]
  Triangulation<1> times;
  GridGenerator::subdivided_hyper_cube(times, 50);

  triangulation.refine_global(4);
  MethodOfLines_SpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR, DIM,
      DIM> DOFH(triangulation, control_fe, state_fe, times,
      DOpEtypes::ControlType::stationary);

  NoConstraints<ElementDataContainer, FaceDataContainer, DOFHANDLER, VECTOR, DIM,
      DIM> Constraints;
  OP1 P1(LFunc, LPDE, Constraints, DOFH);
  OP2 P2(LFunc, LPDE, Constraints, DOFH);
  OP3 P3(LFunc, LPDE, Constraints, DOFH);
  OP4 P4(LFunc, LPDE, Constraints, DOFH);
  OP5 P5(LFunc, LPDE, Constraints, DOFH);

  P1.AddFunctional(&LPF);
  P2.AddFunctional(&LPF);
  P3.AddFunctional(&LPF);
  P4.AddFunctional(&LPF);
  P5.AddFunctional(&LPF);

  std::vector<bool> comp_mask(1);
  comp_mask[0] = true;

  //Here we use zero boundary values
  DOpEWrapper::ZeroFunction<DIM> zf;
  SimpleDirichletData<VECTOR, DIM> DD1(zf);

  P1.SetDirichletBoundaryColors(0, comp_mask, &DD1);
  P1.SetDirichletBoundaryColors(1, comp_mask, &DD1);
  P2.SetDirichletBoundaryColors(0, comp_mask, &DD1);
  P2.SetDirichletBoundaryColors(1, comp_mask, &DD1);
  P3.SetDirichletBoundaryColors(0, comp_mask, &DD1);
  P3.SetDirichletBoundaryColors(1, comp_mask, &DD1);
  P4.SetDirichletBoundaryColors(0, comp_mask, &DD1);
  P4.SetDirichletBoundaryColors(1, comp_mask, &DD1);
  P5.SetDirichletBoundaryColors(0, comp_mask, &DD1);
  P5.SetDirichletBoundaryColors(1, comp_mask, &DD1);

  //prepare the initial data
  InitialData initial_data;
  P1.SetInitialValues(&initial_data);
  P2.SetInitialValues(&initial_data);
  P3.SetInitialValues(&initial_data);
  P4.SetInitialValues(&initial_data);
  P5.SetInitialValues(&initial_data);

  RP1 solver1(&P1, DOpEtypes::VectorStorageType::fullmem, pr, idc);
  RP2 solver2(&P2, DOpEtypes::VectorStorageType::fullmem, pr, idc);
  RP3 solver3(&P3, DOpEtypes::VectorStorageType::fullmem, pr, idc);
  RP4 solver4(&P4, DOpEtypes::VectorStorageType::fullmem, pr, idc);
  RP5 solver5(&P5, DOpEtypes::VectorStorageType::fullmem, pr, idc);

  //Use one outputhandler for all problems
  DOpEOutputHandler<VECTOR> output(&solver1, pr);
  DOpEExceptionHandler<VECTOR> ex(&output);

  RNA1 Alg1(&P1, &solver1, pr, &ex, &output);
  RNA2 Alg2(&P2, &solver2, pr, &ex, &output);
  RNA3 Alg3(&P3, &solver3, pr, &ex, &output);
  RNA4 Alg4(&P4, &solver4, pr, &ex, &output);
  RNA5 Alg5(&P5, &solver5, pr, &ex, &output);

  try
  {

    Alg1.ReInit();
    Alg2.ReInit();
    Alg3.ReInit();
    Alg4.ReInit();
    Alg5.ReInit();

    Vector<double> solution;
    ControlVector<VECTOR> q(&DOFH, DOpEtypes::VectorStorageType::fullmem);

    Alg1.SolveForward(q);
    Alg2.SolveForward(q);
    Alg3.SolveForward(q);
    Alg4.SolveForward(q);
    Alg5.SolveForward(q);

    // The soluction extractor class allows us
    // to get the solution vector from the solver
    SolutionExtractor<RP1, VECTOR> a1(solver1);
    const StateVector<VECTOR> &statevec1 = a1.GetU();
    SolutionExtractor<RP2, VECTOR> a2(solver2);
    const StateVector<VECTOR> &statevec2 = a2.GetU();
    SolutionExtractor<RP3, VECTOR> a3(solver3);
    const StateVector<VECTOR> &statevec3 = a3.GetU();
    SolutionExtractor<RP4, VECTOR> a4(solver4);
    const StateVector<VECTOR> &statevec4 = a4.GetU();
    SolutionExtractor<RP5, VECTOR> a5(solver5);
    const StateVector<VECTOR> &statevec5 = a5.GetU();

    stringstream out;
    double product1 = statevec1 * statevec1;
    double product2 = statevec2 * statevec2;
    double product3 = statevec3 * statevec3;
    double product4 = statevec4 * statevec4;
    double product5 = statevec5 * statevec5;
    out << "Forward euler: u * u = " << product1 << std::endl;
    out << "Backward euler: u * u = " << product2 << std::endl;
    out << "CN: u * u = " << product3 << std::endl;
    out << "ShiftedCN: u * u = " << product4 << std::endl;
    out << "FractionalStepTheta: u * u = " << product5 << std::endl;
    output.Write(out, 0);

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

#undef FDC
#undef CDC
#undef FE
#undef DOFHANDLER


