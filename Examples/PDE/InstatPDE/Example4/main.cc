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
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgp.h> //for discont. finite elements
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_generator.h>

//DOpE includes
#include <include/parameterreader.h>
#include <templates/directlinearsolver.h>
#include <templates/integrator.h>
#include <basic/mol_statespacetimehandler.h>
#include <problemdata/simpledirichletdata.h>
#include <container/integratordatacontainer.h>
#include <templates/newtonsolver.h>
#include <templates/fractional_step_theta_step_newtonsolver.h>

#include <reducedproblems/instatpdeproblem.h>
#include <templates/instat_step_newtonsolver.h>
#include <container/instatpdeproblemcontainer.h>

#include <tsschemes/forward_euler_problem.h>
#include <tsschemes/backward_euler_problem.h>
#include <tsschemes/crank_nicolson_problem.h>
#include <tsschemes/shifted_crank_nicolson_problem.h>
#include <tsschemes/fractional_step_theta_problem.h>

//Problem specific includes
#include "localpde.h"
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

typedef PDEProblemContainer<
LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
         SimpleDirichletData<VECTOR, DIM>,
         SPARSITYPATTERN,
         VECTOR, DIM> OP_BASE;

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

typedef InstatPDEProblemContainer<TSP1, DTSP1,
        LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
        SimpleDirichletData<VECTOR, DIM>,
        SPARSITYPATTERN,
        VECTOR, DIM> OP1;
typedef InstatPDEProblemContainer<TSP2, DTSP2,
        LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
        SimpleDirichletData<VECTOR, DIM>,
        SPARSITYPATTERN,
        VECTOR, DIM> OP2;
typedef InstatPDEProblemContainer<TSP3, DTSP3,
        LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
        SimpleDirichletData<VECTOR, DIM>,
        SPARSITYPATTERN,
        VECTOR, DIM> OP3;
typedef InstatPDEProblemContainer<TSP4, DTSP4,
        LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
        SimpleDirichletData<VECTOR, DIM>,
        SPARSITYPATTERN,
        VECTOR, DIM> OP4;
typedef InstatPDEProblemContainer<TSP5, DTSP5,
        LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
        SimpleDirichletData<VECTOR, DIM>,
        SPARSITYPATTERN,
        VECTOR, DIM> OP5;

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

typedef InstatPDEProblem<NLS, INTEGRATOR, OP1, VECTOR,
        DIM> RP1;
typedef InstatPDEProblem<NLS, INTEGRATOR, OP2, VECTOR,
        DIM> RP2;
typedef InstatPDEProblem<NLS, INTEGRATOR, OP3, VECTOR,
        DIM> RP3;
typedef InstatPDEProblem<NLS, INTEGRATOR, OP4, VECTOR,
        DIM> RP4;
typedef InstatPDEProblem<NLS2, INTEGRATOR, OP5, VECTOR,
        DIM> RP5;

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
  DOpEOutputHandler<VECTOR>::declare_params(pr);
  pr.read_parameters(paramfile);

  std::string cases = "solve";

  //Create the triangulation.
  Triangulation<DIM> triangulation;
  GridGenerator::hyper_cube(triangulation, 0., 1.);

  //Define the Finite Elements and quadrature formulas for the state.
  FESystem<DIM> state_fe(FE_Q<DIM>(1), 1);

  QGauss<DIM> quadrature_formula(3);
  QGauss<DIM - 1> face_quadrature_formula(3);
  IDC idc(quadrature_formula, face_quadrature_formula);

  //Define the localPDE and the functionals we are interested in. Here, LFunc is a dummy necessary for the control,
  //LPF is a SpaceTimePointevaluation
  LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM> LPDE;
  LocalPointFunctional<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM> LPF;

  //Time grid of [0,1]
  Triangulation<1> times;
  GridGenerator::subdivided_hyper_cube(times, 50);

  triangulation.refine_global(4);
  MethodOfLines_StateSpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR,
                                      DIM> DOFH(triangulation, state_fe, times);

  OP1 P1(LPDE, DOFH);
  OP2 P2(LPDE, DOFH);
  OP3 P3(LPDE, DOFH);
  OP4 P4(LPDE, DOFH);
  OP5 P5(LPDE, DOFH);

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

  P1.RegisterOutputHandler(&output);
  P1.RegisterExceptionHandler(&ex);
  P2.RegisterOutputHandler(&output);
  P2.RegisterExceptionHandler(&ex);
  P3.RegisterOutputHandler(&output);
  P3.RegisterExceptionHandler(&ex);
  P4.RegisterOutputHandler(&output);
  P4.RegisterExceptionHandler(&ex);
  P5.RegisterOutputHandler(&output);
  P5.RegisterExceptionHandler(&ex);
  solver1.RegisterOutputHandler(&output);
  solver1.RegisterExceptionHandler(&ex);
  solver2.RegisterOutputHandler(&output);
  solver2.RegisterExceptionHandler(&ex);
  solver3.RegisterOutputHandler(&output);
  solver3.RegisterExceptionHandler(&ex);
  solver4.RegisterOutputHandler(&output);
  solver4.RegisterExceptionHandler(&ex);
  solver5.RegisterOutputHandler(&output);
  solver5.RegisterExceptionHandler(&ex);


  try
    {

      solver1.ReInit();
      solver2.ReInit();
      solver3.ReInit();
      solver4.ReInit();
      solver5.ReInit();
      output.ReInit();

      stringstream outp;
      {
        outp << "**************************************************\n";
        outp << "*             Starting Forward Solve             *\n";
        outp << "*   Solving : " << P1.GetName() << "\t*\n";
        outp << "*   SDoFs   : ";
        solver1.StateSizeInfo(outp);
        outp << "**************************************************";
        //We print this header with priority 1 and 1 empty line in front and after.
        output.Write(outp, 1, 1, 1);

        //We compute the value of the functionals. To this end, we have to solve
        //the PDE at hand.
        solver1.ComputeReducedFunctionals();
      }
      {
        outp << "**************************************************\n";
        outp << "*             Starting Forward Solve             *\n";
        outp << "*   Solving : " << P2.GetName() << "\t*\n";
        outp << "*   SDoFs   : ";
        solver2.StateSizeInfo(outp);
        outp << "**************************************************";
        //We print this header with priority 1 and 1 empty line in front and after.
        output.Write(outp, 1, 1, 1);

        //We compute the value of the functionals. To this end, we have to solve
        //the PDE at hand.
        solver2.ComputeReducedFunctionals();
      }
      {
        outp << "**************************************************\n";
        outp << "*             Starting Forward Solve             *\n";
        outp << "*   Solving : " << P3.GetName() << "\t*\n";
        outp << "*   SDoFs   : ";
        solver3.StateSizeInfo(outp);
        outp << "**************************************************";
        //We print this header with priority 1 and 1 empty line in front and after.
        output.Write(outp, 1, 1, 1);

        //We compute the value of the functionals. To this end, we have to solve
        //the PDE at hand.
        solver3.ComputeReducedFunctionals();
      }
      {
        outp << "**************************************************\n";
        outp << "*             Starting Forward Solve             *\n";
        outp << "*   Solving : " << P5.GetName() << "\t*\n";
        outp << "*   SDoFs   : ";
        solver4.StateSizeInfo(outp);
        outp << "**************************************************";
        //We print this header with priority 1 and 1 empty line in front and after.
        output.Write(outp, 1, 1, 1);

        //We compute the value of the functionals. To this end, we have to solve
        //the PDE at hand.
        solver4.ComputeReducedFunctionals();
      }
      {
        outp << "**************************************************\n";
        outp << "*             Starting Forward Solve             *\n";
        outp << "*   Solving : " << P5.GetName() << "\t*\n";
        outp << "*   SDoFs   : ";
        solver5.StateSizeInfo(outp);
        outp << "**************************************************";
        //We print this header with priority 1 and 1 empty line in front and after.
        output.Write(outp, 1, 1, 1);

        //We compute the value of the functionals. To this end, we have to solve
        //the PDE at hand.
        solver5.ComputeReducedFunctionals();
      }

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

      double product1 = statevec1 * statevec1;
      double product2 = statevec2 * statevec2;
      double product3 = statevec3 * statevec3;
      double product4 = statevec4 * statevec4;
      double product5 = statevec5 * statevec5;
      outp << "Forward euler: u * u = " << product1 << std::endl;
      outp << "Backward euler: u * u = " << product2 << std::endl;
      outp << "CN: u * u = " << product3 << std::endl;
      outp << "ShiftedCN: u * u = " << product4 << std::endl;
      outp << "FractionalStepTheta: u * u = " << product5 << std::endl;
      output.Write(outp, 0);

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


