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

#include <reducedproblems/instatpdeproblem.h>
#include <templates/instat_step_newtonsolver.h>
#include <container/instatpdeproblemcontainer.h>

#include <tsschemes/shifted_crank_nicolson_problem.h>

//Problem specific includes
#include "localpde.h"
#include "functionals.h"
#include "my_functions.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

const static int DIM = 2;

#define DOFHANDLER DoFHandler
#define FE FESystem
#define CDC ElementDataContainer
#define FDC FaceDataContainer

typedef QGauss<DIM> QUADRATURE;
typedef QGauss<DIM - 1> FACEQUADRATURE;
typedef BlockSparseMatrix<double> MATRIX;
typedef BlockSparsityPattern SPARSITYPATTERN;
typedef BlockVector<double> VECTOR;

typedef PDEProblemContainer<
LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
         SimpleDirichletData<VECTOR, DIM>,
         SPARSITYPATTERN,
         VECTOR, DIM> OP_BASE;

typedef StateProblem<OP_BASE, LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
        SimpleDirichletData<VECTOR, DIM>, SPARSITYPATTERN, VECTOR, DIM> PROB;

// Typedefs for timestep problem
#define TSP ShiftedCrankNicolsonProblem
//FIXME: This should be a reasonable dual timestepping scheme
#define DTSP ShiftedCrankNicolsonProblem
typedef InstatPDEProblemContainer<TSP, DTSP,
        LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
        SimpleDirichletData<VECTOR, DIM>,
        SPARSITYPATTERN,
        VECTOR, DIM> OP;
#undef TSP
#undef DTSP

typedef IntegratorDataContainer<DOFHANDLER, QUADRATURE,
        FACEQUADRATURE, VECTOR, DIM> IDC;
typedef Integrator<IDC, VECTOR, double, DIM> INTEGRATOR;
typedef DirectLinearSolverWithMatrix<SPARSITYPATTERN, MATRIX, VECTOR> LINEARSOLVER;
typedef NewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR> CNLS;
typedef InstatStepNewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR> NLS;
typedef InstatPDEProblem<NLS, INTEGRATOR, OP, VECTOR,
        DIM> RP;

/**
 * In this example we solve the two dimensional Black-Scholes equation.
 * As the initial conditions are only H1 regular, we use the shifted
 * Crank Nicolson time stepping.
 */

void
ColorizeTriangulation(Triangulation<2> &coarse_grid, double upper_bound)
{
  /**
   * Colorize the spatial triangulation, i.e. set the correct boundary colors.
   */
  Triangulation<2>::cell_iterator element = coarse_grid.begin();
  Triangulation<2>::cell_iterator endc = coarse_grid.end();
  for (; element != endc; ++element)
    for (unsigned int face = 0; face < GeometryInfo<2>::faces_per_cell; ++face)
      {
        if (std::fabs(element->face(face)->center()(1) - (0)) < 1e-12)
          {
#if DEAL_II_VERSION_GTE(8,3,0)
            element->face(face)->set_boundary_id(1);
#else
            element->face(face)->set_boundary_indicator(1);
#endif
          }
        else if (std::fabs(element->face(face)->center()(0) - (upper_bound)) < 1e-12)
          {
#if DEAL_II_VERSION_GTE(8,3,0)
            element->face(face)->set_boundary_id(0);
#else
            element->face(face)->set_boundary_indicator(0);
#endif
          }
        else if (std::fabs(element->face(face)->center()(1) - (upper_bound)) < 1e-12)
          {
#if DEAL_II_VERSION_GTE(8,3,0)
            element->face(face)->set_boundary_id(0);
#else
            element->face(face)->set_boundary_indicator(0);
#endif
          }
        else if (std::fabs(element->face(face)->center()(0) - (0)) < 1e-12)
          {
#if DEAL_II_VERSION_GTE(8,3,0)
            element->face(face)->set_boundary_id(1);
#else
            element->face(face)->set_boundary_indicator(1);
#endif
          }
      }
}

int
main(int argc, char **argv)
{
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
  RP::declare_params(pr);
  DOpEOutputHandler<VECTOR>::declare_params(pr);
  LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>::declare_params(pr);
  InitialData::declare_params(pr);
  pr.SetSubsection("Discretization parameters");
  pr.declare_entry("upper bound", "0.0", Patterns::Double(0));
  pr.read_parameters(paramfile);

  //Create the triangulation.
  double upper_bound = pr.get_double("upper bound");
  Triangulation<DIM> triangulation;
  GridGenerator::hyper_cube(triangulation, 0., upper_bound);
  ColorizeTriangulation(triangulation, upper_bound);

  //Define the Finite Elements and quadrature formulas for the state.
  FESystem<DIM> state_fe(FE_Q<DIM>(1), 1); //Q1

  QGauss<DIM> quadrature_formula(3);
  QGauss<DIM - 1> face_quadrature_formula(3);
  IDC idc(quadrature_formula, face_quadrature_formula);

  //Define the localPDE and the functionals we are interested in. Here, LFunc is a dummy necessary for the control,
  //LPF is a SpaceTimePointevaluation
  LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, 2> LPDE(pr);
  LocalPointFunctional<CDC, FDC, DOFHANDLER, VECTOR, DIM, DIM> LPF;

  //Time grid of [0,expiration date] with 20 subintervalls.
  Triangulation<1> times;
  pr.SetSubsection("Local PDE parameters");
  GridGenerator::subdivided_hyper_cube(times, 20,0.,pr.get_double("expiration date"));

  triangulation.refine_global(5);
  MethodOfLines_StateSpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR,
                                      DIM> DOFH(triangulation, state_fe, times);

  OP P(LPDE, DOFH);

  P.AddFunctional(&LPF);

  std::vector<bool> comp_mask(1);
  comp_mask[0] = true;

  //Here we use zero boundary values
  DOpEWrapper::ZeroFunction<DIM> zf(1);
  SimpleDirichletData<VECTOR, DIM> DD1(zf);

  P.SetDirichletBoundaryColors(0, comp_mask, &DD1);

  //prepare the initial data
  InitialData initial_data(pr);
  P.SetInitialValues(&initial_data);

  try
    {
      RP solver(&P, DOpEtypes::VectorStorageType::fullmem, pr, idc);
      DOpEOutputHandler<VECTOR> out(&solver, pr);
      DOpEExceptionHandler<VECTOR> ex(&out);
      P.RegisterOutputHandler(&out);
      P.RegisterExceptionHandler(&ex);
      solver.RegisterOutputHandler(&out);
      solver.RegisterExceptionHandler(&ex);

      //Before solving we have to reinitialize the stateproblem and outputhandler.
      solver.ReInit();
      out.ReInit();

      stringstream outp;
      outp << "**************************************************\n";
      outp << "*             Starting Forward Solve             *\n";
      outp << "*   Solving : " << P.GetName() << "\t*\n";
      outp << "*   SDoFs   : ";
      solver.StateSizeInfo(outp);
      outp << "**************************************************";
      //We print this header with priority 1 and 1 empty line in front and after.
      out.Write(outp, 1, 1, 1);

      //We compute the value of the functionals. To this end, we have to solve
      //the PDE at hand.
      solver.ComputeReducedFunctionals();


      SolutionExtractor<RP, VECTOR> a(solver);
      const StateVector<VECTOR> &statevec = a.GetU();


      double product = statevec * statevec;
      outp << " u * u = " << product << std::endl;
      P.GetOutputHandler()->Write(outp, 0);

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

