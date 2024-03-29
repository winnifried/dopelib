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
#include <fstream>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_in.h>
#if DEAL_II_VERSION_GTE(9,1,1)
#else
#include <deal.II/grid/tria_boundary_lib.h>
#endif
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/grid/manifold_lib.h>

#include <container/pdeproblemcontainer.h>
#include <interfaces/functionalinterface.h>
#include <interfaces/pdeinterface.h>
#include <reducedproblems/statpdeproblem.h>
#include <templates/directlinearsolver.h>
#include <include/userdefineddofconstraints.h>
#include <include/sparsitymaker.h>
#include <container/integratordatacontainer.h>

#include <templates/newtonsolver.h>
#include <templates/integrator.h>
#include <include/parameterreader.h>

#include <basic/mol_statespacetimehandler.h>
#include <problemdata/simpledirichletdata.h>
#include <interfaces/active_fe_index_setter_interface.h>

#include "localpde.h"
#include "functionals.h"
#include "functions.h"
#include "obstacleestimator.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

const static int DIM = 2;

#if DEAL_II_VERSION_GTE(9,3,0)
#define DOFHANDLER false
#else
#define DOFHANDLER DoFHandler
#endif

#define FE FESystem
#define EDC ElementDataContainer
#define FDC FaceDataContainer

typedef QGaussLobatto<DIM> QUADRATURE;
typedef QGaussLobatto<DIM - 1> FACEQUADRATURE;
typedef SparseMatrix<double> MATRIX;
typedef SparsityPattern SPARSITYPATTERN;
typedef Vector<double> VECTOR;

typedef PDEProblemContainer<LocalPDE<EDC, FDC, DOFHANDLER, VECTOR, DIM>,
        SimpleDirichletData<VECTOR, DIM>, SPARSITYPATTERN, VECTOR, DIM> OP;
typedef IntegratorDataContainer<DOFHANDLER, QUADRATURE, FACEQUADRATURE,
        VECTOR, DIM> IDC;
typedef Integrator<IDC, VECTOR, double, DIM> INTEGRATOR;
typedef DirectLinearSolverWithMatrix<SPARSITYPATTERN, MATRIX, VECTOR> LINEARSOLVER;

typedef NewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR> NLS;
typedef StatPDEProblem<NLS, INTEGRATOR, OP, VECTOR, DIM> RP;
typedef MethodOfLines_StateSpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN,
        VECTOR, DIM> STH;
typedef ObstacleResidualErrorContainer<STH, VECTOR, DIM> OBSTACLE_RESC;

void
declare_params(ParameterReader &param_reader)
{
  param_reader.SetSubsection("main parameters");
  param_reader.declare_entry("max_iter", "1", Patterns::Integer(0),
                             "How many iterations?");
  param_reader.declare_entry("post_ref", "1", Patterns::Integer(0),
                             "How many global refinements for reference?");
  param_reader.declare_entry("quad order", "2", Patterns::Integer(1),
                             "Order of the quad formula?");
  param_reader.declare_entry("facequad order", "2", Patterns::Integer(1),
                             "Order of the face quad formula?");
  param_reader.declare_entry("order fe", "2", Patterns::Integer(1),
                             "Order of the finite element?");
  param_reader.declare_entry("prerefine", "1", Patterns::Integer(0),
                             "How often should we refine the coarse grid?");
}

int
main(int argc, char **argv)
{
  /**
   * We solve an obstacle method by indtroducing a complementarity function
   * for the Lagrange multiplier
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
  DOpEOutputHandler<VECTOR>::declare_params(pr);
  declare_params(pr);

  pr.read_parameters(paramfile);

  //************************************************
  //define some constants
  pr.SetSubsection("main parameters");
  int max_iter = pr.get_integer("max_iter");
  int post_ref = pr.get_integer("post_ref");
  int prerefine = pr.get_integer("prerefine");
  //*************************************************

  //Make triangulation *************************************************
  Triangulation<DIM> triangulation;
  GridGenerator::hyper_cube(triangulation,-1,1);
  triangulation.refine_global(prerefine);
  //*************************************************************

  //FiniteElemente*************************************************
  pr.SetSubsection("main parameters");
  FE<DIM> state_fe(FE_Q<DIM>(pr.get_integer("order fe")), 2);
  //Component 0 = phasen-Feld
  //Component 1 = lagrangemult.

  //Quadrature formulas*************************************************
  pr.SetSubsection("main parameters");
  QGaussLobatto<DIM> quadrature_formula(pr.get_integer("quad order"));
  QGaussLobatto<1> face_quadrature_formula(pr.get_integer("facequad order"));
  IDC idc(quadrature_formula, face_quadrature_formula);
  //**************************************************************************

  //Functionals*************************************************
  LocalFaceFunctional<EDC, FDC, DOFHANDLER, VECTOR, DIM> LFF;
  LocalPDE<EDC, FDC, DOFHANDLER, VECTOR, DIM> LPDE;
  //*************************************************

  //space time handler***********************************/
  STH DOFH(triangulation, state_fe);
  /***********************************/

  OP P(LPDE, DOFH);
  P.AddFunctional(&LFF);
  //Boundary conditions************************************************
  std::vector<bool> comp_mask(2,true);
  comp_mask[1] = false;
  DOpEWrapper::ZeroFunction<DIM> zf(2);
  SimpleDirichletData<VECTOR, DIM> DD1(zf);
  //Set dirichlet boundary values all around
  P.SetDirichletBoundaryColors(0, comp_mask, &DD1);
  /************************************************/
  RP solver(&P, DOpEtypes::VectorStorageType::fullmem, pr, idc);

  //Only needed for pure PDE Problems
  DOpEOutputHandler<VECTOR> out(&solver, pr);
  DOpEExceptionHandler<VECTOR> ex(&out);
  P.RegisterOutputHandler(&out);
  P.RegisterExceptionHandler(&ex);
  solver.RegisterOutputHandler(&out);
  solver.RegisterExceptionHandler(&ex);
  /**********************************************************************/
  OBSTACLE_RESC resc(DOFH, DOpEtypes::VectorStorageType::fullmem, pr, DOpEtypes::primal_only);

  solver.ReInit();
  out.ReInit();

  //Generating Storage for the reference values
  vector<StateVector<VECTOR>*> store(max_iter,NULL);
  for (int i = 0; i < max_iter; i++)
    {
      store[i] = new StateVector<VECTOR>(&DOFH, DOpEtypes::VectorStorageType::fullmem,pr);
      (*store[i]) = 0.;
    }
  vector<double> h1errs(max_iter);
  vector<unsigned int> dofs(max_iter);

  for (int i = 0; i < max_iter; i++)
    {
      try
        {
          stringstream outp;

          //Initialize Obstacle
          StateVector<VECTOR> obstacle(&DOFH,DOpEtypes::VectorStorageType::fullmem,pr);
          local::Obstacle exact_obstacle;
          VectorTools::interpolate(DOFH.GetStateDoFHandler().GetDEALDoFHandler(),exact_obstacle,obstacle.GetSpacialVector());
          P.AddAuxiliaryState(&obstacle,"obstacle");
          //Done with Obstacle

          outp << "**************************************************\n";
          outp << "*             Starting Forward Solve             *\n";
          outp << "*   Solving : " << P.GetName() << "\t*\n";
          outp << "*   SDoFs   : ";
          solver.StateSizeInfo(outp);
          outp << "**************************************************";
          out.Write(outp, 1, 1, 1);

          solver.ComputeReducedFunctionals();
          solver.ComputeRefinementIndicators(resc, LPDE);
          P.DeleteAuxiliaryState("obstacle");
          outp << "Obstacle-Error estimator: " << sqrt(resc.GetError()) << std::endl;
          out.Write(outp, 1, 1, 1);

          //Store solution
          h1errs[i] = sqrt(resc.GetError());
          dofs[i] =  DOFH.GetStateNDoFs();
          SolutionExtractor<RP,VECTOR >  a(solver);
          store[i]->GetSpacialVector() = a.GetU().GetSpacialVector();
        }
      catch (DOpEException &e)
        {
          std::cout
              << "Warning: During execution of `" + e.GetThrowingInstance()
              + "` the following Problem occurred!" << std::endl;
          std::cout << e.GetErrorMessage() << std::endl;
          P.DeleteAuxiliaryState("obstacle");
        }
      if (i != max_iter - 1)
        {
          //For global mesh refinement, uncomment the next line
          DOFH.RefineSpace(RefineOptimized(resc.GetErrorIndicators()[0]));

          solver.ReInit();
          out.ReInit();
          for (int j = 0; j < max_iter; j++)
            {
              store[j]->ReInit();
            }
        }
    }
  //Postrefinement to get the reference solution
  for (int i = 0; i< post_ref; i++)
    {
      DOFH.RefineSpace(DOpEtypes::RefinementType::global);
      solver.ReInit();
      out.ReInit();
      for (int j = 0; j < max_iter; j++)
        {
          store[j]->ReInit();
        }
    }
  {
    //Calculate reference
    stringstream outp;

    //Initialize Obstacle
    StateVector<VECTOR> obstacle(&DOFH,DOpEtypes::VectorStorageType::fullmem,pr);
    local::Obstacle exact_obstacle;
    VectorTools::interpolate(DOFH.GetStateDoFHandler().GetDEALDoFHandler(),exact_obstacle,obstacle.GetSpacialVector());
    P.AddAuxiliaryState(&obstacle,"obstacle");
    //Done with Obstacle

    outp << "**************************************************\n";
    outp << "*             Starting Forward Solve             *\n";
    outp << "*             for Reference Solution             *\n";
    outp << "*   Solving : " << P.GetName() << "\t*\n";
    outp << "*   SDoFs   : ";
    solver.StateSizeInfo(outp);
    outp << "**************************************************";
    out.Write(outp, 1, 1, 1);

    solver.ComputeReducedFunctionals();
    P.DeleteAuxiliaryState("obstacle");
    SolutionExtractor<RP, VECTOR > SE(solver);

    //Calculate error to reference
    for (int i = 0; i < max_iter; i++)
      {
#if DEAL_II_VERSION_GTE(8,4,0)
        Vector<float> difference_per_element (DOFH.GetStateDoFHandler().get_triangulation().n_active_cells());
#else
        Vector<float> difference_per_element (DOFH.GetStateDoFHandler().get_tria().n_active_cells());
#endif
        //First component (of two) for error calculation
        ComponentSelectFunction<2> select_comp(0,2);
        //Prepare the error
        store[i]->GetSpacialVector() -=SE.GetU().GetSpacialVector();
        VectorTools::integrate_difference( DOFH.GetStateDoFHandler(),
                                           store[i]->GetSpacialVector(),
#if DEAL_II_VERSION_GTE(9,3,0)
                                           Functions::ZeroFunction<2>(2),
#else
                                           ZeroFunction<2>(2),
#endif
                                           difference_per_element,
                                           QGaussLobatto<2>(5),
                                           VectorTools::H1_norm,
                                           &select_comp);
        double error = difference_per_element.l2_norm();
        outp<<"Level "<<i<<" DoFs "<<dofs[i]<<" Estimate: "<<h1errs[i]<<" Error: "<<error<<" Ieff: "<<h1errs[i]/error<<std::endl;
        out.Write(outp,1,1,1);
      }

  }

  //clean storage
  for (int i = 0; i < max_iter; i++)
    {
      delete store[i];
    }

  return 0;
}
#undef FDC
#undef EDC
#undef FE
#undef DOFHANDLER
