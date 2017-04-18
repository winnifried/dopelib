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

#include <container/pdeproblemcontainer.h>
#include <reducedproblems/statpdeproblem.h>
#include <templates/newtonsolver.h>
#include <templates/directlinearsolver.h>
#include <templates/integrator.h>
#include <include/parameterreader.h>
#include <basic/mol_statespacetimehandler.h>
#include <problemdata/simpledirichletdata.h>
#include <include/sparsitymaker.h>
#include <include/userdefineddofconstraints.h>
#include <container/integratordatacontainer.h>
#include <wrapper/mapping_wrapper.h>

#include <iostream>
#include <fstream>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/convergence_table.h>

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
typedef SparseMatrix<double> MATRIX;
typedef SparsityPattern SPARSITYPATTERN;
typedef Vector<double> VECTOR;

typedef PDEProblemContainer<LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
        SimpleDirichletData<VECTOR, DIM>, SPARSITYPATTERN, VECTOR, DIM, FE,
        DOFHANDLER> OP;
typedef IntegratorDataContainer<DOFHANDLER, QUADRATURE, FACEQUADRATURE, VECTOR,
        DIM> IDC;
typedef Integrator<IDC, VECTOR, double, DIM> INTEGRATOR;
typedef DirectLinearSolverWithMatrix<SPARSITYPATTERN, MATRIX, VECTOR> LINEARSOLVER;
typedef NewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR> NLS;
typedef StatPDEProblem<NLS, INTEGRATOR, OP, VECTOR, DIM> RP;
typedef MethodOfLines_StateSpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN,
        VECTOR, DIM> STH;

void
declare_params(ParameterReader &param_reader)
{
  param_reader.SetSubsection("main parameters");
  param_reader.declare_entry("max_iter", "1", Patterns::Integer(0),
                             "How many iterations?");
  param_reader.declare_entry("order fe", "2", Patterns::Integer(1),
                             "Order of the finite element?");
  param_reader.declare_entry("order mapping", "2", Patterns::Integer(1),
                             "Order of the finite element?");
}

int
main(int argc, char **argv)
{
  /**
   *  In this example we do not really solve
   *  a PDE but test the functionality of the higher order mappings.
   *  To this end we approximate PI by the computation of the
   *  circumference of a circle. See the deal.II tutorial step 10.
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
  ConvergenceTable convergence_table;
  RP::declare_params(pr);
  DOpEOutputHandler<VECTOR>::declare_params(pr);
  declare_params(pr);

  pr.read_parameters(paramfile);

  //define some constants
  pr.SetSubsection("main parameters");
  const int max_iter = pr.get_integer("max_iter");
  const int order_fe = pr.get_integer("order fe");
  const int order_mapping = pr.get_integer("order mapping");

  Triangulation<DIM> triangulation;
  Triangulation<DIM> triangulation_q1;

  GridGenerator::hyper_ball(triangulation);
  static const HyperBallBoundary<DIM> boundary;
  triangulation.set_boundary(0, boundary);

  GridGenerator::hyper_ball(triangulation_q1);
  triangulation_q1.set_boundary(0, boundary);

  FE<DIM> state_fe(FE_Q<DIM>(order_fe), 1);

  QUADRATURE quadrature_formula(2);
  FACEQUADRATURE face_quadrature_formula(2);
  IDC idc(quadrature_formula, face_quadrature_formula);

  LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM> LPDE;
  BoundaryFunctional<CDC, FDC, DOFHANDLER, VECTOR, DIM> BF;

  DOpEWrapper::Mapping<DIM, DOFHANDLER > mapping(order_mapping);
  STH DOFH(triangulation, mapping, state_fe);
  STH DOFH_q1(triangulation_q1, state_fe);

  OP P(LPDE, DOFH);
  OP P_q1(LPDE, DOFH_q1);

  P.AddFunctional(&BF);
  P.SetBoundaryFunctionalColors(0);

  P_q1.AddFunctional(&BF);
  P_q1.SetBoundaryFunctionalColors(0);

  std::vector<bool> comp_mask(1);

  comp_mask[0] = true;

  ExactSolution ex_sol;
  DOpEWrapper::ConstantFunction<DIM> cs(1., 1);
  SimpleDirichletData<VECTOR, DIM> DD1(cs);

  P.SetDirichletBoundaryColors(0, comp_mask, &DD1);

  P_q1.SetDirichletBoundaryColors(0, comp_mask, &DD1);

  RP solver(&P, DOpEtypes::VectorStorageType::fullmem, pr, idc);

  RP solver_q1(&P_q1, DOpEtypes::VectorStorageType::fullmem, pr, idc);

  //Only needed for pure PDE Problems
  DOpEOutputHandler<VECTOR> out(&solver, pr);
  DOpEExceptionHandler<VECTOR> ex(&out);

  DOpEOutputHandler<VECTOR> out_q1(&solver_q1, pr);
  DOpEExceptionHandler<VECTOR> ex_q1(&out_q1);

  P.RegisterOutputHandler(&out);
  P.RegisterExceptionHandler(&ex);

  P_q1.RegisterOutputHandler(&out_q1);
  P_q1.RegisterExceptionHandler(&ex_q1);

  solver.RegisterOutputHandler(&out);
  solver.RegisterExceptionHandler(&ex);

  solver_q1.RegisterOutputHandler(&out_q1);
  solver_q1.RegisterExceptionHandler(&ex_q1);

  VECTOR solution;
  auto &dof_handler = DOFH.GetStateDoFHandler().GetDEALDoFHandler();
  auto &dof_handler_q1 = DOFH_q1.GetStateDoFHandler().GetDEALDoFHandler();
  for (int i = 0; i < max_iter; i++)
    {
      //grid output*****************************************
      // if you want to take a look at the resulting
      // grids, uncomment the following section

      /*std::string name = "grid" + dealii::Utilities::int_to_string(i, 2) + ".gpl";
       std::ofstream outs(name.c_str());
       dealii::GridOut grid_out;
       GridOutFlags::Gnuplot gnuplot_flags(false, 30);
       grid_out.set_flags(gnuplot_flags);
       grid_out.write_gnuplot(triangulation, outs, &mapping);*/
      //grid output*****************************************
      try
        {
          solver.ReInit();
          solver_q1.ReInit();
          out.ReInit();
          out_q1.ReInit();
          stringstream outp;

          outp << "**************************************************\n";
          outp << "*             Starting Forward Solve             *\n";
          outp << "*   Solving : " << P.GetName() << "\t*\n";
          outp << "*   SDoFs   : ";
          solver.StateSizeInfo(outp);
          outp << "**************************************************\n";
          outp << "Computing solution and functionals with higher order mapping:";
          out.Write(outp, 1, 1, 1);

          solver.ComputeReducedFunctionals();

          SolutionExtractor<RP, VECTOR> a(solver);
          const StateVector<VECTOR> &gu = a.GetU();

          solution = gu.GetSpacialVector();

          Vector<float> difference_per_element(triangulation.n_active_cells());
          VectorTools::integrate_difference(mapping, dof_handler, solution,
                                            ExactSolution(), difference_per_element, QGauss<DIM>(4),
                                            VectorTools::L2_norm);
          outp << "L2-error: " << difference_per_element.l2_norm() << "\n";
          convergence_table.add_value("n-dofs ||", DOFH.GetStateNDoFs());
          convergence_table.add_value("L2-error ||", difference_per_element.l2_norm());

          /********************/
          outp << "Computing solution and functionals with 1st order mapping:";
          out_q1.Write(outp, 1, 1, 1);

          solver_q1.ComputeReducedFunctionals();

          SolutionExtractor<RP, VECTOR> a_q1(solver_q1);
          const StateVector<VECTOR> &gu_q1 = a_q1.GetU();

          solution = gu_q1.GetSpacialVector();

          VectorTools::integrate_difference(dof_handler_q1, solution,
                                            ExactSolution(), difference_per_element, QGauss<DIM>(4),
                                            VectorTools::L2_norm);
          outp << "L2-error: " << difference_per_element.l2_norm() << "\n";
          convergence_table.add_value("L2-error Q1 ||",
                                      difference_per_element.l2_norm());

          convergence_table.add_value("Error PI by boundary Q1 ||",
                                      solver_q1.GetFunctionalValue(BF.GetName()) - dealii::numbers::PI);
          convergence_table.add_value("Error PI by boundary ||",
                                      solver.GetFunctionalValue(BF.GetName()) - dealii::numbers::PI);

          out.Write(outp, 1, 1, 1);
          out_q1.Write(outp, 1, 1, 1);
        }
      catch (DOpEException &e)
        {
          std::cout
              << "Warning: During execution of `" + e.GetThrowingInstance()
              + "` the following Problem occurred!" << std::endl;
          std::cout << e.GetErrorMessage() << std::endl;
        }
      if (i != max_iter - 1)
        {
          DOFH.RefineSpace();
          DOFH_q1.RefineSpace();
        }
    }
  convergence_table.set_scientific("L2-error ||", true);
  convergence_table.set_precision("L2-error ||", 2);
  convergence_table.evaluate_convergence_rates("L2-error ||", "n-dofs ||",
                                               ConvergenceTable::reduction_rate_log2);

  convergence_table.set_scientific("L2-error Q1 ||", true);
  convergence_table.set_precision("L2-error Q1 ||", 2);
  convergence_table.evaluate_convergence_rates("L2-error Q1 ||", "n-dofs ||",
                                               ConvergenceTable::reduction_rate_log2);

  convergence_table.set_scientific("Error PI by boundary ||", true);
  convergence_table.set_precision("Error PI by boundary ||", 2);
  convergence_table.evaluate_convergence_rates("Error PI by boundary ||",
                                               "n-dofs ||", ConvergenceTable::reduction_rate_log2);

  convergence_table.set_scientific("Error PI by boundary Q1 ||", true);
  convergence_table.set_precision("Error PI by boundary Q1 ||", 2);
  convergence_table.evaluate_convergence_rates("Error PI by boundary Q1 ||",
                                               "n-dofs ||", ConvergenceTable::reduction_rate_log2);
  stringstream outp;
  convergence_table.write_text(outp);
  out.Write(outp, 1, 1, 1);
  return 0;
}
#undef FDC
#undef CDC
#undef FE
#undef DOFHANDLER
