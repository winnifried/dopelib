#include "pdeproblemcontainer.h"
#include "functionalinterface.h"
#include "pdeinterface.h"
#include "statpdeproblem.h"
#include "newtonsolver.h"
#include "directlinearsolver.h"
#include "integrator.h"
#include "parameterreader.h"
#include "mol_statespacetimehandler.h"
#include "simpledirichletdata.h"
#include "celldatacontainer.h"
#include "facedatacontainer.h"
#include "sparsitymaker.h"
#include "userdefineddofconstraints.h"
#include "integratordatacontainer.h"
#include "mapping_wrapper.h"

#include <iostream>
#include <fstream>

#include <grid/tria.h>
#include <grid/grid_in.h>
#include <dofs/dof_handler.h>
#include <grid/grid_generator.h>
#include <grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_out.h>
#include <fe/fe_q.h>
#include <dofs/dof_tools.h>
#include <base/quadrature_lib.h>
#include <base/function.h>
#include <numerics/vector_tools.h>
#include <deal.II/base/convergence_table.h>

#include "localpde.h"
#include "functionals.h"
#include "my_functions.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

#define MATRIX SparseMatrix<double>
#define SPARSITYPATTERN SparsityPattern
#define VECTOR Vector<double>
#define DOFHANDLER DoFHandler<2>
#define FE FESystem<2>
#define FACEDATACONTAINER FaceDataContainer<DOFHANDLER, VECTOR, 2>

typedef PDEProblemContainer<LocalPDE<VECTOR, 2>,
    DirichletDataInterface<VECTOR, 2>, SPARSITYPATTERN, VECTOR, 2> OP;
typedef IntegratorDataContainer<DOFHANDLER, Quadrature<2>, Quadrature<1>,
    VECTOR, 2> IDC;
typedef Integrator<IDC, VECTOR, double, 2> INTEGRATOR;
typedef DirectLinearSolverWithMatrix<SPARSITYPATTERN, MATRIX, VECTOR, 2> LINEARSOLVER;
typedef NewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR, 2> NLS;
typedef StatPDEProblem<NLS, INTEGRATOR, OP, VECTOR, 2> SSolver;

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
   *  circumference of a circle. See the deal.II tutorail, step 10.
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
  SSolver::declare_params(pr);
  DOpEOutputHandler<VECTOR>::declare_params(pr);
  declare_params(pr);

  pr.read_parameters(paramfile);

  //define some constants
  pr.SetSubsection("main parameters");
  const int max_iter = pr.get_integer("max_iter");
  const int order_fe = pr.get_integer("order fe");
  const int order_mapping = pr.get_integer("order mapping");

  Triangulation<2> triangulation;
  Triangulation<2> triangulation_q1;

  GridGenerator::hyper_ball(triangulation);
  static const HyperBallBoundary<2> boundary;
  triangulation.set_boundary(0, boundary);

  GridGenerator::hyper_ball(triangulation_q1);
  triangulation_q1.set_boundary(0, boundary);

  FESystem<2> state_fe(FE_Q<2>(order_fe), 1);

  QGauss<2> quadrature_formula(2);
  QGauss<1> face_quadrature_formula(2);
  IDC idc(quadrature_formula, face_quadrature_formula);

  LocalPDE<VECTOR, 2> LPDE(order_fe);
  BoundaryFunctional<VECTOR, FACEDATACONTAINER, 2> BF;

  DOpEWrapper::Mapping<2, dealii::DoFHandler<2> > mapping(order_mapping);
  MethodOfLines_StateSpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR, 2> DOFH(
      triangulation, mapping, state_fe);
  MethodOfLines_StateSpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR, 2> DOFH_q1(
      triangulation_q1, state_fe);

  OP P(LPDE, DOFH);
  OP P_q1(LPDE, DOFH_q1);

  P.AddFunctional(&BF);
  P.SetBoundaryFunctionalColors(0);

  P_q1.AddFunctional(&BF);
  P_q1.SetBoundaryFunctionalColors(0);

  std::vector<bool> comp_mask(1);

  comp_mask[0] = true;

  ExactSolution ex_sol(order_fe);
  DOpEWrapper::ConstantFunction<2> cs(1., 1);
  SimpleDirichletData<VECTOR, 2> DD1(cs);

  P.SetDirichletBoundaryColors(0, comp_mask, &DD1);

  P_q1.SetDirichletBoundaryColors(0, comp_mask, &DD1);

  SSolver solver(&P, "fullmem", pr, idc);

  SSolver solver_q1(&P_q1, "fullmem", pr, idc);

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
  auto& dof_handler = DOFH.GetStateDoFHandler().GetDEALDoFHandler();
  auto& dof_handler_q1 = DOFH_q1.GetStateDoFHandler().GetDEALDoFHandler();
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

      SolutionExtractor<SSolver, VECTOR > a(solver);
      const StateVector<VECTOR > &gu = a.GetU();

      solution = gu.GetSpacialVector();

      Vector<float> difference_per_cell(triangulation.n_active_cells());
      VectorTools::integrate_difference(dof_handler, solution,
          ExactSolution(order_fe), difference_per_cell, QGauss<2>(4),
          VectorTools::L2_norm);
      outp << "L2-error: " << difference_per_cell.l2_norm() << "\n";
      convergence_table.add_value("n-dofs ||", DOFH.GetStateNDoFs());
      convergence_table.add_value("L2-error ||", difference_per_cell.l2_norm());

      /********************/
      outp << "Computing solution and functionals with 1st order mapping:";
      out_q1.Write(outp, 1, 1, 1);

      solver_q1.ComputeReducedFunctionals();

      SolutionExtractor<SSolver, VECTOR > a_q1(solver_q1);
      const StateVector<VECTOR > &gu_q1 = a_q1.GetU();

      solution = gu_q1.GetSpacialVector();

      VectorTools::integrate_difference(dof_handler_q1, solution,
          ExactSolution(order_fe), difference_per_cell, QGauss<2>(4),
          VectorTools::L2_norm);
      outp << "L2-error: " << difference_per_cell.l2_norm()<< "\n";
      convergence_table.add_value("L2-error Q1 ||", difference_per_cell.l2_norm());

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
      DOFH.RefineSpace("global");
      DOFH_q1.RefineSpace("global");
    }
  }
  convergence_table.set_scientific("L2-error ||", true);
  convergence_table.set_precision("L2-error ||", 2);
  convergence_table.evaluate_convergence_rates("L2-error ||", "n-dofs ||",
      ConvergenceTable::reduction_rate_log2);

  convergence_table.set_scientific("L2-error Q1", true);
  convergence_table.set_precision("L2-error Q1", 2);
  convergence_table.evaluate_convergence_rates("L2-error Q1 ||", "n-dofs ||",
      ConvergenceTable::reduction_rate_log2);

  convergence_table.set_scientific("Error PI by boundary ||", true);
  convergence_table.set_precision("Error PI by boundary ||", 2);
  convergence_table.evaluate_convergence_rates("Error PI by boundary ||", "n-dofs ||",
      ConvergenceTable::reduction_rate_log2);

  convergence_table.set_scientific("Error PI by boundary Q1 ||", true);
  convergence_table.set_precision("Error PI by boundary Q1 ||", 2);
  convergence_table.evaluate_convergence_rates("Error PI by boundary Q1 ||", "n-dofs ||",
      ConvergenceTable::reduction_rate_log2);
  stringstream outp;
  convergence_table.write_text(outp);
  out.Write(outp, 1, 1, 1);
  return 0;
}
