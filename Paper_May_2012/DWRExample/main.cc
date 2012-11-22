//C++
#include <iostream>
#include <fstream>
#include <numeric>

//DOpE
#include "pdeproblemcontainer.h"
#include "functionalinterface.h"
#include "pdeinterface.h"
#include "statpdeproblem.h"
#include "newtonsolver.h"
#include "directlinearsolver.h"
#include "userdefineddofconstraints.h"
#include "sparsitymaker.h"
#include "integratordatacontainer.h"
#include "integrator.h"
#include "parameterreader.h"
#include "mol_statespacetimehandler.h"
#include "simpledirichletdata.h"
#include "active_fe_index_setter_interface.h"

//deal.ii
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/numerics/vectors.h>

//local
#include "localpde.h"
#include "functionals.h"
#include "higher_order_dwrc.h"
#include "residualestimator.h"
#include "myfunctions.h"
#include "mapping_wrapper.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

//some defines and typedefs for cleaner code
#define DIM 2
#define VECTOR BlockVector<double>
#define MATRIX BlockSparseMatrix<double>
#define SPARSITYPATTERN BlockSparsityPattern
#define DOFHANDLER DoFHandler<DIM>
#define FE FESystem<DIM>
#define FACEDATACONTAINER FaceDataContainer<DOFHANDLER, VECTOR, DIM>

typedef PDEProblemContainer<
    PDEInterface<CellDataContainer, FaceDataContainer, DOFHANDLER, VECTOR, DIM>
    ,
    DirichletDataInterface<VECTOR, DIM> ,SPARSITYPATTERN,VECTOR, DIM> OP;
typedef IntegratorDataContainer<DOFHANDLER, Quadrature<DIM>,
    Quadrature<DIM - 1>, VECTOR, DIM> IDC;
typedef Integrator<IDC, VECTOR, double, DIM> INTEGRATOR;
//********************Linearsolver**********************************
typedef DirectLinearSolverWithMatrix<SPARSITYPATTERN, MATRIX, VECTOR, DIM> LINEARSOLVER;
//********************Linearsolver**********************************
typedef NewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR, DIM> NLS;
//typedef MyStatPDEProblem<NLS, INTEGRATOR, OP, 2> SSolver;
typedef StatPDEProblem<NLS, INTEGRATOR, OP, VECTOR, DIM> SSolver;
typedef MethodOfLines_StateSpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN,
    VECTOR, DIM> STH;
typedef CellDataContainer<DOFHANDLER, VECTOR, DIM> CDC;
typedef FaceDataContainer<DOFHANDLER, VECTOR, DIM> FDC;
typedef HigherOrderDWRContainer<STH, IDC, CDC, FDC, VECTOR> HO_DWRC;
typedef L2ResidualErrorContainer<STH, VECTOR, DIM> L2_RESC;
typedef H1ResidualErrorContainer<STH, VECTOR, DIM> H1_RESC;

void
declare_params(ParameterReader &param_reader)
{
  param_reader.SetSubsection("main parameters");
  param_reader.declare_entry("refinement", "global",
      Patterns::Selection("global|adaptive"), "How do we refine?");
  param_reader.declare_entry("estimator", "DWR",
      Patterns::Selection("DWR|H1|L2"), "Which error estimator should we use?");
  param_reader.declare_entry("max dofs", "1000000", Patterns::Integer(1),
      "Which error estimator should we use?");
  param_reader.declare_entry("max_iter", "1", Patterns::Integer(0),
      "How many iterations?");
  param_reader.declare_entry("quad order", "3", Patterns::Integer(1),
      "Order of the quad formula?");
  param_reader.declare_entry("facequad order", "3", Patterns::Integer(1),
      "Order of the face quad formula?");
  param_reader.declare_entry("prerefine", "1", Patterns::Integer(1),
      "How often should we refine the coarse grid?");

  param_reader.declare_entry("order fe v", "2", Patterns::Integer(0),
      "Order of the finite element for the velocity");
  param_reader.declare_entry("order fe p", "1", Patterns::Integer(0),
      "Order of the finite element for the pressure");
  param_reader.declare_entry("compute error", "true", Patterns::Bool(),
      "Shall we estimate the error?");
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
  ParameterReader pr;
  ConvergenceTable convergence_table;

  SSolver::declare_params(pr);
//  LocalPDEStokes<VECTOR, DIM>::declare_params(pr);
  LocalPDENStokes<VECTOR, DIM>::declare_params(pr);
//  LocalPDENStokesStab<VECTOR, DIM>::declare_params(pr);

  BoundaryParabel::declare_params(pr);
  LocalBoundaryFunctionalDrag<VECTOR, DIM>::declare_params(pr);
  LocalBoundaryFunctionalLift<VECTOR, DIM>::declare_params(pr);
  DOpEOutputHandler<VECTOR>::declare_params(pr);
  declare_params(pr);

  pr.read_parameters(paramfile);

  //************************************************
  //define some constants
  pr.SetSubsection("main parameters");
  const int max_iter = pr.get_integer("max_iter");
  const int prerefine = pr.get_integer("prerefine");
  const std::string ref_type = pr.get_string("refinement");
  const std::string ee_type = pr.get_string("estimator");
  const unsigned int max_n_dofs = pr.get_integer("max dofs");
  const unsigned int fe_order_v = pr.get_integer("order fe v");
  const unsigned int fe_order_p = pr.get_integer("order fe p");

//Werte von Nabh G., On higher order methods for the stationary incompressible Navier-Stokes equations. PhD thesis, Heidelberg, 1998

  //ns
  const double exact_delta_p = 0.11752016697;
  const double exact_cd = 5.57953523384;
  const double exact_cl = 0.010618948146;

  //stokes
//  const double exact_delta_p = 0.04557938237;
//  const double exact_cd = 3.142426688/*3.142425296*/;
//  const double exact_cl = 0.03019601524;

  const bool compute_error = pr.get_bool("compute error");

  //*************************************************

  //Make triangulation *************************************************
  Triangulation<DIM> triangulation(
      Triangulation<DIM>::MeshSmoothing::patch_level_1);
  GridIn<2> grid_in;
  grid_in.attach_triangulation(triangulation);
  std::ifstream input_file("nsbench4_original.inp");
  grid_in.read_ucd(input_file);

  Point<DIM> p(0.2, 0.2);
  double radius = 0.05;
  static const HyperBallBoundary<DIM> boundary(p, radius);
  triangulation.set_boundary(80, boundary);

  if (prerefine > 0)
    triangulation.refine_global(prerefine);
//  for (unsigned int i = 1; i < 1; i++)
//  {
//    for (auto it = triangulation.begin_active(); it != triangulation.end();
//        it++)
//        {
//      if (it->at_boundary())
//      {
//        for (unsigned int j = 0; j < 4; j++)
//          if (it->face(j)->boundary_indicator() == 80)
//            it->set_refine_flag();
//      }
//    }
//    triangulation.execute_coarsening_and_refinement();
//  }
  //*************************************************************

  //FiniteElemente*************************************************
  pr.SetSubsection("main parameters");
  FESystem<DIM> state_fe(FE_Q<DIM>(fe_order_v), DIM, FE_Q<DIM>(fe_order_p), 1);

  //Quadrature formulas*************************************************
  pr.SetSubsection("main parameters");
  QGauss<DIM> quadrature_formula(pr.get_integer("quad order"));
  QGauss<DIM - 1> face_quadrature_formula(pr.get_integer("facequad order"));
  IDC idc(quadrature_formula, face_quadrature_formula);
  //**************************************************************************

  //Functionals*************************************************
  LocalPointFunctionalPressure<VECTOR, DIM> LPFP;
  LocalBoundaryFunctionalDrag<VECTOR, DIM> LBFD(pr);
  LocalBoundaryFunctionalLift<VECTOR, DIM> LBFL(pr);
  LocalCellFunctionalDrag<VECTOR, DIM> LCFD(pr);

//  LocalPDEStokes<VECTOR, DIM> LPDE(pr, false);
  LocalPDENStokes<VECTOR, DIM> LPDE(pr);
//  LocalPDENStokesStab<VECTOR, DIM> LPDE(pr);
  //*************************************************

  //space time handler***********************************/
  DOpEWrapper::Mapping<2, DOFHANDLER> mapping(2);
  STH DOFH(triangulation, mapping, state_fe);
  /***********************************/

  OP P(LPDE, DOFH);
  P.AddFunctional(&LPFP);
  P.AddFunctional(&LBFD);
  P.AddFunctional(&LBFL);
  P.AddFunctional(&LCFD);
  //Boundary conditions************************************************
  // fuer Drag und Lift Auswertung am Zylinder
  P.SetBoundaryFunctionalColors(80);

  std::vector<bool> comp_mask(DIM + 1, true);
  comp_mask[0] = true;
  comp_mask[1] = true;
  comp_mask[2] = false;

  DOpEWrapper::ZeroFunction<DIM> zf(3);
  SimpleDirichletData<VECTOR, DIM> DD1(zf);

  BoundaryParabel boundary_parabel(pr);
  SimpleDirichletData<VECTOR, DIM> DD2(boundary_parabel);

  P.SetDirichletBoundaryColors(0, comp_mask, &DD2);
//  P.SetDirichletBoundaryColors(0, comp_mask, &DD1);
  P.SetDirichletBoundaryColors(2, comp_mask, &DD1);
  P.SetDirichletBoundaryColors(80, comp_mask, &DD1);

//  P.SetBoundaryEquationColors(80);

  /************************************************/
  SSolver solver(&P, "fullmem", pr, idc);

  //Only needed for pure PDE Problems
  DOpEOutputHandler<VECTOR> out(&solver, pr);
  DOpEExceptionHandler<VECTOR> ex(&out);
  P.RegisterOutputHandler(&out);
  P.RegisterExceptionHandler(&ex);
  solver.RegisterOutputHandler(&out);
  solver.RegisterExceptionHandler(&ex);
  /**********************************************************************/
  //DWR**********************************************************************/
  //Set dual functional for ee
  P.SetFunctionalForErrorEstimation(LBFD.GetName());
//  P.SetFunctionalForErrorEstimation(LCFD.GetName());
  //FiniteElemente for DWR*************************************************
  pr.SetSubsection("main parameters");
  FESystem<DIM> state_fe_high(FE_Q<DIM>(2 * fe_order_v), DIM,
      FE_Q<DIM>(2 * fe_order_p), 1);
  //Quadrature formulas for DWR*************************************************
  pr.SetSubsection("main parameters");
  QGauss<DIM> quadrature_formula_high(pr.get_integer("quad order") + 2);
  QGauss<1> face_quadrature_formula_high(pr.get_integer("facequad order") + 2);
  IDC idc_high(quadrature_formula, face_quadrature_formula);
  STH DOFH_higher_order(triangulation, mapping, state_fe_high);

//  HO_DWRC dwrc(DOFH_higher_order, idc_high, "fullmem", pr, DOpEtypes::mixed);
  HO_DWRC dwrc(DOFH_higher_order, idc_high, "fullmem", pr, DOpEtypes::mixed);
//  HO_DWRC dwrc(DOFH_higher_order, idc_high, "fullmem", pr, DOpEtypes::dual_only);
  L2_RESC l2resc(DOFH, "fullmem", pr, DOpEtypes::primal_only);
  H1_RESC h1resc(DOFH, "fullmem", pr, DOpEtypes::primal_only);

  solver.InitializeDWRC(dwrc);
  //**************************************************************************************************
  int i = 0;
  unsigned int n_dofs = 0;
  while (i < max_iter && n_dofs < max_n_dofs)
  {
    try
    {
      solver.ReInit();
      out.ReInit();
      stringstream outp;

      outp << "**************************************************\n";
      outp << "*             Starting Forward Solve             *\n";
      outp << "*   Solving : " << P.GetName() << "\t*\n";
      outp << "*   SDoFs   : ";
      solver.StateSizeInfo(outp);
      outp << "**************************************************";
      out.Write(outp, 1, 1, 1);

      std::vector<unsigned int> dofs_per_block = DOFH.GetStateDoFsPerBlock();
      LCFD.PreparePhiD(DOFH, dofs_per_block, mapping);
      solver.AddUserDomainData("phid", LCFD.GetPhiD());
      solver.ComputeReducedFunctionals();

      if (ee_type == "DWR")
      {
        solver.ComputeRefinementIndicators(dwrc, LPDE);
        solver.DeleteUserDomainData("phid");
      }
      else if (ee_type == "L2")
        solver.ComputeRefinementIndicators(l2resc, LPDE);
      else if (ee_type == "H1")
        solver.ComputeRefinementIndicators(h1resc, LPDE);

////
//      double errord = exact_value - solver.GetFunctionalValue( LBFD.GetName());
      const double error_cd = exact_cd
          - solver.GetFunctionalValue(LBFD.GetName());
//          -solver.GetFunctionalValue(LCFD.GetName());
      const double error_cl = exact_cl
          - solver.GetFunctionalValue(LBFL.GetName());
      const double error_delta_p = exact_delta_p
          - solver.GetFunctionalValue(LPFP.GetName());
      const double error_cd_domain = exact_cd
          - solver.GetFunctionalValue(LCFD.GetName());

      double est_error = 0;
      if (compute_error)
      {
        if (ee_type == "DWR")
          est_error = dwrc.GetError();
        else if (ee_type == "L2")
          est_error = sqrt(l2resc.GetError());
        else if (ee_type == "H1")
          est_error = sqrt(h1resc.GetError());
      }

      outp << "Error Drag Domain: " << error_cd_domain <<" Error Drag: "
          << error_cd << "  Est.Error Drag: " << est_error << "  Ieff: "
          << est_error / error_cd << "  Error Lift: " << error_cl
          << "  Error Delta p: " << error_delta_p << std::endl;

//      outp << "Mean value error: " << error << "  Ieff (eh/e)= "
//          << dwrc.GetError() / error << std::endl;

      // outp << "L2-Error estimator: " << sqrt(l2resc.GetError()) << std::endl;
      //  outp << "H1-Error estimator: " << sqrt(h1resc.GetError()) << std::endl;
      out.Write(outp, 1, 1, 1);

      /*******************************************************************/
      n_dofs = DOFH.GetStateNDoFs();
      convergence_table.add_value("n-dofs", n_dofs);
      convergence_table.add_value("v-dofs", dofs_per_block[0]);
      convergence_table.add_value("p-dofs", dofs_per_block[1]);

      convergence_table.add_value("Drag",
          solver.GetFunctionalValue(LBFD.GetName()));
      convergence_table.add_value("Drag-error", std::fabs(error_cd));

      convergence_table.add_value("DragD",
          solver.GetFunctionalValue(LCFD.GetName()));
      convergence_table.add_value("DragD-error", std::fabs(error_cd_domain));

      convergence_table.add_value("Lift",
          solver.GetFunctionalValue(LBFL.GetName()));
      convergence_table.add_value("Lift-error", std::fabs(error_cl));
      convergence_table.add_value("Delta p",
          solver.GetFunctionalValue(LPFP.GetName()));
      convergence_table.add_value("Delta p-error", std::fabs(error_delta_p));
    }
    catch (DOpEException &e)
    {
      std::cout
          << "Warning: During execution of `" + e.GetThrowingInstance()
              + "` the following Problem occurred!" << std::endl;
      std::cout << e.GetErrorMessage() << std::endl;
    }

    if (compute_error)
    {
      Vector<float> error_ind_p(dwrc.GetPrimalErrorIndicators());
      Vector<float> error_ind_d(dwrc.GetDualErrorIndicators());
      float primal = std::accumulate(error_ind_p.begin(), error_ind_p.end(),
          0.);
      float dual = std::accumulate(error_ind_d.begin(), error_ind_d.end(), 0.);
      std::cout << "Primal gives: " << primal << " Dual gives: " << dual
          << std::endl;
    }

    if (i != max_iter - 1)
    {
      if (ref_type == "global")
      {
        DOFH.RefineSpace("global");
      }
      else if (ref_type == "adaptive")
      {
        Vector<float> error_ind(
            DOFH.GetStateDoFHandler().get_tria().n_active_cells());

        if (ee_type == "DWR")
        {
          error_ind = dwrc.GetErrorIndicators();
          for (unsigned int i = 0; i < error_ind.size(); i++)
            error_ind(i) = std::fabs(error_ind(i));
        }
        else if (ee_type == "L2")
        {
          error_ind = l2resc.GetErrorIndicators();
          for (unsigned int i = 0; i < error_ind.size(); i++)
            error_ind(i) = sqrt(error_ind(i));
        }
        else if (ee_type == "H1")
        {
          error_ind = h1resc.GetErrorIndicators();
          for (unsigned int i = 0; i < error_ind.size(); i++)
          {
            error_ind(i) = sqrt(error_ind(i));
          }
        }
        DOFH.RefineSpace("optimized", &error_ind);
//        DOFH.RefineSpace("fixedfraction", &error_ind, 0.8);
//        DOFH.RefineSpace("fixednumber", &error_ind, 0.4);
      }
//      DOFH.RefineSpace("fixednumber", &error_ind, 0.4);
//      DOFH.RefineSpace("fixedfraction", &error_ind, 0.8);
    }
    i++;
  }
  convergence_table.set_scientific("Lift-error", true);
  convergence_table.set_scientific("Drag-error", true);
  convergence_table.set_scientific("DragD-error", true);
  convergence_table.set_scientific("Delta p-error", true);

  convergence_table.evaluate_convergence_rates("Lift-error", "n-dofs",
      ConvergenceTable::reduction_rate_log2);
  convergence_table.evaluate_convergence_rates("Drag-error", "n-dofs",
      ConvergenceTable::reduction_rate_log2);
  convergence_table.evaluate_convergence_rates("DragD-error", "n-dofs",
      ConvergenceTable::reduction_rate_log2);
  convergence_table.evaluate_convergence_rates("Delta p-error", "n-dofs",
      ConvergenceTable::reduction_rate_log2);
  stringstream outp;
  convergence_table.write_text(outp);
  out.Write(outp, 1, 1, 1);

  return 0;
}
