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

#include <iostream>
#include <fstream>

#include <grid/tria.h>
#include <grid/grid_in.h>
#include <grid/tria_boundary_lib.h>
#include <dofs/dof_handler.h>
#include <grid/grid_generator.h>
#include <fe/fe_q.h>
#include <fe/fe_nothing.h>
#include <dofs/dof_tools.h>
#include <base/quadrature_lib.h>
#include <base/function.h>
#include <deal.II/numerics/vectors.h>

#include "localpde.h"
#include "functionals.h"
#include "higher_order_dwrc.h"
#include "residualestimator.h"
#include "myfunctions.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

#define DIM 2
#define VECTOR Vector<double>
#define MATRIX SparseMatrix<double>
#define SPARSITYPATTERN SparsityPattern
#define DOFHANDLER DoFHandler<DIM>
#define FE FESystem<DIM>
#define FACEDATACONTAINER FaceDataContainer<DOFHANDLER, VECTOR, DIM>

typedef PDEProblemContainer<
    PDEInterface<CellDataContainer, FaceDataContainer, DOFHANDLER, VECTOR, DIM>,
    DirichletDataInterface<VECTOR, DIM>, SPARSITYPATTERN, VECTOR, DIM> OP;
typedef IntegratorDataContainer<DOFHANDLER, Quadrature<DIM>,
    Quadrature<DIM - 1>, VECTOR, DIM> IDC;
typedef Integrator<IDC, VECTOR, double, DIM> INTEGRATOR;
//********************Linearsolver**********************************
typedef DirectLinearSolverWithMatrix<SPARSITYPATTERN, MATRIX, VECTOR, DIM> LINEARSOLVER;
//********************Linearsolver**********************************

typedef NewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR, DIM> NLS;
typedef StatPDEProblem<NLS, INTEGRATOR, OP, VECTOR, DIM> SSolver;
typedef MethodOfLines_StateSpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN,
    VECTOR, DIM> STH;
typedef CellDataContainer<DOFHANDLER, VECTOR, DIM> CDC;
typedef FaceDataContainer<DOFHANDLER, VECTOR, DIM> FDC;
typedef HigherOrderDWRContainer<STH, IDC, CDC, FDC, VECTOR> HO_DWRC;
typedef L2ResidualErrorContainer<STH,  VECTOR, DIM> L2_RESC;
typedef H1ResidualErrorContainer<STH,  VECTOR, DIM> H1_RESC;

void
declare_params(ParameterReader &param_reader)
{
  param_reader.SetSubsection("main parameters");
  param_reader.declare_entry("max_iter", "1", Patterns::Integer(0),
      "How many iterations?");
  param_reader.declare_entry("quad order", "3", Patterns::Integer(1),
      "Order of the quad formula?");
  param_reader.declare_entry("facequad order", "3", Patterns::Integer(1),
      "Order of the face quad formula?");
  param_reader.declare_entry("prerefine", "1", Patterns::Integer(1),
      "How often should we refine the coarse grid?");
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

  SSolver::declare_params(pr);
  LocalPDEStokes<VECTOR, DIM>::declare_params(pr);
  BoundaryParabel::declare_params(pr);
  LocalBoundaryFunctionalDrag<VECTOR, DIM>::declare_params(
      pr);
  LocalBoundaryFunctionalLift<VECTOR, DIM>::declare_params(
      pr);
  DOpEOutputHandler<VECTOR>::declare_params(pr);
  declare_params(pr);

  pr.read_parameters(paramfile);

  //************************************************
  //define some constants
  pr.SetSubsection("main parameters");
  const int max_iter = pr.get_integer("max_iter");
  const int prerefine = pr.get_integer("prerefine");

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
  //*************************************************************

  //FiniteElemente*************************************************
  pr.SetSubsection("main parameters");
  FESystem<DIM> state_fe(FE_Q<DIM>(2), DIM, FE_Q<DIM>(1), 1);

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

  LocalPDEStokes<VECTOR, DIM> LPDE(pr);
  //*************************************************

  //space time handler***********************************/
  STH DOFH(triangulation, state_fe);
  /***********************************/

  OP P(LPDE, DOFH);
  P.AddFunctional(&LPFP);
  P.AddFunctional(&LBFD);
  P.AddFunctional(&LBFL);
  //Boundary conditions************************************************
  // fuer Drag und Lift Auswertung am Zylinder
  P.SetBoundaryFunctionalColors(80);


  std::vector<bool> comp_mask(DIM+1,true);
  comp_mask[0] = true;
  comp_mask[1] = true;
  comp_mask[2] = false;



  DOpEWrapper::ZeroFunction<DIM> zf(3);
  SimpleDirichletData<VECTOR,  DIM> DD1(zf);

  BoundaryParabel boundary_parabel(pr);
  SimpleDirichletData<VECTOR,  DIM> DD2(boundary_parabel);
  P.SetDirichletBoundaryColors(0, comp_mask, &DD2);
  P.SetDirichletBoundaryColors(2, comp_mask, &DD1);
  P.SetDirichletBoundaryColors(80, comp_mask, &DD1);

  P.SetBoundaryEquationColors(80);

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
  P.SetFunctionalForErrorEstimation( LBFD.GetName());
  //FiniteElemente for DWR*************************************************
  pr.SetSubsection("main parameters");
  FESystem<DIM> state_fe_high(FE_Q<DIM>(4), DIM, FE_Q<DIM>(2), 1);
  //Quadrature formulas for DWR*************************************************
  pr.SetSubsection("main parameters");
  QGauss<DIM> quadrature_formula_high(pr.get_integer("quad order") + 1);
  QGauss<1> face_quadrature_formula_high(pr.get_integer("facequad order") + 1);
  IDC idc_high(quadrature_formula, face_quadrature_formula);
  STH DOFH_higher_order(triangulation, state_fe_high);
  HO_DWRC dwrc(DOFH_higher_order, idc_high, "fullmem", pr,
      DOpEtypes::mixed);
 // L2_RESC l2resc(DOFH, "fullmem", pr, DOpEtypes::primal_only);
 // H1_RESC h1resc(DOFH, "fullmem", pr, DOpEtypes::primal_only);

  solver.InitializeDWRC(dwrc);
  //**************************************************************************************************

  for (int i = 0; i < max_iter; i++)
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

      solver.ComputeReducedFunctionals();
      solver.ComputeRefinementIndicators(dwrc, LPDE);

     // solver.ComputeRefinementIndicators(l2resc, LPDE);
    //  solver.ComputeRefinementIndicators(h1resc, LPDE);

//      double exact_value = 5.5755;
////
//      double error = exact_value - solver.GetFunctionalValue( LBFD.GetName());
//      outp << "Mean value error: " << error << "  Ieff (eh/e)= "
//          << dwrc.GetError() / error << std::endl;


     // outp << "L2-Error estimator: " << sqrt(l2resc.GetError()) << std::endl;
    //  outp << "H1-Error estimator: " << sqrt(h1resc.GetError()) << std::endl;
      out.Write(outp, 1, 1, 1);
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
//      Vector<float> error_ind(dwrc.GetErrorIndicators());
//      for (unsigned int i = 0; i < error_ind.size(); i++)
//        error_ind(i) = std::fabs(error_ind(i));
//      DOFH.RefineSpace("optimized", &error_ind);
//      DOFH.RefineSpace("fixednumber", &error_ind, 0.4);
//      DOFH.RefineSpace("fixedfraction", &error_ind, 0.8);
    }
  }
  return 0;
}
