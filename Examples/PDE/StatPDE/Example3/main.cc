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
#include "solutionextractor.h"
#include "celldatacontainer.h"
#include "facedatacontainer.h"
#include "sparsitymaker.h"
#include "userdefineddofconstraints.h"
#include "preconditioner_wrapper.h"
#include "integratordatacontainer.h"

#include <iostream>
#include <fstream>

#include <grid/tria.h>
#include <grid/grid_in.h>
#include <grid/tria_boundary_lib.h>
#include <dofs/dof_handler.h>
#include <grid/grid_generator.h>
#include <fe/fe_q.h>
#include <fe/fe_dgp.h>
#include <dofs/dof_tools.h>
#include <base/quadrature_lib.h>
#include <base/function.h>

// for grid local refinement
#include <numerics/error_estimator.h>
#include <grid/grid_refinement.h>

#include "localpde.h"
#include "functionals.h"

#include "my_functions.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

#define VECTOR BlockVector<double>
#define DOFHANDLER DoFHandler<2>
#define FE FESystem<2>

typedef PDEProblemContainer<LocalPDE<DOFHANDLER, VECTOR, 2>,
    DirichletDataInterface<VECTOR, 2>, BlockSparsityPattern, VECTOR, 2> OP;

typedef IntegratorDataContainer<DOFHANDLER, Quadrature<2>,
    Quadrature<1>, VECTOR, 2> IDC;
typedef Integrator<IDC, VECTOR, double, 2> INTEGRATOR;

typedef DirectLinearSolverWithMatrix<BlockSparsityPattern,
    BlockSparseMatrix<double>, VECTOR, 2> LINEARSOLVER;

typedef NewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR, 2> NLS;
typedef StatPDEProblem<NLS, INTEGRATOR, OP, VECTOR, 2> SSolver;

int
main(int argc, char **argv)
{
  /**
   * Stationary FSI problem in an ALE framework
   * Fluid: Stokes equ.
   * Structure: Incompressible INH model and compressible STVK material
   * We use the Q2^c-P1^dc element for discretization.
   * Computation of PointFunctionals, i.e. pressure, x-and y-deflections
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
  SSolver::declare_params(pr);
  DOpEOutputHandler<VECTOR>::declare_params(pr);
  LocalPDE<DOFHANDLER, VECTOR, 2>::declare_params(pr);
  BoundaryParabel::declare_params(pr);
  LocalBoundaryFaceFunctionalDrag<DOFHANDLER, VECTOR, 2>::declare_params(pr);
  LocalBoundaryFaceFunctionalLift<DOFHANDLER, VECTOR, 2>::declare_params(pr);
  pr.read_parameters(paramfile);

  Triangulation<2> triangulation;

  GridIn<2> grid_in;
  grid_in.attach_triangulation(triangulation);
  // Grid for "normal" fluid Benchmark
  //std::ifstream input_file("nsbench4_original.inp");

  // Grid for Benchmark with flag (FSI)
  std::ifstream input_file("bench_fs_t0100_tw.inp");

  grid_in.read_ucd(input_file);

  Point<2> p(0.2, 0.2);
  double radius = 0.05;
  static const HyperBallBoundary<2> boundary(p, radius);
  triangulation.set_boundary(80, boundary);
  triangulation.set_boundary(81, boundary);

  FESystem<2> state_fe(FE_Q<2>(2), 2,
  //FE_DGP<2>(1),1,
      FE_Q<2>(1), 1, FE_Q<2>(2), 2);

  QGauss<2> quadrature_formula(3);
  QGauss<1> face_quadrature_formula(3);
  IDC idc(quadrature_formula, face_quadrature_formula);

  LocalPDE<DOFHANDLER, VECTOR, 2> LPDE(pr);

  LocalPointFunctionalPressure<DOFHANDLER, VECTOR, 2> LPFP;
  LocalPointFunctionalDeflectionX<DOFHANDLER, VECTOR, 2> LPFDX;
  LocalPointFunctionalDeflectionY<DOFHANDLER, VECTOR, 2> LPFDY;
  LocalBoundaryFaceFunctionalDrag<DOFHANDLER, VECTOR, 2> LBFD(pr);
  LocalBoundaryFaceFunctionalLift<DOFHANDLER, VECTOR, 2> LBFL(pr);

  //pseudo time
  std::vector<double> times(1, 0.);
  triangulation.refine_global(1);

  MethodOfLines_StateSpaceTimeHandler<FE, DOFHANDLER, BlockSparsityPattern,
      VECTOR, 2> DOFH(triangulation, state_fe);

  OP P(LPDE, DOFH);

  P.AddFunctional(&LPFP);
  P.AddFunctional(&LPFDX);
  P.AddFunctional(&LPFDY);
  P.AddFunctional(&LBFD);
  P.AddFunctional(&LBFL);

  // fuer Drag und Lift Auswertung am Zylinder
  P.SetBoundaryFunctionalColors(80);
  P.SetBoundaryFunctionalColors(81);

  std::vector<bool> comp_mask(5);

  comp_mask[0] = true;
  comp_mask[1] = true;
  comp_mask[2] = false;
  comp_mask[3] = true;
  comp_mask[4] = true;

  DOpEWrapper::ZeroFunction<2> zf(5);
  SimpleDirichletData<VECTOR, 2> DD1(zf);

  BoundaryParabel boundary_parabel(pr);
  SimpleDirichletData<VECTOR, 2> DD2(boundary_parabel);
  P.SetDirichletBoundaryColors(0, comp_mask, &DD2);
  P.SetDirichletBoundaryColors(2, comp_mask, &DD1);
  //P.SetDirichletBoundaryColors(3,comp_mask,&DD1);
  P.SetDirichletBoundaryColors(80, comp_mask, &DD1);
  P.SetDirichletBoundaryColors(81, comp_mask, &DD1);

  P.SetBoundaryEquationColors(1);

  SSolver solver(&P, "fullmem", pr, idc);
  //Only needed for pure PDE Problems
  DOpEOutputHandler<VECTOR> out(&solver, pr);
  DOpEExceptionHandler<VECTOR> ex(&out);
  P.RegisterOutputHandler(&out);
  P.RegisterExceptionHandler(&ex);
  solver.RegisterOutputHandler(&out);
  solver.RegisterExceptionHandler(&ex);

  // Mesh-refinement cycles
  int niter = 3;

  Vector<double> solution;
  solver.ReInit();
  out.ReInit();

  for (int i = 0; i < niter; i++)
    {
      try
        {
          stringstream outp;

          outp << "**************************************************\n";
          outp << "*             Starting Forward Solve             *\n";
          outp << "*   Solving : " << P.GetName() << "\t*\n";
          outp << "*   SDoFs   : ";
          solver.StateSizeInfo(outp);
          outp << "**************************************************";
          out.Write(outp, 1, 1, 1);

          solver.ComputeReducedFunctionals();
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
          SolutionExtractor<SSolver, VECTOR > a(solver);
          const StateVector<VECTOR > &gu = a.GetU();
          solution = 0;
          solution = gu.GetSpacialVector();
          Vector<float> estimated_error_per_cell(
              triangulation.n_active_cells());

          std::vector<bool> component_mask(5, false);
          component_mask[2] = true;

          KellyErrorEstimator<2>::estimate(
              static_cast<const DoFHandler<2>&>(DOFH.GetStateDoFHandler()),
              QGauss<1>(2), FunctionMap<2>::type(), solution,
              estimated_error_per_cell, component_mask);
          DOFH.RefineSpace("fixednumber", &estimated_error_per_cell, 0.3, 0.0);
          solver.ReInit();
          out.ReInit();
        }
    }

  return 0;
}
