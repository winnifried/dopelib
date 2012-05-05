#include "reducednewtonalgorithm.h"
#include "instatoptproblemcontainer.h"
#include "forward_euler_problem.h"
#include "backward_euler_problem.h"
#include "crank_nicolson_problem.h"
#include "shifted_crank_nicolson_problem.h"
#include "fractional_step_theta_problem.h"
#include "functionalinterface.h"
#include "pdeinterface.h"
#include "instatreducedproblem.h"
#include "instat_step_newtonsolver.h"
#include "fractional_step_theta_step_newtonsolver.h"
#include "newtonsolvermixeddims.h"
#include "gmreslinearsolver.h"
#include "cglinearsolver.h"
#include "directlinearsolver.h"
#include "voidlinearsolver.h"
#include "integrator.h"
#include "integratormixeddims.h"
#include "parameterreader.h"
#include "mol_spacetimehandler.h"
#include "simpledirichletdata.h"
#include "noconstraints.h"
#include "sparsitymaker.h"
#include "userdefineddofconstraints.h"
#include "integratordatacontainer.h"

#include <iostream>
#include <fstream>

#include <base/quadrature_lib.h>
#include <base/function.h>

#include <grid/tria.h>
#include <grid/grid_in.h>
#include <grid/grid_generator.h>
#include <grid/tria_boundary_lib.h>

#include <dofs/dof_handler.h>
#include <dofs/dof_tools.h>

#include <fe/fe_q.h>

#include "localpde.h"
#include "localfunctional.h"
#include "functionals.h"

#include "my_functions.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

// Define dimensions for control- and state problem
#define LOCALDOPEDIM 2
#define LOCALDEALDIM 2
#define VECTOR BlockVector<double>
#define SPARSITYPATTERN BlockSparsityPattern
#define DOFHANDLER DoFHandler<LOCALDEALDIM>
#define FE FESystem<2>
#define FUNC FunctionalInterface<CellDataContainer,FaceDataContainer,DOFHANDLER,VECTOR,LOCALDOPEDIM,LOCALDEALDIM>
#define PDE PDEInterface<CellDataContainer,FaceDataContainer,DOFHANDLER,VECTOR,LOCALDOPEDIM,LOCALDEALDIM>
#define DD DirichletDataInterface<VECTOR,LOCALDOPEDIM,LOCALDEALDIM>
#define CONS ConstraintInterface<CellDataContainer,FaceDataContainer,DOFHANDLER,VECTOR,LOCALDOPEDIM,LOCALDEALDIM>

typedef OptProblemContainer<FUNC, FUNC, PDE, DD, CONS, SPARSITYPATTERN, VECTOR,
    LOCALDOPEDIM, LOCALDEALDIM> OP_BASE;

#define PROB StateProblem<OP_BASE,PDE,DD,SPARSITYPATTERN,VECTOR,LOCALDOPEDIM,LOCALDEALDIM>

// Typedefs for timestep problem
#define TSP ShiftedCrankNicolsonProblem
//FIXME: This should be a reasonable dual timestepping scheme
#define DTSP ShiftedCrankNicolsonProblem

typedef InstatOptProblemContainer<TSP, DTSP, FUNC, FUNC, PDE, DD, CONS,
    SPARSITYPATTERN, VECTOR, LOCALDOPEDIM, LOCALDEALDIM> OP;

#undef TSP
#undef DTSP

typedef IntegratorDataContainer<DOFHANDLER, Quadrature<LOCALDEALDIM>,
    Quadrature<LOCALDEALDIM - 1>, VECTOR, LOCALDEALDIM> IDC;

typedef Integrator<IDC, VECTOR, double, LOCALDEALDIM> INTEGRATOR;

typedef DirectLinearSolverWithMatrix<BlockSparsityPattern,
    BlockSparseMatrix<double>, BlockVector<double>, LOCALDEALDIM> LINEARSOLVER;

typedef InstatStepNewtonSolver<INTEGRATOR, LINEARSOLVER, BlockVector<double>,
    LOCALDEALDIM> NLS;

typedef ReducedNewtonAlgorithm<OP, VECTOR, LOCALDOPEDIM, LOCALDEALDIM> RNA;
typedef InstatReducedProblem<NLS, NLS, INTEGRATOR, INTEGRATOR, OP,
    VECTOR, LOCALDOPEDIM, LOCALDEALDIM> SSolver;

int
main(int argc, char **argv)
{
  /**
   *  As extension to Example 3 we are going to solve
   *  instationary Stokes' equations. We use the well-known
   *  Taylor-Hood element
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
  RNA::declare_params(pr);
  LocalPDE<VECTOR, LOCALDOPEDIM, LOCALDEALDIM>::declare_params(pr);
  BoundaryParabel::declare_params(pr);
  LocalBoundaryFunctionalDrag<VECTOR, LOCALDOPEDIM, LOCALDEALDIM>::declare_params(
      pr);
  LocalBoundaryFunctionalLift<VECTOR, LOCALDOPEDIM, LOCALDEALDIM>::declare_params(
      pr);
  pr.read_parameters(paramfile);

  std::string cases = "solve";

  Triangulation<LOCALDEALDIM> triangulation;

  GridIn<LOCALDEALDIM> grid_in;
  grid_in.attach_triangulation(triangulation);
  //std::ifstream input_file("channel.inp");
  std::ifstream input_file("nsbench4_original.inp");
  grid_in.read_ucd(input_file);

  Point<LOCALDEALDIM> p(0.2, 0.2);
  double radius = 0.05;
  static const HyperBallBoundary<LOCALDEALDIM> boundary(p, radius);
  triangulation.set_boundary(80, boundary);

  FESystem<LOCALDEALDIM> control_fe(FE_Q<LOCALDOPEDIM>(1), 1); //Q1 geht auch P0?
  FESystem<LOCALDEALDIM> state_fe(FE_Q<LOCALDEALDIM>(2), 2,
      FE_Q<LOCALDEALDIM>(1), 1); //Q1

  QGauss<LOCALDEALDIM> quadrature_formula(3);
  QGauss<LOCALDEALDIM - 1> face_quadrature_formula(3);
  IDC idc(quadrature_formula, face_quadrature_formula);

  LocalPDE<VECTOR, LOCALDOPEDIM, LOCALDEALDIM> LPDE(pr);
  LocalFunctional<VECTOR, LOCALDOPEDIM, LOCALDEALDIM> LFunc;

  LocalPointFunctionalPressure<VECTOR, LOCALDOPEDIM, LOCALDEALDIM> LPFP;
  LocalBoundaryFunctionalDrag<VECTOR, LOCALDOPEDIM, LOCALDEALDIM> LBFD(pr);
  LocalBoundaryFunctionalLift<VECTOR, LOCALDOPEDIM, LOCALDEALDIM> LBFL(pr);

  //Time grid of [0,8]
  Triangulation<1> times;
  GridGenerator::subdivided_hyper_cube(times, 80, 0, 8);

  triangulation.refine_global(2);
  MethodOfLines_SpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR,
      LOCALDOPEDIM, LOCALDEALDIM> DOFH(triangulation, control_fe, state_fe,
      times);

  NoConstraints<CellDataContainer, FaceDataContainer, DOFHANDLER, VECTOR,
      LOCALDOPEDIM, LOCALDEALDIM> Constraints;
  OP P(LFunc, LPDE, Constraints, DOFH);

  P.AddFunctional(&LPFP);
  P.AddFunctional(&LBFD);
  P.AddFunctional(&LBFL);

  // fuer Drag und Lift Auswertung am Zylinder
  P.SetBoundaryFunctionalColors(80);

  std::vector<bool> comp_mask(3);

  comp_mask[0] = true;
  comp_mask[1] = true;
  comp_mask[2] = false;

  DOpEWrapper::ZeroFunction<LOCALDEALDIM> zf(3);
  SimpleDirichletData<VECTOR, LOCALDOPEDIM, LOCALDEALDIM> DD1(zf);

  BoundaryParabel boundary_parabel(pr);
  SimpleDirichletData<VECTOR, LOCALDOPEDIM, LOCALDEALDIM> DD2(boundary_parabel);
  P.SetDirichletBoundaryColors(0, comp_mask, &DD2);
  P.SetDirichletBoundaryColors(2, comp_mask, &DD1);
  P.SetDirichletBoundaryColors(80, comp_mask, &DD1);

  P.SetBoundaryEquationColors(1);

  BoundaryParabelExact boundary_parabel_ex;
  P.SetInitialValues(&zf);
  //P.SetInitialValues(&boundary_parabel_ex);

  SSolver solver(&P, "fullmem", pr, idc);
  RNA Alg(&P, &solver, pr);

  try
    {
      Alg.ReInit();

      ControlVector<VECTOR> q(&DOFH, "fullmem");
      if (cases == "check")
        {
          ControlVector<VECTOR> dq(q);
          Alg.CheckGrads(1., q, dq, 2);
          Alg.CheckHessian(1., q, dq, 2);
        }
      else
        {
          Alg.SolveForward(q);
        }
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

#undef LOCALDOPEDIM
#undef LOCALDEALDIM
#undef VECTOR
#undef SPARSITYPATTERN
