#include "reducednewtonalgorithm.h"
#include "instatoptproblemcontainer.h"
#include "forward_euler_problem.h"
#include "backward_euler_problem.h"
#include "crank_nicolson_problem.h"
#include "shifted_crank_nicolson_problem.h"
#include "fractional_step_theta_problem.h"
#include "functionalinterface.h"
#include "pdeinterface.h"
#include "forwardtimestepreducedproblem.h"
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
#include "solutionextractor.h"
#include "sparsitymaker.h"
#include "integratordatacontainer.h"
#include "constraintsmaker.h"

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
#define LOCALDOPEDIM 1
#define LOCALDEALDIM 1


#define VECTOR dealii::BlockVector<double>
#define SPARSITYPATTERN dealii::BlockSparsityPattern
#define MATRIX dealii::BlockSparseMatrix<double>
#define DOFHANDLER dealii::DoFHandler<LOCALDOPEDIM>
#define FE DOpEWrapper::FiniteElement<LOCALDEALDIM>
#define FUNC DOpE::FunctionalInterface<CellDataContainer,FaceDataContainer,DOFHANDLER,VECTOR,LOCALDOPEDIM,LOCALDEALDIM>
#define PDE DOpE::PDEInterface<CellDataContainer,FaceDataContainer,DOFHANDLER,VECTOR,LOCALDOPEDIM,LOCALDEALDIM>
#define DD DOpE::DirichletDataInterface<VECTOR,LOCALDOPEDIM,LOCALDEALDIM>
#define CONS DOpE::ConstraintInterface<CellDataContainer,FaceDataContainer,DOFHANDLER,VECTOR,LOCALDOPEDIM,LOCALDEALDIM>

typedef OptProblem<FUNC,FUNC,PDE,DD,CONS,SPARSITYPATTERN, VECTOR, LOCALDOPEDIM,LOCALDEALDIM> OP_BASE;

typedef DOpE::StateProblem<OP_BASE,PDE,DD,SPARSITYPATTERN,VECTOR,LOCALDOPEDIM,LOCALDEALDIM> PROB;
// Typedefs for timestep problem

typedef ForwardEulerProblem<PROB, SPARSITYPATTERN, VECTOR, LOCALDOPEDIM,LOCALDEALDIM> TSP1;
typedef BackwardEulerProblem<PROB, SPARSITYPATTERN, VECTOR, LOCALDOPEDIM,LOCALDEALDIM> TSP2;
typedef CrankNicolsonProblem<PROB, SPARSITYPATTERN, VECTOR, LOCALDOPEDIM,LOCALDEALDIM> TSP3;
typedef ShiftedCrankNicolsonProblem<PROB, SPARSITYPATTERN, VECTOR, LOCALDOPEDIM,LOCALDEALDIM> TSP4;
typedef FractionalStepThetaProblem<PROB, SPARSITYPATTERN, VECTOR, LOCALDOPEDIM,LOCALDEALDIM> TSP5;

typedef InstatOptProblemContainer<TSP1,FUNC,FUNC,PDE,DD,CONS,SPARSITYPATTERN, VECTOR, LOCALDOPEDIM,LOCALDEALDIM> OP1;
typedef InstatOptProblemContainer<TSP2,FUNC,FUNC,PDE,DD,CONS,SPARSITYPATTERN, VECTOR, LOCALDOPEDIM,LOCALDEALDIM> OP2;
typedef InstatOptProblemContainer<TSP3,FUNC,FUNC,PDE,DD,CONS,SPARSITYPATTERN, VECTOR, LOCALDOPEDIM,LOCALDEALDIM> OP3;
typedef InstatOptProblemContainer<TSP4,FUNC,FUNC,PDE,DD,CONS,SPARSITYPATTERN, VECTOR, LOCALDOPEDIM,LOCALDEALDIM> OP4;
typedef InstatOptProblemContainer<TSP5,FUNC,FUNC,PDE,DD,CONS,SPARSITYPATTERN, VECTOR, LOCALDOPEDIM,LOCALDEALDIM> OP5;

typedef IntegratorDataContainer<DOFHANDLER, dealii::Quadrature<LOCALDEALDIM>, dealii::Quadrature<LOCALDEALDIM-1>, VECTOR, LOCALDEALDIM > IDC;

typedef Integrator<IDC , VECTOR , double, LOCALDEALDIM> INTEGRATOR;

typedef DirectLinearSolverWithMatrix<SPARSITYPATTERN, MATRIX , VECTOR,
    LOCALDEALDIM> LINEARSOLVER;

typedef InstatStepNewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR , LOCALDEALDIM>
    NLS;

typedef FractionalStepThetaStepNewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR , LOCALDEALDIM>
    NLS2;

typedef ReducedNewtonAlgorithm<OP1, VECTOR, LOCALDOPEDIM, LOCALDEALDIM> RNA1;
typedef ReducedNewtonAlgorithm<OP2, VECTOR, LOCALDOPEDIM, LOCALDEALDIM> RNA2;
typedef ReducedNewtonAlgorithm<OP3, VECTOR, LOCALDOPEDIM, LOCALDEALDIM> RNA3;
typedef ReducedNewtonAlgorithm<OP4, VECTOR, LOCALDOPEDIM, LOCALDEALDIM> RNA4;
typedef ReducedNewtonAlgorithm<OP5, VECTOR, LOCALDOPEDIM, LOCALDEALDIM> RNA5;

typedef ForwardTimestepReducedProblem<NLS, NLS, INTEGRATOR, INTEGRATOR, OP1,VECTOR, LOCALDOPEDIM, LOCALDEALDIM> SSolver1;
typedef ForwardTimestepReducedProblem<NLS, NLS, INTEGRATOR, INTEGRATOR, OP2,VECTOR, LOCALDOPEDIM, LOCALDEALDIM> SSolver2;
typedef ForwardTimestepReducedProblem<NLS, NLS, INTEGRATOR, INTEGRATOR, OP3,VECTOR, LOCALDOPEDIM, LOCALDEALDIM> SSolver3;
typedef ForwardTimestepReducedProblem<NLS, NLS, INTEGRATOR, INTEGRATOR, OP4,VECTOR, LOCALDOPEDIM, LOCALDEALDIM> SSolver4;
typedef ForwardTimestepReducedProblem<NLS, NLS2, INTEGRATOR, INTEGRATOR, OP5,VECTOR, LOCALDOPEDIM, LOCALDEALDIM> SSolver5;


int
main(int argc, char **argv)
{
  /**
   * In this example we will solve the one dimensional heat equation.
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
  SSolver1::declare_params(pr);
  SSolver2::declare_params(pr);
  SSolver3::declare_params(pr);
  SSolver4::declare_params(pr);
  SSolver5::declare_params(pr);
  RNA1::declare_params(pr);
  pr.read_parameters(paramfile);

  std::string cases = "solve";

  //Create the triangulation.
  Triangulation < LOCALDEALDIM > triangulation;
  GridGenerator::hyper_cube(triangulation, 0., 1.);


  //Define the Finite Elements and quadrature formulas for control and state.
  DOpEWrapper::FiniteElement<LOCALDEALDIM>
      control_fe(FE_Q<LOCALDOPEDIM> (1), 1); //Q1 geht auch P0?
  DOpEWrapper::FiniteElement<LOCALDEALDIM> state_fe(FE_Q<LOCALDEALDIM> (1), 1);//Q1


  QGauss<LOCALDEALDIM> quadrature_formula(3);
  QGauss<LOCALDEALDIM - 1> face_quadrature_formula(3);
  IDC idc(quadrature_formula, face_quadrature_formula);


  //Define the localPDE and the functionals we are interested in. Here, LFunc is a dummy necessary for the control,
  //LPF is a SpaceTimePointevaluation
  LocalPDE<VECTOR, LOCALDOPEDIM, LOCALDEALDIM> LPDE;
  LocalFunctional<VECTOR, LOCALDOPEDIM, LOCALDEALDIM> LFunc;
  LocalPointFunctional<VECTOR, LOCALDOPEDIM, LOCALDEALDIM> LPF;

  //Time grid of [0,1]
  dealii::Triangulation<1> times;
  dealii::GridGenerator::subdivided_hyper_cube(times, 50);

  triangulation.refine_global(4);
  MethodOfLines_SpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR,
      SparsityMaker<DOFHANDLER, 1> , ConstraintsMaker<DOFHANDLER, 1> ,
      LOCALDOPEDIM, LOCALDEALDIM> DOFH(triangulation, control_fe, state_fe,
      times);


  NoConstraints<CellDataContainer,FaceDataContainer,DOFHANDLER,VECTOR, LOCALDOPEDIM, LOCALDEALDIM> Constraints;
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
  DOpEWrapper::ZeroFunction<LOCALDEALDIM> zf;
  SimpleDirichletData<VECTOR, LOCALDOPEDIM, LOCALDEALDIM> DD1(zf);

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

  SSolver1 solver1(&P1, "fullmem", pr, idc);
  SSolver2 solver2(&P2, "fullmem", pr, idc);
  SSolver3 solver3(&P3,  "fullmem", pr, idc);
  SSolver4 solver4(&P4,  "fullmem", pr, idc);
  SSolver5 solver5(&P5,  "fullmem", pr, idc);
  
  //Use one outputhandler for all problems
  DOpEOutputHandler<VECTOR> output(&solver1,pr);
  DOpEExceptionHandler<VECTOR> ex(&output);

  RNA1 Alg1(&P1, &solver1, pr,&ex,&output);
  RNA2 Alg2(&P2, &solver2, pr,&ex,&output);
  RNA3 Alg3(&P3, &solver3, pr,&ex,&output);
  RNA4 Alg4(&P4, &solver4, pr,&ex,&output);
  RNA5 Alg5(&P5, &solver5, pr,&ex,&output);

  try
    {

      Alg1.ReInit();
      Alg2.ReInit();
      Alg3.ReInit();
      Alg4.ReInit();
      Alg5.ReInit();

      Vector<double> solution;
      ControlVector<VECTOR> q(&DOFH, "fullmem");

      Alg1.SolveForward(q);
      Alg2.SolveForward(q);
      Alg3.SolveForward(q);
      Alg4.SolveForward(q);
      Alg5.SolveForward(q);

      SolutionExtractor<SSolver1, VECTOR> a1(solver1);
      const StateVector<VECTOR> &statevec1 = a1.GetU();
      SolutionExtractor<SSolver2, VECTOR> a2(solver2);
      const StateVector<VECTOR> &statevec2 = a2.GetU();
      SolutionExtractor<SSolver3, VECTOR> a3(solver3);
      const StateVector<VECTOR> &statevec3 = a3.GetU();
      SolutionExtractor<SSolver4, VECTOR> a4(solver4);
      const StateVector<VECTOR> &statevec4 = a4.GetU();
      SolutionExtractor<SSolver5, VECTOR> a5(solver5);
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
      std::cout << "Warning: During execution of `" + e.GetThrowingInstance()
          + "` the following Problem occurred!" << std::endl;
      std::cout << e.GetErrorMessage() << std::endl;
    }

  return 0;
}

#undef LOCALDOPEDIM
#undef LOCALDEALDIM
