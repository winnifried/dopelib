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
#include "gmreslinearsolver.h"
#include "cglinearsolver.h"
#include "directlinearsolver.h"
#include "integrator.h"
#include "parameterreader.h"
#include "mol_spacetimehandler.h"
#include "simpledirichletdata.h"
#include "noconstraints.h"
#include "sparsitymaker.h"
#include "constraintsmaker.h"
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
#define VECTOR dealii::BlockVector<double>
#define SPARSITYPATTERN dealii::BlockSparsityPattern
#define MATRIX dealii::BlockSparseMatrix<double>
#define DOFHANDLER dealii::DoFHandler<2>
#define FE DOpEWrapper::FiniteElement<2>
#define FUNC DOpE::FunctionalInterface<CellDataContainer,FaceDataContainer,DOFHANDLER,VECTOR,LOCALDOPEDIM,LOCALDEALDIM>
#define PDE DOpE::PDEInterface<CellDataContainer,FaceDataContainer,DOFHANDLER,VECTOR,LOCALDOPEDIM,LOCALDEALDIM>
#define DD DOpE::DirichletDataInterface<VECTOR,LOCALDOPEDIM,LOCALDEALDIM>
#define CONS DOpE::ConstraintInterface<CellDataContainer,FaceDataContainer,DOFHANDLER,VECTOR,LOCALDOPEDIM,LOCALDEALDIM>


typedef OptProblem<FUNC,FUNC,PDE,DD,CONS,SPARSITYPATTERN, VECTOR, LOCALDOPEDIM,LOCALDEALDIM> OP_BASE;
#define PROB DOpE::StateProblem<OP_BASE,PDE,DD,SPARSITYPATTERN,VECTOR,LOCALDOPEDIM,LOCALDEALDIM>


// Typedefs for timestep problem
typedef BackwardEulerProblem<PROB,SPARSITYPATTERN, VECTOR,2,2> TSP;
typedef InstatOptProblemContainer<TSP,FUNC,FUNC,PDE,DD,CONS,SPARSITYPATTERN, VECTOR, LOCALDOPEDIM,LOCALDEALDIM> OP;
typedef IntegratorDataContainer<DOFHANDLER, dealii::Quadrature<LOCALDEALDIM>, dealii::Quadrature<LOCALDEALDIM-1>, VECTOR, LOCALDEALDIM > IDC;

typedef Integrator<IDC,VECTOR,double,2> INTEGRATOR;

typedef DirectLinearSolverWithMatrix<SPARSITYPATTERN,
				     MATRIX,
				     VECTOR,2> LINEARSOLVER;

typedef InstatStepNewtonSolver<INTEGRATOR,LINEARSOLVER,VECTOR,2> NLS;
typedef ReducedNewtonAlgorithm<OP,VECTOR,2,2> RNA;
typedef InstatReducedProblem<NLS,NLS,INTEGRATOR,INTEGRATOR,OP,VECTOR,2,2> SSolver;


int main(int argc, char **argv)
{
  /**
   * Instationary FSI problem in an ALE framework
   * Fluid: NSE
   * Structure: INH or STVK
   */

  string paramfile = "dope.prm";

  if(argc == 2)
  {
    paramfile = argv[1];
  }
  else if (argc > 2)
  {
    std::cout<<"Usage: "<<argv[0]<< " [ paramfile ] "<<std::endl;
    return -1;
  }

  ParameterReader pr;
  SSolver::declare_params(pr);
  RNA::declare_params(pr);
  LocalPDE<VECTOR,2,2>::declare_params(pr);
  BoundaryParabel::declare_params(pr);
  LocalBoundaryFaceFunctionalDrag<VECTOR,LOCALDOPEDIM,LOCALDEALDIM>::declare_params(pr);
  LocalBoundaryFaceFunctionalLift<VECTOR,LOCALDOPEDIM,LOCALDEALDIM>::declare_params(pr);
  pr.read_parameters(paramfile);

  std::string cases = "solve";

  Triangulation<2>     triangulation;

  GridIn<2> grid_in;
  grid_in.attach_triangulation (triangulation);

  // Grid for Benchmark with flag
  std::ifstream input_file("bench_fs_t0100_tw.inp");

  grid_in.read_ucd (input_file);

  Point<2> p(0.2,0.2);
  double radius = 0.05;
  static const HyperBallBoundary<2> boundary(p,radius);

  // cylinder boundary
  triangulation.set_boundary (80, boundary);
  // cylinder boundary attached to the flag
  triangulation.set_boundary (81, boundary);


  DOpEWrapper::FiniteElement<2>       control_fe(FE_Q<2>(1),1);

  // FE for the state equation: v,u,p
  DOpEWrapper::FiniteElement<2>       state_fe(FE_Q<2>(2),2,
					       FE_Q<2>(2),2,
					       FE_DGP<2>(1),1
					       );

  QGauss<2> quadrature_formula(3);
  QGauss<1> face_quadrature_formula(3);
  IDC idc(quadrature_formula, face_quadrature_formula);

  LocalPDE<VECTOR, 2,2> LPDE(pr);
  LocalFunctional<VECTOR,2,2> LFunc;

  LocalPointFunctionalPressure<VECTOR,2,2> LPFP;
  LocalPointFunctionalDeflectionX<VECTOR,2,2> LPFDX;
  LocalPointFunctionalDeflectionY<VECTOR, 2,2> LPFDY;

  LocalBoundaryFaceFunctionalDrag<VECTOR,LOCALDOPEDIM,LOCALDEALDIM>  LBFD(pr);
  LocalBoundaryFaceFunctionalLift<VECTOR,LOCALDOPEDIM,LOCALDEALDIM>  LBFL(pr);




 //Time grid of [0,25]
  dealii::Triangulation<1> times;
  dealii::GridGenerator::subdivided_hyper_cube(times, 25,0,25);

  triangulation.refine_global (1);
  MethodOfLines_SpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR,  SparsityMaker<DOFHANDLER,2>, ConstraintsMaker<DOFHANDLER,2>,LOCALDOPEDIM,LOCALDEALDIM> DOFH(triangulation, control_fe, state_fe, times);

  NoConstraints<CellDataContainer,FaceDataContainer,DOFHANDLER,VECTOR,LOCALDOPEDIM,LOCALDEALDIM> Constraints;

  OP P(LFunc,
       LPDE,
       Constraints,
       DOFH);

  //P.HasFaces();
  P.AddFunctional(&LPFP);   // pressure difference
  P.AddFunctional(&LPFDX);  // deflection of x
  P.AddFunctional(&LPFDY);  // deflection of y

  P.AddFunctional(&LBFD);   // drag at cylinder and interface
  P.AddFunctional(&LBFL);   // lift at cylinder and interface

  // fuer Drag und Lift Auswertung am Zylinder
  P.SetBoundaryFunctionalColors(80);
  P.SetBoundaryFunctionalColors(81);


  std::vector<bool> comp_mask(5);

  comp_mask[0] = true;   // vx
  comp_mask[1] = true;   // vy
  comp_mask[2] = true;   // ux
  comp_mask[3] = true;   // uy
  comp_mask[4] = false;  // pressure

  DOpEWrapper::ZeroFunction<2> zf(5);
  SimpleDirichletData<VECTOR, LOCALDOPEDIM,LOCALDEALDIM> DD1(zf);

  BoundaryParabel boundary_parabel(pr);
  SimpleDirichletData<VECTOR, LOCALDOPEDIM,LOCALDEALDIM> DD2(boundary_parabel);
  P.SetDirichletBoundaryColors(0,comp_mask,&DD2);  // inflow boundary
  P.SetDirichletBoundaryColors(2,comp_mask,&DD1);  // rigid walls
  P.SetDirichletBoundaryColors(80,comp_mask,&DD1); // cylinder
  P.SetDirichletBoundaryColors(81,comp_mask,&DD1); // cylinder attached to flag

  P.SetBoundaryEquationColors(1);  // outflow boundary

  BoundaryParabelExact boundary_parabel_ex;
  P.SetInitialValues(&zf);
  //P.SetInitialValues(&boundary_parabel_ex);


  SSolver solver(&P,"fullmem",pr,idc);
  RNA Alg(&P,&solver, pr);



  // Mesh-refinement cycles
  int niter = 1;
  Alg.ReInit();
  ControlVector<VECTOR> q(&DOFH,"fullmem");

  for(int i = 0; i < niter; i++)
    {
      try
	{
	  if( cases == "check" )
	    {
	      ControlVector<VECTOR> dq(q);
	      Alg.CheckGrads(1.,q,dq,2);
	      Alg.CheckHessian(1.,q,dq,2);
	    }
	  else
	    {
	      Alg.SolveForward(q);
	    }
	}
      catch(DOpEException &e)
	{
	  std::cout<<"Warning: During execution of `" + e.GetThrowingInstance() + "` the following Problem occurred!"<<std::endl;
	  std::cout<<e.GetErrorMessage()<<std::endl;
	}
      if(i != niter-1)
	{
	  //triangulation.refine_global (1);
	  DOFH.RefineSpace("global");
	  Alg.ReInit();
	}
}

  return 0;
}
#undef LOCALDOPEDIM
#undef LOCALDEALDIM

