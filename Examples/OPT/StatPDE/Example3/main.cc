#include "reducednewtonalgorithm.h"
#include "optproblem.h"
#include "functionalinterface.h"
#include "pdeinterface.h"
#include "statreducedproblem.h"
#include "newtonsolver.h"
#include "gmreslinearsolver.h"
#include "cglinearsolver.h"
#include "directlinearsolver.h"
#include "voidlinearsolver.h" // for mixed dim opt. control
#include "integrator.h"
#include "noconstraints.h"
#include "solutionextractor.h"

#include "integratormixeddims.h"    // for mixed dim opt. control
#include "newtonsolvermixeddims.h"  // for mixed dim opt. control
#include "finiteelement_wrapper.h"

#include "parameterreader.h"
#include "mol_spacetimehandler.h"
#include "simpledirichletdata.h"
#include "sparsitymaker.h"
#include "constraintsmaker.h"
#include "integratordatacontainer.h"

// C++
#include <iostream>
#include <fstream>


// deal.II
#include <grid/tria.h>
#include <grid/grid_in.h>
#include <grid/tria_boundary_lib.h>
#include <dofs/dof_handler.h>
#include <grid/grid_generator.h>
#include <fe/fe_q.h>
#include <fe/fe_dgp.h>
#include <fe/fe_nothing.h>
#include <dofs/dof_tools.h>
#include <base/quadrature_lib.h>
#include <base/function.h>

// for grid local refinement
#include <numerics/error_estimator.h>
#include <grid/grid_refinement.h>


#include "localpde.h"
#include "localfunctional.h"
#include "functionals.h"

#include "my_functions.h"



using namespace std;
using namespace dealii;
using namespace DOpE;

#define VECTOR dealii::BlockVector<double>
#define DOFHANDLER dealii::DoFHandler<2>
#define FE DOpEWrapper::FiniteElement<2>

typedef OptProblem<FunctionalInterface<CellDataContainer,FaceDataContainer,DOFHANDLER, VECTOR,0,2>,
		   FunctionalInterface<CellDataContainer,FaceDataContainer,DOFHANDLER, VECTOR,0,2>,
		   PDEInterface<CellDataContainer,FaceDataContainer,DOFHANDLER, VECTOR,0,2>,
		   DirichletDataInterface<VECTOR,0,2>,
		   ConstraintInterface<CellDataContainer,FaceDataContainer,DOFHANDLER, VECTOR,0,2>,
		   BlockSparsityPattern,VECTOR,0,2> OP;

typedef IntegratorDataContainer<DOFHANDLER, dealii::Quadrature<2>, dealii::Quadrature<1>, VECTOR, 2 > IDC;
typedef Integrator<IDC,VECTOR,double,2> INTEGRATOR;
typedef IntegratorMixedDimensions<IDC,VECTOR,double,0,2> INTEGRATORM;

typedef DirectLinearSolverWithMatrix<BlockSparsityPattern,
				     BlockSparseMatrix<double>,
				     VECTOR,2> LINEARSOLVER;

typedef VoidLinearSolver<BlockSparsityPattern,BlockSparseMatrix<double>,VECTOR,2> VOIDLS;  // mixed dim optimal control

typedef NewtonSolverMixedDimensions<INTEGRATORM,VOIDLS,VECTOR,0,2> NLSM;    // mixed dim optimal control
typedef NewtonSolver<INTEGRATOR,LINEARSOLVER,VECTOR,2> NLS;
typedef ReducedNewtonAlgorithm<OP,VECTOR,0,2> RNA;
typedef StatReducedProblem<NLSM,NLS,INTEGRATORM,INTEGRATOR,OP,VECTOR,0,2> SSolver;


int main(int argc, char **argv)
{
  /**
   * In this example we study
   * stationary flow control. The configuration
   * comes from the original fluid benchmark problem
   * and has been modified to reduce drag around the
   * cylinder. The gain the solvability of
   * the optimization problem we add a quadratic
   * regularization term to the cost functional.
   * Please note: although we provide a fluid example
   * program we consider five solution variables
   * v_x,v_y,p,u_x,u_y since this example is the basis
   * for FSI optimization.
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
  LocalPDE<VECTOR,0,2>::declare_params(pr);
  LocalFunctional<VECTOR,0,2>::declare_params(pr);
  BoundaryParabel::declare_params(pr);
  LocalBoundaryFaceFunctionalDrag<VECTOR,0,2>::declare_params(pr);
  LocalBoundaryFaceFunctionalLift<VECTOR,0,2>::declare_params(pr);
  pr.read_parameters(paramfile);



  Triangulation<2>     triangulation;

  GridIn<2> grid_in;
  grid_in.attach_triangulation (triangulation);
  // Grid for "normal" fluid Benchmark
  std::ifstream input_file("nsbench4_original.inp");

  grid_in.read_ucd (input_file);

  Point<2> p(0.2,0.2);
  double radius = 0.05;
  static const HyperBallBoundary<2> boundary(p,radius);
  triangulation.set_boundary (80, boundary);

  // DOpEWrapper::FiniteElement<0>          control_fe(2); //2 Parameter
  DOpEWrapper::FiniteElement<2>          control_fe(FE_Nothing<2>(1),2); //2 Parameter

  // Implementation for state
  // includes finite element for the displacements, too.
  // That is, because this example is the basis for
  // FSI optimal control and is extended in example 5.
  DOpEWrapper::FiniteElement<2>      state_fe(FE_Q<2>(2),2,   // velocities
					      FE_Q<2>(1),1, // pressure with CG(1)
 					      FE_Q<2>(2),2);  // displacements

  QGauss<2>   quadrature_formula(3);
  QGauss<1> face_quadrature_formula(3);
  IDC idc(quadrature_formula, face_quadrature_formula);

  LocalPDE<VECTOR,0,2> LPDE(pr);
  LocalFunctional<VECTOR,0,2> LFunc(pr);

  LocalPointFunctionalPressure<VECTOR,0,2> LPFP;
  LocalPointFunctionalDeflectionX<VECTOR,0,2> LPFDX;
  LocalPointFunctionalDeflectionY<VECTOR,0,2> LPFDY;
  LocalBoundaryFaceFunctionalDrag<VECTOR,0,2>  LBFD(pr);
  LocalBoundaryFaceFunctionalLift<VECTOR,0,2>  LBFL(pr);



  //pseudo time
  std::vector<double> times(1,0.);
  triangulation.refine_global (2);

  MethodOfLines_SpaceTimeHandler<FE, DOFHANDLER, BlockSparsityPattern, VECTOR, SparsityMaker<DOFHANDLER,2>, ConstraintsMaker<DOFHANDLER,2>, 0,2> DOFH(triangulation,control_fe, state_fe);

  NoConstraints<CellDataContainer,FaceDataContainer,DOFHANDLER,VECTOR, 0,2> Constraints;

  OP P(LFunc,
       LPDE,
       Constraints,
       DOFH);


  P.AddFunctional(&LPFP);
  P.AddFunctional(&LBFD);
  P.AddFunctional(&LBFL);

  // fuer Drag und Lift Auswertung am Zylinder
  P.SetBoundaryFunctionalColors(80);

  // Due to regularization
  P.SetBoundaryFunctionalColors(50);
  P.SetBoundaryFunctionalColors(51);

  std::vector<bool> comp_mask(5);

  comp_mask[0] = true;
  comp_mask[1] = true;
  comp_mask[2] = false;
  comp_mask[3] = true;
  comp_mask[4] = true;

  DOpEWrapper::ZeroFunction<2> zf(5);
  SimpleDirichletData<VECTOR,0,2> DD1(zf);

  BoundaryParabel boundary_parabel(pr);
  SimpleDirichletData<VECTOR,0,2> DD2(boundary_parabel);
  P.SetDirichletBoundaryColors(0,comp_mask,&DD2);  // flow by Dirichlet data
  P.SetDirichletBoundaryColors(2,comp_mask,&DD1);
  P.SetDirichletBoundaryColors(80,comp_mask,&DD1);

  P.SetBoundaryEquationColors(1);   // do-nothing at outflow boundary
  P.SetBoundaryEquationColors(50); // upper control bc \Gamma_q1
  P.SetBoundaryEquationColors(51); // lower control bc \Gamma_q2

   // We need these functions to evaluate
   // BoundaryEquation_Q, etc.
   P.SetControlBoundaryEquationColors(50); // upper control bc \Gamma_q1
   P.SetControlBoundaryEquationColors(51); // lower control bc \Gamma_q2


  SSolver solver(&P,"fullmem",pr,idc);
  RNA Alg(&P,&solver, pr);


   std::string cases = "solve";
  // Mesh-refinement cycles
  int niter = 1;

  Vector<double> solution;
  Alg.ReInit(); 
  ControlVector<VECTOR > q(&DOFH,"fullmem");
  q = 0.1;
  for(int i = 0; i < niter; i++)
    {
      try
	{
	  if( cases == "check" )
	    {

	      ControlVector<VECTOR > dq(q);
	      // eps: step size for difference quotient
	      // choose: 1.0, 0.1, 0.01, etc.
	      double eps_diff = 1.0e-2;
	      Alg.CheckGrads(eps_diff,q,dq,2);
	      Alg.CheckHessian(eps_diff,q,dq,2);
	    }
	  else
	    {
	      //Alg.SolveForward(q);  // just solves the forward problem
	      Alg.Solve(q);
	    }
	}
      catch(DOpEException &e)
	{
	  std::cout<<"Warning: During execution of `" + e.GetThrowingInstance() + "` the following Problem occurred!"<<std::endl;
	  std::cout<<e.GetErrorMessage()<<std::endl;
	}
      if(i != niter-1)
	{
	  SolutionExtractor<SSolver,VECTOR >  a(solver);
	  const StateVector<VECTOR > &gu = a.GetU();
	  solution = 0;
	  solution = gu.GetSpacialVector();
	  Vector<float> estimated_error_per_cell (triangulation.n_active_cells());

	  std::vector<bool> component_mask (5, false);
	  component_mask[2] = true;


 	  KellyErrorEstimator<2>::estimate (static_cast<const DoFHandler<2>&>(DOFH.GetStateDoFHandler()),
 					      QGauss<1>(2),
 					      FunctionMap<2>::type(),
 					      solution,
 					      estimated_error_per_cell,
 					      component_mask
 					      );
//	  GridRefinement::refine_and_coarsen_fixed_number (triangulation,
//							   estimated_error_per_cell,
//							   0.5, 0.0);
//	  triangulation.prepare_coarsening_and_refinement();
//	  triangulation.execute_coarsening_and_refinement ();

	  //triangulation.refine_global (1);
	  DOFH.RefineSpace("fixednumber",&estimated_error_per_cell,0.5,0.0);
	  Alg.ReInit();
	}


}


  return 0;
}
