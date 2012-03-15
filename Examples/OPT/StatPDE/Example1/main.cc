
#include "reducednewtonalgorithm.h"
#include "reducedtrustregionnewton.h"
#include "optproblem.h"
#include "functionalinterface.h"
#include "pdeinterface.h"
#include "statreducedproblem.h"
#include "newtonsolver.h"
#include "cglinearsolver.h"
#include "directlinearsolver.h"
#include "integrator.h"
#include "parameterreader.h"
#include "mol_spacetimehandler.h"
#include "simpledirichletdata.h"
#include "noconstraints.h"
#include "sparsitymaker.h"
#include "constraintsmaker.h"
#include "preconditioner_wrapper.h"
#include "integratordatacontainer.h"

#include <iostream>

#include <grid/tria.h>
#include <dofs/dof_handler.h>
#include <grid/grid_generator.h>
#include <fe/fe_q.h>
#include <dofs/dof_tools.h>
#include <base/quadrature_lib.h>
#include <base/function.h>
#include <lac/precondition_block.h>

#include "localpde.h"
#include "localfunctional.h"
#include "functionals.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

#define VECTOR dealii::BlockVector<double>
#define DOFHANDLER dealii::DoFHandler<2>
#define FE DOpEWrapper::FiniteElement<2>

typedef OptProblem<FunctionalInterface<CellDataContainer,FaceDataContainer,DOFHANDLER, VECTOR, 2,2>,
		   FunctionalInterface<CellDataContainer,FaceDataContainer,DOFHANDLER, VECTOR, 2,2>,
		   PDEInterface<CellDataContainer,FaceDataContainer,DOFHANDLER, VECTOR,2,2>,
		   DirichletDataInterface<VECTOR,2,2>,
		   ConstraintInterface<CellDataContainer,FaceDataContainer,DOFHANDLER, VECTOR, 2,2>,
		   BlockSparsityPattern, VECTOR, 2,2> OP;

typedef IntegratorDataContainer<DOFHANDLER, dealii::Quadrature<2>, dealii::Quadrature<1>, VECTOR, 2 > IDC;

typedef Integrator<IDC,VECTOR,double,2> INTEGRATOR;

//Uncomment to use a CG-Method with Identity Preconditioner
typedef CGLinearSolverWithMatrix<DOpEWrapper::PreconditionIdentity_Wrapper<BlockSparseMatrix<double> >,BlockSparsityPattern,BlockSparseMatrix<double>,VECTOR,2> LINEARSOLVER;
//Uncomment to use UMFPACK
//typedef DirectLinearSolverWithMatrix<OP,BlockSparsityPattern,BlockSparseMatrix<double>,VECTOR,2> LINEARSOLVER;

typedef NewtonSolver<INTEGRATOR,LINEARSOLVER,VECTOR,2> NLS;
typedef ReducedNewtonAlgorithm<OP,VECTOR,2,2> RNA;
typedef ReducedTrustregion_NewtonAlgorithm<OP,VECTOR,2,2> RNA2;
typedef StatReducedProblem<NLS,NLS,INTEGRATOR,INTEGRATOR,OP,VECTOR,2,2> SSolver;

int main(int argc, char **argv)
{

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
  RNA2::declare_params(pr);
  DOpEOutputHandler<VECTOR>::declare_params(pr);

  pr.read_parameters(paramfile);

  std::string cases = "solve";

  Triangulation<2>     triangulation;
  GridGenerator::hyper_cube (triangulation, 0, 1);

  DOpEWrapper::FiniteElement<2>          control_fe(FE_Q<2>(1),1);
  DOpEWrapper::FiniteElement<2>          state_fe(FE_Q<2>(2),1);

  QGauss<2>   quadrature_formula(2);
  QGauss<1> face_quadrature_formula(2);
  IDC idc(quadrature_formula, face_quadrature_formula);

  LocalPDE<VECTOR, 2,2> LPDE;
  LocalFunctional<VECTOR, 2,2> LFunc;

  //AuxFunctionals
  LocalPointFunctional<VECTOR,2,2> LPF;
  LocalMeanValueFunctional<VECTOR,2,2> LMF;

//  std::vector<double> times(1,0.);
  dealii::Triangulation<1> times;
  dealii::GridGenerator::hyper_cube(times);
  triangulation.refine_global (5);

  MethodOfLines_SpaceTimeHandler<FE, DOFHANDLER, BlockSparsityPattern,VECTOR, SparsityMaker<DOFHANDLER,2>, ConstraintsMaker<DOFHANDLER,2>, 2,2> DOFH(triangulation,control_fe, state_fe);

  NoConstraints<CellDataContainer,FaceDataContainer,DOFHANDLER,VECTOR, 2,2> Constraints;

  OP P(LFunc,
       LPDE,
       Constraints,
       DOFH);

  P.AddFunctional(&LPF);
  P.AddFunctional(&LMF);


  std::vector<bool> comp_mask(1);
  comp_mask[0] = true;
  DOpEWrapper::ZeroFunction<2> zf(1);
  SimpleDirichletData<VECTOR,2,2> DD(zf);
  P.SetDirichletBoundaryColors(0,comp_mask,&DD);

  SSolver solver(&P, "fullmem", pr, idc,2);
  //Make shure we use the same outputhandler
  DOpEOutputHandler<VECTOR> out(&solver,pr);
  DOpEExceptionHandler<VECTOR> ex(&out);
  RNA Alg(&P,&solver, pr,&ex,&out);
  RNA2 Alg2(&P,&solver, pr,&ex,&out);

  int niter = 2;
  Alg.ReInit();
  out.ReInit();
  ControlVector<VECTOR > q(&DOFH,"fullmem");
 
  for(int i = 0; i < niter; i++)
  {
    try
    {
      if( cases == "check" )
      {
	ControlVector<VECTOR > dq(q);
	Alg.CheckGrads(1.,q,dq,2);
	Alg.CheckHessian(1.,q,dq,2);
      }
      else
      {
	
	Alg2.Solve(q);
	q = 0.;
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
      //triangulation.refine_global (1);
      DOFH.RefineSpace("global");
      Alg.ReInit();
      out.ReInit();
    }
  }

  return 0;
}
