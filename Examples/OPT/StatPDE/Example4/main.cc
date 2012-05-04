
#include "reducednewtonalgorithm.h"
#include "optproblemcontainer.h"
#include "functionalinterface.h"
#include "pdeinterface.h"
#include "statreducedproblem.h"
#include "cglinearsolver.h"
#include "directlinearsolver.h"
#include "voidlinearsolver.h"
#include "integrator.h"
#include "newtonsolver.h"
#include "integratormixeddims.h"
#include "newtonsolvermixeddims.h"
#include "parameterreader.h"
#include "finiteelement_wrapper.h"
#include "mol_spacetimehandler.h"
#include "localdirichletdata.h"
#include "noconstraints.h"
#include "sparsitymaker.h"
#include "userdefineddofconstraints.h"
#include "integratordatacontainer.h"

#include <iostream>

#include <grid/tria.h>
#include <dofs/dof_handler.h>
#include <grid/grid_generator.h>
#include <fe/fe_q.h>
#include <fe/fe_nothing.h>
#include <dofs/dof_tools.h>
#include <base/quadrature_lib.h>
#include <base/function.h>
#include <base/point.h>

#include "localpde.h"
#include "localfunctional.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

#define VECTOR dealii::BlockVector<double>
#define DOFHANDLER dealii::DoFHandler<2>
#define FE DOpEWrapper::FiniteElement<2>

typedef OptProblemContainer<FunctionalInterface<CellDataContainer,FaceDataContainer,DOFHANDLER, VECTOR, 0,2>,
FunctionalInterface<CellDataContainer,FaceDataContainer,DOFHANDLER, VECTOR, 0,2>,
PDEInterface<CellDataContainer,FaceDataContainer,DOFHANDLER, VECTOR,0,2>,
DirichletDataInterface<VECTOR,0,2>,
ConstraintInterface<CellDataContainer,FaceDataContainer,DOFHANDLER, VECTOR,0,2>,
BlockSparsityPattern, VECTOR,0,2> OP;

typedef IntegratorDataContainer<DOFHANDLER, dealii::Quadrature<2>, dealii::Quadrature<1>, VECTOR, 2 > IDC;
typedef Integrator<IDC,VECTOR,double,2> INTEGRATOR;
typedef IntegratorMixedDimensions<IDC,VECTOR,double,0,2> INTEGRATORM;
//Uncomment to use UMFPACK
typedef DirectLinearSolverWithMatrix<BlockSparsityPattern,BlockSparseMatrix<double>,VECTOR,2> LINEARSOLVER;

typedef VoidLinearSolver<BlockSparsityPattern,BlockSparseMatrix<double>,VECTOR,2> VOIDLS;

typedef NewtonSolverMixedDimensions<INTEGRATORM,VOIDLS,VECTOR,0,2> NLSM;
typedef NewtonSolver<INTEGRATOR,LINEARSOLVER,VECTOR,2> NLS;
typedef ReducedNewtonAlgorithm<OP,VECTOR,0,2> RNA;
typedef StatReducedProblem<NLSM,NLS,INTEGRATORM,INTEGRATOR,OP,VECTOR,0,2> SSolver;


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
  pr.read_parameters(paramfile);


  Triangulation<2>     triangulation;
  Point<2> p1(0,0),p2(1,1);

  GridGenerator::hyper_rectangle (triangulation, p1,p2,true);

  //DOpEWrapper::FiniteElement<0>          control_fe(5); //5 Parameter
  DOpEWrapper::FiniteElement<2>          control_fe(FE_Nothing<2>(1),5); //5 Parameter
  DOpEWrapper::FiniteElement<2>          state_fe(FE_Q<2>(1),2);   //


  QGauss<2>   quadrature_formula(2);
  QGauss<1> face_quadrature_formula(2);
  IDC idc(quadrature_formula, face_quadrature_formula);

  LocalPDE<VECTOR,0,2> LPDE;
  LocalFunctional<VECTOR,0,2> LFunc;

  std::vector<double> times(1,0.);
  triangulation.refine_global (5);

  MethodOfLines_SpaceTimeHandler<FE, DOFHANDLER, BlockSparsityPattern,VECTOR, 0,2> DOFH(triangulation,control_fe, state_fe);

  NoConstraints<CellDataContainer,FaceDataContainer,DOFHANDLER,VECTOR,0,2> Constraints;

  OP P(LFunc,
       LPDE,
       Constraints,
       DOFH);

  std::vector<bool> comp_mask(2);
  comp_mask[0] = true;
  comp_mask[1] = true;

  LocalDirichletData<VECTOR,0,2> DD;
  P.SetDirichletBoundaryColors(0,comp_mask,&DD);
  P.SetDirichletBoundaryColors(1,comp_mask,&DD);
  P.SetDirichletBoundaryColors(2,comp_mask,&DD);
  P.SetDirichletBoundaryColors(3,comp_mask,&DD);

  SSolver solver(&P,"fullmem", pr, idc);
  RNA Alg(&P,&solver, pr);


  int niter = 1;
  dealii::Vector<double> qinit(5);
  {
    qinit(0) =0.;
    qinit(1) =0.;
    qinit(2) =0.;
    qinit(3) =0.;
    qinit(4) =1.;
  }
  Alg.ReInit();
  ControlVector<VECTOR > q(&DOFH,"fullmem");
  q.GetSpacialVector() = qinit;

  for(int i = 0; i < niter; i++)
  {
    try
    {
      Alg.Solve(q);
    }
    catch(DOpEException &e)
    {
      std::cout<<"Warning: During execution of `"
	+ e.GetThrowingInstance()
	+ "` the following Problem occurred!"
	       <<std::endl;

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
