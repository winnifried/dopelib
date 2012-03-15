#include "reduced_snopt_algorithm.h"
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
#include "localconstraints.h"
#include "localconstraintaccessor.h"

using namespace dealii;
using namespace DOpE;

#define CDC CellDataContainer
#define FDC FaceDataContainer
#define VECTOR dealii::BlockVector<double>
#define SPARSITYPATTERN dealii::BlockSparsityPattern
#define MATRIX dealii::BlockSparseMatrix<double>
#define DOFHANDLER dealii::DoFHandler<2>
#define FE DOpEWrapper::FiniteElement<2>
#define FUNC DOpE::FunctionalInterface<CDC,FDC,DOFHANDLER,VECTOR,2,2>
#define PDE DOpE::PDEInterface<CDC,FDC,DOFHANDLER,VECTOR,2,2>
#define DD DOpE::DirichletDataInterface<VECTOR,2,2>
#define CONS DOpE::ConstraintInterface<CDC,FDC,DOFHANDLER,VECTOR,2,2>

typedef OptProblem<FUNC, FUNC, PDE, DD, CONS, SPARSITYPATTERN,VECTOR,2,2> OP;

typedef IntegratorDataContainer<DOFHANDLER, dealii::Quadrature<2>, dealii::Quadrature<1>, VECTOR, 2> IDC;
typedef Integrator<IDC,VECTOR,double,2> INTEGRATOR;

//Uncomment to use a CG-Method with Identity Preconditioner
typedef CGLinearSolverWithMatrix<DOpEWrapper::PreconditionIdentity_Wrapper<BlockSparseMatrix<double> >,BlockSparsityPattern,BlockSparseMatrix<double>,BlockVector<double>,2> LINEARSOLVER;
//Uncomment to use UMFPACK
//typedef DirectLinearSolverWithMatrix<SPARSITYPATTERN,MATRIX,VECTOR,2> LINEARSOLVER;

typedef NewtonSolver<INTEGRATOR,LINEARSOLVER,VECTOR,2> NLS;
typedef StatReducedProblem<NLS,NLS,INTEGRATOR,INTEGRATOR,OP,VECTOR,2,2> SSolver;

typedef Reduced_SnoptAlgorithm<OP,dealii::BlockVector<double>,2,2> MMA;

int main(int argc, char **argv)
{  
  std::string paramfile = "dope.prm";
    
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
  MMA::declare_params(pr);
  NLS::declare_params(pr);

  pr.read_parameters(paramfile);

  Triangulation<2>     triangulation;
  GridGenerator::hyper_cube (triangulation, 0, 1);
 
  FE  control_fe(FE_Q<2>(1),1);
  FE  state_fe(FE_Q<2>(2),1);
  
  QGauss<2> quadrature_formula(2);
  QGauss<1> face_quadrature_formula(2);
  IDC idc(quadrature_formula, face_quadrature_formula);

  LocalPDE<VECTOR, 2,2> LPDE;
  LocalFunctional<VECTOR, 2,2> LFunc;
 
  //AuxFunctionals
  LocalPointFunctional<VECTOR,2,2> LPF;
  LocalMeanValueFunctional<VECTOR,2,2> LMF;

  triangulation.refine_global (5);

  //Add Constrained description
  std::vector<std::vector<unsigned int> > lcc(1);//1 Control Block
  lcc[0].resize(2);
  lcc[0][0]=1; //each component is constrained individualy
  lcc[0][1]=2; // each two constraints (lower and upper bound)
  Constraints constraints(lcc,0); //Second entry defines the numer of global constraints, here we have none
  
  MethodOfLines_SpaceTimeHandler<FE,DOFHANDLER,SPARSITYPATTERN,VECTOR,
    SparsityMaker<DOFHANDLER,2>, ConstraintsMaker<DOFHANDLER,2>,2,2> DOFH(triangulation, 
									  control_fe,
									  state_fe,
									  constraints);
  LocalConstraintAccessor CA;
  LocalConstraint<CDC,FDC,DOFHANDLER,VECTOR,2,2> LC(CA);

  OP P(LFunc,
       LPDE,
       LC,
       DOFH);

  P.AddFunctional(&LPF);
  P.AddFunctional(&LMF);
 
  std::vector<bool> comp_mask(1);
  comp_mask[0] = true;
  DOpEWrapper::ZeroFunction<2> zf(1);
  SimpleDirichletData<BlockVector<double>,2,2> DD_1(zf);
  P.SetDirichletBoundaryColors(0,comp_mask,&DD_1);

  SSolver solver(&P,"fullmem",pr,idc,2);
  
  MMA Alg(&P,&solver,"fullmem",pr);
    
  int niter = 1;
  
  Alg.ReInit();
  ControlVector<BlockVector<double> > q(&DOFH,"fullmem");
 
  for(int i = 0; i < niter; i++)
  {
    try
    {
      Alg.Solve(q);
    }
    catch(DOpEException &e)
    {
      std::cout<<"Warning: During execution of `" + e.GetThrowingInstance() + "` the following Problem occurred!"<<std::endl;
      std::cout<<e.GetErrorMessage()<<std::endl;      
    }
    if(i != niter-1)
    {
      DOFH.RefineSpace("global");
      Alg.ReInit();
    }
  }
  return 0;
}
