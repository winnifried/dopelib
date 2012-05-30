
#include "pdeproblemcontainer.h"
#include "functionalinterface.h"
#include "pdeinterface.h"
#include "statpdeproblem.h"
#include "newtonsolver.h"
#include "cglinearsolver.h"
#include "directlinearsolver.h"
#include "integrator.h"
#include "parameterreader.h"
#include "mol_statespacetimehandler.h"
#include "simpledirichletdata.h"
#include "noconstraints.h"
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
#include <grid/tria_boundary_lib.h>
#include "localpde.h"
#include "functionals.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

#define MATRIX dealii::BlockSparseMatrix<double>
#define SPARSITYPATTERN dealii::BlockSparsityPattern
#define VECTOR dealii::BlockVector<double>
#define DOFHANDLER dealii::DoFHandler<2>
#define FE FESystem<2>

typedef PDEProblemContainer<LocalPDE<VECTOR, 2,2>,DirichletDataInterface<VECTOR,2>,SPARSITYPATTERN, VECTOR,2> OP;
typedef IntegratorDataContainer<DOFHANDLER, dealii::Quadrature<2>, dealii::Quadrature<1>, VECTOR, 2 > IDC;
typedef Integrator<IDC,VECTOR,double,2> INTEGRATOR;

typedef DirectLinearSolverWithMatrix<BlockSparsityPattern,BlockSparseMatrix<double>,VECTOR,2> LINEARSOLVER;

typedef NewtonSolver<INTEGRATOR,LINEARSOLVER,VECTOR,2> NLS;
typedef StatPDEProblem<NLS,INTEGRATOR,OP,VECTOR,2> SSolver;

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
  DOpEOutputHandler<VECTOR>::declare_params(pr);

  {
    pr.SetSubsection("main");
    pr.declare_entry("niter","6",Patterns::Integer(0));
  }

  pr.read_parameters(paramfile);

  int niter;
  {
    pr.SetSubsection("main");
    niter = pr.get_integer("niter");
  }

  Triangulation<2>     triangulation;
  Point<2> p1(0,0);
  Point<2> p2(30,12);
  std::vector<std::vector<double> > spacing(2);
  spacing[0].resize(30,1.);
  spacing[1].resize(12,1.);
  GridGenerator::subdivided_hyper_rectangle(triangulation,spacing,p1,p2,false);

  FE          state_fe(FE_Q<2>(2),2);

  QGauss<2>   quadrature_formula(2);
  QGauss<1> face_quadrature_formula(2);
  IDC idc(quadrature_formula, face_quadrature_formula);

  LocalPDE<VECTOR, 2,2> LPDE;

  //AuxFunctionals
  LocalPointFunctional<VECTOR,2,2> LPF;
  LocalValueFunctional<VECTOR,2,2> LMF;
  LocalValueFunctional2<VECTOR,2,2> LMF2;

  MethodOfLines_StateSpaceTimeHandler<FE, DOFHANDLER,BlockSparsityPattern, 
    VECTOR, 2> DOFH(triangulation,state_fe);

  OP P(LPDE,
       DOFH);

  P.AddFunctional(&LPF);
  P.AddFunctional(&LMF);
  P.AddFunctional(&LMF2);


  std::vector<bool> comp_mask(2);
  comp_mask[0] = comp_mask[1] = true;
  DOpEWrapper::ZeroFunction<2> zf(2);
  SimpleDirichletData<VECTOR,2,2> DD(zf);
  P.SetDirichletBoundaryColors(0,comp_mask,&DD);
  P.SetDirichletBoundaryColors(1,comp_mask,&DD);
 

  SSolver solver(&P,"fullmem",pr,idc);

  //Only needed for pure PDE Problems
  DOpEOutputHandler<VECTOR> out(&solver,pr);
  DOpEExceptionHandler<VECTOR> ex(&out);
  P.RegisterOutputHandler(&out);
  P.RegisterExceptionHandler(&ex);
  solver.RegisterOutputHandler(&out);
  solver.RegisterExceptionHandler(&ex); 
 
  for(int i = 0; i < niter; i++)
  {
    solver.ReInit();
    out.ReInit();
    stringstream outp;
    
    outp << "**************************************************\n";
    outp << "*             Starting Forward Solve             *\n";
    outp << "*   Solving : "<<P.GetName()<<"\t*\n";
    outp << "*   SDoFs   : ";
    solver.StateSizeInfo(outp);
    outp << "**************************************************";
    out.Write(outp,1,1,1);
    
    try
    {
      solver.ComputeReducedFunctionals();
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
    }
  }

  return 0;
}
