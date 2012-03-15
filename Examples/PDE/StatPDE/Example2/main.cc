#include "pdeproblemcontainer.h"
#include "functionalinterface.h"
#include "pdeinterface.h"
#include "statpdeproblem.h"
#include "newtonsolver.h"
#include "directlinearsolver.h"
#include "integrator.h"
#include "parameterreader.h"
#include "finiteelement_wrapper.h"
#include "mol_statespacetimehandler.h"
#include "simpledirichletdata.h"
#include "sparsitymaker.h"
#include "constraintsmaker.h"
#include "integratordatacontainer.h"

#include <iostream>
#include <fstream>

#include <grid/tria.h>
#include <grid/grid_in.h>
#include <dofs/dof_handler.h>
#include <grid/grid_generator.h>
#include <fe/fe_q.h>
#include <fe/fe_dgp.h>
#include <dofs/dof_tools.h>
#include <base/quadrature_lib.h>
#include <base/function.h>

#include "localpde.h"
#include "functionals.h"

#include "my_functions.h"



using namespace std;
using namespace dealii;
using namespace DOpE;

#define VECTOR dealii::BlockVector<double>
#define DOFHANDLER dealii::DoFHandler<2>
#define FE DOpEWrapper::FiniteElement<2>

typedef PDEProblemContainer<PDEInterface<CellDataContainer,FaceDataContainer,DOFHANDLER, VECTOR,2>,DirichletDataInterface<VECTOR,2,2>,BlockSparsityPattern,VECTOR,2> OP;
typedef IntegratorDataContainer<DOFHANDLER, dealii::Quadrature<2>, dealii::Quadrature<1>, VECTOR, 2 > IDC;

typedef Integrator<IDC,VECTOR,double,2> INTEGRATOR;

typedef DirectLinearSolverWithMatrix<BlockSparsityPattern,
				     BlockSparseMatrix<double>,
				     VECTOR,2> LINEARSOLVER;

typedef NewtonSolver<INTEGRATOR,LINEARSOLVER,VECTOR,2> NLS;
typedef StatPDEProblem<NLS,INTEGRATOR,OP,VECTOR,2> SSolver;


int main(int argc, char **argv)
{
  /**
   * Stationary FSI problem in an ALE framework
   * Fluid: Stokes equ.
   * Structure: Incompressible INH model
   * We use the Q2^c-P1^dc element for discretization.
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
  DOpEOutputHandler<VECTOR>::declare_params(pr);

  pr.read_parameters(paramfile);

  Triangulation<2>     triangulation;

  GridIn<2> grid_in;
  grid_in.attach_triangulation (triangulation);
  std::ifstream input_file("channel.inp");
  grid_in.read_ucd (input_file);

  DOpEWrapper::FiniteElement<2>      state_fe(FE_Q<2>(2),2,
					      FE_DGP<2>(1),1,
					      FE_Q<2>(2),2);

  QGauss<2>   quadrature_formula(3);
  QGauss<1> face_quadrature_formula(3);
  IDC idc( quadrature_formula, face_quadrature_formula);

  LocalPDE<VECTOR,2> LPDE;
 
  LocalPointFunctionalX<VECTOR,2> LPFX;
  LocalBoundaryFluxFunctional<VECTOR,2>  LBFF;

  //pseudo time
  std::vector<double> times(1,0.);
  triangulation.refine_global (2);

  MethodOfLines_StateSpaceTimeHandler<FE, DOFHANDLER, BlockSparsityPattern,VECTOR, SparsityMaker<DOFHANDLER, 2>, ConstraintsMaker<DOFHANDLER, 2>,2> DOFH(triangulation,state_fe);

  OP P(LPDE,
       DOFH);

  P.AddFunctional(&LPFX);
  P.AddFunctional(&LBFF);

  // fuer Flux Auswertung am Ausflussrand
  P.SetBoundaryFunctionalColors(1);

  std::vector<bool> comp_mask(5);

  comp_mask[0] = true;
  comp_mask[1] = true;
  comp_mask[2] = false;
  comp_mask[3] = true;
  comp_mask[4] = true;

  DOpEWrapper::ZeroFunction<2> zf(5);
  SimpleDirichletData<VECTOR,2> DD1(zf);

  BoundaryParabel boundary_parabel;
  SimpleDirichletData<VECTOR,2> DD2(boundary_parabel);
  P.SetDirichletBoundaryColors(0,comp_mask,&DD2);
  P.SetDirichletBoundaryColors(2,comp_mask,&DD1);
  P.SetDirichletBoundaryColors(3,comp_mask,&DD1);
  P.SetBoundaryEquationColors(1);

  //comp_mask[3] = true;
  //comp_mask[4] = true;
  //P.SetDirichletBoundaryColors(1,comp_mask,&zf);

  SSolver solver(&P,"fullmem",pr,idc);
  //Only needed for pure PDE Problems
  DOpEOutputHandler<VECTOR> out(&solver,pr);
  DOpEExceptionHandler<VECTOR> ex(&out);
  P.RegisterOutputHandler(&out);
  P.RegisterExceptionHandler(&ex);
  solver.RegisterOutputHandler(&out);
  solver.RegisterExceptionHandler(&ex); 
 

  try
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
    
    solver.ComputeReducedFunctionals();    
  }
  catch(DOpEException &e)
  {
    std::cout<<"Warning: During execution of `" + e.GetThrowingInstance() + "` the following Problem occurred!"<<std::endl;
    std::cout<<e.GetErrorMessage()<<std::endl;
  }

  return 0;
}
