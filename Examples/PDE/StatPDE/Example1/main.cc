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
#include "celldatacontainer.h"
#include "facedatacontainer.h"
#include "sparsitymaker.h"
#include "userdefineddofconstraints.h"
#include "integratordatacontainer.h"

#include <iostream>
#include <fstream>

#include <grid/tria.h>
#include <grid/grid_in.h>
#include <dofs/dof_handler.h>
#include <grid/grid_generator.h>
#include <fe/fe_q.h>
#include <dofs/dof_tools.h>
#include <base/quadrature_lib.h>
#include <base/function.h>

#include "localpde.h"
#include "functionals.h"

#include "my_functions.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

#define MATRIX dealii::BlockSparseMatrix<double>
#define SPARSITYPATTERN dealii::BlockSparsityPattern
#define VECTOR dealii::BlockVector<double>
#define DOFHANDLER dealii::DoFHandler<2>
#define FE DOpEWrapper::FiniteElement<2>

//typedef PDEProblemContainer<PDEInterface<DOFHANDLER,VECTOR, 2>,DirichletDataInterface<VECTOR,2>,SPARSITYPATTERN, VECTOR,2> OP;
typedef PDEProblemContainer<LocalPDE<VECTOR, 2>,
    DirichletDataInterface<VECTOR, 2>, SPARSITYPATTERN, VECTOR, 2> OP;
typedef IntegratorDataContainer<DOFHANDLER, dealii::Quadrature<2>,
    dealii::Quadrature<1>, VECTOR, 2> IDC;

typedef Integrator<IDC, VECTOR, double, 2> INTEGRATOR;

typedef DirectLinearSolverWithMatrix<SPARSITYPATTERN, MATRIX, VECTOR, 2> LINEARSOLVER;

typedef NewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR, 2> NLS;
typedef StatPDEProblem<NLS, INTEGRATOR, OP, VECTOR, 2> SSolver;

int
main(int argc, char **argv)
{
  /**
   *  In this example we solve stationary (linear) Stokes' equations
   *  with symmetric stress tensor and do-nothing condition on
   *  the outflow boundary. In this case we employ an additional
   *  term on the outlfow boundary due the symmetry of the stress tensor
   */
  //TODO Make this run with hp
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

  pr.read_parameters(paramfile);

  Triangulation<2> triangulation;

  GridIn<2> grid_in;
  grid_in.attach_triangulation(triangulation);
  std::ifstream input_file("channel.inp");
  grid_in.read_ucd(input_file);

  DOpEWrapper::FiniteElement<2> state_fe(FE_Q<2>(2), 2, FE_Q<2>(1), 1); //Q1
  /******hp******************/
//  DOpEWrapper::FECollection<2>  state_fe_collection(state_fe);
  /******hp******************/

  QGauss<2> quadrature_formula(3);
  QGauss<1> face_quadrature_formula(3);
  IDC idc(quadrature_formula, face_quadrature_formula);

  /******hp******************/
//  dealii::hp::QCollection<2> q_coll(quadrature_formula);
//  dealii::hp::QCollection<1> face_q_coll(face_quadrature_formula);
  /******hp******************/

  LocalPDE<VECTOR, 2> LPDE;

  LocalPointFunctionalX<VECTOR, 2> LPFX;
  LocalBoundaryFluxFunctional<VECTOR, 2> LBFF;

//  //pseudo time
//  dealii::Triangulation<1> times;
//  dealii::GridGenerator::hyper_cube(times);
  triangulation.refine_global(3);

  MethodOfLines_StateSpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR, 2> DOFH(
      triangulation, state_fe);

  /***************hp********************/
//  MethodOfLines_StateSpaceTimeHandler<2> DOFH(triangulation,state_fe_collection,times);
  /***************hp********************/

  OP P(LPDE, DOFH);

  P.AddFunctional(&LPFX);
  P.AddFunctional(&LBFF);

  // fuer Flux Auswertung am Ausflussrand
  P.SetBoundaryFunctionalColors(1);

  std::vector<bool> comp_mask(3);

  comp_mask[0] = true;
  comp_mask[1] = true;
  comp_mask[2] = false;

  DOpEWrapper::ZeroFunction<2> zf(3);
  SimpleDirichletData<VECTOR, 2> DD1(zf);

  BoundaryParabel boundary_parabel;
  SimpleDirichletData<VECTOR, 2> DD2(boundary_parabel);

  P.SetDirichletBoundaryColors(0, comp_mask, &DD2);

  P.SetDirichletBoundaryColors(2, comp_mask, &DD1);
  P.SetDirichletBoundaryColors(3, comp_mask, &DD1);

  P.SetBoundaryEquationColors(1);

  SSolver solver(&P, "fullmem", pr, idc);

  /***************hp********************/
//  SSolver solver(&P,"fullmem",pr,
//                 q_coll,
//                 face_q_coll);
  /***************hp********************/
  //Only needed for pure PDE Problems
  DOpEOutputHandler<VECTOR> out(&solver, pr);
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

  return 0;
}
