#include "pdeproblemcontainer.h"
#include "functionalinterface.h"
#include "pdeinterface.h"
#include "statpdeproblem.h"
#include "newtonsolver.h"
#include "directlinearsolver.h"
#include "userdefineddofconstraints.h"
#include "myconstraintsmaker.h"
#include "sparsitymaker.h"
#include "integratordatacontainer.h"

#include "integrator.h"
#include "parameterreader.h"

#include "mol_statespacetimehandler.h"
#include "simpledirichletdata.h"
#include "active_fe_index_setter_interface.h"

#include <iostream>
#include <fstream>

#include <grid/tria.h>
#include <grid/grid_in.h>
#include <grid/tria_boundary_lib.h>
#include <dofs/dof_handler.h>
#include <grid/grid_generator.h>
#include <fe/fe_q.h>
#include <fe/fe_nothing.h>
#include <dofs/dof_tools.h>
#include <base/quadrature_lib.h>
#include <base/function.h>

#include "localpde.h"
#include "functionals.h"

using namespace std;
using namespace dealii;
using namespace DOpE;



#define VECTOR Vector<double>
#define MATRIX SparseMatrix<double>
#define SPARSITYPATTERN SparsityPattern
#define DOFHANDLER DoFHandler<2>
#define FE FESystem<2>
#define FACEDATACONTAINER FaceDataContainer<DOFHANDLER, VECTOR, 2>

typedef PDEProblemContainer<PDEInterface<CellDataContainer,FaceDataContainer,DOFHANDLER, VECTOR, 2> , DirichletDataInterface<VECTOR, 2> ,
			    SPARSITYPATTERN, VECTOR, 2> OP;
typedef IntegratorDataContainer<DOFHANDLER, Quadrature<2>, Quadrature<1>, VECTOR, 2 > IDC;
typedef Integrator<IDC, VECTOR, double, 2> INTEGRATOR;
//********************Linearsolver**********************************
typedef DirectLinearSolverWithMatrix<SPARSITYPATTERN, MATRIX,
    VECTOR, 2> LINEARSOLVER;
//********************Linearsolver**********************************

typedef NewtonSolver<INTEGRATOR, LINEARSOLVER, VECTOR, 2> NLS;
typedef StatPDEProblem<NLS, INTEGRATOR, OP, VECTOR, 2> SSolver;

void declare_params(ParameterReader &param_reader)
{
	param_reader.SetSubsection("main parameters");
	param_reader.declare_entry("max_iter", "1", Patterns::Integer(0),
	                           "How many iterations?");
	param_reader.declare_entry("quad order", "2", Patterns::Integer(1),
	                           "Order of the quad formula?");
	param_reader.declare_entry("facequad order", "2", Patterns::Integer(1),
	                           "Order of the face quad formula?");
	param_reader.declare_entry("order fe", "2", Patterns::Integer(1),
	                           "Order of the finite element?");
	param_reader.declare_entry("prerefine", "1", Patterns::Integer(1),
	                           "How often should we refine the coarse grid?");
}

int main(int argc, char **argv)
{
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
 	declare_params(pr);

	pr.read_parameters(paramfile);

	//************************************************
	//define some constants
	pr.SetSubsection("main parameters");
	int max_iter = pr.get_integer("max_iter");
	int prerefine = pr.get_integer("prerefine");

	//*************************************************

	//Make triangulation *************************************************
	const Point<2> center(0, 0);
	const HyperShellBoundary<2> boundary_description(center);
	Triangulation<2> triangulation;
	GridGenerator::hyper_cube_with_cylindrical_hole(triangulation, 0.1,
	                                                        1., 1, 1, true);
	triangulation.set_boundary(4, boundary_description);
	if (prerefine > 0)
		triangulation.refine_global(prerefine);
	//*************************************************

	//FiniteElemente*************************************************
	pr.SetSubsection("main parameters");
	FESystem < 2 > state_fe(FE_Q<2>(pr.get_integer("order fe")), 2);

	//Quadrature formulas*************************************************
	pr.SetSubsection("main parameters");
	QGauss<2> quadrature_formula(pr.get_integer("quad order"));
	QGauss<1> face_quadrature_formula(pr.get_integer("facequad order"));
	IDC idc(quadrature_formula, face_quadrature_formula);
	//**************************************************************************************************

	//Functionals*************************************************
	LocalBoundaryFunctionalMassFlux<VECTOR, FACEDATACONTAINER, 2> LBFMF;
	LocalPDELaplace<VECTOR,  2> LPDE;
	//*************************************************

	//pseudo time*************************************************
	std::vector<double> times(1, 0.);
	//*************************************************

	/***********************************/
	DOpE::PeriodicityConstraints<2> constraints_mkr;
	MethodOfLines_StateSpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR, 2> DOFH(triangulation,  state_fe);
	//Add the periodicity constraints through the following:
	DOFH.SetUserDefinedDoFConstraints(constraints_mkr);
	/***********************************/

	OP P(LPDE, DOFH);
	P.AddFunctional(&LBFMF);
	P.SetBoundaryFunctionalColors(1);
	//Boundary conditions************************************************
	std::vector<bool> comp_mask(2);
	comp_mask[0] = true;
	comp_mask[1] = true;

	DOpEWrapper::ZeroFunction < 2 > zf(2);
	SimpleDirichletData<VECTOR, 2> DD1(zf);
	//Set zero dirichlet at the hole in the middle of the domain
	P.SetDirichletBoundaryColors(4, comp_mask, &DD1);
	/************************************************/
	SSolver solver(&P, "fullmem", pr,idc);


        //Only needed for pure PDE Problems
        DOpEOutputHandler<VECTOR> out(&solver,pr);
        DOpEExceptionHandler<VECTOR> ex(&out);
        P.RegisterOutputHandler(&out);
        P.RegisterExceptionHandler(&ex);
        solver.RegisterOutputHandler(&out);
        solver.RegisterExceptionHandler(&ex); 
	/**********************************************************************/
	for (int i = 0; i < max_iter; i++)
	{
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
		} catch (DOpEException &e)
		{
			std::cout
			    << "Warning: During execution of `" + e.GetThrowingInstance()
			        + "` the following Problem occurred!" << std::endl;
			std::cout << e.GetErrorMessage() << std::endl;
		}
		if (i != max_iter - 1)
		{
		  DOFH.RefineSpace("global");
		}
	}
	return 0;
}
