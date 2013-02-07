/**
*
* Copyright (C) 2012 by the DOpElib authors
*
* This file is part of DOpElib
*
* DOpElib is free software: you can redistribute it
* and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either
* version 3 of the License, or (at your option) any later
* version.
*
* DOpElib is distributed in the hope that it will be
* useful, but WITHOUT ANY WARRANTY; without even the implied
* warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
* PURPOSE.  See the GNU General Public License for more
* details.
*
* Please refer to the file LICENSE.TXT included in this distribution
* for further information on this license.
*
**/


#include "reducednewtonalgorithm.h"
#include "optproblemcontainer.h"
#include "functionalinterface.h"
#include "pdeinterface.h"
#include "statreducedproblem.h"
#include "newtonsolver.h"
#include "cglinearsolver.h"
#include "directlinearsolver.h"
#include "integrator_multimesh.h"
#include "parameterreader.h"
#include "mol_multimesh_spacetimehandler.h"
#include "simpledirichletdata.h"
#include "noconstraints.h"
#include "sparsitymaker.h"
#include "userdefineddofconstraints.h"
#include "preconditioner_wrapper.h"
#include "integratordatacontainer.h"

#include <iostream>

#include <grid/tria.h>
#include <dofs/dof_handler.h>
#include <grid/grid_generator.h>
#include <fe/fe_q.h>
#include <fe/fe_dgp.h>
#include <dofs/dof_tools.h>
#include <base/quadrature_lib.h>
#include <base/function.h>
#include <lac/precondition_block.h>
#include <numerics/error_estimator.h>

#include "localpde.h"
#include "localfunctional.h"
#include "functionals.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

#define VECTOR BlockVector<double>
#define DOFHANDLER DoFHandler<2>
#define FE FESystem<2>

typedef OptProblemContainer<FunctionalInterface<Multimesh_CellDataContainer,Multimesh_FaceDataContainer,DOFHANDLER, VECTOR, 2,2>,
		   FunctionalInterface<Multimesh_CellDataContainer,Multimesh_FaceDataContainer,DOFHANDLER, VECTOR, 2,2>,
		   PDEInterface<Multimesh_CellDataContainer,Multimesh_FaceDataContainer,DOFHANDLER, VECTOR,2,2>,
		   DirichletDataInterface<VECTOR,2,2>,
		   ConstraintInterface<Multimesh_CellDataContainer,Multimesh_FaceDataContainer,DOFHANDLER, VECTOR, 2,2>,BlockSparsityPattern, VECTOR, 2,2> OP;

typedef IntegratorDataContainer<DOFHANDLER, Quadrature<2>, Quadrature<1>, VECTOR, 2 > IDC;

typedef IntegratorMultiMesh<IDC,VECTOR,double,2> INTEGRATOR;

//Uncomment to use a CG-Method with Identity Preconditioner
typedef CGLinearSolverWithMatrix<DOpEWrapper::PreconditionIdentity_Wrapper<BlockSparseMatrix<double> >,BlockSparsityPattern,BlockSparseMatrix<double>,VECTOR,2> LINEARSOLVER;
//Uncomment to use UMFPACK
//typedef DirectLinearSolverWithMatrix<BlockSparsityPattern,BlockSparseMatrix<double>,VECTOR,2> LINEARSOLVER;

typedef NewtonSolver<INTEGRATOR,LINEARSOLVER,VECTOR,2> NLS;
typedef ReducedNewtonAlgorithm<OP,VECTOR,2,2> RNA;
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

  pr.read_parameters(paramfile);

  Triangulation<2>     triangulation;
  GridGenerator::hyper_cube (triangulation, 0, 1);

//FESystem<2>          control_fe(FE_DGP<2>(0),1);
  FESystem<2>          control_fe(FE_Q<2>(1),1);
  FESystem<2>          state_fe(FE_Q<2>(1),1);

  QGauss<2>   quadrature_formula(2);
  QGauss<1> face_quadrature_formula(2);
  IDC idc(quadrature_formula, face_quadrature_formula);

  LocalPDE<VECTOR, 2,2> LPDE;
  LocalFunctional<VECTOR, 2,2> LFunc;

  //AuxFunctionals
  LocalPointFunctional<VECTOR,2,2> LPF;
  LocalMeanValueFunctional<VECTOR,2,2> LMF;
  QErrorFunctional<VECTOR,2,2> QEF;
  UErrorFunctional<VECTOR,2,2> UEF;

//  std::vector<double> times(1,0.);
  Triangulation<1> times;
  GridGenerator::hyper_cube(times);
  triangulation.refine_global (3);

  MethodOfLines_MultiMesh_SpaceTimeHandler<FE, DOFHANDLER, BlockSparsityPattern,VECTOR, 2> DOFH(triangulation,control_fe, state_fe, DOpEtypes::stationary);

  NoConstraints<Multimesh_CellDataContainer,Multimesh_FaceDataContainer,DOFHANDLER,VECTOR, 2,2> Constraints;

  OP P(LFunc,
       LPDE,
       Constraints,
       DOFH);

  P.AddFunctional(&LPF);
  P.AddFunctional(&LMF);
  P.AddFunctional(&QEF);
  P.AddFunctional(&UEF);


  std::vector<bool> comp_mask(1);
  comp_mask[0] = true;
  DOpEWrapper::ZeroFunction<2> zf(1);
  SimpleDirichletData<VECTOR,2,2> DD(zf);
  P.SetDirichletBoundaryColors(0,comp_mask,&DD);

  SSolver solver(&P, "fullmem", pr, idc);

  RNA Alg(&P,&solver, pr);
 

  int niter = 3;
  Alg.ReInit();
  ControlVector<VECTOR > q(&DOFH,"fullmem");
  DOFH.RefineStateSpace();
  Alg.ReInit();

  for(int i = 0; i < niter; i++)
  {
    try
    {
      q = 0.;
      Alg.Solve(q);
    }
    catch(DOpEException &e)
    {
      std::cout<<"Warning: During execution of `" + e.GetThrowingInstance() + "` the following Problem occurred!"<<std::endl;
      std::cout<<e.GetErrorMessage()<<std::endl;
    }
    if(i != niter-1)
    {
      //DOFH.RefineSpace();
      DOFH.RefineControlSpace();
      Alg.ReInit();
    }
  }
  DOFH.RefineStateSpace();
  Alg.ReInit();
 
  for(int i = 0; i < niter; i++)
  {
    try
    {
      q = 0.;
      Alg.Solve(q);
    }
    catch(DOpEException &e)
    {
      std::cout<<"Warning: During execution of `" + e.GetThrowingInstance() + "` the following Problem occurred!"<<std::endl;
      std::cout<<e.GetErrorMessage()<<std::endl;
    }
    if(i != niter-1)
    {
      if( i%2 == 0)
      {
	SolutionExtractor<SSolver,VECTOR >  a(solver);
	const StateVector<VECTOR > &gu = a.GetU();
	Vector<double> solution;
	solution = 0;
	solution = gu.GetSpacialVector();
	Vector<float> estimated_error_per_cell (triangulation.n_active_cells());
	
	std::vector<bool> component_mask (1, true);
	
	KellyErrorEstimator<2>::estimate (static_cast<const DoFHandler<2>&>(DOFH.GetStateDoFHandler()),
					  QGauss<1>(2),
					  FunctionMap<2>::type(),
					  solution,
					  estimated_error_per_cell,
					  component_mask);
	DOFH.RefineStateSpace(RefineFixedNumber(estimated_error_per_cell,0.1,0.0));
	Alg.ReInit();
      }
      if( i%2 == 1)
      {
	Vector<double> solution;
	solution = 0;
	solution = q.GetSpacialVector();
	Vector<float> estimated_error_per_cell (triangulation.n_active_cells());
	
	std::vector<bool> component_mask (1, true);
	
	KellyErrorEstimator<2>::estimate (static_cast<const DoFHandler<2>&>(DOFH.GetControlDoFHandler()),
					  QGauss<1>(2),
					  FunctionMap<2>::type(),
					  solution,
					  estimated_error_per_cell,
					  component_mask);
	DOFH.RefineControlSpace(RefineFixedNumber(estimated_error_per_cell,0.1,0.0));
	Alg.ReInit();
      }
    }
  }
  return 0;
}
