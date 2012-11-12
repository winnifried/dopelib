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

#include "generalized_mma_algorithm.h"
#include "optproblemcontainer.h"
#include "functionalinterface.h"
#include "pdeinterface.h"
#include "statreducedproblem.h" 
#include "voidreducedproblem.h"
#include "newtonsolver.h"
#include "cglinearsolver.h"
#include "directlinearsolver.h"
#include "integrator.h"
#include "parameterreader.h"
#include "mol_spacetimehandler.h"
#include "simpledirichletdata.h"
#include "constraints.h"
#include "localconstraints.h"
#include "localconstraintaccessor.h"
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

#include "localpde.h"
#include "localfunctional.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

#define CDC CellDataContainer
#define FDC FaceDataContainer

#define VECTOR BlockVector<double>
#define SPARSITYPATTERN BlockSparsityPattern
#define MATRIX BlockSparseMatrix<double>
#define DOFHANDLER DoFHandler<2>
#define FE FESystem<2>

#define FUNC FunctionalInterface<CDC,FDC,DOFHANDLER,VECTOR,2,2>
#define PDE PDEInterface<CDC,FDC,DOFHANDLER,VECTOR,2,2>
#define DD DirichletDataInterface<VECTOR,2,2>
#define CONS ConstraintInterface<CDC,FDC,DOFHANDLER,VECTOR,2,2>


typedef SpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR, 2,2> STH;

typedef OptProblemContainer<FUNC, FUNC, PDE, DD, CONS, SPARSITYPATTERN,VECTOR,2,2> OP;

typedef AugmentedLagrangianProblem<LocalConstraintAccessor,STH,OP, 2, 2,1> ALagOP;
typedef IntegratorDataContainer<DOFHANDLER, Quadrature<2>, Quadrature<1>, VECTOR, 2> IDC;
typedef Integrator<IDC,VECTOR,double,2> INTEGRATOR;

//Uncomment to use a CG-Method with Identity Preconditioner
//typedef CGLinearSolverWithMatrix<INTEGRATOR,PreconditionIdentity,OP,BlockSparsityPattern,BlockSparseMatrix<double>,BlockVector<double>,2> LINEARSOLVER;
//Uncomment to use UMFPACK
typedef DirectLinearSolverWithMatrix<SPARSITYPATTERN,MATRIX,VECTOR,2> LINEARSOLVER;

typedef NewtonSolver<INTEGRATOR,LINEARSOLVER,VECTOR,2> NLS;
typedef StatReducedProblem<NLS,NLS,INTEGRATOR,INTEGRATOR,OP,VECTOR,2,2> SSolver;
typedef VoidReducedProblem<NLS,INTEGRATOR,ALagOP,VECTOR,2,2> ALagSSolver;
//typedef GeneralizedMMAAlgorithm<LocalConstraintAccessor,IDC,STH,OP,BlockVector<double>,ALagSSolver,2,2,1> MMA;
typedef GeneralizedMMAAlgorithm<LocalConstraintAccessor,IDC,STH,OP,BlockVector<double>,ALagSSolver,2,2,1> MMA;

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
  MMA::declare_params(pr);
  NLS::declare_params(pr);

  pr.read_parameters(paramfile);

  Triangulation<2>     triangulation;
  std::vector<unsigned int> rep(2);
  rep[0]  = 2;
  rep[1]  = 1;
  GridGenerator::subdivided_hyper_rectangle (triangulation,rep,Point<2>(0,0),Point<2>(2,1),true);
 
  FE  control_fe(FE_DGP<2>(0),1);
  FE  state_fe(FE_Q<2>(2),2);
  
  QGauss<2> quadrature_formula(3);
  QGauss<1> face_quadrature_formula(2);
  IDC idc(quadrature_formula, face_quadrature_formula);

  LocalPDE<CDC,FDC,DOFHANDLER,VECTOR,2,2> LPDE;
  LocalFunctional<CDC,FDC,DOFHANDLER,VECTOR,2,2> LFunc;

  //triangulation.refine_global (5);
  triangulation.refine_global (3);

  {//Set Dirichlet Boundary!
    for (Triangulation<2>::active_cell_iterator
	   cell = triangulation.begin_active();
	 cell != triangulation.end(); ++cell)
      for (unsigned int f=0; f<GeometryInfo<2>::faces_per_cell; ++f)
      {
	if(cell->face(f)->at_boundary())
	{
	  if (cell->face(f)->center()[1] == 0)
	  {
	    cell->face(f)->set_all_boundary_indicators(5);
	    if(fabs(cell->face(f)->center()[0]-2.) < std::max(0.25,cell->face(f)->diameter()))
	    {
	      cell->face(f)->set_all_boundary_indicators(2);
	    }
	  }
	}
      }
  }

  //Add Constrained description
  std::vector<std::vector<unsigned int> > lcc(1);//1 Control Block
  lcc[0].resize(2);
  lcc[0][0]=1; //each component is constrained individualy
  lcc[0][1]=2; // each two constraints (lower and upper bound)
  Constraints constraints(lcc,1);

  MethodOfLines_SpaceTimeHandler<FE,DOFHANDLER,SPARSITYPATTERN,VECTOR,2,2> DOFH(triangulation, 
									  control_fe,
									  state_fe,
									  constraints,
 DOpEtypes::stationary);
  
  LocalConstraintAccessor CA;
  LocalConstraint<CDC,FDC,DOFHANDLER,VECTOR,2,2> LC(CA);
  
  OP P(LFunc,
       LPDE,
       LC,
       DOFH);  


  std::vector<bool> comp_mask(2);
  comp_mask[0] = false;
  comp_mask[1] = true;
  std::vector<bool> comp_mask_2(2);
  comp_mask_2[0] = true;
  comp_mask_2[1] = false;
  DOpEWrapper::ZeroFunction<2> zf(2);
  SimpleDirichletData<BlockVector<double>,2,2> DD_1(zf);
  P.SetDirichletBoundaryColors(2,comp_mask,&DD_1);
  P.SetDirichletBoundaryColors(0,comp_mask_2,&DD_1);
  
  P.SetBoundaryFunctionalColors(3);
  P.SetBoundaryEquationColors(3);

  //SSolver solver(&P,"fullmem",pr,quadrature_formula,face_quadrature_formula);
  SSolver solver(&P,"fullmem",pr,idc);
  
  //MMA Alg(&P,&CA,&solver,"fullmem",pr,quadrature_formula,face_quadrature_formula);
  MMA Alg(&P,&CA,&solver,"fullmem",pr,idc);
    
  
    
//int niter = 4;
  int niter = 1;
  
  Alg.ReInit();
  ControlVector<BlockVector<double> > q(&DOFH,"fullmem");
  //init q
  {
    q=0.4;
  }
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
      {//Set Dirichlet Boundary!
	for (Triangulation<2>::active_cell_iterator
	       cell = triangulation.begin_active();
	     cell != triangulation.end(); ++cell)
	  for (unsigned int f=0; f<GeometryInfo<2>::faces_per_cell; ++f)
	  {
	    if(cell->face(f)->at_boundary())
	    {
	      if (cell->face(f)->center()[1] == 0)
	      {
		cell->face(f)->set_all_boundary_indicators(5);
		if((fabs(cell->face(f)->center()[0]-2.) < std::max(0.25,cell->face(f)->diameter())))
		{
		  cell->face(f)->set_all_boundary_indicators(2);
		}
	      }
	    }
	  }
      } 
      Alg.ReInit();
    }
  }
  return 0;
}
