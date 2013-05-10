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

#include "reduced_ipopt_algorithm.h"
#include "reduced_snopt_algorithm.h"
#include "optproblemcontainer.h"
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
#include "userdefineddofconstraints.h"
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
#define VECTOR BlockVector<double>
#define SPARSITYPATTERN BlockSparsityPattern
#define MATRIX BlockSparseMatrix<double>
#define DOFHANDLER DoFHandler
#define FE FESystem
#define FUNC FunctionalInterface<CDC,FDC,DOFHANDLER,VECTOR,2,2>
#define PDE PDEInterface<CDC,FDC,DOFHANDLER,VECTOR,2>
#define DD DirichletDataInterface<VECTOR,2>
#define CONS ConstraintInterface<CDC,FDC,DOFHANDLER,VECTOR,2,2>

typedef OptProblemContainer<FUNC, FUNC, PDE, DD, CONS, SPARSITYPATTERN,VECTOR,2,2> OP;

typedef IntegratorDataContainer<DOFHANDLER, Quadrature<2>, Quadrature<1>, VECTOR, 2> IDC;
typedef Integrator<IDC,VECTOR,double,2> INTEGRATOR;

//Uncomment to use a CG-Method with Identity Preconditioner
typedef CGLinearSolverWithMatrix<DOpEWrapper::PreconditionIdentity_Wrapper<MATRIX>,SPARSITYPATTERN,MATRIX,VECTOR> LINEARSOLVER;
//Uncomment to use UMFPACK
//typedef DirectLinearSolverWithMatrix<SPARSITYPATTERN,MATRIX,VECTOR> LINEARSOLVER;

typedef NewtonSolver<INTEGRATOR,LINEARSOLVER,VECTOR> NLS;
typedef StatReducedProblem<NLS,NLS,INTEGRATOR,INTEGRATOR,OP,VECTOR,2,2> SSolver;

typedef Reduced_SnoptAlgorithm<OP,VECTOR> SNOPT_Alg;
typedef Reduced_IpoptAlgorithm<OP,VECTOR> IPOPT_Alg;

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
  SNOPT_Alg::declare_params(pr);
  IPOPT_Alg::declare_params(pr);
  NLS::declare_params(pr);

  pr.read_parameters(paramfile);

  Triangulation<2>     triangulation;
  GridGenerator::hyper_cube (triangulation, 0, 1);
 
  FE<2>  control_fe(FE_Q<2>(1),1);
  FE<2>  state_fe(FE_Q<2>(2),1);
  
  QGauss<2> quadrature_formula(2);
  QGauss<1> face_quadrature_formula(2);
  IDC idc(quadrature_formula, face_quadrature_formula);

  LocalPDE<DOFHANDLER, VECTOR,2> LPDE;
  LocalFunctional<DOFHANDLER, VECTOR, 2,2> LFunc;
 
  //AuxFunctionals
  LocalPointFunctional<DOFHANDLER, VECTOR,2,2> LPF;
  LocalMeanValueFunctional<DOFHANDLER, VECTOR,2,2> LMF;

  triangulation.refine_global (5);

  //Add Constrained description
  std::vector<std::vector<unsigned int> > lcc(1);//1 Control Block
  lcc[0].resize(2);
  lcc[0][0]=1; //each component is constrained individualy
  lcc[0][1]=2; // each two constraints (lower and upper bound)
  Constraints constraints(lcc,0); //Second entry defines the numer of global constraints, here we have none
  
  MethodOfLines_SpaceTimeHandler<FE,DOFHANDLER,SPARSITYPATTERN,VECTOR,
    2,2> DOFH(triangulation,
									  control_fe,
									  state_fe,
									  constraints, DOpEtypes::stationary);
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
  SimpleDirichletData<BlockVector<double>,2> DD_1(zf);
  P.SetDirichletBoundaryColors(0,comp_mask,&DD_1);

  SSolver solver(&P,"fullmem",pr,idc,2);
  
  //SNOPT_Alg Alg(&P,&solver,"fullmem",pr);
  IPOPT_Alg Alg(&P,&solver,"fullmem",pr);
    
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
      DOFH.RefineSpace();
      Alg.ReInit();
    }
  }
  return 0;
}
