/**
 *
 * Copyright (C) 2012-2018 by the DOpElib authors
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
#include <iostream>
#include <chrono>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/precondition_block.h>

#include <deal.II/lac/sparse_direct.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/table.h>

#include <opt_algorithms/reduceddampedbfgsalgorithm.h>

#include <container/eigenvalueproblemcontainer.h>
#include <interfaces/functionalinterface.h>
#include <include/parameterreader.h>
#include <problemdata/simpledirichletdata.h>
#include <problemdata/noconstraints.h>
#include <wrapper/preconditioner_wrapper.h>
#include <container/integratordatacontainer.h>
#include <reducedproblems/eigenvaluereducedproblem.h>

#include <include/solutionextractor.h>

#include <templates/integratoreigenvalue.h>
#include <templates/newtonsolver_eigenvalueproblems.h>

#include <templates/voidlinearsolver.h>
#include <templates/directlinearsolver.h>
#include <templates/cglinearsolver.h>
#include <templates/eigenvalue_solver.h>

#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_full_matrix.h>
#if DEAL_II_VERSION_GTE(9,4,0)
#include <deal.II/lac/petsc_vector.h>
#else
#include <deal.II/lac/petsc_parallel_vector.h>
#endif
#include <deal.II/lac/petsc_precondition.h>


#include "localpde.h"
#include "localfunctional.h"
#include "deformation_functions.h"
#include <basic/test_mol_spacetimehandler.h>
#include "functionals.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

const static int DIM = 2;
const static int CDIM = 2;

#if DEAL_II_VERSION_GTE(9,3,0)
#define DOFHANDLER false
#else
#define DOFHANDLER DoFHandler
#endif
#define FE FESystem
#define CDC ElementDataContainer
#define FDC FaceDataContainer

typedef QGauss<DIM> QUADRATURE;
typedef QGauss<DIM - 1> FACEQUADRATURE;

typedef PETScWrappers::SparseMatrix MATRIX;
typedef SparseMatrix<double> MATRIXFORLINSOLVE;
typedef SparsityPattern SPARSITYPATTERN;

typedef Vector<double> VECTOR;
typedef double EIGENVALUE;

typedef LocalFunctional<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> COSTFUNCTIONAL;
typedef FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> FUNCTIONALINTERFACE;

typedef EigenvalueProblemContainer<FUNCTIONALINTERFACE, COSTFUNCTIONAL,
        LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM>,
        SimpleDirichletData<VECTOR, DIM>,
        NoConstraints<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM>, SPARSITYPATTERN,
        VECTOR, CDIM, DIM> OP;

typedef IntegratorDataContainer<DOFHANDLER, QUADRATURE, FACEQUADRATURE, VECTOR,
        DIM> IDC;

typedef Integrator_eigenval<IDC, VECTOR, double, DIM> INTEGRATOR;
typedef Integrator_eigenval<IDC, VECTOR, double, DIM> INTEGRATOR_CONTROL;

typedef CGLinearSolverWithMatrix<DOpEWrapper::PreconditionIdentity_Wrapper<MATRIXFORLINSOLVE>,SPARSITYPATTERN, MATRIXFORLINSOLVE, VECTOR> LINEARSOLVER;

typedef Newtonsolver_Eigenvalueproblems<INTEGRATOR,VECTOR,EIGENVALUE ,MATRIX, SPARSITYPATTERN, LINEARSOLVER> NS;
typedef EigenvalueSolver<INTEGRATOR,VECTOR, MATRIX> EVS;
typedef EigenvalueReducedProblem<NS, EVS,INTEGRATOR_CONTROL, INTEGRATOR, OP, VECTOR, CDIM,
        DIM> RP;

typedef ReducedDampedBFGSAlgorithm<OP, VECTOR> RNA;


typedef MethodOfLines_SpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR,
        CDIM, DIM> STH;

int

main(int argc, char **argv){
  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv,1);



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
  RP::declare_params(pr);
  NS::declare_params(pr);
  EVS::declare_params(pr);
  RNA::declare_params(pr);
  COSTFUNCTIONAL::declare_params(pr);
  DOpEOutputHandler<VECTOR>::declare_params(pr);

  pr.read_parameters(paramfile);

  Triangulation<DIM> triangulation;



 GridGenerator::hyper_rectangle(triangulation,
  		  Point<2>(0,0),
  		  Point<2>(M_PI/3,M_PI/2));   //1. EW ~4.0*/


  triangulation.refine_global(3);


  //------------- FE-System ---------------------------------------------
   FE<DIM> control_fe(FE_Q<DIM>(1), 2);
   FESystem<DIM> state_fe(FE_Q<DIM>(1), 1 , FE_Nedelec<DIM>(0),1);

  QUADRATURE quadrature_formula(8);
  FACEQUADRATURE face_quadrature_formula(8);
  IDC idc(quadrature_formula, face_quadrature_formula);

  LocalPDE<CDC, FDC, DOFHANDLER, VECTOR, DIM> LPDE(pr);

  COSTFUNCTIONAL LFunc(pr);



  STH DOFH(triangulation, control_fe, state_fe, DOpEtypes::stationary);

  NoConstraints<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> Constraints;
  OP P(LFunc, LPDE, Constraints, DOFH);

  //Functional which returns acutal eigenvalue
  LocalEigenvalueFunctional<CDC, FDC, DOFHANDLER, VECTOR, CDIM, DIM> EVF;
  P.AddFunctional(&EVF);
  
  std::vector<bool> comp_mask(3);

   comp_mask[0] = true;
   DOpEWrapper::ZeroFunction<DIM> zf(3);
   SimpleDirichletData<VECTOR, DIM> DD1(zf);

//TODO in  mol space time handler angepasst für NedelecRB
   P.SetDirichletBoundaryColors(0, comp_mask, &DD1);
   P.SetDirichletBoundaryColors(1, comp_mask, &DD1);
   P.SetDirichletBoundaryColors(2, comp_mask, &DD1);
   P.SetDirichletBoundaryColors(3, comp_mask, &DD1);

   std::vector<bool> comp_mask_c(2);
   comp_mask_c[0] = true;
    comp_mask_c[1] = false;
    DOpEWrapper::ZeroFunction<CDIM> zf_c(2);
    P.SetControlDirichletBoundaryColors(0, comp_mask_c, &zf_c);
    P.SetControlDirichletBoundaryColors(1, comp_mask_c, &zf_c);
    comp_mask_c[0] = false;
    comp_mask_c[1] = true;

    P.SetControlDirichletBoundaryColors(2, comp_mask_c, &zf_c);
    P.SetControlDirichletBoundaryColors(3, comp_mask_c, &zf_c);

  RP solver(&P, DOpEtypes::VectorStorageType::fullmem, pr, idc, 2);
  RNA Alg(&P, &solver, pr);

  DOpEOutputHandler<VECTOR> out(&solver, pr);
  DOpEExceptionHandler<VECTOR> ex(&out);

  P.RegisterOutputHandler(&out);
  P.RegisterExceptionHandler(&ex);

  solver.RegisterOutputHandler(&out);
  solver.RegisterExceptionHandler(&ex);
  solver.ReInit();


  ControlVector<VECTOR> q(&DOFH, DOpEtypes::VectorStorageType::fullmem,pr);
  q = 0;
  ControlVector<VECTOR> dq(q);


      try
        {
    	  solver.ReInit();
    	  Alg.ReInit();
            Alg.Solve(q);
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

#undef FDC
#undef CDC
#undef FE
#undef DOFHANDLER
