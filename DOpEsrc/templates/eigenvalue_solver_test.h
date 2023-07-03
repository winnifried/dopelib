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

#ifndef EIGENVALUE_SOLVER_TEST_H_
#define EIGENVALUE_SOLVER_TEST_H_

#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/slepc_solver.h>

#include <include/parameterreader.h>
#if DEAL_II_VERSION_GTE(9,4,0)
#include <deal.II/lac/petsc_vector.h>
#else
#include <deal.II/lac/petsc_parallel_vector.h>
#endif

#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>

#include <petscconf.h>
#include <petscksp.h>
#include <slepceps.h>




namespace DOpE
{

  template <typename INTEGRATOR, typename VECTOR, typename MATRIX>
  class EigenvalueSolver_test
  {
  public:
	  EigenvalueSolver_test( INTEGRATOR &integrator, ParameterReader &param_reader);
    ~EigenvalueSolver_test();

    static void declare_params(ParameterReader &param_reader);

    template<typename PROBLEM>
    void ReInit(PROBLEM &pde);

    template<typename PROBLEM>
    void GetNormalizedVectorState(PROBLEM &pde,StateVector<VECTOR> &stateeigenfunction);
    template<typename PROBLEM>
        void GetNormalizedVectorAdjoint(PROBLEM &pde, StateVector<VECTOR> &adjeigenfunction,StateVector<VECTOR> &stateeigenfunction, double value);

    template<typename PROBLEM>
    bool EigenvalueSolve(PROBLEM &pde, std::vector<double> &eigenvalues, std::vector<StateVector<VECTOR>> &eigenfunctions,
    						bool apply_boundary_values=true,
                            bool force_matrix_build=false,
                            int priority = 5, std::string algo_level = "\t\t ");

    inline INTEGRATOR &GetIntegrator();

  private:
    INTEGRATOR &integrator_;
    MATRIX matrixK_, matrixM_;

    IndexSet eigenfunction_index_set;
    std::vector<PETScWrappers::MPI::Vector> eigenvectors_;

	PETScWrappers::MPI::Vector state_for_normalization;
	PETScWrappers::MPI::Vector adjoint_for_normalization;
    std::vector<double> eigenvalues_;
    bool build_matrix_ = true;

    int maxiter_;
    double target_eigenvalue_ , tol_;
  };

  /**********************************Implementation*******************************************/

  template <typename INTEGRATOR, typename VECTOR, typename MATRIX>
  void EigenvalueSolver_test<INTEGRATOR,VECTOR , MATRIX>
  ::declare_params(ParameterReader &param_reader)
  {
	  param_reader.SetSubsection("eigenvalue_solver parameters");
	  param_reader.declare_entry("tol", "0.00001", Patterns::Double(0),"tolerance");
	  param_reader.declare_entry("maxiter", "1000", Patterns::Integer(0),"max iterations");
	  param_reader.declare_entry("target_eigenvalue", "0.001", Patterns::Double(0),"target eigenvalue");

  }

  /*******************************************************************************************/
  template <typename INTEGRATOR, typename VECTOR, typename MATRIX>
  EigenvalueSolver_test<INTEGRATOR,VECTOR, MATRIX>
  ::EigenvalueSolver_test(INTEGRATOR &integrator, ParameterReader &param_reader)
    : integrator_(integrator)
  {
	    param_reader.SetSubsection("eigenvalue_solver parameters");
	    tol_ = param_reader.get_double ("tol");
	    maxiter_ = param_reader.get_integer ("maxiter");
	    target_eigenvalue_ = param_reader.get_double ("target_eigenvalue");
  }

  /*******************************************************************************************/

  template <typename INTEGRATOR, typename VECTOR,  typename MATRIX>
  EigenvalueSolver_test<INTEGRATOR,VECTOR, MATRIX>
  ::~EigenvalueSolver_test()
  {
  }

  /*******************************************************************************************/
  template <typename INTEGRATOR, typename VECTOR, typename MATRIX>
  template<typename PROBLEM>
  void EigenvalueSolver_test<INTEGRATOR, VECTOR, MATRIX>
  ::ReInit(PROBLEM &pde)
  {
	     matrixK_.clear();
	     matrixK_.reinit(pde.GetBaseProblem().GetSpaceTimeHandler()->GetStateDoFHandler().GetDEALDoFHandler().n_dofs(), pde.GetBaseProblem().GetSpaceTimeHandler()->GetStateDoFHandler().GetDEALDoFHandler().n_dofs(),
	     		pde.GetBaseProblem().GetSpaceTimeHandler()->GetStateDoFHandler().GetDEALDoFHandler().max_couplings_between_dofs());
	     matrixM_.clear();
	     matrixM_.reinit(pde.GetBaseProblem().GetSpaceTimeHandler()->GetStateDoFHandler().GetDEALDoFHandler().n_dofs(), pde.GetBaseProblem().GetSpaceTimeHandler()->GetStateDoFHandler().GetDEALDoFHandler().n_dofs(),
	     		pde.GetBaseProblem().GetSpaceTimeHandler()->GetStateDoFHandler().GetDEALDoFHandler().max_couplings_between_dofs());
	     eigenvectors_.clear();
//	     eigenvectors_normalization_.clear();
	     eigenvalues_.clear();
  }

  /*******************************************************************************************/
  template <typename INTEGRATOR, typename VECTOR, typename MATRIX>
  template<typename PROBLEM>
  bool EigenvalueSolver_test<INTEGRATOR, VECTOR, MATRIX>
  ::EigenvalueSolve(PROBLEM &pde,
		  std::vector<double> &eigenvalues,
		  std::vector<StateVector<VECTOR>> &eigenfunctions,
		           bool /*apply_boundary_values*/,
                   bool force_matrix_build,
                   int /*priority*/,
                   std::string /*algo_level*/
				   )
  {

   bool build_matrix = force_matrix_build;
   if (force_matrix_build) {
  	   this->ReInit(pde);
	   integrator_.ComputeMatrix(pde,matrixK_);
	   integrator_.ComputeMassMatrix(pde,matrixM_);
   }
   eigenvectors_.clear();

   eigenvalues_.clear();

   eigenfunction_index_set = pde.GetBaseProblem().GetSpaceTimeHandler()->GetStateDoFHandler().GetDEALDoFHandler().locally_owned_dofs();
   eigenvectors_.resize(eigenvalues.size());
   for (unsigned int i = 0; i < eigenvectors_.size(); ++i) {
	   eigenvectors_[i].reinit(eigenfunction_index_set, MPI_COMM_WORLD);
    }
   	  eigenvalues_.resize(eigenvalues.size());
   	  int max_it = pde.GetBaseProblem().GetSpaceTimeHandler()->GetStateDoFHandler().GetDEALDoFHandler().n_dofs();

   	  EPS eps;
   	  EPSCreate(MPI_COMM_WORLD, &eps);

   	  MatShift(matrixK_, 1e-11);
   	  MatShift(matrixM_, 1e-11);

   	  EPSSetOperators(eps, matrixK_, matrixM_);

//   	  EPSSetType(eps,EPSKRYLOVSCHUR);
   	  ST st;
   	  EPSGetST(eps, &st);
   	  STSetType(st, STSINVERT);


   	  KSP ksp;
   	  STGetKSP(st, &ksp);
   	  KSPSetType(ksp, KSPPREONLY);

   	  PC pc;
   	  KSPGetPC(ksp, &pc);
   	  PCSetType(pc, PCLU);


//   	  PCFactorSetShiftType(pc,MAT_SHIFT_NONZERO);
//   	  PCFactorSetShiftAmount(pc,1e-10);


   	  PetscScalar target = 0.1;
   	  EPSSetTarget(eps, target);
	  EPSSetWhichEigenpairs(eps, EPS_TARGET_MAGNITUDE);

   	  EPSSetProblemType(eps, EPS_GNHEP);
   	  EPSSetDimensions(eps,2,PETSC_DECIDE,PETSC_DECIDE);

   	  EPSSolve(eps);
   	  PetscScalar eig;


   	  EPSGetEigenpair(eps, 0 , &eig, nullptr, eigenvectors_[0], nullptr); //realteil und imaginärteil (imaginärteil nullptr)

//
//  /* 	SolverControl linear_solver_control(pde.GetBaseProblem().GetSpaceTimeHandler()->GetStateDoFHandler().GetDEALDoFHandler().n_dofs(),
//                                 1e-12, false, false);
//   	 PETScWrappers::SolverCG linear_solver(linear_solver_control, MPI_COMM_WORLD);
//
//   	PETScWrappers::PreconditionNone preconditioner;
//
////   	PETScWrappers::PreconditionLU::AdditionalData data;
//   	//data kann noch gesetzt werden, default einstellungen
////   	 PETScWrappers::PreconditionLU preconditioner;
//
////
////   	 PETScWrappers::PreconditionBoomerAMG::AdditionalData data;
////   	 data.symmetric_operator = true;
////   	 PETScWrappers::PreconditionBoomerAMG preconditioner(MPI_COMM_WORLD, data);
//
//
//
//  	 linear_solver.initialize(preconditioner);
////
//      dealii::SolverControl solver_control (/*maxiter_,*/max_it, tol_);
//      SLEPcWrappers::SolverKrylovSchur eigensolver(solver_control, MPI_COMM_WORLD);
//
//      SLEPcWrappers::TransformationShift spectral_transformation(MPI_COMM_WORLD);
//      spectral_transformation.set_solver(linear_solver);
//      eigensolver.set_transformation(spectral_transformation);
//

   	  	for(unsigned int i = 0; i<eigenvalues_.size(); i++){
    	  		eigenvalues[i] = eig;
    	    	eigenfunctions[i].GetSpacialVector() = eigenvectors_[i];
   	      }

    return build_matrix;
  }

  template <typename INTEGRATOR, typename VECTOR, typename MATRIX>
   template<typename PROBLEM>
    void EigenvalueSolver_test<INTEGRATOR, VECTOR, MATRIX>::GetNormalizedVectorState(PROBLEM &pde,  StateVector<VECTOR> &stateeigenfunction){

	  state_for_normalization.clear();
	  state_for_normalization.reinit(eigenfunction_index_set, MPI_COMM_WORLD);
	  state_for_normalization = stateeigenfunction.GetSpacialVector();


		   PETScWrappers::MPI::Vector vec;
		  PetscScalar scalar_factor[1];
	      vec.reinit(eigenfunction_index_set, MPI_COMM_WORLD);
	      MatMult(matrixM_,state_for_normalization,vec);
	      VecDot(state_for_normalization,vec,scalar_factor);
	      scalar_factor[0] = sqrt(scalar_factor[0]);
	      state_for_normalization/= scalar_factor[0];
	      stateeigenfunction.GetSpacialVector() = state_for_normalization;

	   	 PetscScalar test[1];
	   	VecDot(state_for_normalization,state_for_normalization,test);

  }


  template <typename INTEGRATOR, typename VECTOR, typename MATRIX>
     template<typename PROBLEM>
      void EigenvalueSolver_test<INTEGRATOR, VECTOR, MATRIX>::GetNormalizedVectorAdjoint(PROBLEM &pde,  StateVector<VECTOR> &adjointeigenfunction, StateVector<VECTOR> &stateeigenfunction, double value){

	 state_for_normalization.clear();
  	 adjoint_for_normalization.clear();

  	 eigenfunction_index_set = pde.GetBaseProblem().GetSpaceTimeHandler()->GetStateDoFHandler().GetDEALDoFHandler().locally_owned_dofs();

  	state_for_normalization.reinit(eigenfunction_index_set, MPI_COMM_WORLD);
  	state_for_normalization = stateeigenfunction.GetSpacialVector();

  	adjoint_for_normalization.reinit(eigenfunction_index_set, MPI_COMM_WORLD);
  	adjoint_for_normalization =  adjointeigenfunction.GetSpacialVector();

	 PETScWrappers::MPI::Vector vec;
  		  PetscScalar scalar_factor[1];
  		   vec.reinit(eigenfunction_index_set, MPI_COMM_WORLD);
  		   MatMult(matrixM_,adjoint_for_normalization,vec); // = matrixM*adjoint_eigenfunctions[i];
  		   VecDot(state_for_normalization,vec,scalar_factor);//= scalar_product(vec,eigenfunctions[i]);
  		 adjoint_for_normalization/= (scalar_factor[0]);
 		 adjoint_for_normalization *= value;
 		adjointeigenfunction.GetSpacialVector() = adjoint_for_normalization;

  PETScWrappers::MPI::Vector vecTest;
  vecTest.reinit(eigenfunction_index_set, MPI_COMM_WORLD);
  MatMult(matrixM_,adjoint_for_normalization,vecTest);
	PetscScalar test[1];
  	VecDot(state_for_normalization,vecTest,test);

  }


  /*******************************************************************************************/
  template <typename INTEGRATOR,  typename VECTOR, typename MATRIX>
  INTEGRATOR &EigenvalueSolver_test<INTEGRATOR, VECTOR, MATRIX>
  ::GetIntegrator()
  {
    return integrator_;
  }

}
#endif





