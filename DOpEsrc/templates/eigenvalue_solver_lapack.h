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

#ifndef EIGENVALUE_SOLVER_LAPACK_H_
#define EIGENVALUE_SOLVER_LAPACK_H_

#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/slepc_solver.h>

#include <include/parameterreader.h>
#include <deal.II/lac/petsc_parallel_vector.h>

#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>


namespace DOpE
{

  template <typename INTEGRATOR, typename VECTOR, typename MATRIX>
  class EigenvalueSolver_LAPACK
  {
  public:
	  EigenvalueSolver_LAPACK( INTEGRATOR &integrator, ParameterReader &param_reader);
    ~EigenvalueSolver_LAPACK();

    static void declare_params(ParameterReader &param_reader);

    template<typename PROBLEM>
    void ReInit(PROBLEM &pde);

    template<typename PROBLEM>
    void GetNormalizedVectorState(PROBLEM &pde, std::vector<StateVector<VECTOR>> &stateeigenfunctions);
    template<typename PROBLEM>
        void GetNormalizedVectorAdjoint(PROBLEM &pde, std::vector<double> &eigenvalues, std::vector<StateVector<VECTOR>> &adjeigenfunctions,std::vector<StateVector<VECTOR>> &stateeigenfunctions);

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
    std::vector<PETScWrappers::MPI::Vector> eigenvectors_, eigenvectors_normalization_;
    std::vector<double> eigenvalues_;
    bool build_matrix_ = true;

    int maxiter_;
    double target_eigenvalue_ , tol_;
  };

  /**********************************Implementation*******************************************/

  template <typename INTEGRATOR, typename VECTOR, typename MATRIX>
  void EigenvalueSolver_LAPACK<INTEGRATOR,VECTOR , MATRIX>
  ::declare_params(ParameterReader &param_reader)
  {
	  param_reader.SetSubsection("eigenvalue_solver parameters");
	  param_reader.declare_entry("tol", "0.00001", Patterns::Double(0),"tolerance");
	  param_reader.declare_entry("maxiter", "1000", Patterns::Integer(0),"max iterations");
	  param_reader.declare_entry("target_eigenvalue", "0.001", Patterns::Double(0),"target eigenvalue");

  }

  /*******************************************************************************************/
  template <typename INTEGRATOR, typename VECTOR, typename MATRIX>
  EigenvalueSolver_LAPACK<INTEGRATOR,VECTOR, MATRIX>
  ::EigenvalueSolver_LAPACK(INTEGRATOR &integrator, ParameterReader &param_reader)
    : integrator_(integrator)
  {
	    param_reader.SetSubsection("eigenvalue_solver parameters");
	    tol_ = param_reader.get_double ("tol");
	    maxiter_ = param_reader.get_integer ("maxiter");
	    target_eigenvalue_ = param_reader.get_double ("target_eigenvalue");
  }

  /*******************************************************************************************/

  template <typename INTEGRATOR, typename VECTOR,  typename MATRIX>
  EigenvalueSolver_LAPACK<INTEGRATOR,VECTOR, MATRIX>
  ::~EigenvalueSolver_LAPACK()
  {
  }

  /*******************************************************************************************/
  template <typename INTEGRATOR, typename VECTOR, typename MATRIX>
  template<typename PROBLEM>
  void EigenvalueSolver_LAPACK<INTEGRATOR, VECTOR, MATRIX>
  ::ReInit(PROBLEM &pde)
  {
	     matrixK_.clear();
	     matrixK_.reinit(pde.GetBaseProblem().GetSpaceTimeHandler()->GetStateDoFHandler().GetDEALDoFHandler().n_dofs(), pde.GetBaseProblem().GetSpaceTimeHandler()->GetStateDoFHandler().GetDEALDoFHandler().n_dofs(),
	     		pde.GetBaseProblem().GetSpaceTimeHandler()->GetStateDoFHandler().GetDEALDoFHandler().max_couplings_between_dofs());
	     matrixM_.clear();
	     matrixM_.reinit(pde.GetBaseProblem().GetSpaceTimeHandler()->GetStateDoFHandler().GetDEALDoFHandler().n_dofs(), pde.GetBaseProblem().GetSpaceTimeHandler()->GetStateDoFHandler().GetDEALDoFHandler().n_dofs(),
	     		pde.GetBaseProblem().GetSpaceTimeHandler()->GetStateDoFHandler().GetDEALDoFHandler().max_couplings_between_dofs());
	     eigenvectors_.clear();
	     eigenvectors_normalization_.clear();
	     eigenvalues_.clear();
  }

  /*******************************************************************************************/
  template <typename INTEGRATOR, typename VECTOR, typename MATRIX>
  template<typename PROBLEM>
  bool EigenvalueSolver_LAPACK<INTEGRATOR, VECTOR, MATRIX>
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
      dealii::SolverControl solver_control (/*maxiter_,*/max_it, tol_);
      SLEPcWrappers::SolverLAPACK eigensolver(solver_control, MPI_COMM_WORLD);
      eigensolver.set_which_eigenpairs(EPS_TARGET_MAGNITUDE);
      eigensolver.set_target_eigenvalue(target_eigenvalue_);
      eigensolver.solve(matrixK_, matrixM_, eigenvalues_, eigenvectors_, eigenvalues_.size());


   	  	for(unsigned int i = 0; i<eigenvalues_.size(); i++){
    	  		eigenvalues[i] = eigenvalues_[i];
    	    	eigenfunctions[i].GetSpacialVector() = eigenvectors_[i];
   	      }

    return build_matrix;
  }

  template <typename INTEGRATOR, typename VECTOR, typename MATRIX>
   template<typename PROBLEM>
    void EigenvalueSolver_LAPACK<INTEGRATOR, VECTOR, MATRIX>::GetNormalizedVectorState(PROBLEM &pde,  std::vector<StateVector<VECTOR>> &stateeigenfunctions){

	  integrator_.ComputeMassMatrix(pde,matrixM_);

	    for(unsigned int i = 0; i<eigenvectors_.size(); i++){
		   PETScWrappers::MPI::Vector vec;
		  PetscScalar scalar_factor[1];
	      vec.reinit(eigenfunction_index_set, MPI_COMM_WORLD);
	      MatMult(matrixM_,eigenvectors_[i],vec);
	      VecDot(vec,eigenvectors_[i],scalar_factor);
	      scalar_factor[0] = sqrt(scalar_factor[0]);
	      eigenvectors_[i]/= scalar_factor[0];
	      stateeigenfunctions[i].GetSpacialVector() = eigenvectors_[i];
	    }
  }


  template <typename INTEGRATOR, typename VECTOR, typename MATRIX>
     template<typename PROBLEM>
      void EigenvalueSolver_LAPACK<INTEGRATOR, VECTOR, MATRIX>::GetNormalizedVectorAdjoint(PROBLEM &pde, std::vector<double> &eigenvalues, std::vector<StateVector<VECTOR>> &adjointeigenfunctions, std::vector<StateVector<VECTOR>> &stateeigenfunctions){
	 integrator_.ComputeMassMatrix(pde,matrixM_);

	 eigenvectors_normalization_.clear();
  	 eigenvectors_normalization_.resize(eigenvalues.size());
  	 eigenfunction_index_set = pde.GetBaseProblem().GetSpaceTimeHandler()->GetStateDoFHandler().GetDEALDoFHandler().locally_owned_dofs();

  	   for (unsigned int i = 0; i < eigenvectors_.size(); ++i) {
  	 	  eigenvectors_normalization_[i].reinit(eigenfunction_index_set, MPI_COMM_WORLD);
  	 	  eigenvectors_normalization_[i] = stateeigenfunctions[i].GetSpacialVector();
  	  }

  	   for(unsigned int i = 0; i<eigenvalues_.size(); i++){
  		 PETScWrappers::MPI::Vector vec;
  		  PetscScalar scalar_factor[1];
  		   vec.reinit(eigenfunction_index_set, MPI_COMM_WORLD);
  		   MatMult(matrixM_,eigenvectors_[i],vec); // = matrixM*adjoint_eigenfunctions[i];
  		   VecDot(vec,eigenvectors_normalization_[i],scalar_factor);//= scalar_product(vec,eigenfunctions[i]);
  		   eigenvectors_[i]/= scalar_factor[0];
  		   adjointeigenfunctions[i].GetSpacialVector() = eigenvectors_[i];
      	  }
  }



  /*******************************************************************************************/
  template <typename INTEGRATOR,  typename VECTOR, typename MATRIX>
  INTEGRATOR &EigenvalueSolver_LAPACK<INTEGRATOR, VECTOR, MATRIX>
  ::GetIntegrator()
  {
    return integrator_;
  }

}
#endif





