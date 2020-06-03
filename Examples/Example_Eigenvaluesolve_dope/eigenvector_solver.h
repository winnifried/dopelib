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

#ifndef EIGENVECTOR_SOLVER_H_
#define EIGENVECTOR_SOLVER_H_

#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/slepc_solver.h>

#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>

#include <include/parameterreader.h>
#include <deal.II/lac/petsc_parallel_vector.h>



namespace DOpE
{
  /**
   * A solver class to compute eigenvalue problems
   *
   * @tparam <INTEGRATOR>          Integration routines to compute domain-, face-, and right-hand side values.
   * @tparam <VECTOR>              A template class for arbitrary vectors which are given to the
   *                               FS scheme and where the solution is stored in.
   * @tparam <EIGENVALUES>
   * @tparam <EIGENVECTORS>
   * @tparam <MATRIX>
   * @tparam <SPARSITYPATTERN>
   */

  template <typename INTEGRATOR, typename VECTOR,typename EIGENVALUES, typename EIGENVECTORS, typename MATRIX,typename SPARSITYPATTERN>
  class EigenvectorSolver
  {
  public:
	  EigenvectorSolver(INTEGRATOR &integrator, ParameterReader &param_reader);
    ~EigenvectorSolver();

    static void declare_params(ParameterReader &param_reader);

    template<typename PROBLEM>
    void ReInit(PROBLEM &pde);

    template<typename PROBLEM>
    bool NonlinearSolve(PROBLEM &pde, VECTOR &solution, EIGENVALUES &eigenvalues, EIGENVECTORS &eigenfunctions, bool apply_boundary_values=true,
                        bool force_matrix_build=false,
                        int priority = 5, std::string algo_level = "\t\t ");

    template<typename PROBLEM>
    void EigenvalueDerivativeSolve(PROBLEM &pde, double eigenvalue, PETScWrappers::MPI::Vector &eigenvec, double eigenval_derivative);



    inline INTEGRATOR &GetIntegrator();

  private:
    INTEGRATOR &integrator_;
    SPARSITYPATTERN sparsity_pattern_;
    MATRIX matrixK_, matrixM_;

//    std::vector<PETScWrappers::MPI::Vector> eigenfunctions;
//    std::vector<double> eigenvalues;
    IndexSet eigenfunction_index_set;
    bool build_matrix_;
    double linear_global_tol_= 0.000001, linear_tol_ = 0.00001;
    int  linear_maxiter_=1000,n_eigenval_;
  };

  /**********************************Implementation*******************************************/

  template <typename INTEGRATOR, typename VECTOR,typename EIGENVALUES, typename EIGENVECTORS, typename MATRIX, typename SPARSITYPATTERN>
  void EigenvectorSolver<INTEGRATOR,VECTOR,EIGENVALUES, EIGENVECTORS, MATRIX, SPARSITYPATTERN>
  ::declare_params(ParameterReader &/*param_reader*/)
  {

  }

  /*******************************************************************************************/

  template <typename INTEGRATOR, typename VECTOR, typename EIGENVALUES, typename EIGENVECTORS, typename MATRIX, typename SPARSITYPATTERN>
  EigenvectorSolver<INTEGRATOR,VECTOR,EIGENVALUES, EIGENVECTORS, MATRIX, SPARSITYPATTERN>
  ::EigenvectorSolver(INTEGRATOR &integrator, ParameterReader &/*param_reader*/)
    :  integrator_(integrator)
  {


  }

  /*******************************************************************************************/

  template <typename INTEGRATOR, typename VECTOR, typename EIGENVALUES, typename EIGENVECTORS, typename MATRIX, typename SPARSITYPATTERN>
  EigenvectorSolver<INTEGRATOR,VECTOR, EIGENVALUES, EIGENVECTORS, MATRIX, SPARSITYPATTERN>
  ::~EigenvectorSolver()
  {
  }

  /*******************************************************************************************/
  template <typename INTEGRATOR, typename VECTOR, typename EIGENVALUES, typename EIGENVECTORS, typename MATRIX, typename SPARSITYPATTERN>
  template<typename PROBLEM>
  void EigenvectorSolver<INTEGRATOR, VECTOR, EIGENVALUES, EIGENVECTORS, MATRIX, SPARSITYPATTERN>
  ::ReInit(PROBLEM &pde)
  {
	   	 matrixM_.clear();
	     matrixM_.reinit(pde.GetBaseProblem().GetSpaceTimeHandler()->GetStateDoFHandler().GetDEALDoFHandler().n_dofs(), pde.GetBaseProblem().GetSpaceTimeHandler()->GetStateDoFHandler().GetDEALDoFHandler().n_dofs(),
	     		pde.GetBaseProblem().GetSpaceTimeHandler()->GetStateDoFHandler().GetDEALDoFHandler().max_couplings_between_dofs());
	     matrixK_.clear();
	     matrixK_.reinit(pde.GetBaseProblem().GetSpaceTimeHandler()->GetStateDoFHandler().GetDEALDoFHandler().n_dofs(), pde.GetBaseProblem().GetSpaceTimeHandler()->GetStateDoFHandler().GetDEALDoFHandler().n_dofs(),
	     		pde.GetBaseProblem().GetSpaceTimeHandler()->GetStateDoFHandler().GetDEALDoFHandler().max_couplings_between_dofs());

  }

  /*******************************************************************************************/
  template <typename INTEGRATOR, typename VECTOR, typename EIGENVALUES, typename EIGENVECTORS, typename MATRIX, typename SPARSITYPATTERN>
  template<typename PROBLEM>
  bool EigenvectorSolver<INTEGRATOR, VECTOR, EIGENVALUES, EIGENVECTORS, MATRIX, SPARSITYPATTERN>
  ::NonlinearSolve(PROBLEM &pde, VECTOR &solution,
		  EIGENVALUES &eigenvalues,
		  EIGENVECTORS &eigenfunctions,
		           bool /*apply_boundary_values*/,
                   bool force_matrix_build,
                   int /*priority*/,
                   std::string /*algo_level*//*, int n_eigenval*/)
  {

   bool build_matrix = force_matrix_build;

   if (force_matrix_build)
        {
	   integrator_.ComputeMatrix(pde,matrixK_);
	   integrator_.ComputeMassMatrix(pde,matrixM_);
        }

      eigenfunction_index_set = pde.GetBaseProblem().GetSpaceTimeHandler()->GetStateDoFHandler().GetDEALDoFHandler().locally_owned_dofs();
      for (int i = 0; i < eigenfunctions.size(); ++i) {
       	eigenfunctions[i].reinit(eigenfunction_index_set, MPI_COMM_WORLD);
      }
      linear_maxiter_ = pde.GetBaseProblem().GetSpaceTimeHandler()->GetStateDoFHandler().GetDEALDoFHandler().n_dofs();
      dealii::SolverControl solver_control (linear_maxiter_, 1e-5/*,false,false*/);
          SLEPcWrappers::SolverLAPACK/*SolverJacobiDavidson*/eigensolver(solver_control, MPI_COMM_WORLD);
          eigensolver.set_which_eigenpairs(EPS_TARGET_MAGNITUDE);
          eigensolver.set_target_eigenvalue(0.001);

          eigensolver.solve(matrixK_, matrixM_, eigenvalues, eigenfunctions, eigenvalues.size());
              	for (unsigned int i = 0; i < eigenfunctions.size(); ++i) {
             		eigenfunctions[i] /= eigenfunctions[i].linfty_norm();
             	}


    return build_matrix;
  }

  /*******************************************************************************************/
   template <typename INTEGRATOR, typename VECTOR, typename EIGENVALUES, typename EIGENVECTORS, typename MATRIX, typename SPARSITYPATTERN>
   template<typename PROBLEM>
   void EigenvectorSolver<INTEGRATOR, VECTOR, EIGENVALUES, EIGENVECTORS, MATRIX, SPARSITYPATTERN>
   ::EigenvalueDerivativeSolve(PROBLEM &pde,  double eigenvalue, PETScWrappers::MPI::Vector &eigenvec, double eigenval_derivative)
   {

	   std::cout << "in Eigenvalue_Derivative_Solve" << std::endl;
	   integrator_.ComputeEigenvalueDerivative(pde, eigenvalue, eigenvec, eigenval_derivative);
		std::cout << "derivative in eigenvalue_solver.h" << eigenval_derivative << std::endl;

//
//          	integrator_.ComputeEigenpair(pde, eigenvalues[0], eigenfunctions[0], derivative_eigenvalues, derivative_eigenfunction);
//
// //  //       	solution = eigenfunctions.at(0);//[0];
// //  //    	solution = eigenfunctions.[0];
//        //  	pde.GetDoFConstraints().distribute(solution);
//
// //    GetIntegrator().DeleteDomainData("last_newton_solution");

   }

  /*******************************************************************************************/
  template <typename INTEGRATOR,  typename VECTOR, typename EIGENVALUES, typename EIGENVECTORS,typename MATRIX, typename SPARSITYPATTERN>
  INTEGRATOR &EigenvectorSolver<INTEGRATOR, VECTOR, EIGENVALUES, EIGENVECTORS, MATRIX, SPARSITYPATTERN>
  ::GetIntegrator()
  {
    return integrator_;
  }

  /*******************************************************************************************/

}
#endif





