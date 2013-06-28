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

#ifndef _QMRS_LINEAR_SOLVER_H_
#define _QMRS_LINEAR_SOLVER_H_

#include <lac/vector.h>
#include <lac/block_sparsity_pattern.h>
#include <lac/block_sparse_matrix.h>
#include <lac/compressed_simple_sparsity_pattern.h>
#include <lac/solver_qmrs.h>
#include <lac/precondition.h>
#include <lac/full_matrix.h>

#include <dofs/dof_tools.h>

#include <numerics/vector_tools.h>

#include <vector>

namespace DOpE
{
  /**
   * @class QMRSLinearSolverWithMatrix
   *
   * This class provides a linear solve for the nonlinear solvers of DOpE.
   * Here we interface to the CG-Solver of dealii
   *
   * @tparam <PRECONDITIONER>     The preconditioner class to be used with the solver
   * @tparam <SPARSITYPATTERN>    The sparsity pattern for the matrix
   * @tparam <MATRIX>             The matrix type that is used for the storage of the system_matrix
   * @tparam <VECTOR>             The vector type for the solution and righthandside data,
   *
   */

  template <typename PRECONDITIONER, typename SPARSITYPATTERN, typename MATRIX, typename VECTOR>
    class QMRSLinearSolverWithMatrix
  {
  public:
    QMRSLinearSolverWithMatrix(ParameterReader &param_reader);
    ~QMRSLinearSolverWithMatrix();

    static void declare_params(ParameterReader &param_reader);
    
    /**
       This Function should be called once after grid refinement, or changes in boundary values
       to  recompute sparsity patterns, and constraint matrices.
     */
    template<typename PROBLEM>
      void ReInit(PROBLEM& pde);

    /**
     * Solves the linear PDE in the form Ax = b using dealii::SolverQMRS
     *
     *
     * @tparam <PROBLEM>            The problem that we want to solve, this is passed on to the INTEGRATOR
     *                              to calculate the matrix.
     * @tparam <INTEGRATOR>         The integrator used to calculate the matrix A.
     * @param rhs                   Right Hand Side of the Equation, i.e., the VECTOR b.
     *                              Note that rhs is not const, this is because we need to apply 
     *                              the boundary values to this vector!
     * @param solution              The Approximate Solution of the Linear Equation.
     *                              It is assumed to be zero! Upon completion this VECTOR stores x
     * @param force_build_matrix    A boolean value, that indicates whether the Matrix
     *                              should be build by the linear solver in the first iteration.
     *				    The default is false, meaning that if we have no idea we don't
     *				    want to build a matrix.
     *
     */
    template<typename PROBLEM, typename INTEGRATOR>
      void Solve(PROBLEM& pde, INTEGRATOR& integr, VECTOR &rhs, VECTOR &solution, bool force_matrix_build=false);
     
  protected:

  private:
    SPARSITYPATTERN _sparsity_pattern;
    MATRIX _matrix;
 
    double _linear_global_tol, _linear_tol;
    int  _linear_maxiter;
  };

/*********************************Implementation************************************************/

template <typename PRECONDITIONER,typename SPARSITYPATTERN, typename MATRIX, typename VECTOR>
 void QMRSLinearSolverWithMatrix<PRECONDITIONER,SPARSITYPATTERN,MATRIX,VECTOR>::declare_params(ParameterReader &param_reader)
  {
    param_reader.SetSubsection("qmrslinearsolver_withmatrix parameters");
    param_reader.declare_entry("linear_global_tol", "1.e-16",Patterns::Double(0),"global tolerance for the cg iteration");
    param_reader.declare_entry("linear_tol", "1.e-12",Patterns::Double(0),"relative tolerance for the cg iteration");
    param_reader.declare_entry("linear_maxiter", "1000",Patterns::Integer(0),"maximal number of cg steps");
  }
/******************************************************/

  template <typename PRECONDITIONER,typename SPARSITYPATTERN, typename MATRIX, typename VECTOR>
    QMRSLinearSolverWithMatrix<PRECONDITIONER,SPARSITYPATTERN,MATRIX,VECTOR>
    ::QMRSLinearSolverWithMatrix(ParameterReader &param_reader) 
{
  param_reader.SetSubsection("qmrslinearsolver_withmatrix parameters");
  _linear_global_tol = param_reader.get_double ("linear_global_tol");
  _linear_tol        = param_reader.get_double ("linear_tol"); 
  _linear_maxiter    = param_reader.get_integer ("linear_maxiter"); 

}

/******************************************************/

template <typename PRECONDITIONER,typename SPARSITYPATTERN, typename MATRIX, typename VECTOR>
  QMRSLinearSolverWithMatrix<PRECONDITIONER,SPARSITYPATTERN,MATRIX,VECTOR>::~QMRSLinearSolverWithMatrix()
{
}

/******************************************************/

template <typename PRECONDITIONER,typename SPARSITYPATTERN, typename MATRIX, typename VECTOR>
  template<typename PROBLEM>
  void  QMRSLinearSolverWithMatrix<PRECONDITIONER,SPARSITYPATTERN,MATRIX,VECTOR>::ReInit(PROBLEM& pde)
{
  _matrix.clear();
  pde.ComputeSparsityPattern(_sparsity_pattern);
  _matrix.reinit(_sparsity_pattern);
}

/******************************************************/
template <typename PRECONDITIONER,typename SPARSITYPATTERN, typename MATRIX, typename VECTOR>
  template<typename PROBLEM, typename INTEGRATOR>
  void QMRSLinearSolverWithMatrix<PRECONDITIONER,SPARSITYPATTERN,MATRIX,VECTOR>::Solve(PROBLEM& pde, 
										     INTEGRATOR& integr,
										     VECTOR &rhs, 
										     VECTOR &solution, 
										     bool force_matrix_build)
{
  if(force_matrix_build)
  { 
    integr.ComputeMatrix (pde,_matrix);
  }
 
  integr.ApplyNewtonBoundaryValues(pde,_matrix,rhs,solution);

  dealii::SolverControl solver_control (_linear_maxiter, _linear_global_tol,false,true);//letzte Arg = false!
  dealii::SolverQMRS<VECTOR> qmres (solver_control);
  PRECONDITIONER precondition;
  precondition.initialize(_matrix);
  qmres.solve (_matrix, solution, rhs,
	    precondition);
  VECTOR tmp(solution.size());
  _matrix.vmult(tmp,solution);
  tmp-= rhs;
  std::cout<<"XXX"<<tmp.linfty_norm()<<" ---- "<<tmp.l2_norm()<<std::endl;
  pde.GetDoFConstraints().distribute(solution);
}
/******************************************************/



}
#endif
