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

#ifndef _DIRECT_LINEAR_SOLVER_H_
#define _DIRECT_LINEAR_SOLVER_H_

#include <lac/vector.h>
#include <lac/block_sparsity_pattern.h>
#include <lac/block_sparse_matrix.h>
#include <lac/compressed_simple_sparsity_pattern.h>
#include <lac/solver_cg.h>
#include <lac/precondition.h>
#include <lac/full_matrix.h>
#include <lac/sparse_direct.h>

#include <dofs/dof_tools.h>

#include <numerics/vectors.h>

#include <vector>

namespace DOpE
{

  template <typename SPARSITYPATTERN, typename MATRIX, typename VECTOR,int dim>
    class DirectLinearSolverWithMatrix
  {
  public:
    DirectLinearSolverWithMatrix(ParameterReader &param_reader);
    ~DirectLinearSolverWithMatrix();
   
    static void declare_params(ParameterReader &param_reader);
 
    /**
       This Function should be called once after grid refinement, or changes in boundary values
       to  recompute sparsity patterns, and constraint matrices.
     */
    template<typename PROBLEM>
      void ReInit(PROBLEM& pde);

    /**
     * Solves the nonlinear PDE described by the PDEPROBLEM given initialy to the constructor 
     * using UMFPACK
     *
     * @param rhs                   Right Hand Side of the Equation.
     * @param solution              The Approximate Solution of the Linear Equation.
     *                              It is assumed to be zero!
     * @param force_build_matrix    A boolean value, that indicates whether the Matrix
     *                              should be build by the linear solver in the first iteration.
     *				    The default is false, meaning that if we have no idea we don't
     *				    want to build a matrix.
     *
     *
     */
    template<typename PROBLEM, typename INTEGRATOR>
      void Solve(PROBLEM& pde, INTEGRATOR& integr, VECTOR &rhs, VECTOR &solution, bool force_matrix_build=false);
     
  protected:

  private:
    SPARSITYPATTERN _sparsity_pattern;
    MATRIX _matrix;

    dealii::SparseDirectUMFPACK* _A_direct;

    double _linear_global_tol, _linear_tol;
    int  _linear_maxiter;
  };

/*********************************Implementation************************************************/
 
  template <typename SPARSITYPATTERN, typename MATRIX, typename VECTOR,int dim>
 void DirectLinearSolverWithMatrix<SPARSITYPATTERN,MATRIX,VECTOR, dim>::declare_params(ParameterReader &param_reader __attribute__((unused))) 
  {
  }

  /******************************************************/

  template <typename SPARSITYPATTERN, typename MATRIX, typename VECTOR,int dim>
    DirectLinearSolverWithMatrix<SPARSITYPATTERN,MATRIX,VECTOR, dim>::DirectLinearSolverWithMatrix(
      ParameterReader &param_reader __attribute__((unused))) 
  {
    _A_direct = NULL;
  }

/******************************************************/

template <typename SPARSITYPATTERN, typename MATRIX, typename VECTOR,int dim>
 DirectLinearSolverWithMatrix<SPARSITYPATTERN,MATRIX,VECTOR, dim>::~DirectLinearSolverWithMatrix()
{
  if(_A_direct != NULL)
  {
    delete _A_direct;
  }
}

/******************************************************/

template <typename SPARSITYPATTERN, typename MATRIX, typename VECTOR,int dim>
  template<typename PROBLEM>
  void  DirectLinearSolverWithMatrix<SPARSITYPATTERN,MATRIX,VECTOR, dim>::ReInit(PROBLEM& pde)
{
  _matrix.clear();
  pde.ComputeSparsityPattern(_sparsity_pattern);
  _matrix.reinit(_sparsity_pattern);
  
  if(_A_direct != NULL)
  {
    delete _A_direct;
    _A_direct= NULL;
  }
}

/******************************************************/

template <typename SPARSITYPATTERN, typename MATRIX, typename VECTOR,int dim>
  template<typename PROBLEM, typename INTEGRATOR>
  void DirectLinearSolverWithMatrix<SPARSITYPATTERN,MATRIX,VECTOR, dim>::Solve(PROBLEM& pde,
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
  
  if(_A_direct == NULL)
  {
    _A_direct = new dealii::SparseDirectUMFPACK;
    _A_direct->initialize(_matrix);
  }
  else if(force_matrix_build)
  {
     _A_direct->factorize(_matrix);
  }
 
  dealii::Vector<double> sol;
  sol = rhs;
  _A_direct->solve(sol);
  solution = sol;

  pde.GetDoFConstraints().distribute(solution);

}


}
#endif
