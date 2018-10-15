/**
*
* Copyright (C) 2012-2014 by the DOpElib authors
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

#ifndef TRILINOS_DIRECT_LINEAR_SOLVER_H_
#define TRILINOS_DIRECT_LINEAR_SOLVER_H_

#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_sparse_matrix.h>
#if DEAL_II_VERSION_GTE(8,5,0)
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#else
#include <deal.II/lac/compressed_simple_sparsity_pattern.h>
#endif
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>

#include <vector>

#include <include/parameterreader.h>

namespace DOpE
{
  /**
   * @class TrilinosDirectLinearSolverWithMatrix
   *
   * This class provides a linear solve for the nonlinear solvers of DOpE.
   * Here we interface to the  TrilinosWrappers::SolverDirect provided via dealii
   * The use of this function requires that dealii is compiled with Trilinos
   *
   * @tparam <SPARSITYPATTERN>    The sparsity pattern for the matrix
   * @tparam <VECTOR>             The vector type for the solution and righthandside data,
   *
   */

  template <typename SPARSITYPATTERN, typename MATRIX, typename VECTOR>
  class TrilinosDirectLinearSolverWithMatrix
  {
  public:
    TrilinosDirectLinearSolverWithMatrix(ParameterReader &param_reader);
    ~TrilinosDirectLinearSolverWithMatrix();

    static void declare_params(ParameterReader &param_reader);

    /**
       This Function should be called once after grid refinement, or changes in boundary values
       to  recompute sparsity patterns, and constraint matrices.
     */
    template<typename PROBLEM>
    void ReInit(PROBLEM &pde);

    /**
     * Solves the linear PDE in the form Ax = b using direct solvers from Trilinos
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
     *            The default is false, meaning that if we have no idea we don't
     *            want to build a matrix.
     *
     */
    template<typename PROBLEM, typename INTEGRATOR>
    void Solve(PROBLEM &pde, INTEGRATOR &integr, VECTOR &rhs, VECTOR &solution, bool force_matrix_build=false);

  protected:

  private:
    SPARSITYPATTERN sparsity_pattern_;
    MATRIX matrix_;
#ifdef DOPELIB_WITH_TRILINOS
    TrilinosWrappers::SparseMatrix tril_matrix_;
#endif
    std::string trilinos_solver_;
  };

  /*********************************Implementation************************************************/

  template <typename SPARSITYPATTERN, typename MATRIX, typename VECTOR>
  void TrilinosDirectLinearSolverWithMatrix<SPARSITYPATTERN,MATRIX,VECTOR>::declare_params(ParameterReader &param_reader)
  {
    param_reader.SetSubsection("trilinos direct parameters");
    param_reader.declare_entry("direct_solver","Amesos_Klu",Patterns::Selection("Amesos_Klu|Amesos_Lapack|Amesos_Scalapack|Amesos_Umfpack|Amesos_Pardiso|Amesos_Taucs|Amesos_Superlu|Amesos_Superludist|Amesos_Dscpack|Amesos_Mumps"),"Direct Solver to be used by trilinos.");

  }

  /******************************************************/

  template <typename SPARSITYPATTERN, typename MATRIX, typename VECTOR>
  TrilinosDirectLinearSolverWithMatrix<SPARSITYPATTERN,MATRIX,VECTOR>::TrilinosDirectLinearSolverWithMatrix(
    ParameterReader &param_reader)
  {
#ifndef DOPELIB_WITH_TRILINOS
    throw DOpEException("To use this algorithm you need to deal.II compiled with Trilinos!","TrilinosDirectLinearSolverWithMatrix");
#endif
    param_reader.SetSubsection("trilinos direct parameters");
    trilinos_solver_ = param_reader.get_string("direct_solver");
  }

  /******************************************************/

  template <typename SPARSITYPATTERN, typename MATRIX, typename VECTOR>
  TrilinosDirectLinearSolverWithMatrix<SPARSITYPATTERN,MATRIX,VECTOR>::~TrilinosDirectLinearSolverWithMatrix()
  {
  }

  /******************************************************/

  template <typename SPARSITYPATTERN, typename MATRIX, typename VECTOR>
  template<typename PROBLEM>
  void  TrilinosDirectLinearSolverWithMatrix<SPARSITYPATTERN,MATRIX,VECTOR>::ReInit(PROBLEM &pde)
  {
    matrix_.clear();
    pde.ComputeSparsityPattern(sparsity_pattern_);
    matrix_.reinit(sparsity_pattern_);
  }

  /******************************************************/

  template <typename SPARSITYPATTERN, typename MATRIX, typename VECTOR>
  template<typename PROBLEM, typename INTEGRATOR>
  void TrilinosDirectLinearSolverWithMatrix<SPARSITYPATTERN,MATRIX,VECTOR>::Solve(PROBLEM &pde,
      INTEGRATOR &integr,
      VECTOR &rhs,
      VECTOR &solution,
      bool force_matrix_build)
  {
#ifdef DOPELIB_WITH_TRILINOS
    if (force_matrix_build)
      {
        integr.ComputeMatrix (pde,matrix_);
      }


    if (force_matrix_build)
      {
        tril_matrix_.reinit(matrix_);
      }

    SolverControl solver_control (1,0);
    TrilinosWrappers::SolverDirect::AdditionalData data (false);
    TrilinosWrappers::SolverDirect direct (solver_control, data);
    direct.solve (tril_matrix_, solution, rhs);

    pde.GetDoFConstraints().distribute(solution);
#else
    throw DOpEException("To use this algorithm you need to deal.II compiled with Trilinos!","TrilinosDirectLinearSolverWithMatrix::Solve");
#endif
  }


}
#endif
