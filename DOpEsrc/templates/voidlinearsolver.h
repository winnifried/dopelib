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

#ifndef VOID_LINEAR_SOLVER_H_
#define VOID_LINEAR_SOLVER_H_

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

#include <vector>

namespace DOpE
{
  /**
   * @class VoidLinearSolver
   *
   * This class provides a linear solve for the nonlinear solvers of DOpE.
   * This one is a dummy implementation for certain cases where we know that
   * we invert an identity matrix!
   *
   * @tparam <VECTOR>             The vector type for the solution and righthandside data,
   *
   */
  template <typename VECTOR>
  class VoidLinearSolver
  {
  public:
    VoidLinearSolver(ParameterReader &param_reader);
    ~VoidLinearSolver();

    static void declare_params(ParameterReader &param_reader);

    /**
       This Function should be called once after grid refinement, or changes in boundary values
       to  recompute sparsity patterns, and constraint matrices.
     */
    template<typename PROBLEM>
    void ReInit(PROBLEM &pde);

    /**
     *  Copys the Rhs to the Solution Vector, als other params are ignored!
     *
     *
     * @param rhs                   Right Hand Side of the Equation.
     * @param solution              The Approximate Solution of the Linear Equation.
     *                              It is assumed to be zero!
     * @param force_build_matrix    A boolean value, that indicates whether the Matrix
     *                              should be build by the linear solver in the first iteration.
     *            The default is false, meaning that if we have no idea we don't
     *            want to build a matrix.
     *
     *
     */
    template<typename PROBLEM, typename INTEGRATOR>
    void Solve(PROBLEM &pde, INTEGRATOR &integr, VECTOR &rhs, VECTOR &solution, bool force_matrix_build=false);

  protected:

  private:

  };

  /*********************************Implementation************************************************/

  template <typename VECTOR>
  void VoidLinearSolver<VECTOR>::declare_params(ParameterReader &/*param_reader*/)
  {
  }

  /******************************************************/

  template <typename VECTOR>
  VoidLinearSolver<VECTOR>::VoidLinearSolver(ParameterReader &/*param_reader*/)
  {
  }

  /******************************************************/

  template <typename VECTOR>
  VoidLinearSolver<VECTOR>::~VoidLinearSolver()
  {
  }

  /******************************************************/

  template <typename VECTOR>
  template<typename PROBLEM>
  void  VoidLinearSolver<VECTOR>::ReInit(PROBLEM & /*pde*/)
  {

  }

  /******************************************************/

  template <typename VECTOR>
  template<typename PROBLEM, typename INTEGRATOR>
  void VoidLinearSolver<VECTOR>::Solve(PROBLEM & /*pde*/, INTEGRATOR & /*integr*/, VECTOR &rhs, VECTOR &solution, bool /*force_matrix_build*/)
  {
    solution = rhs;
  }


}
#endif
