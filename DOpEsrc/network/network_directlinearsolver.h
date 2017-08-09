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

#ifndef NETWORK_DIRECT_LINEAR_SOLVER_H_
#define NETWORK_DIRECT_LINEAR_SOLVER_H_

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

#include <include/parameterreader.h>

namespace DOpE
{
  namespace Networks
  {
    /**
     * @class DirectLinearSolverWithMatrix
     *
     * This class provides a linear solve for the nonlinear solvers of DOpE.
     *
     */

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
      void ReInit(PROBLEM &pde);

      /**
       * Solves the linear PDE in the form Ax = b using direct solver
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
      void Solve(PROBLEM &pde, INTEGRATOR &integr, BlockVector<double> &rhs, BlockVector<double> &solution, bool force_matrix_build=false);

    protected:

    private:
      dealii::BlockSparsityPattern sparsity_pattern_;
      dealii::BlockSparseMatrix<double> matrix_;

      dealii::SparseDirectUMFPACK *A_direct_;

      MethodOfLines_Network_SpaceTimeHandler<FESystem,DoFHandler,BlockVector<double>,0,1> *sth_;
      unsigned int n_pipes_;
    };

    /*********************************Implementation************************************************/

    void DirectLinearSolverWithMatrix::declare_params(ParameterReader &/*param_reader*/)
    {
    }

    /******************************************************/

    DirectLinearSolverWithMatrix::DirectLinearSolverWithMatrix(
      ParameterReader &/*param_reader*/)
    {
      A_direct_ = NULL;
    }

    /******************************************************/

    DirectLinearSolverWithMatrix::~DirectLinearSolverWithMatrix()
    {
      if (A_direct_ != NULL)
        {
          delete A_direct_;
        }
    }

    /******************************************************/

    template<typename PROBLEM>
    void  DirectLinearSolverWithMatrix::ReInit(PROBLEM &pde)
    {
      sth_ = dynamic_cast<MethodOfLines_Network_SpaceTimeHandler<FESystem,DoFHandler,BlockVector<double>,0,1>*>(pde.GetBaseProblem().GetSpaceTimeHandler());
      if (sth_ == NULL)
        {
          throw DOpEException("Using Networks::DirectLinearSolverWithMatrix with wrong SpaceTimeHandler","DirectLinearSolverWithMatrix::ReInit");
        }
      n_pipes_ =  sth_->GetNPipes();

      matrix_.clear();
      pde.ComputeSparsityPattern(sparsity_pattern_);
      matrix_.reinit(sparsity_pattern_);

      if (A_direct_ != NULL)
        {
          delete A_direct_;
          A_direct_= NULL;
        }
    }

    /******************************************************/

    template<typename PROBLEM, typename INTEGRATOR>
    void DirectLinearSolverWithMatrix::Solve(PROBLEM &pde,
                                             INTEGRATOR &integr,
                                             BlockVector<double> &rhs,
                                             BlockVector<double> &solution,
                                             bool force_matrix_build)
    {
      if (force_matrix_build)
        {
          integr.ComputeMatrix (pde,matrix_);
        }

      if (A_direct_ == NULL)
        {
          A_direct_ = new dealii::SparseDirectUMFPACK;
          A_direct_->initialize(matrix_);
        }
      else if (force_matrix_build)
        {
          A_direct_->factorize(matrix_);
        }

      dealii::Vector<double> sol;
      sol = rhs;
      A_direct_->solve(sol);
      solution = sol;


      for (unsigned int p = 0; p < n_pipes_; p++)
        {
          sth_->SelectPipe(p);
          pde.GetDoFConstraints().distribute(solution.block(p));
        }
      sth_->SelectPipe(n_pipes_);

    }

///////////////Endof Namespaces
  }
}
#endif
