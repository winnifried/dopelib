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

#ifndef Schur_LINEAR_SOLVER_H_
#define Schur_LINEAR_SOLVER_H_

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
#include <deal.II/lac/iterative_inverse.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <vector>

namespace DOpE
{
  /**
   * @class SchurLinearSolverWithMatrix
   *
   * This class implements the schur complement solver provided by
   * the dealii step-20 Example to solve the mixed formulation of
   * Poisson's problem.
   */

  class SchurLinearSolverWithMatrix
  {
  public:
    SchurLinearSolverWithMatrix(ParameterReader &param_reader);
    ~SchurLinearSolverWithMatrix();

    static void declare_params(ParameterReader &param_reader);

    /**
       This Function should be called once after grid refinement, or changes in boundary values
       to  recompute sparsity patterns, and constraint matrices.
    */
    template<typename PROBLEM>
    void ReInit(PROBLEM &pde);

    /**
     * Solves the linear PDE in the form Ax = b using dealii::SolverCG
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
    void Solve(PROBLEM &pde, INTEGRATOR &integr, dealii::BlockVector<double> &rhs, dealii::BlockVector<double> &solution, bool force_matrix_build=false);

  protected:

  private:
    //The SchurComplement and ApproximateSchurComplement Class are taken from dealii step-20
    class SchurComplement : public Subscriptor
    {
    public:
      SchurComplement (const BlockSparseMatrix<double> &A,
                       const IterativeInverse<Vector<double> > &Minv)
        :  system_matrix (&A), m_inverse (&Minv), tmp1 (A.block(0,0).m()), tmp2 (A.block(0,0).m())
      {}

      void vmult (Vector<double>       &dst,
                  const Vector<double> &src) const
      {
        system_matrix->block(0,1).vmult (tmp1, src);
        m_inverse->vmult (tmp2, tmp1);
        system_matrix->block(1,0).vmult (dst, tmp2);
      }

    private:
      const SmartPointer<const BlockSparseMatrix<double> > system_matrix;
      const SmartPointer<const IterativeInverse<Vector<double> > > m_inverse;

      mutable Vector<double> tmp1, tmp2;
    };
    class ApproximateSchurComplement : public Subscriptor
    {
    public:
      ApproximateSchurComplement (const BlockSparseMatrix<double> &A)
        : system_matrix (&A), tmp1 (A.block(0,0).m()), tmp2 (A.block(0,0).m())
      {}

      void vmult (Vector<double>       &dst,
                  const Vector<double> &src) const
      {
        system_matrix->block(0,1).vmult (tmp1, src);
        system_matrix->block(0,0).precondition_Jacobi (tmp2, tmp1);
        system_matrix->block(1,0).vmult (dst, tmp2);
      }
      void Tvmult (Vector<double>       &dst,
                   const Vector<double> &src) const
      {
        vmult (dst, src);
      }

    private:
      const SmartPointer<const BlockSparseMatrix<double> > system_matrix;

      mutable Vector<double> tmp1, tmp2;
    };
    //End of code from step-20

    dealii::BlockSparsityPattern sparsity_pattern_;
    dealii::BlockSparseMatrix<double> matrix_;
  };

  /*********************************Implementation************************************************/

  void SchurLinearSolverWithMatrix::declare_params(ParameterReader &/*param_reader*/)
  {
  }
  /******************************************************/

  SchurLinearSolverWithMatrix::SchurLinearSolverWithMatrix(ParameterReader &/*param_reader*/)
  {
  }

  /******************************************************/

  SchurLinearSolverWithMatrix::~SchurLinearSolverWithMatrix()
  {
  }

  /******************************************************/

  template<typename PROBLEM>
  void  SchurLinearSolverWithMatrix::ReInit(PROBLEM &pde)
  {
    matrix_.clear();
    pde.ComputeSparsityPattern(sparsity_pattern_);
    matrix_.reinit(sparsity_pattern_);
  }

  /******************************************************/
  template<typename PROBLEM, typename INTEGRATOR>
  void SchurLinearSolverWithMatrix::Solve(PROBLEM &pde,
                                          INTEGRATOR &integr,
                                          dealii::BlockVector<double> &rhs,
                                          dealii::BlockVector<double> &solution,
                                          bool force_matrix_build)
  {
    if (force_matrix_build)
      {
        integr.ComputeMatrix (pde,matrix_);
      }


    //This is the code coming from the dealii step-20 solve method:
    dealii::PreconditionIdentity identity;
    dealii::IterativeInverse<dealii::Vector<double> > m_inverse;
    m_inverse.initialize(matrix_.block(0,0), identity);
    m_inverse.solver.select("cg");
    static ReductionControl inner_control(1000, 0., 1.e-13,false,false);
    m_inverse.solver.set_control(inner_control);

    dealii::Vector<double> tmp (solution.block(0).size());

    {
      dealii::Vector<double> schur_rhs (solution.block(1).size());

      m_inverse.vmult (tmp, rhs.block(0));
      matrix_.block(1,0).vmult (schur_rhs, tmp);
      schur_rhs -= rhs.block(1);

      SchurComplement schur_complement (matrix_, m_inverse);

      ApproximateSchurComplement approximate_schur_complement (matrix_);

      dealii::IterativeInverse<Vector<double> > preconditioner;
      preconditioner.initialize(approximate_schur_complement, identity);
      preconditioner.solver.select("cg");
      preconditioner.solver.set_control(inner_control);

      SolverControl solver_control (solution.block(1).size(),
                                    1e-12*schur_rhs.l2_norm(),false,false);
      SolverCG<>    cg (solver_control);

      cg.solve (schur_complement, solution.block(1), schur_rhs,
                preconditioner);
    }
    {
      matrix_.block(0,1).vmult (tmp, solution.block(1));
      tmp *= -1;
      tmp += rhs.block(0);

      m_inverse.vmult (solution.block(0), tmp);
    }

    //Here the code ends, we only have to make sure our constraints are handled appropriately
    pde.GetDoFConstraints().distribute(solution);
  }
  /******************************************************/



}
#endif

