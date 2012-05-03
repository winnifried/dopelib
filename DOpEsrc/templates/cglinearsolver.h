#ifndef _CG_LINEAR_SOLVER_H_
#define _CG_LINEAR_SOLVER_H_

#include <lac/vector.h>
#include <lac/block_sparsity_pattern.h>
#include <lac/block_sparse_matrix.h>
#include <lac/compressed_simple_sparsity_pattern.h>
#include <lac/solver_cg.h>
#include <lac/precondition.h>
#include <lac/full_matrix.h>

#include <dofs/dof_tools.h>

#include <numerics/vectors.h>

#include <vector>

namespace DOpE
{

  template <typename PRECONDITIONER, typename SPARSITYPATTERN, typename MATRIX, typename VECTOR,int dim>
    class CGLinearSolverWithMatrix
  {
  public:
    CGLinearSolverWithMatrix(ParameterReader &param_reader);
    ~CGLinearSolverWithMatrix();

    static void declare_params(ParameterReader &param_reader);
    
    /**
       This Function should be called once after grid refinement, or changes in boundary values
       to  recompute sparsity patterns, and constraint matrices.
     */
    template<typename PROBLEM>
      void ReInit(PROBLEM& pde);

    /**
     * Solves the nonlinear PDE described by the PROBLEM given initialy to the constructor 
     * using a CG-Method
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
 
    double _linear_global_tol, _linear_tol;
    int  _linear_maxiter;
  };

/*********************************Implementation************************************************/

template <typename PRECONDITIONER,typename SPARSITYPATTERN, typename MATRIX, typename VECTOR,int dim>
 void CGLinearSolverWithMatrix<PRECONDITIONER,SPARSITYPATTERN,MATRIX,VECTOR, dim>::declare_params(ParameterReader &param_reader)
  {
    param_reader.SetSubsection("cglinearsolver_withmatrix parameters");
    param_reader.declare_entry("linear_global_tol", "1.e-16",Patterns::Double(0),"global tolerance for the cg iteration");
    param_reader.declare_entry("linear_tol", "1.e-12",Patterns::Double(0),"relative tolerance for the cg iteration");
    param_reader.declare_entry("linear_maxiter", "1000",Patterns::Integer(0),"maximal number of cg steps");
  }
/******************************************************/

  template <typename PRECONDITIONER,typename SPARSITYPATTERN, typename MATRIX, typename VECTOR,int dim>
    CGLinearSolverWithMatrix<PRECONDITIONER,SPARSITYPATTERN,MATRIX,VECTOR, dim>
    ::CGLinearSolverWithMatrix(ParameterReader &param_reader) 
{
  param_reader.SetSubsection("cglinearsolver_withmatrix parameters");
  _linear_global_tol = param_reader.get_double ("linear_global_tol");
  _linear_tol        = param_reader.get_double ("linear_tol"); 
  _linear_maxiter    = param_reader.get_integer ("linear_maxiter"); 

}

/******************************************************/

template <typename PRECONDITIONER,typename SPARSITYPATTERN, typename MATRIX, typename VECTOR,int dim>
  CGLinearSolverWithMatrix<PRECONDITIONER,SPARSITYPATTERN,MATRIX,VECTOR, dim>::~CGLinearSolverWithMatrix()
{
}

/******************************************************/

template <typename PRECONDITIONER,typename SPARSITYPATTERN, typename MATRIX, typename VECTOR,int dim>
  template<typename PROBLEM>
  void  CGLinearSolverWithMatrix<PRECONDITIONER,SPARSITYPATTERN,MATRIX,VECTOR, dim>::ReInit(PROBLEM& pde)
{
  _matrix.clear();
  pde.ComputeSparsityPattern(_sparsity_pattern);
  _matrix.reinit(_sparsity_pattern);
}

/******************************************************/

template <typename PRECONDITIONER,typename SPARSITYPATTERN, typename MATRIX, typename VECTOR,int dim>
  template<typename PROBLEM, typename INTEGRATOR>
  void CGLinearSolverWithMatrix<PRECONDITIONER,SPARSITYPATTERN,MATRIX,VECTOR, dim>::Solve(PROBLEM& pde, 
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

  dealii::SolverControl solver_control (_linear_maxiter, _linear_global_tol,false,false);
  dealii::SolverCG<VECTOR> cg (solver_control);
  PRECONDITIONER precondition;
  precondition.initialize(_matrix);
  cg.solve (_matrix, solution, rhs,
	    precondition);
  
  pde.GetDoFConstraints().distribute(solution);
}
/******************************************************/



}
#endif

