#ifndef _GMRES_LINEAR_SOLVER_H_
#define _GMRES_LINEAR_SOLVER_H_

#include <lac/vector.h>
#include <lac/block_sparsity_pattern.h>
#include <lac/block_sparse_matrix.h>
#include <lac/compressed_simple_sparsity_pattern.h>
#include <lac/solver_gmres.h>
#include <lac/precondition.h>
#include <lac/full_matrix.h>

#include <dofs/dof_tools.h>

#include <numerics/vectors.h>

#include <vector>

namespace DOpE
{

  template <typename PRECONDITIONER, typename SPARSITYPATTERN, typename MATRIX, typename VECTOR,int dim>
    class GMRESLinearSolverWithMatrix
  {
  public:
    GMRESLinearSolverWithMatrix( ParameterReader &param_reader);
    ~GMRESLinearSolverWithMatrix();

    static void declare_params(ParameterReader &param_reader);
    
    /**
       This Function should be called once after grid refinement, or changes in boundary values
       to  recompute sparsity patterns, and constraint matrices.
     */
    template<typename PROBLEM>
      void ReInit(PROBLEM& pde);

    /**
     * Solves the nonlinear PDE described by the PROBLEM given initialy to the constructor 
     * using the GMRES algorithm
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
      void Solve(PROBLEM& pde,INTEGRATOR& integr, VECTOR &rhs, VECTOR &solution, bool force_matrix_build=false);
     
  protected:

  private:
    SPARSITYPATTERN _sparsity_pattern;
    MATRIX _matrix;

    double _linear_global_tol, _linear_tol;
    int  _linear_maxiter, _no_tmp_vectors;
  };

/*********************************Implementation************************************************/

template <typename PRECONDITIONER,typename SPARSITYPATTERN, typename MATRIX, typename VECTOR,int dim>
 void GMRESLinearSolverWithMatrix<PRECONDITIONER,SPARSITYPATTERN,MATRIX,VECTOR, dim>::declare_params(ParameterReader &param_reader)
  {
    param_reader.SetSubsection("gmres_withmatrix parameters");
    param_reader.declare_entry("linear_global_tol", "1.e-10",Patterns::Double(0),"global tolerance for the gmres iteration");
    param_reader.declare_entry("linear_maxiter", "1000",Patterns::Integer(0),"maximal number of gmres steps");
    param_reader.declare_entry("no_tmp_vectors", "100",Patterns::Integer(0),"Number of temporary vectors");
  }
/******************************************************/

  template <typename PRECONDITIONER,typename SPARSITYPATTERN, typename MATRIX, typename VECTOR,int dim>
    GMRESLinearSolverWithMatrix<PRECONDITIONER,SPARSITYPATTERN,MATRIX,VECTOR, dim>
    ::GMRESLinearSolverWithMatrix(ParameterReader &param_reader) 
{
  param_reader.SetSubsection("gmres_withmatrix parameters");
  _linear_global_tol = param_reader.get_double ("linear_global_tol");
  _linear_maxiter    = param_reader.get_integer ("linear_maxiter"); 
  _no_tmp_vectors    = param_reader.get_integer ("no_tmp_vectors"); 

}

/******************************************************/

template <typename PRECONDITIONER,typename SPARSITYPATTERN, typename MATRIX, typename VECTOR,int dim>
  GMRESLinearSolverWithMatrix<PRECONDITIONER,SPARSITYPATTERN,MATRIX,VECTOR, dim>::~GMRESLinearSolverWithMatrix()
{
}

/******************************************************/

template <typename PRECONDITIONER,typename SPARSITYPATTERN, typename MATRIX, typename VECTOR,int dim>
  template<typename PROBLEM>
  void  GMRESLinearSolverWithMatrix<PRECONDITIONER,SPARSITYPATTERN,MATRIX,VECTOR, dim>::ReInit(PROBLEM& pde)
{
  _matrix.clear();
  pde.ComputeSparsityPattern(_sparsity_pattern);
  _matrix.reinit(_sparsity_pattern);
}

/******************************************************/

template <typename PRECONDITIONER,typename SPARSITYPATTERN, typename MATRIX, typename VECTOR,int dim>
  template<typename PROBLEM, typename INTEGRATOR>
  void GMRESLinearSolverWithMatrix<PRECONDITIONER,SPARSITYPATTERN,MATRIX,VECTOR, dim>::Solve(PROBLEM& pde, 
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

  // This is gmres specific
  dealii::GrowingVectorMemory<VECTOR> vector_memory;
  typename dealii::SolverGMRES<VECTOR>::AdditionalData gmres_data;
  gmres_data.max_n_tmp_vectors = _no_tmp_vectors;


  dealii::SolverGMRES<VECTOR> gmres (solver_control, vector_memory, gmres_data);
  PRECONDITIONER precondition;
  precondition.initialize(_matrix);
  gmres.solve (_matrix, solution, rhs,
    precondition);

  pde.GetHangingNodeConstraints().distribute(solution);
}


}
#endif
