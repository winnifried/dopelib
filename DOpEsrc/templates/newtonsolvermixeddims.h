#ifndef _NEWTON_MIXED_SOLVER_H_
#define _NEWTON_MIXED_SOLVER_H_

#include <lac/vector.h>
#include <lac/block_sparsity_pattern.h>
#include <lac/block_sparse_matrix.h>

#include <numerics/vectors.h>
#include <vector>
#include <iostream>
#include <iomanip>

#include "parameterreader.h"




namespace DOpE
{

  template <typename INTEGRATOR, typename LINEARSOLVER, typename VECTOR,int dimlow, int dimhigh>
    class NewtonSolverMixedDimensions : public LINEARSOLVER
  {
  public:
    NewtonSolverMixedDimensions(INTEGRATOR &integrator, ParameterReader &param_reader);
    ~NewtonSolverMixedDimensions();

    static void declare_params(ParameterReader &param_reader);

    /**
       This Function should be called once after grid refinement, or changes in boundary values
       to  recompute sparsity patterns, and constraint matrices.
     */
    template<typename PROBLEM>
      void ReInit(PROBLEM& pde);

    /**
     * Solves the nonlinear PDE described by the PDEPROBLEM given initialy to the constructor 
     * using a Newton-Method
     *
     * @param solution              A  Vector that will store the solution upon completion
     *                              Note that an initial guess for the solution may be stored 
     *                              in this vector as this Vector is used as starting value for the 
     *                              Iteration. 
     * @param apply_boundary_values A boolean that decides whether we apply strong dirichlet boundary values
     *                              to the Vector solution or not. If true (default) the Values will be applied
     *                              However if it is set to false, solution will be used unchanged as
     *                              initial condition. Be aware of the fact that NewtonsMethod can only converge 
     *                              in this case if the initial Value of `solution` has the correct
     *                              boundary values.
     * @param force_build_matrix    A boolean value, that indicates whether the Matrix
     *                              should be build by the linear solver in the first iteration.
     *				    The default is false, meaning that if we have no idea we don't
     *				    want to build a matrix.
     *
     * @return a boolean, that indicates whether it should be required to build the matrix next time that
     *         this method is used, e.g. the value for force_build_matrix of the next call.
     *
     */
    template<typename PROBLEM>
      bool NonlinearSolve(PROBLEM& pde, VECTOR &solution, bool apply_boundary_values=true, bool force_matrix_build=false,
			  int priority = 5, std::string algo_level = "\t\t ");
    
  protected:
    
    inline INTEGRATOR& GetIntegrator();
   
  private:
    INTEGRATOR &_integrator;

    bool _build_matrix;

    double _nonlinear_global_tol, _nonlinear_tol, _nonlinear_rho;
    double _linesearch_rho;
    int _nonlinear_maxiter, _line_maxiter;
  };

  /**********************************Implementation*******************************************/

template <typename INTEGRATOR, typename LINEARSOLVER, typename VECTOR,  int dimlow, int dimhigh>
 void NewtonSolverMixedDimensions<INTEGRATOR,LINEARSOLVER, VECTOR, dimlow, dimhigh>
    ::declare_params(ParameterReader &param_reader)
  {
      param_reader.SetSubsection("newtonsolver parameters");
      param_reader.declare_entry("nonlinear_global_tol", "1.e-12",Patterns::Double(0),"global tolerance for the newton iteration");
      param_reader.declare_entry("nonlinear_tol", "1.e-10",Patterns::Double(0),"relative tolerance for the newton iteration");
      param_reader.declare_entry("nonlinear_maxiter", "10",Patterns::Integer(0),"maximal number of newton iterations");
      param_reader.declare_entry("nonlinear_rho", "0.1",Patterns::Double(0),"minimal  newton reduction, if actual reduction is less, matrix is rebuild ");
     	   
      param_reader.declare_entry("line_maxiter", "4",Patterns::Integer(0),"maximal number of linesearch steps");
      param_reader.declare_entry("linesearch_rho", "0.9",Patterns::Double(0),"reduction rate for the linesearch damping paramete");
 
      LINEARSOLVER::declare_params(param_reader);
  }

  /*******************************************************************************************/

  template <typename INTEGRATOR, typename LINEARSOLVER, typename VECTOR,  int dimlow, int dimhigh>
    NewtonSolverMixedDimensions<INTEGRATOR,LINEARSOLVER, VECTOR, dimlow, dimhigh>
    ::NewtonSolverMixedDimensions(INTEGRATOR &integrator, ParameterReader &param_reader)
    : LINEARSOLVER(param_reader), _integrator(integrator)
    {
      assert(dimlow < dimhigh);

       param_reader.SetSubsection("newtonsolver parameters");
       _nonlinear_global_tol = param_reader.get_double ("nonlinear_global_tol");
       _nonlinear_tol        = param_reader.get_double ("nonlinear_tol"); 
       _nonlinear_maxiter    = param_reader.get_integer ("nonlinear_maxiter"); 
       _nonlinear_rho        = param_reader.get_double ("nonlinear_rho"); 

       _line_maxiter   = param_reader.get_integer ("line_maxiter");
       _linesearch_rho = param_reader.get_double ("linesearch_rho");
     
    }

  /*******************************************************************************************/

    template <typename INTEGRATOR, typename LINEARSOLVER, typename VECTOR,  int dimlow, int dimhigh>
    NewtonSolverMixedDimensions<INTEGRATOR,LINEARSOLVER, VECTOR, dimlow, dimhigh>
                   ::~NewtonSolverMixedDimensions()
    {
    }

  /*******************************************************************************************/
 template <typename INTEGRATOR, typename LINEARSOLVER, typename VECTOR,  int dimlow, int dimhigh>
   template<typename PROBLEM>
   void NewtonSolverMixedDimensions<INTEGRATOR,LINEARSOLVER, VECTOR, dimlow, dimhigh>
                 ::ReInit(PROBLEM& pde)
    {
      LINEARSOLVER::ReInit(pde);
    }

  /*******************************************************************************************/
 template <typename INTEGRATOR, typename LINEARSOLVER, typename VECTOR,  int dimlow, int dimhigh>
   template<typename PROBLEM>
   bool NewtonSolverMixedDimensions<INTEGRATOR,LINEARSOLVER, VECTOR, dimlow, dimhigh>
   ::NonlinearSolve(PROBLEM& pde,
		    VECTOR &solution, 
		    bool apply_boundary_values, 
		    bool force_matrix_build,
		    int priority, 
		    std::string algo_level)
    {
      bool build_matrix = force_matrix_build;
      VECTOR residual;
      VECTOR du;
      std::stringstream out;
      pde.GetOutputHandler()->InitNewtonOut(out);

      du.reinit(solution);
      residual.reinit(solution);
      
      if(apply_boundary_values)
      {
	GetIntegrator().ApplyInitialBoundaryValues(pde,solution);
      }
      
      dealii::Vector<double> copy_solution; 
      copy_solution= solution;
      
      GetIntegrator().AddParamData("last_newton_solution",&copy_solution);
    
      GetIntegrator().ComputeNonlinearResidual(pde,residual,true);
      residual *= -1.;
    
      pde.GetOutputHandler()->SetIterationNumber(0,"PDENewton");
      pde.GetOutputHandler()->Write(residual,"Residual"+pde.GetType(),pde.GetDoFType());
  
      double res = residual.linfty_norm();
      double firstres = res;
      double lastres = res;
      
      out<<algo_level + " Newton step: " <<0<<"\t Residual (abs.):  " <<pde.GetOutputHandler()->ZeroTolerance(res, 1.0)
	 <<"\n";

      out<<algo_level + " Newton step: " <<0<<"\t Residual (rel.): "
	 <<pde.GetOutputHandler()->ZeroTolerance(res/firstres, 1.0);

      pde.GetOutputHandler()->Write(out,priority);
      
      int iter=0;
      while(res > _nonlinear_global_tol && res > firstres * _nonlinear_tol)
      {
	iter++;
	
	if(iter > _nonlinear_maxiter)
	{
	  throw DOpEIterationException("Iteration count exceeded bounds!","NewtonSolverMixedDims::NonlinearSolve");
	}
	
	pde.GetOutputHandler()->SetIterationNumber(iter,"PDENewton");
	
	LINEARSOLVER::Solve(pde,GetIntegrator(),residual,du,build_matrix);
	build_matrix = false;
	//Linesearch
	{
	  solution += du;
	  copy_solution = solution;

	  GetIntegrator().ComputeNonlinearResidual(pde,residual,true);
	  residual *= -1.;
	  pde.GetOutputHandler()->Write(residual,"Residual"+pde.GetType(),pde.GetDoFType());
	  
	  double newres = residual.linfty_norm();
	  int lineiter=0;
	  double rho = _linesearch_rho;
	  double alpha=1;
	  
	  while(newres > res)
	  {
	    out<<algo_level + " Newton step: " <<iter<<"\t Residual (rel.): "
	       << pde.GetOutputHandler()->ZeroTolerance(newres/firstres, 1.0)
	       << "\t LineSearch {"<<lineiter<<"} ";
	    pde.GetOutputHandler()->Write(out,priority+1);

	    lineiter++;
	    if(lineiter > _line_maxiter)
	    {
	      throw DOpEIterationException("Line-Iteration count exceeded bounds!","NewtonSolverMixedDims::NonlinearSolve");
	    }
	    solution.add(alpha*(rho-1.),du);
	    alpha*= rho;
	    
	    GetIntegrator().ComputeNonlinearResidual(pde,residual,true);
	    residual *= -1.;
	    pde.GetOutputHandler()->Write(residual,"Residual"+pde.GetType(),pde.GetDoFType());
	        
	    newres = residual.linfty_norm();	    

	  }
	  if(res/lastres > _nonlinear_rho)
	  {
	    build_matrix=true;
	  }
	  lastres=res;
	  res=newres;

	  out<<algo_level + " Newton step: " <<iter<<"\t Residual (rel.): "
	     << pde.GetOutputHandler()->ZeroTolerance(res/firstres, 1.0)
	     << "\t LineSearch {"<<lineiter<<"} ";
	  pde.GetOutputHandler()->Write(out,priority);

	}//End of Linesearch
	copy_solution = solution;
      }
      GetIntegrator().DeleteParamData("last_newton_solution");

      return build_matrix;
    }


/*******************************************************************************************/
    template <typename INTEGRATOR, typename LINEARSOLVER, typename VECTOR,  int dimlow, int dimhigh>
    INTEGRATOR& NewtonSolverMixedDimensions<INTEGRATOR,LINEARSOLVER,VECTOR, dimlow, dimhigh>
                               ::GetIntegrator()
    {
      return _integrator;
    }

  /*******************************************************************************************/

}
#endif





