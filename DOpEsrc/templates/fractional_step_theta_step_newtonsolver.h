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

#ifndef _FRACTIONAL_STEP_THETA_STEP_NEWTON_SOLVER_H_
#define _FRACTIONAL_STEP_THETA_STEP_NEWTON_SOLVER_H_

#include <lac/vector.h>
#include <lac/block_sparsity_pattern.h>
#include <lac/block_sparse_matrix.h>

#include <numerics/vectors.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "parameterreader.h"




namespace DOpE
{
  /**
   * A nonlinear solver class to compute solutions to time dependent PDE- and optimization problems with
   * help of the Fractional-Step-theta (FS) scheme. This class differ from the instat_step_newtonsolver.h
   * since the time interval is split into three sub-intervals which is characteristic of this scheme.
   *
   * @tparam <INTEGRATOR>          Integration routines to compute domain-, face-, and right-hand side values.
   * @tparam <LINEARSOLVER>        A linear solver to solve the linear subproblems.
   * @tparam <PDEPROBLEM>          The PDE problem to solve.
   * @tparam <VECTOR>              A template class for arbitrary vectors which are given to the 
                                   FS scheme and where the solution is stored in.
   * @tparam <dim>                 The dimension of the problem: 1, 2, or 3.				  
   */
  template <typename INTEGRATOR, typename LINEARSOLVER, typename VECTOR,int dim>
    class FractionalStepThetaStepNewtonSolver : public LINEARSOLVER
  {
  public:
    /**
     * Constructor of this class. Initialization of parameters which are used during computation.
     * E.g., maximal number of newton iterations, global tolerance of the nonlinear solver etc.
     * 
     * @param integrator          A reference of the integrator is given to the nonlinear solver.
     * @param pde_problem         A reference of the PDE problem which is going to solved.
     * @param param_reader        An object which has run time data for the nonlinear solver.
     */
    FractionalStepThetaStepNewtonSolver(INTEGRATOR &integrator, ParameterReader &param_reader);
    ~FractionalStepThetaStepNewtonSolver();

    /******************************************************/

    /**
     * Static member function for run time parameters.
     *
     * @param param_reader      An object which has run time data.
     */
    static void declare_params(ParameterReader &param_reader);

    /******************************************************/

    /**
       This Function should be called once after grid refinement, or changes in boundary values
       to  recompute sparsity patterns, and constraint matrices.
     */
    template<typename PROBLEM>
      void ReInit(PROBLEM& pde);

    /******************************************************/

    /**
     * Solves the nonlinear PDE coming from the backward euler time-discretization 
     * described by the PDEPROBLEM given initialy to the constructor 
     * using a Newton-Method
     *
     * @param last_time_solution    A  Vector stores the solution from the previous timestep
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
      bool NonlinearSolve(PROBLEM& pde, const VECTOR &last_time_solution, VECTOR &solution, bool apply_boundary_values=true, 
			  bool force_matrix_build=false, int priority = 5, std::string algo_level = "\t\t ");
    
    /******************************************************/

    /**
     * Solves the nonlinear PDE to determine the initial value for the	
     * PROBLEM, i.e., there is no time dependence in this problem!
     * using a Newton-Method
     *
     * @param solution              A  Vector that will store the solution upon completion
     *                              It is expected that solution is initially set to the return value
     *                              of residual in NonlinearLastTimeEvals!
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
      bool NonlinearSolve_Initial(PROBLEM& pde, VECTOR &solution, bool apply_boundary_values=true, 
			bool force_matrix_build=false, int priority = 5, std::string algo_level = "\t\t ");

    /******************************************************/
    
    /**
     * Evaluates the timestep Problem at the previous time-point, this is part of the rhs for the Solution
     *
     * @param last_time_solution          A  Vector stores the solution from the previous timestep
     * @param residual                    A  Vector that will store the results upon completion
     *
     */
    template<typename PROBLEM>
      void NonlinearLastTimeEvals(PROBLEM& pde, const VECTOR &last_time_solution, VECTOR &residual);
    
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

template <typename INTEGRATOR, typename LINEARSOLVER, typename VECTOR,  int dim>
 void FractionalStepThetaStepNewtonSolver<INTEGRATOR,LINEARSOLVER, VECTOR,  dim>
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

  template <typename INTEGRATOR, typename LINEARSOLVER, typename VECTOR,  int dim>
    FractionalStepThetaStepNewtonSolver<INTEGRATOR,LINEARSOLVER, VECTOR,  dim>
    ::FractionalStepThetaStepNewtonSolver(INTEGRATOR &integrator, ParameterReader &param_reader)
    : LINEARSOLVER(param_reader), _integrator(integrator)
    {
       param_reader.SetSubsection("newtonsolver parameters");
       _nonlinear_global_tol = param_reader.get_double ("nonlinear_global_tol");
       _nonlinear_tol        = param_reader.get_double ("nonlinear_tol"); 
       _nonlinear_maxiter    = param_reader.get_integer ("nonlinear_maxiter"); 
       _nonlinear_rho        = param_reader.get_double ("nonlinear_rho"); 

       _line_maxiter   = param_reader.get_integer ("line_maxiter");
       _linesearch_rho = param_reader.get_double ("linesearch_rho");
     
    }

  /*******************************************************************************************/

    template <typename INTEGRATOR, typename LINEARSOLVER, typename VECTOR,  int dim>
    FractionalStepThetaStepNewtonSolver<INTEGRATOR,LINEARSOLVER, VECTOR,  dim>
                   ::~FractionalStepThetaStepNewtonSolver()
    {
    }

  /*******************************************************************************************/
 template <typename INTEGRATOR, typename LINEARSOLVER, typename VECTOR,  int dim>
   template<typename PROBLEM>
    void FractionalStepThetaStepNewtonSolver<INTEGRATOR,LINEARSOLVER, VECTOR,  dim>
                        ::ReInit(PROBLEM& pde)
    {
      LINEARSOLVER::ReInit(pde);
    }
 
  /*******************************************************************************************/
 template <typename INTEGRATOR, typename LINEARSOLVER, typename VECTOR,  int dim>
   template<typename PROBLEM>
   void FractionalStepThetaStepNewtonSolver<INTEGRATOR,LINEARSOLVER, VECTOR, dim>
                         ::NonlinearLastTimeEvals(PROBLEM& pde, const VECTOR &last_time_solution, VECTOR &residual)
   { 
      VECTOR tmp_residual;
      tmp_residual.reinit(residual);
      residual =0.; 
      GetIntegrator().AddDomainData("last_newton_solution",&last_time_solution);
      GetIntegrator().AddDomainData("last_time_solution",&last_time_solution);
      pde.SetStepPart("Old_for_1st_cycle");
      GetIntegrator().ComputeNonlinearLhs(pde,residual);   
      GetIntegrator().ComputeNonlinearRhs(pde,tmp_residual);
      tmp_residual *= -1;
      residual += tmp_residual;
      
      GetIntegrator().DeleteDomainData("last_newton_solution");
      GetIntegrator().DeleteDomainData("last_time_solution");
   }
      /*******************************************************************************************/

 template <typename INTEGRATOR, typename LINEARSOLVER, typename VECTOR,  int dim>
   template<typename PROBLEM>
   bool FractionalStepThetaStepNewtonSolver<INTEGRATOR,LINEARSOLVER, VECTOR, dim>
   ::NonlinearSolve_Initial(PROBLEM& pde, VECTOR &solution, bool apply_boundary_values, 
			    bool force_matrix_build, int priority, std::string algo_level)
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
   pde.GetDoFConstraints().distribute(solution);
   
   GetIntegrator().AddDomainData("last_newton_solution",&solution);
   
   GetIntegrator().ComputeNonlinearResidual(pde,residual);
   residual *= -1.;
   
   pde.GetOutputHandler()->SetIterationNumber(0,"PDENewton");
   pde.GetOutputHandler()->Write(residual,"Residual"+pde.GetType(),pde.GetDoFType());
   
   double res = residual.linfty_norm();
   double firstres = res;
   double lastres = res;
   
   
   out<< algo_level << "Newton step: " <<0<<"\t Residual (abs.): "
      <<pde.GetOutputHandler()->ZeroTolerance(res, 1.0)
      <<"\n";
   
   out<< algo_level << "Newton step: " <<0<<"\t Residual (rel.):   " << std::scientific << firstres/firstres; 
   
   
   pde.GetOutputHandler()->Write(out,priority);
   
   int iter=0;
   while(res > _nonlinear_global_tol && res > firstres * _nonlinear_tol)
   {
     iter++;
     
     if(iter > _nonlinear_maxiter)
     {
       GetIntegrator().DeleteDomainData("last_newton_solution");
       throw DOpEIterationException("Iteration count exceeded bounds!","InstatStepNewtonSolver::NonlinearSolve_Initial");
     }
     
     pde.GetOutputHandler()->SetIterationNumber(iter,"PDENewton");
     
     LINEARSOLVER::Solve(pde,GetIntegrator(),residual,du,build_matrix);
     build_matrix = false;
     
     //Linesearch
     {
       solution += du;
       GetIntegrator().ComputeNonlinearResidual(pde,residual);
       residual *= -1.;
       
       pde.GetOutputHandler()->Write(residual,"Residual"+pde.GetType(),pde.GetDoFType());
       pde.GetOutputHandler()->Write(du,"Update"+pde.GetType(),pde.GetDoFType());
       
       double newres = residual.linfty_norm();
       int lineiter=0;
       double rho = _linesearch_rho;
       double alpha=1;
       
       while(newres > res)
       {
	 out<< algo_level << "Newton step: " <<iter<<"\t Residual (rel.): "
	    <<pde.GetOutputHandler()->ZeroTolerance(newres/firstres, 1.0)
	    << "\t LineSearch {"<<lineiter<<"} ";
	 
	 pde.GetOutputHandler()->Write(out,priority+1);
	 
	 lineiter++;
	 if(lineiter > _line_maxiter)
	 {
	   GetIntegrator().DeleteDomainData("last_newton_solution");
	   throw DOpEIterationException("Line-Iteration count exceeded bounds!","InstatStepNewtonSolver::NonlinearSolve_Initial");
	 }
	 solution.add(alpha*(rho-1.),du);
	 alpha*= rho;
	 
	 GetIntegrator().ComputeNonlinearResidual(pde,residual);
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
       
       out << algo_level 
	   << "Newton step: " 
	   <<iter
	   <<"\t Residual (rel.): "
	   << pde.GetOutputHandler()->ZeroTolerance(res/firstres, 1.0)
	   << "\t LineSearch {"
	   <<lineiter
	   <<"} ";
       
	  
       pde.GetOutputHandler()->Write(out,priority);
       
     }//End of Linesearch
   }
   GetIntegrator().DeleteDomainData("last_newton_solution");
   
   return build_matrix; 
 }
  
  /*******************************************************************************************/
 template <typename INTEGRATOR, typename LINEARSOLVER,  typename VECTOR,  int dim>
   template<typename PROBLEM>
   bool FractionalStepThetaStepNewtonSolver<INTEGRATOR,LINEARSOLVER, VECTOR, dim>
                        ::NonlinearSolve(PROBLEM& pde,
					 const VECTOR &last_time_solution, 
					 VECTOR &solution, 
					 bool apply_boundary_values, 
					 bool force_matrix_build,
					 int priority,
					 std::string /*algo_level*/)
    {      
       
      bool build_matrix = force_matrix_build;
      VECTOR residual, time_residual, tmp_residual;
      VECTOR du, tmp_last_time_solution;
      std::stringstream out;
      pde.GetOutputHandler()->InitNewtonOut(out);

      double res = 0.0;
      double firstres = 0.0;
      double lastres = 0.0;

      du.reinit(solution);
      residual.reinit(solution);
      time_residual.reinit(solution);
      tmp_residual.reinit(solution);
      tmp_last_time_solution.reinit(solution);

       //Transfer from previous timestep
      residual +=solution;
      // last_time_solution is very good starting value
      solution = last_time_solution;
   

      // First cycle of FS-scheme
      if(apply_boundary_values)
      {
	GetIntegrator().ApplyInitialBoundaryValues(pde,solution);
      }
      
      // echte rechte Seite zum aktuellen Zeitschritt f^{n+1}
      GetIntegrator().AddDomainData("last_time_solution",&last_time_solution);
      GetIntegrator().AddDomainData("last_newton_solution",&solution);
      pde.SetStepPart("New_for_1st_and_3rd_cycle");
      GetIntegrator().ComputeNonlinearRhs(pde,tmp_residual);    
      tmp_residual *= -1;
      residual += tmp_residual;

      // Speichern des nicht von u^{n+1} abh. Anteils
      time_residual = residual;
      time_residual *=-1;

      // Berechne zum "echte" Residuumsgleichung zum aktuellen Zeitschritt
      GetIntegrator().ComputeNonlinearLhs(pde,tmp_residual); // modi, new
      residual += tmp_residual;
               
      residual *=-1.; // wg. A(U)(\psi) = - A(U)(du,\psi)
     
      pde.GetOutputHandler()->SetIterationNumber(0,"PDENewton");
      pde.GetOutputHandler()->Write(residual,"Residual"+pde.GetType(),pde.GetDoFType());
  
      res = residual.linfty_norm();
      firstres = res;
      lastres = res;
      int iter=0;
     
      out<<"\t\t Newton step: " <<0<<"\t Residual (abs.): "
	 <<pde.GetOutputHandler()->ZeroTolerance(res, 1.0)
	 <<"\n";
      out<<"\t\t Newton step: " <<0<<"\t Residual (rel.):   "<< std::scientific << firstres/firstres; 




      pde.GetOutputHandler()->Write(out,priority);
      while(res > _nonlinear_global_tol && res > firstres * _nonlinear_tol)
      {
	iter++;
	if(iter > _nonlinear_maxiter)
	{
	  throw DOpEIterationException("Iteration count exceeded bounds!","StatSolver::NonlinearSolve");
	}
	
	pde.GetOutputHandler()->SetIterationNumber(iter,"PDENewton");	
	LINEARSOLVER::Solve(pde,GetIntegrator(),residual,du,build_matrix);
	build_matrix = false;	
	//Linesearch
	{
	  solution += du;	 
	  GetIntegrator().ComputeNonlinearLhs(pde,residual);	
	  //GetIntegrator().ComputeNonlinearRhs(residual); // später vielleicht, falls f^{n+1} linearisiert wird (wie bei FSI)	  
	  residual -= time_residual; 
	  residual *= -1.;
	  pde.GetOutputHandler()->Write(residual,"Residual"+pde.GetType(),pde.GetDoFType());
	  
	  double newres = residual.linfty_norm();
	  int lineiter=0;
	  double rho = _linesearch_rho;
	  double alpha=1;
	  
	  while(newres > res)
	  {
	    out<<"\t\t\t Linesearch step: " <<lineiter<<"\t Residual (rel.): "
	       <<pde.GetOutputHandler()->ZeroTolerance(newres/firstres, 1.0);

	    pde.GetOutputHandler()->Write(out,priority+1);
	    lineiter++;
	    if(lineiter > _line_maxiter)
	    {
	      throw DOpEIterationException("Line-Iteration count exceeded bounds!","StatSolver::NonlinearSolve");
	    }
	    solution.add(alpha*(rho-1.),du);
	    alpha*= rho;
	    
	    GetIntegrator().ComputeNonlinearLhs(pde,residual);
	    residual -= time_residual; 
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

	  out<<"\t\t Newton step: " <<iter<<"\t Residual (rel.): "
	     << pde.GetOutputHandler()->ZeroTolerance(res/firstres, 1.0)
	     << "\t LineSearch {"<<lineiter<<"} ";
	  pde.GetOutputHandler()->Write(out,priority);

	}//End of Linesearch
      } 
      GetIntegrator().DeleteDomainData("last_time_solution");
      GetIntegrator().DeleteDomainData("last_newton_solution");
      

      // 2nd cycle of FS-scheme 
      tmp_last_time_solution = 0;
      tmp_last_time_solution = solution;

      // muss bereits hier berechnet werden, da sonst in local_cell_equation die
      // Werte nicht gesetzt werden können
      GetIntegrator().AddDomainData("last_time_solution",&tmp_last_time_solution);

      // Berechne Residuumsanteil der alten Zeitschrittloesung
      GetIntegrator().AddDomainData("last_newton_solution",&tmp_last_time_solution);
      pde.SetStepPart("Old_for_2nd_cycle");
      GetIntegrator().ComputeNonlinearLhs(pde,residual);       
   
      GetIntegrator().ComputeNonlinearRhs(pde,tmp_residual);
      tmp_residual *= -1;
      residual += tmp_residual;

      GetIntegrator().DeleteDomainData("last_newton_solution");
      // alte Zeitloesung ist fertig berechnet

      if(apply_boundary_values)
      {
	GetIntegrator().ApplyInitialBoundaryValues(pde,solution);
      }
      
   
      // echte rechte Seite zum aktuellen Zeitschritt f^{n+1}     
      GetIntegrator().AddDomainData("last_newton_solution",&solution);
      pde.SetStepPart("New_for_2nd_cycle");
      GetIntegrator().ComputeNonlinearRhs(pde,tmp_residual);    
      tmp_residual *= -1;
      residual += tmp_residual;

      // Speichern des nicht von u^{n+1} abh. Anteils
      time_residual = residual;
      time_residual *=-1;

      // Berechne zum "echte" Residuumsgleichung zum aktuellen Zeitschritt
      GetIntegrator().ComputeNonlinearLhs(pde,tmp_residual); // modi, new
      residual += tmp_residual;
               
      residual *=-1.; // wg. A(U)(\psi) = - A(U)(du,\psi)
     
      pde.GetOutputHandler()->SetIterationNumber(0,"PDENewton");
      pde.GetOutputHandler()->Write(residual,"Residual"+pde.GetType(),pde.GetDoFType());
  
      res = residual.linfty_norm();
      firstres = res;
      lastres = res;
      iter=0;
  
      out<<"\t\t Newton step: " <<0<<"\t Residual (abs.): "
	 <<pde.GetOutputHandler()->ZeroTolerance(res, 1.0)
      	 <<"\n";
      
      out<<"\t\t Newton step: " <<0<<"\t Residual (rel.):   "<< std::scientific << firstres/firstres; 


      pde.GetOutputHandler()->Write(out,priority);
      while(res > _nonlinear_global_tol && res > firstres * _nonlinear_tol)
      {
	iter++;
	if(iter > _nonlinear_maxiter)
	{
	  throw DOpEIterationException("Iteration count exceeded bounds!","StatSolver::NonlinearSolve");
	}
	
	pde.GetOutputHandler()->SetIterationNumber(iter,"PDENewton");	
	LINEARSOLVER::Solve(pde,GetIntegrator(),residual,du,build_matrix);
	build_matrix = false;	
	//Linesearch
	{
	  solution += du;	 
	  GetIntegrator().ComputeNonlinearLhs(pde,residual);	
	  //GetIntegrator().ComputeNonlinearRhs(residual); // später vielleicht, falls f^{n+1} linearisiert wird (wie bei FSI)	  
	  residual -= time_residual; 
	  residual *= -1.;
	  pde.GetOutputHandler()->Write(residual,"Residual"+pde.GetType(),pde.GetDoFType());
	  
	  double newres = residual.linfty_norm();
	  int lineiter=0;
	  double rho = _linesearch_rho;
	  double alpha=1;
	  
	  while(newres > res)
	  {
	    out<<"\t\t\t Linesearch step: " <<lineiter<<"\t Residual (rel.): "
	       <<pde.GetOutputHandler()->ZeroTolerance(newres/firstres, 1.0);

	    pde.GetOutputHandler()->Write(out,priority+1);
	    lineiter++;
	    if(lineiter > _line_maxiter)
	    {
	      throw DOpEIterationException("Line-Iteration count exceeded bounds!","StatSolver::NonlinearSolve");
	    }
	    solution.add(alpha*(rho-1.),du);
	    alpha*= rho;
	    
	    GetIntegrator().ComputeNonlinearLhs(pde,residual);
	    residual -= time_residual;
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

	  out<<"\t\t Newton step: " <<iter<<"\t Residual (rel.): "
	     << pde.GetOutputHandler()->ZeroTolerance(res/firstres, 1.0)
	     << "\t LineSearch {"<<lineiter<<"} ";

	  pde.GetOutputHandler()->Write(out,priority);

	}//End of Linesearch
      }   
      GetIntegrator().DeleteDomainData("last_time_solution");
      GetIntegrator().DeleteDomainData("last_newton_solution");


      // 3rd cycle of FS-scheme
      tmp_last_time_solution = 0;
      tmp_last_time_solution = solution;

      // muss bereits hier berechnet werden, da sonst in local_cell_equation die
      // Werte nicht gesetzt werden können
      GetIntegrator().AddDomainData("last_time_solution",&tmp_last_time_solution);

      // Berechne Residuumsanteil der alten Zeitschrittloesung
      GetIntegrator().AddDomainData("last_newton_solution",&tmp_last_time_solution);
      pde.SetStepPart("Old_for_3rd_cycle");
      GetIntegrator().ComputeNonlinearLhs(pde,residual);   
    
      GetIntegrator().ComputeNonlinearRhs(pde,tmp_residual);
      tmp_residual *= -1;
      residual += tmp_residual;

      GetIntegrator().DeleteDomainData("last_newton_solution");
      // alte Zeitloesung ist fertig berechnet
      
      if(apply_boundary_values)
	{
	  GetIntegrator().ApplyInitialBoundaryValues(pde,solution);
	}
          
      // echte rechte Seite zum aktuellen Zeitschritt f^{n+1}    
      GetIntegrator().AddDomainData("last_newton_solution",&solution);
      pde.SetStepPart("New_for_1st_and_3rd_cycle");
      GetIntegrator().ComputeNonlinearRhs(pde,tmp_residual);    
      tmp_residual *= -1;
      residual += tmp_residual;

      // Speichern des nicht von u^{n+1} abh. Anteils
      time_residual = residual;
      time_residual *=-1;

      // Berechne zum "echte" Residuumsgleichung zum aktuellen Zeitschritt
      GetIntegrator().ComputeNonlinearLhs(pde,tmp_residual); // modi, new
      residual += tmp_residual;
               
      residual *=-1.; // wg. A(U)(\psi) = - A(U)(du,\psi)
     
      pde.GetOutputHandler()->SetIterationNumber(0,"PDENewton");
      pde.GetOutputHandler()->Write(residual,"Residual"+pde.GetType(),pde.GetDoFType());
  
      res = residual.linfty_norm();
      firstres = res;
      lastres = res;
      iter=0;
     
      out<<"\t\t Newton step: " <<0<<"\t Residual (abs.): "
	 <<pde.GetOutputHandler()->ZeroTolerance(res, 1.0)
      	 <<"\n";
      
      out<<"\t\t Newton step: " <<0<<"\t Residual (rel.):   "<< std::scientific << firstres/firstres; 

      
      pde.GetOutputHandler()->Write(out,priority);
      while(res > _nonlinear_global_tol && res > firstres * _nonlinear_tol)
      {
	iter++;
	if(iter > _nonlinear_maxiter)
	{
	  throw DOpEIterationException("Iteration count exceeded bounds!","StatSolver::NonlinearSolve");
	}
	
	pde.GetOutputHandler()->SetIterationNumber(iter,"PDENewton");	
	LINEARSOLVER::Solve(pde,GetIntegrator(),residual,du,build_matrix);
	build_matrix = false;	
	//Linesearch
	{
	  solution += du;	 
	  GetIntegrator().ComputeNonlinearLhs(pde,residual);	
	  //GetIntegrator().ComputeNonlinearRhs(residual); // später vielleicht, falls f^{n+1} linearisiert wird (wie bei FSI)	  
	  residual -= time_residual; 
	  residual *= -1.;
	  pde.GetOutputHandler()->Write(residual,"Residual"+pde.GetType(),pde.GetDoFType());
	  
	  double newres = residual.linfty_norm();
	  int lineiter=0;
	  double rho = _linesearch_rho;
	  double alpha=1;
	  
	  while(newres > res)
	  {
	    out<<"\t\t\t Linesearch step: " <<lineiter<<"\t Residual (rel.): "
	       <<pde.GetOutputHandler()->ZeroTolerance(newres/firstres, 1.0);

	    pde.GetOutputHandler()->Write(out,priority+1);
	    lineiter++;
	    if(lineiter > _line_maxiter)
	    {
	      throw DOpEIterationException("Line-Iteration count exceeded bounds!","StatSolver::NonlinearSolve");
	    }
	    solution.add(alpha*(rho-1.),du);
	    alpha*= rho;
	    
	    GetIntegrator().ComputeNonlinearLhs(pde,residual);
	    residual -= time_residual; 
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

	  out<<"\t\t Newton step: " <<iter<<"\t Residual (rel.): "
	     << pde.GetOutputHandler()->ZeroTolerance(res/firstres, 1.0)
	     << "\t LineSearch {"<<lineiter<<"} ";

	  pde.GetOutputHandler()->Write(out,priority);

	}//End of Linesearch
      } 
      GetIntegrator().DeleteDomainData("last_time_solution");
      GetIntegrator().DeleteDomainData("last_newton_solution");



      return build_matrix;        
    }


/*******************************************************************************************/
    template <typename INTEGRATOR, typename LINEARSOLVER, typename VECTOR,  int dim>
    INTEGRATOR& FractionalStepThetaStepNewtonSolver<INTEGRATOR,LINEARSOLVER, VECTOR,  dim>
                               ::GetIntegrator()
    {
      return _integrator;
    }

  /*******************************************************************************************/

}
#endif
