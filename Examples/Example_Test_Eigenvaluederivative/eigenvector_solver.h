/**
*
* Copyright (C) 2012-2018 by the DOpElib authors
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

#ifndef EIGENVECTOR_SOLVER_H_
#define EIGENVECTOR_SOLVER_H_

#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/slepc_solver.h>

#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>

#include <include/parameterreader.h>
#include <deal.II/lac/petsc_parallel_vector.h>



namespace DOpE
{

  template <typename INTEGRATOR, typename VECTOR,typename EIGENVALUES, typename EIGENVECTORS, typename MATRIX,typename SPARSITYPATTERN, typename LINEARSOLVER>
  class EigenvectorSolver : public LINEARSOLVER
  {
  public:
	  EigenvectorSolver( INTEGRATOR &integrator, ParameterReader &param_reader);
    ~EigenvectorSolver();

    static void declare_params(ParameterReader &param_reader);

    template<typename PROBLEM>
    void ReInit(PROBLEM &pde);

    template<typename PROBLEM>
    bool EigenvalueSolve(PROBLEM &pde, EIGENVALUES &eigenvalues, EIGENVECTORS &eigenfunctions, bool apply_boundary_values=true,
                            bool force_matrix_build=false,
                            int priority = 5, std::string algo_level = "\t\t "/*, int n_eigenval=5*/);
    template<typename PROBLEM>
       bool NonlinearSolve(PROBLEM &pde, EIGENVALUES &eigenvalues, VECTOR &solution, bool apply_boundary_values=true,
                               bool force_matrix_build=false,
                               int priority = 5, std::string algo_level = "\t\t "/*, int n_eigenval=5*/);

    inline INTEGRATOR &GetIntegrator();

  private:
    INTEGRATOR &integrator_;
    SPARSITYPATTERN sparsity_pattern_;
    MATRIX matrixK_, matrixM_;

    IndexSet eigenfunction_index_set;

    bool build_matrix_;

//    double linear_global_tol_= 0.000001, linear_tol_ = 0.00001;
   int  linear_maxiter_=1000,n_eigenval_;

    double nonlinear_global_tol_, nonlinear_tol_, nonlinear_rho_;
       double linesearch_rho_;
       int nonlinear_maxiter_, line_maxiter_;
  };

  /**********************************Implementation*******************************************/

  template <typename INTEGRATOR, typename VECTOR,typename EIGENVALUES, typename EIGENVECTORS, typename MATRIX, typename SPARSITYPATTERN, typename LINEARSOLVER>
  void EigenvectorSolver<INTEGRATOR,VECTOR,EIGENVALUES, EIGENVECTORS, MATRIX, SPARSITYPATTERN, LINEARSOLVER>
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
  template <typename INTEGRATOR, typename VECTOR, typename EIGENVALUES, typename EIGENVECTORS, typename MATRIX, typename SPARSITYPATTERN, typename LINEARSOLVER>
  EigenvectorSolver<INTEGRATOR,VECTOR,EIGENVALUES, EIGENVECTORS, MATRIX, SPARSITYPATTERN, LINEARSOLVER>
  ::EigenvectorSolver(INTEGRATOR &integrator, ParameterReader &param_reader)
    : LINEARSOLVER(param_reader), integrator_(integrator)
  {
	    param_reader.SetSubsection("newtonsolver parameters");
	    nonlinear_global_tol_ = param_reader.get_double ("nonlinear_global_tol");
	    nonlinear_tol_        = param_reader.get_double ("nonlinear_tol");
	    nonlinear_maxiter_    = param_reader.get_integer ("nonlinear_maxiter");
	    nonlinear_rho_        = param_reader.get_double ("nonlinear_rho");

	    line_maxiter_   = param_reader.get_integer ("line_maxiter");
	    linesearch_rho_ = param_reader.get_double ("linesearch_rho");
  }

  /*******************************************************************************************/

  template <typename INTEGRATOR, typename VECTOR, typename EIGENVALUES, typename EIGENVECTORS, typename MATRIX, typename SPARSITYPATTERN, typename LINEARSOLVER>
  EigenvectorSolver<INTEGRATOR,VECTOR, EIGENVALUES, EIGENVECTORS, MATRIX, SPARSITYPATTERN, LINEARSOLVER>
  ::~EigenvectorSolver()
  {
  }

  /*******************************************************************************************/
  template <typename INTEGRATOR, typename VECTOR, typename EIGENVALUES, typename EIGENVECTORS, typename MATRIX, typename SPARSITYPATTERN, typename LINEARSOLVER>
  template<typename PROBLEM>
  void EigenvectorSolver<INTEGRATOR, VECTOR, EIGENVALUES, EIGENVECTORS, MATRIX, SPARSITYPATTERN, LINEARSOLVER>
  ::ReInit(PROBLEM &pde)
  {

	   	 matrixM_.clear();
	     matrixM_.reinit(pde.GetBaseProblem().GetSpaceTimeHandler()->GetStateDoFHandler().GetDEALDoFHandler().n_dofs(), pde.GetBaseProblem().GetSpaceTimeHandler()->GetStateDoFHandler().GetDEALDoFHandler().n_dofs(),
	     		pde.GetBaseProblem().GetSpaceTimeHandler()->GetStateDoFHandler().GetDEALDoFHandler().max_couplings_between_dofs());
	     matrixK_.clear();
	     matrixK_.reinit(pde.GetBaseProblem().GetSpaceTimeHandler()->GetStateDoFHandler().GetDEALDoFHandler().n_dofs(), pde.GetBaseProblem().GetSpaceTimeHandler()->GetStateDoFHandler().GetDEALDoFHandler().n_dofs(),
	     		pde.GetBaseProblem().GetSpaceTimeHandler()->GetStateDoFHandler().GetDEALDoFHandler().max_couplings_between_dofs());
	     LINEARSOLVER::ReInit(pde);

  }

  /*******************************************************************************************/
  template <typename INTEGRATOR, typename VECTOR, typename EIGENVALUES, typename EIGENVECTORS, typename MATRIX, typename SPARSITYPATTERN, typename LINEARSOLVER>
  template<typename PROBLEM>
  bool EigenvectorSolver<INTEGRATOR, VECTOR, EIGENVALUES, EIGENVECTORS, MATRIX, SPARSITYPATTERN, LINEARSOLVER>
  ::EigenvalueSolve(PROBLEM &pde,
		  EIGENVALUES &eigenvalues,
		  EIGENVECTORS &eigenfunctions,
		           bool /*apply_boundary_values*/,
                   bool force_matrix_build,
                   int /*priority*/,
                   std::string /*algo_level*//*, int n_eigenval*/)
  {

   bool build_matrix = force_matrix_build;


   if (force_matrix_build) {
   	   this->ReInit(pde);
	   integrator_.ComputeMatrix(pde,matrixK_);
	   integrator_.ComputeMassMatrix(pde,matrixM_);
   }

      eigenfunction_index_set = pde.GetBaseProblem().GetSpaceTimeHandler()->GetStateDoFHandler().GetDEALDoFHandler().locally_owned_dofs();
      for (unsigned int i = 0; i < eigenfunctions.size(); ++i) {
       	eigenfunctions[i].reinit(eigenfunction_index_set, MPI_COMM_WORLD);
      }
      linear_maxiter_ = pde.GetBaseProblem().GetSpaceTimeHandler()->GetStateDoFHandler().GetDEALDoFHandler().n_dofs();
      dealii::SolverControl solver_control (linear_maxiter_, 1e-5/*,false,false*/);
          SLEPcWrappers::SolverLAPACK/*SolverJacobiDavidson*/eigensolver(solver_control, MPI_COMM_WORLD);
          eigensolver.set_which_eigenpairs(EPS_TARGET_MAGNITUDE);
          eigensolver.set_target_eigenvalue(0.001);

          eigensolver.solve(matrixK_, matrixM_, eigenvalues, eigenfunctions, eigenvalues.size());
              	for (unsigned int i = 0; i < eigenfunctions.size(); ++i) {
//             		eigenfunctions[i] /= eigenfunctions[i].linfty_norm();
              		eigenfunctions[i] /= eigenfunctions[i].norm_sqr();
             	}

    return build_matrix;
  }



  /*******************************************************************************************/
  template <typename INTEGRATOR, typename VECTOR, typename EIGENVALUES, typename EIGENVECTORS, typename MATRIX, typename SPARSITYPATTERN, typename LINEARSOLVER>
    template<typename PROBLEM>
  bool EigenvectorSolver<INTEGRATOR, VECTOR, EIGENVALUES, EIGENVECTORS, MATRIX, SPARSITYPATTERN, LINEARSOLVER>
  ::NonlinearSolve(PROBLEM &pde,
		  	  	  EIGENVALUES &eigenvalues,
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

    if (apply_boundary_values)
      {
        GetIntegrator().ApplyInitialBoundaryValues(pde,solution);
      }

    GetIntegrator().AddDomainData("last_newton_solution",&solution);

    GetIntegrator().ComputeNonlinearResidual(pde,residual,eigenvalues[0]);
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
    while (res > nonlinear_global_tol_ && res > firstres * nonlinear_tol_)
      {
        iter++;

        if (iter > nonlinear_maxiter_)
          {
            GetIntegrator().DeleteDomainData("last_newton_solution");
	    GetIntegrator().DeleteAllData();
            throw DOpEIterationException("Iteration count exceeded bounds!","EigenvectorSolver::NonlinearSolve");
          }

        pde.GetOutputHandler()->SetIterationNumber(iter,"PDENewton");

        LINEARSOLVER::Solve(pde,GetIntegrator(),residual,du,build_matrix);

        //Linesearch
        {
          solution += du;
          GetIntegrator().ComputeNonlinearResidual(pde,residual,eigenvalues[0]);
          residual *= -1.;

          pde.GetOutputHandler()->Write(residual,"Residual"+pde.GetType(),pde.GetDoFType());
          pde.GetOutputHandler()->Write(du,"Update"+pde.GetType(),pde.GetDoFType());

          double newres = residual.linfty_norm();
          int lineiter=0;
          pde.GetOutputHandler()->SetIterationNumber(lineiter,"PDENewtonLS");
          double rho = linesearch_rho_;
          double alpha=1;
          if ( newres > res && build_matrix == false)
            {
              build_matrix = true;
              // Reuse of Matrix seems to be a bad idea, rebuild and repeat
              solution -= du;
              GetIntegrator().ComputeNonlinearResidual(pde,residual,eigenvalues[0]);
              residual *= -1.;
              out << algo_level
                  << "Newton step: "
                  <<iter
                  <<"\t Recalculate with new Matrix";
              iter--;
              pde.GetOutputHandler()->Write(out,priority);
            }
          else
            {
	      bool was_build = build_matrix;
              build_matrix = false;
	      pde.GetOutputHandler()->Write(solution,"Intermediate"+pde.GetType(),pde.GetDoFType());
              while (newres > res)
                {
                  out<< algo_level << "Newton step: " <<iter<<"\t Residual (rel.): "
                     <<pde.GetOutputHandler()->ZeroTolerance(newres/firstres, 1.0)
                     << "\t LineSearch {"<<lineiter<<"} ";
		  if(was_build)
		    out<<"M ";
                  pde.GetOutputHandler()->Write(out,priority+1);

                  lineiter++;
		  pde.GetOutputHandler()->SetIterationNumber(lineiter,"PDENewtonLS");
                  if (lineiter > line_maxiter_)
                    {
                      GetIntegrator().DeleteDomainData("last_newton_solution");
		      GetIntegrator().DeleteAllData();
                      throw DOpEIterationException("Line-Iteration count exceeded bounds!","NewtonSolver::NonlinearSolve");
                    }
                  solution.add(alpha*(rho-1.),du);
                  alpha*= rho;

                  GetIntegrator().ComputeNonlinearResidual(pde,residual,eigenvalues[0]);
                  residual *= -1.;
                  pde.GetOutputHandler()->Write(residual,"Residual"+pde.GetType(),pde.GetDoFType());
                  pde.GetOutputHandler()->Write(solution,"Intermediate"+pde.GetType(),pde.GetDoFType());

                  newres = residual.linfty_norm();

                }

              if (res/lastres > nonlinear_rho_)
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
	      if(was_build)
		out<<"M ";


              pde.GetOutputHandler()->Write(out,priority);

            }//End of Linesearch
        }
      }
   GetIntegrator().DeleteDomainData("last_newton_solution");

    return build_matrix;
  }

  /*******************************************************************************************/

  /*******************************************************************************************/
  template <typename INTEGRATOR,  typename VECTOR, typename EIGENVALUES, typename EIGENVECTORS,typename MATRIX, typename SPARSITYPATTERN, typename LINEARSOLVER>
  INTEGRATOR &EigenvectorSolver<INTEGRATOR, VECTOR, EIGENVALUES, EIGENVECTORS, MATRIX, SPARSITYPATTERN, LINEARSOLVER>
  ::GetIntegrator()
  {
    return integrator_;
  }

  /*******************************************************************************************/

}
#endif





