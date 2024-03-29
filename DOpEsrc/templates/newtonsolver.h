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

#ifndef NEWTON_SOLVER_H_
#define NEWTON_SOLVER_H_

#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/numerics/vector_tools.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>

#include <include/parameterreader.h>




namespace DOpE
{
  /**
   * A nonlinear solver class to compute solutions to nonlinear (stationary) problems
   *
   * @tparam <INTEGRATOR>          Integration routines to compute domain-, face-, and right-hand side values.
   * @tparam <LINEARSOLVER>        A linear solver to solve the linear subproblems.
   * @tparam <VECTOR>              A template class for arbitrary vectors which are given to the
                                   FS scheme and where the solution is stored in.
   */

  template <typename INTEGRATOR, typename LINEARSOLVER, typename VECTOR>
  class NewtonSolver : public LINEARSOLVER
  {
  public:
    NewtonSolver(INTEGRATOR &integrator, ParameterReader &param_reader);
    ~NewtonSolver();

    static void declare_params(ParameterReader &param_reader);

    /**
       This Function should be called once after grid refinement, or changes in boundary values
       to  recompute sparsity patterns, and constraint matrices.
     */
    template<typename PROBLEM>
    void ReInit(PROBLEM &pde);

    /**
     * Solves the nonlinear PDE described by the PROBLEM using a Newton-Method
     *
     * @tparam <PROBLEM>            The description of the problem we want to solve.
     *
     * @param pde                   The problem
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
     *            The default is false, meaning that if we have no idea we don't
     *            want to build a matrix.
     * @param priority              A number that defines the offset for the priority of the output
     * @param algo_level            A prefix string to adjust indentation of the output.
     *
     * @return a boolean, that indicates whether it should be required to build the matrix next time that
     *         this method is used, e.g. the value for force_build_matrix of the next call.
     *
     */
    template<typename PROBLEM>
    bool NonlinearSolve(PROBLEM &pde, VECTOR &solution, bool apply_boundary_values=true,
                        bool force_matrix_build=false,
                        int priority = 5, std::string algo_level = "\t\t ");

  protected:

    inline INTEGRATOR &GetIntegrator();

  private:
    INTEGRATOR &integrator_;

    bool build_matrix_;

    double nonlinear_global_tol_, nonlinear_tol_, nonlinear_rho_;
    double linesearch_rho_;
    int nonlinear_maxiter_, line_maxiter_;
  };

  /**********************************Implementation*******************************************/

  template <typename INTEGRATOR, typename LINEARSOLVER, typename VECTOR>
  void NewtonSolver<INTEGRATOR,LINEARSOLVER, VECTOR>
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

  template <typename INTEGRATOR, typename LINEARSOLVER, typename VECTOR>
  NewtonSolver<INTEGRATOR,LINEARSOLVER, VECTOR>
  ::NewtonSolver(INTEGRATOR &integrator, ParameterReader &param_reader)
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

  template <typename INTEGRATOR, typename LINEARSOLVER, typename VECTOR>
  NewtonSolver<INTEGRATOR,LINEARSOLVER, VECTOR>
  ::~NewtonSolver()
  {
  }

  /*******************************************************************************************/
  template <typename INTEGRATOR, typename LINEARSOLVER, typename VECTOR>
  template<typename PROBLEM>
  void NewtonSolver<INTEGRATOR,LINEARSOLVER, VECTOR>
  ::ReInit(PROBLEM &pde)
  {
    LINEARSOLVER::ReInit(pde);
  }

  /*******************************************************************************************/
  template <typename INTEGRATOR, typename LINEARSOLVER, typename VECTOR>
  template<typename PROBLEM>
  bool NewtonSolver<INTEGRATOR,LINEARSOLVER, VECTOR>
  ::NonlinearSolve(PROBLEM &pde,
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
    while (res > nonlinear_global_tol_ && res > firstres * nonlinear_tol_)
      {
        iter++;

        if (iter > nonlinear_maxiter_)
          {
            GetIntegrator().DeleteDomainData("last_newton_solution");
            GetIntegrator().DeleteAllData();
            throw DOpEIterationException("Iteration count exceeded bounds!","NewtonSolver::NonlinearSolve");
          }

        pde.GetOutputHandler()->SetIterationNumber(iter,"PDENewton");

        LINEARSOLVER::Solve(pde,GetIntegrator(),residual,du,build_matrix);

        //Linesearch
        {
          solution += du;
          GetIntegrator().ComputeNonlinearResidual(pde,residual);
          residual *= -1.;

          pde.GetOutputHandler()->Write(residual,"Residual"+pde.GetType(),pde.GetDoFType());
          pde.GetOutputHandler()->Write(du,"Update"+pde.GetType(),pde.GetDoFType());

          res = residual.linfty_norm();
          int lineiter=0;
          pde.GetOutputHandler()->SetIterationNumber(lineiter,"PDENewtonLS");
          double rho = linesearch_rho_;
          double alpha=1;
          if ( res > lastres && build_matrix == false)
            {
              build_matrix = true;
              // Reuse of Matrix seems to be a bad idea, rebuild and repeat
              solution -= du;
              GetIntegrator().ComputeNonlinearResidual(pde,residual);
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
              while (res > lastres)
                {
                  out<< algo_level << "Newton step: " <<iter<<"\t Residual (rel.): "
                     <<pde.GetOutputHandler()->ZeroTolerance(res/firstres, 1.0)
                     << "\t LineSearch {"<<lineiter<<"} ";
                  if (was_build)
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

                  GetIntegrator().ComputeNonlinearResidual(pde,residual);
                  residual *= -1.;
                  pde.GetOutputHandler()->Write(residual,"Residual"+pde.GetType(),pde.GetDoFType());
                  pde.GetOutputHandler()->Write(solution,"Intermediate"+pde.GetType(),pde.GetDoFType());

                  res = residual.linfty_norm();

                }

              if (res/lastres > nonlinear_rho_)
                {
                  build_matrix=true;
                }
              lastres=res;

              out << algo_level
                  << "Newton step: "
                  <<iter
                  <<"\t Residual (rel.): "
                  << pde.GetOutputHandler()->ZeroTolerance(res/firstres, 1.0)
                  << "\t LineSearch {"
                  <<lineiter
                  <<"} ";
              if (was_build)
                out<<"M ";


              pde.GetOutputHandler()->Write(out,priority);

            }//End of Linesearch
        }
      }
    GetIntegrator().DeleteDomainData("last_newton_solution");

    return build_matrix;
  }

  /*******************************************************************************************/
  template <typename INTEGRATOR, typename LINEARSOLVER, typename VECTOR>
  INTEGRATOR &NewtonSolver<INTEGRATOR,LINEARSOLVER, VECTOR>
  ::GetIntegrator()
  {
    return integrator_;
  }

  /*******************************************************************************************/

}
#endif





