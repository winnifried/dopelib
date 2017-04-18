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

#ifndef IPOPT_PROBLEM_H_
#define IPOPT_PROBLEM_H_

#ifdef DOPELIB_WITH_IPOPT
//Make shure the unused variable warnings from ipopt don't bother us
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "IpTNLP.hpp"
#pragma GCC diagnostic pop
#endif

#include <include/controlvector.h>

#include <iostream>

namespace DOpE
{
#ifdef DOPELIB_WITH_IPOPT

  /**
   * This class is used to transfer the problem given by
   * the user to the interface required to solve the problem
   * using IPOPT.
   *
   * @tparam <RPROBLEM>   The reduced problem considered for the solution
   *                      See ReducedProblemInterface for the required methods.
   * @tparam <VECTOR>     The vector type under consideration.
   *
   */
  template <typename RPROBLEM, typename VECTOR>
  class Ipopt_Problem : public Ipopt::TNLP
  {
  public:
    Ipopt_Problem( int &ret_val,
                   RPROBLEM *OP, ControlVector<VECTOR> &q,
                   const ControlVector<VECTOR> *q_min,
                   const ControlVector<VECTOR> *q_max,
                   const ConstraintVector<VECTOR> &c);

    virtual ~Ipopt_Problem() {}

    /**@name Overloaded from TNLP */
    //@{
    /** Method to return some info about the nlp */
    virtual bool get_nlp_info(Ipopt::Index &n, Ipopt::Index &m, Ipopt::Index &nnz_jac_g,
                              Ipopt::Index &nnz_h_lag, IndexStyleEnum &index_style);

    /** Method to return the bounds for my problem */
    virtual bool get_bounds_info(Ipopt::Index n, Ipopt::Number *x_l, Ipopt::Number *x_u,
                                 Ipopt::Index m, Ipopt::Number *g_l, Ipopt::Number *g_u);

    /** Method to return the starting point for the algorithm */
    virtual bool get_starting_point(Ipopt::Index n, bool init_x, Ipopt::Number *x,
                                    bool init_z, Ipopt::Number *z_L, Ipopt::Number *z_U,
                                    Ipopt::Index m, bool init_lambda,
                                    Ipopt::Number *lambda);

    /** Method to return the objective value */
    virtual bool eval_f(Ipopt::Index n, const Ipopt::Number *x, bool new_x, Ipopt::Number &obj_value);

    /** Method to return the gradient of the objective */
    virtual bool eval_grad_f(Ipopt::Index n, const Ipopt::Number *x, bool new_x, Ipopt::Number *grad_f);

    /** Method to return the constraint residuals */
    virtual bool eval_g(Ipopt::Index n, const Ipopt::Number *x, bool new_x, Ipopt::Index m, Ipopt::Number *g);

    /** Method to return:
     *   1) The structure of the jacobian (if "values" is NULL)
     *   2) The values of the jacobian (if "values" is not NULL)
     */
    virtual bool eval_jac_g(Ipopt::Index n, const Ipopt::Number *x, bool new_x,
                            Ipopt::Index m, Ipopt::Index nele_jac, Ipopt::Index *iRow, Ipopt::Index *jCol,
                            Ipopt::Number *values);

    /** Method to return:
     *   1) The structure of the hessian of the lagrangian (if "values" is NULL)
     *   2) The values of the hessian of the lagrangian (if "values" is not NULL)
     */
    virtual bool eval_h(Ipopt::Index n, const Ipopt::Number *x, bool new_x,
                        Ipopt::Number obj_factor, Ipopt::Index m, const Ipopt::Number *lambda,
                        bool new_lambda, Ipopt::Index nele_hess, Ipopt::Index *iRow,
                        Ipopt::Index *jCol, Ipopt::Number *values);

    //@}

    /** @name Solution Methods */
    //@{
    /** This method is called when the algorithm is complete so the TNLP can store/write the solution */
    virtual void finalize_solution(Ipopt::SolverReturn status,
                                   Ipopt::Index n, const Ipopt::Number *x, const Ipopt::Number *z_L, const Ipopt::Number *z_U,
                                   Ipopt::Index m, const Ipopt::Number *g, const Ipopt::Number *lambda,
                                   Ipopt::Number obj_value,
                                   const Ipopt::IpoptData *ip_data,
                                   Ipopt::IpoptCalculatedQuantities *ip_cq);
    //@}
  private:
    int &ret_val_;
    RPROBLEM *P_;
    ControlVector<VECTOR> q_;
    ControlVector<VECTOR> &init_; //Also the return value!
    const ControlVector<VECTOR> *q_min_;
    const ControlVector<VECTOR> *q_max_;
    ConstraintVector<VECTOR> c_;
  };
  /***************************************************************************************/
  /****************************************IMPLEMENTATION*********************************/
  /***************************************************************************************/
  template <typename RPROBLEM, typename VECTOR>
  Ipopt_Problem<RPROBLEM,VECTOR>::Ipopt_Problem(int &ret_val,
                                                RPROBLEM *OP, ControlVector<VECTOR> &q,
                                                const ControlVector<VECTOR> *q_min,
                                                const ControlVector<VECTOR> *q_max,
                                                const ConstraintVector<VECTOR> &c)
    : ret_val_(ret_val), P_(OP), q_(q), init_(q),
      q_min_(q_min), q_max_(q_max), c_(c)
  {
  }

  template <typename RPROBLEM, typename VECTOR>
  bool Ipopt_Problem<RPROBLEM,VECTOR>::get_nlp_info(Ipopt::Index &n, Ipopt::Index &m, Ipopt::Index &nnz_jac_g,
                                                    Ipopt::Index &nnz_h_lag, IndexStyleEnum &index_style)
  {
    n = q_.GetSpacialVector().size();              //n unknowns
    m = c_.GetGlobalConstraints().size(); //Only Global constraints
    nnz_jac_g = m*n; //Size of constraint jacobian
    nnz_h_lag = 0; //Size of hessian! (n * n Don't compute!)
    index_style = TNLP::C_STYLE; //C style indexing (0-based)
    return true;
  }

  template <typename RPROBLEM, typename VECTOR>
  bool Ipopt_Problem<RPROBLEM,VECTOR>::get_bounds_info(Ipopt::Index n, Ipopt::Number *x_l, Ipopt::Number *x_u,
                                                       Ipopt::Index m, Ipopt::Number *g_l, Ipopt::Number *g_u)
  {
    assert(n == (int) q_.GetSpacialVector().size());
    assert(m == (int) c_.GetGlobalConstraints().size());

    //Lower and upper bounds on q
    const VECTOR &lb = q_min_->GetSpacialVector();
    const VECTOR &ub = q_max_->GetSpacialVector();
    for (int i = 0; i < n; i++)
      {
        x_l[i] = lb(i);
        x_u[i] = ub(i);
      }
    //Global constraints are given such that feasible means <= 0
    for (int i = 0; i < m; i++)
      {
        g_l[i] = -1.e+20;
        g_u[i] = 0.;
      }
    return true;
  }

  template <typename RPROBLEM, typename VECTOR>
  bool Ipopt_Problem<RPROBLEM,VECTOR>::get_starting_point(Ipopt::Index n, bool /*init_x*/, Ipopt::Number *x,
                                                          bool /*init_z*/, Ipopt::Number * /*z_L*/, Ipopt::Number * /*z_U*/,
                                                          Ipopt::Index /*m*/, bool /*init_lambda*/,
                                                          Ipopt::Number * /*lambda*/)
  {
//    assert(init_x == true);
//    assert(init_z == false);
//    assert(init_lambda == false);
    const VECTOR &in = init_.GetSpacialVector();

    for (int i = 0; i < n; i++)
      {
        x[i] = in(i);
      }
    return true;
  }

  template <typename RPROBLEM, typename VECTOR>
  bool Ipopt_Problem<RPROBLEM,VECTOR>::eval_f(Ipopt::Index n, const Ipopt::Number *x, bool /*new_x*/, Ipopt::Number &obj_value)
  {
    VECTOR &qval = q_.GetSpacialVector();
    for (int i = 0; i < n; i++)
      {
        qval(i) = x[i];
      }
    try
      {
        obj_value = P_->ComputeReducedCostFunctional(q_);
      }
    catch (DOpEException &e)
      {
        P_->GetExceptionHandler()->HandleCriticalException(e,"IPOPT_RPROBLEM::eval_f");
      }
    return true;
  }

  template <typename RPROBLEM, typename VECTOR>
  bool Ipopt_Problem<RPROBLEM,VECTOR>::eval_grad_f(Ipopt::Index n, const Ipopt::Number *x, bool new_x, Ipopt::Number *grad_f)
  {
    ControlVector<VECTOR> gradient(q_);
    ControlVector<VECTOR> gradient_transposed(q_);
    if ( new_x )
      {
        //Need to calculate J!
        VECTOR &qval = q_.GetSpacialVector();
        for (int i = 0; i < n; i++)
          {
            qval(i) = x[i];
          }
        try
          {
            P_->ComputeReducedCostFunctional(q_);
          }
        catch (DOpEException &e)
          {
            P_->GetExceptionHandler()->HandleCriticalException(e,"IPOPT_PROBLEM::eval_grad_f");
          }
      }
    //Compute Functional Gradient
    try
      {
        P_->ComputeReducedGradient(q_,gradient,gradient_transposed);
      }
    catch (DOpEException &e)
      {
        P_->GetExceptionHandler()->HandleCriticalException(e,"IPOPT_PROBLEM::eval_grad_f");
      }
    const VECTOR &ref_g = gradient_transposed.GetSpacialVector();
    for (int i=0; i < n; i++)
      {
        grad_f[i] = ref_g(i);
      }

    return true;
  }

  template <typename RPROBLEM, typename VECTOR>
  bool Ipopt_Problem<RPROBLEM,VECTOR>::eval_g(Ipopt::Index n, const Ipopt::Number *x, bool new_x, Ipopt::Index /*m*/, Ipopt::Number *g)
  {
    if ( new_x )
      {
        //Need to calculate J!
        VECTOR &qval = q_.GetSpacialVector();
        for (int i = 0; i < n; i++)
          {
            qval(i) = x[i];
          }
        try
          {
            P_->ComputeReducedCostFunctional(q_);
          }
        catch (DOpEException &e)
          {
            P_->GetExceptionHandler()->HandleCriticalException(e,"IPOPT_PROBLEM::eval_g");
          }
      }
    //Calculate constraints
    try
      {
        P_->ComputeReducedConstraints(q_,c_);
      }
    catch (DOpEException &e)
      {
        P_->GetExceptionHandler()->HandleCriticalException(e,"IPOPT_PROBLEM::eval_g");
      }
    const dealii::Vector<double> &gc = c_.GetGlobalConstraints();
    for (unsigned int i=0; i < gc.size(); i++)
      {
        g[i] = gc(i);
      }
    return true;
  }

  template <typename RPROBLEM, typename VECTOR>
  bool Ipopt_Problem<RPROBLEM,VECTOR>::eval_jac_g(Ipopt::Index n, const Ipopt::Number *x, bool new_x,
                                                  Ipopt::Index m, Ipopt::Index /*nele_jac*/, Ipopt::Index *iRow, Ipopt::Index *jCol,
                                                  Ipopt::Number *values)
  {
    if (values != NULL)
      {
        //Compute Jacobian of Constraints
        ControlVector<VECTOR> gradient(q_);
        ControlVector<VECTOR> gradient_transposed(q_);
        if ( new_x )
          {
            //Need to calculate J!
            VECTOR &qval = q_.GetSpacialVector();
            for (int i = 0; i < n; i++)
              {
                qval(i) = x[i];
              }
            try
              {
                P_->ComputeReducedCostFunctional(q_);
              }
            catch (DOpEException &e)
              {
                P_->GetExceptionHandler()->HandleCriticalException(e,"IPOPT_PROBLEM::eval_grad_g");
              }
          }
        //Compute Constraint Gradients
        for (int j=0; j < m; j++)
          {
            try
              {
                P_->ComputeReducedGradientOfGlobalConstraints(j,q_,c_,
                                                              gradient,gradient_transposed);
              }
            catch (DOpEException &e)
              {
                P_->GetExceptionHandler()->HandleCriticalException(e,"IPOPT_PROBLEM::eval_grad_g");
              }
            const VECTOR &ref_g = gradient_transposed.GetSpacialVector();
            for (int i=0; i < n; i++)
              {
                values[n*j+i] = ref_g(i);
              }
          }
      }
    else
      {
        //assert(nele_jac == n*m);
        // return the structure of the Jacobian (here a dense one)
        for (int i = 0; i < n; i++)
          {
            for (int j = 0; j < m; j++)
              {
                iRow[n*j+i] = j;
                jCol[n*j+i] = i;
              }
          }
      }

    return true;
  }

  template <typename RPROBLEM, typename VECTOR>
  bool Ipopt_Problem<RPROBLEM,VECTOR>::eval_h(Ipopt::Index /*n*/, const Ipopt::Number * /*x*/, bool /*new_x*/,
                                              Ipopt::Number /*obj_factor*/, Ipopt::Index /*m*/, const Ipopt::Number * /*lambda*/,
                                              bool /*new_lambda*/, Ipopt::Index /*nele_hess*/, Ipopt::Index * /*iRow*/,
                                              Ipopt::Index * /*jCol*/, Ipopt::Number * /*values*/)
  {
    //Donot calculate the hessian!
    return false;
  }


  template <typename RPROBLEM, typename VECTOR>
  void Ipopt_Problem<RPROBLEM,VECTOR>::finalize_solution(Ipopt::SolverReturn status,
                                                         Ipopt::Index n, const Ipopt::Number *x,
                                                         const Ipopt::Number * /*z_L*/, const Ipopt::Number * /*z_U*/,
                                                         Ipopt::Index /*m*/, const Ipopt::Number * /*g*/,
                                                         const Ipopt::Number * /*lambda*/,
                                                         Ipopt::Number /*obj_value*/,
                                                         const Ipopt::IpoptData * /*ip_data*/,
                                                         Ipopt::IpoptCalculatedQuantities * /*ip_cq*/)
  {
    //Check the result, q_ should be x and then be copied to init_

    VECTOR &ret = init_.GetSpacialVector();
    for (int i = 0; i < n; i++)
      {
        ret(i) = x[i];
      }
    if (status == Ipopt::SUCCESS)
      ret_val_ = 1;
    else
      ret_val_ = 0;
  }

#endif //Endof DOPELIB_WITH_IPOPT



} //Endof Namespace DOpE

#endif
