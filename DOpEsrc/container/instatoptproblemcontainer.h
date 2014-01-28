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

#ifndef _INSTAT_OPT_PROBLEM_CONTAINER_
#define _INSTAT_OPT_PROBLEM_CONTAINER_

#include "optproblemcontainer.h"

namespace DOpE
{
  /**
   * Container class for all nonstationary Optimization problems.
   * This class collects all problem depended data needed to 
   * calculate the solution to the optimization problem.
   *
   * At present also nonstationary PDEs use this container, beeing also an optimization problem
   * over a set with just one point.
   *
   * @tparam PRIMALTSPROBLEM        The description of the time discretization scheme for the PDE
   *                                (and tangent PDE).
   * @tparam ADJOINTTSPROBLEM       The description of the time discretization scheme for the adjoint
   *                                PDE (and all auxilliary adjoint problems).
   * @tparam FUNCTIONAL_INTERFACE   A generic interface to arbitrary functionals to be evaluated.
   * @tparam FUNCTIONAL             The cost functional, see FunctionalInterface for details.
   * @tparam PDE                    The description of the PDE, see PDEInterface for details.
   * @tparam DD                     The description of the Dirichlet data, see 
   *                                DirichletDataInterface for details.
   * @tparam CONSTRAINTS            The description of, possible, additional constraints for the 
   *                                optimization problem, see ConstraintInterface for details.
   * @tparam SPARSITYPATTERN        The sparsity pattern to be used in the stiffness matrix.
   * @tparam VECTOR                 The vector type in which the coordinate vector of the 
   *                                solution is to be stored.
   * @tparam dopedim                The dimension of the domain in which the control is considered.
   * @tparam dealdim                The dimension of the domain in which the PDE is considered.
   * @tparam FE                     The finite element under consideration.
   * @tparam DH                     The spatial DoFHandler to be used when evaluating the 
   *                                weak form.
   */
  template<template<typename BASE_PROB, typename SPARSITYPATTERN, typename VECTOR,int dopedim, int dealdim, template<int, int> class FE, template<int, int> class DH> class PRIMALTSPROBLEM,
    template<typename BASE_PROB, typename SPARSITYPATTERN, typename VECTOR,int dopedim, int dealdim, template<int, int> class FE, template<int, int> class DH> class  ADJOINTTSPROBLEM,
    typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
    typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
    typename VECTOR, int dopedim, int dealdim,
    template<int, int> class FE = FESystem,
    template<int, int> class DH = dealii::DoFHandler>
  class InstatOptProblemContainer : public OptProblemContainer<FUNCTIONAL_INTERFACE,FUNCTIONAL,PDE,DD,CONSTRAINTS,
  SPARSITYPATTERN,VECTOR,dopedim,dealdim,FE, DH>
{
  public:
  InstatOptProblemContainer(FUNCTIONAL& functional,PDE& pde,CONSTRAINTS& constraints,
			    SpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR, dopedim,dealdim>& STH)
  : OptProblemContainer<FUNCTIONAL_INTERFACE,FUNCTIONAL,PDE,DD,CONSTRAINTS,SPARSITYPATTERN,VECTOR,dopedim,dealdim,FE, DH>(
    functional,pde,constraints,STH), _ts_state_problem(NULL), _ts_adjoint_problem(NULL), 
  _ts_tangent_problem(NULL), _ts_adjoint_hessian_problem(NULL)
  {
  }
  
  ~InstatOptProblemContainer()
  {
    if(_ts_state_problem != NULL)
    {
      delete _ts_state_problem;
    }
    if(_ts_adjoint_problem != NULL)
    {
      delete _ts_adjoint_problem;
    }
    if(_ts_tangent_problem != NULL)
    {
      delete _ts_tangent_problem;
    }
    if(_ts_adjoint_hessian_problem != NULL)
    {
      delete _ts_adjoint_hessian_problem;
    }
  }

  void ReInit(std::string algo_type)
  {  
    if(_ts_state_problem != NULL)
    {
      delete _ts_state_problem;
      _ts_state_problem = NULL;
    }
    if(_ts_adjoint_problem != NULL)
    {
      delete _ts_adjoint_problem;
      _ts_adjoint_problem = NULL;
    }
    if(_ts_tangent_problem != NULL)
    {
      delete _ts_tangent_problem;
      _ts_tangent_problem = NULL;
    }
    if(_ts_adjoint_hessian_problem != NULL)
    {
      delete _ts_adjoint_hessian_problem;
      _ts_adjoint_hessian_problem = NULL;
    }
    OptProblemContainer<FUNCTIONAL_INTERFACE,FUNCTIONAL,PDE,DD,CONSTRAINTS,SPARSITYPATTERN,VECTOR,dopedim,dealdim,FE, DH>::ReInit(algo_type);
	
  }

  std::string GetName() const
  {
    return "InstatOptProblemContainer";
  }
  
  /*******************************************************************************************/
  /**
   * Returns a description of the PDE
   */
  PRIMALTSPROBLEM<StateProblem<
  OptProblemContainer<FUNCTIONAL_INTERFACE,FUNCTIONAL,PDE,DD,CONSTRAINTS,SPARSITYPATTERN,VECTOR,dopedim,dealdim,FE, DH>,
  PDE, DD, SPARSITYPATTERN, VECTOR, dealdim>,
  SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>& GetStateProblem()
  {
    if(_ts_state_problem == NULL)
    {
      _ts_state_problem = new PRIMALTSPROBLEM<StateProblem<
      OptProblemContainer<FUNCTIONAL_INTERFACE,FUNCTIONAL,PDE,DD,CONSTRAINTS,SPARSITYPATTERN,VECTOR,dopedim,dealdim,FE, DH>,
      PDE, DD, SPARSITYPATTERN, VECTOR, dealdim>,
      SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>(OptProblemContainer<FUNCTIONAL_INTERFACE,FUNCTIONAL,PDE,DD,CONSTRAINTS,
      					      SPARSITYPATTERN,VECTOR,dopedim,dealdim,FE, DH>::GetStateProblem());
    }
    return *_ts_state_problem;
  }
  /*******************************************************************************************/

  //FIXME: This should use the GetAdjointProblem of OptProblemContainer once availiable simillar for all calls
  // to ADJOINTTSPROBLEM ...
   /**
    * Returns a description of the Adjoint PDE
    */
  ADJOINTTSPROBLEM<OptProblemContainer<FUNCTIONAL_INTERFACE,FUNCTIONAL,PDE,DD,CONSTRAINTS,SPARSITYPATTERN,VECTOR,dopedim,dealdim,FE, DH>,
  SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>& GetAdjointProblem()
  {
    if(_ts_adjoint_problem == NULL)
    {
      _ts_adjoint_problem = new ADJOINTTSPROBLEM<OptProblemContainer<FUNCTIONAL_INTERFACE,FUNCTIONAL,PDE,DD,CONSTRAINTS,SPARSITYPATTERN,VECTOR,dopedim,dealdim,FE, DH>,
      SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>(OptProblemContainer<FUNCTIONAL_INTERFACE,FUNCTIONAL,PDE,DD,CONSTRAINTS,
								 SPARSITYPATTERN,VECTOR,dopedim,dealdim,FE, DH>::GetBaseProblem());
    }
    return *_ts_adjoint_problem;
  }

  /*******************************************************************************************/
  //FIXME: This should use the GetTangentProblem of OptProblemContainer once availiable simillar for all 
  // other related calls
  /**
   * Returns a description of the tangent PDE
   */
  PRIMALTSPROBLEM<OptProblemContainer<FUNCTIONAL_INTERFACE,FUNCTIONAL,PDE,DD,CONSTRAINTS,SPARSITYPATTERN,VECTOR,dopedim,dealdim,FE, DH>,
  SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>& GetTangentProblem()
  {
    if(_ts_tangent_problem == NULL)
    {
      _ts_tangent_problem = new PRIMALTSPROBLEM<OptProblemContainer<FUNCTIONAL_INTERFACE,FUNCTIONAL,PDE,DD,CONSTRAINTS,SPARSITYPATTERN,VECTOR,dopedim,dealdim,FE, DH>,
      SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>(OptProblemContainer<FUNCTIONAL_INTERFACE,FUNCTIONAL,PDE,DD,CONSTRAINTS,
      					      SPARSITYPATTERN,VECTOR,dopedim,dealdim,FE, DH>::GetBaseProblem());
    }
    return *_ts_tangent_problem;
  }
  /*******************************************************************************************/

  //FIXME: This should use the GetAdjointHessianProblem of OptProblemContainer once availiable simillar for all 
  // other related calls
   /**
    * Returns a description of the Auxilliary Adjoint PDE for the Hessian Operator
    */
  ADJOINTTSPROBLEM<OptProblemContainer<FUNCTIONAL_INTERFACE,FUNCTIONAL,PDE,DD,CONSTRAINTS,SPARSITYPATTERN,VECTOR,dopedim,dealdim,FE, DH>,
  SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>& GetAdjointHessianProblem()
  {
    if(_ts_adjoint_hessian_problem == NULL)
    {
      _ts_adjoint_hessian_problem = new ADJOINTTSPROBLEM<OptProblemContainer<FUNCTIONAL_INTERFACE,FUNCTIONAL,PDE,DD,CONSTRAINTS,SPARSITYPATTERN,VECTOR,dopedim,dealdim,FE, DH>,
      SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>(OptProblemContainer<FUNCTIONAL_INTERFACE,FUNCTIONAL,PDE,DD,CONSTRAINTS,
								 SPARSITYPATTERN,VECTOR,dopedim,dealdim,FE, DH>::GetBaseProblem());
    }
    return *_ts_adjoint_hessian_problem;
  }

private:
  PRIMALTSPROBLEM<StateProblem<
      OptProblemContainer<FUNCTIONAL_INTERFACE,FUNCTIONAL,PDE,DD,CONSTRAINTS,SPARSITYPATTERN,VECTOR,dopedim,dealdim,FE, DH>,
      PDE, DD, SPARSITYPATTERN, VECTOR, dealdim>,
      SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>* _ts_state_problem;
  ADJOINTTSPROBLEM<OptProblemContainer<FUNCTIONAL_INTERFACE,FUNCTIONAL,PDE,DD,CONSTRAINTS,SPARSITYPATTERN,VECTOR,dopedim,dealdim,FE, DH>,
      SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>* _ts_adjoint_problem;
  PRIMALTSPROBLEM<OptProblemContainer<FUNCTIONAL_INTERFACE,FUNCTIONAL,PDE,DD,CONSTRAINTS,SPARSITYPATTERN,VECTOR,dopedim,dealdim,FE, DH>,
      SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>* _ts_tangent_problem;
  ADJOINTTSPROBLEM<OptProblemContainer<FUNCTIONAL_INTERFACE,FUNCTIONAL,PDE,DD,CONSTRAINTS,SPARSITYPATTERN,VECTOR,dopedim,dealdim,FE, DH>,
      SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DH>* _ts_adjoint_hessian_problem;
};
}
#endif
