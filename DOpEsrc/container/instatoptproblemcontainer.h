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
  template<template<typename BASE_PROB, typename SPARSITYPATTERN, typename VECTOR,int dopedim, int dealdim, typename FE, typename DOFHANDLER> class PRIMALTSPROBLEM, 
    template<typename BASE_PROB, typename SPARSITYPATTERN, typename VECTOR,int dopedim, int dealdim, typename FE, typename DOFHANDLER> class  ADJOINTTSPROBLEM, 
    typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
    typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
    typename VECTOR, int dopedim, int dealdim,
    typename FE = FESystem<dealdim>,
    typename DOFHANDLER = dealii::DoFHandler<dealdim>>
  class InstatOptProblemContainer : public OptProblemContainer<FUNCTIONAL_INTERFACE,FUNCTIONAL,PDE,DD,CONSTRAINTS,
  SPARSITYPATTERN,VECTOR,dopedim,dealdim,FE, DOFHANDLER>
{
  public:
  InstatOptProblemContainer(FUNCTIONAL& functional,PDE& pde,CONSTRAINTS& constraints,
			    SpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR, dopedim,dealdim>& STH)
  : OptProblemContainer<FUNCTIONAL_INTERFACE,FUNCTIONAL,PDE,DD,CONSTRAINTS,SPARSITYPATTERN,VECTOR,dopedim,dealdim,FE, DOFHANDLER>(
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
    }
    if(_ts_adjoint_hessian_problem != NULL)
    {
      delete _ts_adjoint_hessian_problem;
    }
    OptProblemContainer<FUNCTIONAL_INTERFACE,FUNCTIONAL,PDE,DD,CONSTRAINTS,SPARSITYPATTERN,VECTOR,dopedim,dealdim,FE, DOFHANDLER>::ReInit(algo_type);
	
  }

  std::string GetName() const
  {
    return "InstatOptProblemContainer";
  }
  
  /*******************************************************************************************/

  PRIMALTSPROBLEM<StateProblem<
  OptProblemContainer<FUNCTIONAL_INTERFACE,FUNCTIONAL,PDE,DD,CONSTRAINTS,SPARSITYPATTERN,VECTOR,dopedim,dealdim,FE, DOFHANDLER>,
  PDE, DD, SPARSITYPATTERN, VECTOR, dopedim, dealdim>,
  SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>& GetStateProblem()
  {
    if(_ts_state_problem == NULL)
    {
      _ts_state_problem = new PRIMALTSPROBLEM<StateProblem<
      OptProblemContainer<FUNCTIONAL_INTERFACE,FUNCTIONAL,PDE,DD,CONSTRAINTS,SPARSITYPATTERN,VECTOR,dopedim,dealdim,FE, DOFHANDLER>,
      PDE, DD, SPARSITYPATTERN, VECTOR, dopedim, dealdim>,
      SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>(OptProblemContainer<FUNCTIONAL_INTERFACE,FUNCTIONAL,PDE,DD,CONSTRAINTS,
      					      SPARSITYPATTERN,VECTOR,dopedim,dealdim,FE, DOFHANDLER>::GetStateProblem());
    }
    return *_ts_state_problem;
  }
  /*******************************************************************************************/

  //FIXME: This should use the GetAdjointProblem of OptProblemContainer once availiable simillar for all calls
  // to ADJOINTTSPROBLEM ...
  ADJOINTTSPROBLEM<OptProblemContainer<FUNCTIONAL_INTERFACE,FUNCTIONAL,PDE,DD,CONSTRAINTS,SPARSITYPATTERN,VECTOR,dopedim,dealdim,FE, DOFHANDLER>,
  SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>& GetAdjointProblem()
  {
    if(_ts_adjoint_problem == NULL)
    {
      _ts_adjoint_problem = new ADJOINTTSPROBLEM<OptProblemContainer<FUNCTIONAL_INTERFACE,FUNCTIONAL,PDE,DD,CONSTRAINTS,SPARSITYPATTERN,VECTOR,dopedim,dealdim,FE, DOFHANDLER>,
      SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>(OptProblemContainer<FUNCTIONAL_INTERFACE,FUNCTIONAL,PDE,DD,CONSTRAINTS,
								 SPARSITYPATTERN,VECTOR,dopedim,dealdim,FE, DOFHANDLER>::GetBaseProblem());
    }
    return *_ts_adjoint_problem;
  }

  /*******************************************************************************************/
  //FIXME: This should use the GetTangentProblem of OptProblemContainer once availiable simillar for all 
  // other related calls
  PRIMALTSPROBLEM<OptProblemContainer<FUNCTIONAL_INTERFACE,FUNCTIONAL,PDE,DD,CONSTRAINTS,SPARSITYPATTERN,VECTOR,dopedim,dealdim,FE, DOFHANDLER>,
  SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>& GetTangentProblem()
  {
    if(_ts_tangent_problem == NULL)
    {
      _ts_tangent_problem = new PRIMALTSPROBLEM<OptProblemContainer<FUNCTIONAL_INTERFACE,FUNCTIONAL,PDE,DD,CONSTRAINTS,SPARSITYPATTERN,VECTOR,dopedim,dealdim,FE, DOFHANDLER>,
      SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>(OptProblemContainer<FUNCTIONAL_INTERFACE,FUNCTIONAL,PDE,DD,CONSTRAINTS,
      					      SPARSITYPATTERN,VECTOR,dopedim,dealdim,FE, DOFHANDLER>::GetBaseProblem());
    }
    return *_ts_tangent_problem;
  }
  /*******************************************************************************************/

  //FIXME: This should use the GetAdjointHessianProblem of OptProblemContainer once availiable simillar for all 
  // other related calls
  ADJOINTTSPROBLEM<OptProblemContainer<FUNCTIONAL_INTERFACE,FUNCTIONAL,PDE,DD,CONSTRAINTS,SPARSITYPATTERN,VECTOR,dopedim,dealdim,FE, DOFHANDLER>,
  SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>& GetAdjointHessianProblem()
  {
    if(_ts_adjoint_hessian_problem == NULL)
    {
      _ts_adjoint_hessian_problem = new ADJOINTTSPROBLEM<OptProblemContainer<FUNCTIONAL_INTERFACE,FUNCTIONAL,PDE,DD,CONSTRAINTS,SPARSITYPATTERN,VECTOR,dopedim,dealdim,FE, DOFHANDLER>,
      SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>(OptProblemContainer<FUNCTIONAL_INTERFACE,FUNCTIONAL,PDE,DD,CONSTRAINTS,
								 SPARSITYPATTERN,VECTOR,dopedim,dealdim,FE, DOFHANDLER>::GetBaseProblem());
    }
    return *_ts_adjoint_hessian_problem;
  }

private:
  PRIMALTSPROBLEM<StateProblem<
      OptProblemContainer<FUNCTIONAL_INTERFACE,FUNCTIONAL,PDE,DD,CONSTRAINTS,SPARSITYPATTERN,VECTOR,dopedim,dealdim,FE, DOFHANDLER>,
      PDE, DD, SPARSITYPATTERN, VECTOR, dopedim, dealdim>,
      SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>* _ts_state_problem;
  ADJOINTTSPROBLEM<OptProblemContainer<FUNCTIONAL_INTERFACE,FUNCTIONAL,PDE,DD,CONSTRAINTS,SPARSITYPATTERN,VECTOR,dopedim,dealdim,FE, DOFHANDLER>, 
      SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>* _ts_adjoint_problem;
  PRIMALTSPROBLEM<OptProblemContainer<FUNCTIONAL_INTERFACE,FUNCTIONAL,PDE,DD,CONSTRAINTS,SPARSITYPATTERN,VECTOR,dopedim,dealdim,FE, DOFHANDLER>,
      SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>* _ts_tangent_problem;
  ADJOINTTSPROBLEM<OptProblemContainer<FUNCTIONAL_INTERFACE,FUNCTIONAL,PDE,DD,CONSTRAINTS,SPARSITYPATTERN,VECTOR,dopedim,dealdim,FE, DOFHANDLER>, 
      SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>* _ts_adjoint_hessian_problem;
};
}
#endif
