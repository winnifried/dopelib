#ifndef _INSTAT_OPT_PROBLEM_CONTAINER_
#define _INSTAT_OPT_PROBLEM_CONTAINER_

#include "optproblem.h"

namespace DOpE
{
  template<template<typename BASE_PROB, typename SPARSITYPATTERN, typename VECTOR,int dopedim, int dealdim, typename FE, typename DOFHANDLER> class PRIMALTSPROBLEM, 
    template<typename BASE_PROB, typename SPARSITYPATTERN, typename VECTOR,int dopedim, int dealdim, typename FE, typename DOFHANDLER> class  DUALTSPROBLEM, 
    typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
    typename DD, typename CONSTRAINTS, typename SPARSITYPATTERN,
    typename VECTOR, int dopedim, int dealdim,
    typename FE = DOpEWrapper::FiniteElement<dealdim>,
    typename DOFHANDLER = dealii::DoFHandler<dealdim>>
  class InstatOptProblemContainer : public OptProblem<FUNCTIONAL_INTERFACE,FUNCTIONAL,PDE,DD,CONSTRAINTS,
  SPARSITYPATTERN,VECTOR,dopedim,dealdim,FE, DOFHANDLER>
{
  public:
  InstatOptProblemContainer(FUNCTIONAL& functional,PDE& pde,CONSTRAINTS& constraints,
			    SpaceTimeHandler<FE, DOFHANDLER, SPARSITYPATTERN, VECTOR, dopedim,dealdim>& STH)
  : OptProblem<FUNCTIONAL_INTERFACE,FUNCTIONAL,PDE,DD,CONSTRAINTS,SPARSITYPATTERN,VECTOR,dopedim,dealdim,FE, DOFHANDLER>(
    functional,pde,constraints,STH), _ts_state_problem(NULL)
  {
  }
  
  ~InstatOptProblemContainer()
  {
    if(_ts_state_problem != NULL)
    {
      delete _ts_state_problem;
    }
  }

  void ReInit(std::string algo_type)
  {  
    if(_ts_state_problem != NULL)
    {
      delete _ts_state_problem;
      _ts_state_problem = NULL;
    }

    OptProblem<FUNCTIONAL_INTERFACE,FUNCTIONAL,PDE,DD,CONSTRAINTS,SPARSITYPATTERN,VECTOR,dopedim,dealdim,FE, DOFHANDLER>::ReInit(algo_type);
	
  }

  std::string GetName() const
  {
    return "InstatOptProblemContainer";
  }
  PRIMALTSPROBLEM<StateProblem<
  OptProblem<FUNCTIONAL_INTERFACE,FUNCTIONAL,PDE,DD,CONSTRAINTS,SPARSITYPATTERN,VECTOR,dopedim,dealdim,FE, DOFHANDLER>,
  PDE, DD, SPARSITYPATTERN, VECTOR, dopedim, dealdim>,
  SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>& GetStateProblem()
  {
    if(_ts_state_problem == NULL)
    {
      _ts_state_problem = new PRIMALTSPROBLEM<StateProblem<
      OptProblem<FUNCTIONAL_INTERFACE,FUNCTIONAL,PDE,DD,CONSTRAINTS,SPARSITYPATTERN,VECTOR,dopedim,dealdim,FE, DOFHANDLER>,
      PDE, DD, SPARSITYPATTERN, VECTOR, dopedim, dealdim>,
      SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>(OptProblem<FUNCTIONAL_INTERFACE,FUNCTIONAL,PDE,DD,CONSTRAINTS,
					      SPARSITYPATTERN,VECTOR,dopedim,dealdim,FE, DOFHANDLER>::GetStateProblem());
    }
    return *_ts_state_problem;
  }
  //FIXME: This should use the GetAdjointProblem of OptProblem once availiable simillar for all calls
  // to DUALTSPROBLEM ...
  DUALTSPROBLEM<OptProblem<FUNCTIONAL_INTERFACE,FUNCTIONAL,PDE,DD,CONSTRAINTS,SPARSITYPATTERN,VECTOR,dopedim,dealdim,FE, DOFHANDLER>,
  SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>& GetAdjointProblem()
  {
    if(_ts_dual_problem == NULL)
    {
      _ts_dual_problem = new DUALTSPROBLEM<OptProblem<FUNCTIONAL_INTERFACE,FUNCTIONAL,PDE,DD,CONSTRAINTS,SPARSITYPATTERN,VECTOR,dopedim,dealdim,FE, DOFHANDLER>,
      SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>(OptProblem<FUNCTIONAL_INTERFACE,FUNCTIONAL,PDE,DD,CONSTRAINTS,
								 SPARSITYPATTERN,VECTOR,dopedim,dealdim,FE, DOFHANDLER>::GetBaseProblem());
    }
    return *_ts_dual_problem;
  }
private:
  PRIMALTSPROBLEM<StateProblem<
      OptProblem<FUNCTIONAL_INTERFACE,FUNCTIONAL,PDE,DD,CONSTRAINTS,SPARSITYPATTERN,VECTOR,dopedim,dealdim,FE, DOFHANDLER>,
      PDE, DD, SPARSITYPATTERN, VECTOR, dopedim, dealdim>,
      SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>* _ts_state_problem;
  DUALTSPROBLEM<OptProblem<FUNCTIONAL_INTERFACE,FUNCTIONAL,PDE,DD,CONSTRAINTS,SPARSITYPATTERN,VECTOR,dopedim,dealdim,FE, DOFHANDLER>,
      SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE, DOFHANDLER>* _ts_dual_problem;
};
}
#endif
