#ifndef _INSTAT_OPT_PROBLEM_CONTAINER_
#define _INSTAT_OPT_PROBLEM_CONTAINER_

#include "optproblem.h"

namespace DOpE
{
  template<typename PRIMALTSPROBLEM, typename FUNCTIONAL_INTERFACE, typename FUNCTIONAL, typename PDE,
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
  PRIMALTSPROBLEM& GetStateProblem()
  {
    if(_ts_state_problem == NULL)
    {
      _ts_state_problem = new PRIMALTSPROBLEM(OptProblem<FUNCTIONAL_INTERFACE,FUNCTIONAL,PDE,DD,CONSTRAINTS,
					      SPARSITYPATTERN,VECTOR,dopedim,dealdim,FE, DOFHANDLER>::GetStateProblem());
    }
    return *_ts_state_problem;
  }
private:
  PRIMALTSPROBLEM* _ts_state_problem;
};
}
#endif
