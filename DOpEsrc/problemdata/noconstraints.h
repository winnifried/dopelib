#ifndef _NOCONSTRAINT_INTERFACE_H_
#define _NOCONSTRAINT_INTERFACE_H_

#include "constraintinterface.h"

namespace DOpE
{
  /**
   * A template for an arbitrary Constraints.
   * GlobalConstraints are dealt with as a Functional, hence all functions from Functionals are inherited.
   */
  template<template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC, template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC, typename DOFHANDLER, typename VECTOR, int dopedim,int dealdim>
    class NoConstraints : public ConstraintInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim,dealdim>
  {
  public:
  NoConstraints() : ConstraintInterface<CDC,FDC,DOFHANDLER, VECTOR, dopedim,dealdim>() {}
    ~NoConstraints()  {}

    void EvaluateLocalControlConstraints(const VECTOR& control __attribute__((unused)),
					 VECTOR& constraints __attribute__((unused)))
    {
      throw DOpEException("This should never be called!","NoConstraints::EvaluateLocalControlConstraints");
       abort();
    }    
    void GetControlBoxConstraints(VECTOR& lb,VECTOR& ub) const
    {
      lb = -1.e+20;
      ub = 1.e+20;
    }
    bool IsFeasible(const ConstraintVector<VECTOR>& g __attribute__((unused))) const { return true; }
    bool IsLargerThan(const ConstraintVector<VECTOR>& g __attribute__((unused)),double p) const 
    {
      if(p < 0) return true;
      return false;
    }
    bool IsEpsilonFeasible(const ConstraintVector<VECTOR>&  /*g*/, double p) const
    {
      if(p >= 0) return true;
      return false;
    }
    void PostProcessConstraints(ConstraintVector<VECTOR>&  g __attribute__((unused))) const {} 
    double MaxViolation(const ConstraintVector<VECTOR>&  g __attribute__((unused))) const { return 0.; }
    void FeasibilityShift(const ControlVector<VECTOR>& g_hat __attribute__((unused)),
			  ControlVector<VECTOR>&  g __attribute__((unused)),
			  double lambda __attribute__((unused))) const {}
    double Complementarity(const ConstraintVector<VECTOR>&  /*f*/,const ConstraintVector<VECTOR>&  /*g*/) const {return 0.;}

  protected:
  private:

  };
}

#endif
