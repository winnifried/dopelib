#ifndef _CONSTRAINT_INTERFACE_H_
#define _CONSTRAINT_INTERFACE_H_

#include <map>
#include <string>

#include <fe/fe_system.h>
#include <fe/fe_values.h>
#include <fe/mapping.h>

#include "fevalues_wrapper.h"
#include "dofhandler_wrapper.h"
#include "functionalinterface.h"

namespace DOpE
{
  /**
   * A template for an arbitrary Constraint.
   * GlobalConstraints are dealt with as a Functional, hence all functions from Functionals are inherited.
   */
  template<template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC, template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC, typename DOFHANDLER, typename VECTOR,int dopedim,int dealdim>
    class ConstraintInterface :  public FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim,dealdim>
  {
  public:
    ConstraintInterface() {}
    ~ConstraintInterface()  {}

    virtual void EvaluateLocalControlConstraints(const VECTOR& control,
						 VECTOR& constraints) = 0;
    virtual void GetControlBoxConstraints(VECTOR& lb,VECTOR& ub) const = 0;

    void SetProblemType(std::string type,unsigned int num)
    {
      _problem_type_num = num;
      _problem_type = type;
    }
    
    virtual bool IsFeasible(const ConstraintVector<VECTOR>& g) const=0;
    virtual bool IsLargerThan(const ConstraintVector<VECTOR>& g,double p) const=0;
    virtual bool IsEpsilonFeasible(const ConstraintVector<VECTOR>&  g, double p) const=0;
    virtual void PostProcessConstraints(ConstraintVector<VECTOR>&  g) const =0;
    virtual double MaxViolation(const ConstraintVector<VECTOR>&  g) const =0;
    virtual void FeasibilityShift(const ControlVector<VECTOR>& g_hat,ControlVector<VECTOR>&  g,double lambda) const=0;
    virtual double Complementarity(const ConstraintVector<VECTOR>&  f,const ConstraintVector<VECTOR>&  g) const=0;

  protected:
    std::string GetProblemType() const {return _problem_type; }
    unsigned int GetProblemTypeNum() const { return _problem_type_num; }
  private:
    std::string _problem_type;
    unsigned int _problem_type_num;
  };
}

#endif
