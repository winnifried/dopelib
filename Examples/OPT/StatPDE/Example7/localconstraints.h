#ifndef _LOCAL_CONSTRAINT_H_
#define _LOCAL_CONSTRAINT_H_

#include "constraintinterface.h"
#include "localconstraintaccessor.h"

namespace DOpE
{
  /**
   * A template for an arbitrary Constraints.
   * GlobalConstraints are dealt with as a Functional, hence all functions from Functionals are inherited.
   */
  template<template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC, template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC, typename DOFHANDLER, typename VECTOR,int dopedim,int dealdim>
    class LocalConstraint : public ConstraintInterface<CDC,FDC,DOFHANDLER,VECTOR,dopedim,dealdim>
  {
  public:
    LocalConstraint(LocalConstraintAccessor& CA) : LCA(CA) 
    {
    }
    ~LocalConstraint()  {}
 
    void EvaluateLocalControlConstraints(const dealii::BlockVector<double>& control,
					 dealii::BlockVector<double>& constraints)
    {
      assert(constraints.block(0).size() == 2*control.block(0).size());

      for(unsigned int i=0; i < control.block(0).size(); i++)
      {
	//Add Control Constraints, such that if control is feasible all  entries are not positive!
	// _rho_min <= control <= _rho_max
	LCA.ControlToLowerConstraint(control,constraints);
	LCA.ControlToUpperConstraint(control,constraints);
      }
    }
    void GetControlBoxConstraints(VECTOR& lb, VECTOR& ub) const
    {
      LCA.GetControlBoxConstraints(lb,ub);
    }
   
   std::string GetType() const 
   { 
       throw DOpEException("Unknown problem_type "+this->GetProblemType(),"LocalConstraints::GetType");
   }
   std::string GetName() const 
   { 
       throw DOpEException("Unknown problem_type "+this->GetProblemType(),"LocalConstraints::GetName");
   } 
        
   dealii::UpdateFlags GetUpdateFlags() const
   {
     return update_values | update_quadrature_points;
   }
       
   bool IsFeasible(const ConstraintVector<VECTOR>& g) const
   {
     for(unsigned int block = 0; block < g.GetSpacialVector("local").n_blocks(); block ++)
     {
       for(unsigned int i = 0; i < g.GetSpacialVector("local").block(block).size(); i++)
       {
	 if(g.GetSpacialVector("local").block(block)(i) > 0.)
	 {
	   return false;
	 }
       }
     }
     assert(g.GetGlobalConstraints().size()==0);
     return true;
   }
   bool IsLargerThan(const ConstraintVector<VECTOR>& g,double p) const
   {
     for(unsigned int block = 0; block < g.GetSpacialVector("local").n_blocks(); block ++)
     {
       for(unsigned int i = 0; i < g.GetSpacialVector("local").block(block).size(); i++)
       {
	 if(g.GetSpacialVector("local").block(block)(i) <= p)
	 {
	   return false;
	 }
       }
     }
     assert(g.GetGlobalConstraints().size()==0);
     return true;
   } 
   bool IsEpsilonFeasible(const ConstraintVector<VECTOR>&  g, double p) const
   {
     for(unsigned int block = 0; block < g.GetSpacialVector("local").n_blocks(); block ++)
     {
       for(unsigned int i = 0; i < g.GetSpacialVector("local").block(block).size(); i++)
       {
	 if(g.GetSpacialVector("local").block(block)(i) >= p)
	 {
	   return false;
	 }
       }
     }
     assert(g.GetGlobalConstraints().size()==0);
     return true;
   }
   double MaxViolation(const ConstraintVector<VECTOR>&  g) const
   {
     double ret = 0.;
     for(unsigned int block = 0; block < g.GetSpacialVector("local").n_blocks(); block ++)
     {
       for(unsigned int i = 0; i < g.GetSpacialVector("local").block(block).size(); i++)
       {
	 ret = (std::max)(ret, g.GetSpacialVector("local").block(block)(i));
       }
     }
     assert(g.GetGlobalConstraints().size()==0);
     return ret;
   }

   void PostProcessConstraints(ConstraintVector<VECTOR>&  g __attribute__((unused))) const {} 

   void FeasibilityShift(const ControlVector<VECTOR>& g_hat,ControlVector<VECTOR>&  g,double lambda) const 
   {
     assert( lambda > 0);
     assert( lambda < 1);
     
     for(unsigned int block = 0; block < g.GetSpacialVector().n_blocks(); block ++)
     {
       for(unsigned int i = 0; i < g.GetSpacialVector().block(block).size(); i++)
       {
	 g.GetSpacialVector().block(block)(i)  = (1.- lambda) *  g.GetSpacialVector().block(block)(i);
	 g.GetSpacialVector().block(block)(i)  += lambda * g_hat.GetSpacialVector().block(block)(i);
       }
     }
   }

   double Complementarity(const ConstraintVector<VECTOR>&  f,const ConstraintVector<VECTOR>&  g) const
   {
     double ret = 0.;
     
     for(unsigned int block = 0; block < g.GetSpacialVector("local").n_blocks(); block ++)
     {
       for(unsigned int i = 0; i < g.GetSpacialVector("local").block(block).size(); i++)
       {
	 ret += fabs(g.GetSpacialVector("local").block(block)(i) * f.GetSpacialVector("local").block(block)(i));
       }
     }
     assert(g.GetGlobalConstraints().size()==0);
     return ret;
   }

  private: 
    LocalConstraintAccessor& LCA;
    
  };
}

#endif
