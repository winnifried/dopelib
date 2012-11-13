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
      _vol_max = 0.5;
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

    double Value(const CDC<DOFHANDLER,VECTOR,dealdim>& cdc)
    {
      if(this->GetProblemType() == "global_constraints" && this->GetProblemTypeNum()==0)
      {
	const DOpEWrapper::FEValues<dealdim> & control_fe_values = cdc.GetFEValuesControl();
	unsigned int n_q_points = cdc.GetNQPoints();
 	
	double ret = 0.;
	{
	  _qvalues.resize(n_q_points,Vector<double>(1));
	  cdc.GetValuesControl("control",_qvalues);
	}
	for(unsigned int q_point = 0; q_point < n_q_points; q_point++)
	{
	  ret += (_qvalues[q_point](0) - _vol_max)* control_fe_values.JxW(q_point);
	}
	return ret;
      }
      else
      {
	return 0;
      }
    }

    void Value_U(const CDC<DOFHANDLER,VECTOR,dealdim>& cdc __attribute__((unused)),
		 dealii::Vector<double> &local_cell_vector __attribute__((unused)), 
		 double scale __attribute__((unused)))
    {
    }

    void Value_Q(const CDC<DOFHANDLER,VECTOR,dealdim>& cdc,
		dealii::Vector<double> &local_cell_vector, 
		double scale)
   {
     if(this->GetProblemType() == "global_constraint_gradient" && this->GetProblemTypeNum()==0)
      {
	const DOpEWrapper::FEValues<dealdim> & control_fe_values = cdc.GetFEValuesControl();
	unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
	unsigned int n_q_points = cdc.GetNQPoints();
 
	for(unsigned int q_point = 0; q_point < n_q_points; q_point++)
	{
	  for(unsigned int i = 0; i < n_dofs_per_cell; i++)
	  { 
	    local_cell_vector(i) += scale*control_fe_values.shape_value (i, q_point)* control_fe_values.JxW(q_point);
	  }
	}
      }
     else
     {
       abort();
     }
   } 
 
   void Value_UU(const CDC<DOFHANDLER,VECTOR,dealdim>& cdc __attribute__((unused)),
		 dealii::Vector<double> &local_cell_vector __attribute__((unused)), 
		 double scale __attribute__((unused)))
   {
   }
   void Value_QU(const CDC<DOFHANDLER,VECTOR,dealdim>& cdc __attribute__((unused)),
		 dealii::Vector<double> &local_cell_vector __attribute__((unused)), 
		 double scale __attribute__((unused)))
   {
   }
   void Value_UQ(const CDC<DOFHANDLER,VECTOR,dealdim>& cdc __attribute__((unused)),
		 dealii::Vector<double> &local_cell_vector __attribute__((unused)), 
		 double scale __attribute__((unused)))
   {
   }
   void Value_QQ(const CDC<DOFHANDLER,VECTOR,dealdim>& cdc __attribute__((unused)),
		 dealii::Vector<double> &local_cell_vector __attribute__((unused)), 
		 double scale __attribute__((unused)))
   {
   }
   
   std::string GetType() const 
   { 
     if((this->GetProblemType() == "global_constraints"||this->GetProblemType() == "global_constraint_gradient"  ) && this->GetProblemTypeNum()==0)
       return "domain";
       else
       throw DOpEException("Unknown problem_type "+this->GetProblemType(),"LocalConstraints::GetType");
   }
   std::string GetName() const 
   { 
       if((this->GetProblemType() == "global_constraints"||this->GetProblemType() == "global_constraint_gradient"  ) && this->GetProblemTypeNum()==0)
	 return "volume_constraint";
       else
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
	   //std::cout<<"Failure in block "<<block<<" in index "<<i<<" with value "<<g.GetSpacialVector("local").block(block)(i)<<std::endl;
	   return false;
	 }
       }
     }
     for(unsigned int i = 0; i < g.GetGlobalConstraints().size(); i++)
     {
       if(g.GetGlobalConstraints()(i) > 0)
	 {
	   //std::cout<<"Failure in global constraints in index "<<i<<" with value "<<g.GetGlobalConstraints()(i)<<std::endl;
	   return false;
	 } 
     }
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
	   //std::cout<<"Failure in block "<<block<<" in index "<<i<<" with value "<<g.GetSpacialVector("local").block(block)(i)<<std::endl;
	   return false;
	 }
       }
     }
     for(unsigned int i = 0; i < g.GetGlobalConstraints().size(); i++)
     {
       if(g.GetGlobalConstraints()(i) <= p)
	 {
	   //std::cout<<"Failure in global constraints in index "<<i<<" with value "<<g.GetGlobalConstraints()(i)<<std::endl;
	   return false;
	 } 
     }
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
	   //std::cout<<"Failure in block "<<block<<" in index "<<i<<" with value "<<g.GetSpacialVector("local").block(block)(i)<<std::endl;
	   return false;
	 }
       }
     }
     for(unsigned int i = 0; i < g.GetGlobalConstraints().size(); i++)
     {
       if(g.GetGlobalConstraints()(i) >= p)
	 {
	   //std::cout<<"Failure in global constraints in index "<<i<<" with value "<<g.GetGlobalConstraints()(i)<<std::endl;
	   return false;
	 } 
     }
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
     for(unsigned int i = 0; i < g.GetGlobalConstraints().size(); i++)
     {
       ret = (std::max)(ret,g.GetGlobalConstraints()(i));
     }
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
     for(unsigned int i = 0; i < g.GetGlobalConstraints().size(); i++)
     {
       ret += fabs(g.GetGlobalConstraints()(i) *f.GetGlobalConstraints()(i) );
     }
     return ret;
   }

  private: 
    double _vol_max;
    std::vector<dealii::Vector<double> > _qvalues;
    LocalConstraintAccessor& LCA;
    
  };
}

#endif
