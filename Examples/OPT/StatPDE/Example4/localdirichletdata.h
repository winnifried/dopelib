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

#ifndef _LOCAL_DIRICHLET_INTERFAC_H_
#define _LOCAL_DIRICHLET_INTERFAC_H_

#include "dirichletdatainterface.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

template<typename VECTOR, int dealdim>
  class LocalDirichletData : public DirichletDataInterface<VECTOR,dealdim>
  {
  public:

     double Data(
//                 const DOpEWrapper::DoFHandler<dopedim> * control_dof_handler __attribute__((unused)),
//		 const DOpEWrapper::DoFHandler<dealdim> *state_dof_handler __attribute__((unused)),
		 const std::map<std::string, const dealii::Vector<double>* > *param_values,
		 const std::map<std::string, const VECTOR* > *domain_values __attribute__((unused)),
		 unsigned int color __attribute__((unused)),
		 const dealii::Point<dealdim>& point __attribute__((unused)),
		 unsigned int component) const
     {
       _qvalues.reinit(5);
       GetParams(*param_values,"control",_qvalues);

       if(component == 0)
       {
	 if(color == 0)
	   return _qvalues(0);
	 else if(color == 1)
	   return _qvalues(1);
	 else if(color == 2)
	   return _qvalues(2);
	 else if(color ==3)
	   return _qvalues(3);
       }
       // else
       {
	 return _qvalues(4)*_qvalues(4)*_qvalues(4);
       }
     }

   double Data_Q(
//                 const DOpEWrapper::DoFHandler<dopedim> * control_dof_handler __attribute__((unused)),
//		 const DOpEWrapper::DoFHandler<dealdim> *state_dof_handler __attribute__((unused)),
		 const std::map<std::string, const dealii::Vector<double>* > *param_values,
		 const std::map<std::string, const VECTOR* > *domain_values __attribute__((unused)),
		 unsigned int color __attribute__((unused)),
		 const dealii::Point<dealdim>& point __attribute__((unused)),
		 unsigned int component)  const
   {
     _qvalues.reinit(5);
     GetParams(*param_values,"control",_qvalues);
     _dqvalues.reinit(5);
     GetParams(*param_values,"dq",_dqvalues);

       if(component == 0)
       {
	 if(color == 0)
	   return _dqvalues(0);
	 else if(color == 1)
	   return _dqvalues(1);
	 else if(color == 2)
	   return _dqvalues(2);
	 else if(color ==3)
	   return _dqvalues(3);
       }
       // else
       {
	 return 3*_qvalues(4)*_qvalues(4)*_dqvalues(4);
       }
   }

   void Data_QT (
//                 const DOpEWrapper::DoFHandler<dopedim> * control_dof_handler __attribute__((unused)),
//		 const DOpEWrapper::DoFHandler<dealdim> *state_dof_handler __attribute__((unused)),
		 const std::map<std::string, const dealii::Vector<double>* > *param_values,
		 const std::map<std::string, const VECTOR* > *domain_values,
		 unsigned int color __attribute__((unused)),
		 const dealii::Point<dealdim>& point __attribute__((unused)),
		 unsigned int component,
		 unsigned int  dof_number,
		 dealii::Vector<double>& local_vector) const
   {
     _qvalues.reinit(5);
     GetParams(*param_values,"control",_qvalues);
     GetValues(*domain_values,"adjoint_residual",_resvalues,dof_number);
     if(component == 0)
     {
       if(color == 0)
	 local_vector(0) += _resvalues;
       if(color == 1)
	 local_vector(1) += _resvalues;
       else if(color == 2)
	 local_vector(2) += _resvalues;
       else if(color ==3)
	 local_vector(3) += _resvalues;
     }
     if(component==1)
       local_vector(4) += 3*_qvalues(4)*_qvalues(4)*_resvalues;
   }

   void Data_QQT (
//                  const DOpEWrapper::DoFHandler<dopedim> * control_dof_handler __attribute__((unused)),
//		  const DOpEWrapper::DoFHandler<dealdim> *state_dof_handler __attribute__((unused)),
		  const std::map<std::string, const dealii::Vector<double>* > *param_values,
		  const std::map<std::string, const VECTOR* > *domain_values,
		  unsigned int color __attribute__((unused)),
		  const dealii::Point<dealdim>& point __attribute__((unused)),
		  unsigned int component,
		  unsigned int  dof_number,
		  dealii::Vector<double>& local_vector) const
   {
     _qvalues.reinit(5);
     GetParams(*param_values,"control",_qvalues);
     _dqvalues.reinit(5);
     GetParams(*param_values,"dq",_dqvalues);
     GetValues(*domain_values,"hessian_residual",_resvalues,dof_number);
     if(component==1)
       local_vector(4) +=6*_qvalues(4)*_dqvalues(4)*_resvalues;
   }

   unsigned int n_components() const { return 2;}
   bool NeedsControl() const { return  true; }
   /*****************************************************************************/
   // protected:
   inline void GetValues(const map<string, const VECTOR* >& domain_values,string name, double& values, unsigned int dof_number) const
    {
      typename map<string, const BlockVector<double>* >::const_iterator it = domain_values.find(name);
      if(it == domain_values.end())
	{
	  throw DOpEException("Did not find " + name,"LocalPDE::GetValues");
	}
      values = (*(it->second))(dof_number);
    }
    inline void GetParams(const map<string, const Vector<double>* >& param_values,string name, Vector<double>& values) const
    {
      typename map<string, const Vector<double>* >::const_iterator it = param_values.find(name);
      if(it == param_values.end())
	{
	  throw DOpEException("Did not find " + name,"LocalPDE::GetValues");
	}
      values = *(it->second);
    }
  private:


    /*****************************************************************************/

    mutable Vector<double> _qvalues;
    mutable Vector<double> _dqvalues;
    mutable double _resvalues;
  };
#endif
