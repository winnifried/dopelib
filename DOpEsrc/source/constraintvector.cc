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


#include "constraintvector.h"
#include "dopeexception.h"

#include <iostream>
#include <assert.h>
#include <iomanip>

using namespace dealii;
using namespace DOpE;
/******************************************************/

template<typename VECTOR>
ConstraintVector<VECTOR>::ConstraintVector(const ConstraintVector& ref)
{
  _behavior = ref.GetBehavior();
  _STH      = ref.GetSpaceTimeHandler();
  _sfh_ticket = 0;

  ReInit();
}

/******************************************************/
template<typename VECTOR>
ConstraintVector<VECTOR>::ConstraintVector(const SpaceTimeHandlerBase<VECTOR>* STH, DOpEtypes::VectorStorageType behavior)
{
  _behavior = behavior;
  _STH      = STH;
  _sfh_ticket = 0;

  ReInit();
}


/******************************************************/
template<typename VECTOR>
void ConstraintVector<VECTOR>::ReInit()
{
  _accessor =0;
  if(!GetSpaceTimeHandler()->IsValidControlTicket(_sfh_ticket))
  {
    if(GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
    {
      _local_control_constraint.resize(1,NULL);
      ReSizeLocalSpace(GetSpaceTimeHandler()->GetConstraintNDoFs("local"),
		       GetSpaceTimeHandler()->GetConstraintDoFsPerBlock("local"));
    }
    else
    {
      throw DOpEException("Unknown Behavior " + GetBehavior(),"ConstraintVector<VECTOR>::ReInit");
    }
    
    ReSizeGlobal(GetSpaceTimeHandler()->GetConstraintNDoFs("global"));
    
  }
}

/******************************************************/

template<typename VECTOR>
ConstraintVector<VECTOR>::~ConstraintVector()
{
  if(GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
  {
    for(unsigned int i =0; i<_local_control_constraint.size(); i++)
    {
      assert(_local_control_constraint[i] != NULL);
      delete _local_control_constraint[i];
    }
  }
  else
  {
    throw DOpEException("Unknown Behavior " + GetBehavior(),"ConstraintVector<VECTOR>::~ConstraintVector");
  }
}

/******************************************************/
template<typename VECTOR>
void ConstraintVector<VECTOR>::SetTimeDoFNumber(unsigned int /*time_point*/) const
{
  //TODO if temporal behavior is required one needs to do something here!
  throw DOpEException("Not implemented", "ConstraintVector<VECTOR>::SetTimeDoFNumber");
}

/******************************************************/
template<typename VECTOR>
void ConstraintVector<VECTOR>::SetTime(double /*t*/,const TimeIterator& /*interval*/) const
{
   //TODO if temporal behavior is required one needs to do something here!
  throw DOpEException("Not implemented", "ConstraintVector<VECTOR>::SetTime");
}

/******************************************************/

template<typename VECTOR>
bool ConstraintVector<VECTOR>::HasType(std::string name) const
{
  if(name == "local")
    return true;

  return false;
}

/******************************************************/

template<typename VECTOR>
VECTOR& ConstraintVector<VECTOR>::GetSpacialVector(std::string name)
{
  if(GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
  {
    if(name == "local")
    {
	return *(_local_control_constraint[_accessor]);
    }
    else
    {
      throw DOpEException("Unknown Constraint " + name,"ConstraintVector<VECTOR>::GetSpacialVector");
    }
    
  }
  else
  {
    throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),"ConstraintVector<VECTOR>::GetSpacialVector");
  }
}

/******************************************************/
template<typename VECTOR>
const VECTOR& ConstraintVector<VECTOR>::GetSpacialVector(std::string name) const
{
  if(GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
  {
    if(name == "local")
    {
	return *(_local_control_constraint[_accessor]);
    }
    else
    {
      throw DOpEException("Unknown Constraint " + name,"ConstraintVector<VECTOR>::GetSpacialVector");
    }
    
  }
  else
  {
    throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),"ConstraintVector<VECTOR>::GetSpacialVector");
  }
}

/******************************************************/
template<typename VECTOR>
const dealii::Vector<double>& ConstraintVector<VECTOR>::GetGlobalConstraints() const
{
  return _global_constraint;
}

/******************************************************/
template<typename VECTOR>
dealii::Vector<double>& ConstraintVector<VECTOR>::GetGlobalConstraints()
{
  return _global_constraint;
}

/******************************************************/

template<typename VECTOR>
void ConstraintVector<VECTOR>::ReSizeGlobal(unsigned int ndofs)
{
  _global_constraint.reinit(ndofs);
}

/******************************************************/
namespace DOpE
{

template<>
void DOpE::ConstraintVector<dealii::BlockVector<double> >::ReSizeLocalSpace(
  unsigned int ndofs,
  const std::vector<unsigned int>& dofs_per_block)
{
  if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
  {
    if (_accessor >= 0)
    {
      if (_local_control_constraint[_accessor] == NULL)
        _local_control_constraint[_accessor] = new dealii::BlockVector<double>;

      unsigned int nblocks = dofs_per_block.size();
      if (_local_control_constraint[_accessor]->size() != ndofs)
      {
        _local_control_constraint[_accessor]->reinit(nblocks);
        for (unsigned int i = 0; i < nblocks; i++)
        {
          _local_control_constraint[_accessor]->block(i).reinit(dofs_per_block[i]);
        }
        _local_control_constraint[_accessor]->collect_sizes();
      }
    }
    else
    {
      throw DOpEException("Something is very wrong today!","ConstraintVector<VECTOR>::ReSizeLocalSpace");
    }
  }
  else
  {
    throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),
                        "ConstraintVector<dealii::BlockVector<double> >::ReSizeSpace");
  }

}

/******************************************************/

template<>
void ConstraintVector<dealii::Vector<double> >::ReSizeLocalSpace(unsigned int ndofs, const std::vector<
    unsigned int>& /*dofs_per_block*/)
{
  if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
  {
    if (_accessor >= 0)
    {
      if (_local_control_constraint[_accessor] == NULL)
        _local_control_constraint[_accessor] = new dealii::Vector<double>;

      if (_local_control_constraint[_accessor]->size() != ndofs)
      {
        _local_control_constraint[_accessor]->reinit(ndofs);
      }
    }
    else
    {
      throw DOpEException("Something is very wrong today!","ConstraintVector<VECTOR>::ReSizeLocalSpace");
    }
  }
  else
  {
    throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),
                        "ConstraintVector<dealii::Vector<double> >::ReSizeSpace");
  }

}
}//end of namespace
/******************************************************/

template<typename VECTOR>
void ConstraintVector<VECTOR>::operator=(double value)
{
  if(GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
  {
    for(unsigned int i = 0; i < _local_control_constraint.size(); i++)
    {
      assert(_local_control_constraint[i] != NULL);
      _local_control_constraint[i]->operator=(value);
    }
    if(_global_constraint.size() > 0)
      _global_constraint = value;
  }
  else
  {
    throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),"ConstraintVector<VECTOR>::opterator=");
  }
}

/******************************************************/
template<typename VECTOR>
void ConstraintVector<VECTOR>::operator=(const ConstraintVector& dq)
{
  if(GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
  {
    if(dq._local_control_constraint.size() != _local_control_constraint.size())
    {
      if(dq._local_control_constraint.size() > _local_control_constraint.size())
	{
	  unsigned int s = _local_control_constraint.size();
	  _local_control_constraint.resize(dq._local_control_constraint.size(),NULL);
	  for(unsigned int i = s; i < _local_control_constraint.size(); i++)
	    {
	      assert(_local_control_constraint[i] == NULL);
	      _local_control_constraint[i] = new VECTOR;
	    }
	}
      else
	{
	  for(unsigned int i = _local_control_constraint.size()-1; i >=dq._local_control_constraint.size(); i--)
	    {
	      assert(_local_control_constraint[i] != NULL);
	      delete _local_control_constraint[i];
	      _local_control_constraint[i] = NULL;
	      _local_control_constraint.pop_back();
	    }
	  assert(_local_control_constraint.size() == dq._local_control_constraint.size());
	}
    }

    for(unsigned int i = 0; i < _local_control_constraint.size(); i++)
    {
      assert(_local_control_constraint[i] != NULL);
      assert(dq._local_control_constraint[i] != NULL);
      _local_control_constraint[i]->operator=(*(dq._local_control_constraint[i]));
    }
    
    _global_constraint = dq._global_constraint;

    _accessor = 0;
  }
  else
  {
    throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),"ConstraintVector<VECTOR>::operator=");
  }
}

/******************************************************/
template<typename VECTOR>
void ConstraintVector<VECTOR>::operator+=(const ConstraintVector& dq)
{
  if(GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
  {
    assert(dq._local_control_constraint.size() == _local_control_constraint.size());
    for(unsigned int i = 0; i < _local_control_constraint.size(); i++)
    {
      assert(_local_control_constraint[i] != NULL);
      assert(dq._local_control_constraint[i] != NULL);
      _local_control_constraint[i]->operator+=(*(dq._local_control_constraint[i]));
    }
    if(_global_constraint.size() > 0)
      _global_constraint += dq._global_constraint;
  }
  else
  {
    throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),"ConstraintVector<VECTOR>::operator+=");
  }
}

/******************************************************/
template<typename VECTOR>
void ConstraintVector<VECTOR>::operator*=(double a)
{
  if(GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
  {
    for(unsigned int i = 0; i < _local_control_constraint.size(); i++)
    {
      assert(_local_control_constraint[i] != NULL);
      _local_control_constraint[i]->operator*=(a);
    }
    if(_global_constraint.size() > 0)
      _global_constraint *= a;
  }
  else
  {
    throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),"ConstraintVector<VECTOR>::operator*=");
  }
}

/******************************************************/
template<typename VECTOR>
double ConstraintVector<VECTOR>::operator*(const ConstraintVector& dq) const
{
  if(GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
  {
    assert(dq._local_control_constraint.size() == _local_control_constraint.size());

    double ret = 0.;
    for(unsigned int i = 0; i < _local_control_constraint.size(); i++)
    {
      assert(_local_control_constraint[i] != NULL);
      assert(dq._local_control_constraint[i] != NULL);
      ret += _local_control_constraint[i]->operator*(*(dq._local_control_constraint[i]));
    }
    if(_global_constraint.size() > 0)
      ret += _global_constraint * dq._global_constraint;
    return ret;
  }
  else
  {
    throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),"ConstraintVector<VECTOR>::operator*");
  }
}

/******************************************************/
template<typename VECTOR>
void ConstraintVector<VECTOR>::add(double s, const ConstraintVector& dq)
{
  if(GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
  {
    assert(dq._local_control_constraint.size() == _local_control_constraint.size());

    for(unsigned int i = 0; i < _local_control_constraint.size(); i++)
    {
      assert(_local_control_constraint[i] != NULL);
      assert(dq._local_control_constraint[i] != NULL);
      _local_control_constraint[i]->add(s,*(dq._local_control_constraint[i]));
    }
    if(_global_constraint.size() > 0)
      _global_constraint.add(s,dq._global_constraint);
  }
  else
  {
    throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),"ConstraintVector<VECTOR>::add");
  }
}

/******************************************************/
template<typename VECTOR>
void ConstraintVector<VECTOR>::equ(double s, const ConstraintVector& dq)
{
  if(GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
  {
    assert(dq._local_control_constraint.size() == _local_control_constraint.size());

    for(unsigned int i = 0; i < _local_control_constraint.size(); i++)
    {
      assert(_local_control_constraint[i] != NULL);
      assert(dq._local_control_constraint[i] != NULL);
      _local_control_constraint[i]->equ(s,*(dq._local_control_constraint[i]));
    }
    if(_global_constraint.size() > 0)
      _global_constraint.equ(s,dq._global_constraint);
  }
  else
  {
    throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),"ConstraintVector<VECTOR>::equ");
  }
}

/******************************************************/

template<typename VECTOR>
void ConstraintVector<VECTOR>::PrintInfos(std::stringstream& out)
{
  if(_local_control_constraint.size() ==1)
  {
    out<<"\t"<<_local_control_constraint[0]->size()+_global_constraint.size()<<std::endl;
  }
  else
  {
    if(GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
    {
      out<<"\tNumber of Timepoints: "<<_local_control_constraint.size()<<std::endl;
      unsigned int min_dofs =0;
      unsigned int max_dofs =0;
      unsigned int total_dofs =0;
      unsigned int this_size=0;
      for(unsigned int i = 0; i < _local_control_constraint.size(); i++)
      {
	this_size = _local_control_constraint[i]->size();
	total_dofs += this_size;
	if(i==0)
	  min_dofs = this_size;
	else
	  min_dofs = std::min(min_dofs,this_size);
	max_dofs = std::max(max_dofs,this_size);
      }
      out<<"\tTotal   DoFs: "<<total_dofs+_global_constraint.size()<<std::endl;
      out<<"\tMinimal DoFs: "<<min_dofs+_global_constraint.size()<<std::endl;
      out<<"\tMaximal DoFs: "<<max_dofs+_global_constraint.size()<<std::endl;
    }
    else
    {
      throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),"ConstraintVector<VECTOR>::PrintInfos");
    }
  }
}

/******************************************************/

template<typename VECTOR> 
double ConstraintVector<VECTOR>::Norm(std::string name,std::string restriction) const
{
  double ret = 0.;
  if(name == "infty")
  {
    if( restriction == "all")
    {
      for(unsigned int i = 0; i < _local_control_constraint.size(); i++)
      {
	const VECTOR& tmp = *(_local_control_constraint[i]);
	for( unsigned int j = 0; j < tmp.size(); j++)
	{
	  ret = std::max(ret,std::fabs(tmp(j)));
	}
      }
      for(unsigned int i = 0; i< _global_constraint.size(); i++)
      {
	ret = std::max(std::fabs(_global_constraint(i)),ret);
      }
    }
    else if (restriction == "positive")
    {
      for(unsigned int i = 0; i < _local_control_constraint.size(); i++)
      {
	const VECTOR& tmp = *(_local_control_constraint[i]);
	for( unsigned int j = 0; j < tmp.size(); j++)
	{
	  ret = std::max(ret,std::max(0.,tmp(j)));
	}
      }
      for(unsigned int i = 0; i< _global_constraint.size(); i++)
      {
	ret = std::max(std::max(0.,_global_constraint(i)),ret);
      }
    }
    else
    {
      throw DOpEException("Unknown restriction: " + restriction,"ConstraintVector<VECTOR>::Norm");
    }
  }
  else if(name == "l1")
  {
    if( restriction == "all")
    {
      for(unsigned int i = 0; i < _local_control_constraint.size(); i++)
      {
	const VECTOR& tmp = *(_local_control_constraint[i]);
	for( unsigned int j = 0; j < tmp.size(); j++)
	{
	  ret += std::fabs(tmp(j));
	}
      }
      for(unsigned int i = 0; i< _global_constraint.size(); i++)
      {
	ret += std::fabs(_global_constraint(i));
      }
    }
    else if (restriction == "positive")
    {
      for(unsigned int i = 0; i < _local_control_constraint.size(); i++)
      {
	const VECTOR& tmp = *(_local_control_constraint[i]);
	for( unsigned int j = 0; j < tmp.size(); j++)
	{
	  ret += std::max(0.,tmp(j));
	}
      }
      for(unsigned int i = 0; i< _global_constraint.size(); i++)
      {
	ret += std::max(0.,_global_constraint(i));
      }
    }
    else
    {
      throw DOpEException("Unknown restriction: " + restriction,"ConstraintVector<VECTOR>::Norm");
    }
  }
  else
  {
    throw DOpEException("Unknown type: " + name,"ConstraintVector<VECTOR>::Norm");
  }
  return ret;
}

template<typename VECTOR> 
bool ConstraintVector<VECTOR>::IsFeasible() const
{
  for(unsigned int i = 0; i < _local_control_constraint.size(); i++)
  {
    const VECTOR& tmp = *(_local_control_constraint[i]);
    for( unsigned int j = 0; j < tmp.size(); j++)
    {
      if(tmp(j) > 0.)
	return false;
    }
  }
  for(unsigned int i = 0; i< _global_constraint.size(); i++)
  {
    if(_global_constraint(i) > 0.)
      return false;
  }
  return true;
}

template<typename VECTOR> 
bool ConstraintVector<VECTOR>::IsEpsilonFeasible(double eps) const
{ 
  for(unsigned int i = 0; i < _local_control_constraint.size(); i++)
  {
    const VECTOR& tmp = *(_local_control_constraint[i]);
    for( unsigned int j = 0; j < tmp.size(); j++)
    {
      if(tmp(j) > eps)
	return false;
    }
  }
  for(unsigned int i = 0; i< _global_constraint.size(); i++)
  {
    if(_global_constraint(i) > eps)
      return false;
  }
  return true;
}

template<typename VECTOR> 
bool ConstraintVector<VECTOR>::IsLargerThan(double eps) const
{ 
  for(unsigned int i = 0; i < _local_control_constraint.size(); i++)
  {
    const VECTOR& tmp = *(_local_control_constraint[i]);
    for( unsigned int j = 0; j < tmp.size(); j++)
    {
      if(tmp(j) <= eps)
	return false;
    }
  }
  for(unsigned int i = 0; i< _global_constraint.size(); i++)
  {
    if(_global_constraint(i) <= eps)
      return false;
  }
  return true;
}
        
template<typename VECTOR> 
double ConstraintVector<VECTOR>::Complementarity(const ConstraintVector<VECTOR>& g) const
{
  double ret =0.;
  assert(g._local_control_constraint.size() == _local_control_constraint.size());

  for(unsigned int i = 0; i < _local_control_constraint.size(); i++)
  {
  
    const VECTOR& tmp = *(_local_control_constraint[i]);
    const VECTOR& tmp2 = *(g._local_control_constraint[i]);
    assert(tmp2.size() == tmp.size());
    for( unsigned int j = 0; j < tmp.size(); j++)
    {
      ret += fabs(tmp2[j]*tmp[j]);
    }
  }
  assert(_global_constraint.size() == g._global_constraint.size()); 
  for(unsigned int i = 0; i< _global_constraint.size(); i++)
  {
    ret += fabs(_global_constraint[i]*g._global_constraint[i]);
  }
  return ret;
}

/******************************************************/
/******************************************************/
/******************************************************/

template class ConstraintVector<dealii::Vector<double> >;
template class ConstraintVector<dealii::BlockVector<double> >;
