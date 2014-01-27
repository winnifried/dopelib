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


#include "controlvector.h"
#include "dopeexception.h"

#include <iostream>
#include <assert.h>
#include <iomanip>

using namespace dealii;
using namespace DOpE;
/******************************************************/

template<typename VECTOR>
ControlVector<VECTOR>::ControlVector(const ControlVector& ref)
{
  _behavior = ref.GetBehavior();
  _STH      = ref.GetSpaceTimeHandler();
  _sfh_ticket = 0;
  _c_type = _STH->GetControlType();
 
  //Check consistency
  if(_c_type == DOpEtypes::ControlType::initial || _c_type == DOpEtypes::ControlType::stationary)
  {
    if(_behavior!= DOpEtypes::VectorStorageType::fullmem)
    {
      throw DOpEException("Storage behavior: " + DOpEtypesToString(GetBehavior()) + 
			  " is not compatible with control type: " + DOpEtypesToString(_c_type),
			  "ControlVector<VECTOR>::ControlVector<VECTOR>");
    }
  }
  if(_behavior != DOpEtypes::VectorStorageType::fullmem)
  {
    throw DOpEException("Storage behavior: " + DOpEtypesToString(GetBehavior()) + 
			" not implemented",
			"ControlVector<VECTOR>::ControlVector<VECTOR>");
  }

  ReInit();
}

/******************************************************/
template<typename VECTOR>
ControlVector<VECTOR>::ControlVector(const SpaceTimeHandlerBase<VECTOR>* STH, DOpEtypes::VectorStorageType behavior)
{
  _behavior = behavior;
  _STH      = STH;
  _sfh_ticket = 0;
  _c_type = _STH->GetControlType();

  //Check consistency
  if(_c_type == DOpEtypes::ControlType::initial || _c_type == DOpEtypes::ControlType::stationary)
  {
    if(_behavior!= DOpEtypes::VectorStorageType::fullmem)
    {
      throw DOpEException("Storage behavior: " + DOpEtypesToString(GetBehavior()) + 
			  " is not compatible with control type: " + DOpEtypesToString(_c_type),
			  "ControlVector<VECTOR>::ControlVector<VECTOR>");
    }
  }
  if(_behavior != DOpEtypes::VectorStorageType::fullmem)
  {
    throw DOpEException("Storage behavior: " + DOpEtypesToString(GetBehavior()) + 
			" not implemented",
			"ControlVector<VECTOR>::ControlVector<VECTOR>");
  }
  
  ReInit();
}


/******************************************************/
template<typename VECTOR>
void ControlVector<VECTOR>::ReInit()
{    
  _accessor =0;

  if(!GetSpaceTimeHandler()->IsValidControlTicket(_sfh_ticket))
  {
    if(GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
    {
      
      _control.resize(1,NULL);
      ReSizeSpace(GetSpaceTimeHandler()->GetControlNDoFs(),
		  GetSpaceTimeHandler()->GetControlDoFsPerBlock());
    }
    else
    {
      throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),"ControlVector<VECTOR>::ReInit");
    }

    _lock= false;
  }
}

/******************************************************/

template<typename VECTOR>
ControlVector<VECTOR>::~ControlVector()
{
  if(GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
  {
    for(unsigned int i =0; i<_control.size(); i++)
    {
      assert(_control[i] != NULL);
      delete _control[i];
    }
  }
  else
  {
    throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),"ControlVector<VECTOR>::~ControlVector");
  }
}

/******************************************************/
template<typename VECTOR>
void ControlVector<VECTOR>::SetTimeDoFNumber(unsigned int time_point) const
{
  if(_c_type == DOpEtypes::ControlType::nonstationary)
  {
    //TODO if temporal behavior is required one needs to do something here!
    throw DOpEException("Control type: " + DOpEtypesToString(_c_type) + " is not implemented.",
			"ControlVector<VECTOR>::SetTimeDoFNumber");
  }
  else if(_c_type == DOpEtypes::ControlType::initial)
  {
    if(time_point != 0)
      throw DOpEException("With control type: " + DOpEtypesToString(_c_type) + "the time should never differ from 0.",
			  "ControlVector<VECTOR>::SetTimeDoFNumber");
  }
}

/******************************************************/
template<typename VECTOR>
void ControlVector<VECTOR>::SetTime(double /*t*/,  const TimeIterator& /*interval*/) const
{
  if(_c_type == DOpEtypes::ControlType::nonstationary)
  {
    //TODO if temporal behavior is required one needs to do something here!
    throw DOpEException("control type: " + DOpEtypesToString(_c_type) + " is not implemented.",
			"ControlVector<VECTOR>::Time");
  }
}

/******************************************************/

template<typename VECTOR>
VECTOR& ControlVector<VECTOR>::GetSpacialVector()
{
  if(_c_type == DOpEtypes::ControlType::stationary || _c_type == DOpEtypes::ControlType::initial)
  {
    return *(_control[_accessor]);
  }
  else
  {
    throw DOpEException("Control type: " + DOpEtypesToString(_c_type) + " is not implemented.",
			"ControlVector<VECTOR>::GetSpacialVector");
  }
}

/******************************************************/
template<typename VECTOR>
const VECTOR& ControlVector<VECTOR>::GetSpacialVector() const
{
  if(_c_type == DOpEtypes::ControlType::stationary || _c_type == DOpEtypes::ControlType::initial)
  {
    return *(_control[_accessor]);
  }
  else
  {
    throw DOpEException("Control type: " + DOpEtypesToString(_c_type) + " is not implemented.",
			"ControlVector<VECTOR>::GetSpacialVector");
  }
}

/******************************************************/
template<typename VECTOR>
const Vector<double>& ControlVector<VECTOR>::GetSpacialVectorCopy() const
{
  if(_lock)
  {
    throw DOpEException("Trying to create a new copy while the old is still in use!","ControlVector:GetSpacialVectorCopy");
  }
  _lock = true;
  if(_c_type == DOpEtypes::ControlType::stationary || _c_type == DOpEtypes::ControlType::initial)
  {
    _copy_control = *(_control[_accessor]);
    return _copy_control;
  }
  else
  {
    throw DOpEException("Control type: " + DOpEtypesToString(_c_type) + " is not implemented.",
			"ControlVector<VECTOR>::GetSpacialVectorCopy");
  }
}

/******************************************************/

namespace DOpE
{
template<>
void ControlVector<dealii::BlockVector<double> >::ReSizeSpace(unsigned int ndofs, const std::vector<unsigned int>& dofs_per_block)
{
  if(GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
  {
    if(_accessor >= 0)
    {
      bool existed = true;
      if(_control[_accessor] == NULL)
      {
	_control[_accessor] = new dealii::BlockVector<double>;
	existed = false; 
      }
      unsigned int nblocks = dofs_per_block.size();
      if(_control[_accessor]->size() != ndofs)
      {
	if(existed)
	{
	  _local_control = *(_control[_accessor]);
	}
	     
	_control[_accessor]->reinit(nblocks);
	for(unsigned int i = 0; i < nblocks; i++)
	{
	  _control[_accessor]->block(i).reinit(dofs_per_block[i]);
	}
	_control[_accessor]->collect_sizes();
            
	if(existed)
	{
	  GetSpaceTimeHandler()->SpatialMeshTransferControl(_local_control,*(_control[_accessor]));
	}
      }
    }
    else
    {
      throw DOpEException("Something is very wrong today!","ControlVector<VECTOR>::ReSizeSpace");
    }
  }
  else
  {
    throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),"ControlVector<VECTOR>::ReSizeSpace");
  }

}

/******************************************************/
template<>
void ControlVector<dealii::Vector<double> >::ReSizeSpace(unsigned int ndofs, const std::vector<unsigned int>& /*dofs_per_block*/)
{
  if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
  {
    if (_accessor >= 0)
    {
      bool existed = true;
      if (_control[_accessor] == NULL)
      {
	_control[_accessor] = new dealii::Vector<double>;
	existed = false;
      }
      if (_control[_accessor]->size() != ndofs)
      {	
	if(existed)
	{
	  _local_control = *(_control[_accessor]);
	}
        
	_control[_accessor]->reinit(ndofs);

	if(existed)
	{
	  GetSpaceTimeHandler()->SpatialMeshTransferControl(_local_control,*(_control[_accessor]));
	}
      }
    }
    else
    {
      throw DOpEException("Something is very wrong today!","ControlVector<VECTOR>::ReSizeSpace");
    }
  }
  else
  {
    throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()), "ControlVector<VECTOR>::ReSizeSpace");
  }

}

}//end of namespace
/******************************************************/

template<typename VECTOR>
void ControlVector<VECTOR>::operator=(double value)
{
  if(GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
  {
    for(unsigned int i = 0; i < _control.size(); i++)
    {
      assert(_control[i] != NULL);
      _control[i]->operator=(value);
    }
  }
  else
  {
    throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),"ControlVector<VECTOR>::opterator=");
  }
}

/******************************************************/
template<typename VECTOR>
void ControlVector<VECTOR>::operator=(const ControlVector& dq)
{
  if(GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
  {
    if(dq._control.size() != _control.size())
    {
      if(dq._control.size() > _control.size())
	{
	  unsigned int s = _control.size();
	  _control.resize(dq._control.size(),NULL);
	  for(unsigned int i = s; i < _control.size(); i++)
	    {
	      assert(_control[i] == NULL);
	      _control[i] = new VECTOR;
	    }
	}
      else
	{
	  for(unsigned int i = _control.size()-1; i >=dq._control.size(); i--)
	    {
	      assert(_control[i] != NULL);
	      delete _control[i];
	      _control[i] = NULL;
	      _control.pop_back();
	    }
	  assert(_control.size() == dq._control.size());
	}
    }

    for(unsigned int i = 0; i < _control.size(); i++)
    {
      assert(_control[i] != NULL);
      assert(dq._control[i] != NULL);
      _control[i]->operator=(*(dq._control[i]));
    }
    _accessor = 0;
  }
  else
  {
    throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),"ControlVector<VECTOR>::operator=");
  }
}

/******************************************************/
template<typename VECTOR>
void ControlVector<VECTOR>::operator+=(const ControlVector& dq)
{
  if(GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
  {
    assert(dq._control.size() == _control.size());
    for(unsigned int i = 0; i < _control.size(); i++)
    {
      assert(_control[i] != NULL);
      assert(dq._control[i] != NULL);
      _control[i]->operator+=(*(dq._control[i]));
    }
  }
  else
  {
    throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),"ControlVector<VECTOR>::operator+=");
  }
}

/******************************************************/
template<typename VECTOR>
void ControlVector<VECTOR>::operator*=(double a)
{
  if(GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
  {
    for(unsigned int i = 0; i < _control.size(); i++)
    {
      assert(_control[i] != NULL);
      _control[i]->operator*=(a);
    }
  }
  else
  {
    throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),"ControlVector<VECTOR>::operator*=");
  }
}

/******************************************************/
template<typename VECTOR>
double ControlVector<VECTOR>::operator*(const ControlVector& dq) const
{
  if(GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
  {
    assert(dq._control.size() == _control.size());

    double ret = 0.;
    for(unsigned int i = 0; i < _control.size(); i++)
    {
      assert(_control[i] != NULL);
      assert(dq._control[i] != NULL);
      ret += _control[i]->operator*(*(dq._control[i]));
    }
    return ret;
  }
  else
  {
    throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),"ControlVector<VECTOR>::operator*");
  }
}

/******************************************************/
template<typename VECTOR>
void ControlVector<VECTOR>::add(double s, const ControlVector& dq)
{
  if(GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
  {
    assert(dq._control.size() == _control.size());

    for(unsigned int i = 0; i < _control.size(); i++)
    {
      assert(_control[i] != NULL);
      assert(dq._control[i] != NULL);
      _control[i]->add(s,*(dq._control[i]));
    }
  }
  else
  {
    throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),"ControlVector<VECTOR>::add");
  }
}

/******************************************************/
template<typename VECTOR>
void ControlVector<VECTOR>::equ(double s, const ControlVector& dq)
{
  if(GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
  {
    assert(dq._control.size() == _control.size());

    for(unsigned int i = 0; i < _control.size(); i++)
    {
      assert(_control[i] != NULL);
      assert(dq._control[i] != NULL);
      _control[i]->equ(s,*(dq._control[i]));
    }
  }
  else
  {
    throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),"ControlVector<VECTOR>::equ");
  }
}

/******************************************************/
template<typename VECTOR>
void ControlVector<VECTOR>::max(const ControlVector& dq)
{
  if(GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
  {
    assert(dq._control.size() == _control.size());

    for(unsigned int i = 0; i < _control.size(); i++)
    {
      assert(_control[i] != NULL);
      assert(dq._control[i] != NULL);
      VECTOR& t = *(_control[i]);
      const VECTOR& tn = *(dq._control[i]);
      assert(t.size() == tn.size());
      for(unsigned int j = 0; j < t.size() ; j++)
      {
	t(j) = std::max(t(j),tn(j));
      }
    }
  }
  else
  {
    throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),"ControlVector<VECTOR>::max");
  }
}


/******************************************************/
template<typename VECTOR>
void ControlVector<VECTOR>::min(const ControlVector& dq)
{
  if(GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
  {
    assert(dq._control.size() == _control.size());

    for(unsigned int i = 0; i < _control.size(); i++)
    {
      assert(_control[i] != NULL);
      assert(dq._control[i] != NULL);
      VECTOR& t = *(_control[i]);
      const VECTOR& tn = *(dq._control[i]);
      assert(t.size() == tn.size());
      for(unsigned int j = 0; j < t.size() ; j++)
      {
	t(j) = std::min(t(j),tn(j));
      }
    }
  }
  else
  {
    throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),"ControlVector<VECTOR>::min");
  }
}
/******************************************************/
template<typename VECTOR>
void ControlVector<VECTOR>::comp_mult(const ControlVector& dq)
{
  if(GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
  {
    assert(dq._control.size() == _control.size());

    for(unsigned int i = 0; i < _control.size(); i++)
    {
      assert(_control[i] != NULL);
      assert(dq._control[i] != NULL);
      VECTOR& t = *(_control[i]);
      const VECTOR& tn = *(dq._control[i]);
      assert(t.size() == tn.size());
      for(unsigned int j = 0; j < t.size() ; j++)
      {
	t(j) = t(j)*tn(j);
      }
    }
  }
  else
  {
    throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),"ControlVector<VECTOR>::comp_mult");
  }
}

/******************************************************/
template<typename VECTOR>
void ControlVector<VECTOR>::comp_invert()
{
  if(GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
  {
    for(unsigned int i = 0; i < _control.size(); i++)
    {
      assert(_control[i] != NULL);
      VECTOR& t = *(_control[i]);
      for(unsigned int j = 0; j < t.size() ; j++)
      {
	t(j) = 1./t(j);
      }
    }
  }
  else
  {
    throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),"ControlVector<VECTOR>::comp_invert");
  }
}

/******************************************************/
template<typename VECTOR>
void ControlVector<VECTOR>::init_by_sign(double smaller, double larger, double unclear, double TOL)
{
  if(GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
  {
    
    for(unsigned int i = 0; i < _control.size(); i++)
    {
      assert(_control[i] != NULL);
      VECTOR& t = *(_control[i]);
      for(unsigned int j = 0; j < t.size() ; j++)
      {
	if(t(j) < -TOL)
	{
	  t(j) = smaller;
	}
	else if(t(j) > TOL)
	{
	  t(j) = larger;
	}
	else
	{
	  t(j) = unclear;
	}       
      }
    }
  }
  else
  {
    throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),"ControlVector<VECTOR>::init_by_sign");
  }
}
/******************************************************/

template<typename VECTOR>
void ControlVector<VECTOR>::PrintInfos(std::stringstream& out)
{
  if(_control.size() ==1)
  {
    out<<"\t"<<_control[0]->size()<<std::endl;
  }
  else
  {
    if(GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
    {
      out<<"\tNumber of Timepoints: "<<_control.size()<<std::endl;
      unsigned int min_dofs =0;
      unsigned int max_dofs =0;
      unsigned int total_dofs =0;
      unsigned int this_size=0;
      for(unsigned int i = 0; i < _control.size(); i++)
      {
	this_size = _control[i]->size();
	total_dofs += this_size;
	if(i==0)
	  min_dofs = this_size;
	else
	  min_dofs = std::min(min_dofs,this_size);
	max_dofs = std::max(max_dofs,this_size);
      }
      out<<"\tTotal   DoFs: "<<total_dofs<<std::endl;
      out<<"\tMinimal DoFs: "<<min_dofs<<std::endl;
      out<<"\tMaximal DoFs: "<<max_dofs<<std::endl;
    }
    else
    {
      throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),"ControlVector<VECTOR>::PrintInfos");
    }
  }
}

/******************************************************/

template<typename VECTOR> 
double ControlVector<VECTOR>::Norm(std::string name,std::string restriction) const
{
  double ret = 0.;
  if(name == "infty")
  {
    if( restriction == "all")
    {
      for(unsigned int i = 0; i < _control.size(); i++)
      {
	const VECTOR& tmp = *(_control[i]);
	for( unsigned int j = 0; j < tmp.size(); j++)
	{
	  ret = std::max(ret,std::fabs(tmp(j)));
	}
      }
    }
    else if (restriction == "positive")
    {
      for(unsigned int i = 0; i < _control.size(); i++)
      {
	const VECTOR& tmp = *(_control[i]);
	for( unsigned int j = 0; j < tmp.size(); j++)
	{
	  ret = std::max(ret,std::max(0.,tmp(j)));
	}
      }
    }
    else
    {
      throw DOpEException("Unknown restriction: " + restriction,"ControlVector<VECTOR>::Norm");
    }
  }
  else if(name == "l1")
  {
    if( restriction == "all")
    {
      for(unsigned int i = 0; i < _control.size(); i++)
      {
	const VECTOR& tmp = *(_control[i]);
	for( unsigned int j = 0; j < tmp.size(); j++)
	{
	  ret += std::fabs(tmp(j));
	}
      }
    }
    else if (restriction == "positive")
    {
      for(unsigned int i = 0; i < _control.size(); i++)
      {
	const VECTOR& tmp = *(_control[i]);
	for( unsigned int j = 0; j < tmp.size(); j++)
	{
	  ret += std::max(0.,tmp(j));
	}
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


/******************************************************/
/******************************************************/

template class ControlVector<dealii::Vector<double> >;
template class ControlVector<dealii::BlockVector<double> >;
