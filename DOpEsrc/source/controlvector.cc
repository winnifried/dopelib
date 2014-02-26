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
      if(_c_type == DOpEtypes::ControlType::initial || _c_type == DOpEtypes::ControlType::stationary)
      {
	_control.resize(1,NULL);
	ReSizeSpace(GetSpaceTimeHandler()->GetControlNDoFs(),
		    GetSpaceTimeHandler()->GetControlDoFsPerBlock());
      }
      else
      {
	if( _c_type == DOpEtypes::ControlType::nonstationary)
	{
	  //Time dofs for state and control are equal!
	  _control.resize(GetSpaceTimeHandler()->GetMaxTimePoint() + 1, NULL);
	  for (unsigned int t = 0; t
		 <= GetSpaceTimeHandler()->GetMaxTimePoint(); t++)
	  {
	    SetTimeDoFNumber(t);
	    ReSizeSpace(GetSpaceTimeHandler()->GetControlNDoFs(t),
			GetSpaceTimeHandler()->GetControlDoFsPerBlock(t));
	  }
	  SetTimeDoFNumber(0);
	}
	else
	{
	  throw DOpEException("control type " + DOpEtypesToString(_c_type) + "Not implemented","ControlVector<VECTOR>::ReInit");
	}
      }
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
    if(GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
    {
      _accessor = static_cast<int> (time_point);
      assert(_accessor < static_cast<int>(_control.size()));
    }
    else
    {
      //TODO if temporal behavior is required one needs to do something here!
      throw DOpEException("Control type: " + DOpEtypesToString(_c_type) + " is not implemented for behavior: " + DOpEtypesToString(GetBehavior()),
			  "ControlVector<VECTOR>::SetTimeDoFNumber");
    }
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
void ControlVector<VECTOR>::SetTime(double t,  const TimeIterator& interval) const
{
  if(_c_type == DOpEtypes::ControlType::nonstationary)
  {
    if (interval.GetIndex() != _accessor_index || _local_vectors.size()==0 )
    {
      _accessor_index = interval.GetIndex();
      ComputeLocalVectors(interval);
    }
    GetSpaceTimeHandler()->InterpolateControl(_local_control, _local_vectors, t,
					      interval);
    _accessor = -1;
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
    if(_c_type == DOpEtypes::ControlType::nonstationary)
    {
      if ( GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
      {
	if (_accessor >= 0)
	{
	  assert(_control[_accessor] != NULL);
	  return *(_control[_accessor]);
	}
	else
	  return _local_control;
      }
      else
      {
	throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),
			    "ControlVector<VECTOR>::GetSpacialVector");
      }
    }
    else
    {
      throw DOpEException("Control type: " + DOpEtypesToString(_c_type) + " is not implemented.",
			  "ControlVector<VECTOR>::GetSpacialVector");
    }
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
    if(_c_type == DOpEtypes::ControlType::nonstationary)
    {
      if ( GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
      {
	if (_accessor >= 0)
	{
	  assert(_control[_accessor] != NULL);
	  return *(_control[_accessor]);
	}
	else
	  return _local_control;
      }
      else
      {
	throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),
			    "ControlVector<VECTOR>::GetSpacialVector");
      }
    }
    else
    {
      throw DOpEException("Control type: " + DOpEtypesToString(_c_type) + " is not implemented.",
			  "ControlVector<VECTOR>::GetSpacialVector");
    }
  }
}

/******************************************************/
template<typename VECTOR>
const Vector<double>& ControlVector<VECTOR>::GetSpacialVectorCopy() const
{
  if(_lock)
  {
    throw DOpEException("Trying to create a new copy while the old is still in use!","ControlVector::GetSpacialVectorCopy");
  }
  _lock = true;
  if(_c_type == DOpEtypes::ControlType::stationary || _c_type == DOpEtypes::ControlType::initial)
  {
    _copy_control = *(_control[_accessor]);
    return _copy_control;
  }
  else
  {
    if(_c_type == DOpEtypes::ControlType::nonstationary)
    {
      if ( GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
      {
	if (_accessor >= 0)
	{
	  assert(_control[_accessor] != NULL);
	  _copy_control = *(_control[_accessor]);
	}
	else
	  _copy_control = _local_control;
      } 
      else
      {
	throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),
			    "StateVector<VECTOR>::GetSpacialVectorCopy");
      }
      return _copy_control;
    }
    else
    {
      throw DOpEException("Control type: " + DOpEtypesToString(_c_type) + " is not implemented.",
			  "ControlVector<VECTOR>::GetSpacialVectorCopy");
    }
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
      //Check if reinitialization is needed
      bool reinit = false;
      if (_control[_accessor]->size() != ndofs)
      {
	reinit = true;
      }
      else
      {
	if (_control[_accessor]->n_blocks() != nblocks)
	{
	  reinit = true;
	}
	else
	{
	  for (unsigned int i = 0; i < nblocks; i++)
	  {
	    if (_control[_accessor]->block(i).size()
		!= dofs_per_block[i])
	    {
	      reinit = true;
	    }
	  }
	}
      }
      //Check done if reinitialization is needed
      if(reinit)
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
    }//Done accessor \ge 0
    else
    { //_accessor < 0
      unsigned int nblocks = dofs_per_block.size();
      if (_local_control.size() != ndofs)
      {
	_local_control.reinit(nblocks);
	for (unsigned int i = 0; i < nblocks; i++)
	{
	  _local_control.block(i).reinit(dofs_per_block[i]);
	}
	_local_control.collect_sizes();
      }
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
    { //accessor < 0
      if (_local_control.size() != ndofs)
      {
	_local_control.reinit(ndofs);
      }
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
  if (_lock)
  {
    throw DOpEException(
      "Trying to use operator= while a copy is in use!",
      "ControlVector::operator=");
  }
  if(GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
  {
    for(unsigned int i = 0; i < _control.size(); i++)
    {
      assert(_control[i] != NULL);
      _control[i]->operator=(value);
    }
    SetTimeDoFNumber(0);
  }
  else
  {
    throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),"ControlVector<VECTOR>::operator=");
  }
}

/******************************************************/
template<typename VECTOR>
void ControlVector<VECTOR>::operator=(const ControlVector& dq)
{
  if (_lock)
  {
    throw DOpEException(
      "Trying to use operator= while a copy is in use!",
      "ControlVector::operator=");
  }
  if (GetBehavior() != dq.GetBehavior())
  {
    throw DOpEException(
		"Own Behavior does not match dq.Behavior. Own Behavior:"
		+ DOpEtypesToString(GetBehavior()) + " but dq.Behavior is "
		+ DOpEtypesToString(dq.GetBehavior()),
                  "ControlVector<VECTOR>::operator=");
  }
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
    SetTimeDoFNumber(0);
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
  if (_lock)
  {
    throw DOpEException(
      "Trying to use operator+= while a copy is in use!",
      "ControlVector::operator+=");
  }
  if (GetBehavior() != dq.GetBehavior())
  {
    throw DOpEException(
      "Own Behavior does not match dq.Behavior. Own Behavior:"
      + DOpEtypesToString(GetBehavior()) + " but dq.GetBehavior is "
      + DOpEtypesToString(dq.GetBehavior()),
      "ControlVector<VECTOR>::operator+=");
  }
  if(GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
  {
    assert(dq._control.size() == _control.size());
    for(unsigned int i = 0; i < _control.size(); i++)
    {
      assert(_control[i] != NULL);
      assert(dq._control[i] != NULL);
      _control[i]->operator+=(*(dq._control[i]));
    }
    SetTimeDoFNumber(0);
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
  if (_lock)
  {
    throw DOpEException(
      "Trying to use operator= while a copy is in use!",
      "ControlVector::operator*=");
  }
  if(GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
  {
    for(unsigned int i = 0; i < _control.size(); i++)
    {
      assert(_control[i] != NULL);
      _control[i]->operator*=(a);
    }
    SetTimeDoFNumber(0);
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
  if (_lock || dq._lock)
  {
    throw DOpEException(
      "Trying to use operator* while a copy is in use!",
      "ControlVector::operator*");
  }
  if (GetBehavior() != dq.GetBehavior())
  {
    throw DOpEException(
      "Own Behavior does not match dq.Behavior. Own Behavior:"
      + DOpEtypesToString(GetBehavior()) + " but dq.Behavior is "
      + DOpEtypesToString(dq.GetBehavior()),
      "ControlVector<VECTOR>::operator*");
  }
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
  if (_lock)
  {
    throw DOpEException(
      "Trying to use add while a copy is in use!",
      "ControlVector::add");
  }
  if (GetBehavior() != dq.GetBehavior())
  {
    throw DOpEException(
      "Own Behavior does not match dq.Behavior. Own Behavior:"
      + DOpEtypesToString(GetBehavior()) + " but dq.Behavior is "
      + DOpEtypesToString(dq.GetBehavior()),
      "ControlVector<VECTOR>::add");
  }
  if(GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
  {
    assert(dq._control.size() == _control.size());

    for(unsigned int i = 0; i < _control.size(); i++)
    {
      assert(_control[i] != NULL);
      assert(dq._control[i] != NULL);
      _control[i]->add(s,*(dq._control[i]));
    }
    SetTimeDoFNumber(0);
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
  if (_lock)
  {
    throw DOpEException(
      "Trying to use add while a copy is in use!",
      "ControlVector::equ");
  }
  if (GetBehavior() != dq.GetBehavior())
  {
    throw DOpEException(
      "Own Behavior does not match dq.Behavior. Own Behavior:"
      + DOpEtypesToString(GetBehavior()) + " but dq.Behavior is "
      + DOpEtypesToString(dq.GetBehavior()),
      "ControlVector<VECTOR>::equ");
  }
  if(GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
  {
    assert(dq._control.size() == _control.size());

    for(unsigned int i = 0; i < _control.size(); i++)
    {
      assert(_control[i] != NULL);
      assert(dq._control[i] != NULL);
      _control[i]->equ(s,*(dq._control[i]));
    }
    SetTimeDoFNumber(0);
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
  if (_lock)
  {
    throw DOpEException(
      "Trying to use add while a copy is in use!",
      "ControlVector::max");
  }
  if (GetBehavior() != dq.GetBehavior())
  {
    throw DOpEException(
      "Own Behavior does not match dq.Behavior. Own Behavior:"
      + DOpEtypesToString(GetBehavior()) + " but dq.Behavior is "
      + DOpEtypesToString(dq.GetBehavior()),
      "ControlVector<VECTOR>::max");
  }
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
    SetTimeDoFNumber(0);
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
  if (_lock)
  {
    throw DOpEException(
      "Trying to use add while a copy is in use!",
      "ControlVector::min");
  }
  if (GetBehavior() != dq.GetBehavior())
  {
    throw DOpEException(
      "Own Behavior does not match dq.Behavior. Own Behavior:"
      + DOpEtypesToString(GetBehavior()) + " but dq.Behavior is "
      + DOpEtypesToString(dq.GetBehavior()),
      "ControlVector<VECTOR>::min");
  }
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
    SetTimeDoFNumber(0);
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
  if (_lock)
  {
    throw DOpEException(
      "Trying to use add while a copy is in use!",
      "ControlVector::comp_mult");
  }
  if (GetBehavior() != dq.GetBehavior())
  {
    throw DOpEException(
      "Own Behavior does not match dq.Behavior. Own Behavior:"
      + DOpEtypesToString(GetBehavior()) + " but dq.Behavior is "
      + DOpEtypesToString(dq.GetBehavior()),
      "ControlVector<VECTOR>::comp_mult");
  }
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
    SetTimeDoFNumber(0);
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
  if (_lock)
  {
    throw DOpEException(
      "Trying to use add while a copy is in use!",
      "ControlVector::comp_invert");
  }
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
    SetTimeDoFNumber(0);
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
  if (_lock)
  {
    throw DOpEException(
      "Trying to use add while a copy is in use!",
      "ControlVector::init_by_sign");
  }
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
    SetTimeDoFNumber(0);
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
  if (_lock)
  {
    throw DOpEException(
      "Trying to use Norm while a copy is in use!",
      "ControlVector::Norm");
  }
  if(GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
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
  else
  {
    throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),"ControlVector<VECTOR>::Norm");
  }
}

 /******************************************************/
  template<typename VECTOR>
    void
  ControlVector<VECTOR>::ComputeLocalVectors(const TimeIterator& interval) const
  {
    
    unsigned int n_local_dofs =
      GetSpaceTimeHandler()->GetTimeDoFHandler().GetLocalNbrOfDoFs();
    std::vector<unsigned int> global_indices(n_local_dofs);
    //get the global indices
    interval.get_time_dof_indices(global_indices);
    
    //clear out _global_to_local
    _global_to_local.clear();
    
    if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
    {
      _local_vectors.resize(n_local_dofs);
      for (unsigned int i = 0; i < n_local_dofs; ++i)
      {
	_local_vectors[i] = _control[global_indices[i]];
	_global_to_local[global_indices[i]] = i;
      }
    }
    else
    {
      throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),
			  "ControlVector::ComputeLocalVectors");
    }
  }


/******************************************************/
/******************************************************/

template class ControlVector<dealii::Vector<double> >;
template class ControlVector<dealii::BlockVector<double> >;
