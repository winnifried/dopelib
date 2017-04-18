/**
*
* Copyright (C) 2012-2014 by the DOpElib authors
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


#include <include/controlvector.h>
#include <include/dopeexception.h>

#include <iostream>
#include <assert.h>
#include <iomanip>

using namespace dealii;
using namespace DOpE;
/******************************************************/

template<typename VECTOR>
ControlVector<VECTOR>::ControlVector(const ControlVector &ref)
{
  behavior_ = ref.GetBehavior();
  STH_      = ref.GetSpaceTimeHandler();
  sfh_ticket_ = 0;
  c_type_ = STH_->GetControlType();

  //Check consistency
  if (c_type_ == DOpEtypes::ControlType::initial || c_type_ == DOpEtypes::ControlType::stationary)
    {
      if (behavior_!= DOpEtypes::VectorStorageType::fullmem)
        {
          throw DOpEException("Storage behavior: " + DOpEtypesToString(GetBehavior()) +
                              " is not compatible with control type: " + DOpEtypesToString(c_type_),
                              "ControlVector<VECTOR>::ControlVector<VECTOR>");
        }
    }
  if (behavior_ != DOpEtypes::VectorStorageType::fullmem)
    {
      throw DOpEException("Storage behavior: " + DOpEtypesToString(GetBehavior()) +
                          " not implemented",
                          "ControlVector<VECTOR>::ControlVector<VECTOR>");
    }

  ReInit();
}

/******************************************************/
template<typename VECTOR>
ControlVector<VECTOR>::ControlVector(const SpaceTimeHandlerBase<VECTOR> *STH, DOpEtypes::VectorStorageType behavior)
{
  behavior_ = behavior;
  STH_      = STH;
  sfh_ticket_ = 0;
  c_type_ = STH_->GetControlType();

  //Check consistency
  if (c_type_ == DOpEtypes::ControlType::initial || c_type_ == DOpEtypes::ControlType::stationary)
    {
      if (behavior_!= DOpEtypes::VectorStorageType::fullmem)
        {
          throw DOpEException("Storage behavior: " + DOpEtypesToString(GetBehavior()) +
                              " is not compatible with control type: " + DOpEtypesToString(c_type_),
                              "ControlVector<VECTOR>::ControlVector<VECTOR>");
        }
    }
  if (behavior_ != DOpEtypes::VectorStorageType::fullmem)
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
  accessor_ =0;

  if (!GetSpaceTimeHandler()->IsValidControlTicket(sfh_ticket_))
    {
      if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
        {
          if (c_type_ == DOpEtypes::ControlType::initial || c_type_ == DOpEtypes::ControlType::stationary)
            {
              control_.resize(1,NULL);
              ReSizeSpace(GetSpaceTimeHandler()->GetControlNDoFs(),
                          GetSpaceTimeHandler()->GetControlDoFsPerBlock());
            }
          else
            {
              if ( c_type_ == DOpEtypes::ControlType::nonstationary)
                {
                  //Time dofs for state and control are equal!
                  control_.resize(GetSpaceTimeHandler()->GetMaxTimePoint() + 1, NULL);
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
                  throw DOpEException("control type " + DOpEtypesToString(c_type_) + "Not implemented","ControlVector<VECTOR>::ReInit");
                }
            }
        }
      else
        {
          throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),"ControlVector<VECTOR>::ReInit");
        }

      lock_= false;
    }
}

/******************************************************/

template<typename VECTOR>
ControlVector<VECTOR>::~ControlVector()
{
  if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
    {
      for (unsigned int i =0; i<control_.size(); i++)
        {
          assert(control_[i] != NULL);
          delete control_[i];
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
  if (c_type_ == DOpEtypes::ControlType::nonstationary)
    {
      if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
        {
          accessor_ = static_cast<int> (time_point);
          assert(accessor_ < static_cast<int>(control_.size()));
        }
      else
        {
          //TODO if temporal behavior is required one needs to do something here!
          throw DOpEException("Control type: " + DOpEtypesToString(c_type_) + " is not implemented for behavior: " + DOpEtypesToString(GetBehavior()),
                              "ControlVector<VECTOR>::SetTimeDoFNumber");
        }
    }
  else
    //In all other cases, ignore the argument since no timedependence is present!
    {
      accessor_ = 0;
    }
}

///******************************************************/
//template<typename VECTOR>
//void ControlVector<VECTOR>::SetTime(double t,  const TimeIterator& interval) const
//{
//  if(c_type_ == DOpEtypes::ControlType::nonstationary)
//  {
//    if (interval.GetIndex() != accessor_index_ || local_vectors_.size()==0 )
//    {
//      accessor_index_ = interval.GetIndex();
//      ComputeLocalVectors(interval);
//    }
//    GetSpaceTimeHandler()->InterpolateControl(local_control_, local_vectors_, t,
//                interval);
//    accessor_ = -1;
//  }
//}

/******************************************************/

template<typename VECTOR>
VECTOR &ControlVector<VECTOR>::GetSpacialVector()
{
  if (c_type_ == DOpEtypes::ControlType::stationary || c_type_ == DOpEtypes::ControlType::initial)
    {
      return *(control_[accessor_]);
    }
  else
    {
      if (c_type_ == DOpEtypes::ControlType::nonstationary)
        {
          if ( GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
            {
              if (accessor_ >= 0)
                {
                  assert(control_[accessor_] != NULL);
                  return *(control_[accessor_]);
                }
              else
                return local_control_;
            }
          else
            {
              throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),
                                  "ControlVector<VECTOR>::GetSpacialVector");
            }
        }
      else
        {
          throw DOpEException("Control type: " + DOpEtypesToString(c_type_) + " is not implemented.",
                              "ControlVector<VECTOR>::GetSpacialVector");
        }
    }
}

/******************************************************/
template<typename VECTOR>
const VECTOR &ControlVector<VECTOR>::GetSpacialVector() const
{
  if (c_type_ == DOpEtypes::ControlType::stationary || c_type_ == DOpEtypes::ControlType::initial)
    {
      return *(control_[accessor_]);
    }
  else
    {
      if (c_type_ == DOpEtypes::ControlType::nonstationary)
        {
          if ( GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
            {
              if (accessor_ >= 0)
                {
                  assert(control_[accessor_] != NULL);
                  return *(control_[accessor_]);
                }
              else
                return local_control_;
            }
          else
            {
              throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),
                                  "ControlVector<VECTOR>::GetSpacialVector");
            }
        }
      else
        {
          throw DOpEException("Control type: " + DOpEtypesToString(c_type_) + " is not implemented.",
                              "ControlVector<VECTOR>::GetSpacialVector");
        }
    }
}

/******************************************************/
template<typename VECTOR>
const Vector<double> &ControlVector<VECTOR>::GetSpacialVectorCopy() const
{
  if (lock_)
    {
      throw DOpEException("Trying to create a new copy while the old is still in use!","ControlVector::GetSpacialVectorCopy");
    }
  lock_ = true;
  if (c_type_ == DOpEtypes::ControlType::stationary || c_type_ == DOpEtypes::ControlType::initial)
    {
      copy_control_ = *(control_[accessor_]);
      return copy_control_;
    }
  else
    {
      if (c_type_ == DOpEtypes::ControlType::nonstationary)
        {
          if ( GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
            {
              if (accessor_ >= 0)
                {
                  assert(control_[accessor_] != NULL);
                  copy_control_ = *(control_[accessor_]);
                }
              else
                copy_control_ = local_control_;
            }
          else
            {
              throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),
                                  "StateVector<VECTOR>::GetSpacialVectorCopy");
            }
          return copy_control_;
        }
      else
        {
          throw DOpEException("Control type: " + DOpEtypesToString(c_type_) + " is not implemented.",
                              "ControlVector<VECTOR>::GetSpacialVectorCopy");
        }
    }
}

/******************************************************/

namespace DOpE
{
  template<>
  void ControlVector<dealii::BlockVector<double> >::ReSizeSpace(unsigned int ndofs, const std::vector<unsigned int> &dofs_per_block)
  {
    if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
      {
        if (accessor_ >= 0)
          {
            bool existed = true;
            if (control_[accessor_] == NULL)
              {
                control_[accessor_] = new dealii::BlockVector<double>;
                existed = false;
              }
            unsigned int nblocks = dofs_per_block.size();
            //Check if reinitialization is needed
            bool reinit = false;
            if (control_[accessor_]->size() != ndofs)
              {
                reinit = true;
              }
            else
              {
                if (control_[accessor_]->n_blocks() != nblocks)
                  {
                    reinit = true;
                  }
                else
                  {
                    for (unsigned int i = 0; i < nblocks; i++)
                      {
                        if (control_[accessor_]->block(i).size()
                            != dofs_per_block[i])
                          {
                            reinit = true;
                          }
                      }
                  }
              }
            //Check done if reinitialization is needed
            if (reinit)
              {
                if (existed)
                  {
                    local_control_ = *(control_[accessor_]);
                  }

                control_[accessor_]->reinit(nblocks);
                for (unsigned int i = 0; i < nblocks; i++)
                  {
                    control_[accessor_]->block(i).reinit(dofs_per_block[i]);
                  }
                control_[accessor_]->collect_sizes();

                if (existed)
                  {
                    GetSpaceTimeHandler()->SpatialMeshTransferControl(local_control_,*(control_[accessor_]));
                  }
              }
          }//Done accessor \ge 0
        else
          {
            //accessor_ < 0
            unsigned int nblocks = dofs_per_block.size();
            if (local_control_.size() != ndofs)
              {
                local_control_.reinit(nblocks);
                for (unsigned int i = 0; i < nblocks; i++)
                  {
                    local_control_.block(i).reinit(dofs_per_block[i]);
                  }
                local_control_.collect_sizes();
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
  void ControlVector<dealii::Vector<double> >::ReSizeSpace(unsigned int ndofs, const std::vector<unsigned int> & /*dofs_per_block*/)
  {
    if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
      {
        if (accessor_ >= 0)
          {
            bool existed = true;
            if (control_[accessor_] == NULL)
              {
                control_[accessor_] = new dealii::Vector<double>;
                existed = false;
              }
            if (control_[accessor_]->size() != ndofs)
              {
                if (existed)
                  {
                    local_control_ = *(control_[accessor_]);
                  }

                control_[accessor_]->reinit(ndofs);

                if (existed)
                  {
                    GetSpaceTimeHandler()->SpatialMeshTransferControl(local_control_,*(control_[accessor_]));
                  }
              }
          }
        else
          {
            //accessor < 0
            if (local_control_.size() != ndofs)
              {
                local_control_.reinit(ndofs);
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
  if (lock_)
    {
      throw DOpEException(
        "Trying to use operator= while a copy is in use!",
        "ControlVector::operator=");
    }
  if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
    {
      for (unsigned int i = 0; i < control_.size(); i++)
        {
          assert(control_[i] != NULL);
          control_[i]->operator=(value);
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
void ControlVector<VECTOR>::operator=(const ControlVector &dq)
{
  if (lock_)
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
  if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
    {
      if (dq.control_.size() != control_.size())
        {
          if (dq.control_.size() > control_.size())
            {
              unsigned int s = control_.size();
              control_.resize(dq.control_.size(),NULL);
              for (unsigned int i = s; i < control_.size(); i++)
                {
                  assert(control_[i] == NULL);
                  control_[i] = new VECTOR;
                }
            }
          else
            {
              for (unsigned int i = control_.size()-1; i >=dq.control_.size(); i--)
                {
                  assert(control_[i] != NULL);
                  delete control_[i];
                  control_[i] = NULL;
                  control_.pop_back();
                }
              assert(control_.size() == dq.control_.size());
            }
        }

      for (unsigned int i = 0; i < control_.size(); i++)
        {
          assert(control_[i] != NULL);
          assert(dq.control_[i] != NULL);
          control_[i]->operator=(*(dq.control_[i]));
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
void ControlVector<VECTOR>::operator+=(const ControlVector &dq)
{
  if (lock_)
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
  if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
    {
      assert(dq.control_.size() == control_.size());
      for (unsigned int i = 0; i < control_.size(); i++)
        {
          assert(control_[i] != NULL);
          assert(dq.control_[i] != NULL);
          control_[i]->operator+=(*(dq.control_[i]));
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
  if (lock_)
    {
      throw DOpEException(
        "Trying to use operator= while a copy is in use!",
        "ControlVector::operator*=");
    }
  if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
    {
      for (unsigned int i = 0; i < control_.size(); i++)
        {
          assert(control_[i] != NULL);
          control_[i]->operator*=(a);
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
double ControlVector<VECTOR>::operator*(const ControlVector &dq) const
{
  if (lock_ || dq.lock_)
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
  if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
    {
      assert(dq.control_.size() == control_.size());

      double ret = 0.;
      for (unsigned int i = 0; i < control_.size(); i++)
        {
          assert(control_[i] != NULL);
          assert(dq.control_[i] != NULL);
          ret += control_[i]->operator*(*(dq.control_[i]));
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
void ControlVector<VECTOR>::add(double s, const ControlVector &dq)
{
  if (lock_)
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
  if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
    {
      assert(dq.control_.size() == control_.size());

      for (unsigned int i = 0; i < control_.size(); i++)
        {
          assert(control_[i] != NULL);
          assert(dq.control_[i] != NULL);
          control_[i]->add(s,*(dq.control_[i]));
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
void ControlVector<VECTOR>::equ(double s, const ControlVector &dq)
{
  if (lock_)
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
  if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
    {
      assert(dq.control_.size() == control_.size());

      for (unsigned int i = 0; i < control_.size(); i++)
        {
          assert(control_[i] != NULL);
          assert(dq.control_[i] != NULL);
          control_[i]->equ(s,*(dq.control_[i]));
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
void ControlVector<VECTOR>::max(const ControlVector &dq)
{
  if (lock_)
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
  if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
    {
      assert(dq.control_.size() == control_.size());

      for (unsigned int i = 0; i < control_.size(); i++)
        {
          assert(control_[i] != NULL);
          assert(dq.control_[i] != NULL);
          VECTOR &t = *(control_[i]);
          const VECTOR &tn = *(dq.control_[i]);
          assert(t.size() == tn.size());
          for (unsigned int j = 0; j < t.size() ; j++)
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
void ControlVector<VECTOR>::min(const ControlVector &dq)
{
  if (lock_)
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
  if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
    {
      assert(dq.control_.size() == control_.size());

      for (unsigned int i = 0; i < control_.size(); i++)
        {
          assert(control_[i] != NULL);
          assert(dq.control_[i] != NULL);
          VECTOR &t = *(control_[i]);
          const VECTOR &tn = *(dq.control_[i]);
          assert(t.size() == tn.size());
          for (unsigned int j = 0; j < t.size() ; j++)
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
void ControlVector<VECTOR>::comp_mult(const ControlVector &dq)
{
  if (lock_)
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
  if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
    {
      assert(dq.control_.size() == control_.size());

      for (unsigned int i = 0; i < control_.size(); i++)
        {
          assert(control_[i] != NULL);
          assert(dq.control_[i] != NULL);
          VECTOR &t = *(control_[i]);
          const VECTOR &tn = *(dq.control_[i]);
          assert(t.size() == tn.size());
          for (unsigned int j = 0; j < t.size() ; j++)
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
  if (lock_)
    {
      throw DOpEException(
        "Trying to use add while a copy is in use!",
        "ControlVector::comp_invert");
    }
  if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
    {
      for (unsigned int i = 0; i < control_.size(); i++)
        {
          assert(control_[i] != NULL);
          VECTOR &t = *(control_[i]);
          for (unsigned int j = 0; j < t.size() ; j++)
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
  if (lock_)
    {
      throw DOpEException(
        "Trying to use add while a copy is in use!",
        "ControlVector::init_by_sign");
    }
  if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
    {
      for (unsigned int i = 0; i < control_.size(); i++)
        {
          assert(control_[i] != NULL);
          VECTOR &t = *(control_[i]);
          for (unsigned int j = 0; j < t.size() ; j++)
            {
              if (t(j) < -TOL)
                {
                  t(j) = smaller;
                }
              else if (t(j) > TOL)
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
void ControlVector<VECTOR>::PrintInfos(std::stringstream &out)
{
  if (control_.size() ==1)
    {
      out<<"\t"<<control_[0]->size()<<std::endl;
    }
  else
    {
      if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
        {
          out<<"\tNumber of Timepoints: "<<control_.size()<<std::endl;
          unsigned int min_dofs =0;
          unsigned int max_dofs =0;
          unsigned int total_dofs =0;
          unsigned int this_size=0;
          for (unsigned int i = 0; i < control_.size(); i++)
            {
              this_size = control_[i]->size();
              total_dofs += this_size;
              if (i==0)
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
  if (lock_)
    {
      throw DOpEException(
        "Trying to use Norm while a copy is in use!",
        "ControlVector::Norm");
    }
  if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
    {
      double ret = 0.;
      if (name == "infty")
        {
          if ( restriction == "all")
            {
              for (unsigned int i = 0; i < control_.size(); i++)
                {
                  const VECTOR &tmp = *(control_[i]);
                  for ( unsigned int j = 0; j < tmp.size(); j++)
                    {
                      ret = std::max(ret,std::fabs(tmp(j)));
                    }
                }
            }
          else if (restriction == "positive")
            {
              for (unsigned int i = 0; i < control_.size(); i++)
                {
                  const VECTOR &tmp = *(control_[i]);
                  for ( unsigned int j = 0; j < tmp.size(); j++)
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
      else if (name == "l1")
        {
          if ( restriction == "all")
            {
              for (unsigned int i = 0; i < control_.size(); i++)
                {
                  const VECTOR &tmp = *(control_[i]);
                  for ( unsigned int j = 0; j < tmp.size(); j++)
                    {
                      ret += std::fabs(tmp(j));
                    }
                }
            }
          else if (restriction == "positive")
            {
              for (unsigned int i = 0; i < control_.size(); i++)
                {
                  const VECTOR &tmp = *(control_[i]);
                  for ( unsigned int j = 0; j < tmp.size(); j++)
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
ControlVector<VECTOR>::ComputeLocalVectors(const TimeIterator &interval) const
{

  unsigned int n_local_dofs =
    GetSpaceTimeHandler()->GetTimeDoFHandler().GetLocalNbrOfDoFs();
  std::vector<unsigned int> global_indices(n_local_dofs);
  //get the global indices
  interval.get_time_dof_indices(global_indices);

  //clear out global_to_local_
  global_to_local_.clear();

  if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
    {
      local_vectors_.resize(n_local_dofs);
      for (unsigned int i = 0; i < n_local_dofs; ++i)
        {
          local_vectors_[i] = control_[global_indices[i]];
          global_to_local_[global_indices[i]] = i;
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

template class DOpE::ControlVector<dealii::Vector<double> >;
template class DOpE::ControlVector<dealii::BlockVector<double> >;
