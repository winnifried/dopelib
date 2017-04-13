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


#include <include/constraintvector.h>
#include <include/dopeexception.h>

#include <iostream>
#include <assert.h>
#include <iomanip>

using namespace dealii;
using namespace DOpE;
/******************************************************/

template<typename VECTOR>
ConstraintVector<VECTOR>::ConstraintVector(const ConstraintVector &ref)
{
  behavior_ = ref.GetBehavior();
  STH_      = ref.GetSpaceTimeHandler();
  sfh_ticket_ = 0;

  ReInit();
}

/******************************************************/
template<typename VECTOR>
ConstraintVector<VECTOR>::ConstraintVector(const SpaceTimeHandlerBase<VECTOR> *STH, DOpEtypes::VectorStorageType behavior)
{
  behavior_ = behavior;
  STH_      = STH;
  sfh_ticket_ = 0;

  ReInit();
}


/******************************************************/
template<typename VECTOR>
void ConstraintVector<VECTOR>::ReInit()
{
  accessor_ =0;
  if (!GetSpaceTimeHandler()->IsValidControlTicket(sfh_ticket_))
    {
      if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
        {
          local_control_constraint_.resize(1,NULL);
          ReSizeLocalSpace(GetSpaceTimeHandler()->GetConstraintNDoFs("local"),
                           GetSpaceTimeHandler()->GetConstraintDoFsPerBlock("local"));
        }
      else
        {
          throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),"ConstraintVector<VECTOR>::ReInit");
        }

      ReSizeGlobal(GetSpaceTimeHandler()->GetConstraintNDoFs("global"));

    }
}

/******************************************************/

template<typename VECTOR>
ConstraintVector<VECTOR>::~ConstraintVector()
{
  if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
    {
      for (unsigned int i =0; i<local_control_constraint_.size(); i++)
        {
          assert(local_control_constraint_[i] != NULL);
          delete local_control_constraint_[i];
        }
    }
  else
    {
      throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),"ConstraintVector<VECTOR>::~ConstraintVector");
    }
}

/******************************************************/
template<typename VECTOR>
void ConstraintVector<VECTOR>::SetTimeDoFNumber(unsigned int /*time_point*/) const
{
  //TODO if temporal behavior is required one needs to do something here!
  throw DOpEException("Not implemented", "ConstraintVector<VECTOR>::SetTimeDoFNumber");
}

///******************************************************/
//template<typename VECTOR>
//void ConstraintVector<VECTOR>::SetTime(double /*t*/,const TimeIterator& /*interval*/) const
//{
//   //TODO if temporal behavior is required one needs to do something here!
//  throw DOpEException("Not implemented", "ConstraintVector<VECTOR>::SetTime");
//}

/******************************************************/

template<typename VECTOR>
bool ConstraintVector<VECTOR>::HasType(std::string name) const
{
  if (name == "local")
    return true;

  return false;
}

/******************************************************/

template<typename VECTOR>
VECTOR &ConstraintVector<VECTOR>::GetSpacialVector(std::string name)
{
  if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
    {
      if (name == "local")
        {
          return *(local_control_constraint_[accessor_]);
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
const VECTOR &ConstraintVector<VECTOR>::GetSpacialVector(std::string name) const
{
  if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
    {
      if (name == "local")
        {
          return *(local_control_constraint_[accessor_]);
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
const dealii::Vector<double> &ConstraintVector<VECTOR>::GetGlobalConstraints() const
{
  return global_constraint_;
}

/******************************************************/
template<typename VECTOR>
dealii::Vector<double> &ConstraintVector<VECTOR>::GetGlobalConstraints()
{
  return global_constraint_;
}

/******************************************************/

template<typename VECTOR>
void ConstraintVector<VECTOR>::ReSizeGlobal(unsigned int ndofs)
{
  global_constraint_.reinit(ndofs);
}

/******************************************************/
namespace DOpE
{

  template<>
  void DOpE::ConstraintVector<dealii::BlockVector<double> >::ReSizeLocalSpace(
    unsigned int ndofs,
    const std::vector<unsigned int> &dofs_per_block)
  {
    if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
      {
        if (accessor_ >= 0)
          {
            if (local_control_constraint_[accessor_] == NULL)
              local_control_constraint_[accessor_] = new dealii::BlockVector<double>;

            unsigned int nblocks = dofs_per_block.size();
            if (local_control_constraint_[accessor_]->size() != ndofs)
              {
                local_control_constraint_[accessor_]->reinit(nblocks);
                for (unsigned int i = 0; i < nblocks; i++)
                  {
                    local_control_constraint_[accessor_]->block(i).reinit(dofs_per_block[i]);
                  }
                local_control_constraint_[accessor_]->collect_sizes();
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
        if (accessor_ >= 0)
          {
            if (local_control_constraint_[accessor_] == NULL)
              local_control_constraint_[accessor_] = new dealii::Vector<double>;

            if (local_control_constraint_[accessor_]->size() != ndofs)
              {
                local_control_constraint_[accessor_]->reinit(ndofs);
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
  if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
    {
      for (unsigned int i = 0; i < local_control_constraint_.size(); i++)
        {
          assert(local_control_constraint_[i] != NULL);
          local_control_constraint_[i]->operator=(value);
        }
      if (global_constraint_.size() > 0)
        global_constraint_ = value;
    }
  else
    {
      throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),"ConstraintVector<VECTOR>::opterator=");
    }
}

/******************************************************/
template<typename VECTOR>
void ConstraintVector<VECTOR>::operator=(const ConstraintVector &dq)
{
  if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
    {
      if (dq.local_control_constraint_.size() != local_control_constraint_.size())
        {
          if (dq.local_control_constraint_.size() > local_control_constraint_.size())
            {
              unsigned int s = local_control_constraint_.size();
              local_control_constraint_.resize(dq.local_control_constraint_.size(),NULL);
              for (unsigned int i = s; i < local_control_constraint_.size(); i++)
                {
                  assert(local_control_constraint_[i] == NULL);
                  local_control_constraint_[i] = new VECTOR;
                }
            }
          else
            {
              for (unsigned int i = local_control_constraint_.size()-1; i >=dq.local_control_constraint_.size(); i--)
                {
                  assert(local_control_constraint_[i] != NULL);
                  delete local_control_constraint_[i];
                  local_control_constraint_[i] = NULL;
                  local_control_constraint_.pop_back();
                }
              assert(local_control_constraint_.size() == dq.local_control_constraint_.size());
            }
        }

      for (unsigned int i = 0; i < local_control_constraint_.size(); i++)
        {
          assert(local_control_constraint_[i] != NULL);
          assert(dq.local_control_constraint_[i] != NULL);
          local_control_constraint_[i]->operator=(*(dq.local_control_constraint_[i]));
        }

      global_constraint_ = dq.global_constraint_;

      accessor_ = 0;
    }
  else
    {
      throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),"ConstraintVector<VECTOR>::operator=");
    }
}

/******************************************************/
template<typename VECTOR>
void ConstraintVector<VECTOR>::operator+=(const ConstraintVector &dq)
{
  if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
    {
      assert(dq.local_control_constraint_.size() == local_control_constraint_.size());
      for (unsigned int i = 0; i < local_control_constraint_.size(); i++)
        {
          assert(local_control_constraint_[i] != NULL);
          assert(dq.local_control_constraint_[i] != NULL);
          local_control_constraint_[i]->operator+=(*(dq.local_control_constraint_[i]));
        }
      if (global_constraint_.size() > 0)
        global_constraint_ += dq.global_constraint_;
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
  if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
    {
      for (unsigned int i = 0; i < local_control_constraint_.size(); i++)
        {
          assert(local_control_constraint_[i] != NULL);
          local_control_constraint_[i]->operator*=(a);
        }
      if (global_constraint_.size() > 0)
        global_constraint_ *= a;
    }
  else
    {
      throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),"ConstraintVector<VECTOR>::operator*=");
    }
}

/******************************************************/
template<typename VECTOR>
double ConstraintVector<VECTOR>::operator*(const ConstraintVector &dq) const
{
  if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
    {
      assert(dq.local_control_constraint_.size() == local_control_constraint_.size());

      double ret = 0.;
      for (unsigned int i = 0; i < local_control_constraint_.size(); i++)
        {
          assert(local_control_constraint_[i] != NULL);
          assert(dq.local_control_constraint_[i] != NULL);
          ret += local_control_constraint_[i]->operator*(*(dq.local_control_constraint_[i]));
        }
      if (global_constraint_.size() > 0)
        ret += global_constraint_ * dq.global_constraint_;
      return ret;
    }
  else
    {
      throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),"ConstraintVector<VECTOR>::operator*");
    }
}

/******************************************************/
template<typename VECTOR>
void ConstraintVector<VECTOR>::add(double s, const ConstraintVector &dq)
{
  if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
    {
      assert(dq.local_control_constraint_.size() == local_control_constraint_.size());

      for (unsigned int i = 0; i < local_control_constraint_.size(); i++)
        {
          assert(local_control_constraint_[i] != NULL);
          assert(dq.local_control_constraint_[i] != NULL);
          local_control_constraint_[i]->add(s,*(dq.local_control_constraint_[i]));
        }
      if (global_constraint_.size() > 0)
        global_constraint_.add(s,dq.global_constraint_);
    }
  else
    {
      throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),"ConstraintVector<VECTOR>::add");
    }
}

/******************************************************/
template<typename VECTOR>
void ConstraintVector<VECTOR>::equ(double s, const ConstraintVector &dq)
{
  if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
    {
      assert(dq.local_control_constraint_.size() == local_control_constraint_.size());

      for (unsigned int i = 0; i < local_control_constraint_.size(); i++)
        {
          assert(local_control_constraint_[i] != NULL);
          assert(dq.local_control_constraint_[i] != NULL);
          local_control_constraint_[i]->equ(s,*(dq.local_control_constraint_[i]));
        }
      if (global_constraint_.size() > 0)
        global_constraint_.equ(s,dq.global_constraint_);
    }
  else
    {
      throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),"ConstraintVector<VECTOR>::equ");
    }
}

/******************************************************/

template<typename VECTOR>
void ConstraintVector<VECTOR>::PrintInfos(std::stringstream &out)
{
  if (local_control_constraint_.size() ==1)
    {
      out<<"\t"<<local_control_constraint_[0]->size()+global_constraint_.size()<<std::endl;
    }
  else
    {
      if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
        {
          out<<"\tNumber of Timepoints: "<<local_control_constraint_.size()<<std::endl;
          unsigned int min_dofs =0;
          unsigned int max_dofs =0;
          unsigned int total_dofs =0;
          unsigned int this_size=0;
          for (unsigned int i = 0; i < local_control_constraint_.size(); i++)
            {
              this_size = local_control_constraint_[i]->size();
              total_dofs += this_size;
              if (i==0)
                min_dofs = this_size;
              else
                min_dofs = std::min(min_dofs,this_size);
              max_dofs = std::max(max_dofs,this_size);
            }
          out<<"\tTotal   DoFs: "<<total_dofs+global_constraint_.size()<<std::endl;
          out<<"\tMinimal DoFs: "<<min_dofs+global_constraint_.size()<<std::endl;
          out<<"\tMaximal DoFs: "<<max_dofs+global_constraint_.size()<<std::endl;
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
  if (name == "infty")
    {
      if ( restriction == "all")
        {
          for (unsigned int i = 0; i < local_control_constraint_.size(); i++)
            {
              const VECTOR &tmp = *(local_control_constraint_[i]);
              for ( unsigned int j = 0; j < tmp.size(); j++)
                {
                  ret = std::max(ret,std::fabs(tmp(j)));
                }
            }
          for (unsigned int i = 0; i< global_constraint_.size(); i++)
            {
              ret = std::max(std::fabs(global_constraint_(i)),ret);
            }
        }
      else if (restriction == "positive")
        {
          for (unsigned int i = 0; i < local_control_constraint_.size(); i++)
            {
              const VECTOR &tmp = *(local_control_constraint_[i]);
              for ( unsigned int j = 0; j < tmp.size(); j++)
                {
                  ret = std::max(ret,std::max(0.,tmp(j)));
                }
            }
          for (unsigned int i = 0; i< global_constraint_.size(); i++)
            {
              ret = std::max(std::max(0.,global_constraint_(i)),ret);
            }
        }
      else
        {
          throw DOpEException("Unknown restriction: " + restriction,"ConstraintVector<VECTOR>::Norm");
        }
    }
  else if (name == "l1")
    {
      if ( restriction == "all")
        {
          for (unsigned int i = 0; i < local_control_constraint_.size(); i++)
            {
              const VECTOR &tmp = *(local_control_constraint_[i]);
              for ( unsigned int j = 0; j < tmp.size(); j++)
                {
                  ret += std::fabs(tmp(j));
                }
            }
          for (unsigned int i = 0; i< global_constraint_.size(); i++)
            {
              ret += std::fabs(global_constraint_(i));
            }
        }
      else if (restriction == "positive")
        {
          for (unsigned int i = 0; i < local_control_constraint_.size(); i++)
            {
              const VECTOR &tmp = *(local_control_constraint_[i]);
              for ( unsigned int j = 0; j < tmp.size(); j++)
                {
                  ret += std::max(0.,tmp(j));
                }
            }
          for (unsigned int i = 0; i< global_constraint_.size(); i++)
            {
              ret += std::max(0.,global_constraint_(i));
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
  for (unsigned int i = 0; i < local_control_constraint_.size(); i++)
    {
      const VECTOR &tmp = *(local_control_constraint_[i]);
      for ( unsigned int j = 0; j < tmp.size(); j++)
        {
          if (tmp(j) > 0.)
            return false;
        }
    }
  for (unsigned int i = 0; i< global_constraint_.size(); i++)
    {
      if (global_constraint_(i) > 0.)
        return false;
    }
  return true;
}

template<typename VECTOR>
bool ConstraintVector<VECTOR>::IsEpsilonFeasible(double eps) const
{
  for (unsigned int i = 0; i < local_control_constraint_.size(); i++)
    {
      const VECTOR &tmp = *(local_control_constraint_[i]);
      for ( unsigned int j = 0; j < tmp.size(); j++)
        {
          if (tmp(j) > eps)
            return false;
        }
    }
  for (unsigned int i = 0; i< global_constraint_.size(); i++)
    {
      if (global_constraint_(i) > eps)
        return false;
    }
  return true;
}

template<typename VECTOR>
bool ConstraintVector<VECTOR>::IsLargerThan(double eps) const
{
  for (unsigned int i = 0; i < local_control_constraint_.size(); i++)
    {
      const VECTOR &tmp = *(local_control_constraint_[i]);
      for ( unsigned int j = 0; j < tmp.size(); j++)
        {
          if (tmp(j) <= eps)
            return false;
        }
    }
  for (unsigned int i = 0; i< global_constraint_.size(); i++)
    {
      if (global_constraint_(i) <= eps)
        return false;
    }
  return true;
}

template<typename VECTOR>
double ConstraintVector<VECTOR>::Complementarity(const ConstraintVector<VECTOR> &g) const
{
  double ret =0.;
  assert(g.local_control_constraint_.size() == local_control_constraint_.size());

  for (unsigned int i = 0; i < local_control_constraint_.size(); i++)
    {

      const VECTOR &tmp = *(local_control_constraint_[i]);
      const VECTOR &tmp2 = *(g.local_control_constraint_[i]);
      assert(tmp2.size() == tmp.size());
      for ( unsigned int j = 0; j < tmp.size(); j++)
        {
          ret += fabs(tmp2[j]*tmp[j]);
        }
    }
  assert(global_constraint_.size() == g.global_constraint_.size());
  for (unsigned int i = 0; i< global_constraint_.size(); i++)
    {
      ret += fabs(global_constraint_[i]*g.global_constraint_[i]);
    }
  return ret;
}

/******************************************************/
/******************************************************/
/******************************************************/

template class DOpE::ConstraintVector<dealii::Vector<double> >;
template class DOpE::ConstraintVector<dealii::BlockVector<double> >;
