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


#include "statevector.h"
#include "dopeexception.h"
#include "helper.h"

#include <iostream>
#include <assert.h>
#include <iomanip>

using namespace dealii;

namespace DOpE
{
  /******************************************************/
  /**
   * Definition of static member variables
   */
  template<typename VECTOR>
    unsigned int StateVector<VECTOR>::_id_counter = 0;
  template<typename VECTOR>
    unsigned int StateVector<VECTOR>::_num_active = 0;

  /******************************************************/
  template<typename VECTOR>
    StateVector<VECTOR>::StateVector(const StateVector<VECTOR>& ref) :
      _unique_id(_id_counter)
    {
      _behavior = ref.GetBehavior();
      _STH = ref.GetSpaceTimeHandler();
      _sfh_ticket = 0;
      _tmp_dir = ref._tmp_dir;
      _accessor_index = 0;
      if (_behavior == DOpEtypes::VectorStorageType::store_on_disc)
        {
          _local_vectors.resize(1, NULL);
          _local_vectors[0] = new VECTOR;
          _global_to_local.clear();
          _accessor_index = -3;
        }
      _id_counter++;
      _current_dof_number = 0;

      ReInit();
    }

  /******************************************************/
  template<typename VECTOR>
    StateVector<VECTOR>::StateVector(const SpaceTimeHandlerBase<VECTOR>* STH,
        DOpEtypes::VectorStorageType behavior, ParameterReader &param_reader) :
      _unique_id(_id_counter)
    {
      _behavior = behavior;
      _STH = STH;
      _sfh_ticket = 0;
      param_reader.SetSubsection("output parameters");
      _tmp_dir = param_reader.get_string("results_dir") + "tmp_state/";
      if (_behavior == DOpEtypes::VectorStorageType::store_on_disc)
        {
          //make the directory
          std::string command = "mkdir -p " + _tmp_dir;
          if (system(command.c_str()) != 0)
            {
              throw DOpEException("The command " + command + "failed!",
                  "StateVector<VECTOR>::StateVector");
            }
          //check that the directory is not alredy in use by the program
          if (_num_active == 0)
            {
              _filename = _tmp_dir + "StateVector_lock";
              assert(!_filestream.is_open());
              _filestream.open(_filename.c_str(), std::fstream::in);
              if (!_filestream.fail())
                {
                  _filestream.close();
                  throw DOpEException(
                      "The directory " + _tmp_dir
                          + " is probably already in use.",
                      "StateVector<VECTOR>::StateVector");
                }
              else
                {
                  command = "touch " + _tmp_dir + "StateVector_lock";
                  if (system(command.c_str()) != 0)
                    {
                      throw DOpEException("The command " + command + "failed!",
                          "StateVector<VECTOR>::StateVector");
                    }
                }
            }
          _local_vectors.resize(1, NULL);
          _local_vectors[0] = new VECTOR;
          _global_to_local.clear();
          _accessor_index = -3;
        }
      _id_counter++;
      _num_active++;
      _current_dof_number = 0;
      ReInit();
    }

  /******************************************************/
  template<typename VECTOR>
    void
    StateVector<VECTOR>::ReInit()
    {
      if (GetBehavior() == DOpEtypes::VectorStorageType::only_recent)
      {
	//In the only_recent storage case, reinit must be 
	// called to reset the time to zero!
	//Since we don't know if the size of the vectors at 
	//zero corresponds to the length in our given vector
	//we always reinitialize
	//As a specialty, in the ReInit function 
	//in the case only recent, we can't use the 
	//SetTimeDoFNumber function, as it would fail since 
	//we move backward in time!
	_state.resize(2, NULL);
	if( GetSpaceTimeHandler()->GetMaxTimePoint() < 1)
	{
	  throw DOpEException("There are not even two time points. Are you shure this is a non stationary problem?", 
			       "StateVector::ReInit()");
	}
	for (unsigned int t = 0; t
	       <= 1; t++)
	{
	  
	  _accessor = t;
	  ReSizeSpace(GetSpaceTimeHandler()->GetStateNDoFs(t),
		      GetSpaceTimeHandler()->GetStateDoFsPerBlock(t));
	}
	_current_dof_number = 0;
	_accessor = 0;
	_lock = false;
      }//End only_recent
      else if (!GetSpaceTimeHandler()->IsValidStateTicket(_sfh_ticket))
      {
	if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
	{
	  _state.resize(GetSpaceTimeHandler()->GetMaxTimePoint() + 1, NULL);
	  for (unsigned int t = 0; t
		 <= GetSpaceTimeHandler()->GetMaxTimePoint(); t++)
	  {
	    SetTimeDoFNumber(t);
	    ReSizeSpace(GetSpaceTimeHandler()->GetStateNDoFs(t),
			GetSpaceTimeHandler()->GetStateDoFsPerBlock(t));
	  }
	}
	else
	{
	  if (GetBehavior() == DOpEtypes::VectorStorageType::store_on_disc)
	  {
	    _state_information.clear();
	    _state_information.resize(
	      GetSpaceTimeHandler()->GetMaxTimePoint() + 1);
	    
	    //delete all old DOpE-Files in the directory
	    std::string command = "rm -f " + _tmp_dir + "*."
	      + Utilities::int_to_string(_unique_id) + ".dope";
	    if (system(command.c_str()) != 0)
	    {
                      throw DOpEException("The command " + command + "failed!",
					  "StateVector<VECTOR>::ReInit");
	    }
	    for (unsigned int t = 0; t
		   <= GetSpaceTimeHandler()->GetMaxTimePoint(); t++)
	    {
	      //this is important, because in SetTimeDoFNumber, the
                      //vecotr _state_information gets its updates!
	      SetTimeDoFNumber(t);
	      ReSizeSpace(GetSpaceTimeHandler()->GetStateNDoFs(t),
                          GetSpaceTimeHandler()->GetStateDoFsPerBlock(t));
	    }
	    
	  }
	  else
	  {
	    throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),
				"StateVector<VECTOR>::ReInit");
	  }
	}
	SetTimeDoFNumber(0);
	_lock = false;
      }
    }
  /******************************************************/
  template<typename VECTOR>
    StateVector<VECTOR>::~StateVector()
    {
      if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem
	|| GetBehavior() == DOpEtypes::VectorStorageType::only_recent)
        {
          for (unsigned int i = 0; i < _state.size(); i++)
            {
              assert(_state[i] != NULL);
              delete _state[i];
            }
        }
      else
        {
          if (GetBehavior() == DOpEtypes::VectorStorageType::store_on_disc)
            {
              for (unsigned int i = 0; i < _local_vectors.size(); i++)
                {
                  assert(_local_vectors[i] != NULL);
                  delete _local_vectors[i];
                }
              if (1 == _num_active)
                {
                  std::string command = "rm -f " + _tmp_dir + "*."
                      + Utilities::int_to_string(_unique_id) + ".dope; rm -f "
                      + _tmp_dir + "StateVector_lock";
                  if (system(command.c_str()) != 0)
                    {
                      throw DOpEException("The command " + command + "failed!",
                          "StateVector<VECTOR>::~StateVector");
                    }
                }
              else
                {
                  std::string command = "rm -f " + _tmp_dir + "*."
                      + Utilities::int_to_string(_unique_id) + ".dope";
                  if (system(command.c_str()) != 0)
                    {
                      throw DOpEException("The command " + command + "failed!",
                          "StateVector<VECTOR>::~StateVector");
                    }
                }
              _num_active--;
            }
          else
            throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),
                "StateVector<VECTOR>::~StateVector");
        }
    }

  /******************************************************/

  template<typename VECTOR>
    void
    StateVector<VECTOR>::SetTimeDoFNumber(unsigned int dof_number,
        const TimeIterator& interval) const
    {
      if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem
	|| GetBehavior() == DOpEtypes::VectorStorageType::only_recent)
        {
          this->SetTimeDoFNumber(dof_number);
        }
      else
        {
          if (GetBehavior() == DOpEtypes::VectorStorageType::store_on_disc)
            {
              //check, if we have already loaded the interval
              if (_accessor_index == interval.GetIndex())
                {
                  if (_accessor != static_cast<int> (dof_number))
                    {
                      //the we have nothing to do, just store the old one
                      StoreOnDisc();
                      //and set the _accessor to the new number.
                      _accessor = static_cast<int> (dof_number);
                    }
                }
              else
                {
                  //so we have to load everything anew.
                  StoreOnDisc();
                  ComputeLocalVectors(interval);
                  _accessor = dof_number;
                  _accessor_index = interval.GetIndex();

                }
            }
          else
            {
              throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),
                  "StateVector<VECTOR>::SetTimeDoFNumber");
            }
        }
    }

  /******************************************************/
  template<typename VECTOR>
    void
    StateVector<VECTOR>::SetTimeDoFNumber(unsigned int time_point) const
    {
      if (GetBehavior() == DOpEtypes::VectorStorageType::only_recent)
      {
	if ( time_point - _current_dof_number == 0)
	{
	  //nothing to do
	}
	else if ( time_point - _current_dof_number == 1)
	{
	  //We moved to the next time point.
	  _accessor = (_accessor + 1)%2;
	  _current_dof_number = time_point;
	  //Resize spatial vector.
	  ReSizeSpace(GetSpaceTimeHandler()->GetStateNDoFs(time_point),
		      GetSpaceTimeHandler()->GetStateDoFsPerBlock(time_point));
	}
	else 
	{
	  //invalid movement in time
	  throw DOpEException("Invalid movement in time. Using the only_recent behavior you may only move forward in time by exatly one time_dof per update. To reset the time to the initial value call the ReInit method of this vector.","StateVector::SetTimeDoFNumber");
	}  
      }
      else if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
      {
	_accessor = static_cast<int> (time_point);
	assert(_accessor < static_cast<int>(_state.size()));
      }
      else
        {
          if (GetBehavior() == DOpEtypes::VectorStorageType::store_on_disc)
            {
              StoreOnDisc();
              _accessor = time_point;
              _accessor_index = -3;
              _global_to_local.clear();
              _global_to_local[_accessor] = 0;
              ResizeLocalVectors(1);

              DOpEHelper::ReSizeVector(
                  GetSpaceTimeHandler()->GetStateNDoFs(time_point),
                  GetSpaceTimeHandler()->GetStateDoFsPerBlock(time_point),
                  *(_local_vectors[0]));
              if (FileExists(_accessor))
                {
                  FetchFromDisc(_accessor, *_local_vectors[0]);
                }
              _state_information.at(time_point)._size
                  = _local_vectors[_global_to_local[_accessor]]->size();
            }
          else
            {
              throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),
                  "StateVector<VECTOR>::SetTimeDoFNumber");
            }
        }
    }

  /******************************************************/
  template<typename VECTOR>
    void
    StateVector<VECTOR>::SetTime(double t, const TimeIterator& interval) const
    {
      if (interval.GetIndex() != _accessor_index || _local_vectors.size()==0 )
        {
          _accessor_index = interval.GetIndex();
          ComputeLocalVectors(interval);
        }
      GetSpaceTimeHandler()->InterpolateState(_local_state, _local_vectors, t,
          interval);
      _accessor = -1;
    }

  /******************************************************/
  template<typename VECTOR>
    VECTOR&
    StateVector<VECTOR>::GetSpacialVector()
    {
      if ( GetBehavior() == DOpEtypes::VectorStorageType::fullmem
	|| GetBehavior() == DOpEtypes::VectorStorageType::only_recent)
        {
          if (_accessor >= 0)
            {
              assert(_state[_accessor] != NULL);
              return *(_state[_accessor]);
            }
          else
            return _local_state;
        }
      else
        {
          if (GetBehavior() == DOpEtypes::VectorStorageType::store_on_disc)
            {
              if (_accessor >= 0)
                {
                  assert(_global_to_local[_accessor] <_local_vectors.size());
                  return *(_local_vectors[_global_to_local[_accessor]]);
                }
              else
                return _local_state;
            }
          throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),
              "StateVector<VECTOR>::GetSpacialVector");
        }
    }

  /******************************************************/
  template<typename VECTOR>
    const VECTOR&
    StateVector<VECTOR>::GetSpacialVector() const
    {
      if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem
	|| GetBehavior() == DOpEtypes::VectorStorageType::only_recent)
        {
          if (_accessor >= 0)
            {
              assert(_state[_accessor] != NULL);
              return *(_state[_accessor]);
            }
          else
            return _local_state;
        }
      else
        {
          if (GetBehavior() == DOpEtypes::VectorStorageType::store_on_disc)
            {
              if (_accessor >= 0)
                {
                  return *(_local_vectors[_global_to_local[_accessor]]);
                }
              else
                return _local_state;
            }
          throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),
              "StateVector<VECTOR>::GetSpacialVector const");
        }
    }

  /******************************************************/
  template<typename VECTOR>
    const Vector<double>&
    StateVector<VECTOR>::GetSpacialVectorCopy() const
    {
      if (_lock)
        {
          throw DOpEException(
              "Trying to create a new copy while the old is still in use!",
              "StateVector::GetSpacialVectorCopy");
        }
      _lock = true;

      if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem
	|| GetBehavior() == DOpEtypes::VectorStorageType::only_recent)
        {
          if (_accessor >= 0)
            {
              assert(_state[_accessor] != NULL);
              _copy_state = *(_state[_accessor]);
            }
          else
            _copy_state = _local_state;
        }
      else
        {
          if (GetBehavior() == DOpEtypes::VectorStorageType::store_on_disc)
            {
              if (_accessor >= 0)
                {
                  _copy_state = *(_local_vectors[_global_to_local[_accessor]]);
                }
              else
                _copy_state = _local_state;
            }
          else
            {
              throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),
                  "StateVector<VECTOR>::GetSpacialVectorCopy");
            }
        }
      return _copy_state;
    }

  /******************************************************/
  template<>
    void
    DOpE::StateVector<dealii::BlockVector<double> >::ReSizeSpace(
        unsigned int ndofs, const std::vector<unsigned int>& dofs_per_block) const
    {
      if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem
	|| GetBehavior() == DOpEtypes::VectorStorageType::only_recent)
        {
          if (_accessor >= 0)
            {
              bool existed = true;
	      if (_state[_accessor] == NULL)
                {
                  _state[_accessor] = new dealii::BlockVector<double>;
		  existed = false; 
                }
              unsigned int nblocks = dofs_per_block.size();
              bool reinit = false;
              if (_state[_accessor]->size() != ndofs)
                {
                  reinit = true;
                }
              else
                {
                  if (_state[_accessor]->n_blocks() != nblocks)
                    {
                      reinit = true;
                    }
                  else
                    {
                      for (unsigned int i = 0; i < nblocks; i++)
                        {
                          if (_state[_accessor]->block(i).size()
                              != dofs_per_block[i])
                            {
                              reinit = true;
                            }
                        }
                    }
                }
              if (reinit)
                {
		  if(existed)
		  {
		    _local_state = *(_state[_accessor]);
		  }
                  _state[_accessor]->reinit(nblocks);
                  for (unsigned int i = 0; i < nblocks; i++)
                    {
                      _state[_accessor]->block(i).reinit(dofs_per_block[i]);
                    }
                  _state[_accessor]->collect_sizes();
		  if(existed)
		  {
		    GetSpaceTimeHandler()->SpatialMeshTransferState(_local_state,*(_state[_accessor]));
		  }
                }
            }
          else
	  { //_accessor < 0
              unsigned int nblocks = dofs_per_block.size();
              if (_local_state.size() != ndofs)
                {
                  _local_state.reinit(nblocks);
                  for (unsigned int i = 0; i < nblocks; i++)
                    {
                      _local_state.block(i).reinit(dofs_per_block[i]);
                    }
                  _local_state.collect_sizes();
                }
            }
        }
      else
        {
          if (GetBehavior() == DOpEtypes::VectorStorageType::store_on_disc)
            {
              if (_accessor >= 0)
                {
                  unsigned int nblocks = dofs_per_block.size();
                  bool reinit = false;
                  if (_local_vectors[_global_to_local[_accessor]]->size()
                      != ndofs)
                    {
                      reinit = true;
                    }
                  else
                    {
                      if (_local_vectors[_global_to_local[_accessor]]->n_blocks()
                          != nblocks)
                        {
                          reinit = true;
                        }
                      else
                        {
                          for (unsigned int i = 0; i < nblocks; i++)
                            {
                              if (_local_vectors[_global_to_local[_accessor]]->block(
                                  i).size() != dofs_per_block[i])
                                {
                                  reinit = true;
                                }
                            }
                        }
                    }
                  if (reinit)
                    {                     
		      _local_vectors[_global_to_local[_accessor]]->reinit(
                          nblocks);
                      for (unsigned int i = 0; i < nblocks; i++)
                        {
                          _local_vectors[_global_to_local[_accessor]]->block(i).reinit(
                              dofs_per_block[i]);
                        }
                      _local_vectors[_global_to_local[_accessor]]->collect_sizes();
                    }
                }
              else
                {
                  unsigned int nblocks = dofs_per_block.size();
                  if (_local_state.size() != ndofs)
                    {
                      _local_state.reinit(nblocks);
                      for (unsigned int i = 0; i < nblocks; i++)
                        {
                          _local_state.block(i).reinit(dofs_per_block[i]);
                        }
                      _local_state.collect_sizes();
                    }
                }
            }
          else
            throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),
                "StateVector<dealii::BlockVector<double> >::ReSizeSpace");
        }

    }
  /******************************************************/

  template<>
    void
    DOpE::StateVector<dealii::Vector<double> >::ReSizeSpace(unsigned int ndofs,
        const std::vector<unsigned int>& /*dofs_per_block*/) const
    {
      if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem
	|| GetBehavior() == DOpEtypes::VectorStorageType::only_recent)
        {
          if (_accessor >= 0)
            {
              bool existed = true;
	      if (_state[_accessor] == NULL)
                {
                  _state[_accessor] = new dealii::Vector<double>;
		  existed = false; 
		}

              bool reinit = false;
              if (_state[_accessor]->size() != ndofs)
                {
                  reinit = true;
                }
              if (reinit)
                {
                  if(existed)
		  {
		    _local_state = *(_state[_accessor]);
		  }
		  
		  _state[_accessor]->reinit(ndofs);

		  if(existed)
		  {
		    GetSpaceTimeHandler()->SpatialMeshTransferState(_local_state,*(_state[_accessor]));
		  }
                }
            }
          else
	  {//accessor < 0
	    if (_local_state.size() != ndofs)
	    {
	      _local_state.reinit(ndofs);
	    }
	  }
        }
      else
        {
          if (GetBehavior() == DOpEtypes::VectorStorageType::store_on_disc)
            {
              if (_accessor >= 0)
                {
                  bool reinit = false;
                  if (_local_vectors[_global_to_local[_accessor]]->size()
                      != ndofs)
                    {
                      reinit = true;
                    }
                  if (reinit)
                    {
                      _local_vectors[_global_to_local[_accessor]]->reinit(ndofs);
                    }
                }
              else
                {
                  if (_local_state.size() != ndofs)
                    {
                      _local_state.reinit(ndofs);
                    }
                }
            }
          else
            throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),
                "StateVector<dealii::Vector<double> >::ReSizeSpace");
        }

    }

  /******************************************************/
  template<typename VECTOR>
    void
    StateVector<VECTOR>::operator=(double value)
    {
      if (_lock)
        {
          throw DOpEException(
              "Trying to use operator= while a copy is in use!",
              "StateVector::operator=");
        }
      else
        {
          if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
            {
              for (unsigned int i = 0; i < _state.size(); i++)
                {
                  assert(_state[i] != NULL);
                  _state[i]->operator=(value);
                }
              SetTimeDoFNumber(0);
            }
          else if (GetBehavior() == DOpEtypes::VectorStorageType::only_recent)
            {
	      //No sense in doing so! Only usefull to initialize with zero
	      if(value != 0.)
	      {
		throw DOpEException("Using this function with any value other than zero is not supported in the only_recent behavior",
				    "StateVector::operator=");
	      }
              for (unsigned int i = 0; i < _state.size(); i++)
                {
                  assert(_state[i] != NULL);
                  _state[i]->operator=(value);
                }
	      //We don't reset the time here!
            }
	  else
            {
              if (GetBehavior() == DOpEtypes::VectorStorageType::store_on_disc)
                {
                  for (unsigned int t = 0; t
                      <= GetSpaceTimeHandler()->GetMaxTimePoint(); t++)
                    {
                      SetTimeDoFNumber(t);
                      assert(_global_to_local[_accessor]==0);
                      _local_vectors[0]->operator=(value);
                    }
                  SetTimeDoFNumber(0);
                }
              else
                {
                  throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),
                      "StateVector<VECTOR>::operator=");
                }
            }
        }
    }

  /******************************************************/
  template<typename VECTOR>
    void
    StateVector<VECTOR>::operator=(const StateVector& dq)
    {
      if (_lock)
        {
          throw DOpEException(
              "Trying to use operator= while a copy is in use!",
              "StateVector::operator=");
        }
      else
        {
          if (GetBehavior() == dq.GetBehavior())
            {
              if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
                {
                  if (dq._state.size() != _state.size())
                    {
                      if (dq._state.size() > _state.size())
                        {
                          unsigned int s = _state.size();
                          _state.resize(dq._state.size(), NULL);
                          for (unsigned int i = s; i < _state.size(); i++)
                            {
                              assert(_state[i] == NULL);
                              _state[i] = new VECTOR;
                            }
                        }
                      else
                        {
                          for (unsigned int i = _state.size() - 1; i
                              >= dq._state.size(); i--)
                            {
                              assert(_state[i] != NULL);
                              delete _state[i];
                              _state[i] = NULL;
                              _state.pop_back();
                            }
                          assert(_state.size() == dq._state.size());
                        }
                    }

                  for (unsigned int i = 0; i < _state.size(); i++)
                    {
                      assert(_state[i] != NULL);
                      assert(dq._state[i] != NULL);
                      _state[i]->operator=(*(dq._state[i]));
                    }
                  SetTimeDoFNumber(0);
                }//endif fullmem
	      else if (GetBehavior() == DOpEtypes::VectorStorageType::only_recent)
	      {
		//No sense in doing so! 
		throw DOpEException("Using this function is not supported in the only_recent behavior",
				    "StateVector::operator=");
	      }
	      else
                {
                  if (GetBehavior() == DOpEtypes::VectorStorageType::store_on_disc)
                    {
                      //Delete all vectors on the disc.
                      std::string command = "mkdir -p " + _tmp_dir + "; rm -f "
                          + _tmp_dir + "*." + Utilities::int_to_string(
                          _unique_id) + ".dope";
                      if (system(command.c_str()) != 0)
                        {
                          throw DOpEException(
                              "The command " + command + "failed!",
                              "StateVector<VECTOR>::operator=");
                        }
                      //make sure that all Vectors of dq are stored on the disc
                      dq.StoreOnDisc();
                      for (unsigned int t = 0; t
                          <= dq.GetSpaceTimeHandler()->GetMaxTimePoint(); t++)
                        {
                          //Now just copy all dq.Vectors on the disc.
                          dq.MakeName(t);
                          MakeName(t);
                          assert(dq.FileExists(t));
                          command = "cp " + dq._filename + " " + _filename;
                          if (system(command.c_str()) != 0)
                            {
                              throw DOpEException(
                                  "The command " + command + "failed!",
                                  "StateVector<VECTOR>::operator=");
                            }
                          _state_information.at(t)._on_disc = true;
                        }
                      //Make sure that no old spatial vectores are stored in _local_vectors.
                      ResizeLocalVectors(1);
                      _accessor = -1; //We set this so that SetTimeDoFNumber(0) does not store something!
                      SetTimeDoFNumber(0);
                    }
                  else
                    {
                      throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),
                          "StateVector<VECTOR>::operator=");
                    }
                }
            }
          else
            {
              throw DOpEException(
		"Own Behavior does not match dq.Behavior. Own Behavior:"
		+ DOpEtypesToString(GetBehavior()) + " but dq.Behavior is "
		+ DOpEtypesToString(dq.GetBehavior()),
                  "StateVector<VECTOR>::operator=");
            }
        }
    }

  /******************************************************/
  template<typename VECTOR>
    void
    StateVector<VECTOR>::operator+=(const StateVector& dq)
    {
      if (_lock)
        {
          throw DOpEException(
              "Trying to use operator+= while a copy is in use!",
              "StateVector::operator+=");
        }
      else
        {
          if (GetBehavior() == dq.GetBehavior())
            {
              if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
                {
                  assert(dq._state.size() == _state.size());
                  for (unsigned int i = 0; i < _state.size(); i++)
                    {
                      assert(_state[i] != NULL);
                      assert(dq._state[i] != NULL);
                      _state[i]->operator+=(*(dq._state[i]));
                    }
                  SetTimeDoFNumber(0);
                }//endif fullmem
	      else if (GetBehavior() == DOpEtypes::VectorStorageType::only_recent)
	      {
		//No sense in doing so! 
		throw DOpEException("Using this function is not supported in the only_recent behavior",
				    "StateVector::operator+=");
	      }
              else
                {
                  if (GetBehavior() == DOpEtypes::VectorStorageType::store_on_disc)
                    {
                      assert(dq.GetSpaceTimeHandler()->GetMaxTimePoint() == GetSpaceTimeHandler()->GetMaxTimePoint() );
                      dq.StoreOnDisc();
                      for (unsigned int t = 0; t
                          <= dq.GetSpaceTimeHandler()->GetMaxTimePoint(); t++)
                        {
                          SetTimeDoFNumber(t);//this makes sure that everything is stored an _local_vectors has length 1.
                          DOpEHelper::ReSizeVector(
                              GetSpaceTimeHandler()->GetStateNDoFs(t),
                              GetSpaceTimeHandler()->GetStateDoFsPerBlock(t),
                              _local_state);
                          dq.FetchFromDisc(t, _local_state);
                          assert(_global_to_local[_accessor]==0);
                          _local_vectors[0]->operator+=(_local_state);
                        }
                      SetTimeDoFNumber(0);
                    }
                  else
                    throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),
                        "StateVector<VECTOR>::operator+=");
                }
            }
          else
            {
              throw DOpEException(
                  "Own Behavior does not match dq.Behavior. Own Behavior:"
		  + DOpEtypesToString(GetBehavior()) + " but dq.GetBehavior is "
		  + DOpEtypesToString(dq.GetBehavior()),
                  "StateVector<VECTOR>::operator+=");
            }
        }
    }

  /******************************************************/
  template<typename VECTOR>
    void
    StateVector<VECTOR>::operator*=(double value)
    {
      if (_lock)
        {
          throw DOpEException(
              "Trying to use operator*= while a copy is in use!",
              "StateVector::operator*=");
        }
      else
        {
          if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
            {
              for (unsigned int i = 0; i < _state.size(); i++)
                {
                  assert(_state[i] != NULL);
                  _state[i]->operator*=(value);
                }
              SetTimeDoFNumber(0);
            }//endif fullmem
	      else if (GetBehavior() == DOpEtypes::VectorStorageType::only_recent)
	      {
		//No sense in doing so! 
		throw DOpEException("Using this function is not supported in the only_recent behavior",
				    "StateVector::operator*=");
	      }
          else
            {
              if (GetBehavior() == DOpEtypes::VectorStorageType::store_on_disc)
                {
                  for (unsigned int t = 0; t
                      <= GetSpaceTimeHandler()->GetMaxTimePoint(); t++)
                    {
                      SetTimeDoFNumber(t);//this makes sure that everything is stored an _local_vectors has length 1.
                      assert(_global_to_local[_accessor]==0);
                      _local_vectors[0]->operator*=(value);
                    }
                  SetTimeDoFNumber(0);
                }
              else
                {
                  throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),
                      "StateVector<VECTOR>::operator*=");
                }
            }
        }
    }

  /******************************************************/
  template<typename VECTOR>
    double
    StateVector<VECTOR>::operator*(const StateVector& dq) const
    {
      if (GetBehavior() == dq.GetBehavior())
        {
          if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
            {
              assert(dq._state.size() == _state.size());

              double ret = 0.;
              for (unsigned int i = 0; i < _state.size(); i++)
                {
                  assert(_state[i] != NULL);
                  assert(dq._state[i] != NULL);
                  ret += _state[i]->operator*(*(dq._state[i]));
                }
              return ret;
            }//endif fullmem
	  if (GetBehavior() == DOpEtypes::VectorStorageType::only_recent)
	  {
		//No sense in doing so! 
	    throw DOpEException("Using this function is not supported in the only_recent behavior",
				"StateVector::operator*");
	  }
          if (GetBehavior() == DOpEtypes::VectorStorageType::store_on_disc)
            {
              if (_lock || dq._lock)
                {
                  throw DOpEException(
                      "Trying to use operator* while a copy is in use!",
                      "StateVector::operator*");
                }
              else
                {
                  assert(dq.GetSpaceTimeHandler()->GetMaxTimePoint() == GetSpaceTimeHandler()->GetMaxTimePoint() );

                  double ret = 0;

                  StoreOnDisc();
                  dq.StoreOnDisc();
                  for (unsigned int t = 0; t
                      <= GetSpaceTimeHandler()->GetMaxTimePoint(); t++)
                    {
                      DOpEHelper::ReSizeVector(
                          GetSpaceTimeHandler()->GetStateNDoFs(t),
                          GetSpaceTimeHandler()->GetStateDoFsPerBlock(t),
                          _local_state);
                      FetchFromDisc(t, _local_state);
                      DOpEHelper::ReSizeVector(
                          dq.GetSpaceTimeHandler()->GetStateNDoFs(t),
                          dq.GetSpaceTimeHandler()->GetStateDoFsPerBlock(t),
                          dq._local_state);
                      dq.FetchFromDisc(t, dq._local_state);
                      ret += _local_state.operator*(dq._local_state);
                    }
                  return ret;
                }
            }
          else
            {
              throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),
                  "StateVector<VECTOR>::operator*");
            }
        }
      else
        {
          throw DOpEException(
              "Own Behavior does not match dq.Behavior. Own Behavior:"
	      + DOpEtypesToString(GetBehavior()) + " but dq.Behavior is "
		+ DOpEtypesToString(dq.GetBehavior()),
              "StateVector<VECTOR>::operator*");
        }
    }

  /******************************************************/
  template<typename VECTOR>
    void
    StateVector<VECTOR>::add(double s, const StateVector& dq)
    {
      if (_lock)
        {
          throw DOpEException(
              "Trying to use add while a copy is in use!",
              "StateVector::add");
        }
      else
        {
          if (GetBehavior() == dq.GetBehavior())
            {
              if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
                {
                  assert(dq._state.size() == _state.size());

                  for (unsigned int i = 0; i < _state.size(); i++)
                    {
                      assert(_state[i] != NULL);
                      assert(dq._state[i] != NULL);
                      _state[i]->add(s, *(dq._state[i]));
                    }
                  SetTimeDoFNumber(0);
                }//endif fullmem
	      else if (GetBehavior() == DOpEtypes::VectorStorageType::only_recent)
	      {
		//No sense in doing so! 
		throw DOpEException("Using this function is not supported in the only_recent behavior",
				    "StateVector::add");
	      }
              else
                {
                  if (GetBehavior() == DOpEtypes::VectorStorageType::store_on_disc)
                    {
                      assert(dq.GetSpaceTimeHandler()->GetMaxTimePoint() == GetSpaceTimeHandler()->GetMaxTimePoint() );

                      dq.StoreOnDisc();
                      for (unsigned int t = 0; t
                          <= dq.GetSpaceTimeHandler()->GetMaxTimePoint(); t++)
                        {
                          SetTimeDoFNumber(t);//this makes sure that everything is stored an _local_vectors has length 1.
                          DOpEHelper::ReSizeVector(
                              GetSpaceTimeHandler()->GetStateNDoFs(t),
                              GetSpaceTimeHandler()->GetStateDoFsPerBlock(t),
                              _local_state);
                          dq.FetchFromDisc(t, _local_state);
                          assert(_global_to_local[_accessor]==0);
                          _local_vectors[0]->add(s, _local_state);
                        }
                      SetTimeDoFNumber(0);
                    }
                  else
                    throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),
                        "StateVector<VECTOR>::add");
                }
            }
          else
            {
              throw DOpEException(
                  "Own Behavior does not match dq.Behavior. Own Behavior:"
		  + DOpEtypesToString(GetBehavior()) + " but dq.Behavior is "
		+ DOpEtypesToString(dq.GetBehavior()),
                  "StateVector<VECTOR>::equ");
            }
        }
    }

  /******************************************************/
  template<typename VECTOR>
    void
    StateVector<VECTOR>::equ(double s, const StateVector& dq)
    {
      if (_lock)
        {
          throw DOpEException(
              "Trying to use equ while a copy is in use!",
              "StateVector::equ");
        }
      else
        {
          if (GetBehavior() == dq.GetBehavior())
            {
              if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
                {
                  assert(dq._state.size() == _state.size());
                  for (unsigned int i = 0; i < _state.size(); i++)
                    {
                      assert(_state[i] != NULL);
                      assert(dq._state[i] != NULL);
                      _state[i]->equ(s, *(dq._state[i]));
                    }
                  SetTimeDoFNumber(0);
                }//endif fullmem
	      else if (GetBehavior() == DOpEtypes::VectorStorageType::only_recent)
	      {
		//No sense in doing so! 
		throw DOpEException("Using this function is not supported in the only_recent behavior",
				    "StateVector::equ");
	      }
              else
                {
                  if (GetBehavior() == DOpEtypes::VectorStorageType::store_on_disc)
                    {
                      dq.StoreOnDisc();
                      for (unsigned int t = 0; t
                          <= dq.GetSpaceTimeHandler()->GetMaxTimePoint(); t++)
                        {
                          SetTimeDoFNumber(t);//this makes sure that everything is stored an _local_vectors has length 1.
                          DOpEHelper::ReSizeVector(
                              GetSpaceTimeHandler()->GetStateNDoFs(t),
                              GetSpaceTimeHandler()->GetStateDoFsPerBlock(t),
                              _local_state);
                          dq.FetchFromDisc(t, _local_state);
                          assert(_global_to_local[_accessor]==0);
                          _local_vectors[0]->equ(s, _local_state);

                        }
                      SetTimeDoFNumber(0);
                    }
                  else
                    throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),
                        "StateVector<VECTOR>::equ");
                }
            }
          else
            {
              throw DOpEException(
                  "Own Behavior does not match dq.Behavior. Own Behavior:"
		  + DOpEtypesToString(GetBehavior()) + " but dq.Behavior is "
		+ DOpEtypesToString(dq.GetBehavior()),
                  "StateVector<VECTOR>::equ");
            }
        }
    }

  /******************************************************/
  template<typename VECTOR>
    void
    StateVector<VECTOR>::PrintInfos(std::stringstream& out)
    {
      if (GetSpaceTimeHandler()->GetMaxTimePoint() == 0)
        {
          if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
            {
              assert(_state.size()==1);
              out << "\t" << _state[0]->size() << std::endl;
            }
          else
            {
              if (GetBehavior() == DOpEtypes::VectorStorageType::store_on_disc)
                {
                  SetTimeDoFNumber(0);
                  assert(_global_to_local[_accessor]==0);
                  out << "\t" << _local_vectors[0]->size() << std::endl;
                }
              else
                throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),
                    "StateyVector::PrintInfos");
            }
        }
      else
        {
          if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
            {
              out << "\tNumber of Timepoints: " << _state.size() << std::endl;
              unsigned int min_dofs = 0;
              unsigned int max_dofs = 0;
              unsigned int total_dofs = 0;
              unsigned int this_size = 0;
              for (unsigned int i = 0; i < _state.size(); i++)
                {
                  this_size = _state[i]->size();
                  total_dofs += this_size;
                  if (i == 0)
                    min_dofs = this_size;
                  else
                    min_dofs = std::min(min_dofs, this_size);
                  max_dofs = std::max(max_dofs, this_size);
                }
              out << "\tTotal   DoFs: " << total_dofs << std::endl;
              out << "\tMinimal DoFs: " << min_dofs << std::endl;
              out << "\tMaximal DoFs: " << max_dofs << std::endl;
            }
          else if (GetBehavior() == DOpEtypes::VectorStorageType::only_recent)
            {
              out << "\tNumber of Timepoints: " << 
		GetSpaceTimeHandler()->GetMaxTimePoint()+1<< std::endl;
              unsigned int this_size = 0;
	      this_size = _state[_accessor]->size();
              out << "\tSpatial DoFs: " << this_size << std::endl;
            }
          else
            {
              if (GetBehavior() == DOpEtypes::VectorStorageType::store_on_disc)
                {
                  out << "\tNumber of Timepoints: "
                      << _state_information.size() << std::endl;
                  unsigned int min_dofs = 0;
                  unsigned int max_dofs = 0;
                  unsigned int total_dofs = 0;
                  unsigned int this_size = 0;
                  for (unsigned int i = 0; i < _state_information.size(); i++)
                    {
                      assert(_state_information.at(i)._size!=-1);
                      this_size = _state_information.at(i)._size;
                      total_dofs += this_size;
                      if (i == 0)
                        min_dofs = this_size;
                      else
                        min_dofs = std::min(min_dofs, this_size);
                      max_dofs = std::max(max_dofs, this_size);
                    }
                  out << "\tTotal   DoFs: " << total_dofs << std::endl;
                  out << "\tMinimal DoFs: " << min_dofs << std::endl;
                  out << "\tMaximal DoFs: " << max_dofs << std::endl;
                }
              else
                throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),
                    "StateyVector::PrintInfos");
            }
        }
    }

  /******************************************************/
  template<typename VECTOR>
    bool
    StateVector<VECTOR>::FileExists(unsigned int time_point) const
    {
      return _state_information.at(time_point)._on_disc;
    }

  /******************************************************/
  template<typename VECTOR>
    void
    StateVector<VECTOR>::MakeName(unsigned int time_point) const
    {
      assert(time_point<100000);
      _filename = _tmp_dir + "statevector." + Utilities::int_to_string(
          time_point, 5) + "." + Utilities::int_to_string(_unique_id) + ".dope";
    }

  /******************************************************/
  template<typename VECTOR>
    void
    StateVector<VECTOR>::StoreOnDisc() const
    {
      //make sure that accessor has the chance to indicate sth valid
      if (_accessor >= 0)
        {
          //now if there is something to store, do it
          if (_local_vectors[_global_to_local[_accessor]]->size() != 0)
            {
              MakeName(_accessor);
              assert(!_filestream.is_open());
              _filestream.open(_filename.c_str(), std::fstream::out);
              if (!_filestream.fail())
                {
                  _local_vectors[_global_to_local[_accessor]]->block_write(
                      _filestream);
                  _filestream.close();
                  _state_information.at(_accessor)._on_disc = true;
                }
              else
                {
                  throw DOpEException(
                      "Could not store " + _filename + "on disc.",
                      "StateVector<VECTOR>::StoreOnDisc");
                }
            }
        }
    }

  /******************************************************/
  template<typename VECTOR>
    void
    StateVector<VECTOR>::FetchFromDisc(unsigned int time_point, VECTOR& vector) const
    {
      MakeName(time_point);
      assert(!_filestream.is_open());
      _filestream.open(_filename.c_str(), std::fstream::in);
      if (!_filestream.fail())
        {
          vector.block_read(_filestream);
          _filestream.close();
        }
      else
        {
          throw DOpEException("Could not fetch " + _filename + "from disc.",
              "StateVector<VECTOR>::StoreOnDisc");
        }
    }

  /******************************************************/
  template<typename VECTOR>
    void
    StateVector<VECTOR>::SwapPtr(VECTOR* & a, VECTOR* & b) const
    {
      VECTOR* tmp = a;
      a = b;
      b = tmp;
    }

  /******************************************************/
  template<typename VECTOR>
    void
    StateVector<VECTOR>::ComputeLocalVectors(const TimeIterator& interval) const
    {
      //FIXME: It appears, the _global_to_local stuff is 
      // needed only for store_on_disc? Then why do we 
      //take care of the other cases?
      
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
              _local_vectors[i] = _state[global_indices[i]];
              _global_to_local[global_indices[i]] = i;
            }
        }
      else if (GetBehavior() == DOpEtypes::VectorStorageType::only_recent)
        {
	  if(n_local_dofs != 2)
	  {
	    throw DOpEException("Behavior only_recent can only work with 2 local DoFs per time interval.", "StateVector::ComputeLocalVectors");
	  }
	  if(global_indices[1] != _current_dof_number)
	  {//We do not have the needed information!
	    throw DOpEException("When using the behavior only_recent you may only use the currently selected interval.", "StateVector::ComputeLocalVectors");	    
	  }
          _local_vectors.resize(n_local_dofs);
          for (unsigned int i = 0; i < n_local_dofs; ++i)
            {
              _local_vectors[i] = _state[(_accessor + ((i+1))%2)%2];
	      //FixMe: Is this the right coordinate, or should this be 
	      //something else? 
              _global_to_local[global_indices[i]] = i;
            }
        }
      else
        {
          if (GetBehavior() == DOpEtypes::VectorStorageType::store_on_disc)
            {
              //Resize _local_vectors to the right size.
              //We have to take more care in this case because of the
              //dynamic memory management
              ResizeLocalVectors(n_local_dofs);
              for (unsigned int i = 0; i < n_local_dofs; ++i)
                {
                  DOpEHelper::ReSizeVector(
                      GetSpaceTimeHandler()->GetStateNDoFs(global_indices[i]),
                      GetSpaceTimeHandler()->GetStateDoFsPerBlock(
                          global_indices[i]), *(_local_vectors[i]));
                  if (FileExists(global_indices[i]))
                    {
                      FetchFromDisc(global_indices[i], *_local_vectors[i]);
                    }
                  _global_to_local[global_indices[i]] = i;
                }
            }
          else
            throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),
                "StateVector::ComputeLocalVectors");
        }
    }

  /******************************************************/

  template<typename VECTOR>
    void
    StateVector<VECTOR>::ResizeLocalVectors(unsigned int size) const
    {
      //we need this function only in the store on disc case, because
      //else, we would not have dynamic Speicherverwaltung
      assert(GetBehavior() == DOpEtypes::VectorStorageType::store_on_disc);

      //just do sth, when local_vectors has not the right size
      if (_local_vectors.size() != size)
        {
          unsigned int lvsize = _local_vectors.size();
          if (lvsize < size)
            {
              _local_vectors.resize(size, NULL);
              for (unsigned int i = lvsize; i < size; i++)
                {
                  _local_vectors[i] = new VECTOR;
                }
            }
          else //i.e. (local_vectors.size() > size)
            {
              for (unsigned int i = size; i < lvsize; i++)
                {
                  delete _local_vectors[i];
                }
              //remember, resize does not touch the entrys smaller than size!
              _local_vectors.resize(size, NULL);
            }
        }
    }

}//end of namespace
/******************************************************/
/******************************************************/

template class DOpE::StateVector<dealii::BlockVector<double> >;
template class DOpE::StateVector<dealii::Vector<double> >;

