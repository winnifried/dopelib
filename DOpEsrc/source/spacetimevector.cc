/**
*
* Copyright (C) 2012-2018 by the DOpElib authors
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


#include <include/spacetimevector.h>
#include <include/dopeexception.h>
#include <include/helper.h>

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
  unsigned int SpaceTimeVector<VECTOR>::id_counter_ = 0;
  template<typename VECTOR>
  unsigned int SpaceTimeVector<VECTOR>::num_active_ = 0;

  /******************************************************/
  template<typename VECTOR>
  SpaceTimeVector<VECTOR>::SpaceTimeVector(const SpaceTimeVector<VECTOR> &ref) :
    unique_id_(id_counter_)
  {
    behavior_ = ref.GetBehavior();
    vector_type_ = ref.GetType();
    STH_ = ref.GetSpaceTimeHandler();
    sfh_ticket_ = 0;
    tmp_dir_ = ref.tmp_dir_;
    accessor_index_ = 0;
    if (behavior_ == DOpEtypes::VectorStorageType::store_on_disc)
      {
        local_vectors_.resize(1, NULL);
        local_vectors_[0] = new VECTOR;
        global_to_local_.clear();
        accessor_index_ = -3;
      }
    id_counter_++;
    current_dof_number_ = 0;

    ReInit();
  }

  /******************************************************/
  template<typename VECTOR>
  SpaceTimeVector<VECTOR>::SpaceTimeVector(const SpaceTimeHandlerBase<VECTOR> *STH,
					   DOpEtypes::VectorStorageType behavior,
					   DOpEtypes::VectorType type,
					   ParameterReader &param_reader) :
    unique_id_(id_counter_)
  {
    behavior_ = behavior;
    vector_type_=type;
    STH_ = STH;
    sfh_ticket_ = 0;
    param_reader.SetSubsection("output parameters");
    tmp_dir_ = param_reader.get_string("results_dir") + "tmp_"+DOpEtypesToString(vector_type_)+"/";
    if (behavior_ == DOpEtypes::VectorStorageType::store_on_disc)
      {
        //make the directory
        std::string command = "mkdir -p " + tmp_dir_;
        if (system(command.c_str()) != 0)
          {
            throw DOpEException("The command " + command + "failed!",
                                "SpaceTimeVector<VECTOR>::SpaceTimeVector");
          }
        //check that the directory is not alredy in use by the program
        if (num_active_ == 0)
          {
            filename_ = tmp_dir_ + "SpaceTimeVector_lock";
            assert(!filestream_.is_open());
            filestream_.open(filename_.c_str(), std::fstream::in);
            if (!filestream_.fail())
              {
                filestream_.close();
                throw DOpEException(
                  "The directory " + tmp_dir_
                  + " is probably already in use.",
                  "SpaceTimeVector<VECTOR>::SpaceTimeVector");
              }
            else
              {
                command = "touch " + tmp_dir_ + "SpaceTimeVector_lock";
                if (system(command.c_str()) != 0)
                  {
                    throw DOpEException("The command " + command + "failed!",
                                        "SpaceTimeVector<VECTOR>::SpaceTimeVector");
                  }
              }
          }
        local_vectors_.resize(1, NULL);
        local_vectors_[0] = new VECTOR;
        global_to_local_.clear();
        accessor_index_ = -3;
      }
    id_counter_++;
    num_active_++;
    current_dof_number_ = 0;
    ReInit();
  }

  /******************************************************/
  template<typename VECTOR>
  void
  SpaceTimeVector<VECTOR>::ReInit()
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
        stvector_.resize(2, NULL);
        if ( GetSpaceTimeHandler()->GetMaxTimePoint() < 1)
          {
            throw DOpEException("There are not even two time points. Are you sure this is a non stationary problem?",
                                "SpaceTimeVector::ReInit()");
          }
        for (unsigned int t = 0; t
             <= 1; t++)
          {

            accessor_ = t;
            ReSizeSpace (t);
          }
        current_dof_number_ = 0;
        accessor_ = 0;
        lock_ = false;
      }//End only_recent
    else if (!GetSpaceTimeHandler()->IsValidTicket(vector_type_,sfh_ticket_))
      {
        if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
          {
            stvector_.resize(GetSpaceTimeHandler()->GetMaxTimePoint() + 1, NULL);
            for (unsigned int t = 0; t
                 <= GetSpaceTimeHandler()->GetMaxTimePoint(); t++)
              {
                SetTimeDoFNumber(t);
                ReSizeSpace (t);
              }
          }
        else
          {
            if (GetBehavior() == DOpEtypes::VectorStorageType::store_on_disc)
              {
                stvector_information_.clear();
                stvector_information_.resize(
                  GetSpaceTimeHandler()->GetMaxTimePoint() + 1);

                //delete all old DOpE-Files in the directory
                std::string command = "rm -f " + tmp_dir_ + "*."
                                      + Utilities::int_to_string(unique_id_) + ".dope";
                if (system(command.c_str()) != 0)
                  {
                    throw DOpEException("The command " + command + "failed!",
                                        "SpaceTimeVector<VECTOR>::ReInit");
                  }
                for (unsigned int t = 0; t
                     <= GetSpaceTimeHandler()->GetMaxTimePoint(); t++)
                  {
                    //this is important, because in SetTimeDoFNumber, the
                    //vecotr stvector_information_ gets its updates!
                    SetTimeDoFNumber(t);
                    ReSizeSpace (t);
                  }

              }
            else
              {
                throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),
                                    "SpaceTimeVector<VECTOR>::ReInit");
              }
          }
        SetTimeDoFNumber(0);
        lock_ = false;
      }
  }
  /******************************************************/
  template<typename VECTOR>
  SpaceTimeVector<VECTOR>::~SpaceTimeVector()
  {
    if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem
        || GetBehavior() == DOpEtypes::VectorStorageType::only_recent)
      {
        for (unsigned int i = 0; i < stvector_.size(); i++)
          {
            assert(stvector_[i] != NULL);
            delete stvector_[i];
          }
      }
    else
      {
        if (GetBehavior() == DOpEtypes::VectorStorageType::store_on_disc)
          {
            for (unsigned int i = 0; i < local_vectors_.size(); i++)
              {
                assert(local_vectors_[i] != NULL);
                delete local_vectors_[i];
              }
            if (1 == num_active_)
              {
                std::string command = "rm -f " + tmp_dir_ + "*."
                                      + Utilities::int_to_string(unique_id_) + ".dope; rm -f "
                                      + tmp_dir_ + "SpaceTimeVector_lock";
                if (system(command.c_str()) != 0)
                  {
		    std::cout<<"The command "<< command << "failed! in SpaceTimeVector<VECTOR>::~SpaceTimeVector"<<std::endl;
		    abort();
                  }
              }
            else
              {
                std::string command = "rm -f " + tmp_dir_ + "*."
                                      + Utilities::int_to_string(unique_id_) + ".dope";
                if (system(command.c_str()) != 0)
                  {
		    std::cout<<"The command "<< command << "failed! in SpaceTimeVector<VECTOR>::~SpaceTimeVector"<<std::endl;
		    abort();
                  }
              }
            num_active_--;
          }
        else
	{
	  std::cout<<"Unknown Behavior "<< DOpEtypesToString(GetBehavior())<<" in SpaceTimeVector<VECTOR>::~SpaceTimeVector"<<std::endl;
	  abort();
	}
      }
  }

  /******************************************************/

  template<typename VECTOR>
  void
  SpaceTimeVector<VECTOR>::SetTimeDoFNumber(unsigned int dof_number,
                                        const TimeIterator &interval) const
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
            if (accessor_index_ == interval.GetIndex())
              {
                if (accessor_ != static_cast<int> (dof_number))
                  {
                    //the we have nothing to do, just store the old one
                    StoreOnDisc();
                    //and set the accessor_ to the new number.
                    accessor_ = static_cast<int> (dof_number);
                  }
              }
            else
              {
                //so we have to load everything anew.
                StoreOnDisc();
                ComputeLocalVectors(interval);
                accessor_ = dof_number;
                accessor_index_ = interval.GetIndex();

              }
          }
        else
          {
            throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),
                                "SpaceTimeVector<VECTOR>::SetTimeDoFNumber");
          }
      }
  }

  /******************************************************/
  template<typename VECTOR>
  void
  SpaceTimeVector<VECTOR>::SetTimeDoFNumber(unsigned int time_point) const
  {
     if (GetBehavior() == DOpEtypes::VectorStorageType::only_recent)
      {
        if ( time_point - current_dof_number_ == 0)
          {
            //nothing to do
          }
        else if ( time_point - current_dof_number_ == 1)
          {
            //We moved to the next time point.
            accessor_ = (accessor_ + 1)%2;
            current_dof_number_ = time_point;
            //Resize spatial vector.
            ReSizeSpace (time_point);
          }
        else
          {
            //invalid movement in time
            throw DOpEException("Invalid movement in time. Using the only_recent behavior you may only move forward in time by exatly one time_dof per update. To reset the time to the initial value call the ReInit method of this vector.","SpaceTimeVector::SetTimeDoFNumber");
          }
      }
    else if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
      {
        accessor_ = static_cast<int> (time_point);
        assert(accessor_ < static_cast<int>(stvector_.size()));
      }
    else
      {
        if (GetBehavior() == DOpEtypes::VectorStorageType::store_on_disc)
          {
            StoreOnDisc();
            accessor_ = time_point;
            accessor_index_ = -3;
            global_to_local_.clear();
            global_to_local_[accessor_] = 0;
            ResizeLocalVectors(1);

            GetSpaceTimeHandler ()->ReinitVector (*local_vectors_[0],
                                                  vector_type_,time_point);
            if (FileExists(accessor_))
              {
                FetchFromDisc(accessor_, *local_vectors_[0]);
              }
            stvector_information_.at(time_point).size_
              = local_vectors_[global_to_local_[accessor_]]->size();
          }
        else
          {
            throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),
                                "SpaceTimeVector<VECTOR>::SetTimeDoFNumber");
          }
      }
  }

  /******************************************************/
  template<typename VECTOR>
  VECTOR &
  SpaceTimeVector<VECTOR>::GetSpacialVector()
  {
    if ( GetBehavior() == DOpEtypes::VectorStorageType::fullmem
         || GetBehavior() == DOpEtypes::VectorStorageType::only_recent)
      {
        if (accessor_ >= 0)
          {
            assert(stvector_[accessor_] != NULL);
            return *(stvector_[accessor_]);
          }
        else
          return local_stvector_;
      }
    else
      {
        if (GetBehavior() == DOpEtypes::VectorStorageType::store_on_disc)
          {
            if (accessor_ >= 0)
              {
                assert(global_to_local_[accessor_] <local_vectors_.size());
                return *(local_vectors_[global_to_local_[accessor_]]);
              }
            else
              return local_stvector_;
          }
        throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),
                            "SpaceTimeVector<VECTOR>::GetSpacialVector");
      }
  }

  /******************************************************/
  template<typename VECTOR>
  const VECTOR &
  SpaceTimeVector<VECTOR>::GetSpacialVector() const
  {
    if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem
        || GetBehavior() == DOpEtypes::VectorStorageType::only_recent)
      {
        if (accessor_ >= 0)
          {
            assert(stvector_[accessor_] != NULL);
            return *(stvector_[accessor_]);
          }
        else
          return local_stvector_;
      }
    else
      {
        if (GetBehavior() == DOpEtypes::VectorStorageType::store_on_disc)
          {
            if (accessor_ >= 0)
              {
                return *(local_vectors_[global_to_local_[accessor_]]);
              }
            else
              return local_stvector_;
          }
        throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),
                            "SpaceTimeVector<VECTOR>::GetSpacialVector const");
      }
  }

  /******************************************************/
  template<typename VECTOR>
  VECTOR &
  SpaceTimeVector<VECTOR>::GetNextSpacialVector()
  {
    if ( GetBehavior() == DOpEtypes::VectorStorageType::fullmem
         || GetBehavior() == DOpEtypes::VectorStorageType::only_recent)
      {
        if (accessor_ >= 0 && accessor_ +1 < (int) stvector_.size())
          {
            assert(stvector_[accessor_+1] != NULL);
            return *(stvector_[accessor_+1]);
          }
        throw DOpEException("No Next Vector Available ",
                            "SpaceTimeVector<VECTOR>::GetNextSpacialVector");
        abort();
      }
    else
      {
        if (GetBehavior() == DOpEtypes::VectorStorageType::store_on_disc)
          {
            if (accessor_ >= 0 && accessor_ +1 < (int) global_to_local_.size())
              {
                assert(global_to_local_[accessor_+1] <local_vectors_.size());
                return *(local_vectors_[global_to_local_[accessor_+1]]);
              }
            throw DOpEException("No Next Vector Available ",
                                "SpaceTimeVector<VECTOR>::GetNextSpacialVector");
            abort();
          }
        throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),
                            "SpaceTimeVector<VECTOR>::GetNextSpacialVector");
      }
  }

  /******************************************************/
  template<typename VECTOR>
  const VECTOR &
  SpaceTimeVector<VECTOR>::GetNextSpacialVector() const
  {
    if ( GetBehavior() == DOpEtypes::VectorStorageType::fullmem
         || GetBehavior() == DOpEtypes::VectorStorageType::only_recent)
      {
        if (accessor_ >= 0 && accessor_ +1< (int) stvector_.size())
          {
            assert(stvector_[accessor_+1] != NULL);
            return *(stvector_[accessor_+1]);
          }
        throw DOpEException("No Next Vector Available ",
                            "SpaceTimeVector<VECTOR>::GetNextSpacialVector const");
        abort();
      }
    else
      {
        if (GetBehavior() == DOpEtypes::VectorStorageType::store_on_disc)
          {
            if (accessor_ >= 0 && accessor_ +1 < (int) global_to_local_.size())
              {
                assert(global_to_local_[accessor_+1] <local_vectors_.size());
                return *(local_vectors_[global_to_local_[accessor_+1]]);
              }
            throw DOpEException("No Next Vector Available ",
                                "SpaceTimeVector<VECTOR>::GetNextSpacialVector const");
            abort();
          }
        throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),
                            "SpaceTimeVector<VECTOR>::GetNextSpacialVector const");
      }
  }

  /******************************************************/
  template<typename VECTOR>
  VECTOR &
  SpaceTimeVector<VECTOR>::GetPreviousSpacialVector()
  {
    if ( GetBehavior() == DOpEtypes::VectorStorageType::fullmem
         || GetBehavior() == DOpEtypes::VectorStorageType::only_recent)
      {
        if (accessor_ > 0 )
          {
            assert(stvector_[accessor_-1] != NULL);
            return *(stvector_[accessor_-1]);
          }
        throw DOpEException("No Previous Vector Available ",
                            "SpaceTimeVector<VECTOR>::GetPreviousSpacialVector");
        abort();
      }
    else
      {
        if (GetBehavior() == DOpEtypes::VectorStorageType::store_on_disc)
          {
            if (accessor_ > 0 )
              {
                assert(global_to_local_[accessor_-1] <local_vectors_.size());
                return *(local_vectors_[global_to_local_[accessor_-1]]);
              }
            throw DOpEException("No Previous Vector Available ",
                                "SpaceTimeVector<VECTOR>::GetPreviousSpacialVector");
            abort();
          }
        throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),
                            "SpaceTimeVector<VECTOR>::GetPreviousSpacialVector");
      }
  }
  /******************************************************/
  template<typename VECTOR>
  const VECTOR &
  SpaceTimeVector<VECTOR>::GetPreviousSpacialVector() const
  {
    if ( GetBehavior() == DOpEtypes::VectorStorageType::fullmem
         || GetBehavior() == DOpEtypes::VectorStorageType::only_recent)
      {
        if (accessor_ > 0 )
          {
            assert(stvector_[accessor_-1] != NULL);
            return *(stvector_[accessor_-1]);
          }
        throw DOpEException("No Previous Vector Available ",
                            "SpaceTimeVector<VECTOR>::GetPreviousSpacialVector const");
        abort();
      }
    else
      {
        if (GetBehavior() == DOpEtypes::VectorStorageType::store_on_disc)
          {
            if (accessor_ > 0 )
              {
                assert(global_to_local_[accessor_-1] <local_vectors_.size());
                return *(local_vectors_[global_to_local_[accessor_-1]]);
              }
            throw DOpEException("No Previous Vector Available ",
                                "SpaceTimeVector<VECTOR>::GetPreviousSpacialVector const");
            abort();
          }
        throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),
                            "SpaceTimeVector<VECTOR>::GetPreviousSpacialVector const");
      }
  }

  /******************************************************/
  template<typename VECTOR>
  const Vector<double> &
  SpaceTimeVector<VECTOR>::GetSpacialVectorCopy() const
  {
    if (lock_)
      {
        throw DOpEException(
          "Trying to create a new copy while the old is still in use!",
          "SpaceTimeVector::GetSpacialVectorCopy");
      }
    lock_ = true;

    if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem
        || GetBehavior() == DOpEtypes::VectorStorageType::only_recent)
      {
        if (accessor_ >= 0)
          {
            assert(stvector_[accessor_] != NULL);
            copy_stvector_ = *(stvector_[accessor_]);
          }
        else
          copy_stvector_ = local_stvector_;
      }
    else
      {
        if (GetBehavior() == DOpEtypes::VectorStorageType::store_on_disc)
          {
            if (accessor_ >= 0)
              {
                copy_stvector_ = *(local_vectors_[global_to_local_[accessor_]]);
              }
            else
              copy_stvector_ = local_stvector_;
          }
        else
          {
            throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),
                                "SpaceTimeVector<VECTOR>::GetSpacialVectorCopy");
          }
      }
    return copy_stvector_;
  }

  /******************************************************/
  template<typename VECTOR>
  void
  SpaceTimeVector<VECTOR>::operator=(double value)
  {
    if (lock_)
      {
        throw DOpEException(
          "Trying to use operator= while a copy is in use!",
          "SpaceTimeVector::operator=");
      }
    else
      {
        if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
          {
            for (unsigned int i = 0; i < stvector_.size(); i++)
              {
                assert(stvector_[i] != NULL);
                stvector_[i]->operator=(value);
              }
            SetTimeDoFNumber(0);
          }
        else if (GetBehavior() == DOpEtypes::VectorStorageType::only_recent)
          {
            //No sense in doing so! Only usefull to initialize with zero
            if (value != 0.)
              {
                throw DOpEException("Using this function with any value other than zero is not supported in the only_recent behavior",
                                    "SpaceTimeVector::operator=");
              }
            for (unsigned int i = 0; i < stvector_.size(); i++)
              {
                assert(stvector_[i] != NULL);
                stvector_[i]->operator=(value);
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
                    assert(global_to_local_[accessor_]==0);
                    local_vectors_[0]->operator=(value);
                  }
                SetTimeDoFNumber(0);
              }
            else
              {
                throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),
                                    "SpaceTimeVector<VECTOR>::operator=");
              }
          }
      }
  }

  /******************************************************/
  template<typename VECTOR>
  void
  SpaceTimeVector<VECTOR>::operator=(const SpaceTimeVector &dq)
  {
    if (lock_)
      {
        throw DOpEException(
          "Trying to use operator= while a copy is in use!",
          "SpaceTimeVector::operator=");
      }
    else
      {
        if (GetBehavior() == dq.GetBehavior())
          {
            if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
              {
                if (dq.stvector_.size() != stvector_.size())
                  {
                    if (dq.stvector_.size() > stvector_.size())
                      {
                        unsigned int s = stvector_.size();
                        stvector_.resize(dq.stvector_.size(), NULL);
                        for (unsigned int i = s; i < stvector_.size(); i++)
                          {
                            assert(stvector_[i] == NULL);
                            stvector_[i] = new VECTOR;
                          }
                      }
                    else
                      {
                        for (unsigned int i = stvector_.size() - 1; i
                             >= dq.stvector_.size(); i--)
                          {
                            assert(stvector_[i] != NULL);
                            delete stvector_[i];
                            stvector_[i] = NULL;
                            stvector_.pop_back();
                          }
                        assert(stvector_.size() == dq.stvector_.size());
                      }
                  }

                for (unsigned int i = 0; i < stvector_.size(); i++)
                  {
                    assert(stvector_[i] != NULL);
                    assert(dq.stvector_[i] != NULL);
                    stvector_[i]->operator=(*(dq.stvector_[i]));
                  }
                SetTimeDoFNumber(0);
              }//endif fullmem
            else if (GetBehavior() == DOpEtypes::VectorStorageType::only_recent)
              {
                //No sense in doing so!
                throw DOpEException("Using this function is not supported in the only_recent behavior",
                                    "SpaceTimeVector::operator=");
              }
            else
              {
                if (GetBehavior() == DOpEtypes::VectorStorageType::store_on_disc)
                  {
                    //Delete all vectors on the disc.
                    std::string command = "mkdir -p " + tmp_dir_ + "; rm -f "
                                          + tmp_dir_ + "*." + Utilities::int_to_string(
                                            unique_id_) + ".dope";
                    if (system(command.c_str()) != 0)
                      {
                        throw DOpEException(
                          "The command " + command + "failed!",
                          "SpaceTimeVector<VECTOR>::operator=");
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
                        command = "cp " + dq.filename_ + " " + filename_;
                        if (system(command.c_str()) != 0)
                          {
                            throw DOpEException(
                              "The command " + command + "failed!",
                              "SpaceTimeVector<VECTOR>::operator=");
                          }
                        stvector_information_.at(t).on_disc_ = true;
                      }
                    //Make sure that no old spatial vectores are stored in local_vectors_.
                    ResizeLocalVectors(1);
                    accessor_ = -1; //We set this so that SetTimeDoFNumber(0) does not store something!
                    SetTimeDoFNumber(0);
                  }
                else
                  {
                    throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),
                                        "SpaceTimeVector<VECTOR>::operator=");
                  }
              }
          }
        else
          {
            throw DOpEException(
              "Own Behavior does not match dq.Behavior. Own Behavior:"
              + DOpEtypesToString(GetBehavior()) + " but dq.Behavior is "
              + DOpEtypesToString(dq.GetBehavior()),
              "SpaceTimeVector<VECTOR>::operator=");
          }
      }
  }

  /******************************************************/
  template<typename VECTOR>
  void
  SpaceTimeVector<VECTOR>::operator+=(const SpaceTimeVector &dq)
  {
    if (lock_)
      {
        throw DOpEException(
          "Trying to use operator+= while a copy is in use!",
          "SpaceTimeVector::operator+=");
      }
    else
      {
        if (GetBehavior() == dq.GetBehavior())
          {
            if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
              {
                assert(dq.stvector_.size() == stvector_.size());
                for (unsigned int i = 0; i < stvector_.size(); i++)
                  {
                    assert(stvector_[i] != NULL);
                    assert(dq.stvector_[i] != NULL);
                    stvector_[i]->operator+=(*(dq.stvector_[i]));
                  }
                SetTimeDoFNumber(0);
              }//endif fullmem
            else if (GetBehavior() == DOpEtypes::VectorStorageType::only_recent)
              {
                //No sense in doing so!
                throw DOpEException("Using this function is not supported in the only_recent behavior",
                                    "SpaceTimeVector::operator+=");
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
                        SetTimeDoFNumber(t);//this makes sure that everything is stored an local_vectors_ has length 1.
                        GetSpaceTimeHandler ()->ReinitVector (local_stvector_,
                                                              vector_type_,t);
                        dq.FetchFromDisc(t, local_stvector_);
                        assert(global_to_local_[accessor_]==0);
                        local_vectors_[0]->operator+=(local_stvector_);
                      }
                    SetTimeDoFNumber(0);
                  }
                else
                  throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),
                                      "SpaceTimeVector<VECTOR>::operator+=");
              }
          }
        else
          {
            throw DOpEException(
              "Own Behavior does not match dq.Behavior. Own Behavior:"
              + DOpEtypesToString(GetBehavior()) + " but dq.GetBehavior is "
              + DOpEtypesToString(dq.GetBehavior()),
              "SpaceTimeVector<VECTOR>::operator+=");
          }
      }
  }

  /******************************************************/
  template<typename VECTOR>
  void
  SpaceTimeVector<VECTOR>::operator*=(double value)
  {
    if (lock_)
      {
        throw DOpEException(
          "Trying to use operator*= while a copy is in use!",
          "SpaceTimeVector::operator*=");
      }
    else
      {
        if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
          {
            for (unsigned int i = 0; i < stvector_.size(); i++)
              {
                assert(stvector_[i] != NULL);
                stvector_[i]->operator*=(value);
              }
            SetTimeDoFNumber(0);
          }//endif fullmem
        else if (GetBehavior() == DOpEtypes::VectorStorageType::only_recent)
          {
            //No sense in doing so!
            throw DOpEException("Using this function is not supported in the only_recent behavior",
                                "SpaceTimeVector::operator*=");
          }
        else
          {
            if (GetBehavior() == DOpEtypes::VectorStorageType::store_on_disc)
              {
                for (unsigned int t = 0; t
                     <= GetSpaceTimeHandler()->GetMaxTimePoint(); t++)
                  {
                    SetTimeDoFNumber(t);//this makes sure that everything is stored an local_vectors_ has length 1.
                    assert(global_to_local_[accessor_]==0);
                    local_vectors_[0]->operator*=(value);
                  }
                SetTimeDoFNumber(0);
              }
            else
              {
                throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),
                                    "SpaceTimeVector<VECTOR>::operator*=");
              }
          }
      }
  }

  /******************************************************/
  template<typename VECTOR>
  double
  SpaceTimeVector<VECTOR>::operator*(const SpaceTimeVector &dq) const
  {
    if (GetBehavior() == dq.GetBehavior())
      {
        if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
          {
            assert(dq.stvector_.size() == stvector_.size());

            double ret = 0.;
            for (unsigned int i = 0; i < stvector_.size(); i++)
              {
                assert(stvector_[i] != NULL);
                assert(dq.stvector_[i] != NULL);
                ret += stvector_[i]->operator*(*(dq.stvector_[i]));
              }
            return ret;
          }//endif fullmem
        if (GetBehavior() == DOpEtypes::VectorStorageType::only_recent)
          {
            //No sense in doing so!
            throw DOpEException("Using this function is not supported in the only_recent behavior",
                                "SpaceTimeVector::operator*");
          }
        if (GetBehavior() == DOpEtypes::VectorStorageType::store_on_disc)
          {
            if (lock_ || dq.lock_)
              {
                throw DOpEException(
                  "Trying to use operator* while a copy is in use!",
                  "SpaceTimeVector::operator*");
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
                    GetSpaceTimeHandler ()->ReinitVector (local_stvector_,
                                                          vector_type_,t);
                    FetchFromDisc(t, local_stvector_);
                    dq.GetSpaceTimeHandler ()->ReinitVector (dq.local_stvector_,
                                                             vector_type_,t);
                    dq.FetchFromDisc(t, dq.local_stvector_);
                    ret += local_stvector_.operator*(dq.local_stvector_);
                  }
                return ret;
              }
          }
        else
          {
            throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),
                                "SpaceTimeVector<VECTOR>::operator*");
          }
      }
    else
      {
        throw DOpEException(
          "Own Behavior does not match dq.Behavior. Own Behavior:"
          + DOpEtypesToString(GetBehavior()) + " but dq.Behavior is "
          + DOpEtypesToString(dq.GetBehavior()),
          "SpaceTimeVector<VECTOR>::operator*");
      }
  }

  /******************************************************/
  template<typename VECTOR>
  void
  SpaceTimeVector<VECTOR>::add(double s, const SpaceTimeVector &dq)
  {
    if (lock_)
      {
        throw DOpEException(
          "Trying to use add while a copy is in use!",
          "SpaceTimeVector::add");
      }
    else
      {
        if (GetBehavior() == dq.GetBehavior())
          {
            if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
              {
                assert(dq.stvector_.size() == stvector_.size());

                for (unsigned int i = 0; i < stvector_.size(); i++)
                  {
                    assert(stvector_[i] != NULL);
                    assert(dq.stvector_[i] != NULL);
                    stvector_[i]->add(s, *(dq.stvector_[i]));
                  }
                SetTimeDoFNumber(0);
              }//endif fullmem
            else if (GetBehavior() == DOpEtypes::VectorStorageType::only_recent)
              {
                //No sense in doing so!
                throw DOpEException("Using this function is not supported in the only_recent behavior",
                                    "SpaceTimeVector::add");
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
                        SetTimeDoFNumber(t);//this makes sure that everything is stored an local_vectors_ has length 1.
                        GetSpaceTimeHandler ()->ReinitVector (local_stvector_,
                                                              vector_type_,t);
                        dq.FetchFromDisc(t, local_stvector_);
                        assert(global_to_local_[accessor_]==0);
                        local_vectors_[0]->add(s, local_stvector_);
                      }
                    SetTimeDoFNumber(0);
                  }
                else
                  throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),
                                      "SpaceTimeVector<VECTOR>::add");
              }
          }
        else
          {
            throw DOpEException(
              "Own Behavior does not match dq.Behavior. Own Behavior:"
              + DOpEtypesToString(GetBehavior()) + " but dq.Behavior is "
              + DOpEtypesToString(dq.GetBehavior()),
              "SpaceTimeVector<VECTOR>::equ");
          }
      }
  }

  /******************************************************/
  template<typename VECTOR>
  void
  SpaceTimeVector<VECTOR>::equ(double s, const SpaceTimeVector &dq)
  {
    if (lock_)
      {
        throw DOpEException(
          "Trying to use equ while a copy is in use!",
          "SpaceTimeVector::equ");
      }
    else
      {
        if (GetBehavior() == dq.GetBehavior())
          {
            if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
              {
                assert(dq.stvector_.size() == stvector_.size());
                for (unsigned int i = 0; i < stvector_.size(); i++)
                  {
                    assert(stvector_[i] != NULL);
                    assert(dq.stvector_[i] != NULL);
                    stvector_[i]->equ(s, *(dq.stvector_[i]));
                  }
                SetTimeDoFNumber(0);
              }//endif fullmem
            else if (GetBehavior() == DOpEtypes::VectorStorageType::only_recent)
              {
                //No sense in doing so!
                throw DOpEException("Using this function is not supported in the only_recent behavior",
                                    "SpaceTimeVector::equ");
              }
            else
              {
                if (GetBehavior() == DOpEtypes::VectorStorageType::store_on_disc)
                  {
                    dq.StoreOnDisc();
                    for (unsigned int t = 0; t
                         <= dq.GetSpaceTimeHandler()->GetMaxTimePoint(); t++)
                      {
                        SetTimeDoFNumber(t);//this makes sure that everything is stored an local_vectors_ has length 1.
                        GetSpaceTimeHandler ()->ReinitVector (local_stvector_,
                                                              vector_type_,t);
                        dq.FetchFromDisc(t, local_stvector_);
                        assert(global_to_local_[accessor_]==0);
                        local_vectors_[0]->equ(s, local_stvector_);

                      }
                    SetTimeDoFNumber(0);
                  }
                else
                  throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),
                                      "SpaceTimeVector<VECTOR>::equ");
              }
          }
        else
          {
            throw DOpEException(
              "Own Behavior does not match dq.Behavior. Own Behavior:"
              + DOpEtypesToString(GetBehavior()) + " but dq.Behavior is "
              + DOpEtypesToString(dq.GetBehavior()),
              "SpaceTimeVector<VECTOR>::equ");
          }
      }
  }

  /******************************************************/
  template<typename VECTOR>
  void
  SpaceTimeVector<VECTOR>::PrintInfos(std::stringstream &out)
  {
    if (GetSpaceTimeHandler()->GetMaxTimePoint() == 0)
      {
        if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
          {
            assert(stvector_.size()==1);
            out << "\t" << stvector_[0]->size() << std::endl;
          }
        else
          {
            if (GetBehavior() == DOpEtypes::VectorStorageType::store_on_disc)
              {
                SetTimeDoFNumber(0);
                assert(global_to_local_[accessor_]==0);
                out << "\t" << local_vectors_[0]->size() << std::endl;
              }
            else
              throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),
                                  "SpaceTimeyVector::PrintInfos");
          }
      }
    else
      {
        if (GetBehavior() == DOpEtypes::VectorStorageType::fullmem)
          {
            out << "\tNumber of Timepoints: " << stvector_.size() << std::endl;
            unsigned int min_dofs = 0;
            unsigned int max_dofs = 0;
            unsigned int total_dofs = 0;
            unsigned int this_size = 0;
            for (unsigned int i = 0; i < stvector_.size(); i++)
              {
                this_size = stvector_[i]->size();
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
            this_size = stvector_[accessor_]->size();
            out << "\tSpatial DoFs: " << this_size << std::endl;
          }
        else
          {
            if (GetBehavior() == DOpEtypes::VectorStorageType::store_on_disc)
              {
                out << "\tNumber of Timepoints: "
                    << stvector_information_.size() << std::endl;
                unsigned int min_dofs = 0;
                unsigned int max_dofs = 0;
                unsigned int total_dofs = 0;
                unsigned int this_size = 0;
                for (unsigned int i = 0; i < stvector_information_.size(); i++)
                  {
                    assert(stvector_information_.at(i).size_!=-1);
                    this_size = stvector_information_.at(i).size_;
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
                                  "SpaceTimeVector::PrintInfos");
          }
      }
  }

  /******************************************************/
  template<typename VECTOR>
  bool
  SpaceTimeVector<VECTOR>::FileExists(unsigned int time_point) const
  {
    return stvector_information_.at(time_point).on_disc_;
  }

  /******************************************************/
  template<typename VECTOR>
  void
  SpaceTimeVector<VECTOR>::MakeName(unsigned int time_point) const
  {
    assert(time_point<100000);
    filename_ = tmp_dir_ +DOpEtypesToString(vector_type_)+"vector." + Utilities::int_to_string(
                  time_point, 5) + "." + Utilities::int_to_string(unique_id_) + ".dope";
  }

  /******************************************************/
  template<typename VECTOR>
  void
  SpaceTimeVector<VECTOR>::StoreOnDisc() const
  {
    //make sure that accessor has the chance to indicate sth valid
    if (accessor_ >= 0)
      {
        //now if there is something to store, do it
        if (local_vectors_[global_to_local_[accessor_]]->size() != 0)
          {
            MakeName(accessor_);
            assert(!filestream_.is_open());
            filestream_.open(filename_.c_str(), std::fstream::out);
            if (!filestream_.fail())
              {
                DOpEHelper::write (*local_vectors_[global_to_local_[accessor_]],
                  filestream_);
                filestream_.close();
                stvector_information_.at(accessor_).on_disc_ = true;
              }
            else
              {
                throw DOpEException(
                  "Could not store " + filename_ + "on disc.",
                  "SpaceTimeVector<VECTOR>::StoreOnDisc");
              }
          }
      }
  }

  /******************************************************/
  template<typename VECTOR>
  void
  SpaceTimeVector<VECTOR>::FetchFromDisc(unsigned int time_point, VECTOR &vector) const
  {
    MakeName(time_point);
    assert(!filestream_.is_open());
    filestream_.open(filename_.c_str(), std::fstream::in);
    if (!filestream_.fail())
      {
        DOpEHelper::read (vector, filestream_);
        filestream_.close();
      }
    else
      {
        throw DOpEException("Could not fetch " + filename_ + "from disc.",
                            "SpaceTimeVector<VECTOR>::StoreOnDisc");
      }
  }

  /******************************************************/
  template<typename VECTOR>
  void
  SpaceTimeVector<VECTOR>::SwapPtr(VECTOR *&a, VECTOR *&b) const
  {
    VECTOR *tmp = a;
    a = b;
    b = tmp;
  }

  /******************************************************/
  template<typename VECTOR>
  void
  SpaceTimeVector<VECTOR>::ComputeLocalVectors(const TimeIterator &interval) const
  {
    //FIXME: It appears, the global_to_local_ stuff is
    // needed only for store_on_disc? Then why do we
    //take care of the other cases?

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
            local_vectors_[i] = stvector_[global_indices[i]];
            global_to_local_[global_indices[i]] = i;
          }
      }
    else if (GetBehavior() == DOpEtypes::VectorStorageType::only_recent)
      {
        if (n_local_dofs != 2)
          {
            throw DOpEException("Behavior only_recent can only work with 2 local DoFs per time interval.", "SpaceTimeVector::ComputeLocalVectors");
          }
        if (global_indices[1] != current_dof_number_)
          {
            //We do not have the needed information!
            throw DOpEException("When using the behavior only_recent you may only use the currently selected interval.", "SpaceTimeVector::ComputeLocalVectors");
          }
        local_vectors_.resize(n_local_dofs);
        for (unsigned int i = 0; i < n_local_dofs; ++i)
          {
            local_vectors_[i] = stvector_[(accessor_ + ((i+1))%2)%2];
            //FixMe: Is this the right coordinate, or should this be
            //something else?
            global_to_local_[global_indices[i]] = i;
          }
      }
    else
      {
        if (GetBehavior() == DOpEtypes::VectorStorageType::store_on_disc)
          {
            //Resize local_vectors_ to the right size.
            //We have to take more care in this case because of the
            //dynamic memory management
            ResizeLocalVectors(n_local_dofs);
            for (unsigned int i = 0; i < n_local_dofs; ++i)
              {
                GetSpaceTimeHandler ()->ReinitVector (*local_vectors_[i],
                                                      vector_type_, global_indices[i]);

                if (FileExists(global_indices[i]))
                  {
                    FetchFromDisc(global_indices[i], *local_vectors_[i]);
                  }
                global_to_local_[global_indices[i]] = i;
              }
          }
        else
          throw DOpEException("Unknown Behavior " + DOpEtypesToString(GetBehavior()),
                              "SpaceTimeVector::ComputeLocalVectors");
      }
  }

  /******************************************************/

  template<typename VECTOR>
  void
  SpaceTimeVector<VECTOR>::ResizeLocalVectors(unsigned int size) const
  {
    //we need this function only in the store on disc case, because
    //else, we would not have dynamic Speicherverwaltung
    assert(GetBehavior() == DOpEtypes::VectorStorageType::store_on_disc);

    //just do sth, when local_vectors has not the right size
    if (local_vectors_.size() != size)
      {
        unsigned int lvsize = local_vectors_.size();
        if (lvsize < size)
          {
            local_vectors_.resize(size, NULL);
            for (unsigned int i = lvsize; i < size; i++)
              {
                local_vectors_[i] = new VECTOR;
              }
          }
        else //i.e. (local_vectors.size() > size)
          {
            for (unsigned int i = size; i < lvsize; i++)
              {
                delete local_vectors_[i];
              }
            //remember, resize does not touch the entrys smaller than size!
            local_vectors_.resize(size, NULL);
          }
      }
  }

}//end of namespace
/******************************************************/
/******************************************************/

template class DOpE::SpaceTimeVector<dealii::BlockVector<double> >;
template class DOpE::SpaceTimeVector<dealii::Vector<double> >;

#ifdef DOPELIB_WITH_TRILINOS
#if DEAL_II_VERSION_GTE(9,0,0)
template class DOpE::SpaceTimeVector<dealii::TrilinosWrappers::MPI::BlockVector>;
template class DOpE::SpaceTimeVector<dealii::TrilinosWrappers::MPI::Vector>;
#endif
#endif

