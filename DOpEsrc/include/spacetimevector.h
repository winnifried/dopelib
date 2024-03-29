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

#ifndef SPACETIME_VECTOR_H_
#define SPACETIME_VECTOR_H_

// TODO remove ...
//#pragma GCC diagnostic ignored "-Wterminate"

#include <basic/spacetimehandler_base.h>
#include <basic/dopetypes.h>
#include <include/parameterreader.h>
#include <include/helper.h>
#include <include/parallel_vectors.h>

#include <deal.II/base/utilities.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_vector_base.h>
#include <deal.II/lac/block_vector.h>

#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>

namespace DOpE
{
  /**
   * This class represents the SpaceTimevector.
   *
   * @tparam <VECTOR>     Class summarizing all storage and access capabilities of the
   *                      control-vectors, state-vectors, ...
   *                      (i.e. dealii::Vector<double> or dealii::BlockVector<double>)
   */
  template<typename VECTOR>
  class SpaceTimeVector
  {
  public:
    //FIXME this is not a real copyconstructor, it just
    //uses the information of ref about size and so on. Is this correct?
    SpaceTimeVector(const SpaceTimeVector<VECTOR> &ref);
    SpaceTimeVector(const SpaceTimeHandlerBase<VECTOR> *STH,
                    DOpEtypes::VectorStorageType behavior,
                    DOpEtypes::VectorType type,
                    DOpEtypes::VectorAction action_,
                    ParameterReader &param_reader);
    ~SpaceTimeVector();

    /**
     * Sets the time in the vector. This Function or SetTimeDoFNumber or SetTime
     * must be called before calling GetSpacialVector
     * in order to load the required data.
     *
     * @param dof_number          An unsigned int containing the global time DoF number
     *                            of the given  interval we are interested in.
     * @param interval            A TimeIterator. The interval we are currently looking on.
     *
     */
    void SetTimeDoFNumber(unsigned int dof_number,
                          const TimeIterator &interval) const;
    /**
     * Sets the time in the vector. This Function or SetTime or SetTimeDoFNumber
     * must be called before calling GetSpacialVector
     * in order to load the required data.
     *
     * Only for internal use, externally a time intervall must be specified
     *
     * @param time_point   An unsigned integer. This gives the number of the point in the  time mesh.
     *
     */
    void SetTimeDoFNumber(unsigned int time_point) const;

    /**
     * Returns a reference to the spatial vector associated to the last time given by SetTime*
     */
    VECTOR &GetSpacialVector();
    /**
     * Returns a const reference to the spatial vector associated to the last time given by SetTime* or SetTimeDoFNumber
     */
    const VECTOR &GetSpacialVector() const;
    /**
    * Returns a const reference to the spatial vector associated to the last time given by SetTime* or SetTimeDoFNumber, transfered to the DoFs at a different time point!
    * This locks the vector, so UnlockCopy needs to be called to free the memory!
    */
    const VECTOR &GetSpacialVectorWithTemporalTransfer(unsigned int from_time_dof, unsigned int to_time_dof) const;
    /**
    * Returns a const reference to the spatial vector associated to the last time given by SetTime* or SetTimeDoFNumber, transfered to the DoFs at a different time point!
    * This locks the vector, so UnlockCopy needs to be called to free the memory!
    */
    const VECTOR &GetPreviousSpacialVectorWithTemporalTransfer(unsigned int from_time_dof, unsigned int to_time_dof) const;
    /**
      * Analog to GetSpacialVector, but the next timepoint in natural time direction
      */
    VECTOR &GetNextSpacialVector();
    /**
     * Analog to GetSpacialVector, but the next timepoint in natural time direction
     */
    const VECTOR &GetNextSpacialVector() const;
    /**
     * Analog to GetSpacialVector, but the previous timepoint in natural time direction
     */
    VECTOR &GetPreviousSpacialVector();
    /**
     * Analog to GetSpacialVector, but the previous timepoint in natural time direction
     */
    const VECTOR &GetPreviousSpacialVector() const;
    /**
     * Returns a const reference to the spatial vector associated to the last time given by SetTime*
     * This makes a copy of the real vector  in order to change the vector type.
     * To assert data integrity Only one Copy may  be obtained at any time.
     * Hence prior to calling this Function again UnLockCopy must be called.
     */
    const dealii::Vector<double> &GetSpacialVectorCopy() const;
    /**
     * Sets all the vector to a constant value. This function calls SetTime(0).
     *
     * @param value    The constant value to be assigned to the vector.
     */
    void operator=(double value);
    /**
     * Sets this vector to the values of an other given vector.
     * If required this vector is resized. This function calls SetTime(0).
     *
     * @param dq    The other vector.
     */
    void operator=(const SpaceTimeVector<VECTOR> &dq);
    /**
     * Upon completion each entry of this Vector contains the following
     * Result this = this + dq;
     * It is required that both this and dq have the same structure!
     * This function calls SetTime(0).
     *
     * @param dq    The increment.
     */
    void operator+=(const SpaceTimeVector<VECTOR> &dq);
    /**
     * Multiplies the Vector with a constant.  It expects both vectors  to be of
     * the same structure. This function calls SetTime(0).
     *
     * @param a    A double to be multiplied with the vector.
     */
    void operator*=(double a);
    /**
     * Computes the Euclidean scalar product of this vector with the argument.
     * Both Vectors must have the same structure.
     *
     * @param dq    The argument for the computation of the scalarproduct.
     * @return      A double containing the scalar product.
     */
    double operator*(const SpaceTimeVector<VECTOR> &dq) const;
    /**
     * Sets this vector adds a multiple of an other vector to this vector.
     * this = this + s * dq
     * It expects both vectors  to be of the same structure.
     * This function calls SetTime(0).
     *
     * @param s    A double, by which the other vector is scaled.
     * @param dq   The other vector.
     */
    void add(double s, const SpaceTimeVector<VECTOR> &dq);
    /**
     * Sets this vector to the values of an other given vector.
     * The vector is not resized! It expects both vectors  to be of
     * the same structure. This function calls SetTime(0).
     *
     * @param dq    The other vector.
     */
    void equ(double s, const SpaceTimeVector<VECTOR> &dq);

    /**
     * Sets this vector to the componentwise maximum of its own
     * entries and that of the other vector
     * The vector is not resized!
     *
     * @param dq    The other vector.
     */
    void max(const SpaceTimeVector &dq);
    /**
     * Sets this vector to the componentwise minimum of its own
     * entries and that of the other vector
     * The vector is not resized!
     *
     * @param dq    The other vector.
     */
    void min(const SpaceTimeVector &dq);

    /**
     * Computes the component wise product of this vector with the argument.
     */
    void comp_mult(const SpaceTimeVector &dq);

    /**
     * Inverts the elements of the vector component wise
     */
    void comp_invert();

    /**
     * Initializes this vector according to the signs in it.
     *
     * @param smaller   value to be taken if sign is negative
     * @param larger    value to be taken if sign is positive
     * @param unclear   value to be taken if sign is unclear
     * @param TOL       if abs(value) < TOL we consider the sign to be unclear
     */
    void init_by_sign(double smaller, double larger, double unclear, double TOL = 1.e-10);

    /**
      * Computes the norm given by name of the vector.
      * Feasible values are "infty", and "l1"
      * The string restriction defines if only certain values are
      * to be considered. Currently "all" and "positive" are feasible
      * Meaning that either all or only the positive entries are
      * considered.
      */
    double Norm(std::string name,std::string restriction = "all") const;

    /**
     * Prints Information on this vector into the given stream.
     *
     * @param out    The output stream.
     */
    void PrintInfos(std::stringstream &out);
    /**
     * This unlocks the function GetSpacialVectorCopy
     */
    void UnLockCopy() const
    {
      lock_ = false;
    }
    /**
     * This returns the behavior of the SpaceTimeVector
     * Currently implemented are the following possibilities
     * @par  fullmem          Means there is a spatial vector for each time point. The whole vector
     *                        is stored in main memory.
     *
     * @par  store_on_disc    Means there are only three spatial vectors (for the actual timepoint
     *                        and his two neighbors) stored in the main memory whereas the rest of
     *                        the spacetimevector is stored on the hard disc.
     *
     * @return               A string indicating the behavior.
     */
    DOpEtypes::VectorStorageType GetBehavior() const
    {
      return behavior_;
    }

    /**
     * This returns the type of the SpaceTimeVector
     * i.e. control, state, ...
     */
    DOpEtypes::VectorType GetType() const
    {
      return vector_type_;
    }

    /**
     * This returns the action type of the SpaceTimeVector
     * i.e. initial, stationary, nonstationary
     */
    DOpEtypes::VectorAction GetAction() const
    {
      return vector_action_;
    }

    /**
     * @return               A const pointer to the SpaceTimeHandler associated with this vector.
     */
    const SpaceTimeHandlerBase<VECTOR> *GetSpaceTimeHandler() const
    {
      return STH_;
    }
    /**
     * Call if the SpaceTimeHandler has changed to reinitialize vector sizes.
     *
     */
    void ReInit();

  private:
    struct SpatialVectorInfos
    {
      int size_;
      bool on_disc_;

      SpatialVectorInfos(int size = -1, bool on_disc = false)
      {
        size_ = size;
        on_disc_ = on_disc;
      }
    };
    /**
     * This function resizes the spatial vector at a prior given time point.
     * Hence SetTimeDoFNumber must be called before this function.
     */
    // Note: the template<...> thing can be ignored, this is just a wrapper between block and non-block vectors.
    template <typename _VECTOR = VECTOR, typename std::enable_if<
                IsBlockVector<_VECTOR>::value, int>::type = 0>
    void
    ReSizeSpace (const unsigned int time_point) const;

    template <typename _VECTOR = VECTOR, typename std::enable_if<
                !IsBlockVector<_VECTOR>::value, int>::type = 0>
    void
    ReSizeSpace (const unsigned int time_point) const;

    /**
     * Sets the membervariable '_filename' to the name of the file (e.g. the whole path!) corresponding to 'time_point'.
     *
     * @ param time_point     The timepoint we are actually interested in.
     */
    void MakeName(unsigned int time_point) const;
    /**
     * Stores the BlockVector stored in stvector_[1] on the Disc. The name of the file will be
     * createt by the function call 'MakeName(local_stvector_.at(1))'.
     *
     */
    void StoreOnDisc() const;
    /**
     * This function reads the BlockVector<double> stored in the file with the name
     * 'MakeName(time_point)' and stores him in vector.
     *
     * This function is set const due to compatibility reasons.
     *
     * @ param time_point   The timepoint we are actually interested in.
     * @ param vector       A BlockVector in which the Data read out from Disc will be stored.
     */
    void FetchFromDisc(unsigned int time_point, VECTOR &vector) const;
    /**
     * This function checks if a file named 'filename_' exists in tmp_dir_.
     *
     * @param time_point      The timepoint we are actually interested in.
     *
     * @return                A bool indicating whether the file exists or not.
     */
    bool FileExists(unsigned int time_point) const;
    /**
     * The Function swaps the Pointers a and b.
     *
     * @param a, b      References to two BlockVector<double>-Pointers which will be swapped.
     */
    void SwapPtr(VECTOR *&a,VECTOR *&b ) const;

    /**
     * Writes the vectors corresponding to the current interval
     * into local_vectors_, and adjusts global_to_local_;
     */
    void ComputeLocalVectors(const TimeIterator &interval) const;

    /**
     * Helper function, resizes local_vectors_ to the size given by size.
     */
    void ResizeLocalVectors(unsigned int size) const;

    mutable std::vector<VECTOR *> stvector_;
    mutable std::vector<SpatialVectorInfos> stvector_information_;

    mutable VECTOR local_stvector_;
    mutable dealii::Vector<double> copy_stvector_;
    mutable int accessor_;

    mutable bool lock_;

    //Needed in the store_on_disc case for read/write operations on the hard disc
    mutable std::string filename_;
    mutable std::fstream filestream_;

    //Needed in the only_recent case to decide if the operation is allowed.
    mutable unsigned int current_dof_number_;

    //pointer to the dofs in the actual interval. Is only used if the interval is set!
    mutable std::vector<VECTOR *> local_vectors_;

    //Map: global time dof index - local time DoF index
    mutable std::map<unsigned int, unsigned int> global_to_local_;
    //the index of the interval, to which the vectors stored in local_vectors belong
    mutable int accessor_index_;

    DOpEtypes::VectorStorageType behavior_;
    DOpEtypes::VectorType vector_type_;
    DOpEtypes::VectorAction vector_action_;
    std::string tmp_dir_;
    unsigned int sfh_ticket_;

    const SpaceTimeHandlerBase<VECTOR> *STH_;
    const unsigned int unique_id_;

    static unsigned int id_counter_;
    static unsigned int num_active_;
  };

  template <typename VECTOR>
  template <typename _VECTOR, typename std::enable_if<
              IsBlockVector<_VECTOR>::value, int>::type>
  void
  SpaceTimeVector<VECTOR>::ReSizeSpace (const unsigned int time_point) const
  {
    const auto ndofs = GetSpaceTimeHandler ()->GetNDoFs (vector_type_, time_point);
    const auto &dofs_per_block =
      GetSpaceTimeHandler ()->GetDoFsPerBlock (vector_type_, time_point);

    if (GetBehavior () == DOpEtypes::VectorStorageType::fullmem || GetBehavior ()
        == DOpEtypes::VectorStorageType::only_recent)
      {
        if (accessor_ >= 0)
          {
            assert(stvector_.size() > (unsigned int) accessor_);
            bool existed = true;
            if (stvector_[accessor_] == NULL)
              {
                stvector_[accessor_] = new VECTOR;
                existed = false;
              }
            unsigned int nblocks = dofs_per_block.size ();
            bool reinit = false;
            if (stvector_[accessor_]->size () != ndofs)
              {
                reinit = true;
              }
            else
              {
                if (stvector_[accessor_]->n_blocks () != nblocks)
                  {
                    reinit = true;
                  }
                else
                  {
                    for (unsigned int i = 0; i < nblocks; i++)
                      if (stvector_[accessor_]->block (i).size () != dofs_per_block[i])
                        reinit = true;
                  }
              }
            if (reinit)
              {
                if (existed)
                  local_stvector_ = *(stvector_[accessor_]);

                GetSpaceTimeHandler ()->ReinitVector (*stvector_[accessor_],
                                                      vector_type_,
                                                      time_point);
                if (existed)
                  GetSpaceTimeHandler ()->SpatialMeshTransfer (vector_type_,
                                                               local_stvector_, *(stvector_[accessor_]), time_point);
              }
          }
        else
          {
            //accessor_ < 0
            if (local_stvector_.size () != ndofs)
              GetSpaceTimeHandler ()->ReinitVector (local_stvector_,
                                                    vector_type_,
                                                    time_point);
          }
      }
    else
      {
        if (GetBehavior () == DOpEtypes::VectorStorageType::store_on_disc)
          {
            if (accessor_ >= 0)
              {
                unsigned int nblocks = dofs_per_block.size ();
                bool reinit = false;
                if (local_vectors_[global_to_local_[accessor_]]->size () != ndofs)
                  {
                    reinit = true;
                  }
                else
                  {
                    if (local_vectors_[global_to_local_[accessor_]]->n_blocks () != nblocks)
                      {
                        reinit = true;
                      }
                    else
                      {
                        for (unsigned int i = 0; i < nblocks; i++)
                          if (local_vectors_[global_to_local_[accessor_]]->block (
                                i).size ()
                              != dofs_per_block[i])
                            reinit = true;
                      }
                  }
                if (reinit)
                  GetSpaceTimeHandler ()->ReinitVector (
                    *local_vectors_[global_to_local_[accessor_]],
                    vector_type_,
                    time_point);
              }
            else
              {
                if (local_stvector_.size () != ndofs)
                  GetSpaceTimeHandler ()->ReinitVector (local_stvector_,
                                                        vector_type_,
                                                        time_point);
              }
          }
        else
          throw DOpEException (
            "Unknown Behavior " + DOpEtypesToString (GetBehavior ()),
            "SpaceTimeVector<dealii::BlockVector<double> >::ReSizeSpace");
      }
  }

  template <typename VECTOR>
  template <typename _VECTOR, typename std::enable_if<
              !IsBlockVector<_VECTOR>::value, int>::type>
  void
  SpaceTimeVector<VECTOR>::ReSizeSpace (const unsigned int time_point) const
  {
    const unsigned int ndofs = GetSpaceTimeHandler ()->GetNDoFs (vector_type_, time_point);

    if (ndofs == 0)
      return;

    if (GetBehavior () == DOpEtypes::VectorStorageType::fullmem || GetBehavior ()
        == DOpEtypes::VectorStorageType::only_recent)
      {
        if (accessor_ >= 0)
          {
            bool existed = true;
            if (stvector_[accessor_] == NULL)
              {
                stvector_[accessor_] = new VECTOR;
                existed = false;
              }

            bool reinit = false;
            if (stvector_[accessor_]->size () != ndofs)
              reinit = true;

            if (reinit)
              {
                if (existed)
                  local_stvector_ = *(stvector_[accessor_]);

                GetSpaceTimeHandler ()->ReinitVector (*stvector_[accessor_],
                                                      vector_type_,
                                                      time_point);

                if (existed)
                  GetSpaceTimeHandler ()->SpatialMeshTransfer (vector_type_,
                                                               local_stvector_, *(stvector_[accessor_]),
                                                               time_point);
              }
          }
        else
          {
            //accessor < 0
            if (local_stvector_.size () != ndofs)
              GetSpaceTimeHandler ()->ReinitVector (local_stvector_,
                                                    vector_type_,
                                                    time_point);
          }
      }
    else
      {
        if (GetBehavior () == DOpEtypes::VectorStorageType::store_on_disc)
          {
            if (accessor_ >= 0)
              {
                bool reinit = false;
                if (local_vectors_[global_to_local_[accessor_]]->size () != ndofs)
                  reinit = true;

                if (reinit)
                  GetSpaceTimeHandler ()->ReinitVector (
                    *local_vectors_[global_to_local_[accessor_]],
                    vector_type_,
                    time_point);
              }
            else
              {
                if (local_stvector_.size () != ndofs)
                  GetSpaceTimeHandler ()->ReinitVector (local_stvector_,
                                                        vector_type_,
                                                        time_point);
              }
          }
        else
          throw DOpEException (
            "Unknown Behavior " + DOpEtypesToString (GetBehavior ()),
            "SpaceTimeVector<dealii::Vector<double> >::ReSizeSpace");
      }
  }

}
#endif
