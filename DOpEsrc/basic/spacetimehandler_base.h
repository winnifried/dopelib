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

#ifndef SPACE_TIME_HANDLER_BASE_H_
#define SPACE_TIME_HANDLER_BASE_H_

#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_vector_base.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/base/index_set.h>

#include <vector>
#include <iostream>
#include <sstream>
#include <limits>

#include <include/timedofhandler.h>
#include <include/timeiterator.h>
#include <container/refinementcontainer.h>
#include <basic/dirichletdescriptor.h>
#include <basic/dopetypes.h>
#include <include/parallel_vectors.h>
#include <include/helper.h>

namespace DOpE
{
  /**
   * Interface to the dimension independent functionality of a
   * SpaceTimeDoFHandler
   */
  template<typename VECTOR>//has to be a class template because you can not have virtual member function templates.
  class SpaceTimeHandlerBase
  {
  public:

    SpaceTimeHandlerBase(DOpEtypes::VectorAction control_type = DOpEtypes::stationary) : control_type_(control_type)
    {
      time_triangulation_ = NULL;
      state_ticket_ = 1;
      control_ticket_ = 1;
      time_dof_number_ = std::numeric_limits<unsigned int>::max();
    }

    SpaceTimeHandlerBase(dealii::Triangulation<1> &times, DOpEtypes::VectorAction type = DOpEtypes::stationary) :
      tdfh_(times), interval_(tdfh_.first_interval()), control_type_(type)
    {
      time_triangulation_ = &times;
      state_ticket_ = 1;
      control_ticket_ = 1;
      time_dof_number_ = std::numeric_limits<unsigned int>::max();
    }

    SpaceTimeHandlerBase(dealii::Triangulation<1> &times,
                         const dealii::FiniteElement<1> &fe,
                         DOpEtypes::VectorAction type = DOpEtypes::stationary) :
      tdfh_(times, fe), interval_(tdfh_.first_interval()), control_type_(type)
    {
      time_triangulation_ = &times;
      state_ticket_ = 1;
      control_ticket_ = 1;
      time_dof_number_ = std::numeric_limits<unsigned int>::max();
    }


    virtual ~SpaceTimeHandlerBase()
    {
      tdfh_.clear();
    }

    virtual MPI_Comm
    GetMPIComm () const
    {
      return MPI_COMM_WORLD; // TODO user provided
    }

    /**
     * This function has to get called after temporal refinement.
     */
    void ReInitTime()
    {
      IncrementStateTicket();
      tdfh_.distribute_dofs();
      //FIXME When we have temporal discretization also for control and constraint,
      //one has to increment here the control_ticket_!
    }

    /**
     * Returns the index of the latest time point (i.e. DoF) in the time mesh.
     * Counting starts at zero.
     *
     * @return The maximal feasible time point.
     *
     */
    unsigned int GetMaxTimePoint() const
    {
      return (tdfh_.GetNbrOfDoFs()-1);//because we start counting at 0.
    }

    /**
     * Returns the number of intervals in the actual time triangulation.
     */
    unsigned int GetNbrOfIntervals() const
    {
      return tdfh_.GetNbrOfIntervals();
    }


    /**
     * Sets the current interval
     *
     * @param interval   The current interval.
     */
    void SetInterval(const TimeIterator &it, unsigned int time_dof_number) const
    {
      assert(IsInIntervall(it,time_dof_number));
      interval_ = it;
      time_dof_number_ = time_dof_number;
    }

    /**
     * Returns the actual interval, which has to be set prior to this function through Setinterval.
     *
     * @return An iterator 'pointing' to the prevoisly given interval_.
     */
    const TimeIterator &GetInterval() const
    {
      return interval_;
    }

    /**
     * Returns the actual time_dof_number, which has to be set prior to this function through Setinterval.
     *
     * @return An number for the currently selected time_dof.
     */
    unsigned int GetTimeDoFNumber() const
    {
      return time_dof_number_;
    }


    /**
     *  Returns the time corresponding to the time_point.
     *
     *  @param time_point      The time_point of interest.
     *  @return A double containing the time at the given time point.
     */
    double GetTime(unsigned int time_point) const
    {

      return tdfh_.GetTime(time_point);
    }


    /**
     * Returns the TimeDoFHandler.
     */
    const TimeDoFHandler &GetTimeDoFHandler() const
    {
      return tdfh_;
    }


    /**
     * Returns the Vector of all time points.
     */
    const std::vector<double> &GetTimes() const
    {
      return tdfh_.GetTimes();
    }

    /**
     * Given an interval, this function writes the global
     *  position of the  dofs in the interval into the
     *  vector local_times. It is assumed that local_times has
     *  the correct length beforehand!
     *
     *  @param interval     The interval from which we want to extract the position of the DoFs.
     *
     *  @param local_tiems  A vector of doubles, in which we want to write the position of the
     *                      of the given interval. Needs the correct size beforehand!
     */
    void
    GetTimes(const TimeIterator &interval, std::vector<double> &local_times) const
    {
      return tdfh_.GetTimes(interval, local_times);
    }


    /**
     * Checks if a ticket is still valid, or if the SpaceTimeHandler has been updated in the meantime.
     * This is in order to allow functions that reinitialize statevectors to the size of the
     * DoFs to work only if necessary.
     *
     * @param ticket           A ticket to be checked. After the procedure the ticket is always a valid ticket.
     *                         Note that zero is always invalid.
     * @return                 A Boolean that is true if the ticket is still valid. It is false if the SpaceTimeHandler has been
     *                         Updated.
     */
    bool IsValidStateTicket(unsigned int &ticket) const
    {
      bool ret = (ticket == state_ticket_);
      ticket = state_ticket_;
      return ret;
    }

    /**
     * Checks if a ticket is still valid, or if the SpaceTimeHandler has been updated in the meantime.
     * This is in order to allow functions that reinitialize control- and constraintvectors to the size of the
     * DoFs to work only if necessary.
     *
     * @param ticket           A ticket to be checked. After the procedure the ticket is always a valid ticket.
     *                         Note that zero is always invalid.
     * @return                 A Boolean that is true if the ticket is still valid. It is false if the SpaceTimeHandler has been
     *                         Updated.
     */
    bool IsValidControlTicket(unsigned int &ticket) const
    {
      bool ret = (ticket == control_ticket_);
      ticket = control_ticket_;
      return ret;
    }

    bool IsValidTicket(const DOpEtypes::VectorType type, unsigned int &ticket) const
    {
      switch (type)
        {
        case DOpEtypes::VectorType::state:
          return IsValidStateTicket(ticket);
        case DOpEtypes::VectorType::control:
          return IsValidControlTicket(ticket);
        default:
          assert(false);
          return false;
        }
    }

    /**
     * Returns the ControlActionType.
     */

    DOpEtypes::VectorAction GetControlActionType() const
    {
      return control_type_;
    }


    /**
     * If one requires values at a time not corresponding to a degree of freedom in
     * time, one needs to interpolate this value from the others on the interval.
     * This Function is for vectors associated  with the control DoFs.
     *
     * @param result          A VECTOR where the interpolation is stored into.
     * @param local_vectors   The vectors corresponding to the time-dofs on the actual interval.
     *                        Needed for the interpolation.
     * @param t               The time at which we want to have the  interpolation
     * @param interval        The interval we are currently working on.
     */
    virtual void
    InterpolateControl(VECTOR & /*result*/, const std::vector<VECTOR *> &/*local_vectors*/,
                       double /*t*/, const TimeIterator &/*interval*/) const
    {
      abort();
    }
    /**
     * If one requires values at a time not corresponding to a degree of freedom in
     * time, one needs to interpolate this value from the others on the interval.
     * This Function is for vectors associated  with the state DoFs.
     *
     * @param result          A VECTOR where the interpolation is stored into.
     * @param local_vectors   The vectors corresponding to the time-dofs on the actual interval.
     *                        Needed for the interpolation.
     * @param t               The time at which we want to have the  interpolation
     * @param interval        The interval we are currently working on.
     */
    virtual void
    InterpolateState(VECTOR & /*result*/, const std::vector<VECTOR *> &/*local_vectors*/,
                     double /*t*/, const TimeIterator &/*interval*/) const = 0;
    /**
     * If one requires values at a time not corresponding to a degree of freedom in
     * time, one needs to interpolate this value from the others on the interval.
     * This Function is for vectors associated  with the constraint-DoFs.
     *
     * @param result          A VECTOR where the interpolation is stored into.
     * @param local_vectors   The vectors corresponding to the time-dofs on the actual interval.
     *                        Needed for the interpolation.
     * @param t               The time at which we want to have the  interpolation
     * @param interval        The interval we are currently working on.
     */
    virtual void InterpolateConstraint(VECTOR & /*result*/, const std::vector<VECTOR *> &/*local_vectors*/,
                                       double /*t*/, const TimeIterator &/*interval*/) const
    {
      abort();
    }

    /**
     * Returns the DoFs for the control vector at the given point time_point.
     * If time_point==-1, it returns the DoFs for the current time which has
     * to be set prior to calling this function using SetTimeDoFNumber.
     *
     * @ param time_point Indicating the time at which we want to know the DoFs. -1 means now.
     */
    virtual unsigned int GetControlNDoFs(unsigned int /*time_point*/ = std::numeric_limits<unsigned int>::max()) const
    {
      abort();
    }

    // TODO this function could replace all Get...NDoFs functions, currently just wraps around
    /**
     * Returns the DoFs for the vector type at the given point time_point.
     * If time_point==-1, it returns the DoFs for the current time which has
     * to be set prior to calling this function using SetTimeDoFNumber.
     *
     * @ param type Indicates for which quantity (state, constrol, constraint, local constraint)
     * we want to know the number of DoFs.
     * @ param time_point Indicating the time at which we want to know the DoFs. -1 means now.
     */
    virtual unsigned int
    GetNDoFs (const DOpEtypes::VectorType type,
              unsigned int time_point = std::numeric_limits<unsigned int>::max()) const
    {
      switch (type)
        {
        case DOpEtypes::VectorType::state:
          return GetStateNDoFs (time_point);
        case DOpEtypes::VectorType::constraint:
          return GetConstraintNDoFs ("global");
        case DOpEtypes::VectorType::local_constraint:
          return GetConstraintNDoFs ("local");
        case DOpEtypes::VectorType::control:
          return GetControlNDoFs (time_point);
        default:
          assert(false);
          return 0;
        }
    }

    /**
     * Same as above, but block-wise.
     *
     * @ param type Indicates for which quantity (state, constrol, constraint, local constraint)
     * we want to know the number of DoFs per block.
     * @ param time_point Indicating the time at which we want to know the DoFs. -1 means now.
     */
    virtual std::vector<unsigned int>
    GetDoFsPerBlock (const DOpEtypes::VectorType type,
                     unsigned int time_point = std::numeric_limits<unsigned int>::max()) const
    {
      switch (type)
        {
        case DOpEtypes::VectorType::state:
          return GetStateDoFsPerBlock (time_point);
        case DOpEtypes::VectorType::constraint:
          return GetConstraintDoFsPerBlock ("global");
        case DOpEtypes::VectorType::local_constraint:
          return GetConstraintDoFsPerBlock ("local");
        case DOpEtypes::VectorType::control:
          return GetControlDoFsPerBlock (time_point);
        default:
          abort ();
          return std::vector<unsigned int>
                 { };
        }
    }

    /**
     * Returns the locally owned DoFs for the given type of vector at given time point.
     *
     * @ param type Indicates for which quantity (state, constrol, constraint, local constraint)
     * we want to know the number of DoFs per block.
     * @ param time_point Indicating the time at which we want to know the DoFs. -1 means now.
     */
    virtual dealii::IndexSet
    GetLocallyOwnedDoFs (const DOpEtypes::VectorType type,
                         unsigned int time_point = std::numeric_limits<unsigned int>::max()) const = 0;

    /**
     * Returns the locally relevant DoFs for the given type of vector at given time point.
     *
     * @ param type Indicates for which quantity (state, constrol, constraint, local constraint)
     * we want to know the number of DoFs per block.
     * @ param time_point Indicating the time at which we want to know the DoFs. -1 means now.
     */
    virtual dealii::IndexSet
    GetLocallyRelevantDoFs (const DOpEtypes::VectorType type,
                            unsigned int time_point = std::numeric_limits<unsigned int>::max()) const = 0;

    /**
     * Returns the DoFs for the state vector at the given point time_point.
     * If time_point==-1, it returns the DoFs for the current time which has
     * to be set prior to calling this function using SetTimeDoFNumber.
     *
     * @ param time_point Indicating the time at which we want to know the DoFs. -1 means now.
     */
    virtual unsigned int GetStateNDoFs(unsigned int time_point = std::numeric_limits<unsigned int>::max()) const = 0;
    /**
     * Returns the DoFs for the constraint vector at the current time which has
     *  to be set prior to calling this function using SetTime.
     */
    virtual unsigned int GetConstraintNDoFs(std::string /*name*/) const
    {
      abort();
      return 0;
    }
    /**
     * Returns the DoFs per block for the control vector at the given point time_point.
     * If time_point==-1, it returns the DoFs per block for the current time which has
     * to be set prior to calling this function using SetTime.
     *
     * @ param time_point Indicating the time at which we want to know the DoFs per block. -1 means now.
     */
    virtual const std::vector<unsigned int> &GetControlDoFsPerBlock(unsigned int /*time_point*/ = std::numeric_limits<unsigned int>::max()) const
    {
      abort();
    }
    /**
     * Returns the DoFs per block for the state vector at the given point time_point.
     * If time_point==-1, it returns the DoFs per block for the current time which has
     * to be set prior to calling this function using SetTime.
     *
     * @ param time_point Indicating the time at which we want to know the DoFs per block. -1 means now.
     */
    virtual const std::vector<unsigned int> &GetStateDoFsPerBlock(unsigned int time_point = std::numeric_limits<unsigned int>::max()) const = 0;
    /**
     * Returns the DoFs per  block for the constraint vector at the current time which has
     * to be set prior to calling this function using SetTime.
     */
    virtual const std::vector<unsigned int> &GetConstraintDoFsPerBlock(std::string /*name*/) const
    {
      abort();
    }
    /**
     * Returns the Number of global in space and time Constraints
     */
    virtual unsigned int  GetNGlobalConstraints() const
    {
      abort();
    }
    /**
     * Returns the Number of local in space and time Constraints
     */
    virtual unsigned int  GetNLocalConstraints() const
    {
      abort();
    }

    /**
     * Returns the length of interval_;
     */
    double GetStepSize() const
    {
      return interval_.get_k();
    }
    /**
     * Returns the length of interval_++;
     */
    double GetNextStepSize() const
    {
      assert(interval_!= tdfh_.last_interval());
      double k = (++interval_).get_k();
      --interval_;
      return k;
    }
    /**
     * Returns the length of interval_--;
     */
    double GetPreviousStepSize() const
    {
      assert(interval_!= tdfh_.first_interval());
      double k = (--interval_).get_k();
      ++interval_;
      return k;
    }


    /**
     * This functions is used to interpolate Control Vectors after a refinement of the spatial mesh.
     * It expects that the timepoint of the mesh has been initialized correctly prior to
     * calling this function
     *
     * @param old_values  The Vector on the mesh before refinement.
     * @param new_values  The Vector where the interpolation should be placed
     *
     */
    virtual void SpatialMeshTransferControl(const VECTOR & /*old_values*/, VECTOR & /*new_values*/) const
    {
      abort();
    }

    virtual void SpatialMeshTransferState(const VECTOR & /*old_values*/, VECTOR & /*new_values*/, unsigned int /*time_point*/= std::numeric_limits<unsigned int>::max()) const
    {
      abort();
    }

    void SpatialMeshTransfer(const DOpEtypes::VectorType type, const VECTOR &old_values, VECTOR &new_values, unsigned int time_point= std::numeric_limits<unsigned int>::max()) const
    {
      switch (type)
        {
        case DOpEtypes::VectorType::state:
          SpatialMeshTransferState(old_values,new_values,time_point);
          return ;
        case DOpEtypes::VectorType::control:
          SpatialMeshTransferControl(old_values,new_values);
          return ;
        default:
          assert(false);
          return ;
        }
    }

    /**
     * This functions is used to interpolate Control Vectors between different meshes of different time steps.
     * It expects that the timepoint of the mesh has been initialized correctly prior to
     * calling this function
     *
     * @param new_values  The Vector to be interpolated in the dofs of the mesh at the time from_time_dof
     *        After the function call, it contains the interpolated vector in the dofs of the mesh
     *              measured at the time to_time_dof
     * @param from_time_dof
     * @param to_time_dof
     *
     * @return a boolean indicating whether a mesh transfer was done.
     */

    virtual bool TemporalMeshTransferControl( VECTOR & /*new_values*/, unsigned int /*from_time_dof*/, unsigned int /*to_time_dof*/) const
    {
      abort();
    }

    virtual bool TemporalMeshTransferState(VECTOR & /*new_values*/, unsigned int /*from_time_dof*/, unsigned int /*to_time_dof*/) const
    {
      abort();
    }

    /******************************************************/
    /**
     * This Function is used to refine the temporal mesh globally.
     * After calling a refinement function a reinitialization is required!
     *
     * @param ref_type       A DOpEtypes::RefinementType telling how to refine the
     *                       spatial mesh. Only DOpEtypes::RefinementType::global
     *                       is allowed in this method.
     */
    void
    RefineTime(DOpEtypes::RefinementType /*ref_type*/ =
                 DOpEtypes::RefinementType::global)
    {
      //assert(ref_type == DOpEtypes::RefinementType::global);
      RefinementContainer ref_con_dummy;
      RefineTime(ref_con_dummy);
    }

    /******************************************************/
    /**
     * This Function is used to refine the temporal mesh.
     * After calling a refinement function a reinitialization is required!
     *
     * @param ref_container   Steers the local mesh refinement. Currently availabe are
     *                        RefinementContainer (for global refinement) for the
     *                        time discretization.
     */

    void
    RefineTime(const RefinementContainer &ref_container)
    {
      DOpEtypes::RefinementType ref_type = ref_container.GetRefType();

      //make sure that we do not use any coarsening
      assert(!ref_container.UsesCoarsening());
      assert(time_triangulation_ != NULL);

      if (DOpEtypes::RefinementType::global == ref_type)
        {
          time_triangulation_->set_all_refine_flags();
        }
      else
        {
          throw DOpEException("Not implemented for name =" + DOpEtypesToString(ref_type),
                              "MethodOfLines_SpaceTimeHandler::RefineTime");
        }
      time_triangulation_->prepare_coarsening_and_refinement();

      time_triangulation_->execute_coarsening_and_refinement();
      ReInitTime();
    }

    /******************************************************/
    // TODO we need only the VECTOR one of those ...
    /**
     * Initializes the given vector v at given time point.
     * Type allows to chose between control, state, constraint.
     *
     * @ param v Vector to be initialized
     * @ param type Indicates whether the vector should be state, control, constraint, etc.
     * @ param time_point Indicating the time at which we want to initialize v. -1 means now.
     */
    void
    ReinitVector (dealii::Vector<double> &v,
                  const DOpEtypes::VectorType type,
                  const unsigned int time_point = std::numeric_limits<unsigned int>::max()) const
    {
      const auto dofs = GetNDoFs (type, time_point);
      v.reinit (dofs);
    }

    /**
     * Same as above for BlockVector.
     *
     * @ param v Vector to be initialized
     * @ param type Indicates whether the vector should be state, control, constraint, etc.
     * @ param time_point Indicating the time at which we want to initialize v. -1 means now.
     */
    void
    ReinitVector (dealii::BlockVector<double> &v,
                  const DOpEtypes::VectorType type,
                  const unsigned int time_point = std::numeric_limits<unsigned int>::max()) const
    {
      const auto blocks = GetDoFsPerBlock (type, time_point);
      v.reinit (blocks);
    }

#ifdef DOPELIB_WITH_TRILINOS
    /**
     * Same as above for TrilinosWrappers::MPI::Vector.
     *
     * @ param v Vector to be initialized
     * @ param type Indicates whether the vector should be state, control, constraint, etc.
     * @ param time_point Indicating the time at which we want to initialize v. -1 means now.
     */
    void
    ReinitVector (dealii::TrilinosWrappers::MPI::Vector &v,
                  const DOpEtypes::VectorType type,
                  const unsigned int time_point = std::numeric_limits<unsigned int>::max()) const
    {
      const auto locally_owned = GetLocallyOwnedDoFs (type, time_point);
      const auto locally_relevant = GetLocallyRelevantDoFs (type,
                                                            time_point);
      v.reinit (locally_owned, locally_relevant, GetMPIComm ());
      return;
    }

    /**
     * Same as above for TrilinosWrappers::MPI::BlockVector.
     *
     * @ param v Vector to be initialized
     * @ param type Indicates whether the vector should be state, control, constraint, etc.
     * @ param time_point Indicating the time at which we want to initialize v. -1 means now.
     */
    void
    ReinitVector (dealii::TrilinosWrappers::MPI::BlockVector &v,
                  const DOpEtypes::VectorType type,
                  const unsigned int time_point = std::numeric_limits<unsigned int>::max()) const
    {
      const auto block_locally_owned = DOpEHelper::split_blockwise (
                                         GetLocallyOwnedDoFs (type, time_point),
                                         GetDoFsPerBlock (type, time_point));
      const auto block_locally_relevant = DOpEHelper::split_blockwise (
                                            GetLocallyRelevantDoFs (type, time_point),
                                            GetDoFsPerBlock (type, time_point));
      v.reinit (block_locally_owned, block_locally_relevant, GetMPIComm ());
    }
#endif

    /******************************************************/
    /**
     *  Here, the given Vector v (associated to DoFs) is printed to
     *  a file of *.vtk, *.vtu, *.gpl, or *.txt format. However, in later implementations other
     *  file formats will be available.
     *
     *  @param v           The Vector to write to a file.
     *  @param name        The names of the variables, e.g., in a fluid problem: v1, v2, p.
     *  @param outfile     The basic name for the output file to print.
     *  @param dof_type    Has the DoF type: state or control.
     *  @param filetype    The filetype. Actually, *.vtk outputs are possible.
     */
    // TODO enum
    // TODO param
    virtual void
    WriteToFile (const VECTOR &v,
                 std::string name,
                 std::string outfile,
                 std::string dof_type,
                 std::string filetype) = 0;

    /**
     *  Here, the given Vector v (containing element-related data) is printed to
     *  a file of *.vtk format. However, in later implementations other
     *  file formats will be available.
     *
     *  @param v           The Vector to write to a file.
     *  @param name        The names of the variables, e.g., in a fluid problem: v1, v2, p.
     *  @param outfile     The basic name for the output file to print.
     *  @param dof_type    Has the DoF type: state or control.
     *  @param filetype    The filetype. Actually, *.vtk outputs are possible.
     *  @param n_patches   Patches used for visualization.
     */
    virtual void
    WriteToFileElementwise(const Vector<float> &v, std::string name,
                           std::string outfile, std::string dof_type, std::string filetype, int n_patches) = 0 ;

  protected:
    /**
     * Call this function if any StateDoF related stuff has changed to invalidate all previous tickets.
     */
    void IncrementStateTicket()
    {
      assert( state_ticket_ < std::numeric_limits<unsigned int>::max());
      state_ticket_++;
    }

    /**
     * Call this function if any ControlDoF related stuff has changed to invalidate all previous tickets.
     */
    void IncrementControlTicket()
    {
      assert( control_ticket_ < std::numeric_limits<unsigned int>::max());
      control_ticket_++;
    }

    bool IsInIntervall(const TimeIterator &it, unsigned int time_dof_number) const
    {
      unsigned int n_dofs_per_interval = this->GetTimeDoFHandler().GetLocalNbrOfDoFs();
      std::vector<unsigned int> local_to_global(n_dofs_per_interval);
      it.get_time_dof_indices(local_to_global);
      return (find(local_to_global.begin(),local_to_global.end(),time_dof_number)!=local_to_global.end());
    }
  private:
    mutable TimeDoFHandler tdfh_;//FIXME Is it really necessary for tdfh_ and interval_ to be mutable? this is really ugly
    mutable TimeIterator interval_;
    mutable unsigned int time_dof_number_;
    dealii::Triangulation<1> *time_triangulation_;
    unsigned int control_ticket_;
    unsigned int state_ticket_;
    mutable DOpEtypes::VectorAction control_type_;
  };

}

#endif
