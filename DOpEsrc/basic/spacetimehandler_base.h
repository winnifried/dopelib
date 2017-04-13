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

#ifndef SPACE_TIME_HANDLER_BASE_H_
#define SPACE_TIME_HANDLER_BASE_H_

#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_vector_base.h>
#include <deal.II/lac/block_vector.h>

#include <vector>
#include <iostream>
#include <sstream>
#include <limits>

#include <include/timedofhandler.h>
#include <include/timeiterator.h>
#include <container/refinementcontainer.h>
#include <basic/dirichletdescriptor.h>
#include <basic/dopetypes.h>

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

    SpaceTimeHandlerBase(DOpEtypes::ControlType control_type = DOpEtypes::stationary) : control_type_(control_type)
    {
      time_triangulation_ = NULL;
      state_ticket_ = 1;
      control_ticket_ = 1;
    }

    SpaceTimeHandlerBase(dealii::Triangulation<1> &times, DOpEtypes::ControlType type = DOpEtypes::stationary) :
      tdfh_(times), interval_(tdfh_.first_interval()), control_type_(type)
    {
      time_triangulation_ = &times;
      state_ticket_ = 1;
      control_ticket_ = 1;
    }

    SpaceTimeHandlerBase(dealii::Triangulation<1> &times,
                         const dealii::FiniteElement<1> &fe,
                         DOpEtypes::ControlType type = DOpEtypes::stationary) :
      tdfh_(times, fe), interval_(tdfh_.first_interval()), control_type_(type)
    {
      time_triangulation_ = &times;
      state_ticket_ = 1;
      control_ticket_ = 1;
    }


    virtual ~SpaceTimeHandlerBase()
    {
      tdfh_.clear();
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
    void SetInterval(const TimeIterator &it)
    {
      interval_ = it;
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


    /**
     * Returns the ControlType.
     */

    DOpEtypes::ControlType GetControlType() const
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
    virtual unsigned int GetControlNDoFs(int /*time_point*/ = -1) const
    {
      abort();
    }
    /**
     * Returns the DoFs for the state vector at the given point time_point.
     * If time_point==-1, it returns the DoFs for the current time which has
     * to be set prior to calling this function using SetTimeDoFNumber.
     *
     * @ param time_point Indicating the time at which we want to know the DoFs. -1 means now.
     */
    virtual unsigned int GetStateNDoFs(int time_point = -1) const = 0;
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
    virtual const std::vector<unsigned int> &GetControlDoFsPerBlock(int /*time_point*/ = -1) const
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
    virtual const std::vector<unsigned int> &GetStateDoFsPerBlock(int time_point = -1) const = 0;
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
     * DEPRECATED. FIXME  We have to think about how to realize this with the new
     * TimeDoFHandler.
     *
     * This functions is used to interpolate Vectors after a refinement of the temporal mesh.
     *
     * @param t  An unsigned int specifying the current time point in the new enumeration
     *
     * @return An unsigned int indicating the number of the timepoint t
     *         before the refinement. If the timepoint has not existed before
     *         the return value is identical to argument t given to this function.
     */
    virtual unsigned int NewTimePointToOldTimePoint(unsigned int t) const = 0;


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

    virtual void SpatialMeshTransferState(const VECTOR & /*old_values*/, VECTOR & /*new_values*/) const
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
     *                        RefinementContainer (for global refinement), RefineFixedFraction,
     *                        RefineFixedNumber and RefineOptimized.
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

  private:
    mutable TimeDoFHandler tdfh_;//FIXME Is it really necessary for tdfh_ and interval_ to be mutable? this is really ugly
    mutable TimeIterator interval_;
    dealii::Triangulation<1> *time_triangulation_;
    unsigned int control_ticket_;
    unsigned int state_ticket_;
    mutable DOpEtypes::ControlType control_type_;
  };

}

#endif
