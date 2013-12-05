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

#ifndef _SPACE_TIME_HANDLER_BASE_H_
#define _SPACE_TIME_HANDLER_BASE_H_

#include <lac/vector.h>
#include <lac/block_vector_base.h>
#include <lac/block_vector.h>

#include <vector>
#include <iostream>
#include <sstream>
#include <limits>

#include "timedofhandler.h"
#include "timeiterator.h"
#include "dopetypes.h"

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

    SpaceTimeHandlerBase(DOpEtypes::ControlType control_type = DOpEtypes::undefined) : _control_type(control_type)
    {
      _state_ticket = 1;
      _control_ticket = 1;
    }

    SpaceTimeHandlerBase(const dealii::Triangulation<1> & times, DOpEtypes::ControlType type = DOpEtypes::undefined) :
      _tdfh(times), _interval(_tdfh.first_interval()), _control_type(type)
    {
      _state_ticket = 1;
      _control_ticket = 1;
    }

    SpaceTimeHandlerBase(const dealii::Triangulation<1> & times,
        const dealii::FiniteElement<1>& fe,
        DOpEtypes::ControlType type = DOpEtypes::undefined) :
          _tdfh(times, fe), _interval(_tdfh.first_interval()), _control_type(type)
    {
      _state_ticket = 1;
      _control_ticket = 1;
    }


    virtual ~SpaceTimeHandlerBase()
    {
      _tdfh.clear();
    }

    /**
     * This function has to get called after temporal refinement.
     */
    void ReInitTime()
    {
      IncrementStateTicket();
      _tdfh.distribute_dofs();
      //FIXME When we have temporal discretization also for control and constraint,
      //one has to increment here the _control_ticket!
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
      return (_tdfh.GetNbrOfDoFs()-1);//because we start counting at 0.
    }

    /**
     * Returns the number of intervals in the actual time triangulation.
     */
    unsigned int GetNbrOfIntervals() const
    {
      return _tdfh.GetNbrOfIntervals();
    }


    /**
     * Sets the current interval
     *
     * @param interval   The current interval.
     */
    void SetInterval(const TimeIterator& it)
    {
      _interval = it;
    }


    /**
     * Returns the actual interval, which has to be set prior to this function through Setinterval.
     *
     * @return An iterator 'pointing' to the prevoisly given _interval.
     */
    const TimeIterator& GetInterval() const
    {
      return _interval;
    }


    /**
     *  Returns the time corresponding to the time_point.
     *
     *  @param time_point      The time_point of interest.
     *  @return A double containing the time at the given time point.
     */
    double GetTime(unsigned int time_point) const
    {

      return _tdfh.GetTime(time_point);
    }


    /**
     * Returns the TimeDoFHandler.
     */
    const TimeDoFHandler& GetTimeDoFHandler() const
    {
      return _tdfh;
    }


    /**
     * Returns the Vector of all time points.
     */
    const std::vector<double>& GetTimes() const
    {
      return _tdfh.GetTimes();
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
    GetTimes(const TimeIterator& interval, std::vector<double>& local_times) const
    {
      return _tdfh.GetTimes(interval, local_times);
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
    bool IsValidStateTicket(unsigned int& ticket) const
    {
      bool ret = (ticket == _state_ticket);
      ticket = _state_ticket;
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
    bool IsValidControlTicket(unsigned int& ticket) const
    {
      bool ret = (ticket == _control_ticket);
      ticket = _control_ticket;
      return ret;
    }


    /**
     * Returns the ControlType.
     */

    DOpEtypes::ControlType GetControlType() const
    {
      return _control_type;
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
      InterpolateControl(VECTOR& /*result*/, const std::vector<VECTOR*> &/*local_vectors*/,
          double /*t*/, const TimeIterator&/*interval*/) const { abort(); }
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
      InterpolateState(VECTOR& /*result*/, const std::vector<VECTOR*> &/*local_vectors*/,
		       double /*t*/, const TimeIterator&/*interval*/) const = 0;
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
    virtual void InterpolateConstraint(VECTOR& /*result*/, const std::vector<VECTOR*> &/*local_vectors*/,
        double /*t*/, const TimeIterator&/*interval*/) const { abort(); }

    /**
     * Returns the DoFs for the control vector at the current time which has  to be set prior to calling this function using SetTime.
     */
    virtual unsigned int GetControlNDoFs() const { abort(); }
    /**
     * Returns the DoFs for the state vector at the given point time_point.
     * If time_point==-1, it returns the DoFs for the current time which has
     * to be set prior to calling this function using SetTimeDoFNumber.
     *
     * @ param time_point			Indicating the time at which we want to know the DoFs. -1 means now.
     */
    virtual unsigned int GetStateNDoFs(int time_point = -1) const = 0;
    /**
     * Returns the DoFs for the constraint vector at the current time which has
     *  to be set prior to calling this function using SetTime.
     */
    virtual unsigned int GetConstraintNDoFs(std::string /*name*/) const { abort(); return 0; }
    /**
     * Returns the DoFs per  block for the control vector at the current time which has
     * to be set prior to calling this function using SetTime.
     */
    virtual const std::vector<unsigned int>& GetControlDoFsPerBlock() const { abort(); }
    /**
     * Returns the DoFs per block for the state vector at the given point time_point.
     * If time_point==-1, it returns the DoFs per block for the current time which has
     * to be set prior to calling this function using SetTime.
     *
     * @ param time_point			Indicating the time at which we want to know the DoFs per block. -1 means now.
     */
    virtual const std::vector<unsigned int>& GetStateDoFsPerBlock(int time_point = -1) const = 0;
    /**
     * Returns the DoFs per  block for the constraint vector at the current time which has
     * to be set prior to calling this function using SetTime.
     */
    virtual const std::vector<unsigned int>& GetConstraintDoFsPerBlock(std::string /*name*/) const { abort(); }
    /**
     * Returns the Number of global in space and time Constraints
     */
    virtual unsigned int  GetNGlobalConstraints() const { abort(); }
    /**
     * Returns the Number of local in space and time Constraints
     */
    virtual unsigned int  GetNLocalConstraints() const { abort(); }
    
    /**
     * Returns the length of _interval;
     */
    double GetStepSize() const
    {
      return _interval.get_k();
    }
    /**
     * Returns the length of _interval++;
     */
    double GetNextStepSize() const
    {
      assert(_interval!= _tdfh.last_interval());
      double k = (++_interval).get_k();
      --_interval;
      return k;
    }
    /**
     * Returns the length of _interval--;
     */
    double GetPreviousStepSize() const
    {
      assert(_interval!= _tdfh.first_interval());
      double k = (--_interval).get_k();
      ++_interval;
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
    virtual void SpatialMeshTransferControl(const VECTOR& /*old_values*/, VECTOR& /*new_values*/) const { abort(); }
    
    virtual void SpatialMeshTransferState(const VECTOR& /*old_values*/, VECTOR& /*new_values*/) const { abort(); }

  protected:
    /**
     * Call this function if any StateDoF related stuff has changed to invalidate all previous tickets.
     */
    void IncrementStateTicket()
    {
      assert( _state_ticket < std::numeric_limits<unsigned int>::max());
      _state_ticket++;
    }

    /**
     * Call this function if any ControlDoF related stuff has changed to invalidate all previous tickets.
     */
    void IncrementControlTicket()
    {
      assert( _control_ticket < std::numeric_limits<unsigned int>::max());
      _control_ticket++;
    }

  private:
    mutable TimeDoFHandler _tdfh;//FIXME Is it really necessary for _tdfh and _interval to be mutable? this is really ugly
    mutable TimeIterator _interval;

    unsigned int _control_ticket;
    unsigned int _state_ticket;
    mutable DOpEtypes::ControlType _control_type;
};

}

#endif
