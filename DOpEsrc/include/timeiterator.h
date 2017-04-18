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

#ifndef TIMEITERATOR_H_
#define TIMEITERATOR_H_

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/grid/tria_iterator.h>

namespace DOpE
{
  namespace IteratorState
  {

    /**
     *   The four states an iterator can be in: valid, past-the-end, before-the-beginning and
     *   invalid.
     */
    enum IteratorStates
    {
      /// Iterator points to a valid object
      valid,
      /// Iterator reached end of container
      past_the_end,
      ///
      before_the_beginning,
      /// Iterator is invalid, probably due to an error
      invalid
    };
  }

  typedef dealii::DoFHandler<1>::active_cell_iterator active_cell_it;


  /**
   * An iterator for the timedofhandler that allows us
   * to work with the 1d triangulation of the interval
   * in the same way that we would do if we just had numbers
   * for the endpoints of the intervals.
   */
  class TimeIterator
  {
  public:
    //constructors

    /**
     * The object constructed by this is not usable!
     */
    TimeIterator()
    {
      present_index_ = -3;//i.e. invalid
    }

    /**
     * Self explanatory.
     */
    TimeIterator(const active_cell_it &element, int present_index) :
      element_(element)
    {
      present_index_ = present_index;
    }

    /**
     * Copy constructor
     */
    TimeIterator(const TimeIterator &it)
    {
      element_ = it.element_;
      present_index_ = it.present_index_;
    }

    /**
     * This translates the actual value of present_index_ into an state.
     */
    IteratorState::IteratorStates
    GetState() const
    {
      if (present_index_ >= 0 && present_index_
          < static_cast<int>(element_->get_triangulation().n_active_cells()))
        return IteratorState::valid;
      else
        {
          if (present_index_ == -1)
            return IteratorState::past_the_end;
          else
            {
              if (present_index_ == -2)
                return IteratorState::before_the_beginning;
              else
                return IteratorState::invalid;
            }
        }
    }

    /**
     * Returns present_index_, which is the number of the interval we
     * are currently pointing to, starting at 0 (negative values correspond to
     * different states the iterator is currently in.)
     */
    int
    GetIndex() const
    {
      return present_index_;
    }

    TimeIterator &
    operator=(const TimeIterator &element)
    {
      element_ = element.element_;
      present_index_ = element.present_index_;
      return *this;
    }

    void
    Initialize(const active_cell_it &element, int present_index)
    {
      element_ = element;
      present_index_ = present_index;
    }

    bool
    operator==(const TimeIterator &element)
    {
      if (GetState() == element.GetState())
        return (element_ == element.getelement_());
      else
        return false;
    }

    bool
    operator!=(const TimeIterator &element)
    {
      if (GetState() == element.GetState())
        return (element_ != element.getelement_());
      else
        return true;
    }

    /**
     * Returns the center of the actual element.
     */
    double
    get_center() const
    {
      assert(GetState()==IteratorState::valid);
      return element_->center()(0);
    }

    /**
     * Returns the  location of the left hand side of the actual element.
     */
    double
    get_left() const
    {
      assert(GetState()==IteratorState::valid);
      return element_->face(0)->center()(0);
    }

    /**
     * Returns the location of the right hand side of the actual element.
     */
    double
    get_right() const
    {
      assert(GetState()==IteratorState::valid);
      return element_->face(1)->center()(0);
    }

    /**
     * Returns the length of the actual element.
     */
    double
    get_k() const
    {
      assert(GetState()==IteratorState::valid);
      return element_->diameter();
    }

    const active_cell_it &
    getelement_() const
    {
      return element_;
    }

    void
    get_time_dof_indices(std::vector<unsigned int> &local_dof_indices) const
    {
      assert(GetState()==IteratorState::valid);
      element_->get_dof_indices(local_dof_indices);
    }

    /**
     * Prefix++ operator
     */
    TimeIterator &
    operator++()
    {
      assert(GetState()==IteratorState::valid);
      ++present_index_;
      if (present_index_ < static_cast<int>(element_->get_triangulation().n_active_cells()))
        {
          element_ = element_->neighbor(1);
        }
      else
        {
          present_index_ = -1;
        }

      return *this;
    }

    /**
     * Postfix ++ operator
     */
    TimeIterator
    operator++(int)
    {
      assert(GetState()==IteratorState::valid);
      TimeIterator tmp(element_, present_index_);

      ++present_index_;
      if (present_index_ < static_cast<int>(element_->get_triangulation().n_active_cells()))
        {
          element_ = element_->neighbor(1);
        }
      else
        {
          present_index_ = -1;
        }
      return tmp;
    }

    /**
     * Prefix -- operator
     */
    TimeIterator &
    operator--()
    {
      assert(GetState()==IteratorState::valid);
      --present_index_;
      if (present_index_ >= 0)
        {
          element_ = element_->neighbor(0);
        }
      else
        {
          present_index_ = -2;
        }

      return *this;
    }

    /**
     * Postfix -- operator
     */
    TimeIterator
    operator--(int)
    {
      assert(GetState()==IteratorState::valid);
      TimeIterator tmp(element_, present_index_);
      if (present_index_ >= 0)
        {
          element_ = element_->neighbor(0);
        }
      else
        {
          present_index_ = -2;
        }
      return tmp;
    }
  private:
    active_cell_it element_;
    int present_index_;
  };
}

#endif /* TIMEITERATOR_H_ */
