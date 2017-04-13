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

#ifndef TIMEDOFHANDLER_H_
#define TIMEDOFHANDLER_H_

//DOpE
#include <include/timeiterator.h>
#include <include/dopeexception.h>
#include <include/versionscheck.h>

//deal.ii
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/base/point.h>

//c++
#include <vector>


using namespace dealii;

namespace DOpE
{

  /**
   * TODO
   * 1) make sure that the dofs get renumbered from left to right from 0 to n_dofs!
   *  It is implemented, but is has to get tested.
   * 2) Interpolation after temporal refinement in the dofhandlers!.
   * 3) At the moment, one is not able to change the Finite Element, because we are currently only looking at
   * timestep methods.
   */

  /**
   * DoFHandler responsible for the management of the timedofs.
   *
   */
  class TimeDoFHandler : public dealii::DoFHandler<1>
  {
  public:
    //Constructors
    TimeDoFHandler()
    {
      fe_ = NULL;
      times_.resize(1, 0.);
      dofs_per_element_ = 1;
      need_delete_ = false;
      initialized_ = false;
    }

    /**
     * TODO for later uses, if one wants to use galerkin methods.
     * On top of that, we should get the fe from the timestepping problem.
     */
    TimeDoFHandler(const Triangulation<1> &tria,
                   const dealii::FiniteElement<1> &fe)
      : dealii::DoFHandler<1>(tria)
    {
      fe_ = &fe;
      this->distribute_dofs();
      need_delete_ = false;
    }

    TimeDoFHandler(const Triangulation<1> &tria)
      : dealii::DoFHandler<1>(tria)
    {
      fe_ = new dealii::FE_Q<1>(1);
      this->distribute_dofs();
      need_delete_ = true;
    }

    //Destructor
    ~TimeDoFHandler()
    {
      this->clear();

      if (need_delete_)
        {
          delete fe_;
        }
      fe_ = NULL;
    }

    /**
     * Go through the triangulation and distribute the degrees of freedoms
     * needed for the given finite element.
     */
    void
    distribute_dofs()
    {
      if (fe_ != NULL)
        {
          dealii::DoFHandler<1>::distribute_dofs(*fe_);
          //make sure that the dofs are numbered 'downstream' (referring to the time variable!)

#if DEAL_II_VERSION_GTE(7,3,0)
          dealii::DoFRenumbering::downstream<dealii::DoFHandler<1> >(*this,
                                                                     dealii::Point<1>(1.), true);
#else
          dealii::DoFRenumbering::downstream<dealii::DoFHandler<1>, 1>(*this,
              dealii::Point<1>(1.), true);
#endif
          find_ends();
          compute_times();
          initialized_ = true;
          dofs_per_element_ = this->get_fe().dofs_per_cell;
        }
    }

    /**
     * Returns the number of intervals
     */
    unsigned int
    GetNbrOfIntervals() const
    {
      if (initialized_)
#if DEAL_II_VERSION_GTE(8,4,0)
        return this->get_triangulation().n_active_cells();
#else
        return this->get_tria().n_active_cells();
#endif
      else
        return 0;
    }

    /**
     * Returns the number of dofs.
     */
    unsigned int
    GetNbrOfDoFs() const
    {
      if (initialized_)
        return this->n_dofs();
      else
        return 1;
    }

    /**
     * Returns a vector containing the position of the dofs.
     */
    const std::vector<double> &
    GetTimes() const
    {
      return times_;
    }

    /**
     * Given an interval, this function writes the global
     *  position of the  dofs in the interval into the
     *  vector local_times. It is assumed that local_times has
     *  the right length beforehand!
     */
    void
    GetTimes(const TimeIterator &interval,
             std::vector<double> &local_times) const
    {
      assert(local_times.size() == GetLocalNbrOfDoFs());

      std::vector<unsigned int> global_dof_indices(GetLocalNbrOfDoFs());
      interval.get_time_dof_indices(global_dof_indices);
      for (unsigned int i = 0; i < GetLocalNbrOfDoFs(); ++i)
        {
          local_times[i] = times_[global_dof_indices[i]];
        }
    }

    /**
     * Returns the position of the given timedof.
     */
    double
    GetTime(unsigned int timestep)
    {
      assert(timestep < times_.size());
      return times_[timestep];
    }

    /**
     * Returns an TimeIterator pointing to the first time_interval.
     */
    TimeIterator
    first_interval() const
    {
      assert(initialized_);
      return first_interval_;
    }

    /**
     * Returns an TimeIterator pointing to the element before(!)
     *  the  first time_interval.
     */
    TimeIterator
    before_first_interval() const
    {
      assert(initialized_);
      return before_first_interval_;
    }

    /**
     * Returns an active_cell_iterator pointing to the last time_interval.
     */
    TimeIterator
    last_interval() const
    {
      assert(initialized_);
      return last_interval_;
    }

    /**
     * Returns an active_cell_iterator pointing to the element after(!)
     * the last time_interval.
     */
    TimeIterator
    after_last_interval() const
    {
      assert(initialized_);
      return after_last_interval_;
    }

    /**
     * Returns the number of dofs per interval.
     */
    unsigned int
    GetLocalNbrOfDoFs() const
    {
      return dofs_per_element_;
    }

  private:
    using dealii::DoFHandler<1>::distribute_dofs;   //Remove the clang warning that the function below overloads the distribute_dofs
    /**
     * Find the first and last interval and store them
     * in first_interval_ as well as last_interval_
     */
    void
    find_ends()
    {
      DoFHandler<1>::active_cell_iterator element = this->begin_active();
#if DEAL_II_VERSION_GTE(8,3,0)
      while (element->face(0)->boundary_id() != 0)
#else
      while (element->face(0)->boundary_indicator() != 0)
#endif
        {
          element = element->neighbor(0);
        }
      first_interval_.Initialize(element, 0);
      before_first_interval_.Initialize(element, -2);

      element = this->begin_active();
#if DEAL_II_VERSION_GTE(8,3,0)
      while (element->face(1)->boundary_id() != 1)
#else
      while (element->face(1)-t>boundary_indicator() != 1)
#endif
        {
          element = element->neighbor(1);
        }
#if DEAL_II_VERSION_GTE(8,4,0)
      assert(static_cast<int>(this->get_triangulation().n_active_cells() - 1) >= 0);
      last_interval_.Initialize(element,
                                static_cast<int>(this->get_triangulation().n_active_cells() - 1));
#else
      assert(static_cast<int>(this->get_tria().n_active_cells() - 1) >= 0);
      last_interval_.Initialize(element,
                                static_cast<int>(this->get_tria().n_active_cells() - 1));
#endif
      after_last_interval_.Initialize(element, -1);
    }

    /**
     * After distributing the dofs, compute the location of the dofs.
     * This corresponds to the 'old' times - vector.
     */
    void
    compute_times()
    {
      assert(this->get_fe().has_support_points());
      //get the right length
      times_.resize(this->n_dofs());

      //geht the support points and build a quadrature rule from it
      dealii::Quadrature<1> quadrature_formula(
        this->get_fe().get_unit_support_points());

      //after that, build a fevalues object. We need only the quadrature points!
      //typename dealii::FEValues<1> fe_values(this->get_fe(),
      dealii::FEValues<1> fe_values(this->get_fe(), quadrature_formula,
                                    dealii::UpdateFlags(update_q_points));

      const unsigned int n_q_points = quadrature_formula.size();
      std::vector<unsigned int> global_dof_indices(
        this->get_fe().dofs_per_cell);

      //the usual loops. Go over all elements, in every element go through the quad
      //points and store thei position in times_
      //typename DoFHandler<1>::active_cell_iterator element =
      DoFHandler<1>::active_cell_iterator element = this->begin_active(), endc =
                                                      this->end();
      for (; element != endc; ++element)
        {
          fe_values.reinit(element);
          element->get_dof_indices(global_dof_indices);
          for (unsigned int i = 0; i < n_q_points; ++i)
            {
              times_[global_dof_indices[i]] =
                fe_values.get_quadrature_points()[i](0);
            }
        }
    }
    //member variables

    TimeIterator first_interval_, last_interval_, before_first_interval_,
                 after_last_interval_;

    //FIXME
    //Dummy FE which is used, if no fe is specified from the user.
    //At the moment, this is necessary, because we use an DoFCellAccessor
    //-iterator to go through our timegrid. Is this ok?
    const dealii::FiniteElement<1> *fe_;

    /**
     * A vector containing the place of the time Dofs.
     */
    std::vector<double> times_;
    unsigned int dofs_per_element_;
    bool need_delete_, initialized_;
  };

}      //end of namespace

#endif /* TIMEDOFHANDLER_H_ */
