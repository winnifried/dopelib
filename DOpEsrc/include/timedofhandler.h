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

#ifndef _TIMEDOFHANDLER_H_
#define _TIMEDOFHANDLER_H_

//DOpE
#include <timeiterator.h>
#include <dopeexception.h>
#include <versionscheck.h>

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
        _fe = NULL;
        _times.resize(1, 0.);
        _dofs_per_cell = 1;
        _need_delete = false;
        _initialized = false;
      }

      /**
       * TODO for later uses, if one wants to use galerkin methods.
       * On top of that, we should get the fe from the timestepping problem.
       */
      TimeDoFHandler(const Triangulation<1> & tria,
          const dealii::FiniteElement<1>& fe)
          : dealii::DoFHandler<1>(tria)
      {
        _fe = &fe;
        this->distribute_dofs();
        _need_delete = false;
      }

      TimeDoFHandler(const Triangulation<1>& tria)
          : dealii::DoFHandler<1>(tria)
      {
        _fe = new dealii::FE_Q<1>(1);
        this->distribute_dofs();
        _need_delete = true;
      }

      //Destructor
      ~TimeDoFHandler()
      {
        this->clear();

        if (_need_delete)
        {
          delete _fe;
        }
        _fe = NULL;
      }

      /**
       * Go through the triangulation and distribute the degrees of freedoms
       * needed for the given finite element.
       */
      void
      distribute_dofs()
      {
        if (_fe != NULL)
        {
          dealii::DoFHandler<1>::distribute_dofs(*_fe);
          //make sure that the dofs are numbered 'downstream' (referring to the time variable!)

  #if DEAL_II_VERSION_GTE(7,3)
           dealii::DoFRenumbering::downstream<dealii::DoFHandler<1> >(*this,
               dealii::Point<1>(1.), true);
  #else
           dealii::DoFRenumbering::downstream<dealii::DoFHandler<1>, 1>(*this,
               dealii::Point<1>(1.), true);
  #endif
          find_ends();
          compute_times();
          _initialized = true;
          _dofs_per_cell = this->get_fe().dofs_per_cell;
        }
      }

      /**
       * Returns the number of intervals
       */
      unsigned int
      GetNbrOfIntervals() const
      {
        if (_initialized)
          return this->get_tria().n_active_cells();
        else
          return 0;
      }

      /**
       * Returns the number of dofs.
       */
      unsigned int
      GetNbrOfDoFs() const
      {
        if (_initialized)
          return this->n_dofs();
        else
          return 1;
      }

      /**
       * Returns a vector containing the position of the dofs.
       */
      const std::vector<double>&
      GetTimes() const
      {
        return _times;
      }

      /**
       * Given an interval, this function writes the global
       *  position of the  dofs in the interval into the
       *  vector local_times. It is assumed that local_times has
       *  the right length beforehand!
       */
      void
      GetTimes(const TimeIterator & interval,
          std::vector<double>& local_times) const
      {
        assert(local_times.size() == GetLocalNbrOfDoFs());

        std::vector<unsigned int> global_dof_indices(GetLocalNbrOfDoFs());
        interval.get_time_dof_indices(global_dof_indices);
        for (unsigned int i = 0; i < GetLocalNbrOfDoFs(); ++i)
        {
          local_times[i] = _times[global_dof_indices[i]];
        }
      }

      /**
       * Returns the position of the given timedof.
       */
      double
      GetTime(unsigned int timestep)
      {
        assert(timestep < _times.size());
        return _times[timestep];
      }

      /**
       * Returns an TimeIterator pointing to the first time_interval.
       */
      TimeIterator
      first_interval() const
      {
        assert(_initialized);
        return _first_interval;
      }

      /**
       * Returns an TimeIterator pointing to the element before(!)
       *  the  first time_interval.
       */
      TimeIterator
      before_first_interval() const
      {
        assert(_initialized);
        return _before_first_interval;
      }

      /**
       * Returns an active_cell_iterator pointing to the last time_interval.
       */
      TimeIterator
      last_interval() const
      {
        assert(_initialized);
        return _last_interval;
      }

      /**
       * Returns an active_cell_iterator pointing to the element after(!)
       * the last time_interval.
       */
      TimeIterator
      after_last_interval() const
      {
        assert(_initialized);
        return _after_last_interval;
      }

      /**
       * Returns the number of dofs per interval.
       */
      unsigned int
      GetLocalNbrOfDoFs() const
      {
        return _dofs_per_cell;
      }

    private:
      /**
       * Find the first and last interval and store them
       * in _first_interval as well as _last_interval
       */
      void
      find_ends()
      {
        DoFHandler<1>::active_cell_iterator cell = this->begin_active();
        while (cell->face(0)->boundary_indicator() != 0)
        {
          cell = cell->neighbor(0);
        }
        _first_interval.Initialize(cell, 0);
        _before_first_interval.Initialize(cell, -2);

        cell = this->begin_active();
        while (cell->face(1)->boundary_indicator() != 1)
        {
          cell = cell->neighbor(1);
        }
        assert(static_cast<int>(this->get_tria().n_active_cells() - 1) >= 0);
        _last_interval.Initialize(cell,
            static_cast<int>(this->get_tria().n_active_cells() - 1));_after_last_interval
        .Initialize(cell, -1);
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
        _times.resize(this->n_dofs());

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

        //the usual loops. Go over all cells, in every cell go through the quad
        //points and store thei position in _times
        //typename DoFHandler<1>::active_cell_iterator cell =
        DoFHandler<1>::active_cell_iterator cell = this->begin_active(), endc =
            this->end();
        for (; cell != endc; ++cell)
        {
          fe_values.reinit(cell);
          cell->get_dof_indices(global_dof_indices);
          for (unsigned int i = 0; i < n_q_points; ++i)
          {
            _times[global_dof_indices[i]] =
                fe_values.get_quadrature_points()[i](0);
          }
        }
      }
      //member variables

      TimeIterator _first_interval, _last_interval, _before_first_interval,
          _after_last_interval;

      //FIXME
      //Dummy FE which is used, if no fe is specified from the user.
      //At the moment, this is necessary, because we use an DoFCellAccessor
      //-iterator to go through our timegrid. Is this ok?
      const dealii::FiniteElement<1>* _fe;

      /**
       * A vector containing the place of the time Dofs.
       */
      std::vector<double> _times;
      unsigned int _dofs_per_cell;
      bool _need_delete, _initialized;
  };

}      //end of namespace

#endif /* TIMEDOFHANDLER_H_ */
