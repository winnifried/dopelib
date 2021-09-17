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

#ifndef DOPE_DOFHANDLER_H_
#define DOPE_DOFHANDLER_H_

#include <deal.II/dofs/dof_handler.h>
#if DEAL_II_VERSION_GTE(9,3,0)
#else
#include <deal.II/hp/dof_handler.h>
#endif
#include <deal.II/fe/fe_system.h>

namespace DOpEWrapper
{
  #if DEAL_II_VERSION_GTE(9,3,0)
  /**
   * @class DoFHandler
   *
   * Wrapper for the DoFHandler. This Wrapper is required to allow instantiations
   * of DoFHandlers in dimension 0 as well as between ``normal'' and ``hp''
   * DoFHandlers.
   *
   * @template dim              Dimension of the dofhandler.
   */
  template<int dim>
  class DoFHandler : public dealii::DoFHandler<dim, dim>
  {
  public:
    DoFHandler(const dealii::Triangulation<dim, dim> &tria) :
      dealii::DoFHandler<dim, dim>(tria)
    {
    }

    /**
     * This function is needed to get access to the base class, i.e.,
     * the dealii DoFHandler which is wrapped.
     *
     * This is needed to avoid casts in the program when
     * some functions need a dealii DoFHandler but have the
     * DoFHandler as a template which is deduced by the
     * arguments passed to the function
     */
    const dealii::DoFHandler<dim, dim> &
    GetDEALDoFHandler() const
    {
      return *this;
    }

  };

  /**
   * Template specializations for dim=0.
   */
  template<>
  class DoFHandler<0>
  {
  private:
    unsigned int dofs_ = 0;
    dealii::Triangulation<1> tmp_tria_;
    dealii::DoFHandler<1> tmp_dof_handler_;

  public:
    /**
     * We actually never need the triangulation, this constructor merely exists
     * to allow for dimension independent programming.
     */
    template<int dim>
      DoFHandler(const dealii::Triangulation<dim, dim> &/*tria*/, const bool /*hp_capability_enabled*/=false)
      : tmp_dof_handler_ (tmp_tria_)
    {
    }
    template<int dim>
    void
    distribute_dofs(const dealii::FESystem<dim> &fe,
                    const unsigned int /*offset*/=0)
    {
      dofs_ = fe.element_multiplicity(0);
    }
    template<int dim>
    void
    distribute_dofs(const dealii::hp::FECollection<dim> &fe,
                    const unsigned int /*offset*/=0)
    {
      dofs_ = fe[0].element_multiplicity(0);
    }
    void
    clear()
    {
    }
    unsigned int
    n_dofs() const
    {
      return dofs_;
    }

    // Quick-fix for dim = 0, just return some DoFHandler.
    const dealii::DoFHandler<1> &
    GetDEALDoFHandler () const
    {
      assert(false);
      return tmp_dof_handler_;
    }
  };


#else//Dealii older than 9.3.0
    /**
   * @class DoFHandler
   *
   * Wrapper for the DoFHandler. This Wrapper is required to allow instantiations
   * of DoFHandlers in dimension 0 as well as between ``normal'' and ``hp''
   * DoFHandlers.
   *
   * @template dim              Dimension of the dofhandler.
   * @template DOFHANDLER       With this template argument we distinguish
   *                            between the 'normal' as well as the hp case.
   *                            The class DOFHANDLER is for dim>0 the base class
   *                            of DoFHandler. Feasible at the moment are
   *                            dealii::DoFHandler<dim> and dealii::hp::DoFHandler.
   *                            It has the default value dealii::DoFHandler<dim>
   */
  template<int dim,
           template<int DIM, int spacedim> class DOFHANDLER = dealii::DoFHandler>
  class DoFHandler : public DOFHANDLER<dim, dim>
  {
  public:
  DoFHandler(const dealii::Triangulation<dim, dim> &tria) :
      DOFHANDLER<dim, dim>(tria)
    {
    }

    /**
     * This function is needed to get access to the base class, i.e.,
     * the dealii DoFHandler which is wrapped.
     *
     * This is needed to avoid casts in the program when
     * some functions need a dealii DoFHandler but have the
     * DoFHandler as a template which is deduced by the
     * arguments passed to the function
     */
    const DOFHANDLER<dim, dim> &
    GetDEALDoFHandler() const
    {
      return *this;
    }

  };

  //Template specialization DOFHANDLER = dealii::DoFHandler<dim>
  template<int dim>
  class DoFHandler<dim, dealii::DoFHandler> : public dealii::DoFHandler<dim>
  {
  public:
  DoFHandler(const dealii::Triangulation<dim, dim> &tria) :
      dealii::DoFHandler<dim>(tria)
    {
    }
    const dealii::DoFHandler<dim> &
    GetDEALDoFHandler() const
    {
      return *this;
    }

  };

  //Template specialization DOFHANDLER = dealii::hp::DoFHandler<dim>
  template<int dim>
  class DoFHandler<dim, dealii::hp::DoFHandler> : public dealii::hp::DoFHandler<
    dim>
  {
  public:
    DoFHandler(const dealii::Triangulation<dim, dim> &tria) :
      dealii::hp::DoFHandler<dim>(tria)
    {
    }
    const dealii::hp::DoFHandler<dim> &
    GetDEALDoFHandler() const
    {
      return *this;
    }
  };

  /**
   * Template specializations for dim=0.
   */
  template<>
  class DoFHandler<0, dealii::DoFHandler>
  {
  private:
    unsigned int dofs_ = 0;
    dealii::Triangulation<1> tmp_tria_;
    dealii::DoFHandler<1> tmp_dof_handler_;

  public:
    /**
     * We actually never need the triangulation, this constructur merely exists
     * to allow for dimension independent programming.
     */
    template<int dim>
    DoFHandler(const dealii::Triangulation<dim, dim> &/*tria*/)
      : tmp_dof_handler_ (tmp_tria_)
    {
    }
    template<int dim>
    void
    distribute_dofs(const dealii::FESystem<dim> &fe,
                    const unsigned int /*offset*/=0)
    {
      dofs_ = fe.element_multiplicity(0);
    }
    void
    clear()
    {
    }
    unsigned int
    n_dofs() const
    {
      return dofs_;
    }

    // Quick-fix for dim = 0, just return some DoFHandler.
    const dealii::DoFHandler<1> &
    GetDEALDoFHandler () const
    {
      assert(false);
      return tmp_dof_handler_;
    }
  };

  template<>
  class DoFHandler<0, dealii::hp::DoFHandler>
  {
  private:
    unsigned int dofs_ = 0;
    dealii::Triangulation<1> tmp_tria_;
    dealii::hp::DoFHandler<1> tmp_dof_handler_;

  public:
    /**
     * We actually never need the triangulation, this constructur merely exists
     * to allow for dimension independent programming.
     */
    template<int dim>
    DoFHandler(const dealii::Triangulation<dim, dim> &/*tria*/)
      : tmp_dof_handler_ (tmp_tria_)
    {
    }
    template<int dim>
    void
    distribute_dofs(const dealii::hp::FECollection<dim> &fe_collection,
                    const unsigned int /*offset*/ = 0)
    {
      dofs_ = fe_collection[0].element_multiplicity(0);
    }
    void
    clear()
    {
    }
    unsigned int
    n_dofs() const
    {
      return dofs_;
    }
    // Quick-fix for dim = 0, just return some DoFHandler.
    const dealii::hp::DoFHandler<1> &
    GetDEALDoFHandler () const
    {
      assert(false);
      return tmp_dof_handler_;
    }
  };

#endif//Endof dealii older than 9.2.0
}

#endif
