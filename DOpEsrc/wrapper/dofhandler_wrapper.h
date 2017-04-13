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

#ifndef DOPE_DOFHANDLER_H_
#define DOPE_DOFHANDLER_H_

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/hp/dof_handler.h>
#include <deal.II/fe/fe_system.h>

namespace DOpEWrapper
{

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

    /**
     * Does the DoFHandler need an IndexSetter, i.e. is this
     * an hp dofhandler?
     */
    static bool
    NeedIndexSetter();

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
    static bool
    NeedIndexSetter()
    {
      return false;
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
    static bool
    NeedIndexSetter()
    {
      return true;
    }
    const dealii::hp::DoFHandler<dim> &
    GetDEALDoFHandler() const
    {
      return *this;
    }
  };

// //Template specialization DOFHANDLER = dealii::MGDoFHandler<dim>
//  template<int dim>
//    class DoFHandler<dim, dealii::MGDoFHandler > : public dealii::MGDoFHandler<
//        dim>
//    {
//      public:
//        DoFHandler(const dealii::Triangulation<dim, dim> &tria)
//            : dealii::MGDoFHandler<dim>(tria)
//        {
//        }
//        static bool
//        NeedIndexSetter()
//        {
//          return true;
//        }
//        const dealii::MGDoFHandler<dim>&
//        GetDEALDoFHandler() const
//        {
//          return *this;
//        }
//    };

  /**
   * Template specializations for dim=0.
   */
  template<>
  class DoFHandler<0, dealii::DoFHandler>
  {
  private:
    unsigned int dofs_;

  public:
    /**
     * We actually never need the triangulation, this constructur merely exists
     * to allow for dimension independent programming.
     */
    template<int dim>
    DoFHandler(const dealii::Triangulation<dim, dim> &/*tria*/)
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
    static bool
    NeedIndexSetter()
    {
      return false;
    }
  };

  template<>
  class DoFHandler<0, dealii::hp::DoFHandler>
  {
  private:
    unsigned int dofs_;

  public:
    /**
     * We actually never need the triangulation, this constructur merely exists
     * to allow for dimension independent programming.
     */
    template<int dim>
    DoFHandler(const dealii::Triangulation<dim, dim> &/*tria*/)
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
    static bool
    NeedIndexSetter()
    {
      return false;
    }
  };
}

#endif
