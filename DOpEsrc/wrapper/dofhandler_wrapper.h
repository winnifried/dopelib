#ifndef _DOPE_DOFHANDLER_H_
#define _DOPE_DOFHANDLER_H_

#include <dofs/dof_handler.h>
#include <hp/dof_handler.h>
#include <fe/fe_system.h>

namespace DOpEWrapper
{

  /**
   * Wrapper for the DoFHandler.
   *
   * @template dim              Dimension of the dofhandler.
   * @template DOFHANDLER       With this template argument we distinguish
   *                            between the 'normal' as well as the hp case.
   *                            The class DOFHANDLER is for dim>0 the base class
   *                            odf DoFHandler.Feasible at the moment are
   *                            dealii::DoFHandler<dim> and dealii::hp::DoFHandler.
   *                            It has the default vale dealii::DoFHandler<dim>
   */
  template<int dim, typename DOFHANDLER = dealii::DoFHandler<dim> >
    class DoFHandler : public DOFHANDLER
    {
      public:
        DoFHandler(const dealii::Triangulation<dim, dim> &tria)
            : DOFHANDLER(tria)
        {
        }

        const DOFHANDLER&
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
    class DoFHandler<dim, dealii::DoFHandler<dim> > : public dealii::DoFHandler<
        dim>
    {
      public:
        DoFHandler(const dealii::Triangulation<dim, dim> &tria)
            : dealii::DoFHandler<dim>(tria)
        {
        }
        static bool
        NeedIndexSetter()
        {
          return false;
        }
        const dealii::DoFHandler<dim>&
        GetDEALDoFHandler() const
        {
          return *this;
        }

    };

  //Template specialization DOFHANDLER = dealii::hp::DoFHandler<dim>
  template<int dim>
    class DoFHandler<dim, dealii::hp::DoFHandler<dim> > : public dealii::hp::DoFHandler<
        dim>
    {
      public:
        DoFHandler(const dealii::Triangulation<dim, dim> &tria)
            : dealii::hp::DoFHandler<dim>(tria)
        {
        }
        static bool
        NeedIndexSetter()
        {
          return true;
        }
        const dealii::hp::DoFHandler<dim>&
        GetDEALDoFHandler() const
        {
          return *this;
        }
    };

  /**
   * Template specializations for dim=0.
   */
  template<>
    class DoFHandler<0, dealii::DoFHandler<deal_II_dimension> >
    {
      private:
        unsigned int _dofs;

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
              const unsigned int offset __attribute__((unused)) =0)
          {
            _dofs = fe.element_multiplicity(0);
          }
        void
        clear()
        {
        }
        unsigned int
        n_dofs() const
        {
          return _dofs;
        }
        static bool
        NeedIndexSetter()
        {
          return false;
        }
    };

  template<>
    class DoFHandler<0, dealii::hp::DoFHandler<deal_II_dimension> >
    {
      private:
        unsigned int _dofs;

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
              const unsigned int offset __attribute__((unused)) =0)
          {
            _dofs = fe_collection[0].element_multiplicity(0);
          }
        void
        clear()
        {
        }
        unsigned int
        n_dofs() const
        {
          return _dofs;
        }
        static bool
        NeedIndexSetter()
        {
          return false;
        }
    };
}

#endif
