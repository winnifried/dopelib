/*
 * mapping_wrapper.h
 *
 *  Created on: Oct 4, 2012
 *      Author: cgoll
 */

#ifndef MAPPING_WRAPPER_H_
#define MAPPING_WRAPPER_H_

#include <dofs/dof_handler.h>
#include <hp/dof_handler.h>
#include <hp/mapping_collection.h>
#include <fe/mapping_q.h>

namespace DOpEWrapper
{
  template<int dim, typename DOFHANDLER = dealii::DoFHandler<dim> >
    class Mapping
    {
      private:
        Mapping()
        {
        }

    };

  /************************************************************************************/

  template<int dim>
    class Mapping<dim, dealii::DoFHandler<dim> > : public dealii::MappingQ<dim>
    {
      public:
        Mapping(const unsigned int p, const bool use_mapping_q_on_all_cells =
            false)
            : dealii::MappingQ<dim>(p, use_mapping_q_on_all_cells)
        {
        }

        Mapping(const dealii::MappingQ<dim> &mapping)
            : dealii::MappingQ<dim>(mapping)
        {
        }

    };

  /************************************************************************************/

  template<int dim>
    class Mapping<dim, dealii::hp::DoFHandler<dim> > : public dealii::hp::MappingCollection<
        dim>
    {
      public:
        Mapping()
            : dealii::hp::MappingCollection<dim>()
        {
        }

        Mapping(const dealii::Mapping<dim>& mapping)
            : dealii::hp::MappingCollection<dim>(mapping)
        {
        }
        Mapping(const dealii::hp::MappingCollection<dim> & mapping_collection)
            : dealii::hp::MappingCollection<dim>(mapping_collection)
        {
        }
    };

  /************************************************************************************/

  template<int dim, typename DOFHANDLER>
    struct StaticMappingQ1
    {
    };

  template<int dim>
    struct StaticMappingQ1<dim, dealii::DoFHandler<dim> >
    {
      public:
        static Mapping<dim, dealii::DoFHandler<dim> > mapping_q1;
    };

  template<int dim>
    struct StaticMappingQ1<dim, dealii::hp::DoFHandler<dim> >
    {
      public:
        static Mapping<dim, dealii::hp::DoFHandler<dim> > mapping_q1;
    };
}

#endif /* MAPPING_WRAPPER_H_ */
