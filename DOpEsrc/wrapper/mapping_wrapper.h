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
  template<int dim, template<int, int> class DH = dealii::DoFHandler>
    class Mapping
    {
      private:
        Mapping()
        {
        }

    };

  /************************************************************************************/

  template<int dim>
    class Mapping<dim, dealii::DoFHandler > : public dealii::MappingQ<dim>
    {
      public:
        Mapping(const unsigned int p, const bool use_mapping_q_on_all_cells =
            false) :
            dealii::MappingQ<dim>(p, use_mapping_q_on_all_cells)
        {
        }

        Mapping(const dealii::MappingQ<dim> &mapping) :
            dealii::MappingQ<dim>(mapping)
        {
        }

        /**
         * This function is needed for a workaround
         * linked to the hp-version (i.e. deal.ii is not
         * consistent at the current stage using Mappings
         * or MappingCollections in the hp-framework).
         */
        const typename dealii::MappingQ<dim> &
        operator[](const unsigned int index) const
        {
          assert(index == 0);
          return *this;
        }

    };

  /************************************************************************************/

  /**
   * WARNING: At the current stage, it is note recommended to use MappingCollections
   * with more than one mapping, as deal.ii is not consinstent in using
   * Collections!
   */
  template<int dim>
    class Mapping<dim, dealii::hp::DoFHandler> : public dealii::hp::MappingCollection<
        dim>
    {
      public:
        Mapping() :
            dealii::hp::MappingCollection<dim>()
        {
        }

        Mapping(const dealii::Mapping<dim>& mapping) :
            dealii::hp::MappingCollection<dim>(mapping)
        {
        }
        Mapping(const dealii::hp::MappingCollection<dim> & mapping_collection) :
            dealii::hp::MappingCollection<dim>(mapping_collection)
        {
        }
    };

  /************************************************************************************/

  template<int dim, template<int, int> class DH>
    struct StaticMappingQ1
    {
    };

  template<int dim>
    struct StaticMappingQ1<dim, dealii::DoFHandler>
    {
      public:
        static Mapping<dim, dealii::DoFHandler> mapping_q1;
    };

  template<int dim>
    struct StaticMappingQ1<dim, dealii::hp::DoFHandler>
    {
      public:
        static Mapping<dim, dealii::hp::DoFHandler> mapping_q1;
    };
}

#endif /* MAPPING_WRAPPER_H_ */
