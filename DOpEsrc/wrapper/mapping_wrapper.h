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

#ifndef MAPPING_WRAPPER_H_
#define MAPPING_WRAPPER_H_

#include <deal.II/dofs/dof_handler.h>
//#include <deal.II/multigrid/mg_dof_handler.h>
#include <deal.II/hp/dof_handler.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/fe/mapping_q.h>

namespace DOpEWrapper
{
  /**
   * @class Mapping
   *
   * A Wrapper that is used to automatically use the
   * dealii::MappingCollection for hp::DoFHandlers and
   * dealii::Mapping for all other DoFHandler objects
   * and simultaneously beeing of the same type to
   * allow us the use of DoFHandler as a template.
   *
   * @template dim              Dimension of the dofhandler.
   * @template DOFHANDLER       The dealii DoFHandler Object
   */
  template<int dim, template<int, int> class DH = dealii::DoFHandler>
  class Mapping
  {
  private:
    Mapping()
    {
    }

    ~Mapping()
    {
    }

  };

  /************************************************************************************/

  template<int dim>
  class Mapping<dim, dealii::DoFHandler > : public dealii::MappingQ<dim>
  {
  public:
    Mapping(const unsigned int p, const bool use_mapping_q_on_all_elements =
              false) :
      dealii::MappingQ<dim>(p, use_mapping_q_on_all_elements)
    {
    }

    Mapping(const dealii::MappingQ<dim> &mapping) :
      dealii::MappingQ<dim>(mapping)
    {
    }

    ~Mapping()
    {
    }

    /**
     * This function is needed for a workaround
     * linked to the hp-version (i.e. deal.ii is not
     * consistent at the current stage using Mappings
     * or MappingCollections in the hp-framework).
     */
    const typename dealii::MappingQ<dim> &
    operator[](const unsigned int /*index*/) const
    {
      //assert(index == 0);
      return *this;
    }

  };

//  /************************************************************************************/
//
//  template<int dim>
//    class Mapping<dim, dealii::MGDoFHandler > : public dealii::MappingQ<dim>
//    {
//      public:
//        Mapping(const unsigned int p, const bool use_mapping_q_on_all_elements =
//            false)
//            : dealii::MappingQ<dim>(p, use_mapping_q_on_all_elements)
//        {
//        }
//
//        Mapping(const dealii::MappingQ<dim> &mapping)
//            : dealii::MappingQ<dim>(mapping)
//        {
//        }
//
//      ~Mapping()
//  {
//  }
//
//        /**
//         * This function is needed for a workaround
//         * linked to the hp-version (i.e. deal.ii is not
//         * consistent at the current stage using Mappings
//         * or MappingCollections in the hp-framework).
//         */
//        const typename dealii::MappingQ<dim> &
//        operator[](const unsigned int index) const
//        {
//          assert(index == 0);
//          return *this;
//        }
//
//    };

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

    ~Mapping()
    {
    }

    Mapping(const dealii::Mapping<dim> &mapping)
      : dealii::hp::MappingCollection<dim>(mapping)
    {
    }
    Mapping(const dealii::hp::MappingCollection<dim> &mapping_collection) :
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

// template<int dim>
//    struct StaticMappingQ1<dim, dealii::MGDoFHandler >
//    {
//      public:
//        static Mapping<dim, dealii::MGDoFHandler > mapping_q1;
//    };

  template<int dim>
  struct StaticMappingQ1<dim, dealii::hp::DoFHandler>
  {
  public:
    static Mapping<dim, dealii::hp::DoFHandler> mapping_q1;
  };
}

#endif /* MAPPING_WRAPPER_H_ */
