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

#ifndef DOPE_FEVALUES_H_
#define DOPE_FEVALUES_H_

#include <deal.II/fe/fe_values.h>
#include <deal.II/hp/fe_values.h>

#include <wrapper/mapping_wrapper.h>

namespace DOpEWrapper
{
  /**
   * @class FEValues
   *
   * This class is a wrapper for the FEValues
   * Objects in dealii to allow instatiations with dim = 0
   *
   * @tparam <dim>     The dimension in which the FEValues object lives
   */
  template<int dim>
  class FEValues : public dealii::FEValues<dim>
  {
  public:
    FEValues(const DOpEWrapper::Mapping<dim, dealii::DoFHandler> &mapping,
             const dealii::FiniteElement<dim, dim> &fe,
             const dealii::Quadrature<dim> &quadrature,
             const dealii::UpdateFlags update_flags) :
      dealii::FEValues<dim>(mapping, fe, quadrature, update_flags)
    {
    }

//    FEValues(
//       const DOpEWrapper::Mapping<dim, dealii::MGDoFHandler > & mapping,
//       const dealii::FiniteElement<dim, dim> & fe,
//       const dealii::Quadrature<dim> & quadrature,
//       const dealii::UpdateFlags update_flags)
//      : dealii::FEValues<dim>(mapping, fe, quadrature, update_flags)
//        {
//        }

    FEValues(const dealii::FiniteElement<dim, dim> &fe,
             const dealii::Quadrature<dim> &quadrature,
             const dealii::UpdateFlags update_flags) :
      dealii::FEValues<dim>(fe, quadrature, update_flags)
    {
    }

    FEValues(const dealii::FEValues<dim> &fe_values) :
      dealii::FEValues<dim>(fe_values.get_mapping(), fe_values.get_fe(),
                            fe_values.get_quadrature(), fe_values.get_update_flags())
    {
    }
  };

  /*********************************************************/
  /**
   * @class FEFaceValues
   *
   * This class is a wrapper for the FEFaceValues
   * Objects in dealii to allow instatiations with dim = 0
   *
   * @tparam <dim>     The dimension in which the FEFaceValues object is posed
   *
   */
  template<int dim>
  class FEFaceValues : public dealii::FEFaceValues<dim>
  {
  public:
    FEFaceValues(
      const DOpEWrapper::Mapping<dim, dealii::DoFHandler> &mapping,
      const dealii::FiniteElement<dim, dim> &fe,
      const dealii::Quadrature<dim - 1> &quadrature,
      const dealii::UpdateFlags update_flags) :
      dealii::FEFaceValues<dim>(mapping, fe, quadrature, update_flags)
    {
    }

//    FEFaceValues(
//            const DOpEWrapper::Mapping<dim, dealii::MGDoFHandler > & mapping,
//            const dealii::FiniteElement<dim, dim> &fe,
//            const dealii::Quadrature<dim - 1> &quadrature,
//            const dealii::UpdateFlags update_flags)
//            : dealii::FEFaceValues<dim>(mapping, fe, quadrature, update_flags)
//        {
//        }

    FEFaceValues(const dealii::FiniteElement<dim, dim> &fe,
                 const dealii::Quadrature<dim - 1> &quadrature,
                 const dealii::UpdateFlags update_flags) :
      dealii::FEFaceValues<dim>(fe, quadrature, update_flags)
    {
    }

    FEFaceValues(const dealii::FEFaceValues<dim> &ffe_values) :
      dealii::FEFaceValues<dim>(ffe_values.get_mapping(),
                                ffe_values.get_fe(), ffe_values.get_quadrature(),
                                ffe_values.get_update_flags())
    {
    }

  };

  /*********************************************************/
  /**
   * @class FESubfaceValues
   *
   * This class is a wrapper for the FESubfaceValues and HPSubfaceValues
   * Objects in dealii to allow instatiations with dim = 0
   *
   * @tparam <dim>     The dimension in which the FESubfaceValues object is posed
   *
   */
  template<int dim>
  class FESubfaceValues : public dealii::FESubfaceValues<dim>
  {
  public:

    FESubfaceValues(
      const DOpEWrapper::Mapping<dim, dealii::DoFHandler> &mapping,
      const dealii::FiniteElement<dim, dim> &fe,
      const dealii::Quadrature<dim - 1> &quadrature,
      const dealii::UpdateFlags update_flags) :
      dealii::FESubfaceValues<dim>(mapping, fe, quadrature, update_flags)
    {
    }

//        FESubfaceValues(
//            const DOpEWrapper::Mapping<dim, dealii::MGDoFHandler > & mapping,
//            const dealii::FiniteElement<dim, dim> &fe,
//            const dealii::Quadrature<dim - 1> &quadrature,
//            const dealii::UpdateFlags update_flags)
//            : dealii::FESubfaceValues<dim>(mapping, fe, quadrature,
//                update_flags)
//        {
//        }

    FESubfaceValues(const dealii::FiniteElement<dim, dim> &fe,
                    const dealii::Quadrature<dim - 1> &quadrature,
                    const dealii::UpdateFlags update_flags) :
      dealii::FESubfaceValues<dim>(fe, quadrature, update_flags)
    {
    }

    FESubfaceValues(const dealii::FESubfaceValues<dim> &ffe_values) :
      dealii::FESubfaceValues<dim>(ffe_values.get_mapping(),
                                   ffe_values.get_fe(), ffe_values.get_quadrature(),
                                   ffe_values.get_update_flags())
    {
    }

  };

  /*********************************************************/
  /**
   * @class HpFEValues
   *
   * This class is a wrapper for the FEValues and HpFEValues
   * Objects in dealii to allow instatiations with dim = 0
   *
   * @tparam <dim>     The dimension in which the HpFEValues object lives
   */

  template<int dim>
  class HpFEValues : public dealii::hp::FEValues<dim>
  {
  public:
    HpFEValues(
      const DOpEWrapper::Mapping<dim, dealii::hp::DoFHandler> &mapping_collection,
      const dealii::hp::FECollection<dim, dim> &fe_collection,
      const dealii::hp::QCollection<dim> &q_collection,
      const dealii::UpdateFlags update_flags) :
      dealii::hp::FEValues<dim>(mapping_collection, fe_collection,
                                q_collection, update_flags)
    {
    }

    HpFEValues(const dealii::hp::FECollection<dim, dim> &fe_collection,
               const dealii::hp::QCollection<dim> &q_collection,
               const dealii::UpdateFlags update_flags) :
      dealii::hp::FEValues<dim>(fe_collection, q_collection, update_flags)
    {
    }
  };

  /*********************************************************/
  /**
   * @class HpFEFaceValues
   *
   * This class is a wrapper for the HPFaceFEValues
   * Objects in dealii to allow instatiations with dim = 0
   *
   * @tparam <dim>     The dimension in which the HpFEFaceValues object is posed
   *
   */
  template<int dim>
  class HpFEFaceValues : public dealii::hp::FEFaceValues<dim>
  {
  public:
    HpFEFaceValues(
      const DOpEWrapper::Mapping<dim, dealii::hp::DoFHandler> &mapping_collection,
      const dealii::hp::FECollection<dim, dim> &fe_collection,
      const dealii::hp::QCollection<dim - 1> & q_collection,
      const dealii::UpdateFlags update_flags) :
      dealii::hp::FEFaceValues<dim>(mapping_collection, fe_collection,
                                    q_collection, update_flags)
    {
    }

    HpFEFaceValues(const dealii::hp::FECollection<dim, dim> &fe_collection,
                   const dealii::hp::QCollection<dim - 1> & q_collection,
                   const dealii::UpdateFlags update_flags) :
      dealii::hp::FEFaceValues<dim>(fe_collection, q_collection,
                                    update_flags)
    {
    }

  };

  template<>
  class HpFEFaceValues<0>
  {
  public:

  };

  /*********************************************************/
  /**
    * @class HpFESubfaceValues
    *
    * This class is a wrapper for the HpFESubfaceValues
    * Objects in dealii to allow instatiations with dim = 0
    *
    * @tparam <dim>     The dimension in which the HpFESubfaceValues object is posed
    *
    */
  template<int dim>
  class HpFESubfaceValues : public dealii::hp::FESubfaceValues<dim>
  {
  public:

    HpFESubfaceValues(
      const DOpEWrapper::Mapping<dim, dealii::hp::DoFHandler> &mapping_collection,
      const dealii::hp::FECollection<dim, dim> &fe_collection,
      const dealii::hp::QCollection<dim - 1> & q_collection,
      const dealii::UpdateFlags update_flags) :
      dealii::hp::FESubfaceValues<dim>(mapping_collection, fe_collection,
                                       q_collection, update_flags)
    {
    }
    HpFESubfaceValues(
      const dealii::hp::FECollection<dim, dim> &fe_collection,
      const dealii::hp::QCollection<dim - 1> & q_collection,
      const dealii::UpdateFlags update_flags) :
      dealii::hp::FESubfaceValues<dim>(fe_collection, q_collection,
                                       update_flags)
    {
    }

  };
}

#endif
