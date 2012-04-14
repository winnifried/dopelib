#ifndef _DOPE_FEVALUES_H_
#define _DOPE_FEVALUES_H_

#include <fe/fe_values.h>
#include <hp/fe_values.h>

namespace DOpEWrapper
{

  template<int dim>
    class FEValues : public dealii::FEValues<dim>
    {
      public:
        FEValues(const dealii::Mapping<dim, dim> & mapping,
            const dealii::FiniteElement<dim, dim> & fe,
            const dealii::Quadrature<dim> & quadrature,
            const dealii::UpdateFlags update_flags)
            : dealii::FEValues<dim>(mapping, fe, quadrature, update_flags)
        {
        }

        FEValues(const dealii::FiniteElement<dim, dim> & fe,
            const dealii::Quadrature<dim> & quadrature,
            const dealii::UpdateFlags update_flags)
            : dealii::FEValues<dim>(fe, quadrature, update_flags)
        {
        }

        FEValues(const dealii::FEValues<dim>& fe_values)
            : dealii::FEValues<dim>(fe_values.get_mapping(), fe_values.get_fe(),
                fe_values.get_quadrature(), fe_values.get_update_flags())
        {
        }
    };

  /*********************************************************/
  template<int dim>
    class FEFaceValues : public dealii::FEFaceValues<dim>
    {
      public:
        FEFaceValues(const dealii::Mapping<dim, dim> &mapping,
            const dealii::FiniteElement<dim, dim> &fe,
            const dealii::Quadrature<dim - 1> &quadrature,
            const dealii::UpdateFlags update_flags)
            : dealii::FEFaceValues<dim>(mapping, fe, quadrature, update_flags)
        {
        }

        FEFaceValues(const dealii::FiniteElement<dim, dim> &fe,
            const dealii::Quadrature<dim - 1> &quadrature,
            const dealii::UpdateFlags update_flags)
            : dealii::FEFaceValues<dim>(fe, quadrature, update_flags)
        {
        }

        FEFaceValues(const dealii::FEFaceValues<dim>& ffe_values)
            : dealii::FEFaceValues<dim>(ffe_values.get_mapping(),
                ffe_values.get_fe(), ffe_values.get_quadrature(),
                ffe_values.get_update_flags())
        {
        }

    };

  /*********************************************************/
  template<int dim>
    class FESubfaceValues : public dealii::FESubfaceValues<dim>
    {
      public:
        FESubfaceValues(const dealii::Mapping<dim, dim> &mapping,
            const dealii::FiniteElement<dim, dim> &fe,
            const dealii::Quadrature<dim - 1> &quadrature,
            const dealii::UpdateFlags update_flags)
            : dealii::FESubfaceValues<dim>(mapping, fe, quadrature,
                update_flags)
        {
        }

        FESubfaceValues(const dealii::FiniteElement<dim, dim> &fe,
            const dealii::Quadrature<dim - 1> &quadrature,
            const dealii::UpdateFlags update_flags)
            : dealii::FESubfaceValues<dim>(fe, quadrature, update_flags)
        {
        }

        FESubfaceValues(const dealii::FESubfaceValues<dim>& ffe_values)
            : dealii::FESubfaceValues<dim>(ffe_values.get_mapping(),
                ffe_values.get_fe(), ffe_values.get_quadrature(),
                ffe_values.get_update_flags())
        {
        }

    };

  /*********************************************************/

  template<int dim>
    class HpFEValues : public dealii::hp::FEValues<dim>
    {
      public:
        HpFEValues(
            const dealii::hp::MappingCollection<dim, dim> & mapping_collection,
            const dealii::hp::FECollection<dim, dim> & fe_collection,
            const dealii::hp::QCollection<dim> & q_collection,
            const dealii::UpdateFlags update_flags)
            : dealii::hp::FEValues<dim>(mapping_collection, fe_collection,
                q_collection, update_flags)
        {
        }

        HpFEValues(const dealii::hp::FECollection<dim, dim> & fe_collection,
            const dealii::hp::QCollection<dim> & q_collection,
            const dealii::UpdateFlags update_flags)
            : dealii::hp::FEValues<dim>(fe_collection, q_collection,
                update_flags)
        {
        }
    };

  /*********************************************************/
  template<int dim>
    class HpFEFaceValues : public dealii::hp::FEFaceValues<dim>
    {
      public:
        HpFEFaceValues(
            const dealii::hp::MappingCollection<dim, dim> & mapping_collection,
            const dealii::hp::FECollection<dim, dim> & fe_collection,
            const dealii::hp::QCollection<dim - 1> & q_collection,
            const dealii::UpdateFlags update_flags)
            : dealii::hp::FEFaceValues<dim>(mapping_collection, fe_collection,
                q_collection, update_flags)
        {
        }

        HpFEFaceValues(const dealii::hp::FECollection<dim, dim> & fe_collection,
            const dealii::hp::QCollection<dim - 1> & q_collection,
            const dealii::UpdateFlags update_flags)
            : dealii::hp::FEFaceValues<dim>(fe_collection, q_collection,
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
  template<int dim>
    class HpFESubfaceValues : public dealii::hp::FESubfaceValues<dim>
    {
      public:

        HpFESubfaceValues(
            const dealii::hp::MappingCollection<dim, dim> & mapping_collection,
            const dealii::hp::FECollection<dim, dim> & fe_collection,
            const dealii::hp::QCollection<dim - 1> & q_collection,
            const dealii::UpdateFlags update_flags)
            : dealii::hp::FESubfaceValues<dim>(mapping_collection,
                fe_collection, q_collection, update_flags)
        {
        }
        HpFESubfaceValues(
            const dealii::hp::FECollection<dim, dim> & fe_collection,
            const dealii::hp::QCollection<dim - 1> & q_collection,
            const dealii::UpdateFlags update_flags)
            : dealii::hp::FESubfaceValues<dim>(fe_collection, q_collection,
                update_flags)
        {
        }

    };
}

#endif
