/*
 * facedatacontainer_internal.h
 *
 *  Created on: Apr 4, 2012
 *      Author: cgoll
 */

#ifndef FACEDATACONTAINER_INTERNAL_H_
#define FACEDATACONTAINER_INTERNAL_H_

#include <deal.II/lac/vector.h>

#include "fevalues_wrapper.h"
#include "dopeexception.h"

namespace DOpE
{
  namespace fdcinternal
  {
    /**
     * This class houses all the functionality which is shared between
     * the FaceDataContainer for normal and hp::DoFHandlers.
     *
     * @template VECTOR     Type of the vector we use in our computations
     *                      (i.e. Vector<double> or BlockVector<double>)
     * @template dim        The dimension of the integral we are actually
     *                      interested in.
     */
    template<typename VECTOR, int dim>
      class FaceDataContainerInternal
      {
        public:
          FaceDataContainerInternal(
              const std::map<std::string, const dealii::Vector<double>*> &param_values,
              const std::map<std::string, const VECTOR*> &domain_values,
              bool need_neighbour);

          virtual
          ~FaceDataContainerInternal()
          {
          }
          ;

          /**
           * Looks up the given name in _parameter_data and returns the
           * corresponding value through 'value'.
           */
          void
          GetParamValues(std::string name, dealii::Vector<double>& value) const;

          /**
           * Returns the domain values.
           */
          const std::map<std::string, const VECTOR*> &
          GetDomainValues() const
          {
            return _domain_values;
          }
          ;

          virtual const dealii::FEFaceValuesBase<dim>&
          GetFEFaceValuesState() const =0;
          virtual const dealii::FEFaceValuesBase<dim>&
          GetFEFaceValuesControl() const = 0;

          virtual const dealii::FEFaceValuesBase<dim>&
          GetNbrFEFaceValuesState() const = 0;
          virtual const dealii::FEFaceValuesBase<dim>&
          GetNbrFEFaceValuesControl() const = 0;

          /********************************************************************/
          /**
           * Functions to extract values and gradients out of the FEValues
           */

          /**
           * Writes the values of the state variable at the quadrature points into values.
           */
          void
          GetFaceValuesState(std::string name,
              std::vector<double>& values) const;

          /*********************************************/
          /*
           * Same as above for the Vector valued case.
           */
          void
          GetFaceValuesState(std::string name,
              std::vector<dealii::Vector<double> >& values) const;

          /*********************************************/
          /*
           * Writes the values of the control variable at the quadrature points into values
           */
          void
          GetFaceValuesControl(std::string name,
              std::vector<double>& values) const;

          /*********************************************/
          /*
           * Same as above for the Vector valued case.
           */
          void
          GetFaceValuesControl(std::string name,
              std::vector<dealii::Vector<double> >& values) const;
          /*********************************************/
          /*
           * Writes the values of the state gradient at the quadrature points into values.
           */

          template<int targetdim>
            void
            GetFaceGradsState(std::string name,
                std::vector<dealii::Tensor<1, targetdim> >& values) const;

          /*********************************************/
          /*
           * Same as above for the Vector valued case.
           */
          template<int targetdim>
            void
            GetFaceGradsState(std::string name,
                std::vector<std::vector<dealii::Tensor<1, targetdim> > >& values) const;

          /*********************************************/
          /*
           * Writes the values of the control gradient at the quadrature points into values.
           */
          template<int targetdim>
            void
            GetFaceGradsControl(std::string name,
                std::vector<dealii::Tensor<1, targetdim> >& values) const;

          /*********************************************/
          /*
           * Same as above for the Vector valued case.
           */
          template<int targetdim>
            void
            GetFaceGradsControl(std::string name,
                std::vector<std::vector<dealii::Tensor<1, targetdim> > >& values) const;

          /*
           * Writes the values of the state variable at the quadrature points into values.
           */
          inline void
          GetNbrFaceValuesState(std::string name,
              std::vector<double>& values) const;
          /*********************************************/
          /*
           * Same as above for the Vector valued case.
           */
          inline void
          GetNbrFaceValuesState(std::string name,
              std::vector<Vector<double> >& values) const;

          /*********************************************/

          /*
           * Writes the values of the control variable at the quadrature points into values
           */
          inline void
          GetNbrFaceValuesControl(std::string name,
              std::vector<double>& values) const;
          /*********************************************/

          /*
           * Same as above for the Vector valued case.
           */
          inline void
          GetNbrFaceValuesControl(std::string name,
              std::vector<Vector<double> >& values) const;
          /*********************************************/

          /*
           * Writes the values of the state gradient at the quadrature points into values.
           */

          template<int targetdim>
            inline void
            GetNbrFaceGradsState(std::string name,
                std::vector<dealii::Tensor<1, targetdim> >& values) const;

          /*********************************************/

          /*
           * Same as avoe for the Vector valued case.
           */
          template<int targetdim>
            inline void
            GetNbrFaceGradsState(std::string name,
                std::vector<std::vector<dealii::Tensor<1, targetdim> > >& values) const;

          /*********************************************/

          /*
           * Writes the values of the control gradient at the quadrature points into values.
           */

          template<int targetdim>
            inline void
            GetNbrFaceGradsControl(std::string name,
                std::vector<dealii::Tensor<1, targetdim> >& values) const;

          /*********************************************/
          /*
           * Same as above for the Vector valued case.
           */
          template<int targetdim>
            inline void
            GetNbrFaceGradsControl(std::string name,
                std::vector<std::vector<dealii::Tensor<1, targetdim> > >& values) const;

        protected:
          void
          SetFace(unsigned int face)
          {
            _face = face;
          }
          unsigned int
          GetFace() const
          {
            return _face;
          }
          void
          SetSubFace(unsigned int subface)
          {
            _subface = subface;
          }
          unsigned int
          GetSubFace() const
          {
            return _subface;
          }
          bool
          NeedNeighbour() const
          {
            return _need_neighbour;
          }

        private:
          /***********************************************************/
          /**
           * Helper Function. Vector valued case.
           */
          void
          GetValues(const dealii::FEFaceValuesBase<dim>& fe_values,
              std::string name, std::vector<double>& values) const;
          /***********************************************************/
          /**
           * Helper Function. Vector valued case.
           */
          void
          GetValues(const dealii::FEFaceValuesBase<dim>& fe_values,
              std::string name,
              std::vector<dealii::Vector<double> >& values) const;
          /***********************************************************/
          /**
           * Helper Function.
           */
          template<int targetdim>
            void
            GetGrads(const dealii::FEFaceValuesBase<dim>& fe_values,
                std::string name,
                std::vector<dealii::Tensor<1, targetdim> >& values) const;
          /***********************************************************/
          /**
           * Helper Function. Vector valued case.
           */
          template<int targetdim>
            void
            GetGrads(const dealii::FEFaceValuesBase<dim>& fe_values,
                std::string name,
                std::vector<std::vector<dealii::Tensor<1, targetdim> > >& values) const;

          const std::map<std::string, const dealii::Vector<double>*> &_param_values;
          const std::map<std::string, const VECTOR*> &_domain_values;

          unsigned int _face;
          unsigned int _subface;
          bool _need_neighbour;
      };

    /**********************************************************************/
    template<typename VECTOR, int dim>
      FaceDataContainerInternal<VECTOR, dim>::FaceDataContainerInternal(
          const std::map<std::string, const dealii::Vector<double>*> &param_values,
          const std::map<std::string, const VECTOR*> &domain_values,
          bool need_neighbour)
          : _param_values(param_values), _domain_values(domain_values), _need_neighbour(
              need_neighbour)
      {
      }

    template<typename VECTOR, int dim>
      void
      FaceDataContainerInternal<VECTOR, dim>::GetParamValues(std::string name,
          dealii::Vector<double>& value) const
      {
        typename std::map<std::string, const dealii::Vector<double>*>::const_iterator it =
            _param_values.find(name);
        if (it == _param_values.end())
        {
          throw DOpEException("Did not find " + name,
              "FaceDataContainerInternal::GetParamValues");
        }
        value = *(it->second);
      }

    /*********************************************/
    template<typename VECTOR, int dim>
      void
      FaceDataContainerInternal<VECTOR, dim>::GetFaceValuesState(
          std::string name, std::vector<double>& values) const
      {
        this->GetValues(this->GetFEFaceValuesState(), name, values);
      }
    /*********************************************/
    template<typename VECTOR, int dim>
      void
      FaceDataContainerInternal<VECTOR, dim>::GetFaceValuesState(
          std::string name, std::vector<dealii::Vector<double> >& values) const
      {
        this->GetValues(this->GetFEFaceValuesState(), name, values);

      }

    /*********************************************/
    template<typename VECTOR, int dim>
      void
      FaceDataContainerInternal<VECTOR, dim>::GetFaceValuesControl(
          std::string name, std::vector<double>& values) const
      {
        this->GetValues(this->GetFEFaceValuesControl(), name, values);
      }

    /*********************************************/
    template<typename VECTOR, int dim>
      void
      FaceDataContainerInternal<VECTOR, dim>::GetFaceValuesControl(
          std::string name, std::vector<dealii::Vector<double> >& values) const
      {
        this->GetValues(this->GetFEFaceValuesControl(), name, values);
      }

    /*********************************************/
    template<typename VECTOR, int dim>
      template<int targetdim>
        void
        FaceDataContainerInternal<VECTOR, dim>::GetFaceGradsState(
            std::string name,
            std::vector<dealii::Tensor<1, targetdim> >& values) const
        {
          this->GetGrads<targetdim>(this->GetFEFaceValuesState(), name, values);
        }

    /*********************************************/
    template<typename VECTOR, int dim>
      template<int targetdim>
        void
        FaceDataContainerInternal<VECTOR, dim>::GetFaceGradsState(
            std::string name,
            std::vector<std::vector<dealii::Tensor<1, targetdim> > >& values) const
        {
          this->GetGrads<targetdim>(this->GetFEFaceValuesState(), name, values);
        }

    /***********************************************************************/

    template<typename VECTOR, int dim>
      template<int targetdim>
        void
        FaceDataContainerInternal<VECTOR, dim>::GetFaceGradsControl(
            std::string name,
            std::vector<dealii::Tensor<1, targetdim> >& values) const
        {
          this->GetGrads<targetdim>(this->GetFEFaceValuesControl(), name,
              values);
        }
    /***********************************************************************/

    template<typename VECTOR, int dim>
      template<int targetdim>
        void
        FaceDataContainerInternal<VECTOR, dim>::GetFaceGradsControl(
            std::string name,
            std::vector<std::vector<dealii::Tensor<1, targetdim> > >& values) const
        {
          this->GetGrads<targetdim>(this->GetFEFaceValuesControl(), name,
              values);
        }

    /*********************************************/
    template<typename VECTOR, int dim>
      void
      FaceDataContainerInternal<VECTOR, dim>::GetNbrFaceValuesState(
          std::string name, std::vector<double>& values) const
      {
        this->GetValues(this->GetNbrFEFaceValuesState(), name, values);
      }
    /*********************************************/
    template<typename VECTOR, int dim>
      void
      FaceDataContainerInternal<VECTOR, dim>::GetNbrFaceValuesState(
          std::string name, std::vector<Vector<double> >& values) const
      {
        this->GetValues(this->GetNbrFEFaceValuesState(), name, values);

      }

    /*********************************************/
    template<typename VECTOR, int dim>
      void
      FaceDataContainerInternal<VECTOR, dim>::GetNbrFaceValuesControl(
          std::string name, std::vector<double>& values) const
      {
        this->GetValues(this->GetNbrFEFaceValuesControl(), name, values);
      }

    /*********************************************/
    template<typename VECTOR, int dim>
      void
      FaceDataContainerInternal<VECTOR, dim>::GetNbrFaceValuesControl(
          std::string name, std::vector<Vector<double> >& values) const
      {
        this->GetValues(this->GetNbrFEFaceValuesControl(), name, values);
      }

    /*********************************************/
    template<typename VECTOR, int dim>
      template<int targetdim>
        void
        FaceDataContainerInternal<VECTOR, dim>::GetNbrFaceGradsState(
            std::string name, std::vector<Tensor<1, targetdim> >& values) const
        {
          this->GetGrads<targetdim>(this->GetNbrFEFaceValuesState(), name,
              values);
        }

    /*********************************************/
    template<typename VECTOR, int dim>
      template<int targetdim>
        void
        FaceDataContainerInternal<VECTOR, dim>::GetNbrFaceGradsState(
            std::string name,
            std::vector<std::vector<Tensor<1, targetdim> > >& values) const
        {
          this->GetGrads<targetdim>(this->GetNbrFEFaceValuesState(), name,
              values);
        }

    /***********************************************************************/

    template<typename VECTOR, int dim>
      template<int targetdim>
        void
        FaceDataContainerInternal<VECTOR, dim>::GetNbrFaceGradsControl(
            std::string name, std::vector<Tensor<1, targetdim> >& values) const
        {
          this->GetGrads<targetdim>(this->GetNbrFEFaceValuesControl(), name,
              values);
        }
    /***********************************************************************/

    template<typename VECTOR, int dim>
      template<int targetdim>
        void
        FaceDataContainerInternal<VECTOR, dim>::GetNbrFaceGradsControl(
            std::string name,
            std::vector<std::vector<Tensor<1, targetdim> > >& values) const
        {
          this->GetGrads<targetdim>(this->GetNbrFEFaceValuesControl(), name,
              values);
        }

    /***********************************************************************/

    /***********************************************************************/
    template<typename VECTOR, int dim>
      void
      FaceDataContainerInternal<VECTOR, dim>::GetValues(
          const dealii::FEFaceValuesBase<dim>& fe_values, std::string name,
          std::vector<double>& values) const
      {
        typename std::map<std::string, const VECTOR*>::const_iterator it =
            this->GetDomainValues().find(name);
        if (it == this->GetDomainValues().end())
        {
          throw DOpEException("Did not find " + name,
              "CellDataContainer::GetValues");
        }
        fe_values.get_function_values(*(it->second), values);
      }

    /***********************************************************************/
    template<typename VECTOR, int dim>
      void
      FaceDataContainerInternal<VECTOR, dim>::GetValues(
          const dealii::FEFaceValuesBase<dim>& fe_values, std::string name,
          std::vector<dealii::Vector<double> >& values) const
      {
        typename std::map<std::string, const VECTOR*>::const_iterator it =
            this->GetDomainValues().find(name);
        if (it == this->GetDomainValues().end())
        {
          throw DOpEException("Did not find " + name,
              "CellDataContainer::GetValues");
        }
        fe_values.get_function_values(*(it->second), values);
      }

    /***********************************************************************/

    template<typename VECTOR, int dim>
      template<int targetdim>
        void
        FaceDataContainerInternal<VECTOR, dim>::GetGrads(
            const dealii::FEFaceValuesBase<dim>& fe_values, std::string name,
            std::vector<dealii::Tensor<1, targetdim> >& values) const
        {
          typename std::map<std::string, const VECTOR*>::const_iterator it =
              this->GetDomainValues().find(name);
          if (it == this->GetDomainValues().end())
          {
            throw DOpEException("Did not find " + name,
                "FaceDataContainerInternal::GetGrads");
          }
          fe_values.get_function_gradients(*(it->second), values);
        }

    /***********************************************************************/

    template<typename VECTOR, int dim>
      template<int targetdim>
        void
        FaceDataContainerInternal<VECTOR, dim>::GetGrads(
            const dealii::FEFaceValuesBase<dim>& fe_values, std::string name,
            std::vector<std::vector<dealii::Tensor<1, targetdim> > >& values) const
        {
          typename std::map<std::string, const VECTOR*>::const_iterator it =
              this->GetDomainValues().find(name);
          if (it == this->GetDomainValues().end())
          {
            throw DOpEException("Did not find " + name,
                "FaceDataContainerInternal::GetGrads");
          }
          fe_values.get_function_gradients(*(it->second), values);
        }

  /***********************************************************************/
  }
}

#endif /* FACEDATACONTAINER_INTERNAL_H_ */
