/*
 * CellDataContainer_internal.h
 *
 *  Created on: Apr 3, 2012
 *      Author: cgoll
 */

#ifndef _CELLDATACONTAINER_INTERNAL_H_
#define _CELLDATACONTAINER_INTERNAL_H_

#include <deal.II/lac/vector.h>

#include "fevalues_wrapper.h"
#include "dopeexception.h"

namespace DOpE
{
  namespace cdcinternal
  {
    /**
     * This class houses all the functionality which is shared between
     * the CellDataContainer for normal and hp::DoFHandlers.
     *
     * @template VECTOR     Type of the vector we use in our computations
     *                      (i.e. Vector<double> or BlockVector<double>)
     * @template dim        The dimension of the integral we are actually
     *                      interested in.
     */
    template<typename VECTOR, int dim>
      class CellDataContainerInternal
      {
        public:
          CellDataContainerInternal(
              const std::map<std::string, const dealii::Vector<double>*> &param_values,
              const std::map<std::string, const VECTOR*> &domain_values);

          virtual
          ~CellDataContainerInternal()
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

          virtual const DOpEWrapper::FEValues<dim>&
          GetFEValuesState() const = 0;

          virtual const DOpEWrapper::FEValues<dim>&
          GetFEValuesControl() const = 0;

          /********************************************************************/
          /**
           * Functions to extract values and gradients out of the FEValues
           */

          /**
           * Writes the values of the state variable at the quadrature points into values.
           */
          void
          GetValuesState(std::string name, std::vector<double>& values) const;

          /*********************************************/
          /*
           * Same as above for the Vector valued case.
           */
          void
          GetValuesState(std::string name,
              std::vector<dealii::Vector<double> >& values) const;

          /*********************************************/
          /*
           * Writes the values of the control variable at the quadrature points into values
           */
          void
          GetValuesControl(std::string name, std::vector<double>& values) const;

          /*********************************************/
          /*
           * Same as above for the Vector valued case.
           */
          void
          GetValuesControl(std::string name,
              std::vector<dealii::Vector<double> >& values) const;
          /*********************************************/
          /*
           * Writes the values of the state gradient at the quadrature points into values.
           */

          template<int targetdim>
            void
            GetGradsState(std::string name,
                std::vector<dealii::Tensor<1, targetdim> >& values) const;

          /*********************************************/
          /*
           * Same as above for the Vector valued case.
           */
          template<int targetdim>
            void
            GetGradsState(std::string name,
                std::vector<std::vector<dealii::Tensor<1, targetdim> > >& values) const;

          /*********************************************/
          /*
           * Writes the values of the control gradient at the quadrature points into values.
           */
          template<int targetdim>
            void
            GetGradsControl(std::string name,
                std::vector<dealii::Tensor<1, targetdim> >& values) const;

          /*********************************************/
          /*
           * Same as above for the Vector valued case.
           */
          template<int targetdim>
            void
            GetGradsControl(std::string name,
                std::vector<std::vector<dealii::Tensor<1, targetdim> > >& values) const;

          /*********************************************/
          /*
           * Writes the values of the state laplacian
           * at the quadrature points into values.
           */

          void
          GetLaplaciansState(std::string name,
              std::vector<double> & values) const;

          /*********************************************/
          /*
           * Same as above for the Vector valued case.
           */

          void
          GetLaplaciansState(std::string name,
              std::vector<dealii::Vector<double> >& values) const;

          /*********************************************/
          /*
           * Writes the values of the control laplacian
           * at the quadrature points into values.
           */

          void
          GetLaplaciansControl(std::string name,
              std::vector<double> & values) const;

          /*********************************************/
          /*
           * Same as above for the Vector valued case.
           */

          void
          GetLaplaciansControl(std::string name,
              std::vector<dealii::Vector<double> >& values) const;

        private:
          /***********************************************************/
          /**
           * Helper Function. Vector valued case.
           */
          void
          GetValues(const DOpEWrapper::FEValues<dim>& fe_values,
              std::string name, std::vector<double>& values) const;
          /***********************************************************/
          /**
           * Helper Function. Vector valued case.
           */
          void
          GetValues(const DOpEWrapper::FEValues<dim>& fe_values,
              std::string name,
              std::vector<dealii::Vector<double> >& values) const;
          /***********************************************************/
          /**
           * Helper Function.
           */
          template<int targetdim>
            void
            GetGrads(const DOpEWrapper::FEValues<dim>& fe_values,
                std::string name,
                std::vector<dealii::Tensor<1, targetdim> >& values) const;
          /***********************************************************/
          /**
           * Helper Function. Vector valued case.
           */
          template<int targetdim>
            void
            GetGrads(const DOpEWrapper::FEValues<dim>& fe_values,
                std::string name,
                std::vector<std::vector<dealii::Tensor<1, targetdim> > >& values) const;
          /***********************************************************/
          /**
           * Helper Function.
           */
          void
          GetLaplacians(const DOpEWrapper::FEValues<dim>& fe_values,
              std::string name, std::vector<double>& values) const;

          /***********************************************************/
          /**
           * Helper Function.
           */
          void
          GetLaplacians(const DOpEWrapper::FEValues<dim>& fe_values,
              std::string name,
              std::vector<dealii::Vector<double> >& values) const;

          const std::map<std::string, const dealii::Vector<double>*> &_param_values;
          const std::map<std::string, const VECTOR*> &_domain_values;
      };

    /**********************************************************************/
    template<typename VECTOR, int dim>
      CellDataContainerInternal<VECTOR, dim>::CellDataContainerInternal(
          const std::map<std::string, const dealii::Vector<double>*> &param_values,
          const std::map<std::string, const VECTOR*> &domain_values)
          : _param_values(param_values), _domain_values(domain_values)
      {
      }

    template<typename VECTOR, int dim>
      void
      CellDataContainerInternal<VECTOR, dim>::GetParamValues(std::string name,
          dealii::Vector<double>& value) const
      {
        typename std::map<std::string, const dealii::Vector<double>*>::const_iterator it =
            _param_values.find(name);
        if (it == _param_values.end())
        {
          throw DOpEException("Did not find " + name,
              "CellDataContainerInternal::GetParamValues");
        }
        value = *(it->second);
      }

    /*********************************************/
    template<typename VECTOR, int dim>
      void
      CellDataContainerInternal<VECTOR, dim>::GetValuesState(std::string name,
          std::vector<double>& values) const
      {
        this->GetValues(this->GetFEValuesState(), name, values);
      }
    /*********************************************/
    template<typename VECTOR, int dim>
      void
      CellDataContainerInternal<VECTOR, dim>::GetValuesState(std::string name,
          std::vector<dealii::Vector<double> >& values) const
      {
        this->GetValues(this->GetFEValuesState(), name, values);

      }

    /*********************************************/
    template<typename VECTOR, int dim>
      void
      CellDataContainerInternal<VECTOR, dim>::GetValuesControl(std::string name,
          std::vector<double>& values) const
      {
        this->GetValues(this->GetFEValuesControl(), name, values);
      }

    /*********************************************/
    template<typename VECTOR, int dim>
      void
      CellDataContainerInternal<VECTOR, dim>::GetValuesControl(std::string name,
          std::vector<dealii::Vector<double> >& values) const
      {
        this->GetValues(this->GetFEValuesControl(), name, values);
      }

    /*********************************************/
    template<typename VECTOR, int dim>
      template<int targetdim>
        void
        CellDataContainerInternal<VECTOR, dim>::GetGradsState(std::string name,
            std::vector<dealii::Tensor<1, targetdim> >& values) const
        {
          this->GetGrads<targetdim>(this->GetFEValuesState(), name, values);
        }

    /*********************************************/
    template<typename VECTOR, int dim>
      template<int targetdim>
        void
        CellDataContainerInternal<VECTOR, dim>::GetGradsState(std::string name,
            std::vector<std::vector<dealii::Tensor<1, targetdim> > >& values) const
        {
          this->GetGrads<targetdim>(this->GetFEValuesState(), name, values);
        }

    /***********************************************************************/

    template<typename VECTOR, int dim>
      template<int targetdim>
        void
        CellDataContainerInternal<VECTOR, dim>::GetGradsControl(
            std::string name,
            std::vector<dealii::Tensor<1, targetdim> >& values) const
        {
          this->GetGrads<targetdim>(this->GetFEValuesControl(), name, values);
        }
    /***********************************************************************/

    template<typename VECTOR, int dim>
      template<int targetdim>
        void
        CellDataContainerInternal<VECTOR, dim>::GetGradsControl(
            std::string name,
            std::vector<std::vector<dealii::Tensor<1, targetdim> > >& values) const
        {
          this->GetGrads<targetdim>(this->GetFEValuesControl(), name, values);
        }

    /***********************************************************************/
    template<typename VECTOR, int dim>
      void
      CellDataContainerInternal<VECTOR, dim>::GetLaplaciansState(
          std::string name, std::vector<double> & values) const
      {
        this->GetLaplacians(this->GetFEValuesState(), name, values);
      }

    /***********************************************************************/
    template<typename VECTOR, int dim>
      void
      CellDataContainerInternal<VECTOR, dim>::GetLaplaciansState(
          std::string name, std::vector<dealii::Vector<double> >& values) const
      {
        this->GetLaplacians(this->GetFEValuesState(), name, values);
      }
    /***********************************************************************/
    template<typename VECTOR, int dim>
      void
      CellDataContainerInternal<VECTOR, dim>::GetLaplaciansControl(
          std::string name, std::vector<double> & values) const
      {
        this->GetLaplacians(this->GetFEValuesControl(), name, values);
      }

    /***********************************************************************/
    template<typename VECTOR, int dim>
      void
      CellDataContainerInternal<VECTOR, dim>::GetLaplaciansControl(
          std::string name, std::vector<dealii::Vector<double> >& values) const
      {
        this->GetLaplacians(this->GetFEValuesControl(), name, values);
      }
    /***********************************************************************/
    template<typename VECTOR, int dim>
      void
      CellDataContainerInternal<VECTOR, dim>::GetValues(
          const DOpEWrapper::FEValues<dim>& fe_values, std::string name,
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
      CellDataContainerInternal<VECTOR, dim>::GetValues(
          const DOpEWrapper::FEValues<dim>& fe_values, std::string name,
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
        CellDataContainerInternal<VECTOR, dim>::GetGrads(
            const DOpEWrapper::FEValues<dim>& fe_values, std::string name,
            std::vector<dealii::Tensor<1, targetdim> >& values) const
        {
          typename std::map<std::string, const VECTOR*>::const_iterator it =
              this->GetDomainValues().find(name);
          if (it == this->GetDomainValues().end())
          {
            throw DOpEException("Did not find " + name,
                "CellDataContainerInternal::GetGrads");
          }
          fe_values.get_function_gradients(*(it->second), values);
        }

    /***********************************************************************/

    template<typename VECTOR, int dim>
      template<int targetdim>
        void
        CellDataContainerInternal<VECTOR, dim>::GetGrads(
            const DOpEWrapper::FEValues<dim>& fe_values, std::string name,
            std::vector<std::vector<dealii::Tensor<1, targetdim> > >& values) const
        {
          typename std::map<std::string, const VECTOR*>::const_iterator it =
              this->GetDomainValues().find(name);
          if (it == this->GetDomainValues().end())
          {
            throw DOpEException("Did not find " + name,
                "CellDataContainerInternal::GetGrads");
          }
          fe_values.get_function_gradients(*(it->second), values);
        }

    /***********************************************************************/

    template<typename VECTOR, int dim>
      void
      CellDataContainerInternal<VECTOR, dim>::GetLaplacians(
          const DOpEWrapper::FEValues<dim>& fe_values, std::string name,
          std::vector<double> & values) const
      {
        typename std::map<std::string, const VECTOR*>::const_iterator it =
            this->GetDomainValues().find(name);
        if (it == this->GetDomainValues().end())
        {
          throw DOpEException("Did not find " + name,
              "CellDataContainerInternal::GetLaplacians");
        }
        fe_values.get_function_laplacians(*(it->second), values);
      }

    /***********************************************************************/

    template<typename VECTOR, int dim>
      void
      CellDataContainerInternal<VECTOR, dim>::GetLaplacians(
          const DOpEWrapper::FEValues<dim>& fe_values, std::string name,
          std::vector<dealii::Vector<double> >& values) const
      {
        typename std::map<std::string, const VECTOR*>::const_iterator it =
            this->GetDomainValues().find(name);
        if (it == this->GetDomainValues().end())
        {
          throw DOpEException("Did not find " + name,
              "CellDataContainerInternal::GetLaplacians");
        }
        fe_values.get_function_laplacians(*(it->second), values);
      }
  } //end of namespace cdcinternal
}

#endif /* CellDataContainer_INTERNAL_H_ */
