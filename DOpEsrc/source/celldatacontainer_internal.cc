/*
 * celldatacontainer_internal.cc
 *
 *  Created on: Apr 3, 2012
 *      Author: cgoll
 */

#include "celldatacontainer_internal.h"
#include "dopeexception.h"

namespace DOpE
{
  namespace cdcinternal
  {
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

    //explicit instantiations

    template class CellDataContainerInternal<dealii::Vector<double>,
        deal_II_dimension> ;
    template class CellDataContainerInternal<dealii::BlockVector<double>,
        deal_II_dimension> ;
    template class CellDataContainerInternal<dealii::Vector<float>,
        deal_II_dimension> ;
    template class CellDataContainerInternal<dealii::BlockVector<float>,
        deal_II_dimension> ;
  }
}

