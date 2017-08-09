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

#ifndef NETWORK_ELEMENTDATACONTAINER_INTERNAL_H_
#define NETWORK_ELEMENTDATACONTAINER_INTERNAL_H_

#include <deal.II/lac/vector.h>

#include <wrapper/fevalues_wrapper.h>
#include <include/dopeexception.h>

namespace DOpE
{
  namespace Networks
  {
    namespace edcinternal
    {
      /**
       * This class houses all the functionality which is shared between
       * the Network_ElementDataContainer for normal and hp::DoFHandlers.
       *
       * @template dim        The dimension of the integral we are actually
       *                      interested in.
       */
      template<int dim>
      class Network_ElementDataContainerInternal
      {
      public:
        Network_ElementDataContainerInternal(
          unsigned int pipe,
          const std::map<std::string, const dealii::Vector<double>*> &param_values
          ,
          const std::map<std::string, const dealii::BlockVector<double> *> &domain_values);

        virtual
        ~Network_ElementDataContainerInternal()
        {
        }
        ;

        /**
         * Looks up the given name in parameter_data_ and returns the
         * corresponding value through 'value'.
         */
        void
        GetParamValues(std::string name, dealii::Vector<double> &value) const;

        /**
         * Returns the domain values.
         */
        const std::map<std::string, const dealii::BlockVector<double> *> &
        GetDomainValues() const
        {
          return domain_values_;
        }
        ;

        virtual const DOpEWrapper::FEValues<dim> &
        GetFEValuesState() const = 0;

        virtual const DOpEWrapper::FEValues<dim> &
        GetFEValuesControl() const = 0;

        /*********************************************************************/
        /**
         * Return a triangulation iterator to the current element for the state.
         */
        const typename  Triangulation<dim>::cell_iterator
        GetElementState() const;


        /********************************************************************/
        /**
         * Functions to extract values and gradients out of the FEValues
         */

        /**
         * Writes the values of the state variable at the quadrature points into values.
         */
        void
        GetValuesState(std::string name, std::vector<double> &values) const;

        /*********************************************/
        /*
         * Same as above for the Vector valued case.
         */
        void
        GetValuesState(std::string name,
                       std::vector<dealii::Vector<double> > &values) const;

        /*********************************************/
        /*
         * Writes the values of the control variable at the quadrature points into values
         */
        void
        GetValuesControl(std::string name, std::vector<double> &values) const;

        /*********************************************/
        /*
         * Same as above for the Vector valued case.
         */
        void
        GetValuesControl(std::string name,
                         std::vector<dealii::Vector<double> > &values) const;
        /*********************************************/
        /*
         * Writes the values of the state gradient at the quadrature points into values.
         */

        template<int targetdim>
        void
        GetGradsState(std::string name,
                      std::vector<dealii::Tensor<1, targetdim> > &values) const;

        /*********************************************/
        /*
         * Same as above for the Vector valued case.
         */
        template<int targetdim>
        void
        GetGradsState(
          std::string name,
          std::vector<std::vector<dealii::Tensor<1, targetdim> > > &values) const;

        /*********************************************/
        /*
         * Writes the values of the control gradient at the quadrature points into values.
         */
        template<int targetdim>
        void
        GetGradsControl(std::string name,
                        std::vector<dealii::Tensor<1, targetdim> > &values) const;

        /*********************************************/
        /*
         * Same as above for the Vector valued case.
         */
        template<int targetdim>
        void
        GetGradsControl(
          std::string name,
          std::vector<std::vector<dealii::Tensor<1, targetdim> > > &values) const;
        /*********************************************/
        /*
         * Writes the values of the state hessian at the quadrature points into values.
         */
        template<int targetdim>
        void
        GetHessiansState(std::string name,
                         std::vector<dealii::Tensor<2, targetdim> > &values) const;

        /*********************************************/
        /*
         * Same as above for the Vector valued case.
         */
        template<int targetdim>
        void
        GetHessiansState(
          std::string name,
          std::vector<std::vector<dealii::Tensor<2, targetdim> > > &values) const;

        /*********************************************/
        /*
         * Writes the values of the control hessian at the quadrature points into values.
         */
        template<int targetdim>
        void
        GetHessiansControl(std::string name,
                           std::vector<dealii::Tensor<2, targetdim> > &values) const;

        /*********************************************/
        /*
         * Same as above for the Vector valued case.
         */
        template<int targetdim>
        void
        GetHessiansControl(
          std::string name,
          std::vector<std::vector<dealii::Tensor<2, targetdim> > > &values) const;

        /*********************************************/
        /*
         * Writes the values of the state laplacian
         * at the quadrature points into values.
         */

        void
        GetLaplaciansState(std::string name,
                           std::vector<double> &values) const;

        /*********************************************/
        /*
         * Same as above for the Vector valued case.
         */

        void
        GetLaplaciansState(std::string name,
                           std::vector<dealii::Vector<double> > &values) const;

        /*********************************************/
        /*
         * Writes the values of the control laplacian
         * at the quadrature points into values.
         */

        void
        GetLaplaciansControl(std::string name,
                             std::vector<double> &values) const;

        /*********************************************/
        /*
         * Same as above for the Vector valued case.
         */

        void
        GetLaplaciansControl(std::string name,
                             std::vector<dealii::Vector<double> > &values) const;

      private:
        /***********************************************************/
        /**
         * Helper Function. Vector valued case.
         */
        void
        GetValues(const DOpEWrapper::FEValues<dim> &fe_values,
                  std::string name, std::vector<double> &values) const;
        /***********************************************************/
        /**
         * Helper Function. Vector valued case.
         */
        void
        GetValues(const DOpEWrapper::FEValues<dim> &fe_values,
                  std::string name,
                  std::vector<dealii::Vector<double> > &values) const;
        /***********************************************************/
        /**
         * Helper Function.
         */
        template<int targetdim>
        void
        GetGrads(const DOpEWrapper::FEValues<dim> &fe_values,
                 std::string name,
                 std::vector<dealii::Tensor<1, targetdim> > &values) const;
        /***********************************************************/
        /**
         * Helper Function. Vector valued case.
         */
        template<int targetdim>
        void
        GetGrads(
          const DOpEWrapper::FEValues<dim> &fe_values,
          std::string name,
          std::vector<std::vector<dealii::Tensor<1, targetdim> > > &values) const;
        /***********************************************************/
        /**
         * Helper Function.
         */
        void
        GetLaplacians(const DOpEWrapper::FEValues<dim> &fe_values,
                      std::string name, std::vector<double> &values) const;

        /***********************************************************/
        /**
         * Helper Function.
         */
        void
        GetLaplacians(const DOpEWrapper::FEValues<dim> &fe_values,
                      std::string name,
                      std::vector<dealii::Vector<double> > &values) const;

        /***********************************************************/
        /**
         * Helper Function.
         */
        template<int targetdim>
        void
        GetHessians(const DOpEWrapper::FEValues<dim> &fe_values,
                    std::string name,
                    std::vector<dealii::Tensor<2, targetdim> > &values) const;

        /***********************************************************/
        /**
         * Helper Function.
         */
        template<int targetdim>
        void
        GetHessians(
          const DOpEWrapper::FEValues<dim> &fe_values,
          std::string name,
          std::vector<std::vector<dealii::Tensor<2, targetdim> > > &values) const;

        const std::map<std::string, const dealii::Vector<double>*> &param_values_;
        const std::map<std::string, const dealii::BlockVector<double> *> &domain_values_;
        unsigned int pipe_;
      };

      /**********************************************************************/
      template<int dim>
      Network_ElementDataContainerInternal<dim>::Network_ElementDataContainerInternal(
        unsigned int pipe,
        const std::map<std::string, const dealii::Vector<double>*> &param_values,
        const std::map<std::string, const dealii::BlockVector<double> *> &domain_values)
        : param_values_(param_values), domain_values_(domain_values)
      {
        pipe_ = pipe;
      }

      template<int dim>
      void
      Network_ElementDataContainerInternal<dim>::GetParamValues(std::string name,
                                                                dealii::Vector<double> &value) const
      {
        typename std::map<std::string, const dealii::Vector<double>*>::const_iterator it =
          param_values_.find(name);
        if (it == param_values_.end())
          {
            throw DOpEException("Did not find " + name,
                                "Network_ElementDataContainerInternal::GetParamValues");
          }
        value = *(it->second);
      }

      /*********************************************/
      template<int dim>
      void
      Network_ElementDataContainerInternal<dim>::GetValuesState(std::string name,
                                                                std::vector<double> &values) const
      {
        this->GetValues(this->GetFEValuesState(), name, values);
      }
      /*********************************************/
      template<int dim>
      void
      Network_ElementDataContainerInternal<dim>::GetValuesState(std::string name,
                                                                std::vector<dealii::Vector<double> > &values) const
      {
        this->GetValues(this->GetFEValuesState(), name, values);

      }


      /*********************************************/
      template<int dim>
      const typename Triangulation<dim>::cell_iterator
      Network_ElementDataContainerInternal<dim>::GetElementState() const
      {
        return this->GetFEValuesState().get_element();
      }

      /*********************************************/
      template<int dim>
      void
      Network_ElementDataContainerInternal<dim>::GetValuesControl(std::string name,
                                                                  std::vector<double> &values) const
      {
        this->GetValues(this->GetFEValuesControl(), name, values);
      }

      /*********************************************/
      template<int dim>
      void
      Network_ElementDataContainerInternal<dim>::GetValuesControl(std::string name,
                                                                  std::vector<dealii::Vector<double> > &values) const
      {
        this->GetValues(this->GetFEValuesControl(), name, values);
      }

      /*********************************************/
      template<int dim>
      template<int targetdim>
      void
      Network_ElementDataContainerInternal<dim>::GetGradsState(std::string name,
                                                               std::vector<dealii::Tensor<1, targetdim> > &values) const
      {
        this->GetGrads<targetdim>(this->GetFEValuesState(), name, values);
      }

      /*********************************************/
      template<int dim>
      template<int targetdim>
      void
      Network_ElementDataContainerInternal<dim>::GetGradsState(
        std::string name,
        std::vector<std::vector<dealii::Tensor<1, targetdim> > > &values) const
      {
        this->GetGrads<targetdim>(this->GetFEValuesState(), name, values);
      }

      /***********************************************************************/

      template<int dim>
      template<int targetdim>
      void
      Network_ElementDataContainerInternal<dim>::GetGradsControl(
        std::string name,
        std::vector<dealii::Tensor<1, targetdim> > &values) const
      {
        this->GetGrads<targetdim>(this->GetFEValuesControl(), name, values);
      }

      /***********************************************************************/

      template<int dim>
      template<int targetdim>
      void
      Network_ElementDataContainerInternal<dim>::GetGradsControl(
        std::string name,
        std::vector<std::vector<dealii::Tensor<1, targetdim> > > &values) const
      {
        this->GetGrads<targetdim>(this->GetFEValuesControl(), name, values);
      }

      /***********************************************************************/

      template<int dim>
      template<int targetdim>
      void
      Network_ElementDataContainerInternal<dim>::GetHessiansState(
        std::string name,
        std::vector<std::vector<dealii::Tensor<2, targetdim> > > &values) const
      {
        this->GetHessians<targetdim>(this->GetFEValuesState(), name, values);
      }

      /***********************************************************************/

      template<int dim>
      template<int targetdim>
      void
      Network_ElementDataContainerInternal<dim>::GetHessiansState(
        std::string name,
        std::vector<dealii::Tensor<2, targetdim> > &values) const
      {
        this->GetHessians<targetdim>(this->GetFEValuesState(), name, values);
      }

      /***********************************************************************/

      template<int dim>
      template<int targetdim>
      void
      Network_ElementDataContainerInternal<dim>::GetHessiansControl(
        std::string name,
        std::vector<std::vector<dealii::Tensor<2, targetdim> > > &values) const
      {
        this->GetHessians<targetdim>(this->GetFEValuesControl(), name, values);
      }

      /***********************************************************************/

      template<int dim>
      template<int targetdim>
      void
      Network_ElementDataContainerInternal<dim>::GetHessiansControl(
        std::string name,
        std::vector<dealii::Tensor<2, targetdim> > &values) const
      {
        this->GetHessians<targetdim>(this->GetFEValuesControl(), name,
                                     values);
      }

      /***********************************************************************/
      template<int dim>
      void
      Network_ElementDataContainerInternal<dim>::GetLaplaciansState(
        std::string name, std::vector<double> &values) const
      {
        this->GetLaplacians(this->GetFEValuesState(), name, values);
      }

      /***********************************************************************/
      template<int dim>
      void
      Network_ElementDataContainerInternal<dim>::GetLaplaciansState(
        std::string name, std::vector<dealii::Vector<double> > &values) const
      {
        this->GetLaplacians(this->GetFEValuesState(), name, values);
      }
      /***********************************************************************/
      template<int dim>
      void
      Network_ElementDataContainerInternal<dim>::GetLaplaciansControl(
        std::string name, std::vector<double> &values) const
      {
        this->GetLaplacians(this->GetFEValuesControl(), name, values);
      }

      /***********************************************************************/
      template<int dim>
      void
      Network_ElementDataContainerInternal<dim>::GetLaplaciansControl(
        std::string name, std::vector<dealii::Vector<double> > &values) const
      {
        this->GetLaplacians(this->GetFEValuesControl(), name, values);
      }
      /***********************************************************************/
      template<int dim>
      void
      Network_ElementDataContainerInternal<dim>::GetValues(
        const DOpEWrapper::FEValues<dim> &fe_values, std::string name,
        std::vector<double> &values) const
      {
        typename std::map<std::string, const dealii::BlockVector<double> *>::const_iterator it =
          this->GetDomainValues().find(name);
        if (it == this->GetDomainValues().end())
          {
            throw DOpEException("Did not find " + name,
                                "Network_ElementDataContainer::GetValues");
          }
        fe_values.get_function_values(it->second->block(pipe_), values);
      }

      /***********************************************************************/
      template<int dim>
      void
      Network_ElementDataContainerInternal<dim>::GetValues(
        const DOpEWrapper::FEValues<dim> &fe_values, std::string name,
        std::vector<dealii::Vector<double> > &values) const
      {
        typename std::map<std::string, const dealii::BlockVector<double> *>::const_iterator it =
          this->GetDomainValues().find(name);
        if (it == this->GetDomainValues().end())
          {
            throw DOpEException("Did not find " + name,
                                "Network_ElementDataContainer::GetValues");
          }
        fe_values.get_function_values(it->second->block(pipe_), values);
      }

      /***********************************************************************/

      template<int dim>
      template<int targetdim>
      void
      Network_ElementDataContainerInternal<dim>::GetGrads(
        const DOpEWrapper::FEValues<dim> &fe_values, std::string name,
        std::vector<dealii::Tensor<1, targetdim> > &values) const
      {
        typename std::map<std::string, const dealii::BlockVector<double> *>::const_iterator it =
          this->GetDomainValues().find(name);
        if (it == this->GetDomainValues().end())
          {
            throw DOpEException("Did not find " + name,
                                "Network_ElementDataContainerInternal::GetGrads");
          }
        fe_values.get_function_gradients(it->second->block(pipe_), values);
      }

      /***********************************************************************/

      template<int dim>
      template<int targetdim>
      void
      Network_ElementDataContainerInternal<dim>::GetGrads(
        const DOpEWrapper::FEValues<dim> &fe_values,
        std::string name,
        std::vector<std::vector<dealii::Tensor<1, targetdim> > > &values) const
      {
        typename std::map<std::string, const dealii::BlockVector<double> *>::const_iterator it =
          this->GetDomainValues().find(name);
        if (it == this->GetDomainValues().end())
          {
            throw DOpEException("Did not find " + name,
                                "Network_ElementDataContainerInternal::GetGrads");
          }
        fe_values.get_function_gradients(it->second->block(pipe_), values);
      }

      /***********************************************************************/

      template<int dim>
      void
      Network_ElementDataContainerInternal<dim>::GetLaplacians(
        const DOpEWrapper::FEValues<dim> &fe_values, std::string name,
        std::vector<double> &values) const
      {
        typename std::map<std::string, const dealii::BlockVector<double> *>::const_iterator it =
          this->GetDomainValues().find(name);
        if (it == this->GetDomainValues().end())
          {
            throw DOpEException("Did not find " + name,
                                "Network_ElementDataContainerInternal::GetLaplacians");
          }
        fe_values.get_function_laplacians(it->second->block(pipe_), values);
      }

      /***********************************************************************/

      template<int dim>
      void
      Network_ElementDataContainerInternal<dim>::GetLaplacians(
        const DOpEWrapper::FEValues<dim> &fe_values, std::string name,
        std::vector<dealii::Vector<double> > &values) const
      {
        typename std::map<std::string, const dealii::BlockVector<double> *>::const_iterator it =
          this->GetDomainValues().find(name);
        if (it == this->GetDomainValues().end())
          {
            throw DOpEException("Did not find " + name,
                                "Network_ElementDataContainerInternal::GetLaplacians");
          }
        fe_values.get_function_laplacians(it->second->block(pipe_), values);
      }

      /***********************************************************************/

      template<int dim>
      template<int targetdim>
      void
      Network_ElementDataContainerInternal<dim>::GetHessians(
        const DOpEWrapper::FEValues<dim> &fe_values,
        std::string name,
        std::vector<std::vector<dealii::Tensor<2, targetdim> > > &values) const
      {
        typename std::map<std::string, const dealii::BlockVector<double> *>::const_iterator it =
          this->GetDomainValues().find(name);
        if (it == this->GetDomainValues().end())
          {
            throw DOpEException("Did not find " + name,
                                "Network_ElementDataContainerInternal::GetGrads");
          }
        fe_values.get_function_hessians(it->second->block(pipe_), values);
      }

      /***********************************************************************/

      template<int dim>
      template<int targetdim>
      void
      Network_ElementDataContainerInternal<dim>::GetHessians(
        const DOpEWrapper::FEValues<dim> &fe_values, std::string name,
        std::vector<dealii::Tensor<2, targetdim> > &values) const
      {
        typename std::map<std::string, const dealii::BlockVector<double> *>::const_iterator it =
          this->GetDomainValues().find(name);
        if (it == this->GetDomainValues().end())
          {
            throw DOpEException("Did not find " + name,
                                "Network_ElementDataContainerInternal::GetGrads");
          }
        fe_values.get_function_hessians(it->second->block(pipe_), values);
      }

    } //end of namespace edcinternal
  }
}

#endif /* Network_ElementDataContainer_INTERNAL_H_ */
