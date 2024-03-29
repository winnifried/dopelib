/**
*
* Copyright (C) 2012-2018 by the DOpElib authors
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

#ifndef NETWORK_FACEDATACONTAINER_INTERNAL_H_
#define NETWORK_FACEDATACONTAINER_INTERNAL_H_

#include <deal.II/lac/vector.h>

#include <wrapper/fevalues_wrapper.h>
#include <include/dopeexception.h>

namespace DOpE
{
  namespace Networks
  {
    namespace fdcinternal
    {
      /**
       * This class houses all the functionality which is shared between
       * the Network_FaceDataContainer for normal and hp::DoFHandlers.
       *
       * @template dim        The dimension of the integral we are actually
       *                      interested in.
       */
      template<int dim>
      class Network_FaceDataContainerInternal
      {
      public:
        Network_FaceDataContainerInternal(
          unsigned int pipe_,
          unsigned int n_pipes,
          unsigned int n_comp,
          const std::map<std::string, const dealii::Vector<double>*> &param_values,
          const std::map<std::string, const dealii::BlockVector<double> *> &domain_values,
          bool need_neighbour);

        virtual
        ~Network_FaceDataContainerInternal()
        {
        }

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

        virtual const dealii::FEFaceValuesBase<dim> &
        GetFEFaceValuesState() const =0;
        virtual const dealii::FEFaceValuesBase<dim> &
        GetFEFaceValuesControl() const = 0;

        virtual const dealii::FEFaceValuesBase<dim> &
        GetNbrFEFaceValuesState() const = 0;
        virtual const dealii::FEFaceValuesBase<dim> &
        GetNbrFEFaceValuesControl() const = 0;

        /*********************************************************************/
        /**
         * Return a triangulation iterator to the current element for the state.
         */
        const typename Triangulation<dim>::cell_iterator
        GetElementState() const;

        /********************************************************************/
        /**
         * Functions to extract values and gradients out of the FEValues
         */

        /**
         * Writes the values of the state variable at the quadrature points into values.
         */
        void
        GetFaceValuesState(std::string name,
                           std::vector<double> &values) const;

        /*********************************************/
        /*
         * Same as above for the Vector valued case.
         */
        void
        GetFaceValuesState(std::string name,
                           std::vector<dealii::Vector<double> > &values) const;

        /*********************************************/
        /*
         * Writes the values of the control variable at the quadrature points into values
         */
        void
        GetFaceValuesControl(std::string name,
                             std::vector<double> &values) const;

        /*********************************************/
        /*
         * Same as above for the Vector valued case.
         */
        void
        GetFaceValuesControl(std::string name,
                             std::vector<dealii::Vector<double> > &values) const;
        /*********************************************/
        /*
         * Writes the values of the state gradient at the quadrature points into values.
         */

        template<int targetdim>
        void
        GetFaceGradsState(std::string name,
                          std::vector<dealii::Tensor<1, targetdim> > &values) const;

        /*********************************************/
        /*
         * Same as above for the Vector valued case.
         */
        template<int targetdim>
        void
        GetFaceGradsState(std::string name,
                          std::vector<std::vector<dealii::Tensor<1, targetdim> > > &values) const;

        /*********************************************/
        /*
         * Writes the values of the control gradient at the quadrature points into values.
         */
        template<int targetdim>
        void
        GetFaceGradsControl(std::string name,
                            std::vector<dealii::Tensor<1, targetdim> > &values) const;

        /*********************************************/
        /*
         * Same as above for the Vector valued case.
         */
        template<int targetdim>
        void
        GetFaceGradsControl(std::string name,
                            std::vector<std::vector<dealii::Tensor<1, targetdim> > > &values) const;

        /*
         * Writes the values of the state variable at the quadrature points into values.
         */
        inline void
        GetNbrFaceValuesState(std::string name,
                              std::vector<double> &values) const;
        /*********************************************/
        /*
         * Same as above for the Vector valued case.
         */
        inline void
        GetNbrFaceValuesState(std::string name,
                              std::vector<Vector<double> > &values) const;

        /*********************************************/

        /*
         * Writes the values of the control variable at the quadrature points into values
         */
        inline void
        GetNbrFaceValuesControl(std::string name,
                                std::vector<double> &values) const;
        /*********************************************/

        /*
         * Same as above for the Vector valued case.
         */
        inline void
        GetNbrFaceValuesControl(std::string name,
                                std::vector<Vector<double> > &values) const;
        /*********************************************/

        /*
         * Writes the values of the state gradient at the quadrature points into values.
         */

        template<int targetdim>
        inline void
        GetNbrFaceGradsState(std::string name,
                             std::vector<dealii::Tensor<1, targetdim> > &values) const;

        /*********************************************/

        /*
         * Same as avoe for the Vector valued case.
         */
        template<int targetdim>
        inline void
        GetNbrFaceGradsState(std::string name,
                             std::vector<std::vector<dealii::Tensor<1, targetdim> > > &values) const;

        /*********************************************/

        /*
         * Writes the values of the control gradient at the quadrature points into values.
         */

        template<int targetdim>
        inline void
        GetNbrFaceGradsControl(std::string name,
                               std::vector<dealii::Tensor<1, targetdim> > &values) const;

        /*********************************************/
        /*
         * Same as above for the Vector valued case.
         */
        template<int targetdim>
        inline void
        GetNbrFaceGradsControl(std::string name,
                               std::vector<std::vector<dealii::Tensor<1, targetdim> > > &values) const;

      protected:
        void
        SetFace(unsigned int face)
        {
          face_ = face;
        }
        unsigned int
        GetFace() const
        {
          return face_;
        }
        void
        SetSubFace(unsigned int subface)
        {
          subface_ = subface;
        }
        unsigned int
        GetSubFace() const
        {
          return subface_;
        }
        bool
        NeedNeighbour() const
        {
          return need_neighbour_;
        }

        unsigned int GetPipe() const
        {
          return pipe_;
        }
        unsigned int GetNPipes() const
        {
          return n_pipes_;
        }
        unsigned int GetNComp() const
        {
          return n_comp_;
        }
      private:
        /***********************************************************/
        /**
         * Helper Function. Vector valued case.
         */
        void
        GetValues(const dealii::FEFaceValuesBase<dim> &fe_values,
                  std::string name, std::vector<double> &values) const;
        /***********************************************************/
        /**
         * Helper Function. Vector valued case.
         */
        void
        GetValues(const dealii::FEFaceValuesBase<dim> &fe_values,
                  std::string name,
                  std::vector<dealii::Vector<double> > &values) const;
        /***********************************************************/
        /**
         * Helper Function.
         */
        template<int targetdim>
        void
        GetGrads(const dealii::FEFaceValuesBase<dim> &fe_values,
                 std::string name,
                 std::vector<dealii::Tensor<1, targetdim> > &values) const;
        /***********************************************************/
        /**
         * Helper Function. Vector valued case.
         */
        template<int targetdim>
        void
        GetGrads(const dealii::FEFaceValuesBase<dim> &fe_values,
                 std::string name,
                 std::vector<std::vector<dealii::Tensor<1, targetdim> > > &values) const;

        const std::map<std::string, const dealii::Vector<double>*> &param_values_;
        const std::map<std::string, const dealii::BlockVector<double> *> &domain_values_;

        unsigned int face_ = 0;
        unsigned int subface_ = 0;
        bool need_neighbour_;
        unsigned int pipe_, n_pipes_, n_comp_;
      };

      /**********************************************************************/
      template<int dim>
      Network_FaceDataContainerInternal<dim>::Network_FaceDataContainerInternal(
        unsigned int pipe,
        unsigned int n_pipes,
        unsigned int n_comp,
        const std::map<std::string, const dealii::Vector<double>*> &param_values,
        const std::map<std::string, const dealii::BlockVector<double> *> &domain_values,
        bool need_neighbour)
        : param_values_(param_values), domain_values_(domain_values), need_neighbour_(
            need_neighbour)
      {
        pipe_ = pipe;
        n_pipes_ = n_pipes;
        n_comp_ = n_comp;
      }

      template<int dim>
      void
      Network_FaceDataContainerInternal<dim>::GetParamValues(std::string name,
                                                             dealii::Vector<double> &value) const
      {
        typename std::map<std::string, const dealii::Vector<double>*>::const_iterator it =
          param_values_.find(name);
        if (it == param_values_.end())
          {
            throw DOpEException("Did not find " + name,
                                "Network_FaceDataContainerInternal::GetParamValues");
          }
        value = *(it->second);
      }

      /*********************************************/
      template<int dim>
      const typename Triangulation<dim>::cell_iterator
      Network_FaceDataContainerInternal<dim>::GetElementState() const
      {
        return this->GetFEFaceValuesState().get_element();
      }

      /*********************************************/
      template<int dim>
      void
      Network_FaceDataContainerInternal<dim>::GetFaceValuesState(
        std::string name, std::vector<double> &values) const
      {
        this->GetValues(this->GetFEFaceValuesState(), name, values);
      }
      /*********************************************/
      template<int dim>
      void
      Network_FaceDataContainerInternal<dim>::GetFaceValuesState(
        std::string name, std::vector<dealii::Vector<double> > &values) const
      {
        this->GetValues(this->GetFEFaceValuesState(), name, values);

      }

      /*********************************************/
      template<int dim>
      void
      Network_FaceDataContainerInternal<dim>::GetFaceValuesControl(
        std::string name, std::vector<double> &values) const
      {
        this->GetValues(this->GetFEFaceValuesControl(), name, values);
      }

      /*********************************************/
      template<int dim>
      void
      Network_FaceDataContainerInternal<dim>::GetFaceValuesControl(
        std::string name, std::vector<dealii::Vector<double> > &values) const
      {
        this->GetValues(this->GetFEFaceValuesControl(), name, values);
      }

      /*********************************************/
      template<int dim>
      template<int targetdim>
      void
      Network_FaceDataContainerInternal<dim>::GetFaceGradsState(
        std::string name,
        std::vector<dealii::Tensor<1, targetdim> > &values) const
      {
        this->GetGrads<targetdim>(this->GetFEFaceValuesState(), name, values);
      }

      /*********************************************/
      template<int dim>
      template<int targetdim>
      void
      Network_FaceDataContainerInternal<dim>::GetFaceGradsState(
        std::string name,
        std::vector<std::vector<dealii::Tensor<1, targetdim> > > &values) const
      {
        this->GetGrads<targetdim>(this->GetFEFaceValuesState(), name, values);
      }

      /***********************************************************************/

      template<int dim>
      template<int targetdim>
      void
      Network_FaceDataContainerInternal<dim>::GetFaceGradsControl(
        std::string name,
        std::vector<dealii::Tensor<1, targetdim> > &values) const
      {
        this->GetGrads<targetdim>(this->GetFEFaceValuesControl(), name,
                                  values);
      }
      /***********************************************************************/

      template<int dim>
      template<int targetdim>
      void
      Network_FaceDataContainerInternal<dim>::GetFaceGradsControl(
        std::string name,
        std::vector<std::vector<dealii::Tensor<1, targetdim> > > &values) const
      {
        this->GetGrads<targetdim>(this->GetFEFaceValuesControl(), name,
                                  values);
      }

      /*********************************************/
      template<int dim>
      void
      Network_FaceDataContainerInternal<dim>::GetNbrFaceValuesState(
        std::string name, std::vector<double> &values) const
      {
        this->GetValues(this->GetNbrFEFaceValuesState(), name, values);
      }
      /*********************************************/
      template<int dim>
      void
      Network_FaceDataContainerInternal<dim>::GetNbrFaceValuesState(
        std::string name, std::vector<Vector<double> > &values) const
      {
        this->GetValues(this->GetNbrFEFaceValuesState(), name, values);

      }

      /*********************************************/
      template<int dim>
      void
      Network_FaceDataContainerInternal<dim>::GetNbrFaceValuesControl(
        std::string name, std::vector<double> &values) const
      {
        this->GetValues(this->GetNbrFEFaceValuesControl(), name, values);
      }

      /*********************************************/
      template<int dim>
      void
      Network_FaceDataContainerInternal<dim>::GetNbrFaceValuesControl(
        std::string name, std::vector<Vector<double> > &values) const
      {
        this->GetValues(this->GetNbrFEFaceValuesControl(), name, values);
      }

      /*********************************************/
      template<int dim>
      template<int targetdim>
      void
      Network_FaceDataContainerInternal<dim>::GetNbrFaceGradsState(
        std::string name, std::vector<Tensor<1, targetdim> > &values) const
      {
        this->GetGrads<targetdim>(this->GetNbrFEFaceValuesState(), name,
                                  values);
      }

      /*********************************************/
      template<int dim>
      template<int targetdim>
      void
      Network_FaceDataContainerInternal<dim>::GetNbrFaceGradsState(
        std::string name,
        std::vector<std::vector<Tensor<1, targetdim> > > &values) const
      {
        this->GetGrads<targetdim>(this->GetNbrFEFaceValuesState(), name,
                                  values);
      }

      /***********************************************************************/

      template<int dim>
      template<int targetdim>
      void
      Network_FaceDataContainerInternal<dim>::GetNbrFaceGradsControl(
        std::string name, std::vector<Tensor<1, targetdim> > &values) const
      {
        this->GetGrads<targetdim>(this->GetNbrFEFaceValuesControl(), name,
                                  values);
      }
      /***********************************************************************/

      template<int dim>
      template<int targetdim>
      void
      Network_FaceDataContainerInternal<dim>::GetNbrFaceGradsControl(
        std::string name,
        std::vector<std::vector<Tensor<1, targetdim> > > &values) const
      {
        this->GetGrads<targetdim>(this->GetNbrFEFaceValuesControl(), name,
                                  values);
      }

      /***********************************************************************/

      /***********************************************************************/
      template<int dim>
      void
      Network_FaceDataContainerInternal<dim>::GetValues(
        const dealii::FEFaceValuesBase<dim> &fe_values, std::string name,
        std::vector<double> &values) const
      {
        typename std::map<std::string, const dealii::BlockVector<double> *>::const_iterator it =
          this->GetDomainValues().find(name);
        if (it == this->GetDomainValues().end())
          {
            throw DOpEException("Did not find " + name,
                                "ElementDataContainer::GetValues");
          }
        fe_values.get_function_values(it->second->block(pipe_), values);
      }

      /***********************************************************************/
      template<int dim>
      void
      Network_FaceDataContainerInternal<dim>::GetValues(
        const dealii::FEFaceValuesBase<dim> &fe_values, std::string name,
        std::vector<dealii::Vector<double> > &values) const
      {
        typename std::map<std::string, const dealii::BlockVector<double> *>::const_iterator it =
          this->GetDomainValues().find(name);
        if (it == this->GetDomainValues().end())
          {
            throw DOpEException("Did not find " + name,
                                "ElementDataContainer::GetValues");
          }
        fe_values.get_function_values(it->second->block(pipe_), values);
      }

      /***********************************************************************/

      template<int dim>
      template<int targetdim>
      void
      Network_FaceDataContainerInternal<dim>::GetGrads(
        const dealii::FEFaceValuesBase<dim> &fe_values, std::string name,
        std::vector<dealii::Tensor<1, targetdim> > &values) const
      {
        typename std::map<std::string, const dealii::BlockVector<double> *>::const_iterator it =
          this->GetDomainValues().find(name);
        if (it == this->GetDomainValues().end())
          {
            throw DOpEException("Did not find " + name,
                                "Network_FaceDataContainerInternal::GetGrads");
          }
        fe_values.get_function_gradients(it->second->block(pipe_), values);
      }

      /***********************************************************************/

      template<int dim>
      template<int targetdim>
      void
      Network_FaceDataContainerInternal<dim>::GetGrads(
        const dealii::FEFaceValuesBase<dim> &fe_values, std::string name,
        std::vector<std::vector<dealii::Tensor<1, targetdim> > > &values) const
      {
        typename std::map<std::string, const dealii::BlockVector<double> *>::const_iterator it =
          this->GetDomainValues().find(name);
        if (it == this->GetDomainValues().end())
          {
            throw DOpEException("Did not find " + name,
                                "Network_FaceDataContainerInternal::GetGrads");
          }
        fe_values.get_function_gradients(it->second->block(pipe_), values);
      }

      /***********************************************************************/


      /***********************************************************************/

    }
  }
}

#endif /* NETWORK_FACEDATACONTAINER_INTERNAL_H_ */
