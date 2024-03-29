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

#ifndef ELEMENTDATACONTAINER_INTERNAL_H_
#define ELEMENTDATACONTAINER_INTERNAL_H_

#include <deal.II/lac/vector.h>

#include <wrapper/fevalues_wrapper.h>
#include <include/dopeexception.h>
#include <sstream>

namespace DOpE
{
  namespace edcinternal
  {
    /**
     * This class houses all the functionality which is shared between
     * the ElementDataContainer for normal and hp::DoFHandlers.
     *
     * @template VECTOR     Type of the vector we use in our computations
     *                      (i.e. Vector<double> or BlockVector<double>)
     * @template dim        The dimension of the integral we are actually
     *                      interested in.
     */
    template<typename VECTOR, int dim>
    class ElementDataContainerInternal
    {
    public:
      template<typename STH>
      ElementDataContainerInternal(
        const std::map<std::string, const dealii::Vector<double>*> &param_values,
        const std::map<std::string, const VECTOR *> &domain_values,
        STH &sth,
        const typename Triangulation<dim>::cell_iterator element_iter,
        bool need_vertices
      );

      virtual
      ~ElementDataContainerInternal()
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
      const std::map<std::string, const VECTOR *> &
      GetDomainValues() const
      {
        return domain_values_;
      }

      virtual const DOpEWrapper::FEValues<dim> &
      GetFEValuesState() const = 0;

      virtual const DOpEWrapper::FEValues<dim> &
      GetFEValuesControl() const = 0;

      /*********************************************************************/
      /**
       * Return a triangulation iterator to the current element for the state.
       */
      const typename  Triangulation<dim>::cell_iterator
      GetElement() const;


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

      /*
       * Returns the number of neighbouring elements to the vertex located at the given point
       */
      inline unsigned int GetNNeighbourElementsOfVertex(const Point<dim> &p) const
      {
        if ( !has_vertices_)
          {
            throw DOpEException("No Vertex information prepared. Have you set HasVertices() in the PDE?",
                                "ElementDataContainerInternal::GetNNeighbourElementsOfVertex");
          }
        const typename Triangulation<dim>::cell_iterator element = GetElement();
        for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_cell; i++)
          {
            if ( p == element->vertex(i))
              {
                return (*n_neighbour_to_vertex_)[element->vertex_index(i)];
              }
          }
        std::stringstream out;
        out <<"Could not find Point "<<p<<" !";
        throw DOpEException(out.str(),
                            "ElementDataContainerInternal::GetNNeighbourElementsOfVertex");
      }

      void ReInit(typename Triangulation<dim>::cell_iterator element_iter)
      {
        element_iter_ = element_iter;
      }
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
      const std::map<std::string, const VECTOR *> &domain_values_;
      const std::vector<unsigned int> *n_neighbour_to_vertex_;
      typename Triangulation<dim>::cell_iterator element_iter_;
      bool has_vertices_;
    };

    /**********************************************************************/
    template<typename VECTOR, int dim>
    template<typename STH>
    ElementDataContainerInternal<VECTOR, dim>::ElementDataContainerInternal(
      const std::map<std::string, const dealii::Vector<double>*> &param_values,
      const std::map<std::string, const VECTOR *> &domain_values,
      STH &sth,
      const typename Triangulation<dim>::cell_iterator element_iter,
      bool need_vertices)
      : param_values_(param_values), domain_values_(domain_values),
        n_neighbour_to_vertex_(NULL), element_iter_(element_iter)
    {
      has_vertices_ = false;
      if (need_vertices)
        {
          n_neighbour_to_vertex_ = sth.GetNNeighbourElements();
          has_vertices_ = true;
        }
    }

    template<typename VECTOR, int dim>
    void
    ElementDataContainerInternal<VECTOR, dim>::GetParamValues(std::string name,
                                                              dealii::Vector<double> &value) const
    {
      const auto it = param_values_.find (name);
      if (it == param_values_.end())
        {
          throw DOpEException("Did not find " + name,
                              "ElementDataContainerInternal::GetParamValues");
        }
      value = *(it->second);
    }

    /*********************************************/
    template<typename VECTOR, int dim>
    void
    ElementDataContainerInternal<VECTOR, dim>::GetValuesState(std::string name,
                                                              std::vector<double> &values) const
    {
      this->GetValues(this->GetFEValuesState(), name, values);
    }
    /*********************************************/
    template<typename VECTOR, int dim>
    void
    ElementDataContainerInternal<VECTOR, dim>::GetValuesState(std::string name,
                                                              std::vector<dealii::Vector<double> > &values) const
    {
      this->GetValues(this->GetFEValuesState(), name, values);

    }


    /*********************************************/
    template<typename VECTOR, int dim>
    const typename Triangulation<dim>::cell_iterator
    ElementDataContainerInternal<VECTOR, dim>::GetElement() const
    {
      return element_iter_;
    }

    /*********************************************/
    template<typename VECTOR, int dim>
    void
    ElementDataContainerInternal<VECTOR, dim>::GetValuesControl(std::string name,
                                                                std::vector<double> &values) const
    {
      this->GetValues(this->GetFEValuesControl(), name, values);
    }

    /*********************************************/
    template<typename VECTOR, int dim>
    void
    ElementDataContainerInternal<VECTOR, dim>::GetValuesControl(std::string name,
                                                                std::vector<dealii::Vector<double> > &values) const
    {
      this->GetValues(this->GetFEValuesControl(), name, values);
    }

    /*********************************************/
    template<typename VECTOR, int dim>
    template<int targetdim>
    void
    ElementDataContainerInternal<VECTOR, dim>::GetGradsState(std::string name,
                                                             std::vector<dealii::Tensor<1, targetdim> > &values) const
    {
      this->GetGrads<targetdim>(this->GetFEValuesState(), name, values);
    }

    /*********************************************/
    template<typename VECTOR, int dim>
    template<int targetdim>
    void
    ElementDataContainerInternal<VECTOR, dim>::GetGradsState(
      std::string name,
      std::vector<std::vector<dealii::Tensor<1, targetdim> > > &values) const
    {
      this->GetGrads<targetdim>(this->GetFEValuesState(), name, values);
    }

    /***********************************************************************/

    template<typename VECTOR, int dim>
    template<int targetdim>
    void
    ElementDataContainerInternal<VECTOR, dim>::GetGradsControl(
      std::string name,
      std::vector<dealii::Tensor<1, targetdim> > &values) const
    {
      this->GetGrads<targetdim>(this->GetFEValuesControl(), name, values);
    }

    /***********************************************************************/

    template<typename VECTOR, int dim>
    template<int targetdim>
    void
    ElementDataContainerInternal<VECTOR, dim>::GetGradsControl(
      std::string name,
      std::vector<std::vector<dealii::Tensor<1, targetdim> > > &values) const
    {
      this->GetGrads<targetdim>(this->GetFEValuesControl(), name, values);
    }

    /***********************************************************************/

    template<typename VECTOR, int dim>
    template<int targetdim>
    void
    ElementDataContainerInternal<VECTOR, dim>::GetHessiansState(
      std::string name,
      std::vector<std::vector<dealii::Tensor<2, targetdim> > > &values) const
    {
      this->GetHessians<targetdim>(this->GetFEValuesState(), name, values);
    }

    /***********************************************************************/

    template<typename VECTOR, int dim>
    template<int targetdim>
    void
    ElementDataContainerInternal<VECTOR, dim>::GetHessiansState(
      std::string name,
      std::vector<dealii::Tensor<2, targetdim> > &values) const
    {
      this->GetHessians<targetdim>(this->GetFEValuesState(), name, values);
    }

    /***********************************************************************/

    template<typename VECTOR, int dim>
    template<int targetdim>
    void
    ElementDataContainerInternal<VECTOR, dim>::GetHessiansControl(
      std::string name,
      std::vector<std::vector<dealii::Tensor<2, targetdim> > > &values) const
    {
      this->GetHessians<targetdim>(this->GetFEValuesControl(), name, values);
    }

    /***********************************************************************/

    template<typename VECTOR, int dim>
    template<int targetdim>
    void
    ElementDataContainerInternal<VECTOR, dim>::GetHessiansControl(
      std::string name,
      std::vector<dealii::Tensor<2, targetdim> > &values) const
    {
      this->GetHessians<targetdim>(this->GetFEValuesControl(), name,
                                   values);
    }

    /***********************************************************************/
    template<typename VECTOR, int dim>
    void
    ElementDataContainerInternal<VECTOR, dim>::GetLaplaciansState(
      std::string name, std::vector<double> &values) const
    {
      this->GetLaplacians(this->GetFEValuesState(), name, values);
    }

    /***********************************************************************/
    template<typename VECTOR, int dim>
    void
    ElementDataContainerInternal<VECTOR, dim>::GetLaplaciansState(
      std::string name, std::vector<dealii::Vector<double> > &values) const
    {
      this->GetLaplacians(this->GetFEValuesState(), name, values);
    }
    /***********************************************************************/
    template<typename VECTOR, int dim>
    void
    ElementDataContainerInternal<VECTOR, dim>::GetLaplaciansControl(
      std::string name, std::vector<double> &values) const
    {
      this->GetLaplacians(this->GetFEValuesControl(), name, values);
    }

    /***********************************************************************/
    template<typename VECTOR, int dim>
    void
    ElementDataContainerInternal<VECTOR, dim>::GetLaplaciansControl(
      std::string name, std::vector<dealii::Vector<double> > &values) const
    {
      this->GetLaplacians(this->GetFEValuesControl(), name, values);
    }
    /***********************************************************************/
    template<typename VECTOR, int dim>
    void
    ElementDataContainerInternal<VECTOR, dim>::GetValues(
      const DOpEWrapper::FEValues<dim> &fe_values, std::string name,
      std::vector<double> &values) const
    {
      const auto it = this->GetDomainValues().find(name);
      if (it == this->GetDomainValues().end())
        {
          throw DOpEException("Did not find " + name,
                              "ElementDataContainer::GetValues");
        }
      fe_values.get_function_values(*(it->second), values);
    }

    /***********************************************************************/
    template<typename VECTOR, int dim>
    void
    ElementDataContainerInternal<VECTOR, dim>::GetValues(
      const DOpEWrapper::FEValues<dim> &fe_values, std::string name,
      std::vector<dealii::Vector<double> > &values) const
    {
      const auto it = this->GetDomainValues().find(name);
      if (it == this->GetDomainValues().end())
        {
          throw DOpEException("Did not find " + name,
                              "ElementDataContainer::GetValues");
        }
      fe_values.get_function_values(*(it->second), values);
    }

    /***********************************************************************/

    template<typename VECTOR, int dim>
    template<int targetdim>
    void
    ElementDataContainerInternal<VECTOR, dim>::GetGrads(
      const DOpEWrapper::FEValues<dim> &fe_values, std::string name,
      std::vector<dealii::Tensor<1, targetdim> > &values) const
    {
      const auto it = this->GetDomainValues().find(name);
      if (it == this->GetDomainValues().end())
        {
          throw DOpEException("Did not find " + name,
                              "ElementDataContainerInternal::GetGrads");
        }
      fe_values.get_function_gradients(*(it->second), values);
    }

    /***********************************************************************/

    template<typename VECTOR, int dim>
    template<int targetdim>
    void
    ElementDataContainerInternal<VECTOR, dim>::GetGrads(
      const DOpEWrapper::FEValues<dim> &fe_values,
      std::string name,
      std::vector<std::vector<dealii::Tensor<1, targetdim> > > &values) const
    {
      const auto it = this->GetDomainValues().find(name);
      if (it == this->GetDomainValues().end())
        {
          throw DOpEException("Did not find " + name,
                              "ElementDataContainerInternal::GetGrads");
        }
      fe_values.get_function_gradients(*(it->second), values);
    }

    /***********************************************************************/

    template<typename VECTOR, int dim>
    void
    ElementDataContainerInternal<VECTOR, dim>::GetLaplacians(
      const DOpEWrapper::FEValues<dim> &fe_values, std::string name,
      std::vector<double> &values) const
    {
      const auto it = this->GetDomainValues().find(name);
      if (it == this->GetDomainValues().end())
        {
          throw DOpEException("Did not find " + name,
                              "ElementDataContainerInternal::GetLaplacians");
        }
      fe_values.get_function_laplacians(*(it->second), values);
    }

    /***********************************************************************/

    template<typename VECTOR, int dim>
    void
    ElementDataContainerInternal<VECTOR, dim>::GetLaplacians(
      const DOpEWrapper::FEValues<dim> &fe_values, std::string name,
      std::vector<dealii::Vector<double> > &values) const
    {
      const auto it = this->GetDomainValues().find(name);
      if (it == this->GetDomainValues().end())
        {
          throw DOpEException("Did not find " + name,
                              "ElementDataContainerInternal::GetLaplacians");
        }
      fe_values.get_function_laplacians(*(it->second), values);
    }

    /***********************************************************************/

    template<typename VECTOR, int dim>
    template<int targetdim>
    void
    ElementDataContainerInternal<VECTOR, dim>::GetHessians(
      const DOpEWrapper::FEValues<dim> &fe_values,
      std::string name,
      std::vector<std::vector<dealii::Tensor<2, targetdim> > > &values) const
    {
      const auto it = this->GetDomainValues().find(name);
      if (it == this->GetDomainValues().end())
        {
          throw DOpEException("Did not find " + name,
                              "ElementDataContainerInternal::GetGrads");
        }
      fe_values.get_function_hessians(*(it->second), values);
    }

    /***********************************************************************/

    template<typename VECTOR, int dim>
    template<int targetdim>
    void
    ElementDataContainerInternal<VECTOR, dim>::GetHessians(
      const DOpEWrapper::FEValues<dim> &fe_values, std::string name,
      std::vector<dealii::Tensor<2, targetdim> > &values) const
    {
      const auto it = this->GetDomainValues().find(name);
      if (it == this->GetDomainValues().end())
        {
          throw DOpEException("Did not find " + name,
                              "ElementDataContainerInternal::GetGrads");
        }
      fe_values.get_function_hessians(*(it->second), values);
    }

  } //end of namespace edcinternal
}

#endif /* ElementDataContainer_INTERNAL_H_ */
