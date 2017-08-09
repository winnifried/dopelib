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

#ifndef NETWORK_INTEGRATORDATACONTAINER_H_
#define NETWORK_INTEGRATORDATACONTAINER_H_

#include <deal.II/base/quadrature.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/hp/q_collection.h>
#include <deal.II/hp/dof_handler.h>
#include <deal.II/lac/vector.h>

#include <wrapper/dofhandler_wrapper.h>
#include <network/network_elementdatacontainer.h>
#include <network/network_facedatacontainer.h>
#include <include/dopeexception.h>

namespace DOpE
{
  namespace Networks
  {
    /**
     * This class manages the different kind of element- and facedatacontainers
     * needed in the integrator.
     */
    template<template<int, int> class DH, typename QUADRATURE, typename FACEQUADRATURE,
             typename VECTOR, int dim>
    class Network_IntegratorDataContainer
    {
    public:
      Network_IntegratorDataContainer(const QUADRATURE &quad,
                                      const FACEQUADRATURE &face_quad)
        : quad_(&quad), face_quad_(&face_quad), fdc_(NULL), edc_(NULL)
      {
      }

      ~Network_IntegratorDataContainer()
      {
        if (fdc_ != NULL)
          {
            delete fdc_;
            fdc_ = NULL;
          }
        if (edc_ != NULL)
          {
            delete edc_;
            edc_ = NULL;
          }
      }

      /**
       * Initializes the FaceDataContainer. See the documentation there.
       */
      template<typename STH>
      void
      InitializeFDC(unsigned int pipe, unsigned int n_pipes, unsigned int n_comp,
                    const FACEQUADRATURE &fquad, UpdateFlags update_flags,
                    STH &sth,
                    const std::vector<
                    typename DOpEWrapper::DoFHandler<dim, DH>::active_cell_iterator>& element,
                    const std::map<std::string, const Vector<double>*> &param_values,
                    const std::map<std::string, const dealii::BlockVector<double> *> &domain_values,
                    bool need_interfaces = false)
      {
        delete fdc_;
        fdc_ = new Network_FaceDataContainer<DH, VECTOR, dim>(pipe, n_pipes, n_comp, fquad,
                                                              update_flags, sth, element, param_values, domain_values,
                                                              need_interfaces);
      }

      /**
       * Initializes the FaceDataContainer. See the documentation there.
       * This one uses the previously given facequadrature.
       */
      template<typename STH>
      void
      InitializeFDC(unsigned int pipe, unsigned int n_pipes, unsigned int n_comp,
                    UpdateFlags update_flags, STH &sth,
                    const std::vector<
                    typename DOpEWrapper::DoFHandler<dim, DH>::active_cell_iterator>& element,
                    const std::map<std::string, const Vector<double>*> &param_values,
                    const std::map<std::string, const dealii::BlockVector<double> *> &domain_values,
                    bool need_interfaces = false)
      {
        InitializeFDC(pipe, n_pipes, n_comp, GetFaceQuad(), update_flags, sth, element, param_values,
                      domain_values, need_interfaces);
      }

      /**
       * Initializes the ElementDataContainer. See the documentation there.
       */
      template<typename STH>
      void
      InitializeEDC(unsigned int pipe, const QUADRATURE &quad, UpdateFlags update_flags,
                    STH &sth,
                    const std::vector<
                    typename DOpEWrapper::DoFHandler<dim, DH>::active_cell_iterator>& element,
                    const std::map<std::string, const Vector<double>*> &param_values,
                    const std::map<std::string, const dealii::BlockVector<double> *> &domain_values)
      {
        if (edc_ != NULL)
          delete edc_;
        edc_ = new Network_ElementDataContainer<DH, VECTOR, dim>(pipe, quad,
                                                                 update_flags, sth, element, param_values, domain_values);
      }

      /**
       * Initializes the ElementDataContainer. See the documentation there.
       * This one uses the previously given quadrature.
       */
      template<typename STH>
      void
      InitializeEDC(unsigned int pipe, UpdateFlags update_flags, STH &sth,
                    const std::vector<
                    typename DOpEWrapper::DoFHandler<dim, DH>::active_cell_iterator>& element,
                    const std::map<std::string, const Vector<double>*> &param_values,
                    const std::map<std::string, const dealii::BlockVector<double> *> &domain_values)
      {
        InitializeEDC(pipe, GetQuad(), update_flags, sth, element, param_values,
                      domain_values);
      }

      const QUADRATURE &
      GetQuad() const
      {
        return *quad_;
      }

      const FACEQUADRATURE &
      GetFaceQuad() const
      {
        return *face_quad_;
      }

      Network_FaceDataContainer<DH, VECTOR, dim> &
      GetFaceDataContainer() const
      {
        if (fdc_ != NULL)
          return *fdc_;
        else
          throw DOpEException("Pointer has to be initialized.",
                              "Network_IntegratorDataContainer::GetFaceDataContainer");
      }

      Network_ElementDataContainer<DH, VECTOR, dim> &
      GetElementDataContainer() const
      {
        if (edc_ != NULL)
          return *edc_;
        else
          throw DOpEException("Pointer has to be initialized.",
                              "Network_IntegratorDataContainer::GetElementDataContainer");
      }

    private:
      QUADRATURE const *quad_;
      FACEQUADRATURE const *face_quad_;
      Network_FaceDataContainer<DH, VECTOR, dim> *fdc_;
      Network_ElementDataContainer<DH, VECTOR, dim> *edc_;
    };

  }
} //end of namespace

#endif /* NETWORK_INTEGRATORDATACONTAINER_H_ */
