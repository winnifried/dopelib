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

#ifndef INTEGRATORDATACONTAINER_H_
#define INTEGRATORDATACONTAINER_H_

#include <deal.II/base/quadrature.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/hp/q_collection.h>
#include <deal.II/hp/dof_handler.h>
#include <deal.II/lac/vector.h>

#include <wrapper/dofhandler_wrapper.h>
#include <container/elementdatacontainer.h>
#include <container/facedatacontainer.h>
#include <container/multimesh_elementdatacontainer.h>
#include <container/multimesh_facedatacontainer.h>
#include <include/dopeexception.h>

namespace DOpE
{
  /**
   * This class manages the different kind of element- and facedatacontainers
   * needed in the integrator.
   */
  template<template<int, int> class DH, typename QUADRATURE, typename FACEQUADRATURE,
           typename VECTOR, int dim>
  class IntegratorDataContainer
  {
  public:
    IntegratorDataContainer(const QUADRATURE &quad,
                            const FACEQUADRATURE &face_quad)
      : quad_(&quad), face_quad_(&face_quad), fdc_(NULL), edc_(NULL), mm_fdc_(
        NULL), mm_edc_(NULL)
    {
    }

    ~IntegratorDataContainer()
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
      if (mm_fdc_ != NULL)
        {
          delete mm_fdc_;
          mm_fdc_ = NULL;
        }
      if (mm_edc_ != NULL)
        {
          delete mm_edc_;
          mm_edc_ = NULL;
        }
    }

    /**
     * Initializes the FaceDataContainer. See the documentation there.
     */
    template<typename STH>
    void
    InitializeFDC(const FACEQUADRATURE &fquad, UpdateFlags update_flags,
                  STH &sth,
                  const std::vector<
                  typename DOpEWrapper::DoFHandler<dim, DH>::active_cell_iterator>& element,
                  const std::map<std::string, const Vector<double>*> &param_values,
                  const std::map<std::string, const VECTOR *> &domain_values,
                  bool need_interfaces = false)
    {
      delete fdc_;
      fdc_ = new FaceDataContainer<DH, VECTOR, dim>(fquad,
                                                    update_flags, sth, element, param_values, domain_values,
                                                    need_interfaces);
    }

    /**
     * Initializes the FaceDataContainer. See the documentation there.
     * This one uses the previously given facequadrature.
     */
    template<typename STH>
    void
    InitializeFDC(UpdateFlags update_flags, STH &sth,
                  const std::vector<
                  typename DOpEWrapper::DoFHandler<dim, DH>::active_cell_iterator>& element,
                  const std::map<std::string, const Vector<double>*> &param_values,
                  const std::map<std::string, const VECTOR *> &domain_values,
                  bool need_interfaces = false)
    {
      InitializeFDC(GetFaceQuad(), update_flags, sth, element, param_values,
                    domain_values, need_interfaces);
    }

    /**
     * Initializes the ElementDataContainer. See the documentation there.
     */
    template<typename STH>
    void
    InitializeEDC(const QUADRATURE &quad, UpdateFlags update_flags,
                  STH &sth,
                  const std::vector<
                  typename DOpEWrapper::DoFHandler<dim, DH>::active_cell_iterator>& element,
                  const std::map<std::string, const Vector<double>*> &param_values,
                  const std::map<std::string, const VECTOR *> &domain_values)
    {
      if (edc_ != NULL)
        delete edc_;
      edc_ = new ElementDataContainer<DH, VECTOR, dim>(quad,
                                                       update_flags, sth, element, param_values, domain_values);
    }

    /**
     * Initializes the ElementDataContainer. See the documentation there.
     * This one uses the previously given quadrature.
     */
    template<typename STH>
    void
    InitializeEDC(UpdateFlags update_flags, STH &sth,
                  const std::vector<
                  typename DOpEWrapper::DoFHandler<dim, DH>::active_cell_iterator>& element,
                  const std::map<std::string, const Vector<double>*> &param_values,
                  const std::map<std::string, const VECTOR *> &domain_values)
    {
      InitializeEDC(GetQuad(), update_flags, sth, element, param_values,
                    domain_values);
    }

    /**
     * Initializes the MMFaceDataContainer. See the documentation there.
     */
    template<typename STH>
    void
    InitializeMMFDC(UpdateFlags update_flags, STH &sth,
                    const typename std::vector<typename DH<dim, dim>::cell_iterator> &element,
                    const typename std::vector<
                    typename dealii::Triangulation<dim>::cell_iterator>& tria_element,
                    const std::map<std::string, const Vector<double>*> &param_values,
                    const std::map<std::string, const VECTOR *> &domain_values,
                    bool need_interfaces = false)
    {
      if (mm_fdc_ != NULL)
        delete mm_fdc_;
      mm_fdc_ = new Multimesh_FaceDataContainer<DH, VECTOR, dim>(
        GetFaceQuad(), update_flags, sth, element, tria_element, param_values,
        domain_values, need_interfaces);
    }

    /**
     * Initializes the MMElementDataContainer. See the documentation there.
     */
    template<typename STH>
    void
    InitializeMMEDC(UpdateFlags update_flags, STH &sth,
                    const typename std::vector<typename DH<dim, dim>::cell_iterator> &element,
                    const typename std::vector<
                    typename dealii::Triangulation<dim>::cell_iterator>& tria_element,
                    const std::map<std::string, const Vector<double>*> &param_values,
                    const std::map<std::string, const VECTOR *> &domain_values)
    {
      if (mm_edc_ != NULL)
        delete mm_edc_;
      mm_edc_ = new Multimesh_ElementDataContainer<DH, VECTOR, dim>(
        GetQuad(), update_flags, sth, element, tria_element, param_values,
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

    FaceDataContainer<DH, VECTOR, dim> &
    GetFaceDataContainer() const
    {
      if (fdc_ != NULL)
        return *fdc_;
      else
        throw DOpEException("Pointer has to be initialized.",
                            "IntegratorDataContainer::GetFaceDataContainer");
    }

    ElementDataContainer<DH, VECTOR, dim> &
    GetElementDataContainer() const
    {
      if (edc_ != NULL)
        return *edc_;
      else
        throw DOpEException("Pointer has to be initialized.",
                            "IntegratorDataContainer::GetElementDataContainer");
    }

    Multimesh_FaceDataContainer<DH, VECTOR, dim> &
    GetMultimeshFaceDataContainer() const
    {
      if (mm_fdc_ != NULL)
        return *mm_fdc_;
      else
        throw DOpEException("Pointer has to be initialized.",
                            "IntegratorDataContainer::GetMultimeshFaceDataContainer");
    }

    Multimesh_ElementDataContainer<DH, VECTOR, dim> &
    GetMultimeshElementDataContainer() const
    {
      if (mm_edc_ != NULL)
        return *mm_edc_;
      else
        throw DOpEException("Pointer has to be initialized.",
                            "IntegratorDataContainer::GetMultimeshElementDataContainer");
    }
  private:
    QUADRATURE const *quad_;
    FACEQUADRATURE const *face_quad_;
    FaceDataContainer<DH, VECTOR, dim> *fdc_;
    ElementDataContainer<DH, VECTOR, dim> *edc_;
    Multimesh_FaceDataContainer<DH, VECTOR, dim> *mm_fdc_;
    Multimesh_ElementDataContainer<DH, VECTOR, dim> *mm_edc_;
  };

} //end of namespace

#endif /* INTEGRATORDATACONTAINER_H_ */
