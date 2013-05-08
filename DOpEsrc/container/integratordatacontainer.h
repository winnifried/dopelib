/**
*
* Copyright (C) 2012 by the DOpElib authors
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

/*
 * integratordatacontainer.h
 *
 *  Created on: 18.01.2012
 *      Author: cgoll
 */

#ifndef _INTEGRATORDATACONTAINER_H_
#define _INTEGRATORDATACONTAINER_H_

#include <base/quadrature.h>
#include <dofs/dof_handler.h>
#include <hp/q_collection.h>
#include <hp/dof_handler.h>
#include <lac/vector.h>

#include "dofhandler_wrapper.h"
#include "celldatacontainer.h"
#include "facedatacontainer.h"
#include "multimesh_celldatacontainer.h"
#include "multimesh_facedatacontainer.h"
#include "dopeexception.h"

namespace DOpE
{
  /**
   * This class manages the different kind of cell- and facedatacontainers
   * needed in the integrator.
   */
  template<template<int, int> class DH, typename QUADRATURE, typename FACEQUADRATURE,
      typename VECTOR, int dim>
    class IntegratorDataContainer
    {
      public:
        IntegratorDataContainer(const QUADRATURE& quad,
            const FACEQUADRATURE & face_quad)
            : _quad(&quad), _face_quad(&face_quad), _fdc(NULL), _cdc(NULL), _mm_fdc(
                NULL), _mm_cdc(NULL)
        {
        }

        ~IntegratorDataContainer()
        {
          if (_fdc != NULL)
          {
            delete _fdc;
            _fdc = NULL;
          }
          if (_cdc != NULL)
          {
            delete _cdc;
            _cdc = NULL;
          }
          if (_mm_fdc != NULL)
          {
            delete _mm_fdc;
            _mm_fdc = NULL;
          }
          if (_mm_cdc != NULL)
          {
            delete _mm_cdc;
            _mm_cdc = NULL;
          }
        }

        /**
         * Initializes the FaceDataContainer. See the documentation there.
         */
        template<typename STH>
          void
          InitializeFDC(const FACEQUADRATURE& fquad, UpdateFlags update_flags,
              STH& sth,
              const std::vector<
                  typename DOpEWrapper::DoFHandler<dim, DH>::active_cell_iterator>& cell,
              const std::map<std::string, const Vector<double>*> &param_values,
              const std::map<std::string, const VECTOR*> &domain_values,
              bool need_interfaces = false)
          {
              delete _fdc;
            _fdc = new FaceDataContainer<DH, VECTOR, dim>(fquad,
                update_flags, sth, cell, param_values, domain_values,
                need_interfaces);
          }

        /**
         * Initializes the FaceDataContainer. See the documentation there.
         * This one uses the previously given facequadrature.
         */
        template<typename STH>
          void
          InitializeFDC(UpdateFlags update_flags, STH& sth,
              const std::vector<
                  typename DOpEWrapper::DoFHandler<dim, DH>::active_cell_iterator>& cell,
              const std::map<std::string, const Vector<double>*> &param_values,
              const std::map<std::string, const VECTOR*> &domain_values,
              bool need_interfaces = false)
          {
            InitializeFDC(GetFaceQuad(), update_flags, sth, cell, param_values,
                domain_values, need_interfaces);
          }

        /**
         * Initializes the CellDataContainer. See the documentation there.
         */
        template<typename STH>
          void
          InitializeCDC(const QUADRATURE& quad, UpdateFlags update_flags,
              STH& sth,
              const std::vector<
                  typename DOpEWrapper::DoFHandler<dim, DH>::active_cell_iterator>& cell,
              const std::map<std::string, const Vector<double>*> &param_values,
              const std::map<std::string, const VECTOR*> &domain_values)
          {
            if (_cdc != NULL)
              delete _cdc;
            _cdc = new CellDataContainer<DH, VECTOR, dim>(quad,
                update_flags, sth, cell, param_values, domain_values);
          }

        /**
         * Initializes the CellDataContainer. See the documentation there.
         * This one uses the previously given quadrature.
         */
        template<typename STH>
          void
          InitializeCDC(UpdateFlags update_flags, STH& sth,
              const std::vector<
                  typename DOpEWrapper::DoFHandler<dim, DH>::active_cell_iterator>& cell,
              const std::map<std::string, const Vector<double>*> &param_values,
              const std::map<std::string, const VECTOR*> &domain_values)
          {
            InitializeCDC(GetQuad(), update_flags, sth, cell, param_values,
                domain_values);
          }

        /**
         * Initializes the MMFaceDataContainer. See the documentation there.
         */
        template<typename STH>
          void
          InitializeMMFDC(UpdateFlags update_flags, STH& sth,
              const typename std::vector<typename DH<dim, dim>::cell_iterator>& cell,
              const typename std::vector<
                  typename dealii::Triangulation<dim>::cell_iterator>& tria_cell,
              const std::map<std::string, const Vector<double>*> &param_values,
              const std::map<std::string, const VECTOR*> &domain_values,
              bool need_interfaces = false)
          {
            if (_mm_fdc != NULL)
              delete _mm_fdc;
            _mm_fdc = new Multimesh_FaceDataContainer<DH, VECTOR, dim>(
                GetFaceQuad(), update_flags, sth, cell, tria_cell, param_values,
                domain_values, need_interfaces);
          }

        /**
         * Initializes the MMCellDataContainer. See the documentation there.
         */
        template<typename STH>
          void
          InitializeMMCDC(UpdateFlags update_flags, STH& sth,
              const typename std::vector<typename DH<dim, dim>::cell_iterator>& cell,
              const typename std::vector<
                  typename dealii::Triangulation<dim>::cell_iterator>& tria_cell,
              const std::map<std::string, const Vector<double>*> &param_values,
              const std::map<std::string, const VECTOR*> &domain_values)
          {
            if (_mm_cdc != NULL)
              delete _mm_cdc;
            _mm_cdc = new Multimesh_CellDataContainer<DH, VECTOR, dim>(
                GetQuad(), update_flags, sth, cell, tria_cell, param_values,
                domain_values);
          }

        const QUADRATURE&
        GetQuad() const
        {
          return *_quad;
        }

        const FACEQUADRATURE&
        GetFaceQuad() const
        {
          return *_face_quad;
        }

        FaceDataContainer<DH, VECTOR, dim>&
        GetFaceDataContainer() const
        {
          if (_fdc != NULL)
            return *_fdc;
          else
            throw DOpEException("Pointer has to be initialized.",
                "IntegratorDataContainer::GetFaceDataContainer");
        }

        CellDataContainer<DH, VECTOR, dim>&
        GetCellDataContainer() const
        {
          if (_cdc != NULL)
            return *_cdc;
          else
            throw DOpEException("Pointer has to be initialized.",
                "IntegratorDataContainer::GetCellDataContainer");
        }

        Multimesh_FaceDataContainer<DH, VECTOR, dim>&
        GetMultimeshFaceDataContainer() const
        {
          if (_mm_fdc != NULL)
            return *_mm_fdc;
          else
            throw DOpEException("Pointer has to be initialized.",
                "IntegratorDataContainer::GetMultimeshFaceDataContainer");
        }

        Multimesh_CellDataContainer<DH, VECTOR, dim>&
        GetMultimeshCellDataContainer() const
        {
          if (_mm_cdc != NULL)
            return *_mm_cdc;
          else
            throw DOpEException("Pointer has to be initialized.",
                "IntegratorDataContainer::GetMultimeshCellDataContainer");
        }
      private:
        QUADRATURE const* _quad;
        FACEQUADRATURE const* _face_quad;
        FaceDataContainer<DH, VECTOR, dim>* _fdc;
        CellDataContainer<DH, VECTOR, dim>* _cdc;
        Multimesh_FaceDataContainer<DH, VECTOR, dim>* _mm_fdc;
        Multimesh_CellDataContainer<DH, VECTOR, dim>* _mm_cdc;
    };

} //end of namespace

#endif /* INTEGRATORDATACONTAINER_H_ */
