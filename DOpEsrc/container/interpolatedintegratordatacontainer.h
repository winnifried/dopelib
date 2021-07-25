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

#ifndef INTERPOLATEDINTEGRATORDATACONTAINER_H_
#define INTERPOLATEDINTEGRATORDATACONTAINER_H_

#include <container/integratordatacontainer.h>
#include <container/elementdatacontainer.h>
#include <container/facedatacontainer.h>
#include <container/interpolatedfacedatacontainer.h>
#include <container/interpolatedelementdatacontainer.h>

using namespace dealii;

namespace DOpE
{
  /**
   * This class manages the different kind of element- and facedatacontainers
   * needed in the integrator when InterpolatedFEValues are used.
   */
#if DEAL_II_VERSION_GTE(9,3,0)
   template<bool DH, typename QUADRATURE, typename FACEQUADRATURE,
#else
   template<template<int, int> class DH, typename QUADRATURE, typename FACEQUADRATURE,
#endif
	    typename VECTOR, int dim>
   class InterpolatedIntegratorDataContainer : public IntegratorDataContainer<DH, QUADRATURE, FACEQUADRATURE, VECTOR, dim>
   {
      public:
      InterpolatedIntegratorDataContainer(
		const FEValuesExtractors::Vector selected_component,
		const Mapping<dim> 		&map,
		const FiniteElement<dim>	&fe_interpolate,
		const QUADRATURE &quad, const FACEQUADRATURE &face_quad) :
		IntegratorDataContainer<DH, QUADRATURE, FACEQUADRATURE, VECTOR, dim>
		(quad,  face_quad),
		selected_component_(selected_component), map_(map), fe_interpolate_(fe_interpolate), interp_edc_(NULL), interp_fdc_(NULL)
      {
      }

     ~InterpolatedIntegratorDataContainer()
     {
       if (interp_fdc_ != NULL)
       {
	 delete interp_fdc_;
	 interp_fdc_ = NULL;
       }
       if (interp_edc_ != NULL)
       {
	 delete interp_edc_;
	 interp_edc_ = NULL;
       }
     }

    /**
      * Initializes the ElementDataContainer. See the documentation there.
     */
     template<typename STH>
     void
     InitializeEDC(const QUADRATURE &quad, UpdateFlags update_flags,
		   STH &sth,
#if DEAL_II_VERSION_GTE(9,3,0)
		   const std::vector<typename DOpEWrapper::DoFHandler<dim>::active_cell_iterator> &element,
#else
		   const std::vector<typename DOpEWrapper::DoFHandler<dim, DH>::active_cell_iterator> &element,
#endif
		   const std::map<std::string, const Vector<double>*> &param_values,
		   const std::map<std::string, const VECTOR*> &domain_values,
		   bool need_vertices)
     {
       if (interp_edc_ != NULL)
	 delete interp_edc_;

       interp_edc_ = new InterpolatedElementDataContainer<DH, VECTOR, dim>(selected_component_,
									   map_, 
									   fe_interpolate_,
									   quad, 
									   update_flags,
									   sth,
									   element,
									   param_values,
									   domain_values,
									   need_vertices);
     }
     
     /**
      * Initializes the ElementDataContainer. See the documentation there.
      * This one uses the previously given quadrature.
      */
     template<typename STH>
     void
     InitializeEDC(UpdateFlags update_flags, STH &sth,
#if DEAL_II_VERSION_GTE(9,3,0)
		   const std::vector<typename DOpEWrapper::DoFHandler<dim>::active_cell_iterator> &element,
#else
		   const std::vector<typename DOpEWrapper::DoFHandler<dim, DH>::active_cell_iterator>& element,
#endif
		   const std::map<std::string, const Vector<double>*> &param_values,
		   const std::map<std::string, const VECTOR *> &domain_values,
		   bool need_vertices)
     {
       InitializeEDC(this->GetQuad(), update_flags, sth, element, param_values,
		     domain_values, need_vertices);
     }

         /**
     * Initializes the FaceDataContainer. See the documentation there.
     */
    template<typename STH>
    void
    InitializeFDC(const FACEQUADRATURE &fquad, UpdateFlags update_flags,
                  STH &sth,
                  const std::vector<
#if DEAL_II_VERSION_GTE(9,3,0)
		  typename DOpEWrapper::DoFHandler<dim>::active_cell_iterator>& element,
#else
		  typename DOpEWrapper::DoFHandler<dim, DH>::active_cell_iterator>& element,
#endif
                  const std::map<std::string, const Vector<double>*> &param_values,
                  const std::map<std::string, const VECTOR *> &domain_values,
                  bool need_interfaces = false)
    {
      if (interp_fdc_ != NULL)
	 delete interp_fdc_;
      //FIXME: FaceData should only depend on facequadrature not element quadarture
      interp_fdc_ = new InterpolatedFaceDataContainer<DH, VECTOR, dim>(selected_component_,
								       map_, 
								       fe_interpolate_,
								       this->GetQuad(),
								       fquad,
								       update_flags, sth,
								       element, param_values,
								       domain_values,
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
#if DEAL_II_VERSION_GTE(9,3,0)
		  typename DOpEWrapper::DoFHandler<dim>::active_cell_iterator>& element,
#else
		  typename DOpEWrapper::DoFHandler<dim, DH>::active_cell_iterator>& element,
#endif
                  const std::map<std::string, const Vector<double>*> &param_values,
                  const std::map<std::string, const VECTOR *> &domain_values,
                  bool need_interfaces = false)
    {
      InitializeFDC(this->GetFaceQuad(), update_flags, sth, element, param_values,
                    domain_values, need_interfaces);
    }

     
     InterpolatedElementDataContainer<DH, VECTOR, dim> &
     GetElementDataContainer() const
     {
       if (interp_edc_ != NULL)
         return *interp_edc_;
       else
         throw DOpEException("Pointer has to be initialized.",
			     "IntegratorDataContainer::GetElementDataContainer");
     }
     InterpolatedFaceDataContainer<DH, VECTOR, dim> &
     GetFaceDataContainer() const
     {
       if (interp_fdc_ != NULL)
         return *interp_fdc_;
       else
         throw DOpEException("Pointer has to be initialized.",
			     "IntegratorDataContainer::GetFaceDataContainer");
     }
     
   private:
     const FEValuesExtractors::Vector selected_component_;
     const Mapping<dim>		&map_;
     const FiniteElement<dim>	&fe_interpolate_;
     InterpolatedElementDataContainer<DH, VECTOR, dim> *interp_edc_;
     InterpolatedFaceDataContainer<DH, VECTOR, dim> *interp_fdc_;
     
   };
}

#endif
