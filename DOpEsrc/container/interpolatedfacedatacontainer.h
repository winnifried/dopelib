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

#ifndef INTERPOLATEDFACEDATACONTAINER_H_
#define INTERPOLATEDFACEDATACONTAINER_H_

#include <container/facedatacontainer.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <include/interpolated_fe_face_values.h>

using namespace dealii;
using namespace std;

namespace DOpE
{
#if DEAL_II_VERSION_GTE(9,3,0)
  template<bool DH, typename VECTOR, int dim>
#else
   template<template<int, int> class DH, typename VECTOR, int dim>
#endif
   class InterpolatedFaceDataContainer : public FaceDataContainer<DH, VECTOR, dim>
   {
      public :
     /**
      * Constructor. Initializes the FaceFEValues objects.
      *
      * @template FE                   The type of Finite Element in use here.
      * @template SPARSITYPATTERN      The corresponding Sparsitypattern to the class-template VECTOR.
      * @template dopedim              The dimension of the control variable.
      * @template dealdim              The dimension of the state variable.
      *
      * @param selected_component      A FEValuesExtractor::Vector indicating which
      *                                components of FE need to be interpolated
      * @param map                     The mapping for the target element of the interpolation
      * @param fe_interpolate          The finite element to which the interpolation should go
      * @param fquad                   Reference to the face quadrature-rule which we use at the moment.
      * @param update_flags            The update flags we need to initialize the FEValues obejcts
      * @param sth                     A reference to the SpaceTimeHandler in use.
      * @param element                    A vector of element iterators through which we gain most of the needed information (like
      *                                material_ids, n_dfos, etc.)
      * @param param_values            A std::map containing parameter data (e.g. non space dependent data). If the control
      *                                is done by parameters, it is contained in this map at the position "control".
      * @param domain_values           A std::map containing domain data (e.g. nodal vectors for FE-Functions). If the control
      *                                is distributed, it is contained in this map at the position "control". The state may always
      *                                be found in this map at the position "state"
      * @param need_neighbour          Describes whether we need all the GetNbr (= Get Neighbor) functions.
      *
      */
    	 template<template<int, int> class FE, typename SPARSITYPATTERN, int dopedim, int dealdim>
         InterpolatedFaceDataContainer(
			 const FEValuesExtractors::Vector selected_component,
			 const Mapping<dim>		&map,
			 const FiniteElement<dim>	&fe_interpolate,
			 const Quadrature<dim-1> &fquad, UpdateFlags update_flags,
#if DEAL_II_VERSION_GTE(9,3,0)
			 SpaceTimeHandler<FE, false, SPARSITYPATTERN, VECTOR,
#else
               		 SpaceTimeHandler<FE, dealii::DoFHandler, SPARSITYPATTERN, VECTOR,
#endif
			 dopedim, dealdim>& sth,
#if DEAL_II_VERSION_GTE(9,3,0)
			 const std::vector<typename DOpEWrapper::DoFHandler<dim>::active_cell_iterator>& element,
#else
                         const std::vector<typename DOpEWrapper::DoFHandler<dim, 
				dealii::DoFHandler>::active_cell_iterator>& element,
#endif
                         const std::map<std::string, const Vector<double>*> &param_values,
                         const std::map<std::string, const VECTOR *> &domain_values,
                         bool need_neighbour) : 
     FaceDataContainer<DH, VECTOR, dim>( fquad, update_flags, sth, element, 
					 param_values, domain_values, need_neighbour), 
       interpolated_fe_face_values_(selected_component, map, sth.GetFESystem("state"), 
				    fe_interpolate, fquad, update_flags),
       element_(element)
	 {
         }

     /**
      * Constructor. Initializes the FaceFEValues objects for PDE problems.
      *
      * @template FE                   The type of Finite Element in use here.
      * @template SPARSITYPATTERN      The corresponding Sparsitypattern to the class-template VECTOR.
      * @template dopedim              The dimension of the control variable.
      * @template dealdim              The dimension of the state variable.
      *
      * @param selected_component      A FEValuesExtractor::Vector indicating which
      *                                components of FE need to be interpolated
      * @param map                     The mapping for the target element of the interpolation
      * @param fe_interpolate          The finite element to which the interpolation should go
      * @param fquad                   Reference to the face quadrature-rule which we use at the moment.
      * @param update_flags            The update flags we need to initialize the FEValues obejcts
      * @param sth                     A reference to the SpaceTimeHandler in use.
      * @param element                    A vector of element iterators through which we gain most of the needed information (like
      *                                material_ids, n_dfos, etc.)
      * @param param_values            A std::map containing parameter data (e.g. non space dependent data). If the control
      *                                is done by parameters, it is contained in this map at the position "control".
      * @param domain_values           A std::map containing domain data (e.g. nodal vectors for FE-Functions). If the control
      *                                is distributed, it is contained in this map at the position "control". The state may always
      *                                be found in this map at the position "state"
      * @param need_neighbour          Describes whether we need all the GetNbr (= Get Neighbor) functions.
      *
      */
     template<template<int, int> class FE, typename SPARSITYPATTERN>
	 InterpolatedFaceDataContainer(
		const FEValuesExtractors::Vector selected_component,
		const Mapping<dim>		&map,
		const FiniteElement<dim>	&fe_interpolate,
		const Quadrature<dim-1> &fquad, UpdateFlags update_flags, 
#if DEAL_II_VERSION_GTE(9,3,0)
		StateSpaceTimeHandler<FE, false, SPARSITYPATTERN,
#else
		StateSpaceTimeHandler<FE, dealii::DoFHandler, SPARSITYPATTERN,
#endif
		VECTOR, dim> &sth,
		const std::vector<
#if DEAL_II_VERSION_GTE(9,3,0)
		typename DOpEWrapper::DoFHandler<dim>::active_cell_iterator>& element,
#else
		typename DOpEWrapper::DoFHandler<dim,dealii::DoFHandler>::active_cell_iterator> &element,
#endif
		const std::map<std::string, const Vector<double>*> &param_values,
		const std::map<std::string, const VECTOR *> &domain_values,
		bool need_neighbour) :
       FaceDataContainer<DH, VECTOR, dim>(fquad, update_flags, sth, element,
					  param_values, domain_values,
					  need_neighbour), 
       interpolated_fe_face_values_(selected_component, map, sth.GetFESystem("state"), 
				    fe_interpolate, fquad, update_flags),
       element_(element)
	 {
 	 }

         void ReInit(unsigned int face_no)
	 {
	    FaceDataContainer<DH, VECTOR, dim>::ReInit(face_no);
	    interpolated_fe_face_values_.reinit(element_[FaceDataContainer<DH, VECTOR, dim>::GetStateIndex()],face_no);
	 }

         void ReInit(unsigned int face_no, unsigned int subface_no)
	 {
	    FaceDataContainer<DH, VECTOR, dim>::ReInit(face_no, subface_no);
	    throw DOpEException("Not implemented","InterpolatedFaceDataContainer::ReInit(unsigned int, unsigned int)");
	 }

         InterpolatedFEFaceValues<dim> 
	 GetInterpolatedFEFaceValuesState() const
	 {
	    return interpolated_fe_face_values_;
	 }

         private:
  	    InterpolatedFEFaceValues<dim>	interpolated_fe_face_values_;
	    const std::vector<typename dealii::DoFHandler<dim>::active_cell_iterator> &element_; 

   };
}





#endif

