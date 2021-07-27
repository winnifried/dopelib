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

#ifndef INTERPOLATED_FE_FACE_VALUES_H
#define INTERPOLATED_FE_FACE_VALUES_H

#include <deal.II/base/config.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_generic.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_raviart_thomas.h>

DEAL_II_NAMESPACE_OPEN

/**
 * This class provides an dealii::FEFaceValues type object, where the returned 
 * value and gradient values correspond to the interpolation 
 * of the shape functions of fe_from by the shape functions of fe_to
 */
template <int dim, int spacedim = dim>
class InterpolatedFEFaceValues
{
  public:
   InterpolatedFEFaceValues(const FEValuesExtractors::Vector component_view,
			    const Mapping<dim, spacedim>		&map,
			    const FiniteElement<dim, spacedim>	&fe_from,
			    const FiniteElement<dim, spacedim>	&fe_to,
			    const Quadrature<dim-1>	        &quadrature,
			    UpdateFlags				flags);

   // TODO fe_to is Raviart Thomas space for now and will have to be upgraded to 
   // a more generalized one later.


   void reinit(const typename Triangulation<dim, spacedim>::cell_iterator &cell, const unsigned int face_no);

   Tensor<1, spacedim> value(const unsigned int shape_function,
			const unsigned int q_point) const;

   Tensor<2, spacedim> gradient(const unsigned int shape_function,
			const unsigned int q_point) const;

  private:
   const FEValuesExtractors::Vector 	selected_vector;
   const Mapping<dim, spacedim>		&map_;			
   const FiniteElement<dim, spacedim>	&fe_base;		// FE space from which we interpolate
   const FiniteElement<dim, spacedim> 	&fe_interpolated;	// FE space to which we interpolate,
								// This is RT for now.
   const Quadrature<dim-1>		&quad;
   UpdateFlags				update_flags;
   std::vector<std::vector<Tensor<1, spacedim> > > interpolated_values;
   std::vector<std::vector<Tensor<2, spacedim> > > interpolated_gradients;
};

DEAL_II_NAMESPACE_CLOSE

#endif  // INTERPOLATED_FE_FACE_VALUES_H 

