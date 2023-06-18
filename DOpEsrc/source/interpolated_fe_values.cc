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

#include <include/interpolated_fe_values.h>

#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_bdm.h>

DEAL_II_NAMESPACE_OPEN

template <int dim, int spacedim>
InterpolatedFEValues<dim, spacedim>::InterpolatedFEValues(
const FEValuesExtractors::Vector 	component_view,
const Mapping<dim, spacedim>		&map,
const FiniteElement<dim, spacedim>	&fe_from,
const FiniteElement<dim, spacedim>	&fe_to,
const Quadrature<dim>			&quadrature,
UpdateFlags				flags) :
	selected_vector(component_view), map_(map), fe_base(fe_from),
	fe_interpolated(fe_to), 
	quad(quadrature), update_flags(flags)
{
  //Quadrature should be feasible
  Assert(quad.size() != 0, ExcLowerRange(quad.size(),1));
  //The FE where we interpolate should be a Vector of appropriate dimensions
  Assert(fe_to.n_components()==dim, ExcDimensionMismatch(fe_to.n_components(),dim));
  //The FE from where we interpolate the selected components by FEValuesExtractor
  //should have sufficiently many components
  Assert(fe_from.n_components() > component_view.first_vector_component+dim,
	 ExcLowerRange(fe_from.n_components(), component_view.first_vector_component+dim));
  //TODO: Check for element type, since interpolation needs to be done
  //Assert that Div-Conforming image element is used, otherwise the transformation
  //is not correct.
  bool is_div_conforming __attribute__((unused))= dynamic_cast<const FE_RaviartThomas<dim> *>(&fe_to) != nullptr ||
    dynamic_cast<const FE_RaviartThomasNodal<dim> *>(&fe_to) != nullptr ||
    dynamic_cast<const FE_BDM<dim> *>(&fe_to) != nullptr;
  Assert(is_div_conforming, ExcNotImplemented());
  // accordingly, i.e. Hdiv, Hcurl, or just C^0 conforming ...
  
  if (update_flags & update_values)
  {
    interpolated_values.resize(fe_base.dofs_per_cell, 
			       std::vector<Tensor<1,spacedim> >(quad.size(),Tensor<1,spacedim>()));
  }
  
  if (update_flags & update_gradients)
  {
    interpolated_gradients.resize(fe_base.dofs_per_cell, 
				  std::vector<Tensor<2, spacedim> > (quad.size(), Tensor<2, spacedim>()));
  }
}


template<int dim, int spacedim>
void InterpolatedFEValues<dim, spacedim>::reinit(const typename Triangulation<dim, spacedim>::cell_iterator &cell)
{
   // Reinitialize values to the current element.
   const unsigned int fe_base_dofs = fe_base.dofs_per_cell;
   const unsigned int fe_interpolated_dofs = fe_interpolated.dofs_per_cell;
   const unsigned int n_support_points = fe_interpolated.get_generalized_support_points().size();
   const unsigned int quad_size = quad.size();

   Quadrature<dim> 	q2(fe_interpolated.get_generalized_support_points(),
			std::vector<double>(n_support_points, 1.0));

   UpdateFlags u_flag1 = update_default;
   UpdateFlags u_flag2 = update_default;
   if(update_flags & update_values) 
   {
      u_flag1 = u_flag1 | update_values;
      u_flag2 = u_flag2 | update_values;
      u_flag2 = u_flag2 | update_inverse_jacobians;
   }

   if(update_flags & update_values) 
   {
      u_flag1 = u_flag1 | update_gradients;
      u_flag2 = u_flag2 | update_gradients;
      u_flag2 = u_flag2 | update_inverse_jacobians;
   }

   FEValues<dim> 	fe_values_1(map_,fe_base, q2, u_flag1);  // fe values of fe_base on fe_interpolated dofs
   FEValues<dim>     	fe_values_2(map_,fe_interpolated, quad, u_flag2); // fe values of fe_interpolated on target quadrature

   fe_values_1.reinit(cell);
   fe_values_2.reinit(cell);

   const auto &Inv_Jacobians = fe_values_2.get_inverse_jacobians();

   if(update_flags & (update_values | update_gradients))
   {
      // ----- Declaring variables for shape functions values ------ //
      std::vector<Tensor<1, spacedim> > phi_1on2(n_support_points);  // shape function values fe_base on fe_interpolated dofs
      std::vector<Vector<double> > phi_1on2_vec(n_support_points, Vector<double>(dim)); // vec form of phi_1on2
      std::vector<double>	Coeff(fe_interpolated_dofs);	// Coeff or Nodal values of phi_1on2
      std::vector<Tensor<1, spacedim> > phi_2(fe_interpolated_dofs); // shape function values of fe_interpolated on fe_interpolated dofs
      std::vector<Tensor<2, spacedim> > phi_2_grad(fe_interpolated_dofs); // shape function gradient of fe_interpolated on fe_interpolated dofs
 
      const FEValuesExtractors::Vector target_component(0);
      
      for( unsigned int q_index = 0; q_index < quad_size; ++q_index)
      {
         // --- Making Coeff of fe_base on 2 dofs --- //
         for( unsigned int i = 0; i < fe_base_dofs; ++i)
         {
            for( unsigned int j = 0; j < q2.size(); ++j)
            {
	       phi_1on2[j]	= fe_values_1[selected_vector].value(i,j);

	       for( unsigned int k = 0; k < spacedim; ++k)
	          phi_1on2_vec[j][k]	= phi_1on2[j][k];
            }

            fe_interpolated.convert_generalized_support_point_values_to_dof_values(phi_1on2_vec, Coeff);

            if(update_flags & update_values)
            {
            // --- Interpolation of phi_1on2 on fe_interpolated --- //
               interpolated_values[i][q_index] = 0;
               Tensor<1, spacedim> tmp;

               for( unsigned int j = 0; j < fe_interpolated_dofs; ++j)
               {
	          phi_2[j]		= fe_values_2[target_component].value(j, q_index);
	          auto Jinv	= Inv_Jacobians[q_index];

	          phi_2[j]		= apply_transformation(Jinv, phi_2[j]);
	          phi_2[j]		/= Jinv.determinant();
	          phi_2[j]		*= Coeff[j];
	          tmp 		+= phi_2[j];
               }

               interpolated_values[i][q_index] = tmp;
            }

            if (update_flags & update_gradients)
	    {
	       interpolated_gradients[i][q_index] = 0;
	       Tensor<2, spacedim> tmp;
	       for (unsigned int j = 0; j < fe_interpolated_dofs; ++j)
	       {
		  phi_2_grad[j] = fe_values_2[target_component].gradient(j, q_index);
		  auto Jinv 	= Inv_Jacobians[q_index];
		  phi_2_grad[j]	= apply_transformation(Jinv, phi_2_grad[j]);
		  phi_2_grad[j]	/= Jinv.determinant();
		  phi_2_grad[j]	*= Coeff[j];
		  tmp += phi_2_grad[j];
	       }
	       interpolated_gradients[i][q_index] = tmp;
	    } 
	 }
      }
   }
}

template<int dim, int spacedim>
Tensor<1, spacedim> InterpolatedFEValues<dim, spacedim>::value(
		const unsigned int shape_function,
		const unsigned int q_point) const
{
   Assert(shape_function < fe_base.dofs_per_cell,
		ExcIndexRange(shape_function, 0, fe_base.dofs_per_cell));

   Assert(q_point < quad.size(),
		ExcIndexRange(shape_function, 0, quad.size()));

   Assert(update_flags & update_values,
		(typename FEValuesBase<dim, spacedim>::ExcAccessToUninitializedField("update_values")));

   return interpolated_values[shape_function][q_point];
}

template<int dim, int spacedim>
Tensor<2, spacedim> InterpolatedFEValues<dim, spacedim>::gradient(const unsigned int shape_function,
	const unsigned int q_point) const
{
   Assert(shape_function < fe_base.dofs_per_cell,
		ExcIndexRange(shape_function, 0, fe_base.dofs_per_cell));
   Assert(q_point < quad.size(),
		ExcIndexRange(q_point, 0, quad.size()));
   Assert(update_flags & update_gradients,
		(typename FEValuesBase<dim, spacedim>::ExcAccessToUninitializedField( "update_gradients")));
   return interpolated_gradients[shape_function][q_point];
}

template class InterpolatedFEValues<2>;
template class InterpolatedFEValues<3>;

DEAL_II_NAMESPACE_CLOSE
