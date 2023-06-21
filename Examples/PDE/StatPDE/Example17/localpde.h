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

#ifndef LOCALPDE_
#define LOCALPDE_

#include <interfaces/pdeinterface.h>

using namespace DOpE;
using namespace std;

#if DEAL_II_VERSION_GTE(9,3,0)
template<
   template<bool DH, typename VECTOR, int dealdim> class EDC,
   template<bool DH, typename VECTOR, int dealdim> class FDC,
   bool DH, typename VECTOR, int dealdim>
class LocalPDE : public PDEInterface<EDC, FDC, DH, VECTOR, dealdim>
#else
template<
   template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
   template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
   template<int, int> class DH, typename VECTOR, int dealdim>
class LocalPDE : public PDEInterface<EDC, FDC, DH, VECTOR, dealdim>
#endif
{
   public:
      LocalPDE() :
    	 state_block_component_(3, 0)
      {
    	 assert(dealdim==2);
    	 state_block_component_[2] = 1;
      }

   void 
   ElementEquation(const EDC<DH, VECTOR, dealdim> &edc,
        dealii::Vector<double> &local_vector, double scale,
        double /*scale_ico*/) override
   { 
      const DOpEWrapper::FEValues<dealdim> &state_fe_values = 
	 edc.GetFEValuesState();

      const unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
      const unsigned int n_q_points = edc.GetNQPoints();

      unsigned int i, q_point;

      assert(this-> problem_type_ == "state");

      u_values_.resize(n_q_points, Vector<double>(3));
      u_grads_.resize(n_q_points, vector<Tensor<1,dealdim>> (3));

      edc.GetValuesState("last_newton_solution", u_values_);
      edc.GetGradsState("last_newton_solution", u_grads_);

      const FEValuesExtractors::Vector disp(0);
      const FEValuesExtractors::Scalar pressure(2);


      for(q_point = 0; q_point < n_q_points; ++q_point)
      {
	 Tensor<2,dealdim> ugrads;
  	 ugrads.clear();
	
	 ugrads[0][0] = u_grads_[q_point][0][0];
	 ugrads[0][1] = u_grads_[q_point][0][1];
	 ugrads[1][0] = u_grads_[q_point][1][0];
	 ugrads[1][1] = u_grads_[q_point][1][1];
  
	 Tensor<2,dealdim> realgrads;
	 realgrads.clear();

	 realgrads = 0.5*(ugrads + transpose(ugrads));

         const double press = u_values_[q_point][2];
	 const double divergence = ugrads[0][0] + ugrads[1][1]; 

 	 for(i = 0; i < n_dofs_per_element; ++i)
	 {
	    const Tensor<2,dealdim> phi_i_grads_v = 
		state_fe_values[disp].gradient(i,q_point);

	    const Tensor<2,dealdim> phi_i_grads_real =
		0.5*(phi_i_grads_v + transpose(phi_i_grads_v));

	    const double phi_i_q = state_fe_values[pressure].value(i,q_point);
	    const double div_phi_v = state_fe_values[disp].divergence(i,q_point);


	    local_vector(i) += scale* (2. * mu_ *
			scalar_product(realgrads, phi_i_grads_real) 
			+ press * div_phi_v + divergence * phi_i_q -
			lambda_inverse_ * press * phi_i_q)
			* state_fe_values.JxW(q_point);
	 }
      }
   }

   void
   ElementMatrix(const EDC<DH, VECTOR, dealdim> &edc,
        FullMatrix<double> &local_matrix, double scale, double /*scale_ico*/) override
   {
      const DOpEWrapper::FEValues<dealdim> &state_fe_values = 
		edc.GetFEValuesState();
      const unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
      const unsigned int n_q_points = edc.GetNQPoints(); 
      unsigned int i, j, q_point;

      const FEValuesExtractors::Vector disp(0);
      const FEValuesExtractors::Scalar pressure(2);

      vector<Tensor<1,dealdim>> phi_i_v(n_dofs_per_element);
      vector<double> 		phi_i_q(n_dofs_per_element);

      vector<Tensor<2,dealdim>> phi_i_grads_v(n_dofs_per_element);
      vector<Tensor<2,dealdim>> phi_i_grads_v_real(n_dofs_per_element);
      vector<double> div_phi_i_v(n_dofs_per_element);
 
      for(q_point = 0; q_point < n_q_points; ++q_point)
      {
	 for(i = 0; i < n_dofs_per_element; ++i)
	 {
	    phi_i_grads_v[i] = state_fe_values[disp].gradient(i,q_point);
	    phi_i_grads_v_real[i] = 0.5*(phi_i_grads_v[i] + transpose(phi_i_grads_v[i]));

	    phi_i_v[i] = state_fe_values[disp].value(i,q_point);
	    div_phi_i_v[i] = state_fe_values[disp].divergence(i,q_point);

            phi_i_q[i] = state_fe_values[pressure].value(i,q_point);
  	 }

         for(i = 0; i < n_dofs_per_element; ++i)
	 {
	    for(j = 0; j < n_dofs_per_element; ++j)
	    {
	       local_matrix(i,j) += scale*( 2. * mu_ *
			scalar_product(phi_i_grads_v_real[j], phi_i_grads_v_real[i]) +
			phi_i_q[j] * div_phi_i_v[i] + div_phi_i_v[j] * phi_i_q[i] -
			lambda_inverse_ * phi_i_q[j] * phi_i_q[i]) 
			* state_fe_values.JxW(q_point);
	    }
	 }
      }
   }

   void
   ElementRightHandSide(const EDC<DH, VECTOR, dealdim> &edc,
			dealii::Vector<double> &local_vector, double scale) override
   {
      InterpolatedFEValues<dealdim> fe_values_interpolated = edc.GetInterpolatedFEValuesState();
      const DOpEWrapper::FEValues<dealdim> &state_fe_values = 
			edc.GetFEValuesState();

      const unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
      const unsigned int n_q_points 	    = edc. GetNQPoints();

      unsigned int q_point, i;

      assert(this->problem_type_ == "state");
   
      Tensor<1, dealdim>  fvalues; fvalues.clear();

      const FEValuesExtractors::Vector disp(0);
      const FEValuesExtractors::Vector pressure(2);

      for(q_point = 0; q_point < n_q_points; ++q_point)
      {
      
	 const Point<dealdim> P = state_fe_values.quadrature_point(q_point);

	 const double x = P[0];
	 const double y = P[1];

	 fvalues[0] = -400*mu_*y*(1-y)*(1-2*y)*(1- 6*x + 6*x*x) + 
	 	 30*y*y*(x-0.5)*(x-0.5) - 3*(y-0.5)*(y-0.5)*(y-0.5)*(1-x)*(1-x) -
		 1200*mu_*x*x*(1-x)*(1-x)*(2*y-1);

	 fvalues[1] = 1200*mu_*y*y*(1-y)*(1-y)*(2*x-1) +
 	 	20*y*(x-0.5)*(x-0.5)*(x-0.5) + 3*(1-x)*(1-x)*(1-x)*(y-0.5)*(y-0.5) 
		 + 400*mu_*x*(1-x)*(1-2*x)*(1- 6*y + 6*y*y);

         for(i = 0; i < n_dofs_per_element; ++i)
	 {
	    const Tensor<1, dealdim> phi_v = 
			state_fe_values[disp].value(i, q_point);


            if (interpolate == 0)
               local_vector(i) += scale* scalar_product(fvalues, phi_v)
				* state_fe_values.JxW(q_point);

            if (interpolate == 1)
               local_vector(i) += scale* scalar_product(fvalues, fe_values_interpolated.value(i,q_point))
				* state_fe_values.JxW(q_point);
	 }
      }
   }

   void
   BoundaryEquation(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
      dealii::Vector<double> &/*local_vector*/, double /*scale*/, double /*scale_ico*/) override
   { }
     
   void
   BoundaryMatrix(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
      dealii::FullMatrix<double> &/*local_matrix*/, double /*scale*/, double /*scale_ico*/) override
   { }
   
   void
   BoundaryRightHandSide(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                        dealii::Vector<double> & /*local_vector*/, double /*scale*/) override
   { } 

   UpdateFlags
   GetUpdateFlags() const override
   {
      return update_values | update_gradients | update_quadrature_points;
   }

   UpdateFlags
   GetFaceUpdateFlags() const override
   {
      return update_values | update_gradients | update_normal_vectors |
		update_quadrature_points;
   }

   unsigned int
   GetStateNBlocks() const override
   {
      return 2;
   }

   std::vector<unsigned int> &
   GetStateBlockComponent() override
   {
      return state_block_component_;
   }

   const std::vector<unsigned int> &
   GetStateBlockComponent() const override
   {
      return state_block_component_;
   }

   private:
      const double mu_ = 3.4722e-5;
      const double lambda_inverse_ = 1e-4;
      vector<Vector<double>>  u_values_;
      vector<vector<Tensor<1,dealdim>>> u_grads_;
      vector<vector<Tensor<1,dealdim>>> ufacegrads_;
      vector<unsigned int> state_block_component_;
      const bool interpolate = 1;
 };


#endif 
  
    

   

