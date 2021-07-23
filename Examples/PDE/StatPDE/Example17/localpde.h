/**

Nearly incompressible problem

**/

#ifndef LOCALPDE_
#define LOCALPDE_

#include <interfaces/pdeinterface.h>

//RT Stuff
#include <container/interpolatedelementdatacontainer.h>
#include <container/interpolatedintegratordatacontainer.h>

#define InterpIDC InterpolatedIntegratorDataContainer
#define InterpEDC InterpolatedElementDataContainer


using namespace DOpE;
using namespace std;

template<
   template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
   template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
   template<int, int> class DH, typename VECTOR, int dealdim>
class LocalPDE : public PDEInterface<EDC, FDC, DH, VECTOR, dealdim>
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
        double /*scale_ico*/)
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


	    local_vector(i) += scale* (2 * myu_ *
			scalar_product(realgrads, phi_i_grads_real) 
			- press * div_phi_v + divergence * phi_i_q +
			lambda_inverse_ * press * phi_i_q)
			* state_fe_values.JxW(q_point);
	 }
      }
   }

   void
   ElementMatrix(const EDC<DH, VECTOR, dealdim> &edc,
        FullMatrix<double> &local_matrix, double scale, double /*scale_ico*/)
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
	       local_matrix(i,j) += scale*( 2 * myu_ *
			scalar_product(phi_i_grads_v_real[j], phi_i_grads_v_real[i]) -
			phi_i_q[j] * div_phi_i_v[i] + div_phi_i_v[j] * phi_i_q[i] +
			lambda_inverse_ * phi_i_q[j] * phi_i_q[i]) 
			* state_fe_values.JxW(q_point);
	    }
	 }
      }
   }

   // Using InterpEDC instead of EDC, because we need interpolated values //
   void
   ElementRightHandSide(const InterpEDC<DH, VECTOR, dealdim> &
                       edc, dealii::Vector<double> &local_vector, double scale)
   { 
      InterpolatedFEValues<dealdim> fe_values_interpolated = edc.GetInterpolatedFEValues();
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

	 fvalues[0] = -400*myu_*y*(1-y)*(1-2*y)*(1- 6*x + 6*x*x) + 
	 	 30*y*y*(x-0.5)*(x-0.5) - 3*(y-0.5)*(y-0.5)*(y-0.5)*(1-x)*(1-x) -
		 1200*myu_*x*x*(1-x)*(1-x)*(2*y-1);

	 fvalues[1] = 1200*myu_*y*y*(1-y)*(1-y)*(2*x-1) +
 	 	20*y*(x-0.5)*(x-0.5)*(x-0.5) + 3*(1-x)*(1-x)*(1-x)*(y-0.5)*(y-0.5) 
		 + 400*myu_*x*(1-x)*(1-2*x)*(1- 6*y + 6*y*y);

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
      dealii::Vector<double> &/*local_vector*/, double /*scale*/, double /*scale_ico*/)
   { }
     
   void
   BoundaryMatrix(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
      dealii::FullMatrix<double> &/*local_matrix*/, double /*scale*/, double /*scale_ico*/)
   { }
   
   void
   BoundaryRightHandSide(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                        dealii::Vector<double> & /*local_vector*/, double /*scale*/)
   { } 

   UpdateFlags
   GetUpdateFlags() const
   {
      return update_values | update_gradients | update_quadrature_points;
   }

   UpdateFlags
   GetFaceUpdateFlags() const
   {
      return update_values | update_gradients | update_normal_vectors |
		update_quadrature_points;
   }

   unsigned int
   GetStateNBlocks() const
   {
      return 2;
   }

   std::vector<unsigned int> &
   GetStateBlockComponent()
   {
      return state_block_component_;
   }

   const std::vector<unsigned int> &
   GetStateBlockComponent() const
   {
      return state_block_component_;
   }

   private:
      const double myu_ = 3.4722e-5;
      const double lambda_inverse_ = 1e-4;
      vector<Vector<double>>  u_values_;
      vector<vector<Tensor<1,dealdim>>> u_grads_;
      vector<vector<Tensor<1,dealdim>>> ufacegrads_;
      vector<unsigned int> state_block_component_;
      const bool interpolate = 1;
 };


#endif 
  
    

   

