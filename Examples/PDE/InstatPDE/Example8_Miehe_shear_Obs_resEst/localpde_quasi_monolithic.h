/**
 *
 * Copyright (C) 2012-2017 by the DOpElib authors
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

// Aufspalten des Spannungstensors in sigma+ und sigma-
// Nicht notwendig bei Sneddon, auch nicht bei Miehe tension
#include "stress_splitting.h"

#include "functions.h"

using namespace DOpE;
using namespace std;
using namespace dealii;

/**
 * This class describes elementwise the weak formulation of the PDE.
 * See pdeinterface.h for more information.
 */

template<
template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
         template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
         template<int, int> class DH, typename VECTOR, int dealdim>
class LocalPDE : public PDEInterface<EDC, FDC, DH, VECTOR, dealdim>
{
public:

  static void
  declare_params(ParameterReader &param_reader)
  {
    param_reader.SetSubsection("Local PDE parameters");

    param_reader.declare_entry("constant_k", "0.0", Patterns::Double(0));
    param_reader.declare_entry("alpha_eps", "0.0", Patterns::Double(0));
    param_reader.declare_entry("G_c", "0.0", Patterns::Double(0));
    param_reader.declare_entry("lame_coefficient_mu", "0.0", Patterns::Double(0));
    param_reader.declare_entry("lame_coefficient_lambda", "0.0", Patterns::Double(0));
    param_reader.declare_entry("sigma", "1", Patterns::Double(0),
				"Which sigma in complementarity function");

  }

  LocalPDE(ParameterReader &param_reader) :
    state_block_component_(4, 0)
  {
    state_block_component_[2] = 1;
    state_block_component_[3] = 2;
    //state_block_component_[4] = 3;

    param_reader.SetSubsection("Local PDE parameters");

    param_reader.SetSubsection("Local PDE parameters");
    constant_k_ = param_reader.get_double("constant_k");
    alpha_eps_ = param_reader.get_double("alpha_eps");
    G_c_ = param_reader.get_double("G_c");
    lame_coefficient_mu_ = param_reader.get_double("lame_coefficient_mu");
    lame_coefficient_lambda_ = param_reader.get_double("lame_coefficient_lambda");
    
    s_ = param_reader.get_double("sigma");

  }

    // Eigentliche PDE bzw. rechte Seite der Newton-Methode,
    // da wir in DOpElib immer von nicht-linearen Problemen ausgehen
  void
  ElementEquation(const EDC<DH, VECTOR, dealdim> &edc,
                  dealii::Vector<double> &local_vector, double scale,
                  double /*scale_ico*/)
  {
    assert(this->problem_type_ == "state");
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();

    //double element_diameter = edc.GetElementDiameter();
    double element_diameter = 0.044;
   
    uvalues_.resize(n_q_points, Vector<double>(4));
    ugrads_.resize(n_q_points, vector<Tensor<1, 2> >(4));
    last_timestep_uvalues_.resize(n_q_points, Vector<double>(4));

    edc.GetValuesState("last_newton_solution", uvalues_);
    edc.GetGradsState("last_newton_solution", ugrads_);

    edc.GetValuesState("last_time_solution", last_timestep_uvalues_);

    const FEValuesExtractors::Vector displacements(0);
    const FEValuesExtractors::Scalar phasefield(2);
    //const FEValuesExtractors::Scalar pressure(3);
    const FEValuesExtractors::Scalar multiplier(3);

    Tensor<2,2> Identity;
    Identity[0][0] = 1.0;
    Identity[1][1] = 1.0;

    Tensor<2,2> zero_matrix;
    zero_matrix.clear();

    // loop over all quadrature points
    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
	// new for pressure variable
        //double press = uvalues_[q_point](3);

        // new for the pressure equation
        //double div_u = ugrads_[q_point][0][0] + ugrads_[q_point][1][1];


	// displacements gradient
        Tensor<2, 2> grad_u;
        grad_u.clear();
        grad_u[0][0] = ugrads_[q_point][0][0];
        grad_u[0][1] = ugrads_[q_point][0][1];
        grad_u[1][0] = ugrads_[q_point][1][0];
        grad_u[1][1] = ugrads_[q_point][1][1];
	
	// displacements
        Tensor<1, 2> u;
        u.clear();
        u[0] = uvalues_[q_point](0);
        u[1] = uvalues_[q_point](1);

	// phase field gradient
        Tensor<1,2> grad_pf;
        grad_pf.clear();
        grad_pf[0] = ugrads_[q_point][2][0];
        grad_pf[1] = ugrads_[q_point][2][1];

	double pf = uvalues_[q_point](2);
        double old_timestep_pf = last_timestep_uvalues_[q_point](2);

        const Tensor<2,2> E = 0.5 * (grad_u + transpose(grad_u));
        const double tr_E = trace(E);

        Tensor<2,2> stress_term;
        stress_term.clear();
	stress_term = lame_coefficient_lambda_ * tr_E * Identity + 2 * lame_coefficient_mu_ * E;

        Tensor<2,2> stress_term_plus;
        Tensor<2,2> stress_term_minus;


        // Necessary because stress splitting does not work
        // in the very initial time step.

        if (this->GetTime() > 0.005) // to avoid zero matrices in the null lines of the loading
          {
		decompose_stress(stress_term_plus, stress_term_minus,
                             E, tr_E, zero_matrix , 0.0,
                             lame_coefficient_lambda_,
                             lame_coefficient_mu_, false);

	  }
	else if (this->GetTime() == 0.001)
	  {
		//std::cout << "First timestep no stress splitting!" << std::endl;
            	stress_term_plus = stress_term;
             	stress_term_minus = 0;
	  }
	else
	  {
		//std::cout << "Current step with nu = 0.18.." << std::endl;
		decompose_stress(stress_term_plus, stress_term_minus,
                             E, tr_E, zero_matrix , 0.0,
                             121150.0,
                             lame_coefficient_mu_, false);
	  }

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            //const Tensor<1, 2> phi_i_u = state_fe_values[displacements].value(i,q_point);
            const Tensor<2, 2> phi_i_grads_u = state_fe_values[displacements].gradient(i, q_point);
            const double phi_i_pf = state_fe_values[phasefield].value(i, q_point);
	    const Tensor<1, 2> phi_i_grads_pf = state_fe_values[phasefield].gradient(i, q_point);
            //const double phi_i_p = state_fe_values[pressure].value(i, q_point);

            // Solid (Time-lagged version) + pressure (!) // i test function
            local_vector(i) += scale
                               * (
				scalar_product(((1.0-constant_k_) * old_timestep_pf * old_timestep_pf + constant_k_) *
                                                 stress_term_plus, phi_i_grads_u)
                                  + scalar_product(stress_term_minus, phi_i_grads_u)
                                 ) * state_fe_values.JxW(q_point); 

            // Phase-field + pressure (!)
            local_vector(i) += scale
                               * (
				// Main terms
				(1.0 - constant_k_) * scalar_product(stress_term_plus, E) * pf * phi_i_pf
                                 - G_c_/(alpha_eps_ * element_diameter) * (1.0 - pf) * phi_i_pf
                                 + G_c_ * alpha_eps_ * element_diameter * grad_pf * phi_i_grads_pf
                               ) * state_fe_values.JxW(q_point);


	    // Pressure equation a(*,*,p)
	    //if (this->GetTime() > 0.005)
		//{
	    		//local_vector(i) += scale_ico
                           //    * ((-1.0/lame_coefficient_lambda_)* press * phi_i_p
                           //    + div_u * phi_i_p) * state_fe_values.JxW(q_point);
		//}
		//else
		//{
	    		//local_vector(i) += scale_ico
                         //      * ((-1.0/6180.0)* press * phi_i_p
                        //       + div_u * phi_i_p) * state_fe_values.JxW(q_point);
		//}

	    //Now the inequality constraint.
	    //Evaluate only in vertices, so we check whether the lambda test function
	    // is one (i.e. we are in a vertex)

	    if(fabs(state_fe_values[multiplier].value(i,q_point) - 1.) < std::numeric_limits<double>::epsilon())
	    {
	      //Weight to account for multiplicity when running over multiple meshes.
	      unsigned int n_neig = edc.GetNNeighbourElementsOfVertex(state_fe_values.quadrature_point(q_point));
	      double weight = 1./n_neig;
	      if(n_neig == 4)
	      {
		//Equation for multiplier
		local_vector(i) += scale * weight* (uvalues_[q_point][3]
						    - std::max(0.,uvalues_[q_point][3]+s_*(pf-old_timestep_pf)));
		//Add Multiplier to the state equation
		//find corresponding basis of state
		for(unsigned int j = 0; j < n_dofs_per_element; j++)
		{
		  if(fabs(state_fe_values[phasefield].value(j,q_point) - 1.) < std::numeric_limits<double>::epsilon())
		  {
		    local_vector(j) += scale * weight* uvalues_[q_point][3];
		  }
		}
	      }
	      else //Boundary or hanging node (no weight, so it works if hanging)
	      {
		local_vector(i) += scale * uvalues_[q_point][3];
	      }
	    }
          }
      }


  }

  void
  ElementMatrix(const EDC<DH, VECTOR, dealdim> &edc,
                FullMatrix<double> &local_matrix, double scale,
                double /*scale_ico*/)
  {

    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();

    //double element_diameter = edc.GetElementDiameter();
    double element_diameter = 0.044;
 
    const FEValuesExtractors::Vector displacements(0);
    const FEValuesExtractors::Scalar phasefield(2);
    //const FEValuesExtractors::Scalar pressure(3);
    const FEValuesExtractors::Scalar multiplier(3);

    uvalues_.resize(n_q_points, Vector<double>(4));
    ugrads_.resize(n_q_points, vector<Tensor<1, 2> >(4));
    last_timestep_uvalues_.resize(n_q_points, Vector<double>(4));

    edc.GetValuesState("last_newton_solution", uvalues_);
    edc.GetGradsState("last_newton_solution", ugrads_);

    edc.GetValuesState("last_time_solution", last_timestep_uvalues_);

    std::vector<Tensor<1, 2> > phi_u(n_dofs_per_element);
    std::vector<Tensor<2, 2> > phi_grads_u(n_dofs_per_element);
    std::vector<double> div_phi_u(n_dofs_per_element);
    std::vector<double> phi_pf(n_dofs_per_element);
    std::vector<Tensor<1, 2> > phi_grads_pf(n_dofs_per_element);
    //std::vector<double> phi_p(n_dofs_per_element);
    //std::vector<double> phi_p_plus(n_dofs_per_element);
    //std::vector<Tensor<1, 2> > phi_grads_p(n_dofs_per_element);
    std::vector<Tensor<2, 2> > E_test(n_dofs_per_element);
    

    Tensor<2,2> Identity;
    Identity[0][0] = 1.0;
    Identity[1][1] = 1.0;

    Tensor<2,2> zero_matrix;
    zero_matrix.clear();

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        for (unsigned int k = 0; k < n_dofs_per_element; k++)
          {

            phi_u[k] = state_fe_values[displacements].value(k, q_point);
            phi_grads_u[k] = state_fe_values[displacements].gradient(k, q_point);
	    div_phi_u[k] = state_fe_values[displacements].divergence(k, q_point);
            phi_pf[k] = state_fe_values[phasefield].value(k, q_point);
            phi_grads_pf[k] = state_fe_values[phasefield].gradient(k, q_point);
	    //phi_p[k] = state_fe_values[pressure].value(k, q_point);
	    //phi_p_plus[k]=std::max(0.0,phi_p[k]);
    
          }

	// new for pressure variable
        //double press = uvalues_[q_point](3);

        Tensor<2, 2> grad_u;
        grad_u.clear();
        grad_u[0][0] = ugrads_[q_point][0][0];
        grad_u[0][1] = ugrads_[q_point][0][1];
        grad_u[1][0] = ugrads_[q_point][1][0];
        grad_u[1][1] = ugrads_[q_point][1][1];

        Tensor<1, 2> v;
        v[0] = uvalues_[q_point](0);
        v[1] = uvalues_[q_point](1);


        Tensor<1,2> grad_pf;
        grad_pf.clear();
        grad_pf[0] = ugrads_[q_point][2][0];
        grad_pf[1] = ugrads_[q_point][2][1];

        double pf = uvalues_[q_point](2);
        double old_timestep_pf = last_timestep_uvalues_[q_point](2);

	const Tensor<2,2> E = 0.5 * (grad_u + transpose(grad_u));
        const double tr_E = trace(E);

        Tensor<2,2> stress_term;
        stress_term.clear();
        stress_term = lame_coefficient_lambda_ * tr_E * Identity // lame_coefficient_lambda_ * tr_E = press
                      + 2 * lame_coefficient_mu_ * E;
	Tensor<2,2> stress_term_plus;
       	Tensor<2,2> stress_term_minus;


        // Necessary because stress splitting does not work
        // in the very initial time step.
        if (this->GetTime() > 0.005)
          {
		decompose_stress(stress_term_plus, stress_term_minus,
                             E, tr_E, zero_matrix , 0.0,
                             lame_coefficient_lambda_,
                             lame_coefficient_mu_, false);
	  }
	else if (this->GetTime() == 0.001)
          {
			//Tensor<2,2> stress_term_plus;
       			//Tensor<2,2> stress_term_minus;
			stress_term_plus = stress_term;
                        stress_term_minus = 0;
	  }
	else
	  {
		    decompose_stress(stress_term_plus, stress_term_minus,
                             E, tr_E, zero_matrix , 0.0,
                             121150.0,
                             lame_coefficient_mu_, false);
	  }
	

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {

            const Tensor<2, 2> E_LinU = 0.5
                                        * (phi_grads_u[i] + transpose(phi_grads_u[i]));

            const double tr_E_LinU = trace(E_LinU);

            Tensor<2,2> stress_term_LinU;
	    Tensor<2,2> stress_term_plus_LinU;
            Tensor<2,2> stress_term_minus_LinU;

	    if (this->GetTime() > 0.005)
	      {
            	        stress_term_LinU = lame_coefficient_lambda_ * tr_E_LinU * Identity
                               + 2 * lame_coefficient_mu_ * E_LinU;
	      }
	   else
	      {
			stress_term_LinU = 121150.0 * tr_E_LinU * Identity
                               + 2 * lame_coefficient_mu_ * E_LinU;
	      }

            // Necessary because stress splitting does not work
            // in the very initial time step.
             if (this->GetTime() > 0.005) // to avoid zero matrices in the null lineas of the loading
	      {
                	decompose_stress(stress_term_plus_LinU, stress_term_minus_LinU,
                                 E, tr_E, E_LinU, tr_E_LinU,
                                 lame_coefficient_lambda_,
                                 lame_coefficient_mu_,
                                 true);
              }
	     else if (this->GetTime() == 0.001)
              {
			//Tensor<2,2> stress_term_plus_LinU;
            		//Tensor<2,2> stress_term_minus_LinU;
			stress_term_plus_LinU = stress_term_LinU;
                        stress_term_minus_LinU = 0;
	      }
	    else 
	      {
                	decompose_stress(stress_term_plus_LinU, stress_term_minus_LinU,
                                 E, tr_E, E_LinU, tr_E_LinU,
                                 121150.0,
                                 lame_coefficient_mu_,
                                 true);

              } 

            for (unsigned int j = 0; j < n_dofs_per_element; j++)
              {
	        // Solid (time-lagged version) // i ansatz function, j test function
                local_matrix(j, i) += scale
                                      * ( 
					scalar_product(((1-constant_k_) * old_timestep_pf * old_timestep_pf + constant_k_) *
                                                        stress_term_plus_LinU, phi_grads_u[j]) // du
                                         + scalar_product(stress_term_minus_LinU, phi_grads_u[j]) // du
					 //+ scalar_product(((1-constant_k_) * old_timestep_pf * old_timestep_pf + constant_k_) *
                                                       //phi_p_plus[i] * Identity, phi_grads_u[j])  // dp
                                         //+ scalar_product(phi_p[i] * Identity, phi_grads_u[j]) // dp
					 //+ scalar_product(phi_p_plus[i] * Identity, phi_grads_u[j]) // dp
                                        ) * state_fe_values.JxW(q_point);
                // Phase-field + pressure(!)
                local_matrix(j, i) += scale
                                      * (
					// Main terms
					(1-constant_k_) * (scalar_product(stress_term_plus_LinU, E)
                                                             + scalar_product(stress_term_plus, E_LinU)) * pf * phi_pf[j] // du
                                        +(1-constant_k_) * scalar_product(stress_term_plus, E) * phi_pf[i] * phi_pf[j] // d phi
                                        + G_c_/(alpha_eps_ * element_diameter) * phi_pf[i] * phi_pf[j] // d phi
                                        + G_c_ * alpha_eps_ * element_diameter * phi_grads_pf[i] * phi_grads_pf[j] // d phi
					//+ (1-constant_k_) * scalar_product(phi_p_plus[i] * Identity, E) * pf * phi_pf[j] // dp
                                        ) * state_fe_values.JxW(q_point);

		// Pressure; incompressibility
	  	//if (this->GetTime() > 0.005)
		//{
			//local_matrix(j, i) += scale_ico
                         //             * ( (phi_grads_u[i][0][0] + phi_grads_u[i][1][1]) * phi_p[j] 
			//	        + ((-1.0/lame_coefficient_lambda_) * phi_p[i] * phi_p[j]) 
			//	        ) * state_fe_values.JxW(q_point); // ok
		//}
		//else
		//{
			//local_matrix(j, i) += scale_ico
                              //        * ( (phi_grads_u[i][0][0] + phi_grads_u[i][1][1]) * phi_p[j] 
				//        + ((-1.0/6180.0) * phi_p[i] * phi_p[j]) 
				//        ) * state_fe_values.JxW(q_point); // ok
		//}

		
		//Now the Multiplierpart
		//only in vertices, so we check whether one of the 
		//lambda test function
		// is one (i.e. we are in a vertex)
		if(
		  (fabs(state_fe_values[multiplier].value(i,q_point) - 1.) < std::numeric_limits<double>::epsilon())
		  ||
		  (fabs(state_fe_values[multiplier].value(j,q_point) - 1.) < std::numeric_limits<double>::epsilon())
		  )
		{
		  //Weight to account for multiplicity when running over multiple meshes.
		  unsigned int n_neig = edc.GetNNeighbourElementsOfVertex(state_fe_values.quadrature_point(q_point));
		  double weight = 1./n_neig;

		  if(n_neig == 4)
		  {
		    //max = 0
		    if( (uvalues_[q_point][3]+s_*(pf-old_timestep_pf)) <= 0. )
		    {
		      local_matrix(i, j) += scale * weight* state_fe_values[multiplier].value(i,q_point)
			*state_fe_values[multiplier].value(j,q_point);
		    }
		    else //max > 0
		    {
		      //From Complementarity
		      local_matrix(i, j) -= scale * weight* s_*state_fe_values[phasefield].value(j,q_point)
			*state_fe_values[multiplier].value(i,q_point);
		    }
		    //From Equation
		    local_matrix(i, j) += scale * weight* state_fe_values[phasefield].value(i,q_point)
		      *state_fe_values[multiplier].value(j,q_point);
		  }
		  else //Boundary or hanging node no weight so it works when hanging
		  {
		    local_matrix(i, j) += scale *  state_fe_values[multiplier].value(i,q_point)
		      *state_fe_values[multiplier].value(j,q_point);
		  }
		}
              }
          }
      }
  }

/*NEU - ANFANG */
  
  void
    StrongElementResidual(const EDC<DH, VECTOR, dealdim> &edc,
			  const EDC<DH, VECTOR, dealdim> &edc_w, double &sum, double scale)
  {
    if( this->GetTime() != 0.0 )
    {
      unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();       
      unsigned int n_q_points = edc.GetNQPoints();
      const DOpEWrapper::FEValues<dealdim> &state_fe_values =
	edc.GetFEValuesState();
      
      //double element_diameter = edc.GetElementDiameter();
      double element_diameter = 0.044;
    
      double eps_pf;
      eps_pf = element_diameter* alpha_eps_;
   
      
      fvalues_.resize(n_q_points);
      
      PI_h_z_.resize(n_q_points, Vector<double>(4));
      lap_pf_.resize(n_q_points, Vector<double>(4));
      uvalues_.resize(n_q_points, Vector<double>(4));
      last_timestep_uvalues_.resize(n_q_points, Vector<double>(4));
      
      // zur Berechnung der Energie als Faktor
      ugrads_.resize(n_q_points, vector<Tensor<1, 2> >(4));
      
      auxvalues_.resize(n_q_points, Vector<double>(4));
      edc.GetLaplaciansState("state", lap_pf_);
      edc.GetValuesState("state", uvalues_);
      edc.GetGradsState("state", ugrads_);
      edc.GetValuesState("last_time_state", last_timestep_uvalues_);    
      edc_w.GetValuesState("weight_for_primal_residual", PI_h_z_);
      
      //aux_error_0 contains the data computed by evaluating
      //*AuxRhs. Here this means
      //Component 0 is the contact information
      //Component 1 is the mass matrix diagonal
      edc.GetValuesState("aux_error_0", auxvalues_);
      
      const FEValuesExtractors::Vector displacements(0);
      const FEValuesExtractors::Scalar phasefield(2);
      const FEValuesExtractors::Scalar multiplier(3);
      
      // weight the residual depending on the contact status 
      int fullContact =0;
      // need localSum as in sum everything in summed up 
      double elemRes = 0;
      double complRes = 0;
      
      Tensor<2,2> Identity;
      Identity[0][0] = 1.0;
      Identity[1][1] = 1.0;
      
      Tensor<2,2> zero_matrix;
      zero_matrix.clear();
      
    //make sure the binding of the function has worked
      assert(this->ResidualModifier);
      for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
	// dofs not nodes
	for (unsigned int i = 0; i < n_dofs_per_element; i++)
	{
	  if(fabs(state_fe_values[phasefield].value(i,q_point) - 1.) < std::numeric_limits<double>::epsilon())
	  {		
	    if (fabs(auxvalues_[q_point](2)-1.)< std::numeric_limits<double>::epsilon())
	    {
	      //count how many nodes of the element are in full-contact
	      fullContact += 1;		   
	      
	    }
	  }
	}
	
	
	// displacements gradient local
    	Tensor<2, 2> grad_u;
        grad_u.clear();
        grad_u[0][0] = ugrads_[q_point][0][0];
        grad_u[0][1] = ugrads_[q_point][0][1];
        grad_u[1][0] = ugrads_[q_point][1][0];
        grad_u[1][1] = ugrads_[q_point][1][1];
	
	// displacements
        Tensor<1, 2> u;
        u.clear();
        u[0] = uvalues_[q_point](0);
        u[1] = uvalues_[q_point](1);
	
	
	// Youngs modulus
        const Tensor<2,2> E = 0.5 * (grad_u + transpose(grad_u));
	const double tr_E = trace(E);
	
        Tensor<2,2> stress_term;
        stress_term.clear();
        stress_term = lame_coefficient_lambda_ * tr_E * Identity
	  + 2 * lame_coefficient_mu_ * E;
	
	Tensor<2,2> stress_term_plus;
        Tensor<2,2> stress_term_minus;
        stress_term_plus.clear();
	stress_term_minus.clear();
	
	// Necessary because stress splitting does not work
        // in the very initial time step.	
	if (this->GetTime() > 0.005) // to avoid zero matrices in the null lines of the loading
	{
	  
	  decompose_stress(stress_term_plus, stress_term_minus,
			   E, tr_E, zero_matrix , 0.0,
			   lame_coefficient_lambda_,
			   lame_coefficient_mu_, false);
	  
	}
	else if (this->GetTime() == 0.001)
	{
	  //std::cout << "First timestep no stress splitting!" << std::endl;
	  stress_term_plus = stress_term;
	  stress_term_minus = 0;
	}
	else
	{
	  //std::cout << "Current step with nu = 0.18.." << std::endl;
	  decompose_stress(stress_term_plus, stress_term_minus,
			   E, tr_E, zero_matrix , 0.0,
			   121150.0,
			   lame_coefficient_mu_, false);
	}
	
	fvalues_[q_point] = local::rhs(state_fe_values.quadrature_point(q_point)); //rhs sollte in functions.h implementiert sein 
	double res;
	
	double weightEnergy;
	weightEnergy = scalar_product(stress_term_plus, E);
	//	std::cout << weightEnergy << std::endl;
	
	res =  G_c_*fvalues_[q_point] + G_c_*eps_pf*eps_pf*lap_pf_[q_point](2) - G_c_*uvalues_[q_point](2)- 
	  eps_pf*(1-constant_k_)*weightEnergy*uvalues_[q_point](2);
	
        //Modify the residual as required by the error estimator
        // old version
	//this->ResidualModifier(res);
	
	double meshsize;
	meshsize =1.0;
	// h ist die Diagonale und bei Elementen wird h^2 zurueckgegeben
	this->ResidualModifier(meshsize);
	res = res *res * std::min((meshsize/(G_c_*eps_pf*eps_pf)),1/(G_c_ + eps_pf*(1-constant_k_)*weightEnergy));
	elemRes += scale * (res * PI_h_z_[q_point](2))
	  * state_fe_values.JxW(q_point);
	
      }
      elemRes = elemRes*(4-fullContact);
      
      
      for (unsigned int q = 0; q < n_q_points; q++)
      {
	// dofs not nodes
	for (unsigned int i = 0; i < n_dofs_per_element; i++)
	{
	  // test if q = i (Knoten)
	  if(fabs(state_fe_values[phasefield].value(i,q) - 1.) < std::numeric_limits<double>::epsilon())
	  {
	    // if q is no full contact
	    if (fabs(auxvalues_[q](2)-1.)>std::numeric_limits<double>::epsilon())
	      // but if q is in contact
	    {
	      if(uvalues_[q](3) > 0 )
	      {
		//we have a semi contact node
		// real quadrature loop
		for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
		{
		  //multiply with eps_pf da hier andere Gleichung geloest wird??
		  complRes += eps_pf*uvalues_[q](3)*(fabs(last_timestep_uvalues_[q_point][2]-uvalues_[q_point][2]))*state_fe_values[phasefield].value(i,q_point)*state_fe_values.JxW(q_point);
		}
	      }
	    }
	  }
	}
      }
      
      sum += elemRes;
      sum += complRes;
    }//End of if time == 0.0
    else
    {
      //Whatever error estimate we want to have for the initial_values
    }
  }
  

  
  void
    StrongFaceResidual(
		       const FaceDataContainer<dealii::DoFHandler, VECTOR, dealdim> &fdc,
		       const FaceDataContainer<dealii::DoFHandler, VECTOR, dealdim> &fdc_w,
		       double &sum, double scale)
  {
    if( this->GetTime() != 0.0 )
    {
      unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();          
      unsigned int n_q_points = fdc.GetNQPoints();
      // auto findet selber heraus ?
      const  auto &state_fe_values = fdc.GetFEFaceValuesState();
      
      //double element_diameter = fdc.GetElementDiameter();
      double element_diameter = 0.044;
    
      double eps_pf;
      eps_pf = element_diameter* alpha_eps_;
      
      ugrads_.resize(n_q_points, std::vector<Tensor<1, dealdim> >(4));
      ugrads_nbr_.resize(n_q_points, std::vector<Tensor<1, dealdim> >(4));
      PI_h_z_.resize(n_q_points, Vector<double>(4));
      auxvalues_.resize(n_q_points, Vector<double>(4));
     
      fdc.GetFaceValuesState("aux_error_0", auxvalues_);
      
      fdc.GetFaceGradsState("state", ugrads_);
      fdc.GetNbrFaceGradsState("state", ugrads_nbr_);
      fdc_w.GetFaceValuesState("weight_for_primal_residual", PI_h_z_);
      vector<double> jump(n_q_points);
    
      const FEValuesExtractors::Scalar phasefield(2);
      
      Tensor<2,2> Identity;
      Identity[0][0] = 1.0;
      Identity[1][1] = 1.0;
      
      Tensor<2,2> zero_matrix;
      zero_matrix.clear();
      
      // weight the face residual depending on the contact status of the nodes
      int fullContact = 0;
      // need localSum as in sum everything is summed up also the element residual
      double localSum = 0;
      
      for (unsigned int q = 0; q < n_q_points; q++)
      {
	// phase field gradient
        Tensor<1,2> grad_pf;
        grad_pf.clear();
        grad_pf[0] = ugrads_[q][2][0];
        grad_pf[1] = ugrads_[q][2][1];
	
	// phase field gradient of neighbour
        Tensor<1,2> grad_pf_nbr;
        grad_pf_nbr.clear();
        grad_pf_nbr[0] = ugrads_nbr_[q][2][0];
        grad_pf_nbr[1] = ugrads_nbr_[q][2][1];

	jump[q] = G_c_*eps_pf*eps_pf*(grad_pf_nbr[0] - grad_pf[0])
	  * fdc.GetFEFaceValuesState().normal_vector(q)[0]
	  + G_c_*eps_pf*eps_pf*(grad_pf_nbr[1] - grad_pf[1])
	  * fdc.GetFEFaceValuesState().normal_vector(q)[1];
	
	// dofs not nodes
   	for (unsigned int i = 0; i < n_dofs_per_element; i++)
	{
	  if(fabs(state_fe_values[phasefield].value(i,q) - 1.) < std::numeric_limits<double>::epsilon())
	  {
	    if (fabs(auxvalues_[q](2)-1.)< std::numeric_limits<double>::epsilon())
	    {
	      //count how many nodes of the element are in full-contact
		    fullContact += 1;		   
		    
	    }
	  }
	}
      }
      //make sure the binding of the function has worked
      assert(this->ResidualModifier);

      for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        //Modify the residual as required by the error estimator
        double res;
        res = jump[q_point];
	//	 this->ResidualModifier(res);
	
	// displacements gradient local
        Tensor<2, 2> grad_u;
        grad_u.clear();
        grad_u[0][0] = ugrads_[q_point][0][0];
        grad_u[0][1] = ugrads_[q_point][0][1];
        grad_u[1][0] = ugrads_[q_point][1][0];
        grad_u[1][1] = ugrads_[q_point][1][1];
	
	
	// Youngs modulus
        const Tensor<2,2> E = 0.5 * (grad_u + transpose(grad_u));
	const double tr_E = trace(E);
	
        Tensor<2,2> stress_term;
        stress_term.clear();
        stress_term = lame_coefficient_lambda_ * tr_E * Identity
	  + 2 * lame_coefficient_mu_ * E;
	
	Tensor<2,2> stress_term_plus;
        Tensor<2,2> stress_term_minus;
        stress_term_plus.clear();
	stress_term_minus.clear();
	
	// Necessary because stress splitting does not work
        // in the very initial time step.	
	if (this->GetTime() > 0.005) // to avoid zero matrices in the null lines of the loading
	{
	  
	  decompose_stress(stress_term_plus, stress_term_minus,
			   E, tr_E, zero_matrix , 0.0,
			   lame_coefficient_lambda_,
			   lame_coefficient_mu_, false);
	  
	}
	else if (this->GetTime() == 0.001)
	{
	  //std::cout << "First timestep no stress splitting!" << std::endl;
	  stress_term_plus = stress_term;
	  stress_term_minus = 0;
	}
	else
	{
	  //std::cout << "Current step with nu = 0.18.." << std::endl;
	  decompose_stress(stress_term_plus, stress_term_minus,
			   E, tr_E, zero_matrix , 0.0,
			   121150.0,
			   lame_coefficient_mu_, false);
	}
	
	double weightEnergy;
	weightEnergy = scalar_product(stress_term_plus, E);
	
	double meshsize;
	meshsize =1.0;
	// h ist die Diagonale und bei Seiten wird h zurueckgegeben
	this->ResidualModifier(meshsize);
	//weighting according to eps
	res = res *res * min(meshsize/(sqrt(G_c_)*(eps_pf)),1/(sqrt(G_c_ + eps_pf*(1-constant_k_)*weightEnergy)))*(1/(sqrt(G_c_)*eps_pf));
	
	
        localSum += scale * (res * PI_h_z_[q_point](0))
	  * fdc.GetFEFaceValuesState().JxW(q_point);
      }
      localSum = (2.0-fullContact)*localSum;
      
      sum += localSum;
    }
    else
    {
      //Whatever if time == 0.0
    }
  }
   
  
  void
    StrongBoundaryResidual(
			   const FaceDataContainer<dealii::DoFHandler, VECTOR, dealdim> &fdc,
			   const FaceDataContainer<dealii::DoFHandler, VECTOR, dealdim> &fdc_w,
			   double &sum, double scale)
  {
    if( this->GetTime() != 0.0 )
    {
      unsigned int n_q_points = fdc.GetNQPoints();
      unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();          
      
      // auto findet selber heraus ?
      const  auto &state_fe_values = fdc.GetFEFaceValuesState();
      
      //double element_diameter = fdc.GetElementDiameter();
      double element_diameter = 0.044;
    
      double eps_pf;
      eps_pf = element_diameter* alpha_eps_;
      
      ugrads_.resize(n_q_points, std::vector<Tensor<1, dealdim> >(4));
      //ugrads_nbr_.resize(n_q_points, std::vector<Tensor<1, dealdim> >(4));
      PI_h_z_.resize(n_q_points, Vector<double>(4));
      auxvalues_.resize(n_q_points, Vector<double>(4));
    
      fdc.GetFaceValuesState("aux_error_0", auxvalues_);
      
      fdc.GetFaceGradsState("state", ugrads_);
      //fdc.GetNbrFaceGradsState("state", ugrads_nbr_);
      fdc_w.GetFaceValuesState("weight_for_primal_residual", PI_h_z_);
      vector<double> jump(n_q_points);
      
      const FEValuesExtractors::Scalar phasefield(2);
      
      Tensor<2,2> Identity;
      Identity[0][0] = 1.0;
      Identity[1][1] = 1.0;
      
      Tensor<2,2> zero_matrix;
      zero_matrix.clear();
      
      // weight the face residual depending on the contact status of the nodes
      int fullContact = 0;
      // need localSum as in sum everything is summed up also the element residual
      double localSum = 0;
      
      
      for (unsigned int q = 0; q < n_q_points; q++)
      {
	
	// phase field gradient
        Tensor<1,2> grad_pf;
        grad_pf.clear();
        grad_pf[0] = ugrads_[q][2][0];
        grad_pf[1] = ugrads_[q][2][1];
	
	
        jump[q] = G_c_*eps_pf*eps_pf*(0 - grad_pf[0])
    	  * fdc.GetFEFaceValuesState().normal_vector(q)[0]
    	  + G_c_*eps_pf*eps_pf*(0 - grad_pf[1])
    	  * fdc.GetFEFaceValuesState().normal_vector(q)[1];
	
	// dofs not nodes
	for (unsigned int i = 0; i < n_dofs_per_element; i++)
	{
	  if(fabs(state_fe_values[phasefield].value(i,q) - 1.) < std::numeric_limits<double>::epsilon())
	  {
	    if (fabs(auxvalues_[q](2)-1.)< std::numeric_limits<double>::epsilon())
	    {
	      //count how many nodes of the element are in full-contact
	      fullContact += 1;		   
	      
	    }
	  }
	}
	
      }
      
      //make sure the binding of the function has worked
      assert(this->ResidualModifier);
      
      for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        //Modify the residual as required by the error estimator
        double res;
        res = jump[q_point];
	// this->ResidualModifier(res);
	
	// displacements gradient local
        Tensor<2, 2> grad_u;
        grad_u.clear();
        grad_u[0][0] = ugrads_[q_point][0][0];
        grad_u[0][1] = ugrads_[q_point][0][1];
        grad_u[1][0] = ugrads_[q_point][1][0];
        grad_u[1][1] = ugrads_[q_point][1][1];
	
	
	// Youngs modulus
        const Tensor<2,2> E = 0.5 * (grad_u + transpose(grad_u));
	const double tr_E = trace(E);
	
        Tensor<2,2> stress_term;
        stress_term.clear();
        stress_term = lame_coefficient_lambda_ * tr_E * Identity
	  + 2 * lame_coefficient_mu_ * E;
	
	Tensor<2,2> stress_term_plus;
        Tensor<2,2> stress_term_minus;
        stress_term_plus.clear();
	stress_term_minus.clear();
	
	// Necessary because stress splitting does not work
        // in the very initial time step.	
	if (this->GetTime() > 0.005) // to avoid zero matrices in the null lines of the loading
	{
	  
	  decompose_stress(stress_term_plus, stress_term_minus,
			   E, tr_E, zero_matrix , 0.0,
			   lame_coefficient_lambda_,
			   lame_coefficient_mu_, false);
	  
	}
	else if (this->GetTime() == 0.001)
	{
	    //std::cout << "First timestep no stress splitting!" << std::endl;
	  stress_term_plus = stress_term;
	  stress_term_minus = 0;
	}
	else
	{
	  //std::cout << "Current step with nu = 0.18.." << std::endl;
	  decompose_stress(stress_term_plus, stress_term_minus,
			   E, tr_E, zero_matrix , 0.0,
			   121150.0,
			   lame_coefficient_mu_, false);
	}
	
	double weightEnergy;
	weightEnergy = scalar_product(stress_term_plus, E);
	
	double meshsize;
	meshsize =1.0;
	// h ist die Diagonale und bei Seiten wird h zurueckgegeben
	this->ResidualModifier(meshsize);
	//weighting according to eps
	res = res *res * min(meshsize/(sqrt(G_c_)*(eps_pf)),1/(sqrt(G_c_ + eps_pf*(1-constant_k_)*weightEnergy)))*(1/(sqrt(G_c_)*eps_pf));
	
	
	localSum += scale * (res * PI_h_z_[q_point](0))
    	  * fdc.GetFEFaceValuesState().JxW(q_point);
      }
      
      localSum = (2.0-fullContact)*localSum;
      
      sum += localSum;
    }
    else
    {
      //Whatever if time == 0.0
    }

  }
  
/*NEU - ENDE*/

void
  ElementRightHandSide(const EDC<DH, VECTOR, dealdim> &/*edc*/,
                       dealii::Vector<double> & /*local_vector*/,
                       double /*scale*/)
  {
    assert(this->problem_type_ == "state");
  }

  void
  ElementTimeEquationExplicit(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                              dealii::Vector<double> & /*local_vector*/,
                              double /*scale*/)
  {
    assert(this->problem_type_ == "state");
  }

  void
  ElementTimeEquation(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                      dealii::Vector<double> & /*local_vector*/,
                      double /*scale*/)
  {
    assert(this->problem_type_ == "state");

  }

  void
  ElementTimeMatrixExplicit(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                            FullMatrix<double> &/*local_matrix*/)
  {
    assert(this->problem_type_ == "state");
  }

  void
  ElementTimeMatrix(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                    FullMatrix<double> &/*local_matrix*/)
  {
    assert(this->problem_type_ == "state");

  }


 /*NEU- ANFANG */

 //Auxiliary Values for Error Estimation
  void ElementAuxRhs(const EDC<DH, VECTOR, dealdim> & edc,
		    dealii::Vector<double> &local_vector,
		    double scale)
  {
   if( this->GetTime() != 0.0 )
    {
      unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
      unsigned int n_q_points = edc.GetNQPoints();
      const DOpEWrapper::FEValues<dealdim> &state_fe_values =
	edc.GetFEValuesState();
      
      assert(this->problem_type_ == "aux_error");
      assert(this->problem_type_num_ == 0);
      //int test = this->GetTime();
      //std::cout << test << std::endl;
      
      uvalues_.resize(n_q_points, Vector<double>(4));
      last_timestep_uvalues_.resize(n_q_points, Vector<double>(4));
      //obstacle_.resize(n_q_points, Vector<double>(4));
      
      edc.GetValuesState("state", uvalues_);
      // ACHTUNG: findet alte Lsg nicht
      edc.GetValuesState("last_time_state", last_timestep_uvalues_);
      //edc.GetValuesState("obstacle", obstacle_);
      //std::cout << "hier" << std::endl; 
      
      const FEValuesExtractors::Scalar phasefield(2);
      const FEValuesExtractors::Scalar multiplier(3);
      
      unsigned int contact_vertices=0;
      //First component is full contact
      //second is mass
      
      //Check if contact vertex
      for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
	for (unsigned int i = 0; i < n_dofs_per_element; i++)
	{
	  //Only in vertices, so we check whether the u test function
	  // is one (i.e. we are in a vertex)
	  if(fabs(state_fe_values[phasefield].value(i,q_point) - 1.) < std::numeric_limits<double>::epsilon())
	  {
	    //Check if contact vertex
	    if((uvalues_[q_point][2]-last_timestep_uvalues_[q_point][2]) >= 0.) 
	      contact_vertices++;
	  }
	}
      }
      //Now assembling the information
      for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
	for (unsigned int i = 0; i < n_dofs_per_element; i++)
	{
	  //Both are vertex based, so we check if the corresponding q point is a vertex
	  //For contact, set one if all (4) vertices are in contact.
	  if(fabs(state_fe_values[phasefield].value(i,q_point) - 1.) < std::numeric_limits<double>::epsilon())
	  {
	    unsigned int n_neig = edc.GetNNeighbourElementsOfVertex(state_fe_values.quadrature_point(q_point));
	    if(n_neig > 0)
	    {
	      if(contact_vertices==4)
	      {
		local_vector(i) += scale/n_neig;
	      }
	    }
	  }
	  //For Mass: \int_{N(x_i)} \phi_i
	  local_vector(i) += scale * state_fe_values[multiplier].value(i,q_point)
	    * state_fe_values.JxW(q_point);   
	}
      }
    }
   else //Time == 0
   {
     
   }
  }

  void FaceAuxRhs(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
		  dealii::Vector<double> &/*local_vector*/,
		  double /*scale*/)
  {
  }
  
  void BoundaryAuxRhs(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
		      dealii::Vector<double> &/*local_vector*/,
		      double /*scale*/)
  {
  }
 
  /*NEU - ENDE */

  UpdateFlags
  GetUpdateFlags() const
  {
    return update_values | update_gradients| update_hessians | update_quadrature_points;
  }

  UpdateFlags
  GetFaceUpdateFlags() const
  {
    return update_values | update_gradients | update_normal_vectors
           | update_quadrature_points;
  }

  /**
   * Returns the number of blocks. We have two for the
   * state variable, namely velocity and pressure.
   * not two! displacement (2), phasefield (1) and pressure(1)
   */

  unsigned int
  GetControlNBlocks() const
  {
    return 1;
  }

  unsigned int
  GetStateNBlocks() const
  {
    // Three Blocks: u, phi, p, lambda
    return 3;
  }

  std::vector<unsigned int> &
  GetControlBlockComponent()
  {
    return control_block_component_;
  }
  const std::vector<unsigned int> &
  GetControlBlockComponent() const
  {
    return control_block_component_;
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
  bool
  HasVertices() const
  {
    return true;
  }

private:
 // vector<Tensor<1, dealdim> > fvalues_;
  vector<double> fvalues_;
  vector<Vector<double> > uvalues_;
  vector<vector<Tensor<1, dealdim> > > ugrads_;
  vector<std::vector<Tensor<1, dealdim> > > ugrads_nbr_;
 
  /*NEU - ANFANG */
  vector<Vector<double> > PI_h_z_;
  vector<Vector<double> > lap_pf_;  
  //vector<Vector<double> > obstacle_;
  vector<Vector<double> > auxvalues_;
  /*NEU - ENDE */

  vector<Vector<double> > last_timestep_uvalues_;

  // face values
  vector<vector<Tensor<1, dealdim> > > ufacegrads_;

  vector<unsigned int> state_block_component_;
  vector<unsigned int> control_block_component_;

  double constant_k_, alpha_eps_, 
    G_c_, lame_coefficient_mu_, lame_coefficient_lambda_, s_;

};
#endif
