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

#include "stress_splitting.h"

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

    param_reader.SetSubsection("Local PDE parameters");

    param_reader.SetSubsection("Local PDE parameters");
    constant_k_ = param_reader.get_double("constant_k");
    alpha_eps_ = param_reader.get_double("alpha_eps");
    G_c_ = param_reader.get_double("G_c");
    lame_coefficient_mu_     = param_reader.get_double("lame_coefficient_mu");
    lame_coefficient_lambda_ = param_reader.get_double("lame_coefficient_lambda");
    
    s_ = param_reader.get_double("sigma");

  }

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

    uvalues_.resize(n_q_points, Vector<double>(4));
    ugrads_.resize(n_q_points, vector<Tensor<1, 2> >(4));
    last_timestep_uvalues_.resize(n_q_points, Vector<double>(4));

    edc.GetValuesState("last_newton_solution", uvalues_);
    edc.GetGradsState("last_newton_solution", ugrads_);

    edc.GetValuesState("last_time_solution", last_timestep_uvalues_);

    const FEValuesExtractors::Vector displacements(0);
    const FEValuesExtractors::Scalar phasefield(2);
    const FEValuesExtractors::Scalar multiplier(3);

    Tensor<2,2> Identity;
    Identity[0][0] = 1.0;
    Identity[1][1] = 1.0;

    Tensor<2,2> zero_matrix;
    zero_matrix.clear();



    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        Tensor<2, 2> grad_u;
        grad_u.clear();
        grad_u[0][0] = ugrads_[q_point][0][0];
        grad_u[0][1] = ugrads_[q_point][0][1];
        grad_u[1][0] = ugrads_[q_point][1][0];
        grad_u[1][1] = ugrads_[q_point][1][1];

        Tensor<1, 2> u;
        u.clear();
        u[0] = uvalues_[q_point](0);
        u[1] = uvalues_[q_point](1);

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
        stress_term = lame_coefficient_lambda_ * tr_E * Identity
                      + 2 * lame_coefficient_mu_ * E;

        Tensor<2,2> stress_term_plus;
        Tensor<2,2> stress_term_minus;

        // Necessary because stress splitting does not work
        // in the very initial time step.
        if (this->GetTime() > 0.001)
          {
            decompose_stress(stress_term_plus, stress_term_minus,
                             E, tr_E, zero_matrix , 0.0,
                             lame_coefficient_lambda_,
                             lame_coefficient_mu_, false);
          }
        else
          {
            stress_term_plus = stress_term;
            stress_term_minus = 0;
          }


        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            //const Tensor<1, 2> phi_i_u = state_fe_values[displacements].value(i,q_point);
            const Tensor<2, 2> phi_i_grads_u = state_fe_values[displacements].gradient(i, q_point);
            const double phi_i_pf = state_fe_values[phasefield].value(i, q_point);
	    const Tensor<1, 2> phi_i_grads_pf = state_fe_values[phasefield].gradient(i, q_point);

            // Solid (Time-lagged version)
            local_vector(i) += scale
                               * (
				scalar_product(((1.0-constant_k_) * old_timestep_pf * old_timestep_pf + constant_k_) *
                                                 stress_term_plus, phi_i_grads_u)
                                  + scalar_product(stress_term_minus, phi_i_grads_u)
                                 ) * state_fe_values.JxW(q_point); 



            // Phase-field
            local_vector(i) += scale
                               * (
				// Main terms
				(1.0 - constant_k_) * scalar_product(stress_term_plus, E) * pf * phi_i_pf
                                 - G_c_/(alpha_eps_) * (1.0 - pf) * phi_i_pf
                                 + G_c_ * alpha_eps_ * grad_pf * phi_i_grads_pf
                               ) * state_fe_values.JxW(q_point);


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

    const FEValuesExtractors::Vector displacements(0);
    const FEValuesExtractors::Scalar phasefield(2);
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
    
          }

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
        stress_term = lame_coefficient_lambda_ * tr_E * Identity
                      + 2 * lame_coefficient_mu_ * E;

        Tensor<2,2> stress_term_plus;
        Tensor<2,2> stress_term_minus;

        // Necessary because stress splitting does not work
        // in the very initial time step.
        if (this->GetTime() > 0.001)
          {
            decompose_stress(stress_term_plus, stress_term_minus,
                             E, tr_E, zero_matrix , 0.0,
                             lame_coefficient_lambda_,
                             lame_coefficient_mu_, false);
          }
        else
          {
            stress_term_plus = stress_term;
            stress_term_minus = 0;
          }



        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {

            const Tensor<2, 2> E_LinU = 0.5
                                        * (phi_grads_u[i] + transpose(phi_grads_u[i]));

            const double tr_E_LinU = trace(E_LinU);

            Tensor<2,2> stress_term_LinU;
            stress_term_LinU = lame_coefficient_lambda_ * tr_E_LinU * Identity
                               + 2 * lame_coefficient_mu_ * E_LinU;

            Tensor<2,2> stress_term_plus_LinU;
            Tensor<2,2> stress_term_minus_LinU;


            // Necessary because stress splitting does not work
            // in the very initial time step.
            if (this->GetTime() > 0.001)
              {
                decompose_stress(stress_term_plus_LinU, stress_term_minus_LinU,
                                 E, tr_E, E_LinU, tr_E_LinU,
                                 lame_coefficient_lambda_,
                                 lame_coefficient_mu_,
                                 true);
              }
            else
              {
                stress_term_plus_LinU = stress_term_LinU;
                stress_term_minus_LinU = 0;
              }



            for (unsigned int j = 0; j < n_dofs_per_element; j++)
              {
                // Solid (time-lagged version)
                local_matrix(j, i) += scale
                                      * ( 
					scalar_product(((1-constant_k_) * old_timestep_pf * old_timestep_pf + constant_k_) *
                                                        stress_term_plus_LinU, phi_grads_u[j]) // du
                                         + scalar_product(stress_term_minus_LinU, phi_grads_u[j]) // du
                                        ) * state_fe_values.JxW(q_point);

                // Phase-field
                local_matrix(j, i) += scale
                                      * (
					// Main terms
					(1-constant_k_) * (scalar_product(stress_term_plus_LinU, E)
                                                             + scalar_product(stress_term_plus, E_LinU)) * pf * phi_pf[j] // du
                                        +(1-constant_k_) * scalar_product(stress_term_plus, E) * phi_pf[i] * phi_pf[j] // d phi
                                        + G_c_/(alpha_eps_) * phi_pf[i] * phi_pf[j] // d phi
                                        + G_c_ * alpha_eps_ * phi_grads_pf[i] * phi_grads_pf[j] // d phi
                                        ) * state_fe_values.JxW(q_point);


		
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

  void
  ElementRightHandSide(const EDC<DH, VECTOR, dealdim> & /*edc*/,
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
   */

  unsigned int
  GetControlNBlocks() const
  {
    return 1;
  }

  unsigned int
  GetStateNBlocks() const
  {
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

  vector<Vector<double> > last_timestep_uvalues_;

  // face values
  vector<vector<Tensor<1, dealdim> > > ufacegrads_;

  vector<unsigned int> state_block_component_;
  vector<unsigned int> control_block_component_;

  double constant_k_, alpha_eps_,
         G_c_, lame_coefficient_mu_, lame_coefficient_lambda_, s_;

};
#endif
