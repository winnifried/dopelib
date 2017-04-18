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

#ifndef LOCALPDE_
#define LOCALPDE_

#include <interfaces/pdeinterface.h>
#include "ale_transformations.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

template<
template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
         template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
         template<int, int> class DH, typename VECTOR, int dealdim>
class LocalPDE : public PDEInterface<EDC, FaceDataContainer, DH, VECTOR,
  dealdim>
{
public:

  static void
  declare_params(ParameterReader &param_reader)
  {
    param_reader.SetSubsection("Local PDE parameters");
    param_reader.declare_entry("density_fluid", "0.0", Patterns::Double(0));
    param_reader.declare_entry("density_structure", "0.0",
                               Patterns::Double(0));
    param_reader.declare_entry("viscosity", "0.0", Patterns::Double(0));
    param_reader.declare_entry("alpha_u", "0.0", Patterns::Double(0));
    param_reader.declare_entry("mu", "0.0", Patterns::Double(0));
    param_reader.declare_entry("poisson_ratio_nu", "0.0",
                               Patterns::Double(0));
  }

  LocalPDE(ParameterReader &param_reader) :
    control_block_components_(2, 0), state_block_components_(5, 0)
  {
    // control block components
    control_block_components_[0] = 0;
    control_block_components_[1] = 1;

    // state block components
    state_block_components_[2] = 1; // displacement x
    state_block_components_[3] = 1; // displacement y
    state_block_components_[4] = 2; // pressure

    param_reader.SetSubsection("Local PDE parameters");
    density_fluid_ = param_reader.get_double("density_fluid");
    density_structure_ = param_reader.get_double("density_structure");
    viscosity_ = param_reader.get_double("viscosity");
    alpha_u_ = param_reader.get_double("alpha_u");

    lame_coefficient_mu_ = param_reader.get_double("mu");
    poisson_ratio_nu_ = param_reader.get_double("poisson_ratio_nu");
    lame_coefficient_lambda_ =
      (2 * poisson_ratio_nu_ * lame_coefficient_mu_)
      / (1.0 - 2 * poisson_ratio_nu_);
  }

  bool
  HasFaces() const
  {
    // should be true
    return true;
  }

  // Domain values for elements
  void
  ElementEquation(const EDC<DH, VECTOR, dealdim> &edc,
                  dealii::Vector<double> &local_vector, double scale,
                  double /*scale_ico*/)
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();
    unsigned int material_id = edc.GetMaterialId();
    double element_diameter = edc.GetElementDiameter();

    assert(this->problem_type_ == "state");

    uvalues_.resize(n_q_points, Vector<double>(5));
    ugrads_.resize(n_q_points, vector<Tensor<1, 2> >(5));

    // Getting state values
    edc.GetValuesState("last_newton_solution", uvalues_);
    edc.GetGradsState("last_newton_solution", ugrads_);

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Vector displacements(2);
    const FEValuesExtractors::Scalar pressure(4);

    const Tensor<2, dealdim> Identity = ALE_Transformations::get_Identity<
                                        dealdim>();

    // fluid
    if (material_id == 0)
      {

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            const Tensor<2, dealdim> pI = ALE_Transformations::get_pI<dealdim>(
                                            q_point, uvalues_);

            const Tensor<1, dealdim> v = ALE_Transformations::get_v<dealdim>(
                                           q_point, uvalues_);

            const Tensor<2, dealdim> grad_v = ALE_Transformations::get_grad_v<
                                              dealdim>(q_point, ugrads_);

            const Tensor<2, dealdim> grad_v_T =
              ALE_Transformations::get_grad_v_T<dealdim>(grad_v);

            const Tensor<2, dealdim> grad_u = ALE_Transformations::get_grad_u<
                                              dealdim>(q_point, ugrads_);

            const Tensor<2, dealdim> F = ALE_Transformations::get_F<dealdim>(
                                           q_point, ugrads_);

            const Tensor<2, dealdim> F_Inverse =
              ALE_Transformations::get_F_Inverse<dealdim>(F);

            const Tensor<2, dealdim> F_Inverse_T =
              ALE_Transformations::get_F_Inverse_T<dealdim>(F_Inverse);

            const double J = ALE_Transformations::get_J<dealdim>(F);

            const Tensor<2, dealdim> sigma_ALE =
              NSE_in_ALE::get_stress_fluid_except_pressure_ALE<dealdim>(
                density_fluid_, viscosity_, grad_v, grad_v_T, F_Inverse,
                F_Inverse_T);

            const Tensor<2, dealdim> stress_fluid =
              (J * sigma_ALE * F_Inverse_T);

            const Tensor<1, dealdim> convection_fluid = density_fluid_ * J
                                                        * (grad_v * F_Inverse * v);

            const Tensor<2, dealdim> fluid_pressure = (-pI * J * F_Inverse_T);

            const double incompressiblity_fluid =
              NSE_in_ALE::get_Incompressibility_ALE<dealdim>(q_point,
                                                             ugrads_);

            for (unsigned int i = 0; i < n_dofs_per_element; i++)
              {
                const Tensor<1, 2> phi_i_v = state_fe_values[velocities].value(i,
                                             q_point);
                const Tensor<2, 2> phi_i_grads_v =
                  state_fe_values[velocities].gradient(i, q_point);
                const double phi_i_p = state_fe_values[pressure].value(i,
                                                                       q_point);
                const Tensor<2, 2> phi_i_grads_u =
                  state_fe_values[displacements].gradient(i, q_point);

                local_vector(i) += scale
                                   * (convection_fluid * phi_i_v
                                      + scalar_product(fluid_pressure, phi_i_grads_v)
                                      + scalar_product(stress_fluid, phi_i_grads_v)
                                      + incompressiblity_fluid * phi_i_p
                                      + alpha_u_ * element_diameter * element_diameter
                                      * scalar_product(grad_u, phi_i_grads_u))
                                   * state_fe_values.JxW(q_point);
              }
          }

      } // end material_id == 0
    else if (material_id == 1)
      {
        // structure, STVK
        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            const Tensor<1, 2> v = ALE_Transformations::get_v<2>(q_point,
                                                                 uvalues_);

            const Tensor<2, 2> F = ALE_Transformations::get_F<2>(q_point,
                                                                 ugrads_);

            const Tensor<2, 2> F_T = ALE_Transformations::get_F_T<2>(F);

            const Tensor<2, 2> E = Structure_Terms_in_ALE::get_E<2>(F_T, F,
                                                                    Identity);

            const double tr_E = Structure_Terms_in_ALE::get_tr_E<2>(E);

            const Tensor<2, 2> sigma_structure_ALE = (F
                                                      * (lame_coefficient_lambda_ * tr_E * Identity
                                                         + 2 * lame_coefficient_mu_ * E));

            for (unsigned int i = 0; i < n_dofs_per_element; i++)
              {
                const Tensor<2, 2> phi_i_grads_v =
                  state_fe_values[velocities].gradient(i, q_point);
                const double phi_i_p = state_fe_values[pressure].value(i,
                                                                       q_point);
                const Tensor<1, 2> phi_i_u = state_fe_values[displacements].value(
                                               i, q_point);

                local_vector(i) += scale
                                   * (scalar_product(sigma_structure_ALE, phi_i_grads_v)
                                      - density_structure_ * v * phi_i_u
                                      + uvalues_[q_point](4) * phi_i_p)
                                   * state_fe_values.JxW(q_point);
              }
          }
      } // end material_id == 1

  }

  void
  ElementMatrix(const EDC<DH, VECTOR, dealdim> &edc,
                dealii::FullMatrix<double> &local_matrix, double /*scale*/,
                double /*scale_ico*/)
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();
    unsigned int material_id = edc.GetMaterialId();
    double element_diameter = edc.GetElementDiameter();

    uvalues_.resize(n_q_points, Vector<double>(5));
    ugrads_.resize(n_q_points, vector<Tensor<1, 2> >(5));

    // Getting previous Newton solutions via "last_newton_solution"
    // for the nonlinear convection term for ElementEquation
    // (PDE). In contrast the equations for
    // "adjoint", "tangent", etc. need the "state" values
    // for the linearized convection term.
    if (this->problem_type_ == "state")
      {
        edc.GetValuesState("last_newton_solution", uvalues_);
        edc.GetGradsState("last_newton_solution", ugrads_);
      }
    else
      {
        edc.GetValuesState("state", uvalues_);
        edc.GetGradsState("state", ugrads_);
      }

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Vector displacements(2);
    const FEValuesExtractors::Scalar pressure(4);

    std::vector<Tensor<1, 2> > phi_v(n_dofs_per_element);
    std::vector<Tensor<2, 2> > phi_grads_v(n_dofs_per_element);
    std::vector<Tensor<1, 2> > phi_u(n_dofs_per_element);
    std::vector<Tensor<2, 2> > phi_grads_u(n_dofs_per_element);
    std::vector<double> phi_p(n_dofs_per_element);

    const Tensor<2, dealdim> Identity = ALE_Transformations::get_Identity<
                                        dealdim>();

    if (material_id == 0)
      {

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            for (unsigned int k = 0; k < n_dofs_per_element; k++)
              {
                phi_p[k] = state_fe_values[pressure].value(k, q_point);
                phi_v[k] = state_fe_values[velocities].value(k, q_point);
                phi_grads_v[k] = state_fe_values[velocities].gradient(k, q_point);
                phi_u[k] = state_fe_values[displacements].value(k, q_point);
                phi_grads_u[k] = state_fe_values[displacements].gradient(k,
                                                                         q_point);
              }

            const Tensor<2, dealdim> pI = ALE_Transformations::get_pI<dealdim>(
                                            q_point, uvalues_);
            const Tensor<1, dealdim> v = ALE_Transformations::get_v<dealdim>(
                                           q_point, uvalues_);

            const Tensor<2, dealdim> grad_v = ALE_Transformations::get_grad_v<
                                              dealdim>(q_point, ugrads_);

            const Tensor<2, dealdim> grad_v_T =
              ALE_Transformations::get_grad_v_T<dealdim>(grad_v);
            const Tensor<2, dealdim> F = ALE_Transformations::get_F<dealdim>(
                                           q_point, ugrads_);
            const Tensor<2, dealdim> F_Inverse =
              ALE_Transformations::get_F_Inverse<dealdim>(F);

            const Tensor<2, dealdim> F_Inverse_T =
              ALE_Transformations::get_F_Inverse_T<dealdim>(F_Inverse);
            const double J = ALE_Transformations::get_J<dealdim>(F);

            const Tensor<2, dealdim> sigma_ALE =
              NSE_in_ALE::get_stress_fluid_ALE<dealdim>(density_fluid_,
                                                        viscosity_, pI, grad_v, grad_v_T, F_Inverse, F_Inverse_T);

            for (unsigned int j = 0; j < n_dofs_per_element; j++)
              {
                const Tensor<2, dealdim> pI_LinP =
                  ALE_Transformations::get_pI_LinP<dealdim>(phi_p[j]);
                const Tensor<2, dealdim> grad_v_LinV =
                  ALE_Transformations::get_grad_v_LinV<dealdim>(phi_grads_v[j]);
                const double J_LinU = ALE_Transformations::get_J_LinU<dealdim>(
                                        q_point, ugrads_, phi_grads_u[j]);

                const Tensor<2, dealdim> J_F_Inverse_T_LinU =
                  ALE_Transformations::get_J_F_Inverse_T_LinU<dealdim>(
                    phi_grads_u[j]);
                const Tensor<2, dealdim> F_Inverse_LinU =
                  ALE_Transformations::get_F_Inverse_LinU(phi_grads_u[j], J,
                                                          J_LinU, q_point, ugrads_);

                const Tensor<2, dealdim> stress_fluid_ALE_1st_term_LinAll =
                  NSE_in_ALE::get_stress_fluid_ALE_1st_term_LinAll_short<dealdim>(
                    pI, F_Inverse_T, J_F_Inverse_T_LinU, pI_LinP, J);

                const double incompressibility_ALE_LinAll =
                  NSE_in_ALE::get_Incompressibility_ALE_LinAll<dealdim>(
                    phi_grads_v[j], phi_grads_u[j], q_point, ugrads_);

                const Tensor<2, dealdim> stress_fluid_ALE_2nd_term_LinAll =
                  NSE_in_ALE::get_stress_fluid_ALE_2nd_term_LinAll_short(
                    J_F_Inverse_T_LinU, sigma_ALE, grad_v, grad_v_LinV,
                    F_Inverse, F_Inverse_LinU, J, viscosity_, density_fluid_);

                const Tensor<1, dealdim> convection_fluid_LinAll_short =
                  NSE_in_ALE::get_Convection_LinAll_short<dealdim>(
                    phi_grads_v[j], phi_v[j], J, J_LinU, F_Inverse,
                    F_Inverse_LinU, v, grad_v, density_fluid_);

                for (unsigned int i = 0; i < n_dofs_per_element; i++)
                  {

                    local_matrix(i, j) += (convection_fluid_LinAll_short
                                           * phi_v[i]
                                           + scalar_product(stress_fluid_ALE_2nd_term_LinAll,
                                                            phi_grads_v[i])
                                           + scalar_product(stress_fluid_ALE_1st_term_LinAll,
                                                            phi_grads_v[i])
                                           + incompressibility_ALE_LinAll * phi_p[i]
                                           + alpha_u_ * element_diameter * element_diameter
                                           * scalar_product(phi_grads_u[j], phi_grads_u[i]))
                                          * state_fe_values.JxW(q_point);

                  }
              }
          }

      } // end material_id ==0
    else if (material_id == 1)
      {
        // structure, STVK
        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            for (unsigned int k = 0; k < n_dofs_per_element; k++)
              {
                phi_p[k] = state_fe_values[pressure].value(k, q_point);
                phi_v[k] = state_fe_values[velocities].value(k, q_point);
                phi_grads_v[k] = state_fe_values[velocities].gradient(k, q_point);
                phi_u[k] = state_fe_values[displacements].value(k, q_point);
                phi_grads_u[k] = state_fe_values[displacements].gradient(k,
                                                                         q_point);
              }

            const Tensor<2, dealdim> F = ALE_Transformations::get_F<dealdim>(
                                           q_point, ugrads_);

            const Tensor<2, dealdim> F_T =
              ALE_Transformations::get_F_T<dealdim>(F);

            const Tensor<2, dealdim> E = Structure_Terms_in_ALE::get_E<dealdim>(
                                           F_T, F, Identity);

            const double tr_E = Structure_Terms_in_ALE::get_tr_E<dealdim>(E);

            for (unsigned int j = 0; j < n_dofs_per_element; j++)
              {
                const Tensor<2, dealdim> F_LinU = ALE_Transformations::get_F_LinU<
                                                  dealdim>(phi_grads_u[j]);

                const Tensor<2, dealdim> E_LinU = 0.5
                                                  * (transpose(F_LinU) * F + transpose(F) * F_LinU);

                const double tr_E_LinU = Structure_Terms_in_ALE::get_tr_E_LinU<
                                         dealdim>(q_point, ugrads_, phi_grads_u[j]);

                Tensor<2, dealdim> piola_kirchhoff_stress_structure_STVK_LinALL;
                piola_kirchhoff_stress_structure_STVK_LinALL =
                  lame_coefficient_lambda_
                  * (F_LinU * tr_E * Identity + F * tr_E_LinU * Identity)
                  + 2 * lame_coefficient_mu_ * (F_LinU * E + F * E_LinU);

                for (unsigned int i = 0; i < n_dofs_per_element; i++)
                  {
                    local_matrix(i, j) += (scalar_product(
                                             piola_kirchhoff_stress_structure_STVK_LinALL,
                                             phi_grads_v[i]) - density_structure_ * phi_v[j] * phi_u[i]
                                           + phi_p[j] * phi_p[i]) * state_fe_values.JxW(q_point);
                  }
              }
          }
      } // end material_id ==1

  }

  void
  ElementEquation_U(const EDC<DH, VECTOR, dealdim> &edc,
                    dealii::Vector<double> &local_vector, double scale,
                    double /*scale_ico*/)
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();
    unsigned int material_id = edc.GetMaterialId();
    double element_diameter = edc.GetElementDiameter();

    assert(this->problem_type_ == "adjoint");

    zvalues_.resize(n_q_points, Vector<double>(5));
    zgrads_.resize(n_q_points, vector<Tensor<1, 2> >(5));

    edc.GetValuesState("last_newton_solution", zvalues_);
    edc.GetGradsState("last_newton_solution", zgrads_);

    z_state_values_.resize(n_q_points, Vector<double>(5));
    z_state_grads_.resize(n_q_points, vector<Tensor<1, 2> >(5));

    edc.GetValuesState("state", z_state_values_);
    edc.GetGradsState("state", z_state_grads_);

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Vector displacements(2);
    const FEValuesExtractors::Scalar pressure(4);

    std::vector<Tensor<1, 2> > phi_v(n_dofs_per_element);
    std::vector<Tensor<2, 2> > phi_grads_v(n_dofs_per_element);
    std::vector<Tensor<1, 2> > phi_u(n_dofs_per_element);
    std::vector<Tensor<2, 2> > phi_grads_u(n_dofs_per_element);
    std::vector<double> phi_p(n_dofs_per_element);

    const Tensor<2, dealdim> Identity = ALE_Transformations::get_Identity<
                                        dealdim>();

    // fluid fsi
    if (material_id == 0)
      {

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            for (unsigned int k = 0; k < n_dofs_per_element; k++)
              {
                phi_p[k] = state_fe_values[pressure].value(k, q_point);
                phi_v[k] = state_fe_values[velocities].value(k, q_point);
                phi_grads_v[k] = state_fe_values[velocities].gradient(k, q_point);
                phi_u[k] = state_fe_values[displacements].value(k, q_point);
                phi_grads_u[k] = state_fe_values[displacements].gradient(k,
                                                                         q_point);
              }

            // adjoint values and grads
            Tensor<2, 2> zv_grads;
            zv_grads.clear();
            zv_grads[0][0] = zgrads_[q_point][0][0];
            zv_grads[0][1] = zgrads_[q_point][0][1];
            zv_grads[1][0] = zgrads_[q_point][1][0];
            zv_grads[1][1] = zgrads_[q_point][1][1];

            Tensor<1, 2> zv;
            zv.clear();
            zv[0] = zvalues_[q_point](0);
            zv[1] = zvalues_[q_point](1);

            double zp = zvalues_[q_point](4);

            Tensor<2, 2> zu_grads;
            zu_grads.clear();
            zu_grads[0][0] = zgrads_[q_point][2][0];
            zu_grads[0][1] = zgrads_[q_point][2][1];
            zu_grads[1][0] = zgrads_[q_point][3][0];
            zu_grads[1][1] = zgrads_[q_point][3][1];

            // state values which contains
            // solution from previous Newton step
            // Necessary for fluid convection term
            Tensor<2, 2> zv_state_grads;
            zv_state_grads.clear();
            zv_state_grads[0][0] = z_state_grads_[q_point][0][0];
            zv_state_grads[0][1] = z_state_grads_[q_point][0][1];
            zv_state_grads[1][0] = z_state_grads_[q_point][1][0];
            zv_state_grads[1][1] = z_state_grads_[q_point][1][1];

            Tensor<1, 2> zv_state;
            zv_state.clear();
            zv_state[0] = z_state_values_[q_point](0);
            zv_state[1] = z_state_values_[q_point](1);

            Tensor<2, 2> zpI_state;
            zpI_state.clear();
            zpI_state[0][0] = z_state_values_[q_point](4);
            zpI_state[0][1] = 0.0;
            zpI_state[1][0] = 0.0;
            zpI_state[1][1] = z_state_values_[q_point](4);

            // state values and grads
            const Tensor<2, dealdim> F = ALE_Transformations::get_F<dealdim>(
                                           q_point, z_state_grads_);

            const Tensor<2, dealdim> F_Inverse =
              ALE_Transformations::get_F_Inverse<dealdim>(F);

            const Tensor<2, dealdim> F_Inverse_T =
              ALE_Transformations::get_F_Inverse_T<dealdim>(F_Inverse);

            const double J = ALE_Transformations::get_J<dealdim>(F);

            const Tensor<2, dealdim> sigma_ALE =
              NSE_in_ALE::get_stress_fluid_ALE<dealdim>(density_fluid_,
                                                        viscosity_, zpI_state, zv_state_grads,
                                                        transpose(zv_state_grads), F_Inverse, F_Inverse_T);

            for (unsigned int j = 0; j < n_dofs_per_element; j++)
              {
                const Tensor<2, dealdim> pI_LinP =
                  ALE_Transformations::get_pI_LinP<dealdim>(phi_p[j]);

                const Tensor<2, dealdim> grad_v_LinV =
                  ALE_Transformations::get_grad_v_LinV<dealdim>(phi_grads_v[j]);

                const double J_LinU = ALE_Transformations::get_J_LinU<dealdim>(
                                        q_point, z_state_grads_, phi_grads_u[j]);


                const Tensor<2, dealdim> J_F_Inverse_T_LinU =
                  ALE_Transformations::get_J_F_Inverse_T_LinU<dealdim>(
                    phi_grads_u[j]);

                const Tensor<2, dealdim> F_Inverse_LinU =
                  ALE_Transformations::get_F_Inverse_LinU(phi_grads_u[j], J,
                                                          J_LinU, q_point, z_state_grads_);

                // four main equations
                const Tensor<2, dealdim> stress_fluid_ALE_1st_term_LinAll =
                  NSE_in_ALE::get_stress_fluid_ALE_1st_term_LinAll_short<dealdim>(
                    zpI_state, F_Inverse_T, J_F_Inverse_T_LinU, pI_LinP, J);

                const double incompressibility_ALE_LinAll =
                  NSE_in_ALE::get_Incompressibility_ALE_LinAll<dealdim>(
                    phi_grads_v[j], phi_grads_u[j], q_point, z_state_grads_);

                const Tensor<2, dealdim> stress_fluid_ALE_2nd_term_LinAll =
                  NSE_in_ALE::get_stress_fluid_ALE_2nd_term_LinAll_short(
                    J_F_Inverse_T_LinU, sigma_ALE, zv_state_grads,
                    grad_v_LinV, F_Inverse, F_Inverse_LinU, J, viscosity_,
                    density_fluid_);

                const Tensor<1, dealdim> convection_fluid_LinAll_short =
                  NSE_in_ALE::get_Convection_LinAll_short<dealdim>(
                    phi_grads_v[j], phi_v[j], J, J_LinU, F_Inverse,
                    F_Inverse_LinU, zv_state, zv_state_grads, density_fluid_);

                local_vector(j) += scale
                                   * (convection_fluid_LinAll_short * zv
                                      + scalar_product(stress_fluid_ALE_2nd_term_LinAll,
                                                       zv_grads)
                                      + scalar_product(stress_fluid_ALE_1st_term_LinAll,
                                                       zv_grads) + incompressibility_ALE_LinAll * zp
                                      + alpha_u_ * element_diameter * element_diameter
                                      * scalar_product(phi_grads_u[j], zu_grads))
                                   * state_fe_values.JxW(q_point);

              }
          }

      } // end material_id == 0
    else if (material_id == 1)
      {

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            for (unsigned int k = 0; k < n_dofs_per_element; k++)
              {
                phi_p[k] = state_fe_values[pressure].value(k, q_point);
                phi_v[k] = state_fe_values[velocities].value(k, q_point);
                phi_grads_v[k] = state_fe_values[velocities].gradient(k, q_point);
                phi_u[k] = state_fe_values[displacements].value(k, q_point);
                phi_grads_u[k] = state_fe_values[displacements].gradient(k,
                                                                         q_point);
              }

            // adjoint values and grads
            Tensor<2, 2> zv_grads;
            zv_grads.clear();
            zv_grads[0][0] = zgrads_[q_point][0][0];
            zv_grads[0][1] = zgrads_[q_point][0][1];
            zv_grads[1][0] = zgrads_[q_point][1][0];
            zv_grads[1][1] = zgrads_[q_point][1][1];

            Tensor<1, 2> zu;
            zu.clear();
            zu[0] = zvalues_[q_point](2);
            zu[1] = zvalues_[q_point](3);

            double zp = zvalues_[q_point](4);

            // state values and grads
            const Tensor<2, dealdim> F = ALE_Transformations::get_F<dealdim>(
                                           q_point, z_state_grads_);

            const Tensor<2, dealdim> F_T =
              ALE_Transformations::get_F_T<dealdim>(F);

            const Tensor<2, dealdim> E = Structure_Terms_in_ALE::get_E<dealdim>(
                                           F_T, F, Identity);

            const double tr_E = Structure_Terms_in_ALE::get_tr_E<dealdim>(E);

            for (unsigned int j = 0; j < n_dofs_per_element; j++)
              {
                const Tensor<2, dealdim> F_LinU = ALE_Transformations::get_F_LinU<
                                                  dealdim>(phi_grads_u[j]);

                const Tensor<2, dealdim> E_LinU = 0.5
                                                  * (transpose(F_LinU) * F + transpose(F) * F_LinU);

                const double tr_E_LinU = Structure_Terms_in_ALE::get_tr_E_LinU<
                                         dealdim>(q_point, z_state_grads_, phi_grads_u[j]);

                Tensor<2, dealdim> piola_kirchhoff_stress_structure_STVK_LinALL;
                piola_kirchhoff_stress_structure_STVK_LinALL =
                  lame_coefficient_lambda_
                  * (F_LinU * tr_E * Identity + F * tr_E_LinU * Identity)
                  + 2 * lame_coefficient_mu_ * (F_LinU * E + F * E_LinU);

                local_vector(j) += scale
                                   * (scalar_product(
                                        piola_kirchhoff_stress_structure_STVK_LinALL, zv_grads)
                                      - density_structure_ * phi_v[j] * zu + phi_p[j] * zp)
                                   * state_fe_values.JxW(q_point);

              }
          }
      } // end material_id == 1

  }

  void
  ElementEquation_UT(const EDC<DH, VECTOR, dealdim> &edc,
                     dealii::Vector<double> &local_vector, double scale,
                     double /*scale_ico*/)
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();
    unsigned int material_id = edc.GetMaterialId();
    double element_diameter = edc.GetElementDiameter();

    assert(this->problem_type_ == "tangent");

    duvalues_.resize(n_q_points, Vector<double>(5));
    dugrads_.resize(n_q_points, vector<Tensor<1, 2> >(5));

    edc.GetValuesState("last_newton_solution", duvalues_);
    edc.GetGradsState("last_newton_solution", dugrads_);

    du_state_values_.resize(n_q_points, Vector<double>(5));
    du_state_grads_.resize(n_q_points, vector<Tensor<1, 2> >(5));

    edc.GetValuesState("state", du_state_values_);
    edc.GetGradsState("state", du_state_grads_);

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Vector displacements(2);
    const FEValuesExtractors::Scalar pressure(4);

    const Tensor<2, dealdim> Identity = ALE_Transformations::get_Identity<
                                        dealdim>();

    if (material_id == 0)
      {

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            Tensor<2, dealdim> du_pI;
            du_pI.clear();
            du_pI[0][0] = -duvalues_[q_point](4);
            du_pI[1][1] = -duvalues_[q_point](4);

            Tensor<2, 2> duv_grads;
            duv_grads.clear();
            duv_grads[0][0] = dugrads_[q_point][0][0];
            duv_grads[0][1] = dugrads_[q_point][0][1];
            duv_grads[1][0] = dugrads_[q_point][1][0];
            duv_grads[1][1] = dugrads_[q_point][1][1];

            Tensor<1, 2> duv;
            duv.clear();
            duv[0] = duvalues_[q_point](0);
            duv[1] = duvalues_[q_point](1);

            Tensor<2, 2> dupI;
            dupI.clear();
            dupI[0][0] = duvalues_[q_point](4);
            dupI[0][1] = 0.0;
            dupI[1][0] = 0.0;
            dupI[1][1] = duvalues_[q_point](4);

            Tensor<2, 2> duu_grads;
            duu_grads.clear();
            duu_grads[0][0] = dugrads_[q_point][2][0];
            duu_grads[0][1] = dugrads_[q_point][2][1];
            duu_grads[1][0] = dugrads_[q_point][3][0];
            duu_grads[1][1] = dugrads_[q_point][3][1];

            Tensor<2, 2> dupI_state;
            dupI_state.clear();
            dupI_state[0][0] = du_state_values_[q_point](4);
            dupI_state[0][1] = 0.0;
            dupI_state[1][0] = 0.0;
            dupI_state[1][1] = du_state_values_[q_point](4);

            Tensor<2, 2> duv_state_grads;
            duv_state_grads.clear();
            duv_state_grads[0][0] = du_state_grads_[q_point][0][0];
            duv_state_grads[0][1] = du_state_grads_[q_point][0][1];
            duv_state_grads[1][0] = du_state_grads_[q_point][1][0];
            duv_state_grads[1][1] = du_state_grads_[q_point][1][1];

            Tensor<1, 2> duv_state;
            duv_state.clear();
            duv_state[0] = du_state_values_[q_point](0);
            duv_state[1] = du_state_values_[q_point](1);

            const Tensor<2, dealdim> F = ALE_Transformations::get_F<dealdim>(
                                           q_point, du_state_grads_);

            const Tensor<2, dealdim> F_Inverse =
              ALE_Transformations::get_F_Inverse<dealdim>(F);

            const Tensor<2, dealdim> F_Inverse_T =
              ALE_Transformations::get_F_Inverse_T<dealdim>(F_Inverse);

            const double J = ALE_Transformations::get_J<dealdim>(F);

            const Tensor<2, dealdim> sigma_ALE =
              NSE_in_ALE::get_stress_fluid_ALE<dealdim>(density_fluid_,
                                                        viscosity_, dupI_state, duv_state_grads,
                                                        transpose(duv_state_grads), F_Inverse, F_Inverse_T);

            const double J_LinU = ALE_Transformations::get_J_LinU<dealdim>(
                                    q_point, du_state_grads_, duu_grads);

            const Tensor<2, dealdim> F_Inverse_LinU =
              ALE_Transformations::get_F_Inverse_LinU<dealdim>(duu_grads, J,
                                                               J_LinU, q_point, du_state_grads_);

            const Tensor<2, dealdim> J_F_Inverse_T_LinU =
              ALE_Transformations::get_J_F_Inverse_T_LinU<dealdim>(duu_grads);

            const Tensor<2, dealdim> stress_fluid_ALE_1st_term_LinAll =
              NSE_in_ALE::get_stress_fluid_ALE_1st_term_LinAll_short<dealdim>(
                dupI_state, F_Inverse_T, J_F_Inverse_T_LinU, dupI, J);

            const double incompressibility_ALE_LinAll =
              NSE_in_ALE::get_Incompressibility_ALE_LinAll<dealdim>(duv_grads,
                                                                    duu_grads, q_point, du_state_grads_);

            const Tensor<2, dealdim> stress_fluid_ALE_2nd_term_LinAll =
              NSE_in_ALE::get_stress_fluid_ALE_2nd_term_LinAll_short(
                J_F_Inverse_T_LinU, sigma_ALE, duv_state_grads, duv_grads,
                F_Inverse, F_Inverse_LinU, J, viscosity_, density_fluid_);

            const Tensor<1, dealdim> convection_fluid_LinAll_short =
              NSE_in_ALE::get_Convection_LinAll_short<dealdim>(duv_grads, duv,
                                                               J, J_LinU, F_Inverse, F_Inverse_LinU, duv_state,
                                                               duv_state_grads, density_fluid_);

            for (unsigned int i = 0; i < n_dofs_per_element; i++)
              {
                const Tensor<1, 2> phi_i_v = state_fe_values[velocities].value(i,
                                             q_point);
                const Tensor<2, 2> phi_i_grads_v =
                  state_fe_values[velocities].gradient(i, q_point);
                const double phi_i_p = state_fe_values[pressure].value(i,
                                                                       q_point);
                const Tensor<2, 2> phi_i_grads_u =
                  state_fe_values[displacements].gradient(i, q_point);

                local_vector(i) += scale
                                   * (convection_fluid_LinAll_short * phi_i_v
                                      + scalar_product(stress_fluid_ALE_2nd_term_LinAll,
                                                       phi_i_grads_v)
                                      + scalar_product(stress_fluid_ALE_1st_term_LinAll,
                                                       phi_i_grads_v)
                                      + incompressibility_ALE_LinAll * phi_i_p
                                      + alpha_u_ * element_diameter * element_diameter
                                      * scalar_product(duu_grads, phi_i_grads_u))
                                   * state_fe_values.JxW(q_point);

              }
          }

      } // material_id ==0
    else if (material_id == 1)
      {

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            Tensor<1, 2> duv;
            duv.clear();
            duv[0] = duvalues_[q_point](0);
            duv[1] = duvalues_[q_point](1);

            double dup = duvalues_[q_point](4);

            Tensor<2, 2> duu_grads;
            duu_grads.clear();
            duu_grads[0][0] = dugrads_[q_point][2][0];
            duu_grads[0][1] = dugrads_[q_point][2][1];
            duu_grads[1][0] = dugrads_[q_point][3][0];
            duu_grads[1][1] = dugrads_[q_point][3][1];

            // get state values
            const Tensor<2, dealdim> F = ALE_Transformations::get_F<dealdim>(
                                           q_point, du_state_grads_);

            const Tensor<2, dealdim> F_T =
              ALE_Transformations::get_F_T<dealdim>(F);

            const Tensor<2, dealdim> E = Structure_Terms_in_ALE::get_E<dealdim>(
                                           F_T, F, Identity);

            const double tr_E = Structure_Terms_in_ALE::get_tr_E<dealdim>(E);

            const Tensor<2, dealdim> F_LinU = ALE_Transformations::get_F_LinU<
                                              dealdim>(duu_grads);

            const Tensor<2, dealdim> E_LinU = 0.5
                                              * (transpose(F_LinU) * F + transpose(F) * F_LinU);

            const double tr_E_LinU = Structure_Terms_in_ALE::get_tr_E_LinU<
                                     dealdim>(q_point, du_state_grads_, duu_grads);

            Tensor<2, dealdim> piola_kirchhoff_stress_structure_STVK_LinALL;
            piola_kirchhoff_stress_structure_STVK_LinALL =
              lame_coefficient_lambda_
              * (F_LinU * tr_E * Identity + F * tr_E_LinU * Identity)
              + 2 * lame_coefficient_mu_ * (F_LinU * E + F * E_LinU);

            for (unsigned int i = 0; i < n_dofs_per_element; i++)
              {
                const Tensor<2, 2> phi_i_grads_v =
                  state_fe_values[velocities].gradient(i, q_point);
                const double phi_i_p = state_fe_values[pressure].value(i,
                                                                       q_point);

                const Tensor<1, 2> phi_i_u = state_fe_values[displacements].value(
                                               i, q_point);

                local_vector(i) += scale
                                   * (scalar_product(
                                        piola_kirchhoff_stress_structure_STVK_LinALL,
                                        phi_i_grads_v) - density_structure_ * duv * phi_i_u
                                      + dup * phi_i_p) * state_fe_values.JxW(q_point);
              }
          }

      } // material_id ==1

  }

  void
  ElementEquation_UTT(const EDC<DH, VECTOR, dealdim> &edc,
                      dealii::Vector<double> &local_vector, double scale,
                      double /*scale_ico*/)
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();
    unsigned int material_id = edc.GetMaterialId();
    double element_diameter = edc.GetElementDiameter();

    assert(this->problem_type_ == "adjoint_hessian");

    dzvalues_.resize(n_q_points, Vector<double>(5));
    dzgrads_.resize(n_q_points, vector<Tensor<1, 2> >(5));

    edc.GetValuesState("last_newton_solution", dzvalues_);
    edc.GetGradsState("last_newton_solution", dzgrads_);

    dz_state_values_.resize(n_q_points, Vector<double>(5));
    dz_state_grads_.resize(n_q_points, vector<Tensor<1, 2> >(5));

    edc.GetValuesState("state", dz_state_values_);
    edc.GetGradsState("state", dz_state_grads_);

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Vector displacements(2);
    const FEValuesExtractors::Scalar pressure(4);

    std::vector<Tensor<1, 2> > phi_v(n_dofs_per_element);
    std::vector<Tensor<2, 2> > phi_grads_v(n_dofs_per_element);
    std::vector<Tensor<1, 2> > phi_u(n_dofs_per_element);
    std::vector<Tensor<2, 2> > phi_grads_u(n_dofs_per_element);
    std::vector<double> phi_p(n_dofs_per_element);

    const Tensor<2, dealdim> Identity = ALE_Transformations::get_Identity<
                                        dealdim>();

    // fluid
    if (material_id == 0)
      {

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            for (unsigned int k = 0; k < n_dofs_per_element; k++)
              {
                phi_p[k] = state_fe_values[pressure].value(k, q_point);
                phi_v[k] = state_fe_values[velocities].value(k, q_point);
                phi_grads_v[k] = state_fe_values[velocities].gradient(k, q_point);
                phi_u[k] = state_fe_values[displacements].value(k, q_point);
                phi_grads_u[k] = state_fe_values[displacements].gradient(k,
                                                                         q_point);
              }

            // adjoint values and grads
            Tensor<2, 2> dzv_grads;
            dzv_grads.clear();
            dzv_grads[0][0] = dzgrads_[q_point][0][0];
            dzv_grads[0][1] = dzgrads_[q_point][0][1];
            dzv_grads[1][0] = dzgrads_[q_point][1][0];
            dzv_grads[1][1] = dzgrads_[q_point][1][1];

            Tensor<1, 2> dzv;
            dzv.clear();
            dzv[0] = dzvalues_[q_point](0);
            dzv[1] = dzvalues_[q_point](1);

            double dzp = dzvalues_[q_point](4);

            Tensor<2, 2> dzu_grads;
            dzu_grads.clear();
            dzu_grads[0][0] = dzgrads_[q_point][2][0];
            dzu_grads[0][1] = dzgrads_[q_point][2][1];
            dzu_grads[1][0] = dzgrads_[q_point][3][0];
            dzu_grads[1][1] = dzgrads_[q_point][3][1];

            // state values which contains
            // solution from previous Newton step
            // Necessary for fluid convection term
            Tensor<2, 2> dzv_state_grads;
            dzv_state_grads.clear();
            dzv_state_grads[0][0] = dz_state_grads_[q_point][0][0];
            dzv_state_grads[0][1] = dz_state_grads_[q_point][0][1];
            dzv_state_grads[1][0] = dz_state_grads_[q_point][1][0];
            dzv_state_grads[1][1] = dz_state_grads_[q_point][1][1];

            Tensor<1, 2> dzv_state;
            dzv_state.clear();
            dzv_state[0] = dz_state_values_[q_point](0);
            dzv_state[1] = dz_state_values_[q_point](1);

            Tensor<2, 2> dzpI_state;
            dzpI_state.clear();
            dzpI_state[0][0] = dz_state_values_[q_point](4);
            dzpI_state[0][1] = 0.0;
            dzpI_state[1][0] = 0.0;
            dzpI_state[1][1] = dz_state_values_[q_point](4);

            const Tensor<2, dealdim> F = ALE_Transformations::get_F<dealdim>(
                                           q_point, dz_state_grads_);

            const Tensor<2, dealdim> F_Inverse =
              ALE_Transformations::get_F_Inverse<dealdim>(F);

            const Tensor<2, dealdim> F_Inverse_T =
              ALE_Transformations::get_F_Inverse_T<dealdim>(F_Inverse);

            const double J = ALE_Transformations::get_J<dealdim>(F);

            const Tensor<2, dealdim> sigma_ALE =
              NSE_in_ALE::get_stress_fluid_ALE<dealdim>(density_fluid_,
                                                        viscosity_, dzpI_state, dzv_state_grads,
                                                        transpose(dzv_state_grads), F_Inverse, F_Inverse_T);

            for (unsigned int j = 0; j < n_dofs_per_element; j++)
              {
                const Tensor<2, dealdim> pI_LinP =
                  ALE_Transformations::get_pI_LinP<dealdim>(phi_p[j]);

                const Tensor<2, dealdim> grad_v_LinV =
                  ALE_Transformations::get_grad_v_LinV<dealdim>(phi_grads_v[j]);

                const double J_LinU = ALE_Transformations::get_J_LinU<dealdim>(
                                        q_point, dz_state_grads_, phi_grads_u[j]);


                const Tensor<2, dealdim> J_F_Inverse_T_LinU =
                  ALE_Transformations::get_J_F_Inverse_T_LinU<dealdim>(
                    phi_grads_u[j]);

                const Tensor<2, dealdim> F_Inverse_LinU =
                  ALE_Transformations::get_F_Inverse_LinU(phi_grads_u[j], J,
                                                          J_LinU, q_point, dz_state_grads_);

                const Tensor<2, dealdim> stress_fluid_ALE_1st_term_LinAll =
                  NSE_in_ALE::get_stress_fluid_ALE_1st_term_LinAll_short<dealdim>(
                    dzpI_state, F_Inverse_T, J_F_Inverse_T_LinU, pI_LinP, J);

                const double incompressibility_ALE_LinAll =
                  NSE_in_ALE::get_Incompressibility_ALE_LinAll<dealdim>(
                    phi_grads_v[j], phi_grads_u[j], q_point, dz_state_grads_);

                const Tensor<2, dealdim> stress_fluid_ALE_2nd_term_LinAll =
                  NSE_in_ALE::get_stress_fluid_ALE_2nd_term_LinAll_short(
                    J_F_Inverse_T_LinU, sigma_ALE, dzv_state_grads,
                    grad_v_LinV, F_Inverse, F_Inverse_LinU, J, viscosity_,
                    density_fluid_);

                const Tensor<1, dealdim> convection_fluid_LinAll_short =
                  NSE_in_ALE::get_Convection_LinAll_short<dealdim>(
                    phi_grads_v[j], phi_v[j], J, J_LinU, F_Inverse,
                    F_Inverse_LinU, dzv_state, dzv_state_grads,
                    density_fluid_);

                local_vector(j) += scale
                                   * (convection_fluid_LinAll_short * dzv
                                      + scalar_product(stress_fluid_ALE_2nd_term_LinAll,
                                                       dzv_grads)
                                      + scalar_product(stress_fluid_ALE_1st_term_LinAll,
                                                       dzv_grads) + incompressibility_ALE_LinAll * dzp
                                      + alpha_u_ * element_diameter * element_diameter
                                      * scalar_product(phi_grads_u[j], dzu_grads))
                                   * state_fe_values.JxW(q_point);

              }
          }
      } // material_id == 0
    else if (material_id == 1)
      {

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            for (unsigned int k = 0; k < n_dofs_per_element; k++)
              {
                phi_p[k] = state_fe_values[pressure].value(k, q_point);
                phi_v[k] = state_fe_values[velocities].value(k, q_point);
                phi_grads_v[k] = state_fe_values[velocities].gradient(k, q_point);
                phi_u[k] = state_fe_values[displacements].value(k, q_point);
                phi_grads_u[k] = state_fe_values[displacements].gradient(k,
                                                                         q_point);
              }

            // adjoint values and grads
            Tensor<2, 2> dzv_grads;
            dzv_grads.clear();
            dzv_grads[0][0] = dzgrads_[q_point][0][0];
            dzv_grads[0][1] = dzgrads_[q_point][0][1];
            dzv_grads[1][0] = dzgrads_[q_point][1][0];
            dzv_grads[1][1] = dzgrads_[q_point][1][1];

            Tensor<1, 2> dzu;
            dzu.clear();
            dzu[0] = dzvalues_[q_point](2);
            dzu[1] = dzvalues_[q_point](3);

            double dzp = dzvalues_[q_point](4);

            const Tensor<2, dealdim> F = ALE_Transformations::get_F<dealdim>(
                                           q_point, dz_state_grads_);

            const Tensor<2, dealdim> F_T =
              ALE_Transformations::get_F_T<dealdim>(F);

            const Tensor<2, dealdim> E = Structure_Terms_in_ALE::get_E<dealdim>(
                                           F_T, F, Identity);

            const double tr_E = Structure_Terms_in_ALE::get_tr_E<dealdim>(E);

            for (unsigned int j = 0; j < n_dofs_per_element; j++)
              {
                const Tensor<2, dealdim> F_LinU = ALE_Transformations::get_F_LinU<
                                                  dealdim>(phi_grads_u[j]);

                const Tensor<2, dealdim> E_LinU = 0.5
                                                  * (transpose(F_LinU) * F + transpose(F) * F_LinU);

                const double tr_E_LinU = Structure_Terms_in_ALE::get_tr_E_LinU<
                                         dealdim>(q_point, dz_state_grads_, phi_grads_u[j]);

                Tensor<2, dealdim> piola_kirchhoff_stress_structure_STVK_LinALL;
                piola_kirchhoff_stress_structure_STVK_LinALL =
                  lame_coefficient_lambda_
                  * (F_LinU * tr_E * Identity + F * tr_E_LinU * Identity)
                  + 2 * lame_coefficient_mu_ * (F_LinU * E + F * E_LinU);

                local_vector(j) += scale
                                   * (scalar_product(
                                        piola_kirchhoff_stress_structure_STVK_LinALL, dzv_grads)
                                      - density_structure_ * phi_v[j] * dzu + phi_p[j] * dzp)
                                   * state_fe_values.JxW(q_point);
              }
          }
      } // material_id == 1

  }

  void
  ElementEquation_UU(const EDC<DH, VECTOR, dealdim> &edc,
                     dealii::Vector<double> &local_vector, double scale,
                     double /*scale_ico*/)
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();
    unsigned int material_id = edc.GetMaterialId();

    assert(this->problem_type_ == "adjoint_hessian");

    zvalues_.resize(n_q_points, Vector<double>(5));
    zgrads_.resize(n_q_points, vector<Tensor<1, 2> >(5));

    edc.GetValuesState("adjoint", zvalues_);
    edc.GetGradsState("adjoint", zgrads_);

    du_tangent_values_.resize(n_q_points, Vector<double>(5));
    du_tangent_grads_.resize(n_q_points, vector<Tensor<1, 2> >(5));

    edc.GetValuesState("tangent", du_tangent_values_);
    edc.GetGradsState("tangent", du_tangent_grads_);

    du_state_values_.resize(n_q_points, Vector<double>(5));
    du_state_grads_.resize(n_q_points, vector<Tensor<1, 2> >(5));

    edc.GetValuesState("state", du_state_values_);
    edc.GetGradsState("state", du_state_grads_);

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Vector displacements(2);
    const FEValuesExtractors::Scalar pressure(4);

    std::vector<Tensor<1, 2> > phi_v(n_dofs_per_element);
    std::vector<Tensor<2, 2> > phi_grads_v(n_dofs_per_element);
    std::vector<Tensor<1, 2> > phi_u(n_dofs_per_element);
    std::vector<Tensor<2, 2> > phi_grads_u(n_dofs_per_element);
    std::vector<double> phi_p(n_dofs_per_element);

    const Tensor<2, dealdim> Identity = ALE_Transformations::get_Identity<
                                        dealdim>();

    if (material_id == 0)
      {

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            Tensor<2, 2> zv_grads;
            zv_grads.clear();
            zv_grads[0][0] = zgrads_[q_point][0][0];
            zv_grads[0][1] = zgrads_[q_point][0][1];
            zv_grads[1][0] = zgrads_[q_point][1][0];
            zv_grads[1][1] = zgrads_[q_point][1][1];

            Tensor<1, 2> zv;
            zv.clear();
            zv[0] = zvalues_[q_point](0);
            zv[1] = zvalues_[q_point](1);

            // state values which contains
            // solution from previous Newton step
            // Necessary for fluid convection term
            Tensor<2, 2> duv_tangent_grads;
            duv_tangent_grads.clear();
            duv_tangent_grads[0][0] = du_tangent_grads_[q_point][0][0];
            duv_tangent_grads[0][1] = du_tangent_grads_[q_point][0][1];
            duv_tangent_grads[1][0] = du_tangent_grads_[q_point][1][0];
            duv_tangent_grads[1][1] = du_tangent_grads_[q_point][1][1];

            Tensor<1, 2> duv_tangent;
            duv_tangent.clear();
            duv_tangent[0] = du_tangent_values_[q_point](0);
            duv_tangent[1] = du_tangent_values_[q_point](1);

            for (unsigned int i = 0; i < n_dofs_per_element; i++)
              {
                const Tensor<1, 2> phi_i_v = state_fe_values[velocities].value(i,
                                             q_point);
                const Tensor<2, 2> phi_i_grads_v =
                  state_fe_values[velocities].gradient(i, q_point);

                local_vector(i) += scale
                                   * (density_fluid_
                                      * (phi_i_grads_v * duv_tangent
                                         + duv_tangent_grads * phi_i_v) * zv)
                                   * state_fe_values.JxW(q_point);
              }
          }
      } // end material_id == 0
    if (material_id == 1)
      {

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            for (unsigned int k = 0; k < n_dofs_per_element; k++)
              {
                phi_p[k] = state_fe_values[pressure].value(k, q_point);
                phi_v[k] = state_fe_values[velocities].value(k, q_point);
                phi_grads_v[k] = state_fe_values[velocities].gradient(k, q_point);
                phi_u[k] = state_fe_values[displacements].value(k, q_point);
                phi_grads_u[k] = state_fe_values[displacements].gradient(k,
                                                                         q_point);
              }

            Tensor<2, 2> zv_grads;
            zv_grads.clear();
            zv_grads[0][0] = zgrads_[q_point][0][0];
            zv_grads[0][1] = zgrads_[q_point][0][1];
            zv_grads[1][0] = zgrads_[q_point][1][0];
            zv_grads[1][1] = zgrads_[q_point][1][1];

            // tangent values and grads
            const Tensor<2, dealdim> F = ALE_Transformations::get_F<dealdim>(
                                           q_point, du_tangent_grads_);

            const Tensor<2, dealdim> F_T =
              ALE_Transformations::get_F_T<dealdim>(F);

            const Tensor<2, dealdim> E = Structure_Terms_in_ALE::get_E<dealdim>(
                                           F_T, F, Identity);

            const double tr_E = Structure_Terms_in_ALE::get_tr_E<dealdim>(E);

            const Tensor<2, dealdim> F_LinU_state =
              ALE_Transformations::get_F_LinU_state<dealdim>(q_point,
                                                             du_state_grads_);

            for (unsigned int j = 0; j < n_dofs_per_element; j++)
              {
                const Tensor<2, dealdim> F_LinU = ALE_Transformations::get_F_LinU<
                                                  dealdim>(phi_grads_u[j]);

                const Tensor<2, dealdim> E_LinU = 0.5
                                                  * (transpose(F_LinU) * F + transpose(F) * F_LinU);

                const Tensor<2, dealdim> E_LinU_state = 0.5
                                                        * (transpose(F_LinU_state) * F + transpose(F) * F_LinU_state);

                const double tr_E_LinU = Structure_Terms_in_ALE::get_tr_E_LinU<
                                         dealdim>(q_point, du_tangent_grads_, phi_grads_u[j]);

                // 2nd derivatives for tr_E and E
                const double tr_E_LinW_LinU =
                  Structure_Terms_in_ALE::get_tr_E_LinU<dealdim>(q_point,
                                                                 du_state_grads_, phi_grads_u[j]);

                const Tensor<2, dealdim> E_LinW_LinU = 0.5
                                                       * (transpose(F_LinU) * F_LinU_state
                                                          + transpose(F_LinU_state) * F_LinU);

                // STVK: 2nd derivative
                Tensor<2, dealdim> piola_kirchhoff_stress_structure_STVK_LinALL_LinALL;
                piola_kirchhoff_stress_structure_STVK_LinALL_LinALL =
                  lame_coefficient_lambda_
                  * (tr_E_LinW_LinU * F + 2 * tr_E_LinU * F_LinU)
                  + 2 * lame_coefficient_mu_
                  * (F_LinU * E_LinU_state + F_LinU_state * E_LinU
                     + F * E_LinW_LinU);

                Tensor<2, dealdim> piola_kirchhoff_stress_structure_STVK_LinALL;
                piola_kirchhoff_stress_structure_STVK_LinALL =
                  lame_coefficient_lambda_
                  * (F_LinU * tr_E * Identity + F * tr_E_LinU * Identity)
                  + 2 * lame_coefficient_mu_ * (F_LinU * E + F * E_LinU);

              }
          }

      } // end material_id == 1

  }

  // Look for BoundaryEquationQ
  void
  ElementEquation_Q(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
                    double /*scale_ico*/)
  {
    assert(this->problem_type_ == "gradient");
  }

  void
  ElementEquation_QT(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                     dealii::Vector<double> &/*local_vector*/, double /*scale*/,
                     double /*scale_ico*/)
  {
    assert(this->problem_type_ == "tangent");
  }

  void
  ElementEquation_QTT(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                      dealii::Vector<double> &/*local_vector*/, double /*scale*/,
                      double /*scale_ico*/)
  {
    assert(this->problem_type_ == "hessian");
  }

  void
  ElementEquation_QU(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                     dealii::Vector<double> &/*local_vector*/, double /*scale*/,
                     double /*scale_ico*/)
  {
    assert(this->problem_type_ == "adjoint_hessian");
  }
  void
  ElementEquation_UQ(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                     dealii::Vector<double> &/*local_vector*/, double /*scale*/,
                     double /*scale_ico*/)
  {
    assert(this->problem_type_ == "hessian");
  }
  void
  ElementEquation_QQ(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                     dealii::Vector<double> &/*local_vector*/, double /*scale*/,
                     double /*scale_ico*/)
  {
    assert(this->problem_type_ == "hessian");
  }

  void
  ElementRightHandSide(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                       dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
    assert(this->problem_type_ == "state");
  }

  // Values for Boundary integrals
  void
  BoundaryEquation(const FaceDataContainer<DH, VECTOR, dealdim> &fdc,
                   dealii::Vector<double> &local_vector, double scale,
                   double /*scale_ico*/)
  {

    const auto &state_fe_face_values = fdc.GetFEFaceValuesState();
    unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();
    unsigned int n_q_points = fdc.GetNQPoints();
    unsigned int color = fdc.GetBoundaryIndicator();

    assert(this->problem_type_ == "state");

    // do-nothing condition applied at outflow boundary due symmetric part of
    // fluid's stress tensor
    if (color == 1)
      {
        uboundarygrads_.resize(n_q_points, vector<Tensor<1, 2> >(5));

        fdc.GetFaceGradsState("last_newton_solution", uboundarygrads_);

        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(4);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            Tensor<2, 2> v_grad;
            v_grad.clear();
            v_grad[0][0] = uboundarygrads_[q_point][0][0];
            v_grad[0][1] = uboundarygrads_[q_point][0][1];
            v_grad[1][0] = uboundarygrads_[q_point][1][0];
            v_grad[1][1] = uboundarygrads_[q_point][1][1];

            const Tensor<2, 2> do_nothing = density_fluid_ * viscosity_
                                            * transpose(v_grad);

            const Tensor<1, 2> neumann_value = do_nothing
                                               * state_fe_face_values.normal_vector(q_point);

            for (unsigned int i = 0; i < n_dofs_per_element; i++)
              {
                const Tensor<1, 2> phi_i_v =
                  state_fe_face_values[velocities].value(i, q_point);

                local_vector(i) -= 1.0 * scale * neumann_value * phi_i_v
                                   * state_fe_face_values.JxW(q_point);
              }
          }
      }

    // Get Param Values for the Control
    // They are initialized in main.cc
    qvalues_.reinit(2);
    fdc.GetParamValues("control", qvalues_);

    // control value for the upper part: \Gamma_q0
    if (color == 50)
      {
        const FEValuesExtractors::Vector velocities(0);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            for (unsigned int i = 0; i < n_dofs_per_element; i++)
              {
                const Tensor<1, 2> phi_i_v =
                  state_fe_face_values[velocities].value(i, q_point);

                local_vector(i) -= scale * qvalues_(0)
                                   * state_fe_face_values.normal_vector(q_point) * phi_i_v
                                   * state_fe_face_values.JxW(q_point);
              }

          }
      }

    // control value for the lower part: \Gamma_q1
    if (color == 51)
      {
        const FEValuesExtractors::Vector velocities(0);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            for (unsigned int i = 0; i < n_dofs_per_element; i++)
              {
                const Tensor<1, 2> phi_i_v =
                  state_fe_face_values[velocities].value(i, q_point);

                local_vector(i) -= scale * qvalues_(1)
                                   * state_fe_face_values.normal_vector(q_point) * phi_i_v
                                   * state_fe_face_values.JxW(q_point);
              }

          }
      }
  }

  void
  BoundaryMatrix(const FaceDataContainer<DH, VECTOR, dealdim> &fdc,
                 dealii::FullMatrix<double> &local_matrix, double /*scale*/,
                 double /*scale_ico*/)
  {
    const auto &state_fe_face_values = fdc.GetFEFaceValuesState();
    unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();
    unsigned int n_q_points = fdc.GetNQPoints();
    unsigned int color = fdc.GetBoundaryIndicator();

    // do-nothing applied on outflow boundary
    if (color == 1)
      {
        uboundarygrads_.resize(n_q_points, vector<Tensor<1, 2> >(5));

        if (this->problem_type_ == "state")
          fdc.GetFaceGradsState("last_newton_solution", uboundarygrads_);
        else
          fdc.GetFaceGradsState("state", uboundarygrads_);

        const FEValuesExtractors::Vector velocities(0);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            Tensor<2, 2> v_grad;
            v_grad[0][0] = uboundarygrads_[q_point][0][0];
            v_grad[0][1] = uboundarygrads_[q_point][0][1];
            v_grad[1][0] = uboundarygrads_[q_point][1][0];
            v_grad[1][1] = uboundarygrads_[q_point][1][1];

            for (unsigned int i = 0; i < n_dofs_per_element; i++)
              {
                const Tensor<1, 2> phi_i_v =
                  state_fe_face_values[velocities].value(i, q_point);

                for (unsigned int j = 0; j < n_dofs_per_element; j++)
                  {
                    const Tensor<2, 2> phi_j_grads_v =
                      state_fe_face_values[velocities].gradient(j, q_point);

                    Tensor<2, 2> do_nothing_LinAll;
                    do_nothing_LinAll = density_fluid_ * viscosity_
                                        * transpose(phi_j_grads_v);

                    const Tensor<1, 2> neumann_value = do_nothing_LinAll
                                                       * state_fe_face_values.normal_vector(q_point);

                    local_matrix(i, j) -= 1.0 * neumann_value * phi_i_v
                                          * state_fe_face_values.JxW(q_point);
                  }
              }
          }
      }

  }

  void
  BoundaryRightHandSide(const FaceDataContainer<DH, VECTOR, dealdim> & /*fdc*/,
                        dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
    assert(this->problem_type_ == "state");
  }

  void
  BoundaryEquation_Q(const FaceDataContainer<DH, VECTOR, dealdim> &fdc,
                     dealii::Vector<double> &local_vector, double scale,
                     double /*scale_ico*/)
  {
    const auto &state_fe_face_values = fdc.GetFEFaceValuesState();
    unsigned int n_q_points = fdc.GetNQPoints();
    unsigned int color = fdc.GetBoundaryIndicator();
    assert(this->problem_type_ == "gradient");

    zboundaryvalues_.resize(n_q_points, Vector<double>(5));

    fdc.GetFaceValuesState("adjoint", zboundaryvalues_);

    // control values for the upper and lower part
    if (color == 50)
      {
        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            Tensor<1, 2> zvboundary;
            zvboundary.clear();
            zvboundary[0] = zboundaryvalues_[q_point](0);
            zvboundary[1] = zboundaryvalues_[q_point](1);

            local_vector(0) -= scale * 1.0
                               * state_fe_face_values.normal_vector(q_point) * zvboundary
                               * state_fe_face_values.JxW(q_point);
          }
      }
    if (color == 51)
      {
        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            Tensor<1, 2> zvboundary;
            zvboundary.clear();
            zvboundary[0] = zboundaryvalues_[q_point](0);
            zvboundary[1] = zboundaryvalues_[q_point](1);

            local_vector(1) -= scale * 1.0
                               * state_fe_face_values.normal_vector(q_point) * zvboundary
                               * state_fe_face_values.JxW(q_point);

          }
      }
  }

  void
  BoundaryEquation_QT(const FaceDataContainer<DH, VECTOR, dealdim> &fdc,
                      dealii::Vector<double> &local_vector, double scale,
                      double /*scale_ico*/)
  {
    const auto &state_fe_face_values = fdc.GetFEFaceValuesState();
    unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();
    unsigned int n_q_points = fdc.GetNQPoints();
    unsigned int color = fdc.GetBoundaryIndicator();

    assert(this->problem_type_ == "tangent");

    dqvalues_.reinit(2);
    fdc.GetParamValues("dq", dqvalues_);

    if (color == 50)
      {
        const FEValuesExtractors::Vector velocities(0);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            for (unsigned int i = 0; i < n_dofs_per_element; i++)
              {
                const Tensor<1, 2> phi_i_v =
                  state_fe_face_values[velocities].value(i, q_point);

                local_vector(i) -= 1.0 * scale * dqvalues_(0) * phi_i_v
                                   * state_fe_face_values.normal_vector(q_point)
                                   * state_fe_face_values.JxW(q_point);
              }

          }
      }

    if (color == 51)
      {
        const FEValuesExtractors::Vector velocities(0);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            for (unsigned int i = 0; i < n_dofs_per_element; i++)
              {
                const Tensor<1, 2> phi_i_v =
                  state_fe_face_values[velocities].value(i, q_point);

                local_vector(i) -= 1.0 * scale * dqvalues_(1) * phi_i_v
                                   * state_fe_face_values.normal_vector(q_point)
                                   * state_fe_face_values.JxW(q_point);
              }

          }
      }
  }

  void
  BoundaryEquation_QTT(const FaceDataContainer<DH, VECTOR, dealdim> &fdc,
                       dealii::Vector<double> &local_vector, double scale,
                       double /*scale_ico*/)
  {
    const auto &state_fe_face_values = fdc.GetFEFaceValuesState();
    unsigned int n_q_points = fdc.GetNQPoints();
    unsigned int color = fdc.GetBoundaryIndicator();

    assert(this->problem_type_ == "hessian");

    dzboundaryvalues_.resize(n_q_points, Vector<double>(5));

    fdc.GetFaceValuesState("adjoint_hessian", dzboundaryvalues_);

    // control values for both parts
    if (color == 50)
      {
        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            Tensor<1, 2> dzvboundary;
            dzvboundary.clear();
            dzvboundary[0] = dzboundaryvalues_[q_point](0);
            dzvboundary[1] = dzboundaryvalues_[q_point](1);

            local_vector(0) -= scale * 1.0
                               * state_fe_face_values.normal_vector(q_point) * dzvboundary
                               * state_fe_face_values.JxW(q_point);

          }
      }
    if (color == 51)
      {
        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            Tensor<1, 2> dzvboundary;
            dzvboundary.clear();
            dzvboundary[0] = dzboundaryvalues_[q_point](0);
            dzvboundary[1] = dzboundaryvalues_[q_point](1);

            local_vector(1) -= scale * 1.0
                               * state_fe_face_values.normal_vector(q_point) * dzvboundary
                               * state_fe_face_values.JxW(q_point);
          }
      }
  }

  // do-nothing condition at boundary /Gamma_1
  void
  BoundaryEquation_U(const FaceDataContainer<DH, VECTOR, dealdim> &fdc,
                     dealii::Vector<double> &local_vector, double scale,
                     double /*scale_ico*/)
  {
    const auto &state_fe_face_values = fdc.GetFEFaceValuesState();
    unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();
    unsigned int n_q_points = fdc.GetNQPoints();
    unsigned int color = fdc.GetBoundaryIndicator();

    assert(this->problem_type_ == "adjoint");

    // do-nothing applied on outflow boundary due symmetric part of
    // fluid's stress tensor
    if (color == 1)
      {
        zboundaryvalues_.resize(n_q_points, Vector<double>(5));

        fdc.GetFaceValuesState("last_newton_solution", zboundaryvalues_);

        const FEValuesExtractors::Vector velocities(0);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            Tensor<1, 2> zvboundary;
            zvboundary.clear();
            zvboundary[0] = zboundaryvalues_[q_point](0);
            zvboundary[1] = zboundaryvalues_[q_point](1);

            for (unsigned int i = 0; i < n_dofs_per_element; i++)
              {
                const Tensor<2, 2> phi_i_grads_v =
                  state_fe_face_values[velocities].gradient(i, q_point);

                local_vector(i) -= 1.0 * scale * density_fluid_ * viscosity_
                                   * transpose(phi_i_grads_v)
                                   * state_fe_face_values.normal_vector(q_point) * zvboundary
                                   * state_fe_face_values.JxW(q_point);
              }
          }
      }
  }

  void
  BoundaryEquation_UT(const FaceDataContainer<DH, VECTOR, dealdim> &fdc,
                      dealii::Vector<double> &local_vector, double scale,
                      double /*scale_ico*/)
  {
    const auto &state_fe_face_values = fdc.GetFEFaceValuesState();
    unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();
    unsigned int n_q_points = fdc.GetNQPoints();
    unsigned int color = fdc.GetBoundaryIndicator();

    assert(this->problem_type_ == "tangent");

    // do-nothing applied on outflow boundary due symmetric part of
    // fluid's stress tensor
    if (color == 1)
      {
        duboundarygrads_.resize(n_q_points, vector<Tensor<1, 2> >(5));

        fdc.GetFaceGradsState("last_newton_solution", duboundarygrads_);

        const FEValuesExtractors::Vector velocities(0);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            Tensor<2, 2> duv_grad;
            duv_grad[0][0] = duboundarygrads_[q_point][0][0];
            duv_grad[0][1] = duboundarygrads_[q_point][0][1];
            duv_grad[1][0] = duboundarygrads_[q_point][1][0];
            duv_grad[1][1] = duboundarygrads_[q_point][1][1];
            for (unsigned int i = 0; i < n_dofs_per_element; i++)
              {
                const Tensor<1, 2> phi_i_v =
                  state_fe_face_values[velocities].value(i, q_point);

                local_vector(i) -= 1.0 * scale * density_fluid_ * viscosity_
                                   * transpose(duv_grad)
                                   * state_fe_face_values.normal_vector(q_point) * phi_i_v
                                   * state_fe_face_values.JxW(q_point);
              }
          }
      }
  }

  void
  BoundaryEquation_UTT(const FaceDataContainer<DH, VECTOR, dealdim> &fdc,
                       dealii::Vector<double> &local_vector, double scale,
                       double /*scale_ico*/)
  {
    const auto &state_fe_face_values = fdc.GetFEFaceValuesState();
    unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();
    unsigned int n_q_points = fdc.GetNQPoints();
    unsigned int color = fdc.GetBoundaryIndicator();

    assert(this->problem_type_ == "adjoint_hessian");

    // do-nothing applied on outflow boundary due symmetric part of
    // fluid's stress tensor
    if (color == 1)
      {
        dzboundaryvalues_.resize(n_q_points, Vector<double>(5));

        fdc.GetFaceValuesState("last_newton_solution", dzboundaryvalues_);

        const FEValuesExtractors::Vector velocities(0);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            Tensor<1, 2> dzvboundary;
            dzvboundary.clear();
            dzvboundary[0] = dzboundaryvalues_[q_point](0);
            dzvboundary[1] = dzboundaryvalues_[q_point](1);

            for (unsigned int i = 0; i < n_dofs_per_element; i++)
              {
                const Tensor<2, 2> phi_i_grads_v =
                  state_fe_face_values[velocities].gradient(i, q_point);

                local_vector(i) -= 1.0 * scale * density_fluid_ * viscosity_
                                   * transpose(phi_i_grads_v)
                                   * state_fe_face_values.normal_vector(q_point) * dzvboundary
                                   * state_fe_face_values.JxW(q_point);
              }
          }
      }
  }

  void
  BoundaryEquation_UU(const FaceDataContainer<DH, VECTOR, dealdim> & /*fdc*/,
                      dealii::Vector<double> &/*local_vector*/, double /*scale*/,
                      double /*scale_ico*/)
  {
    assert(this->problem_type_ == "adjoint_hessian");
  }

  void
  BoundaryEquation_QU(const FaceDataContainer<DH, VECTOR, dealdim> & /*fdc*/,
                      dealii::Vector<double> &/*local_vector*/, double /*scale*/,
                      double /*scale_ico*/)
  {
    assert(this->problem_type_ == "adjoint_hessian");
  }

  void
  BoundaryEquation_UQ(const FaceDataContainer<DH, VECTOR, dealdim> & /*fdc*/,
                      dealii::Vector<double> &/*local_vector*/, double /*scale*/,
                      double /*scale_ico*/)
  {

  }

  void
  BoundaryEquation_QQ(const FaceDataContainer<DH, VECTOR, dealdim> & /*fdc*/,
                      dealii::Vector<double> &/*local_vector*/, double /*scale*/,
                      double /*scale_ico*/)
  {

  }

///// Hier FaceEquation einfuegen

  void
  FaceEquation(const FaceDataContainer<DH, VECTOR, dealdim> &,
               dealii::Vector<double> &, double, double)
  {

  }
  void
  FaceMatrix(const FaceDataContainer<DH, VECTOR, dealdim> &,
             dealii::FullMatrix<double> &, double, double)
  {
  }
  void
  FaceRightHandSide(const FaceDataContainer<DH, VECTOR, dealdim> &,
                    dealii::Vector<double> &, double)
  {
    assert(this->problem_type_ == "state");
  }

  void
  FaceEquation_Q(const FaceDataContainer<DH, VECTOR, dealdim> &,
                 dealii::Vector<double> &, double, double)
  {
  }

  void
  FaceEquation_QT(const FaceDataContainer<DH, VECTOR, dealdim> &,
                  dealii::Vector<double> &, double, double)
  {
  }

  void
  FaceEquation_QTT(const FaceDataContainer<DH, VECTOR, dealdim> &,
                   dealii::Vector<double> &, double, double)
  {
  }

  void
  FaceEquation_U(const FaceDataContainer<DH, VECTOR, dealdim> &,
                 dealii::Vector<double> &, double, double)
  {
  }

  void
  FaceEquation_UT(const FaceDataContainer<DH, VECTOR, dealdim> &,
                  dealii::Vector<double> &, double, double)
  {
  }

  void
  FaceEquation_UTT(const FaceDataContainer<DH, VECTOR, dealdim> &,
                   dealii::Vector<double> &, double, double)
  {
  }

  void
  FaceEquation_UU(const FaceDataContainer<DH, VECTOR, dealdim> & /*fdc*/,
                  dealii::Vector<double> &/*local_vector*/, double /*scale*/,
                  double /*scale_ico*/)
  {
    assert(this->problem_type_ == "adjoint_hessian");
  }

  void
  FaceEquation_QU(const FaceDataContainer<DH, VECTOR, dealdim> & /*fdc*/,
                  dealii::Vector<double> &/*local_vector*/, double /*scale*/,
                  double /*scale_ico*/)
  {
    assert(this->problem_type_ == "adjoint_hessian");
  }

  void
  FaceEquation_UQ(const FaceDataContainer<DH, VECTOR, dealdim> & /*fdc*/,
                  dealii::Vector<double> &/*local_vector*/, double /*scale*/,
                  double /*scale_ico*/)
  {

  }

  void
  FaceEquation_QQ(const FaceDataContainer<DH, VECTOR, dealdim> & /*fdc*/,
                  dealii::Vector<double> &/*local_vector*/, double /*scale*/,
                  double /*scale_ico*/)
  {

  }

///////// Hier Face zuende

  void
  ControlElementEquation(const EDC<DH, VECTOR, dealdim> &edc,
                         dealii::Vector<double> &local_vector, double scale)
  {
    {
      assert(
        (this->problem_type_ == "gradient")||(this->problem_type_ == "hessian"));
      funcgradvalues_.reinit(local_vector.size());
      edc.GetParamValues("last_newton_solution", funcgradvalues_);
    }

    for (unsigned int i = 0; i < local_vector.size(); i++)
      {
        local_vector(i) += scale * funcgradvalues_(i);
      }
  }

  void
  ControlElementMatrix(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                       FullMatrix<double> &local_matrix, double scale)
  {
    assert(local_matrix.m() == local_matrix.n());
    for (unsigned int i = 0; i < local_matrix.m(); i++)
      {
        local_matrix(i, i) += scale * 1.;
      }
  }

  UpdateFlags
  GetUpdateFlags() const
  {
    if ((this->problem_type_ == "adjoint")
        || (this->problem_type_ == "state")
        || (this->problem_type_ == "tangent")
        || (this->problem_type_ == "adjoint_hessian")
        || (this->problem_type_ == "hessian"))
      return update_values | update_gradients | update_quadrature_points;
    else if ((this->problem_type_ == "gradient"))
      return update_values | update_quadrature_points;
    else
      throw DOpEException("Unknown Problem Type " + this->problem_type_,
                          "LocalPDE::GetUpdateFlags");
  }

  UpdateFlags
  GetFaceUpdateFlags() const
  {
    if ((this->problem_type_ == "adjoint")
        || (this->problem_type_ == "state")
        || (this->problem_type_ == "tangent")
        || (this->problem_type_ == "adjoint_hessian")
        || (this->problem_type_ == "hessian"))
      return update_values | update_gradients | update_normal_vectors
             | update_quadrature_points;
    else if ((this->problem_type_ == "gradient"))
      return update_values | update_quadrature_points
             | update_normal_vectors;
    else
      throw DOpEException("Unknown Problem Type " + this->problem_type_,
                          "LocalPDE::GetUpdateFlags");
  }

  unsigned int
  GetControlNBlocks() const
  {
    return 2;
  }

  unsigned int
  GetStateNBlocks() const
  {
    return 3;
  }

  std::vector<unsigned int> &
  GetControlBlockComponent()
  {
    return control_block_components_;
  }
  const std::vector<unsigned int> &
  GetControlBlockComponent() const
  {
    return control_block_components_;
  }
  std::vector<unsigned int> &
  GetStateBlockComponent()
  {
    return state_block_components_;
  }
  const std::vector<unsigned int> &
  GetStateBlockComponent() const
  {
    return state_block_components_;
  }

private:
  Vector<double> qvalues_;
  Vector<double> dqvalues_;

  Vector<double> funcgradvalues_;
  vector<Vector<double> > fvalues_;

  vector<Vector<double> > uvalues_;
  vector<vector<Tensor<1, dealdim> > > ugrads_;

  vector<Vector<double> > zvalues_;
  vector<vector<Tensor<1, dealdim> > > zgrads_;
  vector<Vector<double> > z_state_values_;
  vector<vector<Tensor<1, dealdim> > > z_state_grads_;

  vector<Vector<double> > duvalues_;
  vector<vector<Tensor<1, dealdim> > > dugrads_;
  vector<Vector<double> > du_state_values_;
  vector<vector<Tensor<1, dealdim> > > du_state_grads_;

  // for ElementEquation_UU
  vector<Vector<double> > du_tangent_values_;
  vector<vector<Tensor<1, dealdim> > > du_tangent_grads_;

  vector<Vector<double> > dzvalues_;
  vector<vector<Tensor<1, dealdim> > > dzgrads_;
  vector<Vector<double> > dz_state_values_;
  vector<vector<Tensor<1, dealdim> > > dz_state_grads_;

  // boundary values
  vector<Vector<double> > qboundaryvalues_;
  vector<Vector<double> > fboundaryvalues_;
  vector<Vector<double> > uboundaryvalues_;

  vector<Vector<double> > zboundaryvalues_;
  vector<Vector<double> > dzboundaryvalues_;

  vector<vector<Tensor<1, dealdim> > > uboundarygrads_;
  vector<vector<Tensor<1, dealdim> > > duboundarygrads_;

  vector<unsigned int> control_block_components_;
  vector<unsigned int> state_block_components_;

  double diameter_;

  // Fluid- and material variables
  double density_fluid_, density_structure_, viscosity_, alpha_u_,
         lame_coefficient_mu_, poisson_ratio_nu_, lame_coefficient_lambda_;

};
#endif

