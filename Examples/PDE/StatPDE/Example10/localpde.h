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

using namespace std;
using namespace dealii;
using namespace DOpE;

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

  static void
  declare_params(ParameterReader &param_reader)
  {
    param_reader.SetSubsection("Local PDE parameters");
    param_reader.declare_entry("density_fluid", "0.0", Patterns::Double(0));
    param_reader.declare_entry("viscosity", "0.0", Patterns::Double(0));
    param_reader.declare_entry("alpha_u", "0.0", Patterns::Double(0));
    param_reader.declare_entry("alpha_p", "0.0", Patterns::Double(0));
    param_reader.declare_entry("mu", "0.0", Patterns::Double(0));
    param_reader.declare_entry("poisson_ratio_nu", "0.0",
                               Patterns::Double(0));
  }

  LocalPDE(ParameterReader &param_reader) :
    state_block_component_(5, 0)
  {
    assert(dealdim==2);

    state_block_component_[2] = 1;
    state_block_component_[3] = 2;
    state_block_component_[4] = 2;

    param_reader.SetSubsection("Local PDE parameters");
    density_fluid = param_reader.get_double("density_fluid");
    viscosity = param_reader.get_double("viscosity");
    alpha_u = param_reader.get_double("alpha_u");
    alpha_p = param_reader.get_double("alpha_p");
    mu = param_reader.get_double("mu");
    poisson_ratio_nu = param_reader.get_double("poisson_ratio_nu");
  }

  // Domain values for elements
  void
  ElementEquation(
    const EDC<DH, VECTOR, 2> &edc,
    dealii::Vector<double> &local_vector, double scale,
    double /*scale_ico*/) override
  {
    assert(this->problem_type_ == "state");

    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_q_points = edc.GetNQPoints();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int material_id = edc.GetMaterialId();
    double element_diameter = edc.GetElementDiameter();

    lambda = (2 * poisson_ratio_nu * mu) / (1.0 - 2 * poisson_ratio_nu);

    uvalues_.resize(n_q_points, Vector<double>(5));
    ugrads_.resize(n_q_points, vector<Tensor<1, 2> >(5));

    edc.GetValuesState("last_newton_solution", uvalues_);
    edc.GetGradsState("last_newton_solution", ugrads_);

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(2);
    const FEValuesExtractors::Vector displacements(3);

    // Declare test functions
    Tensor<1, 2> phi_i_v;
    Tensor<2, 2> phi_i_grads_v;
    double phi_i_p;
    Tensor<1, 2> phi_i_grads_p; // only for STVK
    Tensor<1, 2> phi_i_u;
    Tensor<2, 2> phi_i_grads_u;

    Tensor<2, 2> I;
    I[0][0] = 1.0;
    I[0][1] = 0.0;
    I[1][0] = 0.0;
    I[1][1] = 1.0;

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        // ALE Transformations
        Tensor<2, 2> F;
        F[0][0] = 1.0 + ugrads_[q_point][3][0];
        F[0][1] = ugrads_[q_point][3][1];
        F[1][0] = ugrads_[q_point][4][0];
        F[1][1] = 1.0 + ugrads_[q_point][4][1];

        Tensor<2, 2> F_Inverse;
        F_Inverse = invert(F);

        Tensor<2, 2> F_T;
        F_T = transpose(F);

        Tensor<2, 2> F_Inverse_T;
        F_Inverse_T = transpose(F_Inverse);

        // STVK: Green-Lagrange strain tensor
        Tensor<2, 2> E;
        E = 0.5 * (transpose(F) * F - I);

        double tr_E;
        tr_E = trace(E);

        double J;
        J = determinant(F);

        Tensor<2, 2> pI;
        pI[0][0] = uvalues_[q_point](2);
        pI[1][1] = uvalues_[q_point](2);

        Tensor<2, 2> v_grad;
        v_grad[0][0] = ugrads_[q_point][0][0];
        v_grad[0][1] = ugrads_[q_point][0][1];
        v_grad[1][0] = ugrads_[q_point][1][0];
        v_grad[1][1] = ugrads_[q_point][1][1];

        Tensor<1, 2> v;
        v.clear();
        v[0] = uvalues_[q_point](0);
        v[1] = uvalues_[q_point](1);

        // STVK
        Tensor<1, 2> pressure_grad;
        pressure_grad[0] = ugrads_[q_point][2][0];
        pressure_grad[1] = ugrads_[q_point][2][1];

        Tensor<2, 2> u_grad;
        u_grad[0][0] = ugrads_[q_point][3][0];
        u_grad[0][1] = ugrads_[q_point][3][1];
        u_grad[1][0] = ugrads_[q_point][4][0];
        u_grad[1][1] = ugrads_[q_point][4][1];

        //div(J * F^{-1} * v)
        double incompressiblity_fluid;
        incompressiblity_fluid = (ugrads_[q_point][0][0]
                                  + ugrads_[q_point][4][1] * ugrads_[q_point][0][0]
                                  - ugrads_[q_point][3][1] * ugrads_[q_point][1][0]
                                  - ugrads_[q_point][4][0] * ugrads_[q_point][0][1]
                                  + ugrads_[q_point][1][1]
                                  + ugrads_[q_point][3][0] * ugrads_[q_point][1][1]);

        // constitutive stress tensors for fluid
        Tensor<2, 2> cauchy_stress_fluid;
        cauchy_stress_fluid = -pI
                              + density_fluid * viscosity
                              * (v_grad * F_Inverse + F_Inverse_T * transpose(v_grad));

        // constitutive stress tensors for structure INH
        Tensor<2, 2> cauchy_stress_structure_INH;
        cauchy_stress_structure_INH = -pI + mu * (F * transpose(F) - I);

        // constitutive stress tensors for structure STVK
        Tensor<2, 2> cauchy_stress_structure_STVK;
        cauchy_stress_structure_STVK = 1.0 / J * F
                                       * (lambda * tr_E * I + 2 * mu * E) * F_T;

        Tensor<1, 2> convection_fluid;
        convection_fluid = density_fluid * J * (v_grad * F_Inverse * v);

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            phi_i_v = state_fe_values[velocities].value(i, q_point);
            phi_i_grads_v = state_fe_values[velocities].gradient(i, q_point);
            phi_i_p = state_fe_values[pressure].value(i, q_point);
            phi_i_grads_p = state_fe_values[pressure].gradient(i, q_point); // only for STVK
            phi_i_u = state_fe_values[displacements].value(i, q_point);
            phi_i_grads_u = state_fe_values[displacements].gradient(i, q_point);

            // fluid: material_id == 0, structure: material_id == 1
            if (material_id == 0)
              {
                local_vector(i) += scale
                                   * (convection_fluid * phi_i_v
                                      + scalar_product(J * cauchy_stress_fluid * F_Inverse_T,
                                                       phi_i_grads_v) + incompressiblity_fluid * phi_i_p
                                      + alpha_u * element_diameter * element_diameter
                                      * scalar_product(u_grad, phi_i_grads_u))
                                   * state_fe_values.JxW(q_point);
              }
            else if (material_id == 1)
              {
                /*
                 local_vector(i) += scale * (scalar_product(J * cauchy_stress_structure_INH * F_Inverse_T, phi_i_grads_v)
                 + (J - 1.0) * phi_i_p
                 + v * phi_i_u)
                 * state_fe_values.JxW(q_point);
                 */
                local_vector(i) += scale
                                   * (scalar_product(
                                        J * cauchy_stress_structure_STVK * F_Inverse_T,
                                        phi_i_grads_v)
                                      + alpha_p * element_diameter * element_diameter * pressure_grad
                                      * phi_i_grads_p // harmonic pressure
                                      //+ uvalues_[q_point](2) * phi_i_p  // for STVK
                                      - v * phi_i_u) * state_fe_values.JxW(q_point);

              }
          }
      }
  }

  void
  ElementMatrix(
    const EDC<DH, VECTOR, 2> &edc,
    dealii::FullMatrix<double> &local_matrix, double scale,
    double /*scale_ico*/) override
  {
    assert(this->problem_type_ == "state");

    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_q_points = edc.GetNQPoints();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int material_id = edc.GetMaterialId();
    double element_diameter = edc.GetElementDiameter();

    uvalues_.resize(n_q_points, Vector<double>(5));
    ugrads_.resize(n_q_points, vector<Tensor<1, 2> >(5));

    edc.GetValuesState("last_newton_solution", uvalues_);
    edc.GetGradsState("last_newton_solution", ugrads_);

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(2);
    const FEValuesExtractors::Vector displacements(3);

    Tensor<2, 2> I;
    I[0][0] = 1.0;
    I[0][1] = 0.0;
    I[1][0] = 0.0;
    I[1][1] = 1.0;

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        // ALE Transformations
        Tensor<2, 2> F;
        F[0][0] = 1.0 + ugrads_[q_point][3][0];
        F[0][1] = ugrads_[q_point][3][1];
        F[1][0] = ugrads_[q_point][4][0];
        F[1][1] = 1.0 + ugrads_[q_point][4][1];

        Tensor<2, 2> F_Inverse;
        F_Inverse = invert(F);

        Tensor<2, 2> F_T;
        F_T = transpose(F);

        Tensor<2, 2> F_Inverse_T;
        F_Inverse_T = transpose(F_Inverse);

        double J;
        J = determinant(F);

        Tensor<2, 2> F_tilde;
        F_tilde[0][0] = 1.0 + ugrads_[q_point][4][1];
        F_tilde[0][1] = -ugrads_[q_point][3][1];
        F_tilde[1][0] = -ugrads_[q_point][4][0];
        F_tilde[1][1] = 1.0 + ugrads_[q_point][3][0];

        // STVK: Green-Lagrange strain tensor
        Tensor<2, 2> E;
        E = 0.5 * (F_T * F - I);

        double tr_E;
        tr_E = trace(E);

        Tensor<2, 2> pI;
        pI[0][0] = uvalues_[q_point](2);
        pI[1][1] = uvalues_[q_point](2);

        Tensor<2, 2> v_grad;
        v_grad[0][0] = ugrads_[q_point][0][0];
        v_grad[0][1] = ugrads_[q_point][0][1];
        v_grad[1][0] = ugrads_[q_point][1][0];
        v_grad[1][1] = ugrads_[q_point][1][1];

        Tensor<1, 2> v;
        v.clear();
        v[0] = uvalues_[q_point](0);
        v[1] = uvalues_[q_point](1);

        Tensor<2, 2> u_grad;
        u_grad[0][0] = ugrads_[q_point][3][0];
        u_grad[0][1] = ugrads_[q_point][3][1];
        u_grad[1][0] = ugrads_[q_point][4][0];
        u_grad[1][1] = ugrads_[q_point][4][1];

        for (unsigned int j = 0; j < n_dofs_per_element; j++)
          {
            const Tensor<1, 2> phi_j_v = state_fe_values[velocities].value(j,
                                         q_point);
            const Tensor<2, 2> phi_j_grads_v =
              state_fe_values[velocities].gradient(j, q_point);
            const double phi_j_p = state_fe_values[pressure].value(j, q_point);
            const Tensor<1, 2> phi_j_grads_p =
              state_fe_values[pressure].gradient(j, q_point); // only for STVK
            const Tensor<2, 2> phi_j_grads_u =
              state_fe_values[displacements].gradient(j, q_point);

            // ALE Transformations derivatives
            Tensor<2, 2> J_F_Inverse_T_LinU;
            J_F_Inverse_T_LinU[0][0] = phi_j_grads_u[1][1];
            J_F_Inverse_T_LinU[0][1] = -phi_j_grads_u[1][0];
            J_F_Inverse_T_LinU[1][0] = -phi_j_grads_u[0][1];
            J_F_Inverse_T_LinU[1][1] = phi_j_grads_u[0][0];

            Tensor<2, 2> F_tilde_LinU;
            F_tilde_LinU[0][0] = phi_j_grads_u[1][1];
            F_tilde_LinU[0][1] = -phi_j_grads_u[0][1];
            F_tilde_LinU[1][0] = -phi_j_grads_u[1][0];
            F_tilde_LinU[1][1] = phi_j_grads_u[0][0];

            double J_LinU;
            J_LinU = (phi_j_grads_u[0][0] * (1 + ugrads_[q_point][4][1])
                      + (1 + ugrads_[q_point][3][0]) * phi_j_grads_u[1][1]
                      - phi_j_grads_u[0][1] * ugrads_[q_point][4][0]
                      - ugrads_[q_point][3][1] * phi_j_grads_u[1][0]);

            Tensor<2, 2> F_Inverse_LinU;
            F_Inverse_LinU = (-1.0 / std::pow(J, 2) * J_LinU * F_tilde
                              + 1.0 / J * F_tilde_LinU);

            Tensor<2, 2> F_LinU;
            F_LinU[0][0] = phi_j_grads_u[0][0];
            F_LinU[0][1] = phi_j_grads_u[0][1];
            F_LinU[1][0] = phi_j_grads_u[1][0];
            F_LinU[1][1] = phi_j_grads_u[1][1];

            // STVK: Green_Lagrange strain tensor derivatives
            Tensor<2, 2> E_LinU;
            E_LinU = 0.5 * (transpose(F_LinU) * F + transpose(F) * F_LinU);

            double tr_E_LinU = ((1 + ugrads_[q_point][3][0])
                                * phi_j_grads_u[0][0]
                                + ugrads_[q_point][3][1] * phi_j_grads_u[0][1]
                                + (1 + ugrads_[q_point][4][1]) * phi_j_grads_u[1][1]
                                + ugrads_[q_point][4][0] * phi_j_grads_u[1][0]);

            // more derivatives
            Tensor<2, 2> pI_LinP;
            pI_LinP[0][0] = phi_j_p;
            pI_LinP[0][1] = 0.0;
            pI_LinP[1][0] = 0.0;
            pI_LinP[1][1] = phi_j_p;

            // INH
            // same derivative function for fluid and structure (1st term)
            // sigma * J * F^{-T} = -pI * J * F^{-T}  --> Linearization
            Tensor<2, 2> piola_kirchhoff_stress_1st_term_LinALL;
            piola_kirchhoff_stress_1st_term_LinALL = -pI_LinP * J * F_Inverse_T
                                                     - pI * J_F_Inverse_T_LinU;

            // piola-kirchhoff stress structure linearized in all directions (2nd term)
            // J * (mu * (FF^T - I)) * F^{-T} --> Linearization
            Tensor<2, 2> piola_kirchhoff_stress_structure_INH_2nd_term_LinALL;
            piola_kirchhoff_stress_structure_INH_2nd_term_LinALL = mu
                                                                   * (J_LinU * F + J * F_LinU - J_F_Inverse_T_LinU);

            // STVK
            // piola-kirchhoff stress structure STVK linearized in all directions
            // J * (1/J*F*(lambda*tr_E*I + 2*mu*E)*F^T) * F^{-T} --> Linearization
            Tensor<2, 2> piola_kirchhoff_stress_structure_STVK_LinALL;
            piola_kirchhoff_stress_structure_STVK_LinALL = lambda
                                                           * (F_LinU * tr_E * I + F * tr_E_LinU * I)
                                                           + 2 * mu * (F_LinU * E + F * E_LinU);

            // piola-kirchhoff stress fluid linearized in all directions (2nd term)
            // rho * nu * J * (v_grad * F^{-1} + F^{-T} * v_grad^T ) F^{-T}
            Tensor<2, 2> cauchy_stress_fluid_2nd_term_LinALL;
            cauchy_stress_fluid_2nd_term_LinALL = phi_j_grads_v * F_Inverse
                                                  + transpose(F_Inverse) * transpose(phi_j_grads_v)
                                                  + v_grad * F_Inverse_LinU
                                                  + transpose(F_Inverse_LinU) * transpose(v_grad);

            Tensor<2, 2> piola_kirchhoff_stress_fluid_2nd_term_LinALL;
            piola_kirchhoff_stress_fluid_2nd_term_LinALL = density_fluid
                                                           * viscosity
                                                           * (cauchy_stress_fluid_2nd_term_LinALL * J * F_Inverse_T
                                                              + (v_grad * F_Inverse + F_Inverse_T * transpose(v_grad))
                                                              * J_F_Inverse_T_LinU);

            // Linearization of fluid incompressibiliy: div(JF^{-1}v)'(dv) and div(JF^{-1}v)'(du)
            double incompressiblity_fluid_LinAll;
            incompressiblity_fluid_LinAll = phi_j_grads_v[0][0]
                                            + phi_j_grads_v[1][1]
                                            + (phi_j_grads_u[1][1] * ugrads_[q_point][0][0]
                                               - phi_j_grads_u[0][1] * ugrads_[q_point][1][0]
                                               - phi_j_grads_u[1][0] * ugrads_[q_point][0][1]
                                               + phi_j_grads_u[0][0] * ugrads_[q_point][1][1]);

            // Linearization of fluid convection term
            // rho J(F^{-1}v * \nabla)v = rho J grad(v)F^{-1}v
            Tensor<1, 2> convection_fluid_LinAll;
            convection_fluid_LinAll = density_fluid
                                      * (J_LinU * v_grad * F_Inverse * v
                                         + J * v_grad * F_Inverse_LinU * v
                                         + J
                                         * (phi_j_grads_v * F_Inverse * v
                                            + v_grad * F_Inverse * phi_j_v));

            for (unsigned int i = 0; i < n_dofs_per_element; i++)
              {
                const Tensor<1, 2> phi_i_v = state_fe_values[velocities].value(i,
                                             q_point);
                const Tensor<2, 2> phi_i_grads_v =
                  state_fe_values[velocities].gradient(i, q_point);
                const double phi_i_p = state_fe_values[pressure].value(i,
                                                                       q_point);
                const Tensor<1, 2> phi_i_grads_p =
                  state_fe_values[pressure].gradient(i, q_point); // only for STVK
                const Tensor<1, 2> phi_i_u = state_fe_values[displacements].value(
                                               i, q_point);
                const Tensor<2, 2> phi_i_grads_u =
                  state_fe_values[displacements].gradient(i, q_point);

                // fluid: material_id == 0, structure: material_id == 1
                if (material_id == 0)
                  {
                    local_matrix(i, j) += scale
                                          * (convection_fluid_LinAll * phi_i_v
                                             + scalar_product(piola_kirchhoff_stress_1st_term_LinALL,
                                                              phi_i_grads_v)
                                             + scalar_product(
                                               piola_kirchhoff_stress_fluid_2nd_term_LinALL,
                                               phi_i_grads_v)
                                             + incompressiblity_fluid_LinAll * phi_i_p
                                             + alpha_u * element_diameter * element_diameter
                                             * scalar_product(phi_j_grads_u, phi_i_grads_u))
                                          * state_fe_values.JxW(q_point);

                  }
                else if (material_id == 1)
                  {
                    /*
                     local_matrix(i,j) += scale *  (scalar_product(piola_kirchhoff_stress_1st_term_LinALL, phi_i_grads_v)
                     + scalar_product(piola_kirchhoff_stress_structure_INH_2nd_term_LinALL, phi_i_grads_v)
                     + J_LinU * phi_i_p
                     + phi_j_v * phi_i_u)
                     * state_fe_values.JxW(q_point);
                     */

                    local_matrix(i, j) += scale
                                          * (scalar_product(
                                               piola_kirchhoff_stress_structure_STVK_LinALL,
                                               phi_i_grads_v)
                                             + alpha_p * element_diameter * element_diameter
                                             * phi_j_grads_p * phi_i_grads_p // harmonic pressure
                                             - phi_j_v * phi_i_u) * state_fe_values.JxW(q_point);

                  }
              }
          }
      }
  }

  void
  ElementRightHandSide(
    const EDC<DH, VECTOR, 2> & /*edc*/,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/) override
  {
  }

  // Values for Boundary integrals
  void
  BoundaryEquation(
    const FDC<DH, VECTOR, 2> &fdc,
    dealii::Vector<double> &local_vector, double scale,
    double /*scale_ico*/) override
  {
    assert(this->problem_type_ == "state");
    const auto &state_fe_face_values = fdc.GetFEFaceValuesState();
    unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();
    unsigned int n_q_points = fdc.GetNQPoints();
    unsigned int color = fdc.GetBoundaryIndicator();

    // do-nothing applied on outflow boundary
    if (color == 1)
      {
        uboundarygrads_.resize(n_q_points, vector<Tensor<1, 2> >(5));

        fdc.GetFaceGradsState("last_newton_solution", uboundarygrads_);

        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(2);
        const FEValuesExtractors::Vector displacements(3);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            Tensor<2, 2> F;
            F[0][0] = 1.0 + uboundarygrads_[q_point][3][0];
            F[0][1] = uboundarygrads_[q_point][3][1];
            F[1][0] = uboundarygrads_[q_point][4][0];
            F[1][1] = 1.0 + uboundarygrads_[q_point][4][1];

            double J = determinant(F);

            Tensor<2, 2> v_grad;
            v_grad.clear();
            v_grad[0][0] = uboundarygrads_[q_point][0][0];
            v_grad[0][1] = uboundarygrads_[q_point][0][1];
            v_grad[1][0] = uboundarygrads_[q_point][1][0];
            v_grad[1][1] = uboundarygrads_[q_point][1][1];

            const Tensor<2, 2> do_nothing_ALE = density_fluid * viscosity
                                                * transpose(invert(F)) * transpose(v_grad);

            const Tensor<1, 2> neumann_value = (J * do_nothing_ALE
                                                * transpose(invert(F))
                                                * state_fe_face_values.normal_vector(q_point));

            for (unsigned int i = 0; i < n_dofs_per_element; i++)
              {
                const Tensor<1, 2> phi_i_v =
                  state_fe_face_values[velocities].value(i, q_point);

                local_vector(i) -= scale * neumann_value * phi_i_v
                                   * state_fe_face_values.JxW(q_point);
              }
          }
      }
  }

  void
  BoundaryMatrix(
    const FDC<DH, VECTOR, 2> &fdc,
    dealii::FullMatrix<double> &local_matrix, double scale,
    double /*scale_ico*/) override
  {
    assert(this->problem_type_ == "state");
    const auto &state_fe_face_values = fdc.GetFEFaceValuesState();
    unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();
    unsigned int n_q_points = fdc.GetNQPoints();
    unsigned int color = fdc.GetBoundaryIndicator();

    // do-nothing applied on outflow boundary
    if (color == 1)
      {
        uboundarygrads_.resize(n_q_points, vector<Tensor<1, 2> >(5));

        fdc.GetFaceGradsState("last_newton_solution", uboundarygrads_);

        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Vector displacements(3);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            // ALE Transformations
            Tensor<2, 2> F;
            F[0][0] = 1.0 + uboundarygrads_[q_point][3][0];
            F[0][1] = uboundarygrads_[q_point][3][1];
            F[1][0] = uboundarygrads_[q_point][4][0];
            F[1][1] = 1.0 + uboundarygrads_[q_point][4][1];

            Tensor<2, 2> F_Inverse;
            F_Inverse = invert(F);

            double J;
            J = determinant(F);

            Tensor<2, 2> F_tilde;
            F_tilde[0][0] = 1.0 + uboundarygrads_[q_point][4][1];
            F_tilde[0][1] = -uboundarygrads_[q_point][3][1];
            F_tilde[1][0] = -uboundarygrads_[q_point][4][0];
            F_tilde[1][1] = 1.0 + uboundarygrads_[q_point][3][0];

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
                    const Tensor<2, 2> phi_j_grads_u =
                      state_fe_face_values[displacements].gradient(j, q_point);

                    // ALE Transformations derivatives
                    Tensor<2, 2> J_F_Inverse_T_LinU;
                    J_F_Inverse_T_LinU[0][0] = phi_j_grads_u[1][1];
                    J_F_Inverse_T_LinU[0][1] = -phi_j_grads_u[1][0];
                    J_F_Inverse_T_LinU[1][0] = -phi_j_grads_u[0][1];
                    J_F_Inverse_T_LinU[1][1] = phi_j_grads_u[0][0];

                    Tensor<2, 2> F_tilde_LinU;
                    F_tilde_LinU[0][0] = phi_j_grads_u[1][1];
                    F_tilde_LinU[0][1] = -phi_j_grads_u[0][1];
                    F_tilde_LinU[1][0] = -phi_j_grads_u[1][0];
                    F_tilde_LinU[1][1] = phi_j_grads_u[0][0];

                    double J_LinU;
                    J_LinU = (phi_j_grads_u[0][0]
                              * (1 + uboundarygrads_[q_point][4][1])
                              + (1 + uboundarygrads_[q_point][3][0]) * phi_j_grads_u[1][1]
                              - phi_j_grads_u[0][1] * uboundarygrads_[q_point][4][0]
                              - uboundarygrads_[q_point][3][1] * phi_j_grads_u[1][0]);

                    Tensor<2, 2> F_Inverse_LinU;
                    F_Inverse_LinU = (-1.0 / std::pow(J, 2) * J_LinU * F_tilde
                                      + 1.0 / J * F_tilde_LinU);

                    // do-nothing in ALE
                    Tensor<2, 2> do_nothing_LinAll;
                    do_nothing_LinAll = density_fluid * viscosity
                                        * (J_F_Inverse_T_LinU * transpose(v_grad)
                                           * transpose(F_Inverse)
                                           + J * transpose(F_Inverse) * transpose(phi_j_grads_v)
                                           * transpose(F_Inverse)
                                           + J * transpose(F_Inverse) * transpose(v_grad)
                                           * transpose(F_Inverse_LinU));

                    const Tensor<1, 2> neumann_value = do_nothing_LinAll
                                                       * state_fe_face_values.normal_vector(q_point);

                    local_matrix(i, j) -= scale * neumann_value * phi_i_v
                                          * state_fe_face_values.JxW(q_point);
                  }
              }
          }
      }
  }

  void
  BoundaryRightHandSide(
    const FDC<DH, VECTOR, 2> &,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/) override
  {
  }

  UpdateFlags
  GetUpdateFlags() const override
  {
    return update_values | update_gradients | update_quadrature_points;
  }

  UpdateFlags
  GetFaceUpdateFlags() const override
  {
    return update_values | update_gradients | update_normal_vectors
           | update_quadrature_points;
  }

  unsigned int
  GetStateNBlocks() const override
  {
    return 3;
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
  vector<Vector<double> > uvalues_;
  vector<vector<Tensor<1, dealdim> > > ugrads_;
  vector<vector<Tensor<1, dealdim> > > uboundarygrads_;
  vector<unsigned int> state_block_component_;

  // material variables
  double density_fluid, viscosity, alpha_u, alpha_p, mu, poisson_ratio_nu,
         lambda;

};
#endif
