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

using namespace std;
using namespace dealii;
using namespace DOpE;

template<
template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
         template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
         template<int, int> class DH, typename VECTOR, int dealdim>
class LocalPDE : public PDEInterface<EDC, FDC, DH, VECTOR, dealdim>
{
public:
  LocalPDE() :
    state_block_component_(5, 0)
  {
    assert(dealdim==2);
    //first block velocity, second block pressure, third block displacement
    state_block_component_[2] = 1;
    state_block_component_[3] = 2;
    state_block_component_[4] = 2;
  }

  // Domain values for elements
  void
  ElementEquation(const EDC<DH, VECTOR, dealdim> &edc,
                  dealii::Vector<double> &local_vector, double scale,
                  double /*scale_ico*/)
  {
    assert(this->problem_type_ == "state");
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    const unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    const unsigned int n_q_points = edc.GetNQPoints();
    const unsigned int material_id = edc.GetMaterialId();

    const double density_fluid = 1.0;
    const double viscosity = 1.0;
    const double alpha_u = 1.0e-10;
    const double mu = 1.0e+3;

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

        Tensor<2, 2> F_Inverse_T;
        F_Inverse_T = transpose(F_Inverse);

        double J;
        J = determinant(F);

        Tensor<2, 2> pI;
        pI[0][0] = uvalues_[q_point](2);
        pI[1][1] = uvalues_[q_point](2);

        Tensor<1,2> grad_p;
        grad_p[0] = ugrads_[q_point][2][0];
        grad_p[1] = ugrads_[q_point][2][1];

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

        //div(J * F^{-1} * v)
        double incompressiblity_fluid;
        incompressiblity_fluid = (ugrads_[q_point][0][0]
                                  + ugrads_[q_point][4][1] * ugrads_[q_point][0][0]
                                  - ugrads_[q_point][3][1] * ugrads_[q_point][1][0]
                                  - ugrads_[q_point][4][0] * ugrads_[q_point][0][1]
                                  + ugrads_[q_point][1][1]
                                  + ugrads_[q_point][3][0] * ugrads_[q_point][1][1]);

        // constitutive stress tensors for fluid and structure
        Tensor<2, 2> cauchy_stress_fluid;
        cauchy_stress_fluid = -pI
                              + density_fluid * viscosity
                              * (v_grad * F_Inverse + F_Inverse_T * transpose(v_grad));

        Tensor<2, 2> cauchy_stress_structure;
        cauchy_stress_structure = mu * (F * transpose(F) - I);

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            const Tensor<2, 2> phi_i_grads_v =
              state_fe_values[velocities].gradient(i, q_point);
            const double phi_i_p = state_fe_values[pressure].value(i, q_point);
            const Tensor<1,2> phi_i_grads_p = state_fe_values[pressure].gradient(i, q_point);

            const Tensor<1, 2> phi_i_u = state_fe_values[displacements].value(i,
                                         q_point);
            const Tensor<2, 2> phi_i_grads_u =
              state_fe_values[displacements].gradient(i, q_point);

            // fluid: material_id == 0, structure: material_id == 1
            if (material_id == 0)
              {
                local_vector(i) += scale
                                   * (scalar_product(J * cauchy_stress_fluid * F_Inverse_T,
                                                     phi_i_grads_v) + incompressiblity_fluid * phi_i_p
                                      + alpha_u * scalar_product(u_grad, phi_i_grads_u))
                                   * state_fe_values.JxW(q_point);
              }
            else if (material_id == 1)
              {
                local_vector(i) += scale
                                   * (scalar_product(J * cauchy_stress_structure * F_Inverse_T,
                                                     phi_i_grads_v) + alpha_u * grad_p * phi_i_grads_p
                                      + v * phi_i_u)
                                   * state_fe_values.JxW(q_point);
              }
          }
      }

  }

  void
  ElementMatrix(const EDC<DH, VECTOR, dealdim> &edc,
                FullMatrix<double> &local_matrix, double scale,
                double /*scale_ico*/)
  {
    assert(this->problem_type_ == "state");
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    const unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    const unsigned int n_q_points = edc.GetNQPoints();
    const unsigned int material_id = edc.GetMaterialId();

    const double density_fluid = 1.0;
    const double viscosity = 1.0;
    const double alpha_u = 1.0e-10;
    const double mu = 1.0e+3;

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

        Tensor<2, 2> F_Inverse_T;
        F_Inverse_T = transpose(F_Inverse);

        double J;
        J = determinant(F);

        Tensor<2, 2> F_tilde;
        F_tilde[0][0] = 1.0 + ugrads_[q_point][4][1];
        F_tilde[0][1] = -ugrads_[q_point][3][1];
        F_tilde[1][0] = -ugrads_[q_point][4][0];
        F_tilde[1][1] = 1.0 + ugrads_[q_point][3][0];

        Tensor<2, 2> pI;
        pI[0][0] = uvalues_[q_point](2);
        pI[1][1] = uvalues_[q_point](2);

        Tensor<1,2> grad_p;
        grad_p[0] = ugrads_[q_point][2][0];
        grad_p[1] = ugrads_[q_point][2][1];

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
            double phi_j_p = state_fe_values[pressure].value(j, q_point);
            const Tensor<1,2> phi_j_grads_p = state_fe_values[pressure].gradient(j, q_point);
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

            // more derivatives
            Tensor<2, 2> pI_LinP;
            pI_LinP[0][0] = phi_j_p;
            pI_LinP[0][1] = 0.0;
            pI_LinP[1][0] = 0.0;
            pI_LinP[1][1] = phi_j_p;

            // same derivative function for fluid and structure (1st term)
            // sigma * J * F^{-T} = -pI * J * F^{-T}  --> Linearization
            Tensor<2, 2> piola_kirchhoff_stress_1st_term_LinALL;
            piola_kirchhoff_stress_1st_term_LinALL = -pI_LinP * J * F_Inverse_T
                                                     - pI * J_F_Inverse_T_LinU;

            // piola-kirchhoff stress structure linearized in all directions (2nd term)
            // J * (mu * (FF^T - I)) * F^{-T} --> Linearization
            Tensor<2, 2> piola_kirchhoff_stress_structure_2nd_term_LinALL;
            piola_kirchhoff_stress_structure_2nd_term_LinALL = mu
                                                               * (J_LinU * F + J * F_LinU - J_F_Inverse_T_LinU);

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

            for (unsigned int i = 0; i < n_dofs_per_element; i++)
              {
                const Tensor<2, 2> phi_i_grads_v =
                  state_fe_values[velocities].gradient(i, q_point);
                const double phi_i_p = state_fe_values[pressure].value(i,
                                                                       q_point);
                const Tensor<1,2> phi_i_grads_p = state_fe_values[pressure].gradient(i, q_point);
                const Tensor<1, 2> phi_i_u = state_fe_values[displacements].value(
                                               i, q_point);
                const Tensor<2, 2> phi_i_grads_u =
                  state_fe_values[displacements].gradient(i, q_point);

                // fluid: material_id == 0, structure: material_id == 1
                if (material_id == 0)
                  {
                    local_matrix(i, j) +=
                      scale
                      * (scalar_product(
                           piola_kirchhoff_stress_1st_term_LinALL,
                           phi_i_grads_v)
                         + scalar_product(
                           piola_kirchhoff_stress_fluid_2nd_term_LinALL,
                           phi_i_grads_v)
                         + incompressiblity_fluid_LinAll * phi_i_p
                         + alpha_u
                         * scalar_product(phi_j_grads_u, phi_i_grads_u))
                      * state_fe_values.JxW(q_point);

                  }
                else if (material_id == 1)
                  {
                    local_matrix(i, j) += scale
                                          * (scalar_product(
                                               piola_kirchhoff_stress_structure_2nd_term_LinALL,
                                               phi_i_grads_v) + alpha_u * phi_j_grads_p * phi_i_grads_p
                                             + phi_j_v * phi_i_u) * state_fe_values.JxW(q_point);
                  }
              }
          }
      }

  }

  void
  ElementRightHandSide(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                       dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
    assert(this->problem_type_ == "state");
  }

  // Values for Boundary integrals
  void
  BoundaryEquation(const FDC<DH, VECTOR, dealdim> &fdc,
                   dealii::Vector<double> &local_vector, double scale,
                   double /*scale_ico*/)
  {
    assert(this->problem_type_ == "state");
    const auto &state_fe_face_values = fdc.GetFEFaceValuesState();
    unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();
    unsigned int n_q_points = fdc.GetNQPoints();
    unsigned int color = fdc.GetBoundaryIndicator();

    // do-nothing applied on outflow boundary
    if (color == 1)
      {
        double density_fluid = 1.0;
        double viscosity = 1.0;

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

                local_vector(i) += -scale * neumann_value * phi_i_v
                                   * state_fe_face_values.JxW(q_point);
              }
          }
      }
  }

  void
  BoundaryMatrix(const FDC<DH, VECTOR, dealdim> &fdc,
                 dealii::FullMatrix<double> &local_matrix, double scale,
                 double /*scale_ico*/)
  {
    assert(this->problem_type_ == "state");
    const auto &state_fe_face_values = fdc.GetFEFaceValuesState();
    unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();
    unsigned int n_q_points = fdc.GetNQPoints();
    unsigned int color = fdc.GetBoundaryIndicator();

    // do-nothing applied on outflow boundary
    if (color == 1)
      {
        double density_fluid = 1.0;
        double viscosity = 1.0;

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

                    local_matrix(i, j) += -scale * neumann_value * phi_i_v
                                          * state_fe_face_values.JxW(q_point);
                  }
              }
          }
      }
  }

  void
  BoundaryRightHandSide(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                        dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
  }

  UpdateFlags
  GetUpdateFlags() const
  {
    return update_values | update_gradients | update_quadrature_points;
  }

  UpdateFlags
  GetFaceUpdateFlags() const
  {
    return update_values | update_gradients | update_normal_vectors
           | update_quadrature_points;
  }

  unsigned int
  GetStateNBlocks() const
  {
    return 3;
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
  vector<Vector<double> > uvalues_;

  vector<vector<Tensor<1, dealdim> > > ugrads_;

  // boundary values
  vector<vector<Tensor<1, dealdim> > > uboundarygrads_;

  vector<unsigned int> state_block_component_;
};
#endif
