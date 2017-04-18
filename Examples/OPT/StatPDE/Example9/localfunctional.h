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

#ifndef LOCALFunctional_
#define LOCALFunctional_

#include <interfaces/pdeinterface.h>

using namespace std;
using namespace dealii;
using namespace DOpE;

template<
template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
         template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
         template<int, int> class DH, typename VECTOR, int dopedim, int dealdim =
         dopedim>
class LocalFunctional : public FunctionalInterface<EDC, FDC, DH, VECTOR,
  dopedim, dealdim>
{
public:

  static void
  declare_params(ParameterReader &param_reader)
  {
    param_reader.SetSubsection("Local PDE parameters");
    param_reader.declare_entry("mu_regularization", "0.0",
                               Patterns::Double(0));
    param_reader.declare_entry("density_fluid", "0.0", Patterns::Double(0));
    param_reader.declare_entry("viscosity", "0.0", Patterns::Double(0));
    param_reader.declare_entry("alpha_u", "0.0", Patterns::Double(0));
    param_reader.declare_entry("alpha_p", "0.0", Patterns::Double(0));
    param_reader.declare_entry("mu", "0.0", Patterns::Double(0));
    param_reader.declare_entry("poisson_ratio_nu", "0.0",
                               Patterns::Double(0));

    param_reader.declare_entry("control_constant", "0.0",
                               Patterns::Double());

  }

  LocalFunctional(ParameterReader &param_reader)
  {
    // Control- and regulraization parameters

    // Fluid- and material parameters
    param_reader.SetSubsection("Local PDE parameters");
    mu_regularization_ = param_reader.get_double("mu_regularization");
    density_fluid_ = param_reader.get_double("density_fluid");
    viscosity_ = param_reader.get_double("viscosity");
    alpha_u_ = param_reader.get_double("alpha_u");

    lame_coefficient_mu_ = param_reader.get_double("mu");
    poisson_ratio_nu_ = param_reader.get_double("poisson_ratio_nu");
    lame_coefficient_lambda_ =
      (2 * poisson_ratio_nu_ * lame_coefficient_mu_)
      / (1.0 - 2 * poisson_ratio_nu_);

    control_constant_ = param_reader.get_double("control_constant");
  }

  bool
  HasFaces() const
  {
    return true;
  }

  string
  GetType() const
  {
    return "boundary face";
  }

  string
  GetName() const
  {
    return "cost functional";
  }

  // compute drag value around cylinder
  double
  BoundaryValue(const FDC<DH, VECTOR, dealdim> &fdc)
  {
    const auto &state_fe_face_values = fdc.GetFEFaceValuesState();
    unsigned int n_q_points = fdc.GetNQPoints();
    unsigned int color = fdc.GetBoundaryIndicator();
    double functional_value_J = 0;

    double drag_lift_value = 0.0;
    // Asking for boundary color of the cylinder
    if (color == 80)
      {
        ufacevalues_.resize(n_q_points, Vector<double>(5));
        ufacegrads_.resize(n_q_points, vector<Tensor<1, 2> >(5));

        fdc.GetFaceValuesState("state", ufacevalues_);
        fdc.GetFaceGradsState("state", ufacegrads_);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            Tensor<2, 2> pI;
            pI[0][0] = ufacevalues_[q_point](4);
            pI[1][1] = ufacevalues_[q_point](4);

            Tensor<1, 2> v;
            v.clear();
            v[0] = ufacevalues_[q_point](0);
            v[1] = ufacevalues_[q_point](1);

            Tensor<2, 2> grad_v;
            grad_v[0][0] = ufacegrads_[q_point][0][0];
            grad_v[0][1] = ufacegrads_[q_point][0][1];
            grad_v[1][0] = ufacegrads_[q_point][1][0];
            grad_v[1][1] = ufacegrads_[q_point][1][1];

            Tensor<2, 2> F;
            F[0][0] = 1.0 + ufacegrads_[q_point][2][0];
            F[0][1] = ufacegrads_[q_point][2][1];
            F[1][0] = ufacegrads_[q_point][3][0];
            F[1][1] = 1.0 + ufacegrads_[q_point][3][1];

            Tensor<2, 2> F_Inverse;
            F_Inverse = invert(F);

            Tensor<2, 2> F_T;
            F_T = transpose(F);

            Tensor<2, 2> F_Inverse_T;
            F_Inverse_T = transpose(F_Inverse);

            double J;
            J = determinant(F);

            // constitutive stress tensors for fluid
            Tensor<2, 2> cauchy_stress_fluid;
            cauchy_stress_fluid =
              1.0 * J
              * (-pI
                 + density_fluid_ * viscosity_
                 * (grad_v * F_Inverse
                    + F_Inverse_T * transpose(grad_v)))
              * F_Inverse_T;

            Tensor<1, 2> stress_normal;
            stress_normal = cauchy_stress_fluid
                            * state_fe_face_values.normal_vector(q_point);

            drag_lift_value += 0.5 * stress_normal[0] * stress_normal[0]
                               * state_fe_face_values.JxW(q_point);

          }

      }
    functional_value_J = drag_lift_value; // drag

    // Regularization term of the cost functional
    if (color == 50)
      {
        // Regularization
        qvalues_.reinit(2);
        fdc.GetParamValues("control", qvalues_);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            functional_value_J += mu_regularization_ * 0.5
                                  * ((qvalues_(0) - control_constant_)
                                     * (qvalues_(0) - control_constant_))
                                  * state_fe_face_values.JxW(q_point);
          }

      }
    if (color == 51)
      {
        // Regularization
        qvalues_.reinit(2);
        fdc.GetParamValues("control", qvalues_);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            functional_value_J += mu_regularization_ * 0.5
                                  * ((qvalues_(0) - control_constant_)
                                     * (qvalues_(0) - control_constant_))
                                  * state_fe_face_values.JxW(q_point);
          }

      }
    return functional_value_J;

  }

  void
  BoundaryValue_U(const FDC<DH, VECTOR, dealdim> &fdc,
                  dealii::Vector<double> &local_vector, double scale)
  {
    const auto &state_fe_face_values = fdc.GetFEFaceValuesState();
    unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();
    unsigned int n_q_points = fdc.GetNQPoints();
    unsigned int color = fdc.GetBoundaryIndicator();

    if (color == 80)
      {
        ufacevalues_.resize(n_q_points, Vector<double>(5));
        ufacegrads_.resize(n_q_points, vector<Tensor<1, 2> >(5));

        fdc.GetFaceValuesState("state", ufacevalues_);
        fdc.GetFaceGradsState("state", ufacegrads_);

        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Vector displacements(2);
        const FEValuesExtractors::Scalar pressure(4);

        std::vector<Tensor<1, 2> > phi_v(n_dofs_per_element);
        std::vector<Tensor<2, 2> > phi_grads_v(n_dofs_per_element);
        std::vector<Tensor<1, 2> > phi_u(n_dofs_per_element);
        std::vector<Tensor<2, 2> > phi_grads_u(n_dofs_per_element);
        std::vector<double> phi_p(n_dofs_per_element);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            for (unsigned int k = 0; k < n_dofs_per_element; k++)
              {
                phi_p[k] = state_fe_face_values[pressure].value(k, q_point);
                phi_v[k] = state_fe_face_values[velocities].value(k, q_point);
                phi_grads_v[k] = state_fe_face_values[velocities].gradient(k,
                                                                           q_point);
                phi_u[k] = state_fe_face_values[displacements].value(k, q_point);
                phi_grads_u[k] = state_fe_face_values[displacements].gradient(k,
                                 q_point);
              }

            const Tensor<2, dealdim> pI = ALE_Transformations::get_pI<dealdim>(
                                            q_point, ufacevalues_);
            const Tensor<2, dealdim> grad_v = ALE_Transformations::get_grad_v<
                                              dealdim>(q_point, ufacegrads_);

            const Tensor<2, dealdim> grad_v_T =
              ALE_Transformations::get_grad_v_T<dealdim>(grad_v);
            const Tensor<2, dealdim> F = ALE_Transformations::get_F<dealdim>(
                                           q_point, ufacegrads_);
            const Tensor<2, dealdim> F_Inverse =
              ALE_Transformations::get_F_Inverse<dealdim>(F);

            const Tensor<2, dealdim> F_Inverse_T =
              ALE_Transformations::get_F_Inverse_T<dealdim>(F_Inverse);
            const double J = ALE_Transformations::get_J<dealdim>(F);

            const Tensor<2, dealdim> sigma_ALE =
              NSE_in_ALE::get_stress_fluid_ALE<dealdim>(density_fluid_,
                                                        viscosity_, pI, grad_v, grad_v_T, F_Inverse, F_Inverse_T);

            Tensor<1, 2> stress_normal;
            stress_normal = sigma_ALE
                            * state_fe_face_values.normal_vector(q_point);

            for (unsigned int j = 0; j < n_dofs_per_element; j++)
              {
                const Tensor<2, dealdim> pI_LinP =
                  ALE_Transformations::get_pI_LinP<dealdim>(phi_p[j]);

                const Tensor<2, dealdim> grad_v_LinV =
                  ALE_Transformations::get_grad_v_LinV<dealdim>(phi_grads_v[j]);

                const double J_LinU = ALE_Transformations::get_J_LinU<dealdim>(
                                        q_point, ufacegrads_, phi_grads_u[j]);

                const Tensor<2, dealdim> J_F_Inverse_T_LinU =
                  ALE_Transformations::get_J_F_Inverse_T_LinU<dealdim>(
                    phi_grads_u[j]);

                const Tensor<2, dealdim> F_Inverse_LinU =
                  ALE_Transformations::get_F_Inverse_LinU(phi_grads_u[j], J,
                                                          J_LinU, q_point, ufacegrads_);

                const Tensor<2, dealdim> stress_fluid_ALE_1st_term_LinAll =
                  NSE_in_ALE::get_stress_fluid_ALE_1st_term_LinAll_short<dealdim>(
                    pI, F_Inverse_T, J_F_Inverse_T_LinU, pI_LinP, J);

                const Tensor<2, dealdim> stress_fluid_ALE_2nd_term_LinAll =
                  NSE_in_ALE::get_stress_fluid_ALE_2nd_term_LinAll_short(
                    J_F_Inverse_T_LinU, sigma_ALE, grad_v, grad_v_LinV,
                    F_Inverse, F_Inverse_LinU, J, viscosity_, density_fluid_);

                Tensor<1, 2> neumann_value = (stress_fluid_ALE_1st_term_LinAll
                                              + stress_fluid_ALE_2nd_term_LinAll)
                                             * state_fe_face_values.normal_vector(q_point);

                local_vector(j) += scale * neumann_value[0]
                                   * stress_normal[0] * state_fe_face_values.JxW(q_point);

              }
          }
      }
  }

  void
  BoundaryValue_Q(const FDC<DH, VECTOR, dealdim> &fdc,
                  dealii::Vector<double> &local_vector, double scale)
  {
    const auto &state_fe_face_values = fdc.GetFEFaceValuesState();
    unsigned int n_dofs_per_element = local_vector.size();
    unsigned int n_q_points = fdc.GetNQPoints();
    unsigned int color = fdc.GetBoundaryIndicator();

    if (color == 50)
      {
        // Regularization
        qvalues_.reinit(2);
        fdc.GetParamValues("control", qvalues_);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            for (unsigned int j = 0; j < n_dofs_per_element; j++)
              {
                local_vector(j) += scale * mu_regularization_
                                   * (qvalues_(j) - control_constant_)
                                   * state_fe_face_values.JxW(q_point);
              }
          }
      }
    if (color == 51)
      {
        // Regularization
        qvalues_.reinit(2);
        fdc.GetParamValues("control", qvalues_);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            for (unsigned int j = 0; j < n_dofs_per_element; j++)
              {
                local_vector(j) += scale * mu_regularization_
                                   * (qvalues_(j) - control_constant_)
                                   * state_fe_face_values.JxW(q_point);
              }
          }
      }
  }

  void
  BoundaryValue_QQ(const FDC<DH, VECTOR, dealdim> &fdc,
                   dealii::Vector<double> &local_vector, double scale)
  {
    const auto &state_fe_face_values = fdc.GetFEFaceValuesState();
    unsigned int n_dofs_per_element = local_vector.size();
    unsigned int n_q_points = fdc.GetNQPoints();
    unsigned int color = fdc.GetBoundaryIndicator();

    if (color == 50)
      {
        // Regularization
        dqvalues_.reinit(2);
        fdc.GetParamValues("dq", dqvalues_);

        qvalues_.reinit(2);
        fdc.GetParamValues("control", qvalues_);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            for (unsigned int j = 0; j < n_dofs_per_element; j++)
              {
                local_vector(j) += scale * mu_regularization_
                                   * (dqvalues_(j)) * state_fe_face_values.JxW(q_point);
              }
          }
      }
    if (color == 51)
      {
        // Regularization
        dqvalues_.reinit(2);
        fdc.GetParamValues("dq", dqvalues_);

        qvalues_.reinit(2);
        fdc.GetParamValues("control", qvalues_);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            for (unsigned int j = 0; j < n_dofs_per_element; j++)
              {
                local_vector(j) += scale * mu_regularization_
                                   * (dqvalues_(j)) * state_fe_face_values.JxW(q_point);
              }
          }
      }
  }

  void
  BoundaryValue_UU(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                   dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {

  }

  void
  BoundaryValue_QU(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                   dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {

  }

  void
  BoundaryValue_UQ(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                   dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {

  }

// compute drag value around cylinder
  double
  FaceValue(const FDC<DH, VECTOR, dealdim> &fdc)
  {
    const auto &state_fe_face_values = fdc.GetFEFaceValuesState();
    unsigned int n_q_points = fdc.GetNQPoints();
    unsigned int material_id = fdc.GetMaterialId();
    unsigned int material_id_neighbor = fdc.GetNbrMaterialId();
    bool at_boundary = fdc.GetIsAtBoundary();

    double drag_lift_value = 0.0;
    if (material_id == 0)
      {
        if ((material_id != material_id_neighbor) && (!at_boundary))
          {
            ufacevalues_.resize(n_q_points, Vector<double>(5));
            ufacegrads_.resize(n_q_points, vector<Tensor<1, 2> >(5));

            fdc.GetFaceValuesState("state", ufacevalues_);
            fdc.GetFaceGradsState("state", ufacegrads_);

            for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
              {
                Tensor<2, 2> pI;
                pI[0][0] = ufacevalues_[q_point](4);
                pI[1][1] = ufacevalues_[q_point](4);

                Tensor<1, 2> v;
                v.clear();
                v[0] = ufacevalues_[q_point](0);
                v[1] = ufacevalues_[q_point](1);

                Tensor<2, 2> grad_v;
                grad_v[0][0] = ufacegrads_[q_point][0][0];
                grad_v[0][1] = ufacegrads_[q_point][0][1];
                grad_v[1][0] = ufacegrads_[q_point][1][0];
                grad_v[1][1] = ufacegrads_[q_point][1][1];

                Tensor<2, 2> F;
                F[0][0] = 1.0 + ufacegrads_[q_point][2][0];
                F[0][1] = ufacegrads_[q_point][2][1];
                F[1][0] = ufacegrads_[q_point][3][0];
                F[1][1] = 1.0 + ufacegrads_[q_point][3][1];

                Tensor<2, 2> F_Inverse;
                F_Inverse = invert(F);

                Tensor<2, 2> F_T;
                F_T = transpose(F);

                Tensor<2, 2> F_Inverse_T;
                F_Inverse_T = transpose(F_Inverse);

                double J;
                J = determinant(F);

                // constitutive stress tensors for fluid
                Tensor<2, 2> cauchy_stress_fluid;
                cauchy_stress_fluid = J
                                      * (-pI
                                         + density_fluid_ * viscosity_
                                         * (grad_v * F_Inverse
                                            + F_Inverse_T * transpose(grad_v))) * F_Inverse_T;

                Tensor<1, 2> stress_normal;
                stress_normal = cauchy_stress_fluid
                                * state_fe_face_values.normal_vector(q_point);

                drag_lift_value += 0.5 * stress_normal[0] * stress_normal[0]
                                   * state_fe_face_values.JxW(q_point);

              }
          }
      }
    return drag_lift_value; // drag

  }

  void
  FaceValue_U(const FDC<DH, VECTOR, dealdim> &fdc,
              dealii::Vector<double> &local_vector, double scale)
  {
    const auto &state_fe_face_values = fdc.GetFEFaceValuesState();
    unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();
    unsigned int n_q_points = fdc.GetNQPoints();
    unsigned int material_id = fdc.GetMaterialId();
    unsigned int material_id_neighbor = fdc.GetNbrMaterialId();
    bool at_boundary = fdc.GetIsAtBoundary();

    if (material_id == 0)
      {
        if ((material_id != material_id_neighbor) && (!at_boundary))
          {
            ufacevalues_.resize(n_q_points, Vector<double>(5));
            ufacegrads_.resize(n_q_points, vector<Tensor<1, 2> >(5));

            fdc.GetFaceValuesState("state", ufacevalues_);
            fdc.GetFaceGradsState("state", ufacegrads_);

            const FEValuesExtractors::Vector velocities(0);
            const FEValuesExtractors::Vector displacements(2);
            const FEValuesExtractors::Scalar pressure(4);

            std::vector<Tensor<1, 2> > phi_v(n_dofs_per_element);
            std::vector<Tensor<2, 2> > phi_grads_v(n_dofs_per_element);
            std::vector<Tensor<1, 2> > phi_u(n_dofs_per_element);
            std::vector<Tensor<2, 2> > phi_grads_u(n_dofs_per_element);
            std::vector<double> phi_p(n_dofs_per_element);

            for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
              {
                for (unsigned int k = 0; k < n_dofs_per_element; k++)
                  {
                    phi_p[k] = state_fe_face_values[pressure].value(k, q_point);
                    phi_v[k] = state_fe_face_values[velocities].value(k, q_point);
                    phi_grads_v[k] = state_fe_face_values[velocities].gradient(k,
                                                                               q_point);
                    phi_u[k] = state_fe_face_values[displacements].value(k,
                                                                         q_point);
                    phi_grads_u[k] = state_fe_face_values[displacements].gradient(k,
                                     q_point);
                  }

                const Tensor<2, dealdim> pI =
                  ALE_Transformations::get_pI<dealdim>(q_point, ufacevalues_);
                const Tensor<2, dealdim> grad_v = ALE_Transformations::get_grad_v<
                                                  dealdim>(q_point, ufacegrads_);

                const Tensor<2, dealdim> grad_v_T =
                  ALE_Transformations::get_grad_v_T<dealdim>(grad_v);
                const Tensor<2, dealdim> F = ALE_Transformations::get_F<dealdim>(
                                               q_point, ufacegrads_);
                const Tensor<2, dealdim> F_Inverse =
                  ALE_Transformations::get_F_Inverse<dealdim>(F);

                const Tensor<2, dealdim> F_Inverse_T =
                  ALE_Transformations::get_F_Inverse_T<dealdim>(F_Inverse);
                const double J = ALE_Transformations::get_J<dealdim>(F);

                const Tensor<2, dealdim> sigma_ALE =
                  NSE_in_ALE::get_stress_fluid_ALE<dealdim>(density_fluid_,
                                                            viscosity_, pI, grad_v, grad_v_T, F_Inverse, F_Inverse_T);

                Tensor<1, 2> stress_normal;
                stress_normal = sigma_ALE
                                * state_fe_face_values.normal_vector(q_point);

                for (unsigned int j = 0; j < n_dofs_per_element; j++)
                  {
                    const Tensor<2, dealdim> pI_LinP =
                      ALE_Transformations::get_pI_LinP<dealdim>(phi_p[j]);

                    const Tensor<2, dealdim> grad_v_LinV =
                      ALE_Transformations::get_grad_v_LinV<dealdim>(
                        phi_grads_v[j]);

                    const double J_LinU = ALE_Transformations::get_J_LinU<dealdim>(
                                            q_point, ufacegrads_, phi_grads_u[j]);

                    const Tensor<2, dealdim> J_F_Inverse_T_LinU =
                      ALE_Transformations::get_J_F_Inverse_T_LinU<dealdim>(
                        phi_grads_u[j]);

                    const Tensor<2, dealdim> F_Inverse_LinU =
                      ALE_Transformations::get_F_Inverse_LinU(phi_grads_u[j], J,
                                                              J_LinU, q_point, ufacegrads_);

                    const Tensor<2, dealdim> stress_fluid_ALE_1st_term_LinAll =
                      NSE_in_ALE::get_stress_fluid_ALE_1st_term_LinAll_short<
                      dealdim>(pI, F_Inverse_T, J_F_Inverse_T_LinU, pI_LinP,
                               J);
                    const Tensor<2, dealdim> stress_fluid_ALE_2nd_term_LinAll =
                      NSE_in_ALE::get_stress_fluid_ALE_2nd_term_LinAll_short(
                        J_F_Inverse_T_LinU, sigma_ALE, grad_v, grad_v_LinV,
                        F_Inverse, F_Inverse_LinU, J, viscosity_,
                        density_fluid_);

                    Tensor<1, 2> neumann_value = (stress_fluid_ALE_1st_term_LinAll
                                                  + stress_fluid_ALE_2nd_term_LinAll)
                                                 * state_fe_face_values.normal_vector(q_point);

                    local_vector(j) += scale * neumann_value[0]
                                       * stress_normal[0] * state_fe_face_values.JxW(q_point);

                  }
              }
          }
      }
  }

  void
  FaceValue_UU(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
               dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {

  }

  void
  FaceValue_Q(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
              dealii::Vector<double> & /*local_vector*/, double /*scale*/)
  {

  }

  void
  FaceValue_QU(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
               dealii::Vector<double> & /*local_vector*/, double /*scale*/)
  {

  }

  void
  FaceValue_UQ(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
               dealii::Vector<double> & /*local_vector*/, double /*scale*/)
  {

  }

  void
  FaceValue_QQ(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
               dealii::Vector<double> & /*local_vector*/, double /*scale*/)
  {

  }

  double
  ElementValue(const EDC<DH, VECTOR, dealdim> & /*edc*/)
  {
    return 0.;
  }

  void
  ElementValue_U(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                 dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {

  }

  void
  ElementValue_Q(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                 dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {

  }

  void
  ElementValue_UU(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                  dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {

  }

  void
  ElementValue_QU(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                  dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
  }

  void
  ElementValue_UQ(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                  dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
  }

  void
  ElementValue_QQ(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                  dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {

  }

  UpdateFlags
  GetFaceUpdateFlags() const
  {
    return update_values | update_quadrature_points | update_gradients
           | update_normal_vectors;
  }

  inline void
  GetFaceValues(const DOpEWrapper::FEFaceValues<dealdim> &fe_face_values,
                const map<string, const BlockVector<double>*> &domain_values,
                string name, vector<Vector<double> > &values)
  {
    typename map<string, const BlockVector<double>*>::const_iterator it =
      domain_values.find(name);
    if (it == domain_values.end())
      {
        throw DOpEException("Did not find " + name, "LocalPDE::GetValues");
      }
    fe_face_values.get_function_values(*(it->second), values);
  }

  inline void
  GetFaceGrads(const DOpEWrapper::FEFaceValues<dealdim> &fe_face_values,
               const map<string, const BlockVector<double>*> &domain_values,
               string name, vector<vector<Tensor<1, dealdim> > > &values)
  {
    typename map<string, const BlockVector<double>*>::const_iterator it =
      domain_values.find(name);
    if (it == domain_values.end())
      {
        throw DOpEException("Did not find " + name, "LocalPDE::GetGrads");
      }
    fe_face_values.get_function_grads(*(it->second), values);
  }

  inline void
  GetValues(const DOpEWrapper::FEValues<dealdim> &fe_values,
            const map<string, const BlockVector<double>*> &domain_values,
            string name, vector<Vector<double> > &values)
  {
    typename map<string, const BlockVector<double>*>::const_iterator it =
      domain_values.find(name);
    if (it == domain_values.end())
      {
        throw DOpEException("Did not find " + name, "LocalPDE::GetValues");
      }
    fe_values.get_function_values(*(it->second), values);
  }

  inline void
  GetParams(const map<string, const Vector<double>*> &param_values,
            string name, Vector<double> &values)
  {
    typename map<string, const Vector<double>*>::const_iterator it =
      param_values.find(name);
    if (it == param_values.end())
      {
        throw DOpEException("Did not find " + name, "LocalPDE::GetValues");
      }
    values = *(it->second);
  }
private:
  Vector<double> qvalues_;
  Vector<double> dqvalues_;
  vector<Vector<double> > ufacevalues_;
  vector<Vector<double> > dufacevalues_;

  vector<vector<Tensor<1, dealdim> > > ufacegrads_;
  vector<vector<Tensor<1, dealdim> > > dufacegrads;

  // Artifcial parameter for FSI (later)
  double alpha_u_;

  // Fluid- and material parameters
  double density_fluid_, viscosity_, lame_coefficient_mu_,
         poisson_ratio_nu_, lame_coefficient_lambda_;

  // Control- and regularization parameters
  double mu_regularization_, control_constant_;

};
#endif
