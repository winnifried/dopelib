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

  static void
  declare_params(ParameterReader &param_reader)
  {
    param_reader.SetSubsection("Local PDE parameters");

    param_reader.declare_entry("m_biot", "0.0", Patterns::Double(0));
    param_reader.declare_entry("alpha_biot", "0.0", Patterns::Double(0));
    param_reader.declare_entry("viscosity_biot", "0.0",
                               Patterns::Double(0));
    param_reader.declare_entry("K_biot", "0.0", Patterns::Double(0));
    param_reader.declare_entry("density_biot", "0.0", Patterns::Double(0));
    param_reader.declare_entry("volume_source", "0.0", Patterns::Double(0));

    param_reader.declare_entry("density_structure", "0.0",
                               Patterns::Double(0));
    param_reader.declare_entry("mu", "0.0", Patterns::Double(0));
    param_reader.declare_entry("poisson_ratio_nu", "0.4",
                               Patterns::Double(0));
  }

  LocalPDE(ParameterReader &param_reader) :
    state_block_component_(3, 0)
  {
    state_block_component_[2] = 1;

    param_reader.SetSubsection("Local PDE parameters");
    M_biot = param_reader.get_double("m_biot");
    c_biot = 1.0 / M_biot;

    alpha_biot = param_reader.get_double("alpha_biot");
    viscosity_biot = param_reader.get_double("viscosity_biot");
    K_biot = param_reader.get_double("K_biot");
    density_biot = param_reader.get_double("density_biot");
    volume_source = param_reader.get_double("volume_source");

    density_structure = param_reader.get_double("density_structure");
    lame_coefficient_mu = param_reader.get_double("mu");
    poisson_ratio_nu = param_reader.get_double("poisson_ratio_nu");

    lame_coefficient_lambda = (2 * poisson_ratio_nu * lame_coefficient_mu)
                              / (1.0 - 2 * poisson_ratio_nu);

    traction_x_biot = 0.0;
    traction_y_biot = -1.0e+7;
  }

  bool
  HasFaces() const
  {
    return true;
  }

  // The part of ElementEquation scaled by scale contains all "normal" terms which
  // can be treated by full "theta" time-discretization
  void
  ElementEquation(const EDC<DH, VECTOR, dealdim> &edc,
                  dealii::Vector<double> &local_vector, double scale,
                  double scale_ico)
  {
    assert(this->problem_type_ == "state");

    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();
    unsigned int material_id = edc.GetMaterialId();
//        double element_diameter = edc.GetElementDiameter();

    // old Newton step solution values and gradients
    uvalues_.resize(n_q_points, Vector<double>(3));
    ugrads_.resize(n_q_points, vector<Tensor<1, 2> >(3));

    edc.GetValuesState("last_newton_solution", uvalues_);
    edc.GetGradsState("last_newton_solution", ugrads_);

    // old timestep solution values and gradients
    last_timestep_uvalues_.resize(n_q_points, Vector<double>(3));
    last_timestep_ugrads_.resize(n_q_points, vector<Tensor<1, 2> >(3));

    edc.GetValuesState("last_time_solution", last_timestep_uvalues_);
    edc.GetGradsState("last_time_solution", last_timestep_ugrads_);

    const FEValuesExtractors::Vector displacements(0);
    const FEValuesExtractors::Scalar pressure(2);

    Tensor<2, dealdim> Identity;
    Identity[0][0] = 1.0;
    Identity[0][1] = 0.0;
    Identity[1][0] = 0.0;
    Identity[1][1] = 1.0;

    // pay-zone
    if (material_id == 1)
      {
        for (unsigned int q = 0; q < n_q_points; q++)
          {
            Tensor<2, dealdim> fluid_pressure;
            fluid_pressure.clear();
            fluid_pressure[0][0] = uvalues_[q](2);
            fluid_pressure[1][1] = uvalues_[q](2);

            Tensor<1, dealdim> grad_p;
            grad_p[0] = ugrads_[q][2][0];
            grad_p[1] = ugrads_[q][2][1];

            Tensor<1, dealdim> u;
            u[0] = uvalues_[q](0);
            u[1] = uvalues_[q](1);

//            double p = uvalues_[q](2);

            Tensor<2, dealdim> grad_u;
            grad_u[0][0] = ugrads_[q][0][0];
            grad_u[0][1] = ugrads_[q][0][1];
            grad_u[1][0] = ugrads_[q][1][0];
            grad_u[1][1] = ugrads_[q][1][1];

//            const double divergence_u = ugrads_[q][0][0] + ugrads_[q][1][1];
//            const double old_timestep_divergence_u =
//                last_timestep_ugrads_[q][0][0] + last_timestep_ugrads_[q][1][1];

            double q_biot = volume_source;

            Tensor<2, dealdim> E = 0.5 * (grad_u + transpose(grad_u));

            Tensor<2, dealdim> sigma_s = 2.0 * lame_coefficient_mu * E
                                         + lame_coefficient_lambda * trace(E) * Identity;

            for (unsigned int i = 0; i < n_dofs_per_element; i++)
              {
                // Fluid, explicit
                const Tensor<2, dealdim> phi_i_grads_u =
                  state_fe_values[displacements].gradient(i, q);
                const double phi_i_p = state_fe_values[pressure].value(i, q);
                const Tensor<1, dealdim> phi_i_grads_p =
                  state_fe_values[pressure].gradient(i, q);

                local_vector(i) += scale_ico
                                   * (scalar_product(sigma_s, phi_i_grads_u)
                                      + alpha_biot
                                      * scalar_product(-fluid_pressure, phi_i_grads_u))
                                   * state_fe_values.JxW(q);

                local_vector(i) += scale
                                   * (1.0 / viscosity_biot * K_biot * grad_p * phi_i_grads_p
                                      // Right hand side
                                      - q_biot * phi_i_p
                                     ) * state_fe_values.JxW(q);

              }
          }
      }
    // non pay zone
    else if (material_id == 0)
      {
        for (unsigned int q = 0; q < n_q_points; q++)
          {
//            double p = uvalues_[q](2);

            Tensor<1, dealdim> grad_p;
            grad_p[0] = ugrads_[q][2][0];
            grad_p[1] = ugrads_[q][2][1];

            Tensor<2, dealdim> grad_u;
            grad_u[0][0] = ugrads_[q][0][0];
            grad_u[0][1] = ugrads_[q][0][1];
            grad_u[1][0] = ugrads_[q][1][0];
            grad_u[1][1] = ugrads_[q][1][1];

            Tensor<2, dealdim> E = 0.5 * (grad_u + transpose(grad_u));
            Tensor<2, dealdim> sigma_s = 2.0 * lame_coefficient_mu * E
                                         + lame_coefficient_lambda * trace(E) * Identity;

            for (unsigned int i = 0; i < n_dofs_per_element; i++)
              {
                // Structure, STVK, explicit
                const Tensor<2, dealdim> phi_i_grads_u =
                  state_fe_values[displacements].gradient(i, q);

//              const Tensor<1, dealdim> phi_i_grads_p =
//                  state_fe_values[pressure].gradient(i, q);

//              const double phi_i_p = state_fe_values[pressure].value(i, q);

                local_vector(i) += scale_ico
                                   * scalar_product(sigma_s, phi_i_grads_u)
                                   * state_fe_values.JxW(q);

              }
          }
      }
  }

  void
  ElementMatrix(const EDC<DH, VECTOR, dealdim> &edc,
                FullMatrix<double> &local_matrix, double scale,
                double scale_ico)
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();
    unsigned int material_id = edc.GetMaterialId();
//        double element_diameter = edc.GetElementDiameter();

    // old Newton step solution values and gradients
    uvalues_.resize(n_q_points, Vector<double>(3));
    ugrads_.resize(n_q_points, vector<Tensor<1, 2> >(3));

    edc.GetValuesState("last_newton_solution", uvalues_);
    edc.GetGradsState("last_newton_solution", ugrads_);

    // old timestep solution values and gradients
    last_timestep_uvalues_.resize(n_q_points, Vector<double>(3));
    last_timestep_ugrads_.resize(n_q_points, vector<Tensor<1, 2> >(3));

    edc.GetValuesState("last_time_solution", last_timestep_uvalues_);
    edc.GetGradsState("last_time_solution", last_timestep_ugrads_);

    const FEValuesExtractors::Vector displacements(0);
    const FEValuesExtractors::Scalar pressure(2);

    std::vector<Tensor<1, 2> > phi_i_u(n_dofs_per_element);
    std::vector<Tensor<2, 2> > phi_i_grads_u(n_dofs_per_element);
    std::vector<double> phi_i_p(n_dofs_per_element);
    std::vector<Tensor<1, 2> > phi_i_grads_p(n_dofs_per_element);

    Tensor<2, dealdim> Identity;
    Identity[0][0] = 1.0;
    Identity[0][1] = 0.0;
    Identity[1][0] = 0.0;
    Identity[1][1] = 1.0;

    // pay zone
    if (material_id == 1)
      {
        for (unsigned int q = 0; q < n_q_points; q++)
          {
            for (unsigned int k = 0; k < n_dofs_per_element; k++)
              {
                phi_i_p[k] = state_fe_values[pressure].value(k, q);
                phi_i_grads_p[k] = state_fe_values[pressure].gradient(k, q);
                phi_i_u[k] = state_fe_values[displacements].value(k, q);
                phi_i_grads_u[k] = state_fe_values[displacements].gradient(k, q);
              }

            Tensor<2, dealdim> grad_u;
            grad_u[0][0] = ugrads_[q][0][0];
            grad_u[0][1] = ugrads_[q][0][1];
            grad_u[1][0] = ugrads_[q][1][0];
            grad_u[1][1] = ugrads_[q][1][1];

            for (unsigned int i = 0; i < n_dofs_per_element; i++)
              {
                Tensor<2, dealdim> pI_LinP;
                pI_LinP.clear();
                pI_LinP[0][0] = phi_i_p[i];
                pI_LinP[1][1] = phi_i_p[i];

//              double divergence_u_LinU = phi_i_grads_u[i][0][0]
//                  + phi_i_grads_u[i][1][1];

                Tensor<2, dealdim> E = 0.5
                                       * (phi_i_grads_u[i] + transpose(phi_i_grads_u[i]));

                Tensor<2, dealdim> sigma_s = 2.0 * lame_coefficient_mu * E
                                             + lame_coefficient_lambda * trace(E) * Identity;

                for (unsigned int j = 0; j < n_dofs_per_element; j++)
                  {
                    local_matrix(j, i) += scale_ico
                                          * (scalar_product(sigma_s, phi_i_grads_u[j])
                                             + alpha_biot
                                             * scalar_product(-pI_LinP, phi_i_grads_u[j]))
                                          * state_fe_values.JxW(q);

                    local_matrix(j, i) += scale
                                          * (1.0 / viscosity_biot * K_biot * phi_i_grads_p[i]
                                             * phi_i_grads_p[j]) * state_fe_values.JxW(q);

                  }
              }
          }
      } // end material_id = 1
    else if (material_id == 0)
      {
        for (unsigned int q = 0; q < n_q_points; q++)
          {
            for (unsigned int k = 0; k < n_dofs_per_element; k++)
              {
                phi_i_u[k] = state_fe_values[displacements].value(k, q);
                phi_i_grads_u[k] = state_fe_values[displacements].gradient(k, q);
                phi_i_p[k] = state_fe_values[pressure].value(k, q);
                phi_i_grads_p[k] = state_fe_values[pressure].gradient(k, q);
              }

            for (unsigned int i = 0; i < n_dofs_per_element; i++)
              {
                Tensor<2, dealdim> E = 0.5
                                       * (phi_i_grads_u[i] + transpose(phi_i_grads_u[i]));

                Tensor<2, dealdim> sigma_s = 2.0 * lame_coefficient_mu * E
                                             + lame_coefficient_lambda * trace(E) * Identity;

                for (unsigned int j = 0; j < n_dofs_per_element; j++)
                  {
                    local_matrix(j, i) += scale_ico
                                          * scalar_product(sigma_s, phi_i_grads_u[j])
                                          * state_fe_values.JxW(q);

                  }
              }
          }
      } // end material_id = 0
  }

  void
  ElementRightHandSide(const EDC<DH, VECTOR, dealdim> & /*edc*/,
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
  ElementTimeEquationExplicit(const EDC<DH, VECTOR, dealdim> &edc,
                              dealii::Vector<double> &local_vector, double scale)
  {
    assert(this->problem_type_ == "state");

    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();
    unsigned int material_id = edc.GetMaterialId();

    // old Newton step solution values and gradients
    uvalues_.resize(n_q_points, Vector<double>(3));
    ugrads_.resize(n_q_points, vector<Tensor<1, 2> >(3));

    edc.GetValuesState("last_newton_solution", uvalues_);
    edc.GetGradsState("last_newton_solution", ugrads_);

    // old timestep solution values and gradients
    last_timestep_uvalues_.resize(n_q_points, Vector<double>(3));
    last_timestep_ugrads_.resize(n_q_points, vector<Tensor<1, 2> >(3));

    edc.GetValuesState("last_time_solution", last_timestep_uvalues_);
    edc.GetGradsState("last_time_solution", last_timestep_ugrads_);

    const FEValuesExtractors::Vector displacements(0);
    const FEValuesExtractors::Scalar pressure(2);

    if (material_id == 1)
      {
        for (unsigned int q = 0; q < n_q_points; q++)
          {

            const double divergence_u = ugrads_[q][0][0] + ugrads_[q][1][1];
            const double old_timestep_divergence_u =
              last_timestep_ugrads_[q][0][0] + last_timestep_ugrads_[q][1][1];

            for (unsigned int i = 0; i < n_dofs_per_element; i++)
              {
                // Fluid, ElementTimeEquation, explicit
                const double phi_i_p = state_fe_values[pressure].value(i, q);

                local_vector(i) += scale
                                   * (c_biot * (uvalues_[q](2) - last_timestep_uvalues_[q](2))
                                      * phi_i_p
                                      + alpha_biot * (divergence_u - old_timestep_divergence_u)
                                      * phi_i_p) * state_fe_values.JxW(q);

              }
          }
      }
  }

  void
  ElementTimeMatrix(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                    FullMatrix<double> &/*local_matrix*/)
  {
    assert(this->problem_type_ == "state");
  }

  void
  ElementTimeMatrixExplicit(const EDC<DH, VECTOR, dealdim> &edc,
                            FullMatrix<double> &local_matrix)
  {
    assert(this->problem_type_ == "state");

    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();
    unsigned int material_id = edc.GetMaterialId();
    //double element_diameter = edc.GetElementDiameter();

    // old Newton step solution values and gradients
    uvalues_.resize(n_q_points, Vector<double>(3));
    ugrads_.resize(n_q_points, vector<Tensor<1, 2> >(3));

    edc.GetValuesState("last_newton_solution", uvalues_);
    edc.GetGradsState("last_newton_solution", ugrads_);

    // old timestep solution values and gradients
    last_timestep_uvalues_.resize(n_q_points, Vector<double>(3));
    last_timestep_ugrads_.resize(n_q_points, vector<Tensor<1, 2> >(3));

    edc.GetValuesState("last_time_solution", last_timestep_uvalues_);
    edc.GetGradsState("last_time_solution", last_timestep_ugrads_);

    const FEValuesExtractors::Vector displacements(0);
    const FEValuesExtractors::Scalar pressure(2);

    std::vector<double> phi_i_p(n_dofs_per_element);
    std::vector<Tensor<2, 2> > phi_i_grads_u(n_dofs_per_element);

    if (material_id == 1)
      {
        for (unsigned int q = 0; q < n_q_points; q++)
          {
            for (unsigned int k = 0; k < n_dofs_per_element; k++)
              {
                phi_i_p[k] = state_fe_values[pressure].value(k, q);
                phi_i_grads_u[k] = state_fe_values[displacements].gradient(k, q);
              }

            for (unsigned int i = 0; i < n_dofs_per_element; i++)
              {
                double divergence_u_LinU = phi_i_grads_u[i][0][0]
                                           + phi_i_grads_u[i][1][1];
                for (unsigned int j = 0; j < n_dofs_per_element; j++)
                  {
                    local_matrix(j, i) += (c_biot * phi_i_p[i] * phi_i_p[j]
                                           + alpha_biot * divergence_u_LinU * phi_i_p[j])
                                          * state_fe_values.JxW(q);

                  }
              }
          }
      }

  }

  // Values for boundary integrals
  void
  BoundaryEquation(const FDC<DH, VECTOR, dealdim> &fdc,
                   dealii::Vector<double> &local_vector, double /*scale*/,
                   double scale_ico)
  {

    assert(this->problem_type_ == "state");

    const auto &state_fe_face_values = fdc.GetFEFaceValuesState();
    unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();
    unsigned int n_q_points = fdc.GetNQPoints();
    unsigned int color = fdc.GetBoundaryIndicator();

    // Top boundary
    if (color == 3)
      {
        // old Newton step face_solution values and gradients
        ufacefvalues_.resize(n_q_points, Vector<double>(3));
        ufacegrads_.resize(n_q_points, vector<Tensor<1, 2> >(3));

        fdc.GetFaceValuesState("last_newton_solution", ufacefvalues_);
        fdc.GetFaceGradsState("last_newton_solution", ufacegrads_);

        const FEValuesExtractors::Vector displacements(0);

        for (unsigned int q = 0; q < n_q_points; q++)
          {
            Tensor<1, dealdim> neumann_value;
            neumann_value[0] = traction_x_biot;
            neumann_value[1] = traction_y_biot;

            for (unsigned int i = 0; i < n_dofs_per_element; i++)
              {
                const Tensor<1, 2> phi_i_u =
                  state_fe_face_values[displacements].value(i, q);

                local_vector(i) -= 1.0 * scale_ico * neumann_value * phi_i_u
                                   * state_fe_face_values.JxW(q);
              }
          }
      }

  }

  void
  BoundaryMatrix(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                 dealii::FullMatrix<double> & /*local_matrix*/, double /*scale*/,
                 double /*scale_ico*/)
  {
    assert(this->problem_type_ == "state");

  }

  void
  BoundaryRightHandSide(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                        dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
    assert(this->problem_type_ == "state");
  }

  // Values for boundary integrals
  void
  FaceEquation(const FDC<DH, VECTOR, dealdim> &fdc,
               dealii::Vector<double> &local_vector, double /*scale*/,
               double scale_ico)
  {

    assert(this->problem_type_ == "state");

    const auto &state_fe_face_values = fdc.GetFEFaceValuesState();
    unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();
    unsigned int n_q_points = fdc.GetNQPoints();
    unsigned int material_id = fdc.GetMaterialId();
    unsigned int material_id_neighbor = fdc.GetNbrMaterialId();
    bool at_boundary = fdc.GetIsAtBoundary();

    // Top boundary
    if (material_id == 1)
      if ((material_id != material_id_neighbor) && (!at_boundary))
        {
          // old Newton step face_solution values and gradients
          ufacefvalues_.resize(n_q_points, Vector<double>(3));
          ufacegrads_.resize(n_q_points, vector<Tensor<1, 2> >(3));

          fdc.GetFaceValuesState("last_newton_solution", ufacefvalues_);
          fdc.GetFaceGradsState("last_newton_solution", ufacegrads_);

          // old timestep solution values and gradients
          last_timestep_ufacefvalues_.resize(n_q_points, Vector<double>(3));
          last_timestep_ufacegrads_.resize(n_q_points,
                                           vector<Tensor<1, 2> >(3));

          fdc.GetFaceValuesState("last_time_solution",
                                 last_timestep_ufacefvalues_);
          fdc.GetFaceGradsState("last_time_solution",
                                last_timestep_ufacegrads_);

          const FEValuesExtractors::Vector displacements(0);

          for (unsigned int q = 0; q < n_q_points; q++)
            {
              double fluid_pressure = last_timestep_ufacefvalues_[q](2);

              for (unsigned int i = 0; i < n_dofs_per_element; i++)
                {
                  const Tensor<1, 2> phi_i_u =
                    state_fe_face_values[displacements].value(i, q);

                  local_vector(i) += 1.0 * scale_ico * alpha_biot
                                     * fluid_pressure * state_fe_face_values.normal_vector(q)
                                     * phi_i_u

                                     * state_fe_face_values.JxW(q);
                }
            }
        }

  }

  void
  FaceMatrix(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
             dealii::FullMatrix<double> &/*local_matrix*/, double /*scale*/,
             double /*scale_ico*/)
  {
    assert(this->problem_type_ == "state");

  }

  void
  FaceRightHandSide(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                    dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
    assert(this->problem_type_ == "state");
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
  GetControlNBlocks() const
  {
    return 1;
  }

  unsigned int
  GetStateNBlocks() const
  {
    return 2;
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

private:
  vector<Vector<double> > uvalues_;

  vector<vector<Tensor<1, dealdim> > > ugrads_;

  //last timestep solution values
  vector<Vector<double> > last_timestep_uvalues_;
  vector<vector<Tensor<1, dealdim> > > last_timestep_ugrads_;

  // face values
  vector<Vector<double> > ufacefvalues_;
  vector<vector<Tensor<1, dealdim> > > ufacegrads_;

  vector<Vector<double> > last_timestep_ufacefvalues_;
  vector<vector<Tensor<1, dealdim> > > last_timestep_ufacegrads_;

  vector<unsigned int> state_block_component_;
  vector<unsigned int> control_block_component_;

  // material variables
  double density_structure, lame_coefficient_mu, poisson_ratio_nu,
         lame_coefficient_lambda;

  double M_biot, c_biot, alpha_biot, viscosity_biot, K_biot, density_biot;

  double traction_x_biot, traction_y_biot;

  double volume_source;

};
#endif
