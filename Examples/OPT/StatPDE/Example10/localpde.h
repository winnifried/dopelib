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
    param_reader.declare_entry("density_fluid", "0.0", Patterns::Double(0));
    param_reader.declare_entry("viscosity", "0.0", Patterns::Double(0));
    param_reader.declare_entry("mu", "0.0", Patterns::Double(0));
  }

  LocalPDE(ParameterReader &param_reader) :
    control_block_components_(2, 0), state_block_component_(3, 0)
  {
    // control block components
    control_block_components_[0] = 0;
    control_block_components_[1] = 1;

    // state block components
    state_block_component_[2] = 1; // pressure

    param_reader.SetSubsection("Local PDE parameters");
    density_fluid_ = param_reader.get_double("density_fluid");
    viscosity_ = param_reader.get_double("viscosity");
  }

  void
  ElementEquation(const EDC<DH, VECTOR, dealdim> &edc,
                  dealii::Vector<double> &local_vector, double scale,
                  double /*scale_ico*/)
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();

    assert(this->problem_type_ == "state");

    uvalues_.resize(n_q_points, Vector<double>(3));
    ugrads_.resize(n_q_points, vector<Tensor<1, 2> >(3));

    // Getting state values
    edc.GetValuesState("last_newton_solution", uvalues_);
    edc.GetGradsState("last_newton_solution", ugrads_);

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(2);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        Tensor<2, 2> v_grads;
        v_grads.clear();
        v_grads[0][0] = ugrads_[q_point][0][0];
        v_grads[0][1] = ugrads_[q_point][0][1];
        v_grads[1][0] = ugrads_[q_point][1][0];
        v_grads[1][1] = ugrads_[q_point][1][1];

        Tensor<1, 2> v;
        v.clear();
        v[0] = uvalues_[q_point](0);
        v[1] = uvalues_[q_point](1);

        double v_incompressibility = v_grads[0][0] + v_grads[1][1];

        Tensor<1, 2> convection_fluid = v_grads * v;

        Tensor<2, dealdim> fluid_pressure;
        fluid_pressure.clear();
        fluid_pressure[0][0] = -uvalues_[q_point](2);
        fluid_pressure[1][1] = -uvalues_[q_point](2);

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            const Tensor<1, 2> phi_i_v = state_fe_values[velocities].value(i,
                                         q_point);
            const Tensor<2, 2> phi_i_grads_v =
              state_fe_values[velocities].gradient(i, q_point);
            const double phi_i_p = state_fe_values[pressure].value(i, q_point);

            local_vector(i) += scale
                               * (scalar_product(fluid_pressure, phi_i_grads_v)
                                  + viscosity_
                                  * scalar_product(v_grads + transpose(v_grads),
                                                   phi_i_grads_v) + convection_fluid * phi_i_v
                                  + v_incompressibility * phi_i_p)
                               * state_fe_values.JxW(q_point);
          }
      }

  }

  void
  ElementMatrix(const EDC<DH, VECTOR, dealdim> &edc,
                dealii::FullMatrix<double> &local_matrix, double scale,
                double /*scale_ico*/)
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();

    uvalues_.resize(n_q_points, Vector<double>(3));
    ugrads_.resize(n_q_points, vector<Tensor<1, 2> >(3));

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
    const FEValuesExtractors::Scalar pressure(2);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        Tensor<2, 2> v_grads;
        v_grads.clear();
        v_grads[0][0] = ugrads_[q_point][0][0];
        v_grads[0][1] = ugrads_[q_point][0][1];
        v_grads[1][0] = ugrads_[q_point][1][0];
        v_grads[1][1] = ugrads_[q_point][1][1];

        Tensor<1, 2> v;
        v.clear();
        v[0] = uvalues_[q_point](0);
        v[1] = uvalues_[q_point](1);

        for (unsigned int j = 0; j < n_dofs_per_element; j++)
          {
            const Tensor<1, 2> phi_j_v = state_fe_values[velocities].value(j,
                                         q_point);
            const Tensor<2, 2> phi_j_grads_v =
              state_fe_values[velocities].gradient(j, q_point);
            const double phi_j_p = state_fe_values[pressure].value(j, q_point);

            Tensor<2, dealdim> fluid_pressure_LinP;
            fluid_pressure_LinP.clear();
            fluid_pressure_LinP[0][0] = -phi_j_p;
            fluid_pressure_LinP[1][1] = -phi_j_p;

            for (unsigned int i = 0; i < n_dofs_per_element; i++)
              {
                const Tensor<1, 2> phi_i_v = state_fe_values[velocities].value(i,
                                             q_point);
                const Tensor<2, 2> phi_i_grads_v =
                  state_fe_values[velocities].gradient(i, q_point);
                const double phi_i_p = state_fe_values[pressure].value(i,
                                                                       q_point);

                local_matrix(i, j) += scale
                                      * (scalar_product(fluid_pressure_LinP, phi_i_grads_v)
                                         + viscosity_
                                         * scalar_product(
                                           phi_j_grads_v + transpose(phi_j_grads_v),
                                           phi_i_grads_v)
                                         + (phi_j_grads_v * v + v_grads * phi_j_v) * phi_i_v
                                         + (phi_j_grads_v[0][0] + phi_j_grads_v[1][1]) * phi_i_p)
                                      * state_fe_values.JxW(q_point);
              }
          }
      }
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

    assert(this->problem_type_ == "adjoint");

    zvalues_.resize(n_q_points, Vector<double>(3));
    zgrads_.resize(n_q_points, vector<Tensor<1, 2> >(3));

    edc.GetValuesState("last_newton_solution", zvalues_);
    edc.GetGradsState("last_newton_solution", zgrads_);

    z_state_values_.resize(n_q_points, Vector<double>(3));
    z_state_grads_.resize(n_q_points, vector<Tensor<1, 2> >(3));

    edc.GetValuesState("state", z_state_values_);
    edc.GetGradsState("state", z_state_grads_);

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(2);

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

        double zp = zvalues_[q_point](2);

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            const Tensor<1, 2> phi_i_v = state_fe_values[velocities].value(i,
                                         q_point);
            const Tensor<2, 2> phi_i_grads_v =
              state_fe_values[velocities].gradient(i, q_point);
            const double phi_i_p = state_fe_values[pressure].value(i, q_point);

            Tensor<2, dealdim> fluid_pressure_phi_i;
            fluid_pressure_phi_i.clear();
            fluid_pressure_phi_i[0][0] = -phi_i_p;
            fluid_pressure_phi_i[1][1] = -phi_i_p;

            local_vector(i) += scale
                               * (scalar_product(fluid_pressure_phi_i, zv_grads)
                                  + viscosity_
                                  * scalar_product(
                                    phi_i_grads_v + transpose(phi_i_grads_v), zv_grads)
                                  + (phi_i_grads_v * zv_state + zv_state_grads * phi_i_v) * zv
                                  + (phi_i_grads_v[0][0] + phi_i_grads_v[1][1]) * zp)
                               * state_fe_values.JxW(q_point);

          }
      }
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

    assert(this->problem_type_ == "tangent");

    duvalues_.resize(n_q_points, Vector<double>(3));
    dugrads_.resize(n_q_points, vector<Tensor<1, 2> >(3));

    edc.GetValuesState("last_newton_solution", duvalues_);
    edc.GetGradsState("last_newton_solution", dugrads_);

    du_state_values_.resize(n_q_points, Vector<double>(3));
    du_state_grads_.resize(n_q_points, vector<Tensor<1, 2> >(3));

    edc.GetValuesState("state", du_state_values_);
    edc.GetGradsState("state", du_state_grads_);

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(2);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        Tensor<2, dealdim> du_pI;
        du_pI.clear();
        du_pI[0][0] = -duvalues_[q_point](2);
        du_pI[1][1] = -duvalues_[q_point](2);

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

        double duv_incompressibility = duv_grads[0][0] + duv_grads[1][1];

        // state values which contains
        // solution from previous Newton step
        // Necessary for fluid convection term
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

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            const Tensor<1, 2> phi_i_v = state_fe_values[velocities].value(i,
                                         q_point);
            const Tensor<2, 2> phi_i_grads_v =
              state_fe_values[velocities].gradient(i, q_point);
            const double phi_i_p = state_fe_values[pressure].value(i, q_point);

            local_vector(i) += scale
                               * (scalar_product(du_pI, phi_i_grads_v)
                                  + viscosity_
                                  * scalar_product(duv_grads + transpose(duv_grads),
                                                   phi_i_grads_v)
                                  + (duv_grads * duv_state + duv_state_grads * duv) * phi_i_v
                                  + duv_incompressibility * phi_i_p)
                               * state_fe_values.JxW(q_point);
          }
      }

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

    assert(this->problem_type_ == "adjoint_hessian");

    dzvalues_.resize(n_q_points, Vector<double>(3));
    dzgrads_.resize(n_q_points, vector<Tensor<1, 2> >(3));

    edc.GetValuesState("last_newton_solution", dzvalues_);
    edc.GetGradsState("last_newton_solution", dzgrads_);

    dz_state_values_.resize(n_q_points, Vector<double>(3));
    dz_state_grads_.resize(n_q_points, vector<Tensor<1, 2> >(3));

    edc.GetValuesState("state", dz_state_values_);
    edc.GetGradsState("state", dz_state_grads_);

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(2);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
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

        double dzp = dzvalues_[q_point](2);

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            const Tensor<1, 2> phi_i_v = state_fe_values[velocities].value(i,
                                         q_point);
            const Tensor<2, 2> phi_i_grads_v =
              state_fe_values[velocities].gradient(i, q_point);
            const double phi_i_p = state_fe_values[pressure].value(i, q_point);

            Tensor<2, dealdim> fluid_pressure_phi_i;
            fluid_pressure_phi_i.clear();
            fluid_pressure_phi_i[0][0] = -phi_i_p;
            fluid_pressure_phi_i[1][1] = -phi_i_p;

            local_vector(i) += scale
                               * (scalar_product(fluid_pressure_phi_i, dzv_grads)
                                  + viscosity_
                                  * scalar_product(
                                    phi_i_grads_v + transpose(phi_i_grads_v), dzv_grads)
                                  + (phi_i_grads_v * dzv_state + dzv_state_grads * phi_i_v)
                                  * dzv
                                  + (phi_i_grads_v[0][0] + phi_i_grads_v[1][1]) * dzp)
                               * state_fe_values.JxW(q_point);
          }
      }
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

    assert(this->problem_type_ == "adjoint_hessian");

    zvalues_.resize(n_q_points, Vector<double>(3));
    zgrads_.resize(n_q_points, vector<Tensor<1, 2> >(3));

    edc.GetValuesState("adjoint", zvalues_);
    edc.GetGradsState("adjoint", zgrads_);

    du_state_values_.resize(n_q_points, Vector<double>(3));
    du_state_grads_.resize(n_q_points, vector<Tensor<1, 2> >(3));

    edc.GetValuesState("tangent", du_state_values_);
    edc.GetGradsState("tangent", du_state_grads_);

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(2);

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

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            const Tensor<1, 2> phi_i_v = state_fe_values[velocities].value(i,
                                         q_point);
            const Tensor<2, 2> phi_i_grads_v =
              state_fe_values[velocities].gradient(i, q_point);

            local_vector(i) += scale
                               * ((phi_i_grads_v * duv_state + duv_state_grads * phi_i_v) * zv)
                               * state_fe_values.JxW(q_point);
          }
      }

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
  BoundaryEquation(const FDC<DH, VECTOR, dealdim> &fdc,
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
        uboundarygrads_.resize(n_q_points, vector<Tensor<1, 2> >(3));

        fdc.GetFaceGradsState("last_newton_solution", uboundarygrads_);

        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(2);

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

                local_vector(i) -= scale * neumann_value * phi_i_v
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
  BoundaryMatrix(const FDC<DH, VECTOR, dealdim> &fdc,
                 dealii::FullMatrix<double> &local_matrix, double scale,
                 double /*scale_ico*/)
  {
    const auto &state_fe_face_values = fdc.GetFEFaceValuesState();
    unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();
    unsigned int n_q_points = fdc.GetNQPoints();
    unsigned int color = fdc.GetBoundaryIndicator();

    // do-nothing applied on outflow boundary
    if (color == 1)
      {
        uboundarygrads_.resize(n_q_points, vector<Tensor<1, 2> >(3));

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

                    // do-nothing
                    Tensor<2, 2> do_nothing_LinAll;
                    do_nothing_LinAll = density_fluid_ * viscosity_
                                        * transpose(phi_j_grads_v);

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
  BoundaryRightHandSide(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                        dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
    assert(this->problem_type_ == "state");
  }

  void
  BoundaryEquation_Q(const FDC<DH, VECTOR, dealdim> &fdc,
                     dealii::Vector<double> &local_vector, double scale,
                     double /*scale_ico*/)
  {
    const auto &state_fe_face_values = fdc.GetFEFaceValuesState();
    unsigned int n_q_points = fdc.GetNQPoints();
    unsigned int color = fdc.GetBoundaryIndicator();

    assert(this->problem_type_ == "gradient");

    zboundaryvalues_.resize(n_q_points, Vector<double>(3));

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
  BoundaryEquation_QT(const FDC<DH, VECTOR, dealdim> &fdc,
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
  BoundaryEquation_QTT(const FDC<DH, VECTOR, dealdim> &fdc,
                       dealii::Vector<double> &local_vector, double scale,
                       double /*scale_ico*/)
  {
    const auto &state_fe_face_values = fdc.GetFEFaceValuesState();
    unsigned int n_q_points = fdc.GetNQPoints();
    unsigned int color = fdc.GetBoundaryIndicator();

    assert(this->problem_type_ == "hessian");

    dzboundaryvalues_.resize(n_q_points, Vector<double>(3));

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
  BoundaryEquation_U(const FDC<DH, VECTOR, dealdim> &fdc,
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
        zboundaryvalues_.resize(n_q_points, Vector<double>(3));

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

                local_vector(i) -= scale * density_fluid_ * viscosity_
                                   * transpose(phi_i_grads_v)
                                   * state_fe_face_values.normal_vector(q_point) * zvboundary
                                   * state_fe_face_values.JxW(q_point);
              }
          }
      }
  }

  void
  BoundaryEquation_UT(const FDC<DH, VECTOR, dealdim> &fdc,
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
        duboundarygrads_.resize(n_q_points, vector<Tensor<1, 2> >(3));

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

                local_vector(i) -= scale * density_fluid_ * viscosity_
                                   * transpose(duv_grad)
                                   * state_fe_face_values.normal_vector(q_point) * phi_i_v
                                   * state_fe_face_values.JxW(q_point);
              }
          }
      }
  }

  void
  BoundaryEquation_UTT(const FDC<DH, VECTOR, dealdim> &fdc,
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
        dzboundaryvalues_.resize(n_q_points, Vector<double>(3));

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

                local_vector(i) -= scale * density_fluid_ * viscosity_
                                   * transpose(phi_i_grads_v)
                                   * state_fe_face_values.normal_vector(q_point) * dzvboundary
                                   * state_fe_face_values.JxW(q_point);
              }
          }
      }
  }

  void
  BoundaryEquation_UU(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                      dealii::Vector<double> &/*local_vector*/, double /*scale*/,
                      double /*scale_ico*/)
  {
    assert(this->problem_type_ == "adjoint_hessian");
  }

  void
  BoundaryEquation_QU(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                      dealii::Vector<double> &/*local_vector*/, double /*scale*/,
                      double /*scale_ico*/)
  {
    assert(this->problem_type_ == "adjoint_hessian");
  }

  void
  BoundaryEquation_UQ(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                      dealii::Vector<double> &/*local_vector*/, double /*scale*/,
                      double /*scale_ico*/)
  {

  }

  void
  BoundaryEquation_QQ(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                      dealii::Vector<double> &/*local_vector*/, double /*scale*/,
                      double /*scale_ico*/)
  {

  }

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
    return 2;
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
    return state_block_component_;
  }
  const std::vector<unsigned int> &
  GetStateBlockComponent() const
  {
    return state_block_component_;
  }

protected:

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
  vector<unsigned int> state_block_component_;

  double diameter_;

  double density_fluid_, viscosity_;
};
#endif

