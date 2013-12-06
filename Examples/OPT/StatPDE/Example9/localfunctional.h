/**
 *
 * Copyright (C) 2012 by the DOpElib authors
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

#ifndef _LOCALFunctional_
#define _LOCALFunctional_

#include "pdeinterface.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

template<
    template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
    template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
    template<int, int> class DH, typename VECTOR, int dopedim, int dealdim =
        dopedim>
  class LocalFunctional : public FunctionalInterface<CDC, FDC, DH, VECTOR,
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
        _mu_regularization = param_reader.get_double("mu_regularization");
        _density_fluid = param_reader.get_double("density_fluid");
        _viscosity = param_reader.get_double("viscosity");
        _alpha_u = param_reader.get_double("alpha_u");

        _lame_coefficient_mu = param_reader.get_double("mu");
        _poisson_ratio_nu = param_reader.get_double("poisson_ratio_nu");
        _lame_coefficient_lambda =
            (2 * _poisson_ratio_nu * _lame_coefficient_mu)
                / (1.0 - 2 * _poisson_ratio_nu);

        _control_constant = param_reader.get_double("control_constant");
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
      BoundaryValue(const FDC<DH, VECTOR, dealdim>& fdc)
      {
        const auto & state_fe_face_values = fdc.GetFEFaceValuesState();
        unsigned int n_q_points = fdc.GetNQPoints();
        unsigned int color = fdc.GetBoundaryIndicator();
        double functional_value_J = 0;

        double drag_lift_value = 0.0;
        // Asking for boundary color of the cylinder
        if (color == 80)
        {
          _ufacevalues.resize(n_q_points, Vector<double>(5));
          _ufacegrads.resize(n_q_points, vector<Tensor<1, 2> >(5));

          fdc.GetFaceValuesState("state", _ufacevalues);
          fdc.GetFaceGradsState("state", _ufacegrads);

          for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            Tensor<2, 2> pI;
            pI[0][0] = _ufacevalues[q_point](4);
            pI[1][1] = _ufacevalues[q_point](4);

            Tensor<1, 2> v;
            v.clear();
            v[0] = _ufacevalues[q_point](0);
            v[1] = _ufacevalues[q_point](1);

            Tensor<2, 2> grad_v;
            grad_v[0][0] = _ufacegrads[q_point][0][0];
            grad_v[0][1] = _ufacegrads[q_point][0][1];
            grad_v[1][0] = _ufacegrads[q_point][1][0];
            grad_v[1][1] = _ufacegrads[q_point][1][1];

            Tensor<2, 2> F;
            F[0][0] = 1.0 + _ufacegrads[q_point][2][0];
            F[0][1] = _ufacegrads[q_point][2][1];
            F[1][0] = _ufacegrads[q_point][3][0];
            F[1][1] = 1.0 + _ufacegrads[q_point][3][1];

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
                        + _density_fluid * _viscosity
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
          _qvalues.reinit(2);
          fdc.GetParamValues("control", _qvalues);

          for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            functional_value_J += _mu_regularization * 0.5
                * ((_qvalues(0) - _control_constant)
                    * (_qvalues(0) - _control_constant))
                * state_fe_face_values.JxW(q_point);
          }

        }
        if (color == 51)
        {
          // Regularization
          _qvalues.reinit(2);
          fdc.GetParamValues("control", _qvalues);

          for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            functional_value_J += _mu_regularization * 0.5
                * ((_qvalues(0) - _control_constant)
                    * (_qvalues(0) - _control_constant))
                * state_fe_face_values.JxW(q_point);
          }

        }
        return functional_value_J;

      }

      void
      BoundaryValue_U(const FDC<DH, VECTOR, dealdim>& fdc,
          dealii::Vector<double> &local_cell_vector, double scale)
      {
        const auto & state_fe_face_values = fdc.GetFEFaceValuesState();
        unsigned int n_dofs_per_cell = fdc.GetNDoFsPerCell();
        unsigned int n_q_points = fdc.GetNQPoints();
        unsigned int color = fdc.GetBoundaryIndicator();

        if (color == 80)
        {
          _ufacevalues.resize(n_q_points, Vector<double>(5));
          _ufacegrads.resize(n_q_points, vector<Tensor<1, 2> >(5));

          fdc.GetFaceValuesState("state", _ufacevalues);
          fdc.GetFaceGradsState("state", _ufacegrads);

          const FEValuesExtractors::Vector velocities(0);
          const FEValuesExtractors::Vector displacements(2);
          const FEValuesExtractors::Scalar pressure(4);

          std::vector<Tensor<1, 2> > phi_v(n_dofs_per_cell);
          std::vector<Tensor<2, 2> > phi_grads_v(n_dofs_per_cell);
          std::vector<Tensor<1, 2> > phi_u(n_dofs_per_cell);
          std::vector<Tensor<2, 2> > phi_grads_u(n_dofs_per_cell);
          std::vector<double> phi_p(n_dofs_per_cell);

          for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            for (unsigned int k = 0; k < n_dofs_per_cell; k++)
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
                q_point, _ufacevalues);
            const Tensor<2, dealdim> grad_v = ALE_Transformations::get_grad_v<
                dealdim>(q_point, _ufacegrads);

            const Tensor<2, dealdim> grad_v_T =
                ALE_Transformations::get_grad_v_T<dealdim>(grad_v);
            const Tensor<2, dealdim> F = ALE_Transformations::get_F<dealdim>(
                q_point, _ufacegrads);
            const Tensor<2, dealdim> F_Inverse =
                ALE_Transformations::get_F_Inverse<dealdim>(F);

            const Tensor<2, dealdim> F_Inverse_T =
                ALE_Transformations::get_F_Inverse_T<dealdim>(F_Inverse);
            const double J = ALE_Transformations::get_J<dealdim>(F);

            const Tensor<2, dealdim> sigma_ALE =
                NSE_in_ALE::get_stress_fluid_ALE<dealdim>(_density_fluid,
                    _viscosity, pI, grad_v, grad_v_T, F_Inverse, F_Inverse_T);

            Tensor<1, 2> stress_normal;
            stress_normal = sigma_ALE
                * state_fe_face_values.normal_vector(q_point);

            for (unsigned int j = 0; j < n_dofs_per_cell; j++)
            {
              const Tensor<2, dealdim> pI_LinP =
                  ALE_Transformations::get_pI_LinP<dealdim>(phi_p[j]);

              const Tensor<2, dealdim> grad_v_LinV =
                  ALE_Transformations::get_grad_v_LinV<dealdim>(phi_grads_v[j]);

              const double J_LinU = ALE_Transformations::get_J_LinU<dealdim>(
                  q_point, _ufacegrads, phi_grads_u[j]);

              const Tensor<2, dealdim> J_F_Inverse_T_LinU =
                  ALE_Transformations::get_J_F_Inverse_T_LinU<dealdim>(
                      phi_grads_u[j]);

              const Tensor<2, dealdim> F_Inverse_LinU =
                  ALE_Transformations::get_F_Inverse_LinU(phi_grads_u[j], J,
                      J_LinU, q_point, _ufacegrads);

              const Tensor<2, dealdim> stress_fluid_ALE_1st_term_LinAll =
                  NSE_in_ALE::get_stress_fluid_ALE_1st_term_LinAll_short<dealdim>(
                      pI, F_Inverse_T, J_F_Inverse_T_LinU, pI_LinP, J);

              const Tensor<2, dealdim> stress_fluid_ALE_2nd_term_LinAll =
                  NSE_in_ALE::get_stress_fluid_ALE_2nd_term_LinAll_short(
                      J_F_Inverse_T_LinU, sigma_ALE, grad_v, grad_v_LinV,
                      F_Inverse, F_Inverse_LinU, J, _viscosity, _density_fluid);

              Tensor<1, 2> neumann_value = (stress_fluid_ALE_1st_term_LinAll
                  + stress_fluid_ALE_2nd_term_LinAll)
                  * state_fe_face_values.normal_vector(q_point);

              local_cell_vector(j) += scale * neumann_value[0]
                  * stress_normal[0] * state_fe_face_values.JxW(q_point);

            }
          }
        }
      }

      void
      BoundaryValue_Q(const FDC<DH, VECTOR, dealdim>& fdc,
          dealii::Vector<double> &local_cell_vector, double scale)
      {
        const auto & state_fe_face_values = fdc.GetFEFaceValuesState();
        unsigned int n_dofs_per_cell = local_cell_vector.size();
        unsigned int n_q_points = fdc.GetNQPoints();
        unsigned int color = fdc.GetBoundaryIndicator();

        if (color == 50)
        {
          // Regularization
          _qvalues.reinit(2);
          fdc.GetParamValues("control", _qvalues);

          for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            for (unsigned int j = 0; j < n_dofs_per_cell; j++)
            {
              local_cell_vector(j) += scale * _mu_regularization
                  * (_qvalues(j) - _control_constant)
                  * state_fe_face_values.JxW(q_point);
            }
          }
        }
        if (color == 51)
        {
          // Regularization
          _qvalues.reinit(2);
          fdc.GetParamValues("control", _qvalues);

          for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            for (unsigned int j = 0; j < n_dofs_per_cell; j++)
            {
              local_cell_vector(j) += scale * _mu_regularization
                  * (_qvalues(j) - _control_constant)
                  * state_fe_face_values.JxW(q_point);
            }
          }
        }
      }

      void
      BoundaryValue_QQ(const FDC<DH, VECTOR, dealdim>& fdc,
          dealii::Vector<double> &local_cell_vector, double scale)
      {
        const auto & state_fe_face_values = fdc.GetFEFaceValuesState();
        unsigned int n_dofs_per_cell = local_cell_vector.size();
        unsigned int n_q_points = fdc.GetNQPoints();
        unsigned int color = fdc.GetBoundaryIndicator();

        if (color == 50)
        {
          // Regularization
          _dqvalues.reinit(2);
          fdc.GetParamValues("dq", _dqvalues);

          _qvalues.reinit(2);
          fdc.GetParamValues("control", _qvalues);

          for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            for (unsigned int j = 0; j < n_dofs_per_cell; j++)
            {
              local_cell_vector(j) += scale * _mu_regularization
                  * (_dqvalues(j)) * state_fe_face_values.JxW(q_point);
            }
          }
        }
        if (color == 51)
        {
          // Regularization
          _dqvalues.reinit(2);
          fdc.GetParamValues("dq", _dqvalues);

          _qvalues.reinit(2);
          fdc.GetParamValues("control", _qvalues);

          for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            for (unsigned int j = 0; j < n_dofs_per_cell; j++)
            {
              local_cell_vector(j) += scale * _mu_regularization
                  * (_dqvalues(j)) * state_fe_face_values.JxW(q_point);
            }
          }
        }
      }

      void
      BoundaryValue_UU(const FDC<DH, VECTOR, dealdim>& /*fdc*/,
          dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/)
      {

      }

      void
      BoundaryValue_QU(const FDC<DH, VECTOR, dealdim>& /*fdc*/,
          dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/)
      {

      }

      void
      BoundaryValue_UQ(const FDC<DH, VECTOR, dealdim>& /*fdc*/,
          dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/)
      {

      }

// compute drag value around cylinder
      double
      FaceValue(const FDC<DH, VECTOR, dealdim>& fdc)
      {
        const auto & state_fe_face_values = fdc.GetFEFaceValuesState();
        unsigned int n_q_points = fdc.GetNQPoints();
        unsigned int material_id = fdc.GetMaterialId();
        unsigned int material_id_neighbor = fdc.GetNbrMaterialId();
        bool at_boundary = fdc.GetIsAtBoundary();

        double drag_lift_value = 0.0;
        if (material_id == 0)
        {
          if ((material_id != material_id_neighbor) && (!at_boundary))
          {
            _ufacevalues.resize(n_q_points, Vector<double>(5));
            _ufacegrads.resize(n_q_points, vector<Tensor<1, 2> >(5));

            fdc.GetFaceValuesState("state", _ufacevalues);
            fdc.GetFaceGradsState("state", _ufacegrads);

            for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
            {
              Tensor<2, 2> pI;
              pI[0][0] = _ufacevalues[q_point](4);
              pI[1][1] = _ufacevalues[q_point](4);

              Tensor<1, 2> v;
              v.clear();
              v[0] = _ufacevalues[q_point](0);
              v[1] = _ufacevalues[q_point](1);

              Tensor<2, 2> grad_v;
              grad_v[0][0] = _ufacegrads[q_point][0][0];
              grad_v[0][1] = _ufacegrads[q_point][0][1];
              grad_v[1][0] = _ufacegrads[q_point][1][0];
              grad_v[1][1] = _ufacegrads[q_point][1][1];

              Tensor<2, 2> F;
              F[0][0] = 1.0 + _ufacegrads[q_point][2][0];
              F[0][1] = _ufacegrads[q_point][2][1];
              F[1][0] = _ufacegrads[q_point][3][0];
              F[1][1] = 1.0 + _ufacegrads[q_point][3][1];

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
                      + _density_fluid * _viscosity
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
      FaceValue_U(const FDC<DH, VECTOR, dealdim>& fdc,
          dealii::Vector<double> &local_cell_vector, double scale)
      {
        const auto & state_fe_face_values = fdc.GetFEFaceValuesState();
        unsigned int n_dofs_per_cell = fdc.GetNDoFsPerCell();
        unsigned int n_q_points = fdc.GetNQPoints();
        unsigned int material_id = fdc.GetMaterialId();
        unsigned int material_id_neighbor = fdc.GetNbrMaterialId();
        bool at_boundary = fdc.GetIsAtBoundary();

        if (material_id == 0)
        {
          if ((material_id != material_id_neighbor) && (!at_boundary))
          {
            _ufacevalues.resize(n_q_points, Vector<double>(5));
            _ufacegrads.resize(n_q_points, vector<Tensor<1, 2> >(5));

            fdc.GetFaceValuesState("state", _ufacevalues);
            fdc.GetFaceGradsState("state", _ufacegrads);

            const FEValuesExtractors::Vector velocities(0);
            const FEValuesExtractors::Vector displacements(2);
            const FEValuesExtractors::Scalar pressure(4);

            std::vector<Tensor<1, 2> > phi_v(n_dofs_per_cell);
            std::vector<Tensor<2, 2> > phi_grads_v(n_dofs_per_cell);
            std::vector<Tensor<1, 2> > phi_u(n_dofs_per_cell);
            std::vector<Tensor<2, 2> > phi_grads_u(n_dofs_per_cell);
            std::vector<double> phi_p(n_dofs_per_cell);

            for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
            {
              for (unsigned int k = 0; k < n_dofs_per_cell; k++)
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
                  ALE_Transformations::get_pI<dealdim>(q_point, _ufacevalues);
              const Tensor<2, dealdim> grad_v = ALE_Transformations::get_grad_v<
                  dealdim>(q_point, _ufacegrads);

              const Tensor<2, dealdim> grad_v_T =
                  ALE_Transformations::get_grad_v_T<dealdim>(grad_v);
              const Tensor<2, dealdim> F = ALE_Transformations::get_F<dealdim>(
                  q_point, _ufacegrads);
              const Tensor<2, dealdim> F_Inverse =
                  ALE_Transformations::get_F_Inverse<dealdim>(F);

              const Tensor<2, dealdim> F_Inverse_T =
                  ALE_Transformations::get_F_Inverse_T<dealdim>(F_Inverse);
              const double J = ALE_Transformations::get_J<dealdim>(F);

              const Tensor<2, dealdim> sigma_ALE =
                  NSE_in_ALE::get_stress_fluid_ALE<dealdim>(_density_fluid,
                      _viscosity, pI, grad_v, grad_v_T, F_Inverse, F_Inverse_T);

              Tensor<1, 2> stress_normal;
              stress_normal = sigma_ALE
                  * state_fe_face_values.normal_vector(q_point);

              for (unsigned int j = 0; j < n_dofs_per_cell; j++)
              {
                const Tensor<2, dealdim> pI_LinP =
                    ALE_Transformations::get_pI_LinP<dealdim>(phi_p[j]);

                const Tensor<2, dealdim> grad_v_LinV =
                    ALE_Transformations::get_grad_v_LinV<dealdim>(
                        phi_grads_v[j]);

                const double J_LinU = ALE_Transformations::get_J_LinU<dealdim>(
                    q_point, _ufacegrads, phi_grads_u[j]);

                const Tensor<2, dealdim> J_F_Inverse_T_LinU =
                    ALE_Transformations::get_J_F_Inverse_T_LinU<dealdim>(
                        phi_grads_u[j]);

                const Tensor<2, dealdim> F_Inverse_LinU =
                    ALE_Transformations::get_F_Inverse_LinU(phi_grads_u[j], J,
                        J_LinU, q_point, _ufacegrads);

                const Tensor<2, dealdim> stress_fluid_ALE_1st_term_LinAll =
                    NSE_in_ALE::get_stress_fluid_ALE_1st_term_LinAll_short<
                        dealdim>(pI, F_Inverse_T, J_F_Inverse_T_LinU, pI_LinP,
                        J);
                const Tensor<2, dealdim> stress_fluid_ALE_2nd_term_LinAll =
                    NSE_in_ALE::get_stress_fluid_ALE_2nd_term_LinAll_short(
                        J_F_Inverse_T_LinU, sigma_ALE, grad_v, grad_v_LinV,
                        F_Inverse, F_Inverse_LinU, J, _viscosity,
                        _density_fluid);

                Tensor<1, 2> neumann_value = (stress_fluid_ALE_1st_term_LinAll
                    + stress_fluid_ALE_2nd_term_LinAll)
                    * state_fe_face_values.normal_vector(q_point);

                local_cell_vector(j) += scale * neumann_value[0]
                    * stress_normal[0] * state_fe_face_values.JxW(q_point);

              }
            }
          }
        }
      }

      void
      FaceValue_UU(const FDC<DH, VECTOR, dealdim>& /*fdc*/,
          dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/)
      {
//This derivative is not zero but it is not needed for newton convergence
      }

      void
      FaceValue_Q(const FDC<DH, VECTOR, dealdim>& /*fdc*/,
          dealii::Vector<double> & /*local_cell_vector*/, double /*scale*/)
      {

      }

      void
      FaceValue_QU(const FDC<DH, VECTOR, dealdim>& /*fdc*/,
          dealii::Vector<double> & /*local_cell_vector*/, double /*scale*/)
      {

      }

      void
      FaceValue_UQ(const FDC<DH, VECTOR, dealdim>& /*fdc*/,
          dealii::Vector<double> & /*local_cell_vector*/, double /*scale*/)
      {

      }

      void
      FaceValue_QQ(const FDC<DH, VECTOR, dealdim>& /*fdc*/,
          dealii::Vector<double> & /*local_cell_vector*/, double /*scale*/)
      {

      }

      double
      Value(const CDC<DH, VECTOR, dealdim>& /*cdc*/)
      {
        return 0.;
      }

      void
      Value_U(const CDC<DH, VECTOR, dealdim>& /*cdc*/,
          dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/)
      {

      }

      void
      Value_Q(const CDC<DH, VECTOR, dealdim>& /*cdc*/,
          dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/)
      {

      }

      void
      Value_UU(const CDC<DH, VECTOR, dealdim>& /*cdc*/,
          dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/)
      {

      }

      void
      Value_QU(const CDC<DH, VECTOR, dealdim>& /*cdc*/,
          dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/)
      {
      }

      void
      Value_UQ(const CDC<DH, VECTOR, dealdim>& /*cdc*/,
          dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/)
      {
      }

      void
      Value_QQ(const CDC<DH, VECTOR, dealdim>& /*cdc*/,
          dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/)
      {

      }

      UpdateFlags
      GetFaceUpdateFlags() const
      {
        return update_values | update_quadrature_points | update_gradients
            | update_normal_vectors;
      }

      inline void
      GetFaceValues(const DOpEWrapper::FEFaceValues<dealdim>& fe_face_values,
          const map<string, const BlockVector<double>*>& domain_values,
          string name, vector<Vector<double> >& values)
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
      GetFaceGrads(const DOpEWrapper::FEFaceValues<dealdim>& fe_face_values,
          const map<string, const BlockVector<double>*>& domain_values,
          string name, vector<vector<Tensor<1, dealdim> > >& values)
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
      GetValues(const DOpEWrapper::FEValues<dealdim>& fe_values,
          const map<string, const BlockVector<double>*>& domain_values,
          string name, vector<Vector<double> >& values)
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
      GetParams(const map<string, const Vector<double>*>& param_values,
          string name, Vector<double>& values)
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
      Vector<double> _qvalues;
      Vector<double> _dqvalues;
      vector<Vector<double> > _ufacevalues;
      vector<Vector<double> > _dufacevalues;

      vector<vector<Tensor<1, dealdim> > > _ufacegrads;
      vector<vector<Tensor<1, dealdim> > > _dufacegrads;

      // Artifcial parameter for FSI (later)
      double _alpha_u;

      // Fluid- and material parameters
      double _density_fluid, _viscosity, _lame_coefficient_mu,
          _poisson_ratio_nu, _lame_coefficient_lambda;

      // Control- and regularization parameters
      double _mu_regularization, _control_constant;

  };
#endif
