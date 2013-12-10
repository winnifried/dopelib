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

#ifndef _LOCALPDE_
#define _LOCALPDE_

#include "pdeinterface.h"

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
          _control_block_components(2, 0), _state_block_components(3, 0)
      {
        // control block components
        _control_block_components[0] = 0;
        _control_block_components[1] = 1;

        // state block components
        _state_block_components[2] = 1; // pressure

        param_reader.SetSubsection("Local PDE parameters");
        _density_fluid = param_reader.get_double("density_fluid");
        _viscosity = param_reader.get_double("viscosity");
      }

      void
      ElementEquation(const EDC<DH, VECTOR, dealdim>& edc,
          dealii::Vector<double> &local_vector, double scale,
          double /*scale_ico*/)
      {
        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            edc.GetFEValuesState();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();

        assert(this->_problem_type == "state");

        _uvalues.resize(n_q_points, Vector<double>(3));
        _ugrads.resize(n_q_points, vector<Tensor<1, 2> >(3));

        // Getting state values
        edc.GetValuesState("last_newton_solution", _uvalues);
        edc.GetGradsState("last_newton_solution", _ugrads);

        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(2);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          Tensor<2, 2> v_grads;
          v_grads.clear();
          v_grads[0][0] = _ugrads[q_point][0][0];
          v_grads[0][1] = _ugrads[q_point][0][1];
          v_grads[1][0] = _ugrads[q_point][1][0];
          v_grads[1][1] = _ugrads[q_point][1][1];

          Tensor<1, 2> v;
          v.clear();
          v[0] = _uvalues[q_point](0);
          v[1] = _uvalues[q_point](1);

          double v_incompressibility = v_grads[0][0] + v_grads[1][1];

          Tensor<1, 2> convection_fluid = v_grads * v;

          Tensor<2, dealdim> fluid_pressure;
          fluid_pressure.clear();
          fluid_pressure[0][0] = -_uvalues[q_point](2);
          fluid_pressure[1][1] = -_uvalues[q_point](2);

          for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            const Tensor<1, 2> phi_i_v = state_fe_values[velocities].value(i,
                q_point);
            const Tensor<2, 2> phi_i_grads_v =
                state_fe_values[velocities].gradient(i, q_point);
            const double phi_i_p = state_fe_values[pressure].value(i, q_point);

            local_vector(i) += scale
                * (scalar_product(fluid_pressure, phi_i_grads_v)
                    + _viscosity
                        * scalar_product(v_grads + transpose(v_grads),
                            phi_i_grads_v) + convection_fluid * phi_i_v
                    + v_incompressibility * phi_i_p)
                * state_fe_values.JxW(q_point);
          }
        }

      }

      void
      ElementMatrix(const EDC<DH, VECTOR, dealdim>& edc,
          dealii::FullMatrix<double> &local_matrix, double scale,
          double /*scale_ico*/)
      {
        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            edc.GetFEValuesState();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();

        _uvalues.resize(n_q_points, Vector<double>(3));
        _ugrads.resize(n_q_points, vector<Tensor<1, 2> >(3));

        // Getting previous Newton solutions via "last_newton_solution"
        // for the nonlinear convection term for ElementEquation
        // (PDE). In contrast the equations for
        // "adjoint", "tangent", etc. need the "state" values
        // for the linearized convection term.
        if (this->_problem_type == "state")
        {
          edc.GetValuesState("last_newton_solution", _uvalues);
          edc.GetGradsState("last_newton_solution", _ugrads);
        }
        else
        {
          edc.GetValuesState("state", _uvalues);
          edc.GetGradsState("state", _ugrads);
        }

        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(2);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          Tensor<2, 2> v_grads;
          v_grads.clear();
          v_grads[0][0] = _ugrads[q_point][0][0];
          v_grads[0][1] = _ugrads[q_point][0][1];
          v_grads[1][0] = _ugrads[q_point][1][0];
          v_grads[1][1] = _ugrads[q_point][1][1];

          Tensor<1, 2> v;
          v.clear();
          v[0] = _uvalues[q_point](0);
          v[1] = _uvalues[q_point](1);

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
                      + _viscosity
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
      ElementEquation_U(const EDC<DH, VECTOR, dealdim>& edc,
          dealii::Vector<double> &local_vector, double scale,
          double /*scale_ico*/)
      {
        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            edc.GetFEValuesState();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();

        assert(this->_problem_type == "adjoint");

        _zvalues.resize(n_q_points, Vector<double>(3));
        _zgrads.resize(n_q_points, vector<Tensor<1, 2> >(3));

        edc.GetValuesState("last_newton_solution", _zvalues);
        edc.GetGradsState("last_newton_solution", _zgrads);

        _z_state_values.resize(n_q_points, Vector<double>(3));
        _z_state_grads.resize(n_q_points, vector<Tensor<1, 2> >(3));

        edc.GetValuesState("state", _z_state_values);
        edc.GetGradsState("state", _z_state_grads);

        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(2);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          Tensor<2, 2> zv_grads;
          zv_grads.clear();
          zv_grads[0][0] = _zgrads[q_point][0][0];
          zv_grads[0][1] = _zgrads[q_point][0][1];
          zv_grads[1][0] = _zgrads[q_point][1][0];
          zv_grads[1][1] = _zgrads[q_point][1][1];

          Tensor<1, 2> zv;
          zv.clear();
          zv[0] = _zvalues[q_point](0);
          zv[1] = _zvalues[q_point](1);

          // state values which contains
          // solution from previous Newton step
          // Necessary for fluid convection term
          Tensor<2, 2> zv_state_grads;
          zv_state_grads.clear();
          zv_state_grads[0][0] = _z_state_grads[q_point][0][0];
          zv_state_grads[0][1] = _z_state_grads[q_point][0][1];
          zv_state_grads[1][0] = _z_state_grads[q_point][1][0];
          zv_state_grads[1][1] = _z_state_grads[q_point][1][1];

          Tensor<1, 2> zv_state;
          zv_state.clear();
          zv_state[0] = _z_state_values[q_point](0);
          zv_state[1] = _z_state_values[q_point](1);

          double zp = _zvalues[q_point](2);

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
                    + _viscosity
                        * scalar_product(
                            phi_i_grads_v + transpose(phi_i_grads_v), zv_grads)
                    + (phi_i_grads_v * zv_state + zv_state_grads * phi_i_v) * zv
                    + (phi_i_grads_v[0][0] + phi_i_grads_v[1][1]) * zp)
                * state_fe_values.JxW(q_point);

          }
        }
      }

      void
      ElementEquation_UT(const EDC<DH, VECTOR, dealdim>& edc,
          dealii::Vector<double> &local_vector, double scale,
          double /*scale_ico*/)
      {
        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            edc.GetFEValuesState();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();

        assert(this->_problem_type == "tangent");

        _duvalues.resize(n_q_points, Vector<double>(3));
        _dugrads.resize(n_q_points, vector<Tensor<1, 2> >(3));

        edc.GetValuesState("last_newton_solution", _duvalues);
        edc.GetGradsState("last_newton_solution", _dugrads);

        _du_state_values.resize(n_q_points, Vector<double>(3));
        _du_state_grads.resize(n_q_points, vector<Tensor<1, 2> >(3));

        edc.GetValuesState("state", _du_state_values);
        edc.GetGradsState("state", _du_state_grads);

        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(2);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          Tensor<2, dealdim> du_pI;
          du_pI.clear();
          du_pI[0][0] = -_duvalues[q_point](2);
          du_pI[1][1] = -_duvalues[q_point](2);

          Tensor<2, 2> duv_grads;
          duv_grads.clear();
          duv_grads[0][0] = _dugrads[q_point][0][0];
          duv_grads[0][1] = _dugrads[q_point][0][1];
          duv_grads[1][0] = _dugrads[q_point][1][0];
          duv_grads[1][1] = _dugrads[q_point][1][1];

          Tensor<1, 2> duv;
          duv.clear();
          duv[0] = _duvalues[q_point](0);
          duv[1] = _duvalues[q_point](1);

          double duv_incompressibility = duv_grads[0][0] + duv_grads[1][1];

          // state values which contains
          // solution from previous Newton step
          // Necessary for fluid convection term
          Tensor<2, 2> duv_state_grads;
          duv_state_grads.clear();
          duv_state_grads[0][0] = _du_state_grads[q_point][0][0];
          duv_state_grads[0][1] = _du_state_grads[q_point][0][1];
          duv_state_grads[1][0] = _du_state_grads[q_point][1][0];
          duv_state_grads[1][1] = _du_state_grads[q_point][1][1];

          Tensor<1, 2> duv_state;
          duv_state.clear();
          duv_state[0] = _du_state_values[q_point](0);
          duv_state[1] = _du_state_values[q_point](1);

          for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            const Tensor<1, 2> phi_i_v = state_fe_values[velocities].value(i,
                q_point);
            const Tensor<2, 2> phi_i_grads_v =
                state_fe_values[velocities].gradient(i, q_point);
            const double phi_i_p = state_fe_values[pressure].value(i, q_point);

            local_vector(i) += scale
                * (scalar_product(du_pI, phi_i_grads_v)
                    + _viscosity
                        * scalar_product(duv_grads + transpose(duv_grads),
                            phi_i_grads_v)
                    + (duv_grads * duv_state + duv_state_grads * duv) * phi_i_v
                    + duv_incompressibility * phi_i_p)
                * state_fe_values.JxW(q_point);
          }
        }

      }

      void
      ElementEquation_UTT(const EDC<DH, VECTOR, dealdim>& edc,
          dealii::Vector<double> &local_vector, double scale,
          double /*scale_ico*/)
      {
        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            edc.GetFEValuesState();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();

        assert(this->_problem_type == "adjoint_hessian");

        _dzvalues.resize(n_q_points, Vector<double>(3));
        _dzgrads.resize(n_q_points, vector<Tensor<1, 2> >(3));

        edc.GetValuesState("last_newton_solution", _dzvalues);
        edc.GetGradsState("last_newton_solution", _dzgrads);

        _dz_state_values.resize(n_q_points, Vector<double>(3));
        _dz_state_grads.resize(n_q_points, vector<Tensor<1, 2> >(3));

        edc.GetValuesState("state", _dz_state_values);
        edc.GetGradsState("state", _dz_state_grads);

        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(2);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          Tensor<2, 2> dzv_grads;
          dzv_grads.clear();
          dzv_grads[0][0] = _dzgrads[q_point][0][0];
          dzv_grads[0][1] = _dzgrads[q_point][0][1];
          dzv_grads[1][0] = _dzgrads[q_point][1][0];
          dzv_grads[1][1] = _dzgrads[q_point][1][1];

          Tensor<1, 2> dzv;
          dzv.clear();
          dzv[0] = _dzvalues[q_point](0);
          dzv[1] = _dzvalues[q_point](1);

          // state values which contains
          // solution from previous Newton step
          // Necessary for fluid convection term
          Tensor<2, 2> dzv_state_grads;
          dzv_state_grads.clear();
          dzv_state_grads[0][0] = _dz_state_grads[q_point][0][0];
          dzv_state_grads[0][1] = _dz_state_grads[q_point][0][1];
          dzv_state_grads[1][0] = _dz_state_grads[q_point][1][0];
          dzv_state_grads[1][1] = _dz_state_grads[q_point][1][1];

          Tensor<1, 2> dzv_state;
          dzv_state.clear();
          dzv_state[0] = _dz_state_values[q_point](0);
          dzv_state[1] = _dz_state_values[q_point](1);

          double dzp = _dzvalues[q_point](2);

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
                    + _viscosity
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
      ElementEquation_UU(const EDC<DH, VECTOR, dealdim>& edc,
          dealii::Vector<double> &local_vector, double scale,
          double /*scale_ico*/)
      {
        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            edc.GetFEValuesState();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();

        assert(this->_problem_type == "adjoint_hessian");

        _zvalues.resize(n_q_points, Vector<double>(3));
        _zgrads.resize(n_q_points, vector<Tensor<1, 2> >(3));

        edc.GetValuesState("adjoint", _zvalues);
        edc.GetGradsState("adjoint", _zgrads);

        _du_state_values.resize(n_q_points, Vector<double>(3));
        _du_state_grads.resize(n_q_points, vector<Tensor<1, 2> >(3));

        edc.GetValuesState("tangent", _du_state_values);
        edc.GetGradsState("tangent", _du_state_grads);

        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(2);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          Tensor<2, 2> zv_grads;
          zv_grads.clear();
          zv_grads[0][0] = _zgrads[q_point][0][0];
          zv_grads[0][1] = _zgrads[q_point][0][1];
          zv_grads[1][0] = _zgrads[q_point][1][0];
          zv_grads[1][1] = _zgrads[q_point][1][1];

          Tensor<1, 2> zv;
          zv.clear();
          zv[0] = _zvalues[q_point](0);
          zv[1] = _zvalues[q_point](1);

          // state values which contains
          // solution from previous Newton step
          // Necessary for fluid convection term
          Tensor<2, 2> duv_state_grads;
          duv_state_grads.clear();
          duv_state_grads[0][0] = _du_state_grads[q_point][0][0];
          duv_state_grads[0][1] = _du_state_grads[q_point][0][1];
          duv_state_grads[1][0] = _du_state_grads[q_point][1][0];
          duv_state_grads[1][1] = _du_state_grads[q_point][1][1];

          Tensor<1, 2> duv_state;
          duv_state.clear();
          duv_state[0] = _du_state_values[q_point](0);
          duv_state[1] = _du_state_values[q_point](1);

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
      ElementEquation_Q(const EDC<DH, VECTOR, dealdim>& /*edc*/,
          dealii::Vector<double> &/*local_vector*/, double /*scale*/,
          double /*scale_ico*/)
      {
        assert(this->_problem_type == "gradient");
      }

      void
      ElementEquation_QT(const EDC<DH, VECTOR, dealdim>& /*edc*/,
          dealii::Vector<double> &/*local_vector*/, double /*scale*/,
          double /*scale_ico*/)
      {
        assert(this->_problem_type == "tangent");
      }

      void
      ElementEquation_QTT(const EDC<DH, VECTOR, dealdim>& /*edc*/,
          dealii::Vector<double> &/*local_vector*/, double /*scale*/,
          double /*scale_ico*/)
      {
        assert(this->_problem_type == "hessian");
      }

      void
      ElementEquation_QU(const EDC<DH, VECTOR, dealdim>& /*edc*/,
          dealii::Vector<double> &/*local_vector*/, double /*scale*/,
          double /*scale_ico*/)
      {
        assert(this->_problem_type == "adjoint_hessian");
      }
      void
      ElementEquation_UQ(const EDC<DH, VECTOR, dealdim>& /*edc*/,
          dealii::Vector<double> &/*local_vector*/, double /*scale*/,
          double /*scale_ico*/)
      {
        assert(this->_problem_type == "hessian");
      }
      void
      ElementEquation_QQ(const EDC<DH, VECTOR, dealdim>& /*edc*/,
          dealii::Vector<double> &/*local_vector*/, double /*scale*/,
          double /*scale_ico*/)
      {
        assert(this->_problem_type == "hessian");
      }

      void
      ElementRightHandSide(const EDC<DH, VECTOR, dealdim>& /*edc*/,
          dealii::Vector<double> &/*local_vector*/, double /*scale*/)
      {
        assert(this->_problem_type == "state");
      }

      // Values for Boundary integrals
      void
      BoundaryEquation(const FDC<DH, VECTOR, dealdim>& fdc,
          dealii::Vector<double> &local_vector, double scale,
          double /*scale_ico*/)
      {
        const auto & state_fe_face_values = fdc.GetFEFaceValuesState();
        unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();
        unsigned int n_q_points = fdc.GetNQPoints();
        unsigned int color = fdc.GetBoundaryIndicator();
        assert(this->_problem_type == "state");

        // do-nothing condition applied at outflow boundary due symmetric part of
        // fluid's stress tensor
        if (color == 1)
        {
          _uboundarygrads.resize(n_q_points, vector<Tensor<1, 2> >(3));

          fdc.GetFaceGradsState("last_newton_solution", _uboundarygrads);

          const FEValuesExtractors::Vector velocities(0);
          const FEValuesExtractors::Scalar pressure(2);

          for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            Tensor<2, 2> v_grad;
            v_grad.clear();
            v_grad[0][0] = _uboundarygrads[q_point][0][0];
            v_grad[0][1] = _uboundarygrads[q_point][0][1];
            v_grad[1][0] = _uboundarygrads[q_point][1][0];
            v_grad[1][1] = _uboundarygrads[q_point][1][1];

            const Tensor<2, 2> do_nothing = _density_fluid * _viscosity
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
        _qvalues.reinit(2);
        fdc.GetParamValues("control", _qvalues);

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

              local_vector(i) -= scale * _qvalues(0)
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

              local_vector(i) -= scale * _qvalues(1)
                  * state_fe_face_values.normal_vector(q_point) * phi_i_v
                  * state_fe_face_values.JxW(q_point);
            }

          }
        }
      }

      void
      BoundaryMatrix(const FDC<DH, VECTOR, dealdim>& fdc,
          dealii::FullMatrix<double> &local_matrix, double scale,
          double /*scale_ico*/)
      {
        const auto & state_fe_face_values = fdc.GetFEFaceValuesState();
        unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();
        unsigned int n_q_points = fdc.GetNQPoints();
        unsigned int color = fdc.GetBoundaryIndicator();

        // do-nothing applied on outflow boundary
        if (color == 1)
        {
          _uboundarygrads.resize(n_q_points, vector<Tensor<1, 2> >(3));

          if (this->_problem_type == "state")
            fdc.GetFaceGradsState("last_newton_solution", _uboundarygrads);
          else
            fdc.GetFaceGradsState("state", _uboundarygrads);

          const FEValuesExtractors::Vector velocities(0);

          for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            Tensor<2, 2> v_grad;
            v_grad[0][0] = _uboundarygrads[q_point][0][0];
            v_grad[0][1] = _uboundarygrads[q_point][0][1];
            v_grad[1][0] = _uboundarygrads[q_point][1][0];
            v_grad[1][1] = _uboundarygrads[q_point][1][1];

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
                do_nothing_LinAll = _density_fluid * _viscosity
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
      BoundaryRightHandSide(const FDC<DH, VECTOR, dealdim>& /*fdc*/,
          dealii::Vector<double> &/*local_vector*/, double /*scale*/)
      {
        assert(this->_problem_type == "state");
      }

      void
      BoundaryEquation_Q(const FDC<DH, VECTOR, dealdim>& fdc,
          dealii::Vector<double> &local_vector, double scale,
          double /*scale_ico*/)
      {
        const auto & state_fe_face_values = fdc.GetFEFaceValuesState();
        unsigned int n_q_points = fdc.GetNQPoints();
        unsigned int color = fdc.GetBoundaryIndicator();

        assert(this->_problem_type == "gradient");

        _zboundaryvalues.resize(n_q_points, Vector<double>(3));

        fdc.GetFaceValuesState("adjoint", _zboundaryvalues);

        // control values for the upper and lower part
        if (color == 50)
        {
          for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            Tensor<1, 2> zvboundary;
            zvboundary.clear();
            zvboundary[0] = _zboundaryvalues[q_point](0);
            zvboundary[1] = _zboundaryvalues[q_point](1);

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
            zvboundary[0] = _zboundaryvalues[q_point](0);
            zvboundary[1] = _zboundaryvalues[q_point](1);

            local_vector(1) -= scale * 1.0
                * state_fe_face_values.normal_vector(q_point) * zvboundary
                * state_fe_face_values.JxW(q_point);

          }
        }
      }

      void
      BoundaryEquation_QT(const FDC<DH, VECTOR, dealdim>& fdc,
          dealii::Vector<double> &local_vector, double scale,
          double /*scale_ico*/)
      {
        const auto & state_fe_face_values = fdc.GetFEFaceValuesState();
        unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();
        unsigned int n_q_points = fdc.GetNQPoints();
        unsigned int color = fdc.GetBoundaryIndicator();
        assert(this->_problem_type == "tangent");

        _dqvalues.reinit(2);
        fdc.GetParamValues("dq", _dqvalues);

        if (color == 50)
        {
          const FEValuesExtractors::Vector velocities(0);

          for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            for (unsigned int i = 0; i < n_dofs_per_element; i++)
            {
              const Tensor<1, 2> phi_i_v =
                  state_fe_face_values[velocities].value(i, q_point);
              local_vector(i) -= 1.0 * scale * _dqvalues(0) * phi_i_v
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
              local_vector(i) -= 1.0 * scale * _dqvalues(1) * phi_i_v
                  * state_fe_face_values.normal_vector(q_point)
                  * state_fe_face_values.JxW(q_point);
            }

          }
        }
      }

      void
      BoundaryEquation_QTT(const FDC<DH, VECTOR, dealdim>& fdc,
          dealii::Vector<double> &local_vector, double scale,
          double /*scale_ico*/)
      {
        const auto & state_fe_face_values = fdc.GetFEFaceValuesState();
        unsigned int n_q_points = fdc.GetNQPoints();
        unsigned int color = fdc.GetBoundaryIndicator();

        assert(this->_problem_type == "hessian");

        _dzboundaryvalues.resize(n_q_points, Vector<double>(3));

        fdc.GetFaceValuesState("adjoint_hessian", _dzboundaryvalues);

        // control values for both parts
        if (color == 50)
        {
          for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            Tensor<1, 2> dzvboundary;
            dzvboundary.clear();
            dzvboundary[0] = _dzboundaryvalues[q_point](0);
            dzvboundary[1] = _dzboundaryvalues[q_point](1);

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
            dzvboundary[0] = _dzboundaryvalues[q_point](0);
            dzvboundary[1] = _dzboundaryvalues[q_point](1);

            local_vector(1) -= scale * 1.0
                * state_fe_face_values.normal_vector(q_point) * dzvboundary
                * state_fe_face_values.JxW(q_point);
          }
        }
      }

      // do-nothing condition at boundary /Gamma_1
      void
      BoundaryEquation_U(const FDC<DH, VECTOR, dealdim>& fdc,
          dealii::Vector<double> &local_vector, double scale,
          double /*scale_ico*/)
      {
        const auto & state_fe_face_values = fdc.GetFEFaceValuesState();
        unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();
        unsigned int n_q_points = fdc.GetNQPoints();
        unsigned int color = fdc.GetBoundaryIndicator();

        assert(this->_problem_type == "adjoint");

        // do-nothing applied on outflow boundary due symmetric part of
        // fluid's stress tensor
        if (color == 1)
        {
          _zboundaryvalues.resize(n_q_points, Vector<double>(3));

          fdc.GetFaceValuesState("last_newton_solution", _zboundaryvalues);

          const FEValuesExtractors::Vector velocities(0);

          for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            Tensor<1, 2> zvboundary;
            zvboundary.clear();
            zvboundary[0] = _zboundaryvalues[q_point](0);
            zvboundary[1] = _zboundaryvalues[q_point](1);

            for (unsigned int i = 0; i < n_dofs_per_element; i++)
            {
              const Tensor<2, 2> phi_i_grads_v =
                  state_fe_face_values[velocities].gradient(i, q_point);

              local_vector(i) -= scale * _density_fluid * _viscosity
                  * transpose(phi_i_grads_v)
                  * state_fe_face_values.normal_vector(q_point) * zvboundary
                  * state_fe_face_values.JxW(q_point);
            }
          }
        }
      }

      void
      BoundaryEquation_UT(const FDC<DH, VECTOR, dealdim>& fdc,
          dealii::Vector<double> &local_vector, double scale,
          double /*scale_ico*/)
      {
        const auto & state_fe_face_values = fdc.GetFEFaceValuesState();
        unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();
        unsigned int n_q_points = fdc.GetNQPoints();
        unsigned int color = fdc.GetBoundaryIndicator();

        assert(this->_problem_type == "tangent");

        // do-nothing applied on outflow boundary due symmetric part of
        // fluid's stress tensor
        if (color == 1)
        {
          _duboundarygrads.resize(n_q_points, vector<Tensor<1, 2> >(3));

          fdc.GetFaceGradsState("last_newton_solution", _duboundarygrads);

          const FEValuesExtractors::Vector velocities(0);

          for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            Tensor<2, 2> duv_grad;
            duv_grad[0][0] = _duboundarygrads[q_point][0][0];
            duv_grad[0][1] = _duboundarygrads[q_point][0][1];
            duv_grad[1][0] = _duboundarygrads[q_point][1][0];
            duv_grad[1][1] = _duboundarygrads[q_point][1][1];
            for (unsigned int i = 0; i < n_dofs_per_element; i++)
            {
              const Tensor<1, 2> phi_i_v =
                  state_fe_face_values[velocities].value(i, q_point);

              local_vector(i) -= scale * _density_fluid * _viscosity
                  * transpose(duv_grad)
                  * state_fe_face_values.normal_vector(q_point) * phi_i_v
                  * state_fe_face_values.JxW(q_point);
            }
          }
        }
      }

      void
      BoundaryEquation_UTT(const FDC<DH, VECTOR, dealdim>& fdc,
          dealii::Vector<double> &local_vector, double scale,
          double /*scale_ico*/)
      {
        const auto & state_fe_face_values = fdc.GetFEFaceValuesState();
        unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();
        unsigned int n_q_points = fdc.GetNQPoints();
        unsigned int color = fdc.GetBoundaryIndicator();

        assert(this->_problem_type == "adjoint_hessian");

        // do-nothing applied on outflow boundary due symmetric part of
        // fluid's stress tensor
        if (color == 1)
        {
          _dzboundaryvalues.resize(n_q_points, Vector<double>(3));

          fdc.GetFaceValuesState("last_newton_solution", _dzboundaryvalues);

          const FEValuesExtractors::Vector velocities(0);

          for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            Tensor<1, 2> dzvboundary;
            dzvboundary.clear();
            dzvboundary[0] = _dzboundaryvalues[q_point](0);
            dzvboundary[1] = _dzboundaryvalues[q_point](1);

            for (unsigned int i = 0; i < n_dofs_per_element; i++)
            {
              const Tensor<2, 2> phi_i_grads_v =
                  state_fe_face_values[velocities].gradient(i, q_point);

              local_vector(i) -= scale * _density_fluid * _viscosity
                  * transpose(phi_i_grads_v)
                  * state_fe_face_values.normal_vector(q_point) * dzvboundary
                  * state_fe_face_values.JxW(q_point);
            }
          }
        }
      }

      void
      BoundaryEquation_UU(const FDC<DH, VECTOR, dealdim>& /*fdc*/,
          dealii::Vector<double> &/*local_vector*/, double /*scale*/,
          double /*scale_ico*/)
      {
        assert(this->_problem_type == "adjoint_hessian");
      }

      void
      BoundaryEquation_QU(const FDC<DH, VECTOR, dealdim>& /*fdc*/,
          dealii::Vector<double> &/*local_vector*/, double /*scale*/,
          double /*scale_ico*/)
      {
        assert(this->_problem_type == "adjoint_hessian");
      }

      void
      BoundaryEquation_UQ(const FDC<DH, VECTOR, dealdim>& /*fdc*/,
          dealii::Vector<double> &/*local_vector*/, double /*scale*/,
          double /*scale_ico*/)
      {

      }

      void
      BoundaryEquation_QQ(const FDC<DH, VECTOR, dealdim>& /*fdc*/,
          dealii::Vector<double> &/*local_vector*/, double /*scale*/,
          double /*scale_ico*/)
      {

      }

      void
      ControlElementEquation(const EDC<DH, VECTOR, dealdim>& edc,
          dealii::Vector<double> &local_vector, double scale)
      {
        {
          assert(
              (this->_problem_type == "gradient")||(this->_problem_type == "hessian"));
          _funcgradvalues.reinit(local_vector.size());
          edc.GetParamValues("last_newton_solution", _funcgradvalues);
        }

        for (unsigned int i = 0; i < local_vector.size(); i++)
        {
          local_vector(i) += scale * _funcgradvalues(i);
        }
      }

      void
      ControlElementMatrix(const EDC<DH, VECTOR, dealdim>& /*edc*/,
          FullMatrix<double> &local_matrix)
      {
        assert(local_matrix.m() == local_matrix.n());
        for (unsigned int i = 0; i < local_matrix.m(); i++)
        {
          local_matrix(i, i) += 1.;
        }
      }

      UpdateFlags
      GetUpdateFlags() const
      {
        if ((this->_problem_type == "adjoint")
            || (this->_problem_type == "state")
            || (this->_problem_type == "tangent")
            || (this->_problem_type == "adjoint_hessian")
            || (this->_problem_type == "hessian"))
          return update_values | update_gradients | update_quadrature_points;
        else if ((this->_problem_type == "gradient"))
          return update_values | update_quadrature_points;
        else
          throw DOpEException("Unknown Problem Type " + this->_problem_type,
              "LocalPDE::GetUpdateFlags");
      }

      UpdateFlags
      GetFaceUpdateFlags() const
      {
        if ((this->_problem_type == "adjoint")
            || (this->_problem_type == "state")
            || (this->_problem_type == "tangent")
            || (this->_problem_type == "adjoint_hessian")
            || (this->_problem_type == "hessian"))
          return update_values | update_gradients | update_normal_vectors
              | update_quadrature_points;
        else if ((this->_problem_type == "gradient"))
          return update_values | update_quadrature_points
              | update_normal_vectors;
        else
          throw DOpEException("Unknown Problem Type " + this->_problem_type,
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

      std::vector<unsigned int>&
      GetControlBlockComponent()
      {
        return _control_block_components;
      }
      const std::vector<unsigned int>&
      GetControlBlockComponent() const
      {
        return _control_block_components;
      }
      std::vector<unsigned int>&
      GetStateBlockComponent()
      {
        return _state_block_components;
      }
      const std::vector<unsigned int>&
      GetStateBlockComponent() const
      {
        return _state_block_components;
      }

    protected:

    private:
      Vector<double> _qvalues;
      Vector<double> _dqvalues;

      Vector<double> _funcgradvalues;
      vector<Vector<double> > _fvalues;

      vector<Vector<double> > _uvalues;
      vector<vector<Tensor<1, dealdim> > > _ugrads;

      vector<Vector<double> > _zvalues;
      vector<vector<Tensor<1, dealdim> > > _zgrads;
      vector<Vector<double> > _z_state_values;
      vector<vector<Tensor<1, dealdim> > > _z_state_grads;

      vector<Vector<double> > _duvalues;
      vector<vector<Tensor<1, dealdim> > > _dugrads;
      vector<Vector<double> > _du_state_values;
      vector<vector<Tensor<1, dealdim> > > _du_state_grads;

      vector<Vector<double> > _dzvalues;
      vector<vector<Tensor<1, dealdim> > > _dzgrads;
      vector<Vector<double> > _dz_state_values;
      vector<vector<Tensor<1, dealdim> > > _dz_state_grads;

      // boundary values
      vector<Vector<double> > _qboundaryvalues;
      vector<Vector<double> > _fboundaryvalues;
      vector<Vector<double> > _uboundaryvalues;

      vector<Vector<double> > _zboundaryvalues;
      vector<Vector<double> > _dzboundaryvalues;

      vector<vector<Tensor<1, dealdim> > > _uboundarygrads;
      vector<vector<Tensor<1, dealdim> > > _duboundarygrads;

      vector<unsigned int> _control_block_components;
      vector<unsigned int> _state_block_components;

      double _diameter;

      double _density_fluid, _viscosity;
  };
#endif

