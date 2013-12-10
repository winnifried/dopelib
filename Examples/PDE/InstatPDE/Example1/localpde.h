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
	param_reader.declare_entry("density_fluid", "0.0", Patterns::Double(0));
        param_reader.declare_entry("viscosity", "0.0", Patterns::Double(0));
      }

      LocalPDE(ParameterReader &param_reader) :
          _state_block_components(3, 0)
      {
        _state_block_components[2] = 1;

        param_reader.SetSubsection("Local PDE parameters");
	_density_fluid = param_reader.get_double("density_fluid");
        _viscosity = param_reader.get_double("viscosity");
      }

      void
      ElementEquation(const EDC<DH, VECTOR, dealdim>& edc,
          dealii::Vector<double> &local_vector, double scale,
          double scale_ico)
      {
        assert(this->_problem_type == "state");
        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            edc.GetFEValuesState();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();
        //unsigned int material_id = edc.GetMaterialId();

        _uvalues.resize(n_q_points, Vector<double>(3));
        _ugrads.resize(n_q_points, vector<Tensor<1, 2> >(3));

        edc.GetValuesState("last_newton_solution", _uvalues);
        edc.GetGradsState("last_newton_solution", _ugrads);

        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(2);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          Tensor<2, dealdim> fluid_pressure;
          fluid_pressure.clear();
          fluid_pressure[0][0] = -_uvalues[q_point](2);
          fluid_pressure[1][1] = -_uvalues[q_point](2);

          double incompressibility = _ugrads[q_point][0][0]
              + _ugrads[q_point][1][1];

          Tensor<2, 2> vgrads;
          vgrads.clear();
          vgrads[0][0] = _ugrads[q_point][0][0];
          vgrads[0][1] = _ugrads[q_point][0][1];
          vgrads[1][0] = _ugrads[q_point][1][0];
          vgrads[1][1] = _ugrads[q_point][1][1];

          Tensor<1, 2> v;
          v.clear();
          v[0] = _uvalues[q_point](0);
          v[1] = _uvalues[q_point](1);

          Tensor<1, 2> convection_fluid = vgrads * v;

          for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            const Tensor<1, 2> phi_i_v = state_fe_values[velocities].value(i,
                q_point);
            const Tensor<2, 2> phi_i_grads_v =
                state_fe_values[velocities].gradient(i, q_point);
            const double phi_i_p = state_fe_values[pressure].value(i, q_point);

            local_vector(i) += scale
                * _density_fluid * (convection_fluid * phi_i_v
				    + _viscosity
				    * scalar_product(vgrads + transpose(vgrads),
						     phi_i_grads_v)) * state_fe_values.JxW(q_point);

            local_vector(i) += scale_ico
                * (scalar_product(fluid_pressure, phi_i_grads_v)
                    + incompressibility * phi_i_p)
                * state_fe_values.JxW(q_point);

          }
        }

      }

      void
      ElementMatrix(const EDC<DH, VECTOR, dealdim>& edc,
          FullMatrix<double> &local_matrix, double scale,
          double scale_ico)
      {
        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            edc.GetFEValuesState();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();
        //unsigned int material_id = edc.GetMaterialId();

        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(2);

        _uvalues.resize(n_q_points, Vector<double>(3));
        _ugrads.resize(n_q_points, vector<Tensor<1, 2> >(3));

        edc.GetValuesState("last_newton_solution", _uvalues);
        edc.GetGradsState("last_newton_solution", _ugrads);

        std::vector<Tensor<1, 2> > phi_v(n_dofs_per_element);
        std::vector<Tensor<2, 2> > phi_grads_v(n_dofs_per_element);
        std::vector<double> phi_p(n_dofs_per_element);
        std::vector<double> div_phi_v(n_dofs_per_element);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          for (unsigned int k = 0; k < n_dofs_per_element; k++)
          {
            phi_v[k] = state_fe_values[velocities].value(k, q_point);
            phi_grads_v[k] = state_fe_values[velocities].gradient(k, q_point);
            phi_p[k] = state_fe_values[pressure].value(k, q_point);
            div_phi_v[k] = state_fe_values[velocities].divergence(k, q_point);
          }

          Tensor<2, 2> vgrads;
          vgrads.clear();
          vgrads[0][0] = _ugrads[q_point][0][0];
          vgrads[0][1] = _ugrads[q_point][0][1];
          vgrads[1][0] = _ugrads[q_point][1][0];
          vgrads[1][1] = _ugrads[q_point][1][1];

          Tensor<1, 2> v;
          v[0] = _uvalues[q_point](0);
          v[1] = _uvalues[q_point](1);

          for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            Tensor<2, dealdim> fluid_pressure_LinP;
            fluid_pressure_LinP.clear();
            fluid_pressure_LinP[0][0] = -phi_p[i];
            fluid_pressure_LinP[1][1] = -phi_p[i];

            Tensor<1, 2> convection_fluid_LinV = phi_grads_v[i] * v
                + vgrads * phi_v[i];

            for (unsigned int j = 0; j < n_dofs_per_element; j++)
            {
              local_matrix(j, i) += scale
                  * _density_fluid * (convection_fluid_LinV * phi_v[j]
                      + _viscosity
                          * scalar_product(
                              phi_grads_v[i] + transpose(phi_grads_v[i]),
                              phi_grads_v[j])) * state_fe_values.JxW(q_point);

              local_matrix(j, i) +=
                  scale_ico
                      * (scalar_product(fluid_pressure_LinP, phi_grads_v[j])
                          + (phi_grads_v[i][0][0] + phi_grads_v[i][1][1])
                              * phi_p[j]) * state_fe_values.JxW(q_point);
            }
          }
        }

      }

      void
      ElementRightHandSide(const EDC<DH, VECTOR, dealdim>& /*edc*/,
			dealii::Vector<double> & /*local_vector*/,
			double /*scale*/)
      {
        assert(this->_problem_type == "state");
      }

      void
      ElementTimeEquationExplicit(const EDC<DH, VECTOR, dealdim>& /*edc*/,
			       dealii::Vector<double> & /*local_vector*/,
			       double /*scale*/)
      {
        assert(this->_problem_type == "state");
      }

      void
      ElementTimeEquation(const EDC<DH, VECTOR, dealdim>& edc,
		       dealii::Vector<double> & local_vector,
		       double scale)
      {
        assert(this->_problem_type == "state");

        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            edc.GetFEValuesState();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();

        _uvalues.resize(n_q_points, Vector<double>(3));

        edc.GetValuesState("last_newton_solution", _uvalues);

        const FEValuesExtractors::Vector velocities(0);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          Tensor<1, 2> v;
          v[0] = _uvalues[q_point](0);
          v[1] = _uvalues[q_point](1);

          for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            const Tensor<1, 2> phi_i_v = state_fe_values[velocities].value(i,
                q_point);

            local_vector(i) += scale * _density_fluid * (v * phi_i_v)
                * state_fe_values.JxW(q_point);
          }
        }

      }

      void
	ElementTimeMatrixExplicit(const EDC<DH, VECTOR, dealdim>& /*edc*/,
			       FullMatrix<double> &/*local_matrix*/)
      {
        assert(this->_problem_type == "state");
      }

      void
      ElementTimeMatrix(const EDC<DH, VECTOR, dealdim>& edc,
          FullMatrix<double> &local_matrix)
      {
        assert(this->_problem_type == "state");

        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            edc.GetFEValuesState();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();

        const FEValuesExtractors::Vector velocities(0);

        std::vector<Tensor<1, 2> > phi_v(n_dofs_per_element);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          for (unsigned int k = 0; k < n_dofs_per_element; k++)
          {
            phi_v[k] = state_fe_values[velocities].value(k, q_point);
          }

          for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            for (unsigned int j = 0; j < n_dofs_per_element; j++)
            {
              local_matrix(j, i) += _density_fluid * (phi_v[i] * phi_v[j])
                  * state_fe_values.JxW(q_point);
            }
          }
        }

      }

      // Values for boundary integrals
      void
      BoundaryEquation(const FDC<DH, VECTOR, dealdim>& fdc,
          dealii::Vector<double> &local_vector, double scale,
          double /*scale_ico*/)
      {

        assert(this->_problem_type == "state");

        const auto & state_fe_face_values = fdc.GetFEFaceValuesState();
        unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();
        unsigned int n_q_points = fdc.GetNQPoints();
        unsigned int color = fdc.GetBoundaryIndicator();

        // do-nothing applied on outflow boundary
        if (color == 1)
        {
          _ufacegrads.resize(n_q_points, vector<Tensor<1, 2> >(3));

          fdc.GetFaceGradsState("last_newton_solution", _ufacegrads);

          const FEValuesExtractors::Vector velocities(0);

          for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            Tensor<2, 2> vgrads;
            vgrads[0][0] = _ufacegrads[q_point][0][0];
            vgrads[0][1] = _ufacegrads[q_point][0][1];
            vgrads[1][0] = _ufacegrads[q_point][1][0];
            vgrads[1][1] = _ufacegrads[q_point][1][1];

            for (unsigned int i = 0; i < n_dofs_per_element; i++)
            {
              const Tensor<1, 2> phi_i_v =
                  state_fe_face_values[velocities].value(i, q_point);

              const Tensor<1, 2> neumann_value = _viscosity * _density_fluid 
                  * (transpose(vgrads)
                      * state_fe_face_values.normal_vector(q_point));

              local_vector(i) -= scale * neumann_value * phi_i_v
                  * state_fe_face_values.JxW(q_point);
            }
          }
        }

      }

      void
      BoundaryMatrix(const FDC<DH, VECTOR, dealdim>& fdc,
          dealii::FullMatrix<double> &local_matrix, double /*scale_ico*/,
          double /*scale_ico*/)
      {
        assert(this->_problem_type == "state");

        const auto & state_fe_face_values = fdc.GetFEFaceValuesState();
        unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();
        unsigned int n_q_points = fdc.GetNQPoints();
        unsigned int color = fdc.GetBoundaryIndicator();

        // do-nothing applied on outflow boundary
        if (color == 1)
        {
          const FEValuesExtractors::Vector velocities(0);

          for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            for (unsigned int i = 0; i < n_dofs_per_element; i++)
            {
              const Tensor<2, 2> phi_j_grads_v =
                  state_fe_face_values[velocities].gradient(i, q_point);
              const Tensor<1, 2> neumann_value = _viscosity * _density_fluid 
                  * (transpose(phi_j_grads_v)
                      * state_fe_face_values.normal_vector(q_point));

              for (unsigned int j = 0; j < n_dofs_per_element; j++)
              {
                const Tensor<1, 2> phi_i_v =
                    state_fe_face_values[velocities].value(j, q_point);

                local_matrix(j, i) -= neumann_value * phi_i_v
                    * state_fe_face_values.JxW(q_point);
              }
            }
          }
        }
      }

      void
	BoundaryRightHandSide(const FDC<DH, VECTOR, dealdim>& /*fdc*/,
			      dealii::Vector<double> & /*local_vector*/,
			      double /*scale*/)
      {
        assert(this->_problem_type == "state");
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
        return 2;
      }

      std::vector<unsigned int>&
      GetControlBlockComponent()
      {
        return _block_components;
      }
      const std::vector<unsigned int>&
      GetControlBlockComponent() const
      {
        return _block_components;
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

    private:
      vector<Vector<double> > _uvalues;
      vector<vector<Tensor<1, dealdim> > > _ugrads;

      // face values
      vector<vector<Tensor<1, dealdim> > > _ufacegrads;

      vector<unsigned int> _state_block_components;
      vector<unsigned int> _block_components;

      double _density_fluid, _viscosity;

  };
#endif
