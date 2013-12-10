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
#include "my_functions.h"

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
      LocalPDE(unsigned int order) :
          _exact_solution(order), _state_block_components(1, 0)
      {
      }

      void
      ElementEquation(const EDC<DH, VECTOR, dealdim>& edc,
          dealii::Vector<double> &local_vector, double scale, double)
      {
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();
        const DOpEWrapper::FEValues<dealdim> &state_fe_values =
            edc.GetFEValuesState();

        assert(this->_problem_type == "state");

        _ugrads.resize(n_q_points, Tensor<1, dealdim>());
        edc.GetGradsState("last_newton_solution", _ugrads);

        const FEValuesExtractors::Scalar velocities(0);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          Tensor<1, 2> vgrads;
          vgrads.clear();
          vgrads[0] = _ugrads[q_point][0];
          vgrads[1] = _ugrads[q_point][1];

          for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            const Tensor<1, 2> phi_i_grads_v =
                state_fe_values[velocities].gradient(i, q_point);

            local_vector(i) += scale * (vgrads * phi_i_grads_v)
                * state_fe_values.JxW(q_point);
          }
        }
      }

      void
      ElementMatrix(const EDC<DH, VECTOR, dealdim>& edc,
          FullMatrix<double> &local_matrix, double scale, double)
      {
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();
        //unsigned int material_id = edc.GetMaterialId();
        const DOpEWrapper::FEValues<dealdim> &state_fe_values =
            edc.GetFEValuesState();

        const FEValuesExtractors::Scalar velocities(0);

        std::vector<Tensor<1, 2> > phi_grads_v(n_dofs_per_element);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          for (unsigned int k = 0; k < n_dofs_per_element; k++)
          {
            phi_grads_v[k] = state_fe_values[velocities].gradient(k, q_point);
          }

          for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            for (unsigned int j = 0; j < n_dofs_per_element; j++)
            {

              local_matrix(i, j) += scale * phi_grads_v[j]
                  * phi_grads_v[i] * state_fe_values.JxW(q_point);
            }
          }
        }
      }

      void
      ElementRightHandSide(const EDC<DH, VECTOR, dealdim>& edc,
          dealii::Vector<double> &local_vector, double scale)
      {
        assert(this->_problem_type == "state");
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();
        const DOpEWrapper::FEValues<dealdim> &state_fe_values =
            edc.GetFEValuesState();

        _fvalues.resize(n_q_points);
        const FEValuesExtractors::Scalar velocities(0);

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
          _fvalues[q_point] = -_exact_solution.laplacian(
              state_fe_values.quadrature_point(q_point));
          for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) += scale * _fvalues[q_point]
                * state_fe_values[velocities].value(i, q_point)
                * state_fe_values.JxW(q_point);
          }
        } //endfor qpoint
      }

      void
	BoundaryEquation(const FDC<DH, VECTOR, dealdim>& /*fdc*/,
		       dealii::Vector<double> &, double /*scale*/, double /*scale_ico*/)
      {

      }

      void
      BoundaryMatrix(const FDC<DH, VECTOR, dealdim>& /*fdc*/,
          dealii::FullMatrix<double> & /*local_matrix*/, double /*scale*/,
          double /*scale_ico*/)
      {
      }

      void
      BoundaryRightHandSide(const FDC<DH, VECTOR, dealdim>& /*fdc*/,
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
        return 1;
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
      vector<double> _fvalues;

      vector<Tensor<1, dealdim> > _ugrads;

      ExactSolution _exact_solution;

      vector<unsigned int> _state_block_components;
  };
#endif
