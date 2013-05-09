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
    template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
    template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
    template<int, int> class DH, typename VECTOR, int dealdim>
  class LocalPDE : public PDEInterface<CDC, FDC, DH, VECTOR, dealdim>
  {
    public:
      LocalPDE() :
          _state_block_components(2, 0)
      {
        assert(dealdim==2);
      }

      // Domain values for cells
      void
      CellEquation(const CDC<DH, VECTOR, dealdim>& cdc,
          dealii::Vector<double> &local_cell_vector, double scale, double /*scale_ico*/)
      {
        assert(this->_problem_type == "state");

        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            cdc.GetFEValuesState();
        unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
        unsigned int n_q_points = cdc.GetNQPoints();

        _uvalues.resize(n_q_points, Vector<double>(2));
        _ugrads.resize(n_q_points, vector<Tensor<1, 2> >(2));

        cdc.GetValuesState("last_newton_solution", _uvalues);
        cdc.GetGradsState("last_newton_solution", _ugrads);

        const FEValuesExtractors::Vector displacements(0);

        const double mu = 80193.800283;
        const double kappa = 271131.389455;
        const double lambda = 110743.788889;

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          Tensor<2, 2> vgrads;
          vgrads.clear();
          vgrads[0][0] = _ugrads[q_point][0][0];
          vgrads[0][1] = _ugrads[q_point][0][1];
          vgrads[1][0] = _ugrads[q_point][1][0];
          vgrads[1][1] = _ugrads[q_point][1][1];

          Tensor<2, 2> realgrads;
          realgrads.clear();
          realgrads[0][0] = kappa * vgrads[0][0] + lambda * vgrads[1][1];
          realgrads[0][1] = mu * vgrads[0][1] + mu * vgrads[1][0];
          realgrads[1][0] = mu * vgrads[0][1] + mu * vgrads[1][0];
          realgrads[1][1] = kappa * vgrads[1][1] + lambda * vgrads[0][0];

          for (unsigned int i = 0; i < n_dofs_per_cell; i++)
          {
            const Tensor<2, 2> phi_i_grads_v =
                state_fe_values[displacements].gradient(i, q_point);
            const Tensor<2, 2> phi_i_grads_real = 0.5 * phi_i_grads_v
                + 0.5 * transpose(phi_i_grads_v);

            local_cell_vector(i) += scale
                * (scalar_product(realgrads, phi_i_grads_real))
                * state_fe_values.JxW(q_point);
          }
        }
      }

      void
      CellMatrix(const CDC<DH, VECTOR, dealdim>& cdc,
          FullMatrix<double> &local_entry_matrix, double scale, double /*scale_ico*/)
      {
        assert(this->_problem_type == "state");

        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            cdc.GetFEValuesState();
        unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
        unsigned int n_q_points = cdc.GetNQPoints();

        const FEValuesExtractors::Vector displacements(0);

        const double mu = 80193.800283;
        const double kappa = 271131.389455;
        const double lambda = 110743.788889;

        std::vector<Tensor<2, 2> > phi_grads_v(n_dofs_per_cell);
        std::vector<Tensor<2, 2> > phi_grads_real(n_dofs_per_cell);
        std::vector<Tensor<2, 2> > phi_grads_test(n_dofs_per_cell);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          for (unsigned int k = 0; k < n_dofs_per_cell; k++)
          {

            phi_grads_v[k] = state_fe_values[displacements].gradient(k,
                q_point);
            phi_grads_real[k][0][0] = kappa * phi_grads_v[k][0][0]
                + lambda * phi_grads_v[k][1][1];
            phi_grads_real[k][0][1] = mu * phi_grads_v[k][0][1]
                + mu * phi_grads_v[k][1][0];
            phi_grads_real[k][1][0] = mu * phi_grads_v[k][0][1]
                + mu * phi_grads_v[k][1][0];
            phi_grads_real[k][1][1] = kappa * phi_grads_v[k][1][1]
                + lambda * phi_grads_v[k][0][0];
            phi_grads_test[k] = 0.5 * phi_grads_v[k]
                + 0.5 * transpose(phi_grads_v[k]);
          }

          for (unsigned int i = 0; i < n_dofs_per_cell; i++)
          {
            for (unsigned int j = 0; j < n_dofs_per_cell; j++)
            {

              local_entry_matrix(i, j) += scale
                  * (scalar_product(phi_grads_real[j], phi_grads_test[i]))

                  * state_fe_values.JxW(q_point);
            }
          }
        }

      }

      void
      CellRightHandSide(const CDC<DH, VECTOR, dealdim>& /*cdc*/,
          dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/)
      {
      }

      // Values for boundary integrals
      void
      BoundaryEquation(const FDC<DH, VECTOR, dealdim>& fdc,
          dealii::Vector<double> &local_cell_vector, double scale,
          double /*scale_ico*/)
      {

        assert(this->_problem_type == "state");

        const auto & state_fe_face_values = fdc.GetFEFaceValuesState();
        unsigned int n_dofs_per_cell = fdc.GetNDoFsPerCell();
        unsigned int n_q_points = fdc.GetNQPoints();
        unsigned int color = fdc.GetBoundaryIndicator();

        //traction on the upper boundary segment realized as Neumann condition
        if (color == 3)
        {
          const FEValuesExtractors::Vector displacements(0);

          Tensor<1, 2> g;
          g[0] = 0;
          g[1] = 450;

          for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            for (unsigned int i = 0; i < n_dofs_per_cell; i++)
            {
              const Tensor<1, 2> phi_i_v =
                  state_fe_face_values[displacements].value(i, q_point);

              local_cell_vector(i) += -scale * g * phi_i_v
                  * state_fe_face_values.JxW(q_point);
            }
          }
        }
      }

      void
      BoundaryMatrix(const FDC<DH, VECTOR, dealdim>& /*fdc*/,
          dealii::FullMatrix<double> &/*local_entry_matrix*/, double /*scale*/,
          double/*scale_ico*/)
      {
        assert(this->_problem_type == "state");
      }

      void
      BoundaryRightHandSide(const FDC<DH, VECTOR, dealdim>& /*fdc*/,
          dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/)
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

    protected:

    private:
      vector<Vector<double> > _uvalues;

      vector<vector<Tensor<1, dealdim> > > _ugrads;

      vector<unsigned int> _state_block_components;
  };
#endif
