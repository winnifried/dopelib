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
#include "celldatacontainer.h"
#include "facedatacontainer.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

template<typename VECTOR, int dealdim>
  class LocalPDE : public PDEInterface<CellDataContainer,FaceDataContainer,dealii::DoFHandler<dealdim>, VECTOR, dealdim>
  {
  public:
    LocalPDE() :
      _state_block_components(3, 0)
    {
      _state_block_components[2] = 1;
    }

    // Domain values for cells
    void
    CellEquation(
        const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
        dealii::Vector<double> &local_cell_vector, double scale, double /*scale_ico*/)
    {
      const DOpEWrapper::FEValues<dealdim> & state_fe_values =
          cdc.GetFEValuesState();
      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
      unsigned int n_q_points = cdc.GetNQPoints();
      assert(this->_problem_type == "state");

      _uvalues.resize(n_q_points, Vector<double> (3));
      _ugrads.resize(n_q_points, vector<Tensor<1, 2> > (3));

      cdc.GetValuesState("last_newton_solution", _uvalues);
      cdc.GetGradsState("last_newton_solution", _ugrads);

      const FEValuesExtractors::Vector velocities(0);
      const FEValuesExtractors::Scalar pressure(2);

      for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          Tensor<2, 2> vgrads;
          vgrads.clear();
          vgrads[0][0] = _ugrads[q_point][0][0];
          vgrads[0][1] = _ugrads[q_point][0][1];
          vgrads[1][0] = _ugrads[q_point][1][0];
          vgrads[1][1] = _ugrads[q_point][1][1];

          double press = _uvalues[q_point](2);
          double incompressibility = vgrads[0][0] + vgrads[1][1];

          for (unsigned int i = 0; i < n_dofs_per_cell; i++)
            {
//              const Tensor<1, 2> phi_i_v = state_fe_values[velocities].value(i,
//                  q_point);
              const Tensor<2, 2> phi_i_grads_v =
                  state_fe_values[velocities].gradient(i, q_point);
              const double phi_i_p =
                  state_fe_values[pressure].value(i, q_point);
              const double div_phi_v = state_fe_values[velocities].divergence(
                  i, q_point);

              local_cell_vector(i) += scale * (0.5 * scalar_product(vgrads,
                  phi_i_grads_v) + 0.5 * scalar_product(transpose(vgrads),
                  phi_i_grads_v) - press * div_phi_v + incompressibility
                  * phi_i_p) * state_fe_values.JxW(q_point);

            }
        }

    }

    void
    CellMatrix(
        const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
        FullMatrix<double> &local_entry_matrix,
        double scale, double /*scale_ico*/)
    {

      const DOpEWrapper::FEValues<dealdim> & state_fe_values =
          cdc.GetFEValuesState();
      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
      unsigned int n_q_points = cdc.GetNQPoints();

      const FEValuesExtractors::Vector velocities(0);
      const FEValuesExtractors::Scalar pressure(2);

      std::vector<Tensor<1, 2> > phi_v(n_dofs_per_cell);
      std::vector<Tensor<2, 2> > phi_grads_v(n_dofs_per_cell);
      std::vector<double> phi_p(n_dofs_per_cell);
      std::vector<double> div_phi_v(n_dofs_per_cell);

      for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          for (unsigned int k = 0; k < n_dofs_per_cell; k++)
            {
              phi_v[k] = state_fe_values[velocities].value(k, q_point);
              phi_grads_v[k] = state_fe_values[velocities].gradient(k, q_point);
              phi_p[k] = state_fe_values[pressure].value(k, q_point);
              div_phi_v[k] = state_fe_values[velocities].divergence(k, q_point);
            }

          for (unsigned int i = 0; i < n_dofs_per_cell; i++)
            {
              for (unsigned int j = 0; j < n_dofs_per_cell; j++)
                {
                  //const Tensor<2,2> phi_j_grads_v = state_fe_values[velocities].gradient (j, q_point);
                  //const double phi_j_p = state_fe_values[pressure].value (j, q_point);

                  local_entry_matrix(i, j) += scale * (0.5 * scalar_product(
                      phi_grads_v[j], phi_grads_v[i]) + 0.5 * scalar_product(
                      transpose(phi_grads_v[j]), phi_grads_v[i]) - phi_p[j]
                      * div_phi_v[i] + (phi_grads_v[j][0][0]
                      + phi_grads_v[j][1][1]) * phi_p[i])
                      * state_fe_values.JxW(q_point);
                }
            }
        }

    }

    void
    CellRightHandSide(
        const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& /*cdc*/,
        dealii::Vector<double> &local_cell_vector __attribute__((unused)), double scale __attribute__((unused)))
    {
      
    }

    // Values for boundary integrals
    void
    BoundaryEquation(
        const FaceDataContainer<dealii::DoFHandler<2>, VECTOR, dealdim>& fdc,
        dealii::Vector<double> &local_cell_vector, double scale, double /*scale_ico*/)
    {
      const auto & state_fe_face_values =
          fdc.GetFEFaceValuesState();
      unsigned int n_dofs_per_cell = fdc.GetNDoFsPerCell();
      unsigned int n_q_points = fdc.GetNQPoints();
      unsigned int color = fdc.GetBoundaryIndicator();

      assert(this->_problem_type == "state");

      // do-nothing applied on outflow boundary
      if (color == 1)
        {
          _ufacegrads.resize(n_q_points, vector<Tensor<1, 2> > (3));

          fdc.GetFaceGradsState("last_newton_solution", _ufacegrads);

          const FEValuesExtractors::Vector velocities(0);

          for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
            {
              Tensor<2, 2> vgrads;
              vgrads.clear();
              vgrads[0][0] = _ufacegrads[q_point][0][0];
              vgrads[0][1] = _ufacegrads[q_point][0][1];
              vgrads[1][0] = _ufacegrads[q_point][1][0];
              vgrads[1][1] = _ufacegrads[q_point][1][1];

              for (unsigned int i = 0; i < n_dofs_per_cell; i++)
                {
                  const Tensor<1, 2> phi_i_v =
                      state_fe_face_values[velocities].value(i, q_point);

                  const Tensor<1, 2> neumann_value = (transpose(vgrads)
                      * state_fe_face_values.normal_vector(q_point));

                  local_cell_vector(i) += -scale * 0.5 * neumann_value
                      * phi_i_v * state_fe_face_values.JxW(q_point);
                }
            }
        }

    }

    void
    BoundaryMatrix(
        const FaceDataContainer<dealii::DoFHandler<2>, VECTOR, dealdim>& fdc,
        dealii::FullMatrix<double> &local_entry_matrix, double /*scale*/, double /*scale_ico*/)
    {
      const auto & state_fe_face_values =
          fdc.GetFEFaceValuesState();
      unsigned int n_dofs_per_cell = fdc.GetNDoFsPerCell();
      unsigned int n_q_points = fdc.GetNQPoints();
      unsigned int color = fdc.GetBoundaryIndicator();
      assert(this->_problem_type == "state");

      // do-nothing applied on outflow boundary
      if (color == 1)
        {
          const FEValuesExtractors::Vector velocities(0);

          for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
            {
              for (unsigned int i = 0; i < n_dofs_per_cell; i++)
                {
                  const Tensor<1, 2> phi_i_v =
                      state_fe_face_values[velocities].value(i, q_point);

                  for (unsigned int j = 0; j < n_dofs_per_cell; j++)
                    {
                      const Tensor<2, 2> phi_j_grads_v =
                          state_fe_face_values[velocities].gradient(j, q_point);

                      const Tensor<1, 2> neumann_value = (transpose(
                          phi_j_grads_v) * state_fe_face_values.normal_vector(
                          q_point));

                      local_entry_matrix(i, j) += -0.5 * neumann_value
                          * phi_i_v * state_fe_face_values.JxW(q_point);
                    }
                }
            }
        }

    }

    void
    BoundaryRightHandSide(
        const FaceDataContainer<dealii::DoFHandler<2>, VECTOR, dealdim>&,
        dealii::Vector<double> &local_cell_vector __attribute__((unused)), double scale __attribute__((unused)))
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
      return 2;
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
  };
#endif
