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

#ifndef __LOCALPDE
#define __LOCALPDE

#include "pdeinterface.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

/***********************************************************************************************/
template<template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
  template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
  template<int, int> class DH, typename VECTOR, int dealdim>
  class LocalPDELaplace : public PDEInterface<CDC,FDC,DH, VECTOR, dealdim>
  {
  public:
  LocalPDELaplace() :
      _state_block_components(2, 0)
    {
    }

    // Domain values for cells
    void
    CellEquation(
        const CDC<DH, VECTOR, dealdim>& cdc,
        dealii::Vector<double> &local_cell_vector, double scale, double)
    {
      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
      unsigned int n_q_points = cdc.GetNQPoints();
      const DOpEWrapper::FEValues<dealdim> &state_fe_values =
          cdc.GetFEValuesState();

      assert(this->_problem_type == "state");

      _uvalues.resize(n_q_points, Vector<double> (2));
      _ugrads.resize(n_q_points, vector<Tensor<1, dealdim> > (2));
      cdc.GetValuesState("last_newton_solution", _uvalues);
      cdc.GetGradsState("last_newton_solution", _ugrads);

      const FEValuesExtractors::Vector velocities(0);
      const FEValuesExtractors::Scalar pressure(dealdim);

      for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          Tensor<2, 2> vgrads;
          vgrads.clear();
          vgrads[0][0] = _ugrads[q_point][0][0];
          vgrads[0][1] = _ugrads[q_point][0][1];
          vgrads[1][0] = _ugrads[q_point][1][0];
          vgrads[1][1] = _ugrads[q_point][1][1];

          for (unsigned int i = 0; i < n_dofs_per_cell; i++)
            {
//              const Tensor<1, 2> phi_i_v = state_fe_values[velocities].value(i,
//                  q_point);
              const Tensor<2, 2> phi_i_grads_v =
                  state_fe_values[velocities].gradient(i, q_point);

              local_cell_vector(i) += scale * (scalar_product(vgrads,
                  phi_i_grads_v)) * state_fe_values.JxW(q_point);

            }
        }
    }

    void
    CellMatrix(
        const CDC<DH, VECTOR, dealdim>& cdc,
        FullMatrix<double> &local_entry_matrix, double scale, double)
    {
      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
      unsigned int n_q_points = cdc.GetNQPoints();
      //unsigned int material_id = cdc.GetMaterialId();
      const DOpEWrapper::FEValues<dealdim> &state_fe_values =
          cdc.GetFEValuesState();

      const FEValuesExtractors::Vector velocities(0);
      const FEValuesExtractors::Scalar pressure(dealdim);

      std::vector<Tensor<1, 2> > phi_v(n_dofs_per_cell);
      std::vector<Tensor<2, 2> > phi_grads_v(n_dofs_per_cell);

      for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          for (unsigned int k = 0; k < n_dofs_per_cell; k++)
            {
              phi_grads_v[k] = state_fe_values[velocities].gradient(k, q_point);
            }

          for (unsigned int i = 0; i < n_dofs_per_cell; i++)
            {
              for (unsigned int j = 0; j < n_dofs_per_cell; j++)
                {

                  local_entry_matrix(i, j) += scale * scalar_product(phi_grads_v[j],
                      phi_grads_v[i]) * state_fe_values.JxW(q_point);
                }
            }
        }
    }

    void
    CellRightHandSide(
        const CDC<DH, VECTOR, dealdim>& cdc,
        dealii::Vector<double> &local_cell_vector, double scale)
    {
      assert(this->_problem_type == "state");
      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
      unsigned int n_q_points = cdc.GetNQPoints();
      const DOpEWrapper::FEValues<dealdim> &state_fe_values =
          cdc.GetFEValuesState();

      _fvalues.resize(n_q_points);
      std::vector<Tensor<1, 2> > phi_v(n_dofs_per_cell);
      const FEValuesExtractors::Vector velocities(0);

      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
          const double x = state_fe_values.quadrature_point(q_point)(0);
          const double y = state_fe_values.quadrature_point(q_point)(1);
          for (unsigned int i = 0; i < dealdim; i++)
            {
              _fvalues[q_point][0] = cos(exp(10 * x)) * y * y * x + sin(y);
              _fvalues[q_point][1] = cos(exp(10 * y)) * x * x * y + sin(x);
            }
          for (unsigned int i = 0; i < n_dofs_per_cell; i++)
            {
              local_cell_vector(i) += scale * (contract(_fvalues[q_point],
                  state_fe_values[velocities].value(i, q_point)))
                  * state_fe_values.JxW(q_point);
            }
        } //endfor qpoint
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
    bool
    HasFaces() const
    {
      return false;
    }
    bool
    HasInterfaces() const
    {
      return false;
    }
  private:

    vector<Tensor<1, dealdim> > _fvalues;
    vector<Vector<double> > _uvalues;

    vector<vector<Tensor<1, dealdim> > > _ugrads;

    vector<unsigned int> _state_block_components;
  };
//**********************************************************************************


#endif

