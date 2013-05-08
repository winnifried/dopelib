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

template<template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
  template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
  template<int, int> class DH, typename VECTOR, int dealdim>
  class LocalPDE: public PDEInterface<CDC,FDC,DH, VECTOR, dealdim>
{
  public:
    LocalPDE() :
      _state_block_components(2, 0)
    {

    }

    // Domain values for cells
    void CellEquation(const CDC<DH, VECTOR, dealdim>& cdc,
                      dealii::Vector<double> &local_cell_vector,
                      double scale, double)
    {
      assert(this->_problem_type == "state");

      const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
      unsigned int n_q_points = cdc.GetNQPoints();

      _ugrads.resize(n_q_points, vector<Tensor<1, 2> > (2));

      cdc.GetGradsState("last_newton_solution", _ugrads);

      const FEValuesExtractors::Vector displacements(0);

      for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        Tensor<2, 2> ugrads;
        ugrads.clear();
        ugrads[0][0] = _ugrads[q_point][0][0];
        ugrads[0][1] = _ugrads[q_point][0][1];
        ugrads[1][0] = _ugrads[q_point][1][0];
        ugrads[1][1] = _ugrads[q_point][1][1];

	for (unsigned int i = 0; i < n_dofs_per_cell; i++)
        {
	  const Tensor<2, 2> phi_i_grads_u = state_fe_values[displacements].gradient(i, q_point);

          local_cell_vector(i) += scale * scalar_product(ugrads, phi_i_grads_u) 
	    * state_fe_values.JxW(q_point);
        }
      }

    }

    void CellMatrix(const CDC<DH, VECTOR, dealdim>& cdc, FullMatrix<double> &local_entry_matrix, double, double)
    {

      const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
      unsigned int n_q_points = cdc.GetNQPoints();

      const FEValuesExtractors::Vector displacements(0);

      std::vector<Tensor<2, 2> > phi_grads_u(n_dofs_per_cell);

      for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        for (unsigned int k = 0; k < n_dofs_per_cell; k++)
        {
          phi_grads_u[k] = state_fe_values[displacements].gradient(k, q_point);
        }

        for (unsigned int i = 0; i < n_dofs_per_cell; i++)
        {
          for (unsigned int j = 0; j < n_dofs_per_cell; j++)
          {
            local_entry_matrix(i, j) += scalar_product(phi_grads_u[j], phi_grads_u[i]) 
                * state_fe_values.JxW(q_point);
          }
        }
      }

    }

    void CellRightHandSide(const CDC<DH, VECTOR, dealdim>& cdc,
                           dealii::Vector<double> &local_cell_vector __attribute__((unused)),
                           double scale __attribute__((unused)))
    {
      assert(this->_problem_type == "state");
            
      const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
      unsigned int n_q_points = cdc.GetNQPoints();
 
      const FEValuesExtractors::Vector displacements(0);

      Tensor<1, 2> fvalues;
      fvalues.clear();
      fvalues[0] = 1.0; 
      fvalues[1] = 1.0;

      for (unsigned int q_point=0;q_point<n_q_points; ++q_point)
      {
	for(unsigned int i = 0; i < n_dofs_per_cell; i++)
	{
	  const Tensor<1, 2> phi_i_u = state_fe_values[displacements].value(i, q_point);

	  local_cell_vector(i) += scale * fvalues * phi_i_u
	    * state_fe_values.JxW(q_point);
	}
      }
    }

    UpdateFlags GetUpdateFlags() const
    {      
        return update_values | update_gradients | update_quadrature_points;
    }

    unsigned int GetStateNBlocks() const
    {
      return 1;
    }
    std::vector<unsigned int>& GetStateBlockComponent()
    {
      return _state_block_components;
    }
    const std::vector<unsigned int>& GetStateBlockComponent() const
    {
      return _state_block_components;
    }

  private:
    vector<vector<Tensor<1, dealdim> > > _ugrads;

    vector<unsigned int> _state_block_components;

};
#endif
