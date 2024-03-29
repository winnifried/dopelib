/**
 *
 * Copyright (C) 2012-2018 by the DOpElib authors
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
#include <container/elementdatacontainer.h>
#include <container/facedatacontainer.h>

using namespace std;
using namespace dealii;
using namespace DOpE;

#if DEAL_II_VERSION_GTE(9,3,0)
template<
  template<bool DH, typename VECTOR, int dealdim> class EDC,
  template<bool DH, typename VECTOR, int dealdim> class FDC,
  bool DH, typename VECTOR, int dealdim>
class LocalPDE : public PDEInterface<EDC, FDC, DH, VECTOR, dealdim>
#else
template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
  template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
  template<int, int> class DH, typename VECTOR, int dealdim>
class LocalPDE : public PDEInterface<EDC, FDC, DH, VECTOR, dealdim>
#endif
{
public:
  LocalPDE() :
    state_block_component_(2, 0)
  {
  }

  void
  ElementEquation(
    const EDC<DH, VECTOR, dealdim> &edc,
    dealii::Vector<double> &local_vector, double scale, double) override
  {
    assert(this->problem_type_ == "state");

    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();

    ugrads_.resize(n_q_points, vector<Tensor<1, dealdim> >(2));

    edc.GetGradsState("last_newton_solution", ugrads_);

    const FEValuesExtractors::Vector extracto(0);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        Tensor<2, dealdim> ugrads;
        ugrads.clear();
        ugrads[0][0] = ugrads_[q_point][0][0];
        ugrads[0][1] = ugrads_[q_point][0][1];
        ugrads[1][0] = ugrads_[q_point][1][0];
        ugrads[1][1] = ugrads_[q_point][1][1];

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            const Tensor<2, dealdim> phi_i_grads_u =
              state_fe_values[extracto].gradient(i, q_point);

            local_vector(i) += scale
                               * scalar_product(ugrads, phi_i_grads_u)
                               * state_fe_values.JxW(q_point);
          }
      }

  }

  void
  ElementMatrix(
    const EDC<DH, VECTOR, dealdim> &edc,
    FullMatrix<double> &local_matrix, double, double) override
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();

    const FEValuesExtractors::Vector extracto(0);

    std::vector<Tensor<2, dealdim> > phi_grads_u(n_dofs_per_element);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        for (unsigned int k = 0; k < n_dofs_per_element; k++)
          {
            phi_grads_u[k] = state_fe_values[extracto].gradient(k,
                                                                q_point);
          }

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            for (unsigned int j = 0; j < n_dofs_per_element; j++)
              {
                local_matrix(i, j) += scalar_product(phi_grads_u[j],
                                                     phi_grads_u[i]) * state_fe_values.JxW(q_point);
              }
          }
      }
  }

  void
  ElementRightHandSide(
    const EDC<DH, VECTOR, dealdim> &edc,
    dealii::Vector<double> &local_vector,
    double scale) override
  {
    assert(this->problem_type_ == "state");

    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();

    const FEValuesExtractors::Vector extracto(0);

    Tensor<1, dealdim> fvalues;
    fvalues.clear();
    fvalues[0] = 1.0;
    fvalues[1] = 1.0;

    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
      {
        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            const Tensor<1, dealdim> phi_i_u = state_fe_values[extracto].value(i,
                                               q_point);

            local_vector(i) += scale * fvalues * phi_i_u
                               * state_fe_values.JxW(q_point);
          }
      }
  }

  UpdateFlags
  GetUpdateFlags() const override
  {
    return update_values | update_gradients | update_quadrature_points;
  }

  unsigned int
  GetStateNBlocks() const override
  {
    return 1;
  }
  std::vector<unsigned int> &
  GetStateBlockComponent() override
  {
    return state_block_component_;
  }
  const std::vector<unsigned int> &
  GetStateBlockComponent() const override
  {
    return state_block_component_;
  }

private:
  vector<vector<Tensor<1, dealdim> > > ugrads_;

  vector<unsigned int> state_block_component_;

};
#endif
