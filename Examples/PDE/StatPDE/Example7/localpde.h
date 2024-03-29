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
    assert(dealdim==2);
  }

  // Domain values for elements
  void
  ElementEquation(const EDC<DH, VECTOR, dealdim> &edc,
                  dealii::Vector<double> &local_vector, double scale, double /*scale_ico*/) override
  {
    assert(this->problem_type_ == "state");

    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();

    uvalues_.resize(n_q_points, Vector<double>(2));
    ugrads_.resize(n_q_points, vector<Tensor<1, 2> >(2));

    edc.GetValuesState("last_newton_solution", uvalues_);
    edc.GetGradsState("last_newton_solution", ugrads_);

    const FEValuesExtractors::Vector displacements(0);

    const double mu = 80193.800283;
    const double lambda = 110743.788889;

    // kappa = 2*mu + lambda
    const double kappa = 271131.389455;


    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        Tensor<2, 2> vgrads;
        vgrads.clear();
        vgrads[0][0] = ugrads_[q_point][0][0];
        vgrads[0][1] = ugrads_[q_point][0][1];
        vgrads[1][0] = ugrads_[q_point][1][0];
        vgrads[1][1] = ugrads_[q_point][1][1];

        Tensor<2, 2> realgrads;
        realgrads.clear();
        realgrads[0][0] = kappa * vgrads[0][0] + lambda * vgrads[1][1];
        realgrads[0][1] = mu * vgrads[0][1] + mu * vgrads[1][0];
        realgrads[1][0] = mu * vgrads[0][1] + mu * vgrads[1][0];
        realgrads[1][1] = kappa * vgrads[1][1] + lambda * vgrads[0][0];

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            const Tensor<2, 2> phi_i_grads_v =
              state_fe_values[displacements].gradient(i, q_point);
            const Tensor<2, 2> phi_i_grads_real = 0.5 * phi_i_grads_v
                                                  + 0.5 * transpose(phi_i_grads_v);

            local_vector(i) += scale
                               * (scalar_product(realgrads, phi_i_grads_real))
                               * state_fe_values.JxW(q_point);
          }
      }
  }

  void
  ElementMatrix(const EDC<DH, VECTOR, dealdim> &edc,
                FullMatrix<double> &local_matrix, double scale, double /*scale_ico*/) override
  {
    assert(this->problem_type_ == "state");

    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();

    const FEValuesExtractors::Vector displacements(0);

    const double mu = 80193.800283;
    const double lambda = 110743.788889;

    // kappa = 2*mu + lambda
    const double kappa = 271131.389455;


    std::vector<Tensor<2, 2> > phi_grads_v(n_dofs_per_element);
    std::vector<Tensor<2, 2> > phi_grads_real(n_dofs_per_element);
    std::vector<Tensor<2, 2> > phi_grads_test(n_dofs_per_element);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        for (unsigned int k = 0; k < n_dofs_per_element; k++)
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

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            for (unsigned int j = 0; j < n_dofs_per_element; j++)
              {

                local_matrix(i, j) += scale
                                      * (scalar_product(phi_grads_real[j], phi_grads_test[i]))

                                      * state_fe_values.JxW(q_point);
              }
          }
      }

  }

  void
  ElementRightHandSide(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                       dealii::Vector<double> &/*local_vector*/, double /*scale*/) override
  {
  }

  // Values for boundary integrals
  void
  BoundaryEquation(const FDC<DH, VECTOR, dealdim> &fdc,
                   dealii::Vector<double> &local_vector, double scale,
                   double /*scale_ico*/) override
  {

    assert(this->problem_type_ == "state");

    const auto &state_fe_face_values = fdc.GetFEFaceValuesState();
    unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();
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
            for (unsigned int i = 0; i < n_dofs_per_element; i++)
              {
                const Tensor<1, 2> phi_i_v =
                  state_fe_face_values[displacements].value(i, q_point);

                local_vector(i) += -scale * g * phi_i_v
                                   * state_fe_face_values.JxW(q_point);
              }
          }
      }
  }

  void
  BoundaryMatrix(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                 dealii::FullMatrix<double> &/*local_matrix*/, double /*scale*/,
                 double/*scale_ico*/) override
  {
    assert(this->problem_type_ == "state");
  }

  void
  BoundaryRightHandSide(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                        dealii::Vector<double> &/*local_vector*/, double /*scale*/) override
  {
    assert(this->problem_type_ == "state");
  }

  UpdateFlags
  GetUpdateFlags() const override
  {
    return update_values | update_gradients | update_quadrature_points;
  }

  UpdateFlags
  GetFaceUpdateFlags() const override
  {
    return update_values | update_gradients | update_normal_vectors
           | update_quadrature_points;
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

protected:

private:
  vector<Vector<double> > uvalues_;

  vector<vector<Tensor<1, dealdim> > > ugrads_;

  vector<unsigned int> state_block_component_;
};
#endif
