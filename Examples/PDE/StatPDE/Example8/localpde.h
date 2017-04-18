/**
 *
 * Copyright (C) 2012-2014 by the DOpElib authors
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

template<
template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
         template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
         template<int, int> class DH, typename VECTOR, int dealdim>
class LocalPDE : public PDEInterface<EDC, FDC, DH, VECTOR, dealdim>
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
                  dealii::Vector<double> &local_vector, double scale,
                  double /*scale_ico*/)
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

    // Physical Lame parameters
    const double mu = 80193.800283;
    const double lambda = 110743.788889;

    // Abbrev. to avoid lengthy terms
    // in the following: rho = mu + lambda
    //                   kappa = 2*mu + lambda
    const double rho = 190937.589172;
    const double kappa = 271131.389455;

    const double sigma = sqrt(2. / 3.) * 450.0;
    double norm = 0.;
    double factor = 0.;

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

        Tensor<2, 2> deviator;
        deviator.clear();
        deviator[0][0] = realgrads[0][0] - rho * vgrads[0][0]
                         - rho * vgrads[1][1];
        deviator[0][1] = realgrads[0][1];
        deviator[1][0] = realgrads[1][0];
        deviator[1][1] = realgrads[1][1] - rho * vgrads[0][0]
                         - rho * vgrads[1][1];

        norm = sqrt(
                 deviator[0][0] * deviator[0][0] + deviator[0][1] * deviator[0][1]
                 + deviator[1][0] * deviator[1][0]
                 + deviator[1][1] * deviator[1][1]);

        factor = sigma / norm;

        Tensor<2, 2> projector;
        projector.clear();
        projector[0][0] = factor * deviator[0][0] + rho * vgrads[0][0]
                          + rho * vgrads[1][1];
        projector[0][1] = factor * deviator[0][1];
        projector[1][0] = factor * deviator[1][0];
        projector[1][1] = factor * deviator[1][1] + rho * vgrads[0][0]
                          + rho * vgrads[1][1];

        if (norm <= sigma)
          {

            for (unsigned int i = 0; i < n_dofs_per_element; i++)
              {
                const Tensor<2, 2> phi_i_grads_v =
                  state_fe_values[displacements].gradient(i, q_point);
                const Tensor<2, 2> phi_i_grads = 0.5 * phi_i_grads_v
                                                 + 0.5 * transpose(phi_i_grads_v);

                local_vector(i) += scale
                                   * scalar_product(realgrads, phi_i_grads)
                                   * state_fe_values.JxW(q_point);
              }
          }
        else
          {
            for (unsigned int i = 0; i < n_dofs_per_element; i++)
              {
                const Tensor<2, 2> phi_i_grads_v =
                  state_fe_values[displacements].gradient(i, q_point);
                const Tensor<2, 2> phi_i_grads = 0.5 * phi_i_grads_v
                                                 + 0.5 * transpose(phi_i_grads_v);

                local_vector(i) += scale
                                   * scalar_product(projector, phi_i_grads)
                                   * state_fe_values.JxW(q_point);
              }
          }
      }
  }

  void
  ElementMatrix(const EDC<DH, VECTOR, dealdim> &edc,
                FullMatrix<double> &local_matrix, double scale,
                double /*scale_ico*/)
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

    // Physical Lame parameters
    const double mu = 80193.800283;
    const double lambda = 110743.788889;

    // Abbrev. to avoid lengthy terms
    // in the following: rho = mu + lambda
    //                   kappa = 2*mu + lambda
    const double rho = 190937.589172;
    const double kappa = 271131.389455;

    const double sigma = sqrt(2. / 3.) * 450.0;
    double norm = 0.;

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

        Tensor<2, 2> deviator;
        deviator.clear();
        deviator[0][0] = realgrads[0][0] - rho * vgrads[0][0]
                         - rho * vgrads[1][1];
        deviator[0][1] = realgrads[0][1];
        deviator[1][0] = realgrads[1][0];
        deviator[1][1] = realgrads[1][1] - rho * vgrads[0][0]
                         - rho * vgrads[1][1];

        norm = sqrt(
                 deviator[0][0] * deviator[0][0] + deviator[0][1] * deviator[0][1]
                 + deviator[1][0] * deviator[1][0]
                 + deviator[1][1] * deviator[1][1]);

        for (unsigned int j = 0; j < n_dofs_per_element; j++)
          {
            const Tensor<2, 2> phi_j_grads_v =
              state_fe_values[displacements].gradient(j, q_point);

            Tensor<2, 2> phi_j_grads_real;

            phi_j_grads_real[0][0] = kappa * phi_j_grads_v[0][0]
                                     + lambda * phi_j_grads_v[1][1];
            phi_j_grads_real[0][1] = mu * phi_j_grads_v[0][1]
                                     + mu * phi_j_grads_v[1][0];
            phi_j_grads_real[1][0] = mu * phi_j_grads_v[0][1]
                                     + mu * phi_j_grads_v[1][0];
            phi_j_grads_real[1][1] = kappa * phi_j_grads_v[1][1]
                                     + lambda * phi_j_grads_v[0][0];

            Tensor<2, 2> phi_j_grads_dev;

            phi_j_grads_dev[0][0] = phi_j_grads_real[0][0]
                                    - rho * phi_j_grads_v[0][0] - rho * phi_j_grads_v[1][1];
            phi_j_grads_dev[0][1] = phi_j_grads_real[0][1];
            phi_j_grads_dev[1][0] = phi_j_grads_real[1][0];
            phi_j_grads_dev[1][1] = phi_j_grads_real[1][1]
                                    - rho * phi_j_grads_v[0][0] - rho * phi_j_grads_v[1][1];

            Tensor<2, 2> dev = deviator;

            double newnorm = norm;

            double prod = scalar_product(dev, phi_j_grads_dev);

            Tensor<2, 2> traceterm;
            traceterm[0][0] = 0.5
                              * (phi_j_grads_real[0][0] + phi_j_grads_real[1][1]);
            traceterm[0][1] = 0;
            traceterm[1][0] = 0;
            traceterm[1][1] = 0.5
                              * (phi_j_grads_real[0][0] + phi_j_grads_real[1][1]);

            Tensor<2, 2> fullderivative = -sigma / (newnorm * newnorm * newnorm)
                                          * prod * dev + sigma / newnorm * phi_j_grads_dev + traceterm;

            for (unsigned int i = 0; i < n_dofs_per_element; i++)
              {
                const Tensor<2, 2> phi_i_grads_v =
                  state_fe_values[displacements].gradient(i, q_point);
                const Tensor<2, 2> phi_i_grads_test = 0.5 * phi_i_grads_v
                                                      + 0.5 * transpose(phi_i_grads_v);

                if (norm <= sigma)
                  {
                    local_matrix(i, j) += scale
                                          * scalar_product(phi_j_grads_real, phi_i_grads_test)
                                          * state_fe_values.JxW(q_point);
                  }
                else
                  {
                    local_matrix(i, j) += scale
                                          * scalar_product(fullderivative, phi_i_grads_test)
                                          * state_fe_values.JxW(q_point);
                  }
              }
          }
      }
  }

  void
  ElementRightHandSide(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                       dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
    assert(this->problem_type_ == "state");
  }

  // Values for boundary integrals
  void
  BoundaryEquation(const FDC<DH, VECTOR, dealdim> &fdc,
                   dealii::Vector<double> &local_vector, double scale,
                   double /*scale_ico*/)
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
        g[1] = 400;

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            for (unsigned int i = 0; i < n_dofs_per_element; i++)
              {
                const Tensor<1, 2> phi_i_v =
                  state_fe_face_values[displacements].value(i, q_point);

                local_vector(i) += -scale * (g * phi_i_v)
                                   * state_fe_face_values.JxW(q_point);
              }
          }
      }
  }

  void
  BoundaryMatrix(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                 dealii::FullMatrix<double> &/*local_matrix*/, double /*scale*/,
                 double /*scale_ico*/)
  {
    assert(this->problem_type_ == "state");
  }

  void
  BoundaryRightHandSide(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                        dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
    assert(this->problem_type_ == "state");
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
  std::vector<unsigned int> &
  GetStateBlockComponent()
  {
    return state_block_component_;
  }
  const std::vector<unsigned int> &
  GetStateBlockComponent() const
  {
    return state_block_component_;
  }

private:
  vector<Vector<double> > uvalues_;

  vector<vector<Tensor<1, dealdim> > > ugrads_;

  vector<unsigned int> state_block_component_;
};
#endif
