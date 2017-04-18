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
    state_block_component_[1] = 1;
  }

  void
  ElementEquation(const EDC<DH, VECTOR, dealdim> &edc,
                  dealii::Vector<double> &local_vector, double scale, double)
  {
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();

    assert(this->problem_type_ == "state");
    uvalues_.resize(n_q_points,Vector<double>(3));
    ugrads_.resize(n_q_points, vector<Tensor<1, dealdim> >(3));
    edc.GetGradsState("last_newton_solution", ugrads_);
    edc.GetValuesState("last_newton_solution", uvalues_);

    const FEValuesExtractors::Vector velocities (0);
    const FEValuesExtractors::Scalar pressure (dealdim);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        Tensor<1, dealdim> u;
        u.clear();
        u[0] = uvalues_[q_point](0);
        u[1] = uvalues_[q_point](1);
        double p = uvalues_[q_point](2);
        double div_u = ugrads_[q_point][0][0] + ugrads_[q_point][1][1];

        Tensor<2,dealdim> K;
        K[0][0] = 1.;
        K[0][1] = 0.;
        K[1][0] = 0.;
        K[1][1] = 1.;

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            const Tensor<1,dealdim> phi_i_u     = state_fe_values[velocities].value (i, q_point);
            const double            div_phi_i_u = state_fe_values[velocities].divergence (i, q_point);
            const double            phi_i_p     = state_fe_values[pressure].value (i, q_point);

            local_vector(i) += scale *
                               ( phi_i_u * K * u - div_phi_i_u * p - phi_i_p * div_u)
                               * state_fe_values.JxW(q_point);
          }
      }
  }

  void
  ElementMatrix(const EDC<DH, VECTOR, dealdim> &edc,
                FullMatrix<double> &local_matrix, double scale, double)
  {
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();

    const FEValuesExtractors::Vector velocities (0);
    const FEValuesExtractors::Scalar pressure (dealdim);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        Tensor<2,dealdim> K;
        K[0][0] = 1.;
        K[0][1] = 0.;
        K[1][0] = 0.;
        K[1][1] = 1.;

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            const Tensor<1,dealdim> phi_i_u     = state_fe_values[velocities].value (i, q_point);
            const double            div_phi_i_u = state_fe_values[velocities].divergence (i, q_point);
            const double            phi_i_p     = state_fe_values[pressure].value (i, q_point);
            for (unsigned int j = 0; j < n_dofs_per_element; j++)
              {
                const Tensor<1,dealdim> phi_j_u     = state_fe_values[velocities].value (j, q_point);
                const double            div_phi_j_u = state_fe_values[velocities].divergence (j, q_point);
                const double            phi_j_p     = state_fe_values[pressure].value (j, q_point);

                local_matrix(i,j) += scale *
                                     ( phi_i_u * K * phi_j_u - div_phi_i_u * phi_j_p - phi_i_p * div_phi_j_u)
                                     * state_fe_values.JxW(q_point);
              }
          }
      }
  }

  void
  ElementRightHandSide(const EDC<DH, VECTOR, dealdim> &edc,
                       dealii::Vector<double> &local_vector, double scale)
  {
    assert(this->problem_type_ == "state");
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();

    fvalues_.resize(n_q_points);
    const FEValuesExtractors::Scalar pressure(dealdim);

    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
      {
        fvalues_[q_point] = 0.;
        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) += scale * fvalues_[q_point]
                               * state_fe_values[pressure].value(i, q_point)
                               * state_fe_values.JxW(q_point);
          }
      } //endfor qpoint
  }

  void
  BoundaryEquation(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                   dealii::Vector<double> &, double /*scale*/, double /*scale_ico*/)
  {
    //
  }

  void
  BoundaryMatrix(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                 dealii::FullMatrix<double> & /*local_matrix*/, double /*scale*/,
                 double /*scale_ico*/)
  {
    //
  }

  void
  BoundaryRightHandSide(const FDC<DH, VECTOR, dealdim> &fdc,
                        dealii::Vector<double> &local_vector, double scale)
  {
    assert(this->problem_type_ == "state");
    unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();
    unsigned int n_q_points = fdc.GetNQPoints();
    const auto &state_fe_face_values =
      fdc.GetFEFaceValuesState();

    fvalues_.resize(n_q_points);
    const FEValuesExtractors::Vector velocities(0);
    const double alpha = 0.3;
    const double beta = 1;

    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
      {
        Tensor<1,dealdim> p = state_fe_face_values.quadrature_point(q_point);
        fvalues_[q_point] = (alpha*p[0]*p[1]*p[1]/2 + beta*p[0] - alpha*p[0]*p[0]*p[0]/6);

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) += scale * fvalues_[q_point]
                               * (state_fe_face_values[velocities].value(i, q_point)
                                  * state_fe_face_values.normal_vector(q_point))
                               * state_fe_face_values.JxW(q_point);
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
    return update_values | update_normal_vectors
           | update_quadrature_points;
  }

  unsigned int
  GetStateNBlocks() const
  {
    return 2;
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
  vector<double> fvalues_;
  vector<Vector<double> > uvalues_;
  vector<vector<Tensor<1, dealdim> > > ugrads_;

  vector<unsigned int> state_block_component_;
};
#endif
