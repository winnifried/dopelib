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

#ifndef LOCALPDE_H_
#define LOCALPDE_H_

#include <interfaces/pdeinterface.h>
#include "myfunctions.h"
#include <deal.II/base/numbers.h>

using namespace std;
using namespace dealii;
using namespace DOpE;

/***********************************************************************************************/
template<
template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
         template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
         template<int, int> class DH, typename VECTOR, int dealdim>
class LocalPDE : public PDEInterface<EDC, FDC, DH, VECTOR, dealdim>
{
public:
  LocalPDE(ParameterReader &param_reader) :
    state_block_component_(2, 0)
  {
    param_reader.SetSubsection("localpde parameters");
    R_      = param_reader.get_double("R");
    T_      = param_reader.get_double("T");
    alpha_  = param_reader.get_double("alpha");
    lambda_ = param_reader.get_double("lambda");
    D_      = param_reader.get_double("D");
    g_      = param_reader.get_double("g");
    hprime_ = param_reader.get_double("hprime");

    stab_param_ = 1.;
  }

  void
  ElementEquation(const EDC<DH, VECTOR, dealdim> &edc,
                  dealii::Vector<double> &local_vector, double scale,
                  double/*scale_ico*/)
  {
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();

    assert(this->problem_type_ == "state");

    uvalues_.resize(n_q_points, Vector<double>(2));
    edc.GetValuesState("last_newton_solution", uvalues_);

    const FEValuesExtractors::Scalar w(0);
    const FEValuesExtractors::Scalar rho(1);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) -= scale * (uvalues_[q_point][0] * state_fe_values[rho].gradient(i,q_point)[0])
                               * state_fe_values.JxW(q_point);
            local_vector(i) -= scale * (p(uvalues_[q_point][1]) + v(uvalues_[q_point][0],uvalues_[q_point][1]) )
                               * state_fe_values[w].gradient(i,q_point)[0]
                               * state_fe_values.JxW(q_point);
            local_vector(i) += scale *(gravity(uvalues_[q_point][1]) + friction(uvalues_[q_point][0],uvalues_[q_point][1]))
                               *state_fe_values[w].value(i,q_point)
                               * state_fe_values.JxW(q_point);

          }
      }
  }

  void
  BoundaryEquation(
    const FaceDataContainer<DH, VECTOR, dealdim> &fdc,
    dealii::Vector<double> &local_vector, double scale,
    double /*scale_ico*/)
  {
    unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();
    unsigned int n_q_points = fdc.GetNQPoints();
    unsigned int color = fdc.GetBoundaryIndicator();
    const auto &state_fe_values =
      fdc.GetFEFaceValuesState();

    assert(this->problem_type_ == "state");

    uvalues_.resize(n_q_points, Vector<double>(2));
    fdc.GetFaceValuesState("last_newton_solution", uvalues_);

    const FEValuesExtractors::Scalar w(0);
    const FEValuesExtractors::Scalar rho(1);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        double bn = state_fe_values.normal_vector(q_point)[0];
        SetStab();
        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            if (color == 0 )
              {
                assert(bn < 0);
                //inflow boundary
                double w_in = 1.;
                double rho_in = 1.;
                local_vector(i) += scale
                                   * ((flux_1_plus(uvalues_[q_point][0],uvalues_[q_point][1],bn)
                                       + flux_1_minus(w_in,rho_in,bn)
                                      ) * state_fe_values[w].value(i,q_point)
                                      + (flux_2_plus(uvalues_[q_point][0],uvalues_[q_point][1],bn)
                                         + flux_2_minus(w_in,rho_in,bn)
                                        ) * state_fe_values[rho].value(i,q_point)
                                     )
                                   * state_fe_values.JxW(q_point);
              }
            else
              {
                assert(bn > 0);
                //outflow boundary (reflektive
                double w_out = -uvalues_[q_point][0];
                double rho_out = uvalues_[q_point][1];
                local_vector(i) += scale
                                   * ((flux_1_plus(uvalues_[q_point][0],uvalues_[q_point][1],bn)
                                       + flux_1_minus(w_out,rho_out,bn)
                                      ) * state_fe_values[w].value(i,q_point)
                                      + (flux_2_plus(uvalues_[q_point][0],uvalues_[q_point][1],bn)
                                         + flux_2_minus(w_out,rho_out,bn)
                                        ) * state_fe_values[rho].value(i,q_point)
                                     )* state_fe_values.JxW(q_point);
              }
          }
      }
  }

  void
  FaceEquation(
    const FaceDataContainer<DH, VECTOR, dealdim> &fdc,
    dealii::Vector<double> &local_vector, double scale,
    double /*scale_ico*/)
  {
    //The face equation contains the coupling of the element DOFs
    //with the DOFs from the same element induced by the face integrals
    unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();
    unsigned int n_q_points = fdc.GetNQPoints();
    const auto &state_fe_values = fdc.GetFEFaceValuesState();

    assert(this->problem_type_ == "state");

    uvalues_.resize(n_q_points, Vector<double>(2));
    fdc.GetFaceValuesState("last_newton_solution", uvalues_);

    const FEValuesExtractors::Scalar w(0);
    const FEValuesExtractors::Scalar rho(1);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        double bn = state_fe_values.normal_vector(q_point)[0];
        SetStab();

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) += scale
                               * (flux_1_plus(uvalues_[q_point][0],uvalues_[q_point][1],bn)
                                  *state_fe_values[w].value(i,q_point)
                                  +flux_2_plus(uvalues_[q_point][0],uvalues_[q_point][1],bn)
                                  *state_fe_values[rho].value(i,q_point)
                                 )
                               * state_fe_values.JxW(q_point);
          }
      }
  }
  void
  InterfaceEquation(
    const FaceDataContainer<DH, VECTOR, dealdim> &fdc,
    dealii::Vector<double> &local_vector, double scale,
    double /*scale_ico*/)
  {
    //The interface equation contains the coupling of the element DOFs
    //with the DOFs from the neigbouring element induced by the face integrals
    //The face equation contains the coupling of the element DOFs
    //with the DOFs from the same element induced by the face integrals
    unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();
    unsigned int n_q_points = fdc.GetNQPoints();
    const auto &state_fe_values = fdc.GetFEFaceValuesState();

    assert(this->problem_type_ == "state");

    uvalues_nbr_.resize(n_q_points, Vector<double>(2));
    fdc.GetNbrFaceValuesState("last_newton_solution", uvalues_nbr_);

    const FEValuesExtractors::Scalar w(0);
    const FEValuesExtractors::Scalar rho(1);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        double bn = state_fe_values.normal_vector(q_point)[0];
        SetStab();

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) += scale
                               * (flux_1_minus(uvalues_nbr_[q_point][0],uvalues_nbr_[q_point][1],bn)
                                  *state_fe_values[w].value(i,q_point)
                                  +flux_2_minus(uvalues_nbr_[q_point][0],uvalues_nbr_[q_point][1],bn)
                                  *state_fe_values[rho].value(i,q_point)
                                 )
                               * state_fe_values.JxW(q_point);
          }
      }
  }

  void
  BoundaryMatrix(
    const FaceDataContainer<DH, VECTOR, dealdim> &fdc,
    FullMatrix<double> &local_matrix, double scale,
    double/*scale_ico*/)
  {
    unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();
    unsigned int n_q_points = fdc.GetNQPoints();
    unsigned int color = fdc.GetBoundaryIndicator();
    const auto &state_fe_values =
      fdc.GetFEFaceValuesState();

    uvalues_.resize(n_q_points, Vector<double>(2));
    fdc.GetFaceValuesState("last_newton_solution", uvalues_);

    const FEValuesExtractors::Scalar w(0);
    const FEValuesExtractors::Scalar rho(1);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        double bn = state_fe_values.normal_vector(q_point)[0];
        SetStab();

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            for (unsigned int j = 0; j < n_dofs_per_element; j++)
              {
                if (color == 0)
                  {
                    assert(bn < 0);
                    //inflow boundary
                    local_matrix(i,j) += scale
                                         * (
                                           (flux_1_plus_w(uvalues_[q_point][0],uvalues_[q_point][1],bn)
                                            * state_fe_values[w].value(j,q_point)
                                            + flux_1_plus_r(uvalues_[q_point][0],uvalues_[q_point][1],bn)
                                            * state_fe_values[rho].value(j,q_point)
                                           ) * state_fe_values[w].value(i,q_point)
                                           +
                                           (flux_2_plus_w(uvalues_[q_point][0],uvalues_[q_point][1],bn)
                                            * state_fe_values[w].value(j,q_point)
                                            + flux_2_plus_r(uvalues_[q_point][0],uvalues_[q_point][1],bn)
                                            * state_fe_values[rho].value(j,q_point)
                                           ) * state_fe_values[rho].value(i,q_point)
                                         )
                                         * state_fe_values.JxW(q_point);
                  }
                else
                  {
                    assert(bn > 0);
                    //outflow boundary (reflektive
                    double w_out = -uvalues_[q_point][0];
                    double rho_out = uvalues_[q_point][1];
                    local_matrix(i,j) += scale
                                         * (
                                           ( flux_1_plus_w(uvalues_[q_point][0],uvalues_[q_point][1],bn)
                                             * state_fe_values[w].value(j,q_point)
                                             + flux_1_plus_r(uvalues_[q_point][0],uvalues_[q_point][1],bn)
                                             * state_fe_values[rho].value(j,q_point)
                                           )
                                           * state_fe_values[w].value(i,q_point)
                                           +
                                           ( -1.*flux_1_minus_w(w_out,rho_out,bn)
                                             * state_fe_values[w].value(j,q_point)
                                             + flux_1_minus_r(w_out,rho_out,bn)
                                             * state_fe_values[rho].value(j,q_point)
                                           )
                                           * state_fe_values[w].value(i,q_point)
                                           +
                                           ( flux_2_plus_w(uvalues_[q_point][0],uvalues_[q_point][1],bn)
                                             * state_fe_values[w].value(j,q_point)
                                             + flux_2_plus_r(uvalues_[q_point][0],uvalues_[q_point][1],bn)
                                             * state_fe_values[rho].value(j,q_point)
                                           )
                                           * state_fe_values[rho].value(i,q_point)
                                           +
                                           ( -1.*flux_2_minus_w(w_out,rho_out,bn)
                                             * state_fe_values[w].value(j,q_point)
                                             + flux_2_minus_r(w_out,rho_out,bn)
                                             * state_fe_values[rho].value(j,q_point)
                                           )
                                           * state_fe_values[rho].value(i,q_point)
                                         )* state_fe_values.JxW(q_point);
                  }
              }
          }
      }
  }
  void
  FaceMatrix(
    const FaceDataContainer<DH, VECTOR, dealdim> &fdc,
    FullMatrix<double> &local_matrix, double scale,
    double/*scale_ico*/)
  {
    unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();
    unsigned int n_q_points = fdc.GetNQPoints();

    const auto &state_fe_values = fdc.GetFEFaceValuesState();

    uvalues_.resize(n_q_points, Vector<double>(2));
    fdc.GetFaceValuesState("last_newton_solution", uvalues_);
    const FEValuesExtractors::Scalar w(0);
    const FEValuesExtractors::Scalar rho(1);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        double bn = state_fe_values.normal_vector(q_point)[0];

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            for (unsigned int j = 0; j < n_dofs_per_element; j++)
              {
                local_matrix(i,j) += scale
                                     * (
                                       (flux_1_plus_w(uvalues_[q_point][0],uvalues_[q_point][1],bn)
                                        * state_fe_values[w].value(j,q_point)
                                        + flux_1_plus_r(uvalues_[q_point][0],uvalues_[q_point][1],bn)
                                        * state_fe_values[rho].value(j,q_point)
                                       ) * state_fe_values[w].value(i,q_point)
                                       +
                                       (flux_2_plus_w(uvalues_[q_point][0],uvalues_[q_point][1],bn)
                                        * state_fe_values[w].value(j,q_point)
                                        + flux_2_plus_r(uvalues_[q_point][0],uvalues_[q_point][1],bn)
                                        * state_fe_values[rho].value(j,q_point)
                                       ) * state_fe_values[rho].value(i,q_point)
                                     )
                                     * state_fe_values.JxW(q_point);
              }
          }
      }
  }

  void
  InterfaceMatrix(
    const FaceDataContainer<DH, VECTOR, dealdim> &fdc,
    FullMatrix<double> &local_matrix, double scale,
    double/*scale_ico*/)
  {
    unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();
    unsigned int n_dofs_per_element_nbr = fdc.GetNbrNDoFsPerElement();
    unsigned int n_q_points = fdc.GetNQPoints();

    const auto &state_fe_values = fdc.GetFEFaceValuesState();
    const auto &state_fe_values_nbr = fdc.GetNbrFEFaceValuesState();

    uvalues_nbr_.resize(n_q_points, Vector<double>(2));
    fdc.GetNbrFaceValuesState("last_newton_solution", uvalues_nbr_);

    const FEValuesExtractors::Scalar w(0);
    const FEValuesExtractors::Scalar rho(1);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        double bn = state_fe_values.normal_vector(q_point)[0];
        SetStab();

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            for (unsigned int j = 0; j < n_dofs_per_element_nbr; j++)
              {
                local_matrix(i,j) += scale
                                     * (
                                       ( flux_1_minus_w(uvalues_nbr_[q_point][0],uvalues_nbr_[q_point][1],bn)
                                         *state_fe_values_nbr[w].value(j,q_point)
                                         +flux_1_minus_r(uvalues_nbr_[q_point][0],uvalues_nbr_[q_point][1],bn)
                                         *state_fe_values_nbr[rho].value(j,q_point)
                                       )
                                       *state_fe_values[w].value(i,q_point)
                                       +
                                       ( flux_2_minus_w(uvalues_nbr_[q_point][0],uvalues_nbr_[q_point][1],bn)
                                         *state_fe_values_nbr[w].value(j,q_point)
                                         +flux_2_minus_r(uvalues_nbr_[q_point][0],uvalues_nbr_[q_point][1],bn)
                                         *state_fe_values_nbr[rho].value(j,q_point)
                                       )
                                       *state_fe_values[rho].value(i,q_point)
                                     )
                                     * state_fe_values.JxW(q_point);

              }
          }
      }
  }

  void
  ElementMatrix(const EDC<DH, VECTOR, dealdim> &edc,
                FullMatrix<double> &local_matrix, double scale,
                double/*scale_ico*/)
  {
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();

    uvalues_.resize(n_q_points, Vector<double>(2));
    edc.GetValuesState("last_newton_solution", uvalues_);
    const FEValuesExtractors::Scalar w(0);
    const FEValuesExtractors::Scalar rho(1);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            for (unsigned int j = 0; j < n_dofs_per_element; j++)
              {
                local_matrix(i,j) -= scale * state_fe_values[w].value(j, q_point)
                                     * state_fe_values[rho].gradient(i,q_point)[0]
                                     * state_fe_values.JxW(q_point);
                local_matrix(i,j) -= scale * (
                                       p_r(uvalues_[q_point][1]) * state_fe_values[rho].value(j, q_point)
                                       + v_w(uvalues_[q_point][0],uvalues_[q_point][1]) * state_fe_values[w].value(j, q_point)
                                       + v_r(uvalues_[q_point][0],uvalues_[q_point][1]) * state_fe_values[rho].value(j, q_point)
                                     )
                                     * state_fe_values[w].gradient(i,q_point)[0]
                                     * state_fe_values.JxW(q_point);
                local_matrix(i,j) += scale *
                                     (gravity_r(uvalues_[q_point][1])
                                      * state_fe_values[rho].value(j,q_point)
                                      + friction_w(uvalues_[q_point][0],uvalues_[q_point][1])
                                      * state_fe_values[w].value(j,q_point)
                                      + friction_r(uvalues_[q_point][0],uvalues_[q_point][1])
                                      * state_fe_values[rho].value(j,q_point)
                                     )
                                     *state_fe_values[w].value(i,q_point)
                                     * state_fe_values.JxW(q_point);
              }
          }
      }
  }

  void
  ElementRightHandSide(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                       dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {

  }

  void
  FaceRightHandSide(
    const FaceDataContainer<DH, VECTOR, dealdim> & /*fdc*/,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
  }

  void
  BoundaryRightHandSide(const FaceDataContainer<DH, VECTOR, dealdim> & /*fdc*/,
                        dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
  }

  void
  ElementTimeEquationExplicit(const ElementDataContainer<DH, VECTOR, dealdim> & /*edc*/,
                              dealii::Vector<double> & /*local_vector*/,
                              double /*scale*/)
  {
    assert(this->problem_type_ == "state");
  }

  void
  ElementTimeEquation(const ElementDataContainer<DH, VECTOR, dealdim> &edc,
                      dealii::Vector<double> &local_vector,
                      double scale)
  {
    assert(this->problem_type_ == "state");

    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();

    uvalues_.resize(n_q_points,Vector<double>(2));

    edc.GetValuesState("last_newton_solution", uvalues_);

    const FEValuesExtractors::Scalar w(0);
    const FEValuesExtractors::Scalar rho(1);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) += scale * (uvalues_[q_point][0] * state_fe_values[w].value(i,q_point))
                               * state_fe_values.JxW(q_point);
            local_vector(i) += scale * (uvalues_[q_point][1] * state_fe_values[rho].value(i,q_point))
                               * state_fe_values.JxW(q_point);
          }
      }
  }

  void
  ElementTimeMatrixExplicit(
    const ElementDataContainer<DH, VECTOR, dealdim> & /*edc*/,
    FullMatrix<double> &/*local_matrix*/)
  {
    assert(this->problem_type_ == "state");
  }

  void
  ElementTimeMatrix(const ElementDataContainer<DH, VECTOR, dealdim> &edc,
                    FullMatrix<double> &local_matrix)
  {
    assert(this->problem_type_ == "state");

    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();

    const FEValuesExtractors::Scalar w(0);
    const FEValuesExtractors::Scalar rho(1);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            for (unsigned int j = 0; j < n_dofs_per_element; j++)
              {
                local_matrix(i,j) += state_fe_values[w].value(j, q_point)
                                     * state_fe_values[w].value(i, q_point)
                                     * state_fe_values.JxW(q_point);
                local_matrix(i,j) += state_fe_values[rho].value(j, q_point)
                                     * state_fe_values[rho].value(i, q_point)
                                     * state_fe_values.JxW(q_point);
              }
          }
      }
  }

  UpdateFlags
  GetUpdateFlags() const
  {
    return update_values | update_gradients
           | update_quadrature_points;
  }

  UpdateFlags
  GetFaceUpdateFlags() const
  {
    return update_values | update_gradients | update_normal_vectors
           | update_quadrature_points;
  }

  unsigned int
  GetControlNBlocks() const
  {
    return 1;
  }

  unsigned int
  GetStateNBlocks() const
  {
    return 1;
  }
  std::vector<unsigned int> &
  GetControlBlockComponent()
  {
    return control_block_component_;
  }
  const std::vector<unsigned int> &
  GetControlBlockComponent() const
  {
    return control_block_component_;
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
  bool
  HasFaces() const
  {
    return true;
  }
  bool
  HasInterfaces() const
  {
    return true;
  }
  template<typename ELEMENTITERATOR>
  bool
  AtInterface(ELEMENTITERATOR &element, unsigned int face) const
  {
    if (element[0]->neighbor_index(face) != -1) //make shure its no boundary
      return true;
    return false;
  }

  static void
  declare_params(ParameterReader &param_reader)
  {
    param_reader.SetSubsection("localpde parameters");
    param_reader.declare_entry("R", "1.", Patterns::Double(0),
                               "Gas constant");
    param_reader.declare_entry("T", "1.", Patterns::Double(0),
                               "Temperature");
    param_reader.declare_entry("alpha", "0.", Patterns::Double(0),
                               "Constant in density presure relation");
    param_reader.declare_entry("lambda", "0.", Patterns::Double(0),
                               "friction coefficient");
    param_reader.declare_entry("D", "1.", Patterns::Double(0),
                               "Diameter");
    param_reader.declare_entry("g", "0.", Patterns::Double(0),
                               "gravity");
    param_reader.declare_entry("hprime", "0.", Patterns::Double(0),
                               "slope");
  }

private:
  vector<Vector<double> > uvalues_;
  vector<Vector<double> > uvalues_nbr_;

  vector<unsigned int> control_block_component_;
  vector<unsigned int> state_block_component_;

  double R_, T_, alpha_, lambda_, D_,g_, hprime_;
  double stab_param_;

  void SetStab()
  {
    //At the moment we take a fixed value.
    stab_param_ = 10.;
  }
  double p(double rho) const
  {
    return R_ * rho * T_ / (1-alpha_ * R_ * rho * T_);
  }
  double p_r(double rho) const
  {
    return R_ * T_ / (1-alpha_ * R_ * rho * T_)
           + alpha_ * R_ * R_  * rho * T_ * T_ /((1-alpha_ * R_ * rho * T_) * (1-alpha_ * R_ * rho * T_));
  }
  double v(double w, double rho) const
  {
    return w * w /rho;
  }
  double v_w(double w, double rho) const
  {
    return 2.*w/rho;
  }
  double v_r(double w, double rho) const
  {
    return - w*w/(rho*rho);
  }
  double flux_1_plus(double w, double rho, double n) const
  {
    return 0.5*( p(rho)*n + v(w,rho)*n + stab_param_ * w );
  }
  double flux_1_minus(double w, double rho, double n) const
  {
    return 0.5*( p(rho)*n + v(w,rho)*n - stab_param_ * w);
  }
  double flux_2_plus(double w, double rho, double n) const
  {
    return 0.5*( w*n + stab_param_ * rho);
  }
  double flux_2_minus(double w, double rho, double n) const
  {
    return 0.5*( w*n - stab_param_ * rho);
  }
  double flux_1_plus_w(double w, double rho, double n) const
  {
    return 0.5*( v_w(w,rho)*n + stab_param_);
  }
  double flux_1_plus_r(double w, double rho, double n) const
  {
    return 0.5*( p_r(rho)*n + v_r(w,rho)*n);
  }
  double flux_1_minus_w(double w, double rho, double n) const
  {
    return 0.5*( v_w(w,rho)*n - stab_param_);
  }
  double flux_1_minus_r(double w, double rho, double n) const
  {
    return 0.5*( p_r(rho)*n + v_r(w,rho)*n);
  }
  double flux_2_plus_w(double /*w*/, double /*rho*/, double n) const
  {
    return 0.5*( n );
  }
  double flux_2_plus_r(double /*w*/, double /*rho*/, double /*n*/) const
  {
    return 0.5*( stab_param_);
  }
  double flux_2_minus_w(double /*w*/, double /*rho*/, double n) const
  {
    return 0.5*( n );
  }
  double flux_2_minus_r(double /*w*/, double /*rho*/, double /*n*/) const
  {
    return 0.5*( -stab_param_);
  }
  double gravity(double rho) const
  {
    return g_*hprime_*rho;
  }
  double gravity_r(double /*rho*/) const
  {
    return g_*hprime_;
  }
  double friction(double w, double rho) const
  {
    return lambda_/(2.*D_) *w * fabs(w/rho);
  }
  double friction_w(double w, double rho) const
  {
    if (w/rho > 0)
      {
        return lambda_/D_ *w/rho;
      }
    else
      {
        return -lambda_/D_*w/rho;
      }
  }
  double friction_r(double w, double rho) const
  {
    if (w/rho > 0)
      {
        return -lambda_/D_ *w*w/(rho*rho);
      }
    else
      {
        return lambda_/D_ *w*w/(rho*rho);
      }
  }
};
//**********************************************************************************

#endif

