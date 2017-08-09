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

#include "localnetwork.h"
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
  LocalPDE(ParameterReader &param_reader, const LocalNetwork &net) :
    state_block_component_(2, 0), network_(net)
  {
    param_reader.SetSubsection("localpde parameters");
    win_    = param_reader.get_double("win");
    pin_  = param_reader.get_double("pin");

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
    const FEValuesExtractors::Scalar p(1);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) -= scale * (uvalues_[q_point][0] * state_fe_values[w].gradient(i,q_point)[0])
                               * state_fe_values.JxW(q_point);
            local_vector(i) += scale * uvalues_[q_point][1]
                               * state_fe_values[p].gradient(i,q_point)[0]
                               * state_fe_values.JxW(q_point);
            local_vector(i) -= scale * (1. * state_fe_values[w].value(i,q_point))
                               * state_fe_values.JxW(q_point);
            local_vector(i) -= scale * (2. * state_fe_values[p].value(i,q_point))
                               * state_fe_values.JxW(q_point);
          }
      }
  }

  void
  BoundaryEquation(
    const FDC<DH, VECTOR, dealdim> &fdc,
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
    uflux_.resize(n_q_points, Vector<double>(2));
    fdc.GetFluxValues("last_newton_solution", uflux_);

    const FEValuesExtractors::Scalar w(0);
    const FEValuesExtractors::Scalar p(1);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        double bn = state_fe_values.normal_vector(q_point)[0];
        SetStab(fdc.GetElementDiameter());
        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            if (color == 0 )
              {
                assert(bn < 0);
                double w_in = uflux_[q_point][0];
                double p_in = uvalues_[q_point][1];
                local_vector(i) += scale
                                   * ((flux_1_plus(uvalues_[q_point][0],bn)
                                       + flux_1_minus(w_in,bn)
                                      ) * state_fe_values[w].value(i,q_point)
                                      + (flux_2_plus(uvalues_[q_point][1],bn)
                                         + flux_2_minus(p_in,bn)
                                        ) * state_fe_values[p].value(i,q_point)
                                     )
                                   * state_fe_values.JxW(q_point);
              }
            else
              {
                assert(bn > 0);
                double w_out = uvalues_[q_point][0];
                double p_out = uflux_[q_point][1];
                local_vector(i) += scale
                                   * ((flux_1_plus(uvalues_[q_point][0],bn)
                                       + flux_1_minus(w_out,bn)
                                      ) * state_fe_values[w].value(i,q_point)
                                      + (flux_2_plus(uvalues_[q_point][1],bn)
                                         + flux_2_minus(p_out,bn)
                                        ) * state_fe_values[p].value(i,q_point)
                                     )* state_fe_values.JxW(q_point);
              }
          }
      }
  }

  void
  FaceEquation(
    const FDC<DH, VECTOR, dealdim> &fdc,
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
    const FEValuesExtractors::Scalar p(1);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        double bn = state_fe_values.normal_vector(q_point)[0];
        SetStab(fdc.GetElementDiameter());

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) += scale
                               * (flux_1_plus(uvalues_[q_point][0],bn)
                                  *state_fe_values[w].value(i,q_point)
                                  +flux_2_plus(uvalues_[q_point][1],bn)
                                  *state_fe_values[p].value(i,q_point)
                                 )
                               * state_fe_values.JxW(q_point);
          }
      }
  }
  void
  InterfaceEquation(
    const FDC<DH, VECTOR, dealdim> &fdc,
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
    const FEValuesExtractors::Scalar p(1);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        double bn = state_fe_values.normal_vector(q_point)[0];
        SetStab(fdc.GetElementDiameter());

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) += scale
                               * (
                                 flux_1_minus(uvalues_nbr_[q_point][0],bn)
                                 *state_fe_values[w].value(i,q_point)
                                 +
                                 flux_2_minus(uvalues_nbr_[q_point][1],bn)
                                 *state_fe_values[p].value(i,q_point)
                               )
                               * state_fe_values.JxW(q_point);
          }
      }
  }

  void
  BoundaryMatrix(
    const FDC<DH, VECTOR, dealdim> &fdc,
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
    const FEValuesExtractors::Scalar p(1);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        double bn = state_fe_values.normal_vector(q_point)[0];
        SetStab(fdc.GetElementDiameter());

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            for (unsigned int j = 0; j < n_dofs_per_element; j++)
              {
                if (color == 0)
                  {
                    assert(bn < 0);
                    local_matrix(i,j) += scale
                                         * (
                                           (flux_1_plus_w(uvalues_[q_point][0],bn)
                                            * state_fe_values[w].value(j,q_point)
                                           )
                                           * state_fe_values[w].value(i,q_point)
                                           +
                                           (flux_2_plus_r(uvalues_[q_point][1],bn)
                                            * state_fe_values[p].value(j,q_point)
                                            +flux_2_minus_r(uvalues_[q_point][1],bn)
                                            * state_fe_values[p].value(j,q_point)
                                           )
                                           * state_fe_values[p].value(i,q_point)
                                         )
                                         * state_fe_values.JxW(q_point);
                  }
                else
                  {
                    assert(bn > 0);
                    local_matrix(i,j) += scale
                                         * (
                                           ( flux_1_plus_w(uvalues_[q_point][0],bn)
                                             * state_fe_values[w].value(j,q_point)
                                             +
                                             flux_1_minus_w(uvalues_[q_point][0],bn)
                                             * state_fe_values[w].value(j,q_point)
                                           )
                                           * state_fe_values[w].value(i,q_point)
                                           +
                                           ( flux_2_plus_r(uvalues_[q_point][1],bn)
                                             * state_fe_values[p].value(j,q_point)
                                           )
                                           * state_fe_values[p].value(i,q_point)
                                         )* state_fe_values.JxW(q_point);
                  }
              }
          }
      }
  }
  void
  FaceMatrix(
    const FDC<DH, VECTOR, dealdim> &fdc,
    FullMatrix<double> &local_matrix, double scale,
    double/*scale_ico*/)
  {
    unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();
    unsigned int n_q_points = fdc.GetNQPoints();

    const auto &state_fe_values = fdc.GetFEFaceValuesState();

    uvalues_.resize(n_q_points, Vector<double>(2));
    fdc.GetFaceValuesState("last_newton_solution", uvalues_);
    const FEValuesExtractors::Scalar w(0);
    const FEValuesExtractors::Scalar p(1);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        double bn = state_fe_values.normal_vector(q_point)[0];

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            for (unsigned int j = 0; j < n_dofs_per_element; j++)
              {
                local_matrix(i,j) += scale
                                     * (
                                       (flux_1_plus_w(uvalues_[q_point][0],bn)
                                        * state_fe_values[w].value(j,q_point)
                                       ) * state_fe_values[w].value(i,q_point)
                                       +
                                       (flux_2_plus_r(uvalues_[q_point][1],bn)
                                        * state_fe_values[p].value(j,q_point)
                                       ) * state_fe_values[p].value(i,q_point)
                                     )
                                     * state_fe_values.JxW(q_point);
              }
          }
      }
  }

  void
  InterfaceMatrix(
    const FDC<DH, VECTOR, dealdim> &fdc,
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
    const FEValuesExtractors::Scalar p(1);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        double bn = state_fe_values.normal_vector(q_point)[0];
        SetStab(fdc.GetElementDiameter());

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            for (unsigned int j = 0; j < n_dofs_per_element_nbr; j++)
              {
                local_matrix(i,j) += scale
                                     * (
                                       ( flux_1_minus_w(uvalues_nbr_[q_point][0],bn)
                                         *state_fe_values_nbr[w].value(j,q_point)
                                       )
                                       *state_fe_values[w].value(i,q_point)
                                       +
                                       ( flux_2_minus_r(uvalues_nbr_[q_point][1],bn)
                                         *state_fe_values_nbr[p].value(j,q_point)
                                       )
                                       *state_fe_values[p].value(i,q_point)
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
    const FEValuesExtractors::Scalar p(1);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            for (unsigned int j = 0; j < n_dofs_per_element; j++)
              {
                local_matrix(i,j) -= scale * state_fe_values[w].value(j, q_point)
                                     * state_fe_values[w].gradient(i,q_point)[0]
                                     * state_fe_values.JxW(q_point);
                local_matrix(i,j) += scale *
                                     state_fe_values[p].value(j, q_point)
                                     * state_fe_values[p].gradient(i,q_point)[0]
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
    const FDC<DH, VECTOR, dealdim> & /*fdc*/,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
  }

  void
  BoundaryRightHandSide(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                        dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
  }

  void
  ElementTimeEquationExplicit(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                              dealii::Vector<double> & /*local_vector*/,
                              double /*scale*/)
  {
    assert(this->problem_type_ == "state");
  }

  void
  ElementTimeEquation(const EDC<DH, VECTOR, dealdim> &edc,
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
    const FEValuesExtractors::Scalar p(1);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            //local_vector(i) += scale * (uvalues_[q_point][0] * state_fe_values[w].value(i,q_point))
            //  * state_fe_values.JxW(q_point);
            local_vector(i) += scale * (uvalues_[q_point][1] * state_fe_values[p].value(i,q_point))
                               * state_fe_values.JxW(q_point);
          }
      }
  }

  void
  ElementTimeMatrixExplicit(
    const EDC<DH, VECTOR, dealdim> & /*edc*/,
    FullMatrix<double> &/*local_matrix*/)
  {
    assert(this->problem_type_ == "state");
  }

  void
  ElementTimeMatrix(const EDC<DH, VECTOR, dealdim> &edc,
                    FullMatrix<double> &local_matrix)
  {
    assert(this->problem_type_ == "state");

    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();

    const FEValuesExtractors::Scalar w(0);
    const FEValuesExtractors::Scalar p(1);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            for (unsigned int j = 0; j < n_dofs_per_element; j++)
              {
                //local_matrix(i,j) += state_fe_values[w].value(j, q_point)
                //* state_fe_values[w].value(i, q_point)
                //* state_fe_values.JxW(q_point);
                local_matrix(i,j) += state_fe_values[p].value(j, q_point)
                                     * state_fe_values[p].value(i, q_point)
                                     * state_fe_values.JxW(q_point);
              }
          }
      }
  }

  /*******************Special Methods on Pipes************************************/
  void BoundaryEquation_BV(const FDC<DH, VECTOR, dealdim> &fdc,
                           dealii::Vector<double> &local_vector,
                           double scale,
                           double /*scale_ico*/)
  {
    assert(local_vector.size()==4);
    unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();
    unsigned int n_q_points = fdc.GetNQPoints();
    assert(n_q_points == 1);
    unsigned int color = fdc.GetBoundaryIndicator();
    const auto &state_fe_values =
      fdc.GetFEFaceValuesState();

    uflux_.resize(n_q_points, Vector<double>(2));
    fdc.GetFluxValues("state", uflux_);

    const FEValuesExtractors::Scalar w(0);
    const FEValuesExtractors::Scalar p(1);

    for (unsigned int i = 0; i < n_dofs_per_element; i++)
      {
        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            double bn = state_fe_values.normal_vector(q_point)[0];
            SetStab(fdc.GetElementDiameter());
            if (color == 0 )
              {
                assert(bn < 0);
                //inflow boundary
                double w_in = uflux_[q_point][0];

                local_vector(0) += scale
                                   * ((flux_1_minus_w(w_in,bn)
                                      ) * state_fe_values[w].value(i,q_point)
                                     )
                                   * state_fe_values.JxW(q_point);
              }
            else
              {
                assert(bn > 0);
                double p_in = uflux_[q_point][1];
                local_vector(2+1) += scale
                                     * ((flux_2_minus_r(p_in,bn)
                                        ) * state_fe_values[p].value(i,q_point)
                                       )
                                     * state_fe_values.JxW(q_point);
              }
          }
      }
  }


  void BoundaryMatrix_BV(const FDC<DH, VECTOR, dealdim> &fdc,
                         std::vector<bool> & /*present_in_outflow*/,
                         dealii::FullMatrix<double> &local_matrix,
                         double scale,
                         double /*scale_ico*/)
  {
    assert(local_matrix.m()==4);
    assert(local_matrix.n()==4);//Should be a square matrix of dimension n_comp*2 x n_comp*2

    unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();
    unsigned int n_q_points = fdc.GetNQPoints();
    assert(n_q_points == 1);
    unsigned int color = fdc.GetBoundaryIndicator();
    const auto &state_fe_values =
      fdc.GetFEFaceValuesState();

    if (this->problem_type_ == "state")
      {
        uflux_.resize(n_q_points, Vector<double>(2));
        fdc.GetFluxValues("last_newton_solution", uflux_);
      }
    else
      {
        uflux_.resize(n_q_points, Vector<double>(2));
        fdc.GetFluxValues("state", uflux_);
      }

    const FEValuesExtractors::Scalar w(0);
    const FEValuesExtractors::Scalar p(1);

    for (unsigned int i = 0; i < n_dofs_per_element; i++)
      {
        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            double bn = state_fe_values.normal_vector(q_point)[0];
            SetStab(fdc.GetElementDiameter());
            if (color == 0 )
              {
                assert(bn < 0);
                //inflow boundary
                double w_in = uflux_[q_point][0];//uvalues_[q_point][0];

                local_matrix(0,0) += scale
                                     * ((flux_1_minus_w(w_in,bn)
                                        ) * state_fe_values[w].value(i,q_point)
                                       )
                                     * state_fe_values.JxW(q_point);
              }
            else
              {
                double p_in = uflux_[q_point][1];
                assert(bn > 0);
                local_matrix(2+1,2+1) += scale
                                         * ((flux_2_minus_r(p_in,bn)
                                            ) * state_fe_values[p].value(i,q_point)
                                           )
                                         * state_fe_values.JxW(q_point);
              }
          }
      }
  }

  void OutflowValues(const FDC<DH, VECTOR, dealdim> &fdc,
                     std::vector<bool> &present_in_outflow,
                     dealii::Vector<double> &local_vector,
                     double /*scale*/,
                     double /*scale_ico*/)
  {
    assert(local_vector.size()==8);
    //Values in local_vector
    // n_comp left_outflow, n_comp right_outflow,
    // n_comp left_direct_coupling, n_comp right_direct_coupling
    //Here only the first four are relevant.
    unsigned int n_q_points = fdc.GetNQPoints();
    unsigned int color = fdc.GetBoundaryIndicator();

    assert(this->problem_type_ == "state");

    uvalues_.resize(n_q_points, Vector<double>(2));
    fdc.GetFaceValuesState("last_newton_solution", uvalues_);
    uflux_.resize(n_q_points, Vector<double>(2));
    fdc.GetFluxValues("last_newton_solution", uflux_);

    const FEValuesExtractors::Scalar w(0);
    const FEValuesExtractors::Scalar p(1);

    assert(n_q_points == 1);
    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        //At present fixed flow-direction!
        //Outflow is color ==1
        if (color == 1 )
          {
            assert(bn > 0);
            //Rightside of pipe -> Index n_comp+current_comp
            local_vector(2) += uvalues_[q_point][0];
            present_in_outflow[2] = true;
          }
        else
          {
            assert(bn < 0);
            assert(color == 0);
            //Leftside of pipe  -> current_comp
            local_vector(1) += uvalues_[q_point][1];
            present_in_outflow[1] = true;
          }
      }
  }

  void OutflowMatrix(const FDC<DH, VECTOR, dealdim> &fdc,
                     std::vector<bool> &present_in_outflow,
                     dealii::FullMatrix<double> &local_matrix,
                     double /*scale*/,
                     double /*scale_ico*/)
  {
    assert(local_matrix.m()==4);
    assert(local_matrix.n()==4);//Should be a square matrix of dimension n_comp*2 x n_comp*2

    unsigned int n_q_points = fdc.GetNQPoints();
    unsigned int color = fdc.GetBoundaryIndicator();

    if (this->problem_type_ == "state")
      {
        uvalues_.resize(n_q_points, Vector<double>(2));
        fdc.GetFaceValuesState("last_newton_solution", uvalues_);
      }
    else
      {
        uvalues_.resize(n_q_points, Vector<double>(2));
        fdc.GetFaceValuesState("state", uvalues_);
      }
    const FEValuesExtractors::Scalar w(0);
    const FEValuesExtractors::Scalar p(1);

    assert(n_q_points == 1);
    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        //At present fixed flow-direction!
        //Sort in matrix: first index is component!
        //                second index is component on left boundary and n_comp+comp on right
        //                boundary
        if (color == 1 )
          {
            assert(uvalues_[q_point][0] > 0. || fabs(uvalues_[q_point][0]) < 1.e-13);
            assert(bn > 0);
            local_matrix(0,2+0) += 1.;
            present_in_outflow[2+0] = true; //On right boundary always n_comp+c
          }
        else
          {
            assert(uvalues_[q_point][0] > 0. || fabs(uvalues_[q_point][0]) < 1.e-13);
            assert(bn < 0);
            assert(color == 0);
            local_matrix(0+1,0+1) += 1.;
            present_in_outflow[0+1] = true;
          }
      }
  }

  void PipeCouplingResidual(dealii::Vector<double> &res,
                            const dealii::Vector<double> &u,
                            const std::vector<bool> &present_in_outflow)
  {
    network_.PipeCouplingResidual(res,u,present_in_outflow);
  }
  void CouplingMatrix(dealii::SparseMatrix<double> &matrix,
                      const std::vector<bool> &present_in_outflow)
  {
    network_.CouplingMatrix(matrix,present_in_outflow);
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
    param_reader.declare_entry("win", "1.", Patterns::Double(0),
                               "inflow flux");
    param_reader.declare_entry("pin", "1.", Patterns::Double(0),
                               "outflow flux");
  }

private:
  vector<Vector<double> > uvalues_;
  vector<Vector<double> > uflux_;
  vector<Vector<double> > uvalues_nbr_;

  vector<unsigned int> control_block_component_;
  vector<unsigned int> state_block_component_;

  const LocalNetwork &network_;

  double win_, pin_;
  double stab_param_;

  void SetStab(double h)
  {
    //At the moment we take a fixed value.
    stab_param_ = 0.0001 * h;
  }
  double flux_1_plus(double w, double n) const
  {
    return 0.5*( w*n + stab_param_ * w );
  }
  double flux_1_minus(double w, double n) const
  {
    return 0.5*( w*n - stab_param_ * w);
  }
  double flux_2_plus(double p, double n) const
  {
    return 0.5*( -p*n + stab_param_ * p);
  }
  double flux_2_minus(double p, double n) const
  {
    return 0.5*( -p*n - stab_param_ * p);
  }
  double flux_1_plus_w(double /*w*/, double n) const
  {
    return 0.5* (n+stab_param_);
  }

  double flux_1_minus_w(double /*w*/, double n) const
  {
    return 0.5*(n-stab_param_);
  }
  double flux_2_plus_r(double /*p*/, double n) const
  {
    return 0.5*(stab_param_-n);
  }
  double flux_2_minus_r(double /*p*/, double n) const
  {
    return -0.5*(n+stab_param_);
  }
};
//**********************************************************************************

#endif

