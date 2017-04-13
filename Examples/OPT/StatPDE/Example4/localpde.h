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
         template<int, int> class DH, typename VECTOR,  int dealdim>
class LocalPDE : public PDEInterface<EDC, FDC, DH, VECTOR, dealdim>
{
public:
  LocalPDE() :
    control_block_component_(5, 0), state_block_component_(2, 0)
  {
    state_block_component_[1] = 1;
    control_block_component_[1] = 1;
    control_block_component_[2] = 2;
    control_block_component_[3] = 3;
    control_block_component_[4] = 4;
  }

  void
  ElementEquation(const EDC<DH, VECTOR, dealdim> &edc,
                  dealii::Vector<double> &local_vector, double scale,
                  double /*scale_ico*/)
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();
    {
      //Reading data
      assert(this->problem_type_ == "state");
      ugrads_.resize(n_q_points, vector<Tensor<1, dealdim> >(2));
      fvalues_.resize(n_q_points, Vector<double>(2));
      //Geting u
      edc.GetGradsState("last_newton_solution", ugrads_);
    }
    const FEValuesExtractors::Scalar comp_0(0);
    const FEValuesExtractors::Scalar comp_1(1);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        fvalues_[q_point](0) = 20. * M_PI * M_PI
                               * (sin(M_PI * state_fe_values.quadrature_point(q_point)(0))
                                  * sin(M_PI * state_fe_values.quadrature_point(q_point)(1)));
        fvalues_[q_point](1) = 1.;

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) += scale
                               * (ugrads_[q_point][0]
                                  * state_fe_values[comp_0].gradient(i, q_point)
                                  + ugrads_[q_point][1]
                                  * state_fe_values[comp_1].gradient(i, q_point)
                                  - fvalues_[q_point](0)
                                  * state_fe_values[comp_0].value(i, q_point)
                                  - fvalues_[q_point](1)
                                  * state_fe_values[comp_1].value(i, q_point))
                               * state_fe_values.JxW(q_point);
          }
      }
  }

  void
  ElementEquation_U(const EDC<DH, VECTOR, dealdim> &edc,
                    dealii::Vector<double> &local_vector, double scale,
                    double /*scale_ico*/)
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();
    {
      assert(this->problem_type_ == "adjoint");
      zgrads_.resize(n_q_points, vector<Tensor<1, dealdim> >(2));
      //We don't need u so we don't search for state
      edc.GetGradsState("last_newton_solution", zgrads_);
    }

    const FEValuesExtractors::Scalar comp_0(0);
    const FEValuesExtractors::Scalar comp_1(1);
    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) += scale
                               * (zgrads_[q_point][0]
                                  * state_fe_values[comp_0].gradient(i, q_point)
                                  + zgrads_[q_point][1]
                                  * state_fe_values[comp_1].gradient(i, q_point))
                               * state_fe_values.JxW(q_point);
          }
      }
  }

  void
  ElementEquation_UT(const EDC<DH, VECTOR, dealdim> &edc,
                     dealii::Vector<double> &local_vector, double scale,
                     double /*scale_ico*/)
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();
    {
      assert(this->problem_type_ == "tangent");
      dugrads_.resize(n_q_points, vector<Tensor<1, dealdim> >(2));
      edc.GetGradsState("last_newton_solution", dugrads_);
    }
    const FEValuesExtractors::Scalar comp_0(0);
    const FEValuesExtractors::Scalar comp_1(1);
    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) += scale
                               * (dugrads_[q_point][0]
                                  * state_fe_values[comp_0].gradient(i, q_point)
                                  + dugrads_[q_point][1]
                                  * state_fe_values[comp_1].gradient(i, q_point))
                               * state_fe_values.JxW(q_point);
          }
      }
  }

  void
  ElementEquation_UTT(const EDC<DH, VECTOR, dealdim> &edc,
                      dealii::Vector<double> &local_vector, double scale,
                      double /*scale_ico*/)
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();
    {
      assert(this->problem_type_ == "adjoint_hessian");
      dzgrads_.resize(n_q_points, vector<Tensor<1, dealdim> >(2));
      edc.GetGradsState("last_newton_solution", dzgrads_);
    }
    const FEValuesExtractors::Scalar comp_0(0);
    const FEValuesExtractors::Scalar comp_1(1);
    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) += scale
                               * (dzgrads_[q_point][0]
                                  * state_fe_values[comp_0].gradient(i, q_point)
                                  + dzgrads_[q_point][1]
                                  * state_fe_values[comp_1].gradient(i, q_point))
                               * state_fe_values.JxW(q_point);
          }
      }
  }

  void
  ElementEquation_Q(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
                    double /*scale_ico*/)
  {

  }

  void
  ElementEquation_QT(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                     dealii::Vector<double> &/*local_vector*/, double /*scale*/,
                     double /*scale_ico*/)
  {

  }

  void
  ElementEquation_QTT(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                      dealii::Vector<double> &/*local_vector*/, double /*scale*/,
                      double /*scale_ico*/)
  {

  }

  void
  ElementEquation_UU(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                     dealii::Vector<double> &/*local_vector*/, double /*scale*/,
                     double /*scale_ico*/)
  {
    assert(this->problem_type_ == "adjoint_hessian");
  }
  void
  ElementEquation_QU(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                     dealii::Vector<double> &/*local_vector*/, double /*scale*/,
                     double /*scale_ico*/)
  {
    assert(this->problem_type_ == "adjoint_hessian");
  }
  void
  ElementEquation_UQ(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                     dealii::Vector<double> &/*local_vector*/, double /*scale*/,
                     double /*scale_ico*/)
  {
    assert(this->problem_type_ == "hessian");
  }
  void
  ElementEquation_QQ(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                     dealii::Vector<double> &/*local_vector*/, double /*scale*/,
                     double /*scale_ico*/)
  {
    assert(this->problem_type_ == "hessian");
  }

  void
  ElementRightHandSide(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                       dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
    assert(this->problem_type_ == "state");
  }

  void
  ElementMatrix(const EDC<DH, VECTOR, dealdim> &edc,
                FullMatrix<double> &local_matrix, double scale,
                double /*scale_ico*/)
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();

    const FEValuesExtractors::Scalar comp_0(0);
    const FEValuesExtractors::Scalar comp_1(1);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            for (unsigned int j = 0; j < n_dofs_per_element; j++)
              {
                local_matrix(i, j) += scale
                                      * state_fe_values[comp_0].gradient(i, q_point)
                                      * state_fe_values[comp_0].gradient(j, q_point)
                                      * state_fe_values.JxW(q_point);

                local_matrix(i, j) += scale
                                      * state_fe_values[comp_1].gradient(i, q_point)
                                      * state_fe_values[comp_1].gradient(j, q_point)
                                      * state_fe_values.JxW(q_point);
              }
          }
      }
  }

  void
  ControlElementEquation(const EDC<DH, VECTOR, dealdim> &edc,
                         dealii::Vector<double> &local_vector, double scale)
  {
    {
      assert(
        (this->problem_type_ == "gradient")||(this->problem_type_ == "hessian"));
      funcgradvalues_.reinit(GetControlNBlocks());
      edc.GetParamValues("last_newton_solution", funcgradvalues_);
    }

//      for(unsigned int q_point = 0; q_point < n_q_points; q_point++)
//      {
    for (unsigned int i = 0; i < GetControlNBlocks(); i++)
      {
        local_vector(i) += scale * funcgradvalues_(i);
      }
//      }
  }

  void
  ControlElementMatrix(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                       FullMatrix<double> &local_matrix, double scale)
  {
//        for(unsigned int q_point = 0; q_point < n_q_points; q_point++)
//        {
    for (unsigned int i = 0; i < GetControlNBlocks(); i++)
      {
        local_matrix(i, i) += scale * 1.;
      }
//        }
  }

  UpdateFlags
  GetUpdateFlags() const
  {
    return update_values | update_gradients | update_quadrature_points;
  }
  UpdateFlags
  GetFaceUpdateFlags() const
  {
    return update_values | update_quadrature_points;
  }
  unsigned int
  GetControlNBlocks() const
  {
    return 5;
  }
  unsigned int
  GetStateNBlocks() const
  {
    return 2;
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

private:
  Vector<double> qvalues_;
  Vector<double> dqvalues_;
  Vector<double> funcgradvalues_;
  vector<Vector<double> > fvalues_;
  vector<Vector<double> > zvalues_;
  vector<Vector<double> > dzvalues_;

  vector<vector<Tensor<1, dealdim> > > ugrads_;
  vector<vector<Tensor<1, dealdim> > > zgrads_;
  vector<vector<Tensor<1, dealdim> > > dugrads_;
  vector<vector<Tensor<1, dealdim> > > dzgrads_;
  vector<unsigned int> control_block_component_;
  vector<unsigned int> state_block_component_;
};
#endif
