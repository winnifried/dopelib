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
  LocalPDE() :
    state_block_component_(1, 0)
  {
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

    uvalues_.resize(n_q_points);
    edc.GetValuesState("last_newton_solution", uvalues_);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        double b1 = 0.;
        double b2 = 0.;
        {
          double x = state_fe_values.quadrature_point(q_point)[0];
          double y = state_fe_values.quadrature_point(q_point)[1];
          double r = sqrt(x*x+y*y);
          if ( r != 0 )
            {
              b1 = -y/r;
              b2 = x/r;
            }
        }

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) -= scale * uvalues_[q_point]
                               * (b1 * state_fe_values.shape_grad(i,q_point)[0]
                                  + b2 * state_fe_values.shape_grad(i,q_point)[1]
                                 )
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

    uvalues_.resize(n_q_points);
    fdc.GetFaceValuesState("last_newton_solution", uvalues_);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        double bn = 0.;
        {
          double x = state_fe_values.quadrature_point(q_point)[0];
          double y = state_fe_values.quadrature_point(q_point)[1];
          double r = sqrt(x*x+y*y);
          double n1 = state_fe_values.normal_vector(q_point)[0];
          double n2 = state_fe_values.normal_vector(q_point)[1];
          if ( r != 0 )
            bn = (-y*n1+x*n2)/r;
        }

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            if (color == 1 || color == 2)
              {
                assert(bn < 0);
                local_vector(i) += scale
                                   * (ex_sol_.value(state_fe_values.quadrature_point(q_point))
                                      * bn
                                      * state_fe_values.shape_value(i,q_point)
                                     )
                                   * state_fe_values.JxW(q_point);
              }
            else
              {
                assert(bn > 0);
                local_vector(i) += scale
                                   * (uvalues_[q_point]
                                      * bn
                                      * state_fe_values.shape_value(i,q_point)
                                     )
                                   * state_fe_values.JxW(q_point);
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

    uvalues_.resize(n_q_points);
    fdc.GetFaceValuesState("last_newton_solution", uvalues_);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        double bn = 0.;
        {
          double x = state_fe_values.quadrature_point(q_point)[0];
          double y = state_fe_values.quadrature_point(q_point)[1];
          double r = sqrt(x*x+y*y);
          double n1 = state_fe_values.normal_vector(q_point)[0];
          double n2 = state_fe_values.normal_vector(q_point)[1];
          if ( r != 0 )
            bn = (-y*n1+x*n2)/r;
        }

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            if (bn < 0)
              {
                //This is the downstream element, i.e., $(u^-, [\beta n v_h])$ has no
                //inner element coupling
              }
            else
              {
                assert(bn > 0); //This is the upstream element, i.e., u^- = u
                local_vector(i) += scale
                                   * (uvalues_[q_point]
                                      * bn
                                      * state_fe_values.shape_value(i,q_point)
                                     )
                                   * state_fe_values.JxW(q_point);
              }
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

    uvalues_nbr_.resize(n_q_points);
    fdc.GetNbrFaceValuesState("last_newton_solution", uvalues_nbr_);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        double bn = 0.;
        {
          double x = state_fe_values.quadrature_point(q_point)[0];
          double y = state_fe_values.quadrature_point(q_point)[1];
          double r = sqrt(x*x+y*y);
          double n1 = state_fe_values.normal_vector(q_point)[0];
          double n2 = state_fe_values.normal_vector(q_point)[1];
          if ( r != 0 )
            bn = (-y*n1+x*n2)/r;
        }

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            if (bn < 0)
              {
                //This is the downstream element, i.e., u^- is the u from the adjacent element
                local_vector(i) += scale
                                   * (uvalues_nbr_[q_point]
                                      * bn
                                      * state_fe_values.shape_value(i,q_point)
                                     )
                                   * state_fe_values.JxW(q_point);
              }
            else
              {
                assert(bn >= 0); //This is the upstream element, i.e., there is no interelement coupling on this face
              }
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

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        double bn = 0.;
        {
          double x = state_fe_values.quadrature_point(q_point)[0];
          double y = state_fe_values.quadrature_point(q_point)[1];
          double r = sqrt(x*x+y*y);
          double n1 = state_fe_values.normal_vector(q_point)[0];
          double n2 = state_fe_values.normal_vector(q_point)[1];
          if ( r != 0 )
            bn = (-y*n1+x*n2)/r;
        }

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            for (unsigned int j = 0; j < n_dofs_per_element; j++)
              {
                if (color == 1 || color == 2)
                  {
                    assert(bn < 0);
                  }
                else
                  {
                    assert(bn > 0);
                    local_matrix(i,j) += scale * bn * state_fe_values.shape_value(i, q_point)
                                         * state_fe_values.shape_value(j, q_point)
                                         * state_fe_values.JxW(q_point);
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

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        double bn = 0.;
        {
          double x = state_fe_values.quadrature_point(q_point)[0];
          double y = state_fe_values.quadrature_point(q_point)[1];
          double r = sqrt(x*x+y*y);
          double n1 = state_fe_values.normal_vector(q_point)[0];
          double n2 = state_fe_values.normal_vector(q_point)[1];
          if ( r != 0 )
            bn = (-y*n1+x*n2)/r;
        }

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            for (unsigned int j = 0; j < n_dofs_per_element; j++)
              {
                if (bn < 0)
                  {
                    //This is the downstream element, i.e., $(u^-, [\beta n v_h])$ has no
                    //inner element coupling
                  }
                else
                  {
                    assert(bn > 0); //This is the upstream element, i.e., u^- = u
                    local_matrix(i,j) += scale
                                         * (state_fe_values.shape_value(j,q_point)
                                            * bn
                                            * state_fe_values.shape_value(i,q_point)
                                           )
                                         * state_fe_values.JxW(q_point);
                  }
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

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        double bn = 0.;
        {
          double x = state_fe_values.quadrature_point(q_point)[0];
          double y = state_fe_values.quadrature_point(q_point)[1];
          double r = sqrt(x*x+y*y);
          double n1 = state_fe_values.normal_vector(q_point)[0];
          double n2 = state_fe_values.normal_vector(q_point)[1];
          if ( r != 0 )
            bn = (-y*n1+x*n2)/r;
        }

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            for (unsigned int j = 0; j < n_dofs_per_element_nbr; j++)
              {
                if (bn < 0)
                  {
                    //This is the downstream element, i.e., u^- is the u from the adjacent element
                    local_matrix(i,j) += scale
                                         * (state_fe_values_nbr.shape_value(j,q_point)
                                            * bn
                                            * state_fe_values.shape_value(i,q_point)
                                           )
                                         * state_fe_values.JxW(q_point);
                  }
                else
                  {
                    assert(bn >= 0); //This is the upstream element, i.e., there is no interelement coupling on this face
                  }
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

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        double b1 = 0.;
        double b2 = 0.;
        {
          double x = state_fe_values.quadrature_point(q_point)[0];
          double y = state_fe_values.quadrature_point(q_point)[1];
          double r = sqrt(x*x+y*y);
          if ( r != 0 )
            {
              b1 = -y/r;
              b2 = x/r;
            }
        }

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            for (unsigned int j = 0; j < n_dofs_per_element; j++)
              {
                local_matrix(i,j) -= scale * state_fe_values.shape_value(j, q_point)
                                     * (b1 * state_fe_values.shape_grad(i,q_point)[0]
                                        + b2 * state_fe_values.shape_grad(i,q_point)[1]
                                       )
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
  BoundaryRightHandSide(
    const FaceDataContainer<DH, VECTOR, dealdim> & /*fdc*/,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
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
private:

  vector<double> fvalues_;
  vector<double> uvalues_;
  vector<double> uvalues_nbr_;

  vector<Tensor<1, dealdim> > ugrads_;
  vector<unsigned int> state_block_component_;
  ExactSolution ex_sol_;
}
;
//**********************************************************************************

#endif

