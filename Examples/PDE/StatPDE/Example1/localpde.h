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

using namespace DOpE;

/**
 * This class describes elementwise the weak formulation of the PDE.
 * See pdeinterface.h for more information.
 */
template<
template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
         template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
         template<int, int> class DH, typename VECTOR, int dealdim>
class LocalPDE : public PDEInterface<EDC, FDC, DH, VECTOR, dealdim>
{
public:
  LocalPDE() :
    state_block_component_(3, 0)
  {
    assert(dealdim==2);
    //The solution has dealdim+1 components, and we
    //want to group the components 0,..,dealdim into
    //block zero and the component dealdim+1 into the block 1.
    state_block_component_[2] = 1;
  }

  /**
   * This describes the weak formulation on a element, i.e. the
   * weak formulation of the Stokes equation.
   */
  void
  ElementEquation(const EDC<DH, VECTOR, dealdim> &edc,
                  dealii::Vector<double> &local_vector, double scale,
                  double scale_ico)
  {
    //Get the number of dofs, the number of quad points as
    //well as the finite element values on this element from the edc.
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    const unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    const unsigned int n_q_points = edc.GetNQPoints();
    //This should only get called if the problem type is state.
    assert(this->problem_type_ == "state");

    //Resize uvalues and ugrads properly. These will hold the
    //solution of the last newton iteration.
    u_values_.resize(n_q_points, Vector<double>(3));
    ugrads_.resize(n_q_points, std::vector<Tensor<1, 2> >(3));

    edc.GetValuesState("last_newton_solution", u_values_);
    edc.GetGradsState("last_newton_solution", ugrads_);

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(2);

    //Now loop over all the quadpoints
    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        //An abbreviations to declatter the weak formulation.
        Tensor<2, 2> vgrads;
        vgrads.clear();
        vgrads[0][0] = ugrads_[q_point][0][0];
        vgrads[0][1] = ugrads_[q_point][0][1];
        vgrads[1][0] = ugrads_[q_point][1][0];
        vgrads[1][1] = ugrads_[q_point][1][1];

        double press = u_values_[q_point](2);
        double incompressibility = vgrads[0][0] + vgrads[1][1];

        //loop over all degrees of freedom
        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            //Again abbreviations.
            const Tensor<2, 2> phi_i_grads_v =
              state_fe_values[velocities].gradient(i, q_point);
            const double phi_i_p = state_fe_values[pressure].value(i, q_point);
            const double div_phi_v = state_fe_values[velocities].divergence(i,
                                     q_point);

            //Define the weak formulation. scale_ico makes only
            //sense in the nonstationary context (treating those
            //terms fully implicitly). In a stationary
            //problem it holds scale_ico = scale.
            local_vector(i) += scale
                               * (0.5 * scalar_product(vgrads, phi_i_grads_v)
                                  + 0.5 * scalar_product(transpose(vgrads), phi_i_grads_v))
                               * state_fe_values.JxW(q_point);

            local_vector(i) += scale_ico
                               * (-1. * press * div_phi_v + incompressibility * phi_i_p)
                               * state_fe_values.JxW(q_point);
          }
      }

  }

  void
  ElementMatrix(const EDC<DH, VECTOR, dealdim> &edc,
                FullMatrix<double> &local_matrix, double scale,
                double scale_ico)
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    const unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    const unsigned int n_q_points = edc.GetNQPoints();

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(2);

    std::vector<Tensor<1, 2> > phi_v(n_dofs_per_element);
    std::vector<Tensor<2, 2> > phi_grads_v(n_dofs_per_element);
    std::vector<double> phi_p(n_dofs_per_element);
    std::vector<double> div_phi_v(n_dofs_per_element);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        for (unsigned int k = 0; k < n_dofs_per_element; k++)
          {
            phi_v[k] = state_fe_values[velocities].value(k, q_point);
            phi_grads_v[k] = state_fe_values[velocities].gradient(k, q_point);
            phi_p[k] = state_fe_values[pressure].value(k, q_point);
            div_phi_v[k] = state_fe_values[velocities].divergence(k, q_point);
          }

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            for (unsigned int j = 0; j < n_dofs_per_element; j++)
              {
                local_matrix(i, j) += scale
                                      * (0.5 * scalar_product(phi_grads_v[j], phi_grads_v[i])
                                         + 0.5
                                         * scalar_product(transpose(phi_grads_v[j]),
                                                          phi_grads_v[i])) * state_fe_values.JxW(q_point);
                local_matrix(i, j) +=
                  scale_ico
                  * (-phi_p[j] * div_phi_v[i]
                     + (phi_grads_v[j][0][0] + phi_grads_v[j][1][1])
                     * phi_p[i]) * state_fe_values.JxW(q_point);
              }
          }
      }

  }

  /**
   * Describes the value of the rhs on a element, i.e. the term (f,phi).
   * As we have f=0 in our example, this method is empty.
   */
  void
  ElementRightHandSide(const EDC<DH, VECTOR, dealdim> &
                       /*edc*/, dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {

  }

  /**
   * This describes the weak formulation on a part of the boundary.
   * We need to specify this here as we use the symmetrized gradient
   * for Stokes equation together with a free outflow condition.
   */
  void
  BoundaryEquation(const FDC<DH, VECTOR, dealdim> &fdc,
                   dealii::Vector<double> &local_vector, double scale,
                   double /*scale_ico*/)
  {
    //auto = FEValues
    const auto &state_fe_face_values = fdc.GetFEFaceValuesState();
    const unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();
    const unsigned int n_q_points = fdc.GetNQPoints();
    const unsigned int color = fdc.GetBoundaryIndicator();

    assert(this->problem_type_ == "state");

    //Do-nothing condition applied on the outflow boundary.
    //The latter has boundary color 1 in this example.
    if (color == 1)
      {
        ufacegrads_.resize(n_q_points, std::vector<Tensor<1, 2> >(3));

        fdc.GetFaceGradsState("last_newton_solution", ufacegrads_);

        const FEValuesExtractors::Vector velocities(0);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            Tensor<2, 2> vgrads;
            vgrads.clear();
            vgrads[0][0] = ufacegrads_[q_point][0][0];
            vgrads[0][1] = ufacegrads_[q_point][0][1];
            vgrads[1][0] = ufacegrads_[q_point][1][0];
            vgrads[1][1] = ufacegrads_[q_point][1][1];

            for (unsigned int i = 0; i < n_dofs_per_element; i++)
              {
                const Tensor<1, 2> phi_i_v =
                  state_fe_face_values[velocities].value(i, q_point);

                const Tensor<1, 2> neumann_value = (transpose(vgrads)
                                                    * state_fe_face_values.normal_vector(q_point));

                local_vector(i) += -scale * 0.5 * neumann_value * phi_i_v
                                   * state_fe_face_values.JxW(q_point);
              }
          }
      }

  }

  /**
   * The matrix term corresponding to the above defined boundaryequation.
   */
  void
  BoundaryMatrix(const FDC<DH, VECTOR, dealdim> &fdc,
                 dealii::FullMatrix<double> &local_matrix, double scale,
                 double /*scale_ico*/)
  {
    const auto &state_fe_face_values = fdc.GetFEFaceValuesState();
    const unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();
    const unsigned int n_q_points = fdc.GetNQPoints();
    const unsigned int color = fdc.GetBoundaryIndicator();
    assert(this->problem_type_ == "state");

    // do-nothing applied on outflow boundary
    if (color == 1)
      {
        const FEValuesExtractors::Vector velocities(0);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            for (unsigned int i = 0; i < n_dofs_per_element; i++)
              {
                const Tensor<1, 2> phi_i_v =
                  state_fe_face_values[velocities].value(i, q_point);

                for (unsigned int j = 0; j < n_dofs_per_element; j++)
                  {
                    const Tensor<2, 2> phi_j_grads_v =
                      state_fe_face_values[velocities].gradient(j, q_point);

                    const Tensor<1, 2> neumann_value = (transpose(phi_j_grads_v)
                                                        * state_fe_face_values.normal_vector(q_point));

                    local_matrix(i, j) += -scale * 0.5 * neumann_value
                                          * phi_i_v * state_fe_face_values.JxW(q_point);
                  }
              }
          }
      }

  }

  /**
   * Describes the value of the rhs on a part of the boundary, i.e. the term
   * (f,phi)_\partial\Omega. As we have f=0 in our example, this method is empty.
   */
  void
  BoundaryRightHandSide(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                        dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
  }

  /**
   * Returns the update flags the FEValues.
   */
  UpdateFlags
  GetUpdateFlags() const
  {
    return update_values | update_gradients | update_quadrature_points;
  }

  /**
   * Returns the update flags the FEFaceValues.
   */
  UpdateFlags
  GetFaceUpdateFlags() const
  {
    return update_values | update_gradients | update_normal_vectors
           | update_quadrature_points;
  }

  /**
   * Returns the number of blocks. We have two, namely velocity and pressure.
   */
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
  std::vector<Vector<double> > u_values_;
  std::vector<std::vector<Tensor<1, dealdim> > > ugrads_;
  std::vector<std::vector<Tensor<1, dealdim> > > ufacegrads_;

  std::vector<unsigned int> state_block_component_;
};
#endif
