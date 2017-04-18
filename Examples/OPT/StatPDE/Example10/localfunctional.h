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

#ifndef LOCALFunctional_
#define LOCALFunctional_

#include <interfaces/pdeinterface.h>

using namespace std;
using namespace dealii;
using namespace DOpE;

template<
template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
         template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
         template<int, int> class DH, typename VECTOR, int dopedim, int dealdim =
         dopedim>
class LocalFunctional : public FunctionalInterface<EDC, FDC, DH, VECTOR,
  dopedim, dealdim>
{
public:

  static void
  declare_params(ParameterReader &param_reader)
  {
    param_reader.SetSubsection("Local PDE parameters");
    param_reader.declare_entry("density_fluid", "0.0", Patterns::Double(0));
    param_reader.declare_entry("viscosity", "0.0", Patterns::Double(0));
  }

  LocalFunctional(ParameterReader &param_reader)
  {
    // Control- and regularization parameters
    mu_regularization = 1.0e+1;

    // Fluid parameters
    param_reader.SetSubsection("Local PDE parameters");
    density_fluid_ = param_reader.get_double("density_fluid");
    viscosity_ = param_reader.get_double("viscosity");
  }

  double AlgebraicValue(const std::map<std::string, const dealii::Vector<double>*> &param_values,
                        const std::map<std::string, const VECTOR *> &/*domain_values*/)
  {
    assert(this->GetProblemType() == "cost_functional");
    //Search the vector with the precomputed functional values
    std::map<std::string, const dealii::Vector<double>*>::const_iterator vals = param_values.find("cost_functional_pre");
    //Return the postprocessed value
    return 0.5*(vals->second->operator[](0)*vals->second->operator[](0));
  }

  double
  BoundaryValue(const FDC<DH, VECTOR, dealdim> &fdc)
  {
    if ( this->GetProblemType() == "cost_functional_pre" ||
         this->GetProblemType() == "cost_functional_pre_tangent" )
      {
        //Same precalculations since drag k(v,p) is linear
        // The value k(v,p) = \int k(v,p) and k'(v,p)\delta u are
        //calculated identically - except that the values are taken
        // from state for k and tangent for k'

        //Precalculation of Drag
        const auto &state_fe_face_values = fdc.GetFEFaceValuesState();
        unsigned int n_q_points = fdc.GetNQPoints();
        unsigned int color = fdc.GetBoundaryIndicator();

        Tensor<1, 2> drag_lift_value;
        drag_lift_value.clear();
        // Asking for boundary color of the cylinder
        if (color == 80)
          {
            ufacevalues_.resize(n_q_points, Vector<double>(3));
            ufacegrads_.resize(n_q_points, vector<Tensor<1, 2> >(3));

            if (this->GetProblemType() == "cost_functional_pre" )
              {
                fdc.GetFaceValuesState("state", ufacevalues_);
                fdc.GetFaceGradsState("state", ufacegrads_);
              }
            else if (this->GetProblemType() == "cost_functional_pre_tangent" )
              {
                fdc.GetFaceValuesState("tangent", ufacevalues_);
                fdc.GetFaceGradsState("tangent", ufacegrads_);
              }
            else
              {
                //This should not happen!
                abort();
              }

            for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
              {
                Tensor<2, 2> pI;
                pI[0][0] = ufacevalues_[q_point](2);
                pI[1][1] = ufacevalues_[q_point](2);

                Tensor<1, 2> v;
                v.clear();
                v[0] = ufacevalues_[q_point](0);
                v[1] = ufacevalues_[q_point](1);

                Tensor<2, 2> grad_v;
                grad_v[0][0] = ufacegrads_[q_point][0][0];
                grad_v[0][1] = ufacegrads_[q_point][0][1];
                grad_v[1][0] = ufacegrads_[q_point][1][0];
                grad_v[1][1] = ufacegrads_[q_point][1][1];

                Tensor<2, 2> cauchy_stress_fluid;
                cauchy_stress_fluid =
                  500.0
                  * (-pI
                     + density_fluid_ * viscosity_
                     * (grad_v + transpose(grad_v)));

                drag_lift_value += cauchy_stress_fluid
                                   * state_fe_face_values.normal_vector(q_point)
                                   * state_fe_face_values.JxW(q_point);
              }

          }
        return drag_lift_value[0];
      }
    else
      {
        assert( this->GetProblemType() == "cost_functional" );
        const auto &state_fe_face_values = fdc.GetFEFaceValuesState();
        unsigned int n_q_points = fdc.GetNQPoints();
        unsigned int color = fdc.GetBoundaryIndicator();
        double functional_value_J = 0;
        // Regularization term for the cost functional
        // defined above
        if (color == 50)
          {
            // Regularization
            qvalues_.reinit(2);
            fdc.GetParamValues("control", qvalues_);

            for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
              {
                functional_value_J += mu_regularization * 0.5
                                      * (qvalues_(0) * qvalues_(0))
                                      * state_fe_face_values.JxW(q_point);
              }

          }
        if (color == 51)
          {
            // Regularization
            qvalues_.reinit(2);
            fdc.GetParamValues("control", qvalues_);

            for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
              {
                functional_value_J += mu_regularization * 0.5
                                      * (qvalues_(1) * qvalues_(1))
                                      * state_fe_face_values.JxW(q_point);
              }

          }
        return functional_value_J;
      }
  }

  void
  BoundaryValue_U(const FDC<DH, VECTOR, dealdim> &fdc,
                  dealii::Vector<double> &local_vector, double scale)
  {
    //Calculation of \partial_u g(k(v,p)) \phi is done as
    // g'(k(v,p)) k(phi) = \int_\gamma g'(k(v,p)) (drag) \ds
    const auto &state_fe_face_values = fdc.GetFEFaceValuesState();
    unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();
    unsigned int n_q_points = fdc.GetNQPoints();
    unsigned int color = fdc.GetBoundaryIndicator();
    if (color == 80)
      {
        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(2);
        dealii::Vector<double> pre_values(1);
        //Search the vector with the precomputed functional values
        fdc.GetParamValues("cost_functional_pre", pre_values);
        // The derivative of the outer function
        double g_prime = pre_values[0];

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            for (unsigned int j = 0; j < n_dofs_per_element; j++)
              {
                const Tensor<2, 2> phi_j_grads_v =
                  state_fe_face_values[velocities].gradient(j, q_point);
                const double phi_j_p = state_fe_face_values[pressure].value(j,
                                                                            q_point);
                Tensor<2, 2> pI_LinP;
                pI_LinP[0][0] = phi_j_p;
                pI_LinP[0][1] = 0.0;
                pI_LinP[1][0] = 0.0;
                pI_LinP[1][1] = phi_j_p;

                Tensor<2, 2> cauchy_stress_fluid;
                cauchy_stress_fluid = -pI_LinP
                                      + density_fluid_ * viscosity_
                                      * (phi_j_grads_v + transpose(phi_j_grads_v));

                Tensor<1, 2> neumann_value = cauchy_stress_fluid
                                             * state_fe_face_values.normal_vector(q_point);

                local_vector(j) += scale * g_prime * neumann_value[0] * 500
                                   * state_fe_face_values.JxW(q_point);
              }
          }

      }
  }

  void
  BoundaryValue_Q(const FDC<DH, VECTOR, dealdim> &fdc,
                  dealii::Vector<double> &local_vector, double scale)
  {
    const auto &state_fe_face_values = fdc.GetFEFaceValuesState();
    unsigned int n_q_points = fdc.GetNQPoints();
    unsigned int color = fdc.GetBoundaryIndicator();

    if (color == 50)
      {
        // Regularization
        qvalues_.reinit(2);
        fdc.GetParamValues("control", qvalues_);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            local_vector(0) += scale * mu_regularization * (qvalues_(0))
                               * state_fe_face_values.JxW(q_point);
          }
      }
    if (color == 51)
      {
        // Regularization
        qvalues_.reinit(2);
        fdc.GetParamValues("control", qvalues_);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            local_vector(1) += scale * mu_regularization * (qvalues_(1))
                               * state_fe_face_values.JxW(q_point);
          }
      }
  }

  void
  BoundaryValue_QQ(const FDC<DH, VECTOR, dealdim> &fdc,
                   dealii::Vector<double> &local_vector, double scale)
  {
    const auto &state_fe_face_values = fdc.GetFEFaceValuesState();
    unsigned int n_q_points = fdc.GetNQPoints();
    unsigned int color = fdc.GetBoundaryIndicator();

    if (color == 50)
      {
        // Regularization
        dqvalues_.reinit(2);
        fdc.GetParamValues("dq", dqvalues_);

        qvalues_.reinit(2);
        fdc.GetParamValues("control", qvalues_);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            local_vector(0) += scale * mu_regularization * (dqvalues_(0))
                               * state_fe_face_values.JxW(q_point);
          }
      }
    if (color == 51)
      {
        // Regularization
        dqvalues_.reinit(2);
        fdc.GetParamValues("dq", dqvalues_);

        qvalues_.reinit(2);
        fdc.GetParamValues("control", qvalues_);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            local_vector(1) += scale * mu_regularization * (dqvalues_(1))
                               * state_fe_face_values.JxW(q_point);
          }
      }
  }

  void
  BoundaryValue_UU(const FDC<DH, VECTOR, dealdim> &fdc,
                   dealii::Vector<double> &local_vector, double scale)
  {
    //Calculation of \partial_{uu} g(k(v,p))(du, tu) is done as
    // \partial_{uu} g(k(v,p)) \du \tu =
    // = g''(k(v,p)) k'(v,p)(du) k'(v,p)(tu) + g'(k(v,p)) k''(v,p)(du,tu)
    //By linearity of k only the first term appears
    //we can get k(v,p) and k'(v,p) du = k(du) as given data
    const auto &state_fe_face_values = fdc.GetFEFaceValuesState();
    unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();
    unsigned int n_q_points = fdc.GetNQPoints();
    unsigned int color = fdc.GetBoundaryIndicator();
    if (color == 80)
      {
        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(2);

        //dealii::Vector<double> pre_values(1);
        dealii::Vector<double> pre_values_tangent(1);
        //Search the vector with the precomputed functional values
        //Values of k not needed since g'' is constant
        //fdc.GetParamValues("cost_functional_pre", pre_values); //k(v,p)
        fdc.GetParamValues("cost_functional_pre_tangent", pre_values_tangent); //k(du)
        // The derivative of the outer function
        double g_primeprime = 1.;
        double k_du = pre_values_tangent[0];


        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            for (unsigned int j = 0; j < n_dofs_per_element; j++)
              {
                const Tensor<2, 2> phi_j_grads_v =
                  state_fe_face_values[velocities].gradient(j, q_point);
                const double phi_j_p = state_fe_face_values[pressure].value(j,
                                                                            q_point);
                Tensor<2, 2> pI_LinP;
                pI_LinP[0][0] = phi_j_p;
                pI_LinP[0][1] = 0.0;
                pI_LinP[1][0] = 0.0;
                pI_LinP[1][1] = phi_j_p;

                Tensor<2, 2> cauchy_stress_fluid;
                cauchy_stress_fluid = -pI_LinP
                                      + density_fluid_ * viscosity_
                                      * (phi_j_grads_v + transpose(phi_j_grads_v));

                Tensor<1, 2> neumann_value = cauchy_stress_fluid
                                             * state_fe_face_values.normal_vector(q_point);

                local_vector(j) += scale * g_primeprime * k_du * neumann_value[0] * 500
                                   * state_fe_face_values.JxW(q_point);
              }
          }

      }

  }

  void
  BoundaryValue_QU(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                   dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {

  }

  void
  BoundaryValue_UQ(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                   dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {

  }

  double
  ElementValue(const EDC<DH, VECTOR, dealdim> & /*edc*/)
  {
    return 0.;
  }

  void
  ElementValue_U(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                 dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {

  }

  void
  ElementValue_Q(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                 dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {

  }

  void
  ElementValue_UU(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                  dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {

  }

  void
  ElementValue_QU(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                  dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
  }

  void
  ElementValue_UQ(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                  dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
  }

  void
  ElementValue_QQ(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                  dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {

  }

  UpdateFlags
  GetFaceUpdateFlags() const
  {
    return update_values | update_quadrature_points | update_gradients
           | update_normal_vectors;
  }

  string
  GetType() const
  {
    //More complicated selection to avoid implementing
    //unneeded derivatives
    if ( this->GetProblemType() == "cost_functional_pre" //Only calculates drag
         ||this->GetProblemType() == "cost_functional_pre_tangent" //Only calculates drag of du
         || this->GetProblemType() == "adjoint" //J'_u is calculated as a boundary integral!
         || this->GetProblemType() == "gradient" //J'_q is calculated as a boundary integral!
         || this->GetProblemType() == "adjoint_hessian" //J'_{uu} is calculated as a boundary integral!
         || this->GetProblemType() == "hessian" //J'_{qq} is calculated as a boundary integral!
       )
      return "boundary";
    else
      {
        if ( this->GetProblemType() == "cost_functional" )
          {
            return "boundary algebraic";
          }
        else
          {
            std::cout<<"Unknown type ,,"<<this->GetProblemType()<<"'' in LocalFunctional::GetType"<<std::endl;
            abort();
          }
      }
  }

  string
  GetName() const
  {
    return "cost functional";
  }

  unsigned int NeedPrecomputations() const
  {
    return 1;
  }
private:
  Vector<double> qvalues_;
  Vector<double> dqvalues_;
  vector<Vector<double> > ufacevalues_;
  vector<Vector<double> > dufacevalues_;

  vector<vector<Tensor<1, dealdim> > > ufacegrads_;
  vector<vector<Tensor<1, dealdim> > > dufacegrads_;


  // Fluid parameters
  double density_fluid_, viscosity_;

  // Controlregularization parameters
  double mu_regularization;

};
#endif
