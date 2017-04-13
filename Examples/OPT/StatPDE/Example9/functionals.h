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

#ifndef LOCALFunctionalS_
#define LOCALFunctionalS_

#include <interfaces/pdeinterface.h>
//#include <interfaces/functionalinterface.h>

using namespace std;
using namespace dealii;
using namespace DOpE;

template<
template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
         template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
         template<int, int> class DH, typename VECTOR, int dopedim, int dealdim =
         dopedim>
class LocalPointFunctionalDeflectionX : public FunctionalInterface<EDC, FDC,
  DH, VECTOR, dopedim, dealdim>
{
public:

  double
  PointValue(
    const DOpEWrapper::DoFHandler<dopedim, DH> & /*control_dof_handler*/,
    const DOpEWrapper::DoFHandler<dealdim, DH> &state_dof_handler,
    const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
    const std::map<std::string, const VECTOR *> &domain_values)
  {
    Point<2> p1(0.6, 0.2);

    typename map<string, const VECTOR *>::const_iterator it =
      domain_values.find("state");
    Vector<double> tmp_vector(5);

    VectorTools::point_value(state_dof_handler, *(it->second), p1,
                             tmp_vector);
    double x = tmp_vector(2);

    // Deflection X
    return x;

  }

  string
  GetType() const
  {
    return "point";
  }
  string
  GetName() const
  {
    return "Deflection_X";
  }

};

/****************************************************************************************/
template<
template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
         template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
         template<int, int> class DH, typename VECTOR, int dopedim, int dealdim =
         dopedim>
class LocalPointFunctionalDeflectionY : public FunctionalInterface<EDC, FDC,
  DH, VECTOR, dopedim, dealdim>
{
public:

  double
  PointValue(
    const DOpEWrapper::DoFHandler<dopedim, DH> & /*control_dof_handler*/,
    const DOpEWrapper::DoFHandler<dealdim, DH> &state_dof_handler,
    const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
    const std::map<std::string, const VECTOR *> &domain_values)
  {
    Point<2> p1(0.6, 0.2);

    typename map<string, const VECTOR *>::const_iterator it =
      domain_values.find("state");
    Vector<double> tmp_vector(5);

    VectorTools::point_value(state_dof_handler, *(it->second), p1,
                             tmp_vector);
    double y = tmp_vector(3);

    // Delfection Y
    return y;

  }

  string
  GetType() const
  {
    return "point";
  }
  string
  GetName() const
  {
    return "Deflection_Y";
  }

};

// drag
/****************************************************************************************/
template<
template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
         template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
         template<int, int> class DH, typename VECTOR, int dopedim, int dealdim =
         dopedim>
class LocalBoundaryFaceFunctionalDrag : public FunctionalInterface<EDC, FDC,
  DH, VECTOR, dopedim, dealdim>
{
public:
  static void
  declare_params(ParameterReader &param_reader)
  {
    param_reader.SetSubsection("Local PDE parameters");
    param_reader.declare_entry("density_fluid", "0.0", Patterns::Double(0));
    param_reader.declare_entry("viscosity", "0.0", Patterns::Double(0));
    param_reader.declare_entry("mu", "0.0", Patterns::Double(0));
    param_reader.declare_entry("poisson_ratio_nu", "0.0",
                               Patterns::Double(0));

  }

  LocalBoundaryFaceFunctionalDrag(ParameterReader &param_reader)
  {
    param_reader.SetSubsection("Local PDE parameters");
    density_fluid_ = param_reader.get_double("density_fluid");
    viscosity_ = param_reader.get_double("viscosity");
    lame_coefficient_mu_ = param_reader.get_double("mu");
    poisson_ratio_nu_ = param_reader.get_double("poisson_ratio_nu");

    lame_coefficient_lambda_ =
      (2 * poisson_ratio_nu_ * lame_coefficient_mu_)
      / (1.0 - 2 * poisson_ratio_nu_);
  }

  bool
  HasFaces() const
  {
    return true;
  }

  string
  GetType() const
  {
    return "face boundary";
    // boundary face
  }

  // compute drag value around cylinder
  double
  BoundaryValue(const FDC<DH, VECTOR, dealdim> &fdc)
  {
    unsigned int color = fdc.GetBoundaryIndicator();
    const auto &state_fe_face_values = fdc.GetFEFaceValuesState();
    unsigned int n_q_points = fdc.GetNQPoints();

    Tensor<1, 2> drag_lift_value;
    drag_lift_value.clear();
    if (color == 80)
      {
        vector<Vector<double> > ufacevalues;
        vector<vector<Tensor<1, dealdim> > > ufacegrads;

        ufacevalues.resize(n_q_points, Vector<double>(5));
        ufacegrads.resize(n_q_points, vector<Tensor<1, 2> >(5));

        fdc.GetFaceValuesState("state", ufacevalues);
        fdc.GetFaceGradsState("state", ufacegrads);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {

            Tensor<2, 2> pI;
            pI[0][0] = ufacevalues[q_point](4);
            pI[1][1] = ufacevalues[q_point](4);

            Tensor<1, 2> v;
            v.clear();
            v[0] = ufacevalues[q_point](0);
            v[1] = ufacevalues[q_point](1);

            Tensor<2, 2> grad_v;
            grad_v[0][0] = ufacegrads[q_point][0][0];
            grad_v[0][1] = ufacegrads[q_point][0][1];
            grad_v[1][0] = ufacegrads[q_point][1][0];
            grad_v[1][1] = ufacegrads[q_point][1][1];

            Tensor<2, 2> F;
            F[0][0] = 1.0 + ufacegrads[q_point][2][0];
            F[0][1] = ufacegrads[q_point][2][1];
            F[1][0] = ufacegrads[q_point][3][0];
            F[1][1] = 1.0 + ufacegrads[q_point][3][1];

            Tensor<2, 2> F_Inverse;
            F_Inverse = invert(F);

            Tensor<2, 2> F_T;
            F_T = transpose(F);

            Tensor<2, 2> F_Inverse_T;
            F_Inverse_T = transpose(F_Inverse);

            double J;
            J = determinant(F);

            // constitutive stress tensors for fluid
            Tensor<2, 2> cauchy_stress_fluid;
            cauchy_stress_fluid =
              J
              * (-pI
                 + density_fluid_ * viscosity_
                 * (grad_v * F_Inverse
                    + F_Inverse_T * transpose(grad_v)))
              * F_Inverse_T;

            drag_lift_value -= 1.0 * cauchy_stress_fluid
                               * state_fe_face_values.normal_vector(q_point)
                               * state_fe_face_values.JxW(q_point);
          }
      }
    return drag_lift_value[0];
  }

  double
  FaceValue(const FDC<DH, VECTOR, dealdim> &fdc)
  {
    const auto &state_fe_face_values = fdc.GetFEFaceValuesState();
    unsigned int n_q_points = fdc.GetNQPoints();
    unsigned int material_id = fdc.GetMaterialId();
    unsigned int material_id_neighbor = fdc.GetNbrMaterialId();

    Tensor<1, 2> drag_lift_value;
    drag_lift_value.clear();
    if (material_id == 0)
      {
        if (material_id != material_id_neighbor)
          {
            vector<Vector<double> > ufacevalues;
            vector<vector<Tensor<1, dealdim> > > ufacegrads;

            ufacevalues.resize(n_q_points, Vector<double>(5));
            ufacegrads.resize(n_q_points, vector<Tensor<1, 2> >(5));

            fdc.GetFaceValuesState("state", ufacevalues);
            fdc.GetFaceGradsState("state", ufacegrads);

            for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
              {

                Tensor<2, 2> pI;
                pI[0][0] = ufacevalues[q_point](4);
                pI[1][1] = ufacevalues[q_point](4);

                Tensor<1, 2> v;
                v.clear();
                v[0] = ufacevalues[q_point](0);
                v[1] = ufacevalues[q_point](1);

                Tensor<2, 2> grad_v;
                grad_v[0][0] = ufacegrads[q_point][0][0];
                grad_v[0][1] = ufacegrads[q_point][0][1];
                grad_v[1][0] = ufacegrads[q_point][1][0];
                grad_v[1][1] = ufacegrads[q_point][1][1];

                Tensor<2, 2> F;
                F[0][0] = 1.0 + ufacegrads[q_point][2][0];
                F[0][1] = ufacegrads[q_point][2][1];
                F[1][0] = ufacegrads[q_point][3][0];
                F[1][1] = 1.0 + ufacegrads[q_point][3][1];

                Tensor<2, 2> F_Inverse;
                F_Inverse = invert(F);

                Tensor<2, 2> F_T;
                F_T = transpose(F);

                Tensor<2, 2> F_Inverse_T;
                F_Inverse_T = transpose(F_Inverse);

                double J;
                J = determinant(F);

                // constitutive stress tensors for fluid
                Tensor<2, 2> cauchy_stress_fluid;
                cauchy_stress_fluid = J
                                      * (-pI
                                         + density_fluid_ * viscosity_
                                         * (grad_v * F_Inverse
                                            + F_Inverse_T * transpose(grad_v))) * F_Inverse_T;

                drag_lift_value -= 1.0 * cauchy_stress_fluid
                                   * state_fe_face_values.normal_vector(q_point)
                                   * state_fe_face_values.JxW(q_point);
              }
          }
      }
    return drag_lift_value[0];
  }

  UpdateFlags
  GetFaceUpdateFlags() const
  {
    return update_values | update_quadrature_points | update_gradients
           | update_normal_vectors;
  }

  string
  GetName() const
  {
    return "Drag";
  }

protected:
  inline void
  GetFaceValues(const DOpEWrapper::FEFaceValues<dealdim> &fe_face_values,
                const map<string, const VECTOR *> &domain_values, string name,
                vector<Vector<double> > &values)
  {
    typename map<string, const VECTOR *>::const_iterator it =
      domain_values.find(name);
    if (it == domain_values.end())
      {
        throw DOpEException("Did not find " + name, "LocalPDE::GetValues");
      }
    fe_face_values.get_function_values(*(it->second), values);
  }

  inline void
  GetFaceGrads(const DOpEWrapper::FEFaceValues<dealdim> &fe_face_values,
               const map<string, const VECTOR *> &domain_values, string name,
               vector<vector<Tensor<1, dealdim> > > &values)
  {
    typename map<string, const VECTOR *>::const_iterator it =
      domain_values.find(name);
    if (it == domain_values.end())
      {
        throw DOpEException("Did not find " + name, "LocalPDE::GetGrads");
      }
    fe_face_values.get_function_grads(*(it->second), values);
  }

private:
  double density_fluid_, viscosity_, lame_coefficient_mu_,
         poisson_ratio_nu_, lame_coefficient_lambda_;

};

// lift
/****************************************************************************************/
template<
template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
         template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
         template<int, int> class DH, typename VECTOR, int dopedim, int dealdim =
         dopedim>
class LocalBoundaryFaceFunctionalLift : public FunctionalInterface<EDC, FDC,
  DH, VECTOR, dopedim, dealdim>
{
public:
  static void
  declare_params(ParameterReader &param_reader)
  {
    param_reader.SetSubsection("Local PDE parameters");
    param_reader.declare_entry("density_fluid", "0.0", Patterns::Double(0));
    param_reader.declare_entry("viscosity", "0.0", Patterns::Double(0));
    param_reader.declare_entry("mu", "0.0", Patterns::Double(0));
    param_reader.declare_entry("poisson_ratio_nu", "0.0",
                               Patterns::Double(0));

  }

  LocalBoundaryFaceFunctionalLift(ParameterReader &param_reader)
  {
    param_reader.SetSubsection("Local PDE parameters");
    density_fluid_ = param_reader.get_double("density_fluid");
    viscosity_ = param_reader.get_double("viscosity");
    lame_coefficient_mu_ = param_reader.get_double("mu");
    poisson_ratio_nu_ = param_reader.get_double("poisson_ratio_nu");

    lame_coefficient_lambda_ =
      (2 * poisson_ratio_nu_ * lame_coefficient_mu_)
      / (1.0 - 2 * poisson_ratio_nu_);
  }

  bool
  HasFaces() const
  {
    return true;
  }

  string
  GetType() const
  {
    return "face boundary";
    // boundary face
  }

  // compute drag value around cylinder
  double
  BoundaryValue(const FDC<DH, VECTOR, dealdim> &fdc)
  {
    unsigned int color = fdc.GetBoundaryIndicator();
    const auto &state_fe_face_values = fdc.GetFEFaceValuesState();
    unsigned int n_q_points = fdc.GetNQPoints();

    Tensor<1, 2> drag_lift_value;
    drag_lift_value.clear();
    if (color == 80)
      {
        vector<Vector<double> > ufacevalues;
        vector<vector<Tensor<1, dealdim> > > ufacegrads;

        ufacevalues.resize(n_q_points, Vector<double>(5));
        ufacegrads.resize(n_q_points, vector<Tensor<1, 2> >(5));

        fdc.GetFaceValuesState("state", ufacevalues);
        fdc.GetFaceGradsState("state", ufacegrads);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            Tensor<2, 2> pI;
            pI[0][0] = ufacevalues[q_point](4);
            pI[1][1] = ufacevalues[q_point](4);

            Tensor<1, 2> v;
            v.clear();
            v[0] = ufacevalues[q_point](0);
            v[1] = ufacevalues[q_point](1);

            Tensor<2, 2> grad_v;
            grad_v[0][0] = ufacegrads[q_point][0][0];
            grad_v[0][1] = ufacegrads[q_point][0][1];
            grad_v[1][0] = ufacegrads[q_point][1][0];
            grad_v[1][1] = ufacegrads[q_point][1][1];

            Tensor<2, 2> F;
            F[0][0] = 1.0 + ufacegrads[q_point][2][0];
            F[0][1] = ufacegrads[q_point][2][1];
            F[1][0] = ufacegrads[q_point][3][0];
            F[1][1] = 1.0 + ufacegrads[q_point][3][1];

            Tensor<2, 2> F_Inverse;
            F_Inverse = invert(F);

            Tensor<2, 2> F_T;
            F_T = transpose(F);

            Tensor<2, 2> F_Inverse_T;
            F_Inverse_T = transpose(F_Inverse);

            double J;
            J = determinant(F);

            // constitutive stress tensors for fluid
            Tensor<2, 2> cauchy_stress_fluid;
            cauchy_stress_fluid =
              J
              * (-pI
                 + density_fluid_ * viscosity_
                 * (grad_v * F_Inverse
                    + F_Inverse_T * transpose(grad_v)))
              * F_Inverse_T;

            drag_lift_value -= 1.0 * cauchy_stress_fluid
                               * state_fe_face_values.normal_vector(q_point)
                               * state_fe_face_values.JxW(q_point);
          }

      }
    return drag_lift_value[1];
  }

  // compute drag value at interface
  double
  FaceValue(const FDC<DH, VECTOR, dealdim> &fdc)
  {
    const auto &state_fe_face_values = fdc.GetFEFaceValuesState();
    unsigned int n_q_points = fdc.GetNQPoints();
    unsigned int material_id = fdc.GetMaterialId();
    unsigned int material_id_neighbor = fdc.GetNbrMaterialId();

    Tensor<1, 2> drag_lift_value;
    drag_lift_value.clear();
    if (material_id == 0)
      {

        if ((material_id != material_id_neighbor) //&& (!at_boundary)
           )
          {
            vector<Vector<double> > ufacevalues;
            vector<vector<Tensor<1, dealdim> > > ufacegrads;

            ufacevalues.resize(n_q_points, Vector<double>(5));
            ufacegrads.resize(n_q_points, vector<Tensor<1, 2> >(5));

            fdc.GetFaceValuesState("state", ufacevalues);
            fdc.GetFaceGradsState("state", ufacegrads);

            for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
              {
                Tensor<2, 2> pI;
                pI[0][0] = ufacevalues[q_point](4);
                pI[1][1] = ufacevalues[q_point](4);

                Tensor<1, 2> v;
                v.clear();
                v[0] = ufacevalues[q_point](0);
                v[1] = ufacevalues[q_point](1);

                Tensor<2, 2> grad_v;
                grad_v[0][0] = ufacegrads[q_point][0][0];
                grad_v[0][1] = ufacegrads[q_point][0][1];
                grad_v[1][0] = ufacegrads[q_point][1][0];
                grad_v[1][1] = ufacegrads[q_point][1][1];

                Tensor<2, 2> F;
                F[0][0] = 1.0 + ufacegrads[q_point][2][0];
                F[0][1] = ufacegrads[q_point][2][1];
                F[1][0] = ufacegrads[q_point][3][0];
                F[1][1] = 1.0 + ufacegrads[q_point][3][1];

                Tensor<2, 2> F_Inverse;
                F_Inverse = invert(F);

                Tensor<2, 2> F_T;
                F_T = transpose(F);

                Tensor<2, 2> F_Inverse_T;
                F_Inverse_T = transpose(F_Inverse);

                double J;
                J = determinant(F);

                // constitutive stress tensors for fluid
                Tensor<2, 2> cauchy_stress_fluid;
                cauchy_stress_fluid = J
                                      * (-pI
                                         + density_fluid_ * viscosity_
                                         * (grad_v * F_Inverse
                                            + F_Inverse_T * transpose(grad_v))) * F_Inverse_T;

                drag_lift_value -= cauchy_stress_fluid
                                   * state_fe_face_values.normal_vector(q_point)
                                   * state_fe_face_values.JxW(q_point);

              }
          }
      }
    return drag_lift_value[1];
  }

  UpdateFlags
  GetFaceUpdateFlags() const
  {
    return update_values | update_quadrature_points | update_gradients
           | update_normal_vectors;
  }

  string
  GetName() const
  {
    return "Lift";
  }

protected:
  inline void
  GetFaceValues(const DOpEWrapper::FEFaceValues<dealdim> &fe_face_values,
                const map<string, const VECTOR *> &domain_values, string name,
                vector<Vector<double> > &values)
  {
    typename map<string, const VECTOR *>::const_iterator it =
      domain_values.find(name);
    if (it == domain_values.end())
      {
        throw DOpEException("Did not find " + name, "LocalPDE::GetValues");
      }
    fe_face_values.get_function_values(*(it->second), values);
  }

  inline void
  GetFaceGrads(const DOpEWrapper::FEFaceValues<dealdim> &fe_face_values,
               const map<string, const VECTOR *> &domain_values, string name,
               vector<vector<Tensor<1, dealdim> > > &values)
  {
    typename map<string, const VECTOR *>::const_iterator it =
      domain_values.find(name);
    if (it == domain_values.end())
      {
        throw DOpEException("Did not find " + name, "LocalPDE::GetGrads");
      }
    fe_face_values.get_function_grads(*(it->second), values);
  }

private:
  double density_fluid_, viscosity_, lame_coefficient_mu_,
         poisson_ratio_nu_, lame_coefficient_lambda_;

};

#endif
