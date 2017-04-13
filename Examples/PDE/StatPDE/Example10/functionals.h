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

/****************************************************************************************/

template<
template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
         template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
         template<int, int> class DH, typename VECTOR, int dealdim>
class LocalPointFunctionalPressure : public FunctionalInterface<EDC, FDC, DH,
  VECTOR, dealdim>
{
public:
  LocalPointFunctionalPressure()
  {
    assert(dealdim==2);
  }

  double
  PointValue(
    const DOpEWrapper::DoFHandler<dealdim, DH> & /*control_dof_handler*/,
    const DOpEWrapper::DoFHandler<dealdim, DH> &state_dof_handler,
    const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
    const std::map<std::string, const VECTOR *> &domain_values)
  {
    Point<2> p1(0.15, 0.2);
    Point<2> p2(0.25, 0.2);

    typename map<string, const VECTOR *>::const_iterator it =
      domain_values.find("state");
    Vector<double> tmp_vector(5);

    VectorTools::point_value(state_dof_handler, *(it->second), p1,
                             tmp_vector);
    double p1_value = tmp_vector(2);
    tmp_vector = 0;
    VectorTools::point_value(state_dof_handler, *(it->second), p2,
                             tmp_vector);
    double p2_value = tmp_vector(2);

    // pressure analysis
    return (p1_value - p2_value);

  }

  string
  GetType() const
  {
    return "point";
  }
  string
  GetName() const
  {
    return "Pressure difference";
  }

};

/****************************************************************************************/
template<
template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
         template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
         template<int, int> class DH, typename VECTOR, int dealdim>
class LocalPointFunctionalDeflectionX : public FunctionalInterface<EDC, FDC,
  DH, VECTOR, dealdim>
{
public:

  double
  PointValue(
    const DOpEWrapper::DoFHandler<dealdim, DH> & /*control_dof_handler*/,
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
    double x = tmp_vector(3);

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
         template<int, int> class DH, typename VECTOR, int dealdim>
class LocalPointFunctionalDeflectionY : public FunctionalInterface<EDC, FDC,
  DH, VECTOR, dealdim>
{
public:

  double
  PointValue(
    const DOpEWrapper::DoFHandler<dealdim, DH> & /*control_dof_handler*/,
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
    double y = tmp_vector(4);

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
         template<int, int> class DH, typename VECTOR, int dealdim>
class LocalBoundaryFaceFunctionalDrag : public FunctionalInterface<EDC, FDC,
  DH, VECTOR, dealdim>
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
    poisson_ration_nu_ = param_reader.get_double("poisson_ratio_nu");

    lame_coefficient_lambda_ =
      (2 * poisson_ration_nu_ * lame_coefficient_mu_)
      / (1.0 - 2 * poisson_ration_nu_);
  }

  bool
  HasFaces() const
  {
    return true;
  }

  // compute drag value around cylinder
  double
  BoundaryValue(const FDC<DH, VECTOR, 2> &fdc)
  {
    unsigned int color = fdc.GetBoundaryIndicator();
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
            pI[0][0] = ufacevalues[q_point](2);
            pI[1][1] = ufacevalues[q_point](2);

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
            F[0][0] = 1.0 + ufacegrads[q_point][3][0];
            F[0][1] = ufacegrads[q_point][3][1];
            F[1][0] = ufacegrads[q_point][4][0];
            F[1][1] = 1.0 + ufacegrads[q_point][4][1];

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

            drag_lift_value -= cauchy_stress_fluid
                               * fdc.GetFEFaceValuesState().normal_vector(q_point)
                               * fdc.GetFEFaceValuesState().JxW(q_point);
          }

      }
    return drag_lift_value[0];

  }

  double
  FaceValue(const FDC<DH, VECTOR, 2> &fdc)
  {

    unsigned int material_id = fdc.GetMaterialId();
    unsigned int material_id_neighbor = fdc.GetNbrMaterialId();
    bool at_boundary = fdc.GetIsAtBoundary();
    unsigned int n_q_points = fdc.GetNQPoints();

    Tensor<1, 2> drag_lift_value;
    drag_lift_value.clear();
    if (material_id == 0)
      {

        if ((material_id != material_id_neighbor) && (!at_boundary))
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
                pI[0][0] = ufacevalues[q_point](2);
                pI[1][1] = ufacevalues[q_point](2);

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
                F[0][0] = 1.0 + ufacegrads[q_point][3][0];
                F[0][1] = ufacegrads[q_point][3][1];
                F[1][0] = ufacegrads[q_point][4][0];
                F[1][1] = 1.0 + ufacegrads[q_point][4][1];

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
                                   * fdc.GetFEFaceValuesState().normal_vector(q_point)
                                   * fdc.GetFEFaceValuesState().JxW(q_point);

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
  GetType() const
  {
    return "boundary face";
  }
  string
  GetName() const
  {
    return "Drag";
  }

private:
  double density_fluid_, viscosity_, lame_coefficient_mu_,
         poisson_ration_nu_, lame_coefficient_lambda_;
};

// lift
/****************************************************************************************/
template<
template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
         template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
         template<int, int> class DH, typename VECTOR, int dealdim>
class LocalBoundaryFaceFunctionalLift : public FunctionalInterface<EDC, FDC,
  DH, VECTOR, dealdim>
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
    poisson_ration_nu_ = param_reader.get_double("poisson_ratio_nu");

    lame_coefficient_lambda_ =
      (2 * poisson_ration_nu_ * lame_coefficient_mu_)
      / (1.0 - 2 * poisson_ration_nu_);
  }

  bool
  HasFaces() const
  {
    return true;
  }

  // compute drag value around cylinder
  double
  BoundaryValue(const FDC<DH, VECTOR, 2> &fdc)
  {
    unsigned int n_q_points = fdc.GetNQPoints();
    unsigned int color = fdc.GetBoundaryIndicator();

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
            pI[0][0] = ufacevalues[q_point](2);
            pI[1][1] = ufacevalues[q_point](2);

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
            F[0][0] = 1.0 + ufacegrads[q_point][3][0];
            F[0][1] = ufacegrads[q_point][3][1];
            F[1][0] = ufacegrads[q_point][4][0];
            F[1][1] = 1.0 + ufacegrads[q_point][4][1];

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

            drag_lift_value -= cauchy_stress_fluid
                               * fdc.GetFEFaceValuesState().normal_vector(q_point)
                               * fdc.GetFEFaceValuesState().JxW(q_point);
          }

      }
    return drag_lift_value[1];

  }

  // compute drag value at interface
  double
  FaceValue(const FDC<DH, VECTOR, 2> &fdc)
  {

    unsigned int n_q_points = fdc.GetNQPoints();
    unsigned int material_id = fdc.GetMaterialId();
    unsigned int material_id_neighbor = fdc.GetNbrMaterialId();
    bool at_boundary = fdc.GetIsAtBoundary();

    Tensor<1, 2> drag_lift_value;
    drag_lift_value.clear();
    if (material_id == 0)
      {

        if ((material_id != material_id_neighbor) && (!at_boundary))
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
                pI[0][0] = ufacevalues[q_point](2);
                pI[1][1] = ufacevalues[q_point](2);

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
                F[0][0] = 1.0 + ufacegrads[q_point][3][0];
                F[0][1] = ufacegrads[q_point][3][1];
                F[1][0] = ufacegrads[q_point][4][0];
                F[1][1] = 1.0 + ufacegrads[q_point][4][1];

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
                                   * fdc.GetFEFaceValuesState().normal_vector(q_point)
                                   * fdc.GetFEFaceValuesState().JxW(q_point);

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
  GetType() const
  {
    return "boundary face";
  }
  string
  GetName() const
  {
    return "Lift";
  }

private:
  double density_fluid_, viscosity_, lame_coefficient_mu_,
         poisson_ration_nu_, lame_coefficient_lambda_;

};

#endif
