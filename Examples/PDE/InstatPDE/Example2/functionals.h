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

using namespace std;
using namespace dealii;
using namespace DOpE;

// pressure
/****************************************************************************************/

template<
template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
         template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
         template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
class LocalPointFunctionalPressure : public FunctionalInterface<EDC, FDC, DH,
  VECTOR, dopedim, dealdim>
{
  bool
  NeedTime() const
  {
    return true;
  }

  double
  PointValue(
    const DOpEWrapper::DoFHandler<dopedim, DH> & /*control_dof_handler*/,
    const DOpEWrapper::DoFHandler<dealdim, DH> &state_dof_handler,
    const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
    const std::map<std::string, const VECTOR *> &domain_values)
  {

    Point<2> p1(0.15, 0.2);
    Point<2> p2(0.25, 0.2);

    typename map<string, const VECTOR *>::const_iterator it =
      domain_values.find("state");
    Vector<double> tmp_vector(7);

    VectorTools::point_value(state_dof_handler, *(it->second), p1,
                             tmp_vector);
    double p1_value = tmp_vector(4);
    tmp_vector = 0;
    VectorTools::point_value(state_dof_handler, *(it->second), p2,
                             tmp_vector);
    double p2_value = tmp_vector(4);

    // pressure analysis
    return (p1_value - p2_value);

  }

  string
  GetType() const
  {
    return "point timelocal";
    // 1) point domain boundary face
    // 2) timelocal timedistributed
  }
  string
  GetName() const
  {
    return "Pressure_difference";
  }

};

// deflection x
/****************************************************************************************/
template<
template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
         template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
         template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
class LocalPointFunctionalDeflectionX : public FunctionalInterface<EDC, FDC,
  DH, VECTOR, dopedim, dealdim>
{
public:
  bool
  NeedTime() const
  {
    return true;
  }

  double
  PointValue(
    const DOpEWrapper::DoFHandler<dopedim, DH> & /*control_dof_handler*/,
    const DOpEWrapper::DoFHandler<dealdim, DH> &state_dof_handler,
    const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
    const std::map<std::string, const VECTOR *> &domain_values)
  {

    Point<dealdim> p1(0.6, 0.2);

    typename map<string, const VECTOR *>::const_iterator it =
      domain_values.find("state");
    Vector<double> tmp_vector(7);

    VectorTools::point_value(state_dof_handler, *(it->second), p1,
                             tmp_vector);
    double x = tmp_vector(2);

    // Deflection X
    return x;

  }

  string
  GetType() const
  {
    return "point timelocal";
  }
  string
  GetName() const
  {
    return "Deflection_X";
  }

};

// deflection y
/****************************************************************************************/
template<
template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
         template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
         template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
class LocalPointFunctionalDeflectionY : public FunctionalInterface<EDC, FDC,
  DH, VECTOR, dopedim, dealdim>
{
public:

  bool
  NeedTime() const
  {
    return true;
  }

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
    Vector<double> tmp_vector(7);

    VectorTools::point_value(state_dof_handler, *(it->second), p1,
                             tmp_vector);
    double y = tmp_vector(3);

    // Delfection Y
    return y;

  }

  string
  GetType() const
  {
    return "point timelocal";
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
         template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
class LocalBoundaryFaceFunctionalDrag : public FunctionalInterface<EDC, FDC,
  DH, VECTOR, dopedim, dealdim>
{
private:
  double density_fluid, viscosity_, lame_coefficient_mu_,
         poisson_ratio_nu_, lame_coefficient_lambda_;

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
    density_fluid = param_reader.get_double("density_fluid");
    viscosity_ = param_reader.get_double("viscosity");
    lame_coefficient_mu_ = param_reader.get_double("mu");
    poisson_ratio_nu_ = param_reader.get_double("poisson_ratio_nu");

    lame_coefficient_lambda_ =
      (2 * poisson_ratio_nu_ * lame_coefficient_mu_)
      / (1.0 - 2 * poisson_ratio_nu_);
  }

  bool
  NeedTime() const
  {
    return true;
  }

  bool
  HasFaces() const
  {
    return true;
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
        vector<Vector<double> > _ufacevalues;
        vector<vector<Tensor<1, dealdim> > > _ufacegrads;

        _ufacevalues.resize(n_q_points, Vector<double>(7));
        _ufacegrads.resize(n_q_points, vector<Tensor<1, 2> >(7));

        fdc.GetFaceValuesState("state", _ufacevalues);
        fdc.GetFaceGradsState("state", _ufacegrads);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            const Tensor<2, 2> pI = ALE_Transformations::get_pI<2>(q_point,
                                                                   _ufacevalues);

            const Tensor<2, 2> grad_v = ALE_Transformations::get_grad_v<2>(
                                          q_point, _ufacegrads);

            const Tensor<2, 2> grad_v_T = ALE_Transformations::get_grad_v_T<2>(
                                            grad_v);

            const Tensor<2, 2> F = ALE_Transformations::get_F<2>(q_point,
                                                                 _ufacegrads);

            const Tensor<2, 2> F_Inverse =
              ALE_Transformations::get_F_Inverse<2>(F);

            const Tensor<2, 2> F_Inverse_T =
              ALE_Transformations::get_F_Inverse_T<2>(F_Inverse);

            const double J = ALE_Transformations::get_J<2>(F);

            const Tensor<2, 2> sigma_ALE =
              NSE_in_ALE::get_stress_fluid_except_pressure_ALE<2>(
                density_fluid, viscosity_, grad_v, grad_v_T, F_Inverse,
                F_Inverse_T);

            Tensor<2, 2> stress_fluid;
            stress_fluid.clear();
            stress_fluid = (J * sigma_ALE * F_Inverse_T);

            Tensor<2, 2> fluid_pressure;
            fluid_pressure.clear();
            fluid_pressure = (-pI * J * F_Inverse_T);

            drag_lift_value -= (stress_fluid + fluid_pressure)
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
            vector<Vector<double> > _ufacevalues;
            vector<vector<Tensor<1, dealdim> > > _ufacegrads;

            _ufacevalues.resize(n_q_points, Vector<double>(7));
            _ufacegrads.resize(n_q_points, vector<Tensor<1, 2> >(7));

            fdc.GetFaceValuesState("state", _ufacevalues);
            fdc.GetFaceGradsState("state", _ufacegrads);

            const FEValuesExtractors::Vector velocities(0);
            //const FEValuesExtractors::Scalar pressure (2);  2=3!!!!

            for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
              {
                const Tensor<2, 2> pI = ALE_Transformations::get_pI<2>(q_point,
                                                                       _ufacevalues);

                const Tensor<2, 2> grad_v = ALE_Transformations::get_grad_v<2>(
                                              q_point, _ufacegrads);

                const Tensor<2, 2> grad_v_T =
                  ALE_Transformations::get_grad_v_T<2>(grad_v);

                const Tensor<2, 2> F = ALE_Transformations::get_F<2>(q_point,
                                                                     _ufacegrads);

                const Tensor<2, 2> F_Inverse = ALE_Transformations::get_F_Inverse<
                                               2>(F);

                const Tensor<2, 2> F_Inverse_T =
                  ALE_Transformations::get_F_Inverse_T<2>(F_Inverse);

                const double J = ALE_Transformations::get_J<2>(F);

                const Tensor<2, 2> sigma_ALE =
                  NSE_in_ALE::get_stress_fluid_except_pressure_ALE<2>(
                    density_fluid, viscosity_, grad_v, grad_v_T, F_Inverse,
                    F_Inverse_T);

                Tensor<2, 2> stress_fluid;
                stress_fluid.clear();
                stress_fluid = (J * sigma_ALE * F_Inverse_T);

                Tensor<2, 2> fluid_pressure;
                fluid_pressure.clear();
                fluid_pressure = (-pI * J * F_Inverse_T);

                drag_lift_value -= (stress_fluid + fluid_pressure)
                                   * state_fe_face_values.normal_vector(q_point)
                                   * state_fe_face_values.JxW(q_point);
              }
          }
      } // end material_id  == 0
    return drag_lift_value[0];

  } // end function

  UpdateFlags
  GetFaceUpdateFlags() const
  {
    return update_values | update_quadrature_points | update_gradients
           | update_normal_vectors;
  }

  string
  GetType() const
  {
    return "boundary face timelocal";
    // 1) point domain boundary face
    // 2) timelocal timedistributed
  }
  string
  GetName() const
  {
    return "Drag";
  }

};

// lift
/****************************************************************************************/

template<
template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
         template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
         template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
class LocalBoundaryFaceFunctionalLift : public FunctionalInterface<EDC, FDC,
  DH, VECTOR, dopedim, dealdim>
{
private:
  double density_fluid, viscosity_, lame_coefficient_mu_,
         poisson_ratio_nu_, lame_coefficient_lambda_;

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
    density_fluid = param_reader.get_double("density_fluid");
    viscosity_ = param_reader.get_double("viscosity");
    lame_coefficient_mu_ = param_reader.get_double("mu");
    poisson_ratio_nu_ = param_reader.get_double("poisson_ratio_nu");

    lame_coefficient_lambda_ =
      (2 * poisson_ratio_nu_ * lame_coefficient_mu_)
      / (1.0 - 2 * poisson_ratio_nu_);
  }

  bool
  NeedTime() const
  {
    return true;
  }

  bool
  HasFaces() const
  {
    return true;
  }

  // compute lift value around cylinder
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

        ufacevalues.resize(n_q_points, Vector<double>(7));
        ufacegrads.resize(n_q_points, vector<Tensor<1, 2> >(7));

        fdc.GetFaceValuesState("state", ufacevalues);
        fdc.GetFaceGradsState("state", ufacegrads);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            const Tensor<2, 2> pI = ALE_Transformations::get_pI<2>(q_point,
                                                                   ufacevalues);

            const Tensor<2, 2> grad_v = ALE_Transformations::get_grad_v<2>(
                                          q_point, ufacegrads);

            const Tensor<2, 2> grad_v_T = ALE_Transformations::get_grad_v_T<2>(
                                            grad_v);

            const Tensor<2, 2> F = ALE_Transformations::get_F<2>(q_point,
                                                                 ufacegrads);

            const Tensor<2, 2> F_Inverse =
              ALE_Transformations::get_F_Inverse<2>(F);

            const Tensor<2, 2> F_Inverse_T =
              ALE_Transformations::get_F_Inverse_T<2>(F_Inverse);

            const double J = ALE_Transformations::get_J<2>(F);

            const Tensor<2, 2> sigma_ALE =
              NSE_in_ALE::get_stress_fluid_except_pressure_ALE<2>(
                density_fluid, viscosity_, grad_v, grad_v_T, F_Inverse,
                F_Inverse_T);

            Tensor<2, 2> stress_fluid;
            stress_fluid.clear();
            stress_fluid = (J * sigma_ALE * F_Inverse_T);

            Tensor<2, 2> fluid_pressure;
            fluid_pressure.clear();
            fluid_pressure = (-pI * J * F_Inverse_T);

            drag_lift_value -= (stress_fluid + fluid_pressure)
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
    //unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();
    unsigned int n_q_points = fdc.GetNQPoints();
    //unsigned int color = fdc.GetBoundaryIndicator();
    unsigned int material_id = fdc.GetMaterialId();
    unsigned int material_id_neighbor = fdc.GetNbrMaterialId();
    //bool at_boundary = fdc.GetIsAtBoundary();

    Tensor<1, 2> drag_lift_value;
    drag_lift_value.clear();
    if (material_id == 0)
      {
        if (material_id != material_id_neighbor)
          {
            vector<Vector<double> > ufacevalues;
            vector<vector<Tensor<1, dealdim> > > ufacegrads;

            ufacevalues.resize(n_q_points, Vector<double>(7));
            ufacegrads.resize(n_q_points, vector<Tensor<1, 2> >(7));

            fdc.GetFaceValuesState("state", ufacevalues);
            fdc.GetFaceGradsState("state", ufacegrads);

            const FEValuesExtractors::Vector velocities(0);
            const FEValuesExtractors::Scalar pressure(2);

            for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
              {
                const Tensor<2, 2> pI = ALE_Transformations::get_pI<2>(q_point,
                                                                       ufacevalues);

//      const Tensor<1,2> v = ALE_Transformations
//        ::get_v<2> (q_point, ufacevalues);

                const Tensor<2, 2> grad_v = ALE_Transformations::get_grad_v<2>(
                                              q_point, ufacegrads);

                const Tensor<2, 2> grad_v_T =
                  ALE_Transformations::get_grad_v_T<2>(grad_v);

                const Tensor<2, 2> F = ALE_Transformations::get_F<2>(q_point,
                                                                     ufacegrads);

                const Tensor<2, 2> F_Inverse = ALE_Transformations::get_F_Inverse<
                                               2>(F);

                const Tensor<2, 2> F_Inverse_T =
                  ALE_Transformations::get_F_Inverse_T<2>(F_Inverse);

                const double J = ALE_Transformations::get_J<2>(F);

                const Tensor<2, 2> sigma_ALE =
                  NSE_in_ALE::get_stress_fluid_except_pressure_ALE<2>(
                    density_fluid, viscosity_, grad_v, grad_v_T, F_Inverse,
                    F_Inverse_T);

                Tensor<2, 2> stress_fluid;
                stress_fluid.clear();
                stress_fluid = (J * sigma_ALE * F_Inverse_T);

                Tensor<2, 2> fluid_pressure;
                fluid_pressure.clear();
                fluid_pressure = (-pI * J * F_Inverse_T);

                drag_lift_value -= (stress_fluid + fluid_pressure)
                                   * state_fe_face_values.normal_vector(q_point)
                                   * state_fe_face_values.JxW(q_point);
              }

          }
      } // end material_id  == 0
    return drag_lift_value[1];

  } // end function

  UpdateFlags
  GetFaceUpdateFlags() const
  {
    return update_values | update_quadrature_points | update_gradients
           | update_normal_vectors;
  }

  string
  GetType() const
  {
    return "boundary face timelocal";
    // 1) point domain boundary face
    // 2) timelocal timedistributed
  }
  string
  GetName() const
  {
    return "Lift";
  }
};

#endif
