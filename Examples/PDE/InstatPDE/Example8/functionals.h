/**
 *
 * Copyright (C) 2012-2017 by the DOpElib authors
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


/****************************************************************************************/

template<
template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
         template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
         template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
class LocalBoundaryFunctionalStressX : public FunctionalInterface<EDC, FDC, DH,
  VECTOR, dopedim, dealdim>
{
  double lame_coefficient_mu_, lame_coefficient_lambda_;

public:
  static void
  declare_params(ParameterReader &param_reader)
  {
    param_reader.SetSubsection("Local PDE parameters");
    param_reader.declare_entry("lame_coefficient_mu", "0.0", Patterns::Double(0));
    param_reader.declare_entry("lame_coefficient_lambda", "0.0", Patterns::Double(0));

  }

  LocalBoundaryFunctionalStressX(ParameterReader &param_reader)
  {
    param_reader.SetSubsection("Local PDE parameters");
    lame_coefficient_mu_ = param_reader.get_double("lame_coefficient_mu");
    lame_coefficient_lambda_ = param_reader.get_double("lame_coefficient_lambda");

  }

  bool
  NeedTime() const
  {
    return true;
  }

  double
  BoundaryValue(const FDC<DH, VECTOR, dealdim> &fdc)
  {
    unsigned int color = fdc.GetBoundaryIndicator();
    const auto &state_fe_face_values = fdc.GetFEFaceValuesState();
    unsigned int n_q_points = fdc.GetNQPoints();

    Tensor<2,2> Identity;
    Identity[0][0] = 1.0;
    Identity[1][1] = 1.0;

    if (color == 3)
      {
        Tensor<1, 2> load_value;

        vector<Vector<double> > ufacevalues;
        vector<vector<Tensor<1, dealdim> > > ufacegrads;

        ufacevalues.resize(n_q_points, Vector<double>(3));
        ufacegrads.resize(n_q_points, vector<Tensor<1, 2> >(3));

        fdc.GetFaceValuesState("state", ufacevalues);
        fdc.GetFaceGradsState("state", ufacegrads);

        const FEValuesExtractors::Vector displacements(0);


        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            Tensor<2, 2> grad_u;
            grad_u.clear();
            grad_u[0][0] = ufacegrads[q_point][0][0];
            grad_u[0][1] = ufacegrads[q_point][0][1];
            grad_u[1][0] = ufacegrads[q_point][1][0];
            grad_u[1][1] = ufacegrads[q_point][1][1];

            const Tensor<2,2> E = 0.5 * (grad_u + transpose(grad_u));
            const double tr_E = trace(E);

            Tensor<2,2> stress_term;
            stress_term = lame_coefficient_lambda_ * tr_E * Identity
                          + 2 * lame_coefficient_mu_ * E;


            load_value -= stress_term
                          * state_fe_face_values.normal_vector(q_point)
                          * state_fe_face_values.JxW(q_point);
          }


        return load_value[0];
      }


    return 0.;
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
    return "boundary timelocal";
    // 1) point domain boundary face
    // 2) timelocal timedistributed
  }
  string
  GetName() const
  {
    return "StressX";
  }
};


#endif
