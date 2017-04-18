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

#include <interfaces/functionalinterface.h>

using namespace DOpE;

/**
 * This class describes the functionals we want to evaluate.
 * See pdeinterface.h for more information.
 */
/****************************************************************************************/

/**
 * This functional evaluates the first displacement component at (2,1).
 */
template<
template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
         template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
         template<int, int> class DH, typename VECTOR, int dealdim>
class LocalPointFunctionalX : public FunctionalInterface<EDC, FDC, DH, VECTOR,
  dealdim>
{
public:
  LocalPointFunctionalX()
  {
    assert(dealdim==2);
  }

  double
  PointValue(const DOpEWrapper::DoFHandler<dealdim, DH> &
             /*control_dof_handler*/,
             const DOpEWrapper::DoFHandler<dealdim, DH> &state_dof_handler,
             const std::map<std::string, const dealii::Vector<double>*> &
             /*param_values*/,
             const std::map<std::string, const VECTOR *> &domain_values)
  {
    const dealii::Point<2> p1(2.0, 1.0);

    typename std::map<std::string, const VECTOR *>::const_iterator it =
      domain_values.find("state");
    dealii::Vector<double> tmp_vector(5);

    VectorTools::point_value(state_dof_handler, *(it->second), p1,
                             tmp_vector);
    double x = tmp_vector(3);

    //Displacement in x-direction
    return x;

  }

  /**
   * Describes the type of the functional.
   */
  std::string
  GetType() const
  {
    return "point";
  }
  std::string
  GetName() const
  {
    return "X-displacement";
  }

};

/****************************************************************************************/
/**
 * This functional evaluates the flux over the outflow boundary.
 */

template<
template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
         template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
         template<int, int> class DH, typename VECTOR, int dealdim>
class LocalBoundaryFluxFunctional : public FunctionalInterface<EDC, FDC, DH,
  VECTOR, dealdim>
{
public:
  LocalBoundaryFluxFunctional()
  {
    assert(dealdim==2);
  }

  double
  BoundaryValue(const FaceDataContainer<DH, VECTOR, dealdim> &fdc)
  {
    const unsigned int color = fdc.GetBoundaryIndicator();
    //auto = FEValues
    const auto &state_fe_face_values = fdc.GetFEFaceValuesState();
    const unsigned int n_q_points = fdc.GetNQPoints();
    double flux = 0.0;
    if (color == 1)
      {
        std::vector<dealii::Vector<double> > ufacevalues;
        ufacevalues.resize(n_q_points, dealii::Vector<double>(5));
        fdc.GetFaceValuesState("state", ufacevalues);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            dealii::Tensor<1, 2> v;
            v.clear();
            v[0] = ufacevalues[q_point](0);
            v[1] = ufacevalues[q_point](1);

            flux += v * state_fe_face_values.normal_vector(q_point)
                    * state_fe_face_values.JxW(q_point);
          }
      }
    return flux;

  }

  UpdateFlags
  GetFaceUpdateFlags() const
  {
    return update_values | update_quadrature_points | update_normal_vectors;
  }

  std::string
  GetType() const
  {
    return "boundary";
  }
  std::string
  GetName() const
  {
    return "Flux";
  }
};

#endif
