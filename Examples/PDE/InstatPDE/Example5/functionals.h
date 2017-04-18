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

const static double PI = 3.14159265359;

/****************************************************************************************/

template<
template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
         template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
         template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
class LocalPointFunctional : public FunctionalInterface<ElementDataContainer,
  FaceDataContainer, DH, VECTOR, dopedim, dealdim>
{
public:

  bool
  NeedTime() const
  {
    if (this->GetTime() == 0.)
      return true;
    else
      return false;
  }

  double
  PointValue(
    const DOpEWrapper::DoFHandler<dopedim, DH> &/* control_dof_handler*/,
    const DOpEWrapper::DoFHandler<dealdim, DH> &state_dof_handler,
    const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
    const std::map<std::string, const VECTOR *> &domain_values)
  {

    Point<2> evaluation_point(0.5 * PI, 0.5 * PI);

    typename map<string, const VECTOR *>::const_iterator it =
      domain_values.find("state");

    double point_value = VectorTools::point_value(state_dof_handler,
                                                  *(it->second), evaluation_point);

    return point_value;
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
    return "Start-Time-Point evaluation";
  }

};

/************************************************************************************************************************************************/

template<
template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
         template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
         template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
class LocalPointFunctional2 : public FunctionalInterface<EDC, FDC, DH, VECTOR,
  dopedim, dealdim>
{
public:

  bool
  NeedTime() const
  {
    if (this->GetTime() == 1.)
      return true;
    else
      return false;
  }

  double
  PointValue(
    const DOpEWrapper::DoFHandler<dopedim, DH> &/* control_dof_handler*/,
    const DOpEWrapper::DoFHandler<dealdim, DH> &state_dof_handler,
    const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
    const std::map<std::string, const VECTOR *> &domain_values)
  {

    Point<2> evaluation_point(0.5 * PI, 0.5 * PI);

    typename map<string, const VECTOR *>::const_iterator it =
      domain_values.find("state");

    double point_value = VectorTools::point_value(state_dof_handler,
                                                  *(it->second), evaluation_point);

    return point_value;
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
    return "End-Time-Point evaluation";
  }

};

/****************************************************************************************/

#endif
