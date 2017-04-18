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
class LocalPointFunctionalP1 : public FunctionalInterface<ElementDataContainer,
  FaceDataContainer, DH, VECTOR, dopedim, dealdim>
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

    Point<2> p1(0.0, 0.0);

    typename map<string, const VECTOR *>::const_iterator it =
      domain_values.find("state");
    Vector<double> tmp_vector(3);

    VectorTools::point_value(state_dof_handler, *(it->second), p1,
                             tmp_vector);
    double p1_value = tmp_vector(2);

    return (p1_value);

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
    return "P1";
  }

};

// pressure
/****************************************************************************************/

template<
template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
         template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
         template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
class LocalPointFunctionalP2 : public FunctionalInterface<EDC, FDC, DH,
  VECTOR, dopedim, dealdim>
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

    Point<2> p1(50.0, 0.0);

    typename map<string, const VECTOR *>::const_iterator it =
      domain_values.find("state");
    Vector<double> tmp_vector(3);

    VectorTools::point_value(state_dof_handler, *(it->second), p1,
                             tmp_vector);
    double p1_value = tmp_vector(2);

    return (p1_value);
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
    return "P2";
  }

};

#endif
