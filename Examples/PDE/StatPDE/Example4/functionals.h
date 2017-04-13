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
#include <container/elementdatacontainer.h>
#include <container/facedatacontainer.h>

using namespace std;
using namespace dealii;
using namespace DOpE;

/****************************************************************************************/
template<
template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
         template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
         template<int, int> class DH, typename VECTOR, int dealdim>
class LocalPointFunctionalX : public FunctionalInterface<EDC, FDC, DH, VECTOR,
  dealdim>
{
public:
  double
  PointValue(
    const DOpEWrapper::DoFHandler<dealdim, DH> & /*control_dof_handler*/,
    const DOpEWrapper::DoFHandler<dealdim, DH> &state_dof_handler,
    const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
    const std::map<std::string, const VECTOR *> &domain_values)
  {
    Point<dealdim> p1;
    for (unsigned int i = 0; i < dealdim; i++)
      p1[i] = 0.5;

    typename map<string, const VECTOR *>::const_iterator it =
      domain_values.find("state");
    Vector<double> tmp_vector(2);

    VectorTools::point_value(state_dof_handler, *(it->second), p1,
                             tmp_vector);
    double x = tmp_vector(0);

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
    return "Point value in X";
  }

};

#endif
