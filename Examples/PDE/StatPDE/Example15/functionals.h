/**
 *
 * Copyright (C) 2012-2018 by the DOpElib authors
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

#include <container/elementdatacontainer.h>
#include <container/facedatacontainer.h>
#include <interfaces/pdeinterface.h>

using namespace std;
using namespace dealii;
using namespace DOpE;

/****************************************************************************************/
#if DEAL_II_VERSION_GTE(9,3,0)
template<
template<bool DH, typename VECTOR, int dealdim> class EDC,
  template<bool DH, typename VECTOR, int dealdim> class FDC,
  bool DH, typename VECTOR, int dealdim>
class LocalPointFunctionalX
  : public FunctionalInterface<EDC, FDC, DH, VECTOR, dealdim>
#else
template<
template<template <int, int> class DH, typename VECTOR, int dealdim> class EDC,
  template<template <int, int> class DH, typename VECTOR, int dealdim> class FDC,
  template <int, int> class DH, typename VECTOR, int dealdim>
class LocalPointFunctionalX
    : public FunctionalInterface<EDC, FDC, DH, VECTOR, dealdim>
#endif
{
public:
  LocalPointFunctionalX() { assert(dealdim == 3); }

  double PointValue(
#if DEAL_II_VERSION_GTE(9,3,0)
      const DOpEWrapper::DoFHandler<dealdim> & /*control_dof_handler*/,
      const DOpEWrapper::DoFHandler<dealdim> &state_dof_handler,
#else
      const DOpEWrapper::DoFHandler<dealdim, DH> & /*control_dof_handler*/,
      const DOpEWrapper::DoFHandler<dealdim, DH> &state_dof_handler,
#endif
      const std::map<std::string, const dealii::Vector<double> *>
          & /*param_values*/,
      const std::map<std::string, const VECTOR *> &domain_values) override {
    Point<dealdim> p1(0.5, 0.5, 0.5);

    typename map<string, const VECTOR *>::const_iterator it =
        domain_values.find("state");
    Vector<double> tmp_vector(3);
    tmp_vector = std::numeric_limits<double>::min();

    try {
      VectorTools::point_value(state_dof_handler, *(it->second), p1,
                               tmp_vector);
    } catch (dealii::VectorTools::ExcPointNotAvailableHere &e) {
    }

    return dealii::Utilities::MPI::max(tmp_vector(0), MPI_COMM_WORLD);
  }

  string GetType() const override { return "point"; }
  string GetName() const override { return "Point value in X"; }
};

#endif
