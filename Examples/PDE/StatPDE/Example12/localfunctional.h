/**
 *
 * Copyright (C) 2012 by the DOpElib authors
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

#ifndef _LOCALFunctional_
#define _LOCALFunctional_

#include "functionalinterface.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

template<
    template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
    template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
    template<int, int> class DH, typename VECTOR, int dealdim>
  class LocalFunctional : public FunctionalInterface<CDC, FDC, DH, VECTOR,dealdim>
  {
    public:
      LocalFunctional()
      {
      }

      double
      Value(const CDC<DH, VECTOR, dealdim>& cdc)
      {
        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            cdc.GetFEValuesState();
        unsigned int n_q_points = cdc.GetNQPoints();

        {
	  _uvalues.resize(n_q_points,Vector<double>(3));
          cdc.GetValuesState("state", _uvalues);
        }
	const FEValuesExtractors::Vector velocities (0);
	const FEValuesExtractors::Scalar pressure (dealdim);

        double r = 0.;
        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
	  const double alpha = 0.3;
	  const double beta = 1;
	  Tensor<1,dealdim> p = state_fe_values.quadrature_point(q_point);
	   Vector<double> exact(3);
	  exact(0) = alpha*p[1]*p[1]/2 + beta - alpha*p[0]*p[0]/2;
	  exact(1) = alpha*p[0]*p[1];
	  exact(2) = -(alpha*p[0]*p[1]*p[1]/2 + beta*p[0] - alpha*p[0]*p[0]*p[0]/6);

	  exact -= _uvalues[q_point];
	  r += exact*exact*state_fe_values.JxW(q_point);
        }
        return r;
      }

      UpdateFlags
      GetUpdateFlags() const
      {
        return update_values | update_quadrature_points;
      }

      string
      GetType() const
      {
        return "domain";
      }

      string
      GetName() const
      {
        return "L^2-error";
      }

    private:
      vector<Vector<double> > _uvalues;
  };
#endif
