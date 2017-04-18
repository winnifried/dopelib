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


#ifndef FUNCTIONALS_H_
#define FUNCTIONALS_H_

#include <interfaces/functionalinterface.h>

using namespace std;
using namespace dealii;
using namespace DOpE;

// massflux
/****************************************************************************************/

template<
template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
         template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
         template<int, int> class DH, typename VECTOR, int dealdim>
class LocalFaceFunctional : public FunctionalInterface<EDC, FDC, DH, VECTOR, dealdim>
{
public:
  LocalFaceFunctional()
  {
  }

  double
  FaceValue(const FDC<DH,VECTOR,dealdim> &fdc)
  {
    unsigned int n_q_points = fdc.GetNQPoints();
    unsigned int material_id = fdc.GetMaterialId();
    unsigned int material_id_neighbor = fdc.GetNbrMaterialId();

    double mean = 0;

    if (material_id == 1)
      {
        if (material_id_neighbor == 2)
          {
            vector<double> ufacevalues;

            ufacevalues.resize(n_q_points);

            fdc.GetFaceValuesState("state", ufacevalues);

            for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
              {
                double v;

                v = ufacevalues[q_point];

                mean += 1. / (1.5) * v * fdc.GetFEFaceValuesState().JxW(q_point);
              }
          }
      }
    return mean;
  }

  void
  FaceValue_U(const FDC<DH,VECTOR,dealdim> &fdc,
              dealii::Vector<double> &local_vector, double scale)
  {
    unsigned int n_q_points = fdc.GetNQPoints();
    unsigned int material_id = fdc.GetMaterialId();
    unsigned int material_id_neighbor = fdc.GetNbrMaterialId();

    if (material_id == 1)
      {
        if (material_id_neighbor == 2)
          {
            for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
              {
                for (unsigned int i = 0; i < fdc.GetNDoFsPerElement(); ++i)
                  {
                    local_vector(i) += scale * 1. / (1.5)
                                       * fdc.GetFEFaceValuesState().shape_value(i, q_point)
                                       * fdc.GetFEFaceValuesState().JxW(q_point);
                  }
              }
          }
      }
  }


  UpdateFlags
  GetFaceUpdateFlags() const
  {
    return update_values | update_quadrature_points | update_normal_vectors;
  }

  string
  GetType() const
  {
    return "face";
  }

  bool HasFaces() const
  {
    return true;
  }

  string
  GetName() const
  {
    return "Local Mean value";
  }

private:
  int outflow_fluid_boundary_color_;
};
#endif /* FUNCTIONALS_H_ */
