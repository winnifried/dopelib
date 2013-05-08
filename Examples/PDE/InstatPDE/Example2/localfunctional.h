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

#include "pdeinterface.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

template<typename VECTOR, int dopedim, int dealdim>
  class LocalFunctional : public FunctionalInterface<CellDataContainer,FaceDataContainer,dealii::DoFHandler, VECTOR, dopedim,dealdim>
  {
  public:
  LocalFunctional()
      {
	_alpha = 1.e-3;
      }

  void SetTime(double t) const
    {
      _time = t;
    }


   bool NeedTime() const
    {
      if(fabs(_time-1.)< 1.e-13)
	return true;
      return false;
    }


   double Value(const CellDataContainer<dealii::DoFHandler, VECTOR, dealdim>&)
      {
	return 0.0;
      }

   void Value_U(const CellDataContainer<dealii::DoFHandler, VECTOR, dealdim>&,
		dealii::Vector<double> &local_cell_vector __attribute__((unused)),
		double scale __attribute__((unused)))
    {

    }

    void Value_Q(const CellDataContainer<dealii::DoFHandler, VECTOR, dealdim>&,
                 dealii::Vector<double> &local_cell_vector __attribute__((unused)),
                 double scale __attribute__((unused)))
    {

    }

    void Value_UU(const CellDataContainer<dealii::DoFHandler, VECTOR, dealdim>&,
                  dealii::Vector<double> &local_cell_vector __attribute__((unused)),
                  double scale __attribute__((unused)))
    {

    }

    void Value_QU(const CellDataContainer<dealii::DoFHandler, VECTOR, dealdim>&,
                  dealii::Vector<double> &local_cell_vector __attribute__((unused)),
                  double scale __attribute__((unused)))
    {
    }

    void Value_UQ(const CellDataContainer<dealii::DoFHandler, VECTOR, dealdim>&,
                  dealii::Vector<double> &local_cell_vector __attribute__((unused)),
                  double scale __attribute__((unused)))
    {
    }

    void Value_QQ(const CellDataContainer<dealii::DoFHandler, VECTOR, dealdim>&,
                  dealii::Vector<double> &local_cell_vector __attribute__((unused)),
                  double scale __attribute__((unused)))
    {

    }


    UpdateFlags GetUpdateFlags() const
    {
      return update_values | update_quadrature_points;
    }

    string GetType() const
    {
      return "domain timedistributed";
    }
    
    string GetName() const
    {
	  return "dummy functional";
	}
  private:
    Vector<double> _qvalues;
    vector<Vector<double> > _fvalues;
    vector<Vector<double> > _uvalues;
    vector<Vector<double> > _duvalues;
    double _alpha;
    mutable double _time;
  };
#endif
