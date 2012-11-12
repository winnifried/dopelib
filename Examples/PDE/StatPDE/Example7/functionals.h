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

#ifndef _LOCALFunctionalS_
#define _LOCALFunctionalS_

#include "pdeinterface.h"
#include "celldatacontainer.h"
#include "facedatacontainer.h"

using namespace std;
using namespace dealii;
using namespace DOpE;


/****************************************************************************************/
template<typename VECTOR, int dealdim>
  class LocalPointFunctionalX : public FunctionalInterface<CellDataContainer,FaceDataContainer,dealii::DoFHandler<dealdim>, VECTOR, dealdim>
  {
  public:

    double PointValue(const DOpEWrapper::DoFHandler<dealdim, dealii::DoFHandler<dealdim> > & /*control_dof_handler*/,
		      const DOpEWrapper::DoFHandler<dealdim, dealii::DoFHandler<dealdim> > &state_dof_handler,
		      const std::map<std::string, const dealii::Vector<double>* > &/*param_values*/,
		      const std::map<std::string, const VECTOR* > &domain_values)
  {
    Point<2> p1(0.5, 0.5);

    typename map<string, const VECTOR* >::const_iterator it = domain_values.find("state");
    Vector<double> tmp_vector(2);

    VectorTools::point_value (state_dof_handler, *(it->second), p1, tmp_vector);
    double x = tmp_vector(0);

    return x;

  }

  string GetType() const
  {
    return "point";
  }
  string GetName() const
  {
    return "Point value in X";
  }

  };

#endif
