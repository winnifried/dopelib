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

#ifndef MYFUNCS_
#define MYFUNCS_

//Helper functions to calculate the coefficients
namespace my
{
  double f(const dealii::Point<2> &p)
  {
    return sin(p(0))*sin(p(1));
  }

  double g(double t)
  {
    return 0.25*(-2.*t-3.*exp(-2.*t) + 3);
  }
  double h(double t)
  {
    return 4.*(t-1)/(M_PI*M_PI);
  }
  double k(double t)
  {
    return g(t) + 4./(M_PI*M_PI) + (1.-t)*8./(M_PI*M_PI);
  }
  double ud(double t, const dealii::Point<2> &p)
  {
    return k(t)*f(p);
  }
  double optu(double t, const dealii::Point<2> &p)
  {
    return g(t)*f(p);
  }
  double optq(double t)
  {
    return 1.-t;
  }
}
#endif
