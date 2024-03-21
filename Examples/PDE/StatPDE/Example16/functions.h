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

#ifndef FUNCTIONS_H_
#define FUNCTIONS_H_

using namespace std;
using namespace dealii;

#include <deal.II/base/numbers.h>
#include <deal.II/base/function.h>
#include <wrapper/function_wrapper.h>

namespace local
{
  double rhs(const Point<2> &/*p*/)
  {
    //const double x = p[0];
    //const double y = p[1];

    return -5.;
  }

  class
    Obstacle : public Function<2>
  {
  public:
    Obstacle() : Function<2>(2) {}

    double value(const Point<2> &p, const unsigned int component = 0 ) const override
    {
      //return 0.5;
      if ( component ==0 )
        {
          const double x = p[0];
          const double y = p[1];
          //Calculate dist to \partial \Omega
          const double dist1 = min(1-abs(x),1-abs(y));
          //Calculate dist to \Omega \setminus (-1/4,1/4)
          double dist2 = 0.;
          if (max(abs(x),abs(y))<0.25)
            {
              dist2 = min(0.25-abs(x),0.25-abs(y));
            }
          return dist1 - 2.*dist2 - 1./5.;
        }
      return 1.;
    }

  };
}

#endif
