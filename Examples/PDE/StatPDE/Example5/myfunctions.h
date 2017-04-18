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

#ifndef MYFUNCTIONS_H_
#define MYFUNCTIONS_H_

using namespace std;
using namespace dealii;

#include <deal.II/base/numbers.h>
#include <wrapper/function_wrapper.h>

namespace DOpE
{
  class ExactSolution : public DOpEWrapper::Function<2>
  {
  public:
    ExactSolution() :
      DOpEWrapper::Function<2>(1)
    {
    }

    virtual double
    value(const Point<2> &p, const unsigned int component = 0) const;

    virtual double
    laplacian(const Point<2> &p, const unsigned int component = 0) const;

  };

  /******************************************************/

  double
  ExactSolution::value(const Point<2> &p, const unsigned int component) const
  {
    Assert(component < this->n_components,
           ExcIndexRange(component, 0, this->n_components));

    const double x = p[0];
    const double y = p[1];
    const double pi = numbers::PI;
    double erg = 0;
    switch (component)
      {
      case 0:
        erg = sin(pi / (x * x + y * y));
        break;
      default:
        erg = -123123123.;
        break;
      }
    return erg;
  }

  /******************************************************/

  double
  ExactSolution::laplacian(const Point<2> &p,
                           const unsigned int component) const
  {
    Assert(component < this->n_components,
           ExcIndexRange(component, 0, this->n_components));

    const double x = p[0];
    const double y = p[1];
    const double pi = numbers::PI;
    const double x2 = x * x;
    const double x4 = x2 * x2;
    const double y2 = y * y;
    const double y4 = y2 * y2;
    double erg = 0;
    switch (component)
      {
      case 0:
        erg = -2 * pi
              * (2 * pi * x2 * sin(pi / (x2 + y2))
                 + (-3 * x4 - 2 * x2 * y2 + y4) * cos(pi / (x2 + y2)))
              / (std::pow(x2 + y2, 4.))
              - 2 * pi
              * (2 * pi * y2 * sin(pi / (x2 + y2))
                 + (-3 * y4 - 2 * x2 * y2 + x4) * cos(pi / (x2 + y2)))
              / (std::pow(x2 + y2, 4.));
        break;
      default:
        erg = -123123123.;
        break;
      }
    return erg;
  }

}

#endif /* MYFUNCTIONS_H_ */
