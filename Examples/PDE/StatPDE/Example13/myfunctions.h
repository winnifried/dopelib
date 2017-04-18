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

  };

  /******************************************************/

  double
  ExactSolution::value(const Point<2> &p, const unsigned int /*component*/) const
  {
    const double x = p[0];
    if (x <= 0.5)
      return 1.;
    else
      return 0.;
  }

  /******************************************************/

}

#endif /* MYFUNCTIONS_H_ */
