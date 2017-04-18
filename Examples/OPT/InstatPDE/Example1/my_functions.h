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

#ifndef MY_FUNCTIONS_
#define MY_FUNCTIONS_

#include <wrapper/function_wrapper.h>

using namespace dealii;

/******************************************************/

class RightHandSideFunction : public DOpEWrapper::Function<2>
{
public:
  RightHandSideFunction() :
    DOpEWrapper::Function<2>(), mytime(0)
  {

  }
  virtual double
  value(const Point<2> &p, const unsigned int component = 0) const;

  void
  SetTime(double t) const
  {
    mytime = t;
  }

private:
  mutable double mytime;

};

/******************************************************/

double
RightHandSideFunction::value(const Point<2> &p,
                             const unsigned int/* component*/) const
{
  return ((3 - 2 * mytime) * std::exp(mytime - mytime * mytime) * sin(p[0])
          * sin(p[1])
          + std::exp(mytime - mytime * mytime) * sin(p[0]) * sin(p[1])
          * std::exp(mytime - mytime * mytime) * sin(p[0]) * sin(p[1]));
}

/******************************************************/

#endif
