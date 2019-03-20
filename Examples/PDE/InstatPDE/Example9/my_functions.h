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

#ifndef MY_FUNCTIONS_
#define MY_FUNCTIONS_

#include <wrapper/function_wrapper.h>

using namespace dealii;

class InitialData : public DOpEWrapper::Function<2>
{
public:
  InitialData() :
    DOpEWrapper::Function<2>()
  {

  }
  virtual double
  value(const Point<2> &p, const unsigned int component = 0) const;
  virtual void
  vector_value(const Point<2> &p, Vector<double> &value) const;

private:

};

/******************************************************/

double
InitialData::value(const Point<2> &p, const unsigned int /*component*/) const
{
  double x = p[0];

  return std::min(x, 1. - x);

}

/******************************************************/

void
InitialData::vector_value(const Point<2> &p, Vector<double> &values) const
{
  for (unsigned int c = 0; c < this->n_components; ++c)
    values(c) = InitialData::value(p, c);
}

/******************************************************/


class DirichletValues : public DOpEWrapper::Function<2>
{
public:
  DirichletValues () : DOpEWrapper::Function<2>()
  {
  }

  double value (const Point<2>   &p,
		const unsigned int /*component*/ = 0) const
  {
    double y = p[1];
    return mytime*y;
  }

  void vector_value (const Point<2> &p,
		     Vector<double>   &value) const
  {
    for (unsigned int c=0; c<this->n_components; ++c)
      value (c) = DirichletValues::value (p, c);
  }

  void SetTime(double t) const
  {
    mytime=t;
  }

private:
  mutable double mytime;
};


/******************************************************/

#endif
