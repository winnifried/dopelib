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

#include <wrapper/function_wrapper.h>

using namespace dealii;

/******************************************************/
/**
 * This function defines the inflowcondition.
 */

class BoundaryParabel : public DOpEWrapper::Function<2>
{
public:
  BoundaryParabel() :
    DOpEWrapper::Function<2>(5)
  {
  }

  /**
   * Returns the value of the component 'component' of the function
   * in point 'p'.
   */
  virtual double
  value(const Point<2> &p, const unsigned int component = 0) const;

  /**
   * Returns the value  of the function  in point 'p'.
   */

  virtual void
  vector_value(const Point<2> &p, Vector<double> &value) const;

private:

};

/******************************************************/

double
BoundaryParabel::value(const Point<2> &p, const unsigned int component) const
{
  Assert(component < this->n_components,
         ExcIndexRange (component, 0, this->n_components));

  double damping_inflow = 1.0;

  if (component == 0)
    {
      return (
               (p(0) == -6.0) && (p(1) <= 2.0) ? -damping_inflow
               * (std::pow(p(1), 2) - 2.0 * std::pow(p(1), 1)) :
               0);

    }
  return 0;
}

/******************************************************/

void
BoundaryParabel::vector_value(const Point<2> &p, Vector<double> &values) const
{
  for (unsigned int c = 0; c < this->n_components; ++c)
    values(c) = BoundaryParabel::value(p, c);
}
