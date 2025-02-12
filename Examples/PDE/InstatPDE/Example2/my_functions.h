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

#include <wrapper/function_wrapper.h>

using namespace dealii;

/******************************************************/

class BoundaryParabel : public DOpEWrapper::Function<2>
{
public:
  BoundaryParabel(ParameterReader &param_reader) :
    DOpEWrapper::Function<2>(7), mytime(0)
  {
    param_reader.SetSubsection("My functions parameters");
    mean_inflow_velocity = param_reader.get_double("mean_inflow_velocity");
  }

  virtual double
  value(const Point<2> &p, const unsigned int component = 0) const override;

  virtual void
  vector_value(const Point<2> &p, Vector<double> &value) const override;

  static void
  declare_params(ParameterReader &param_reader)
  {
    param_reader.SetSubsection("My functions parameters");
    param_reader.declare_entry("mean_inflow_velocity", "0.0",
                               Patterns::Double(0));
  }

  void
  SetTime(double t) const override
  {
    mytime = t;
  }

private:
  double mean_inflow_velocity;
  mutable double mytime;

};

/******************************************************/

double
BoundaryParabel::value(const Point<2> &p, const unsigned int component) const
{
  Assert(component < this->n_components,
         ExcIndexRange (component, 0, this->n_components));

  if (component == 0)
    {

      /*
       // Channel problem
       return   ( (p(0) == -6.0) && (p(1) <= 2.0)  ? - mean_inflow_velocity *
       (std::pow(p(1), 2) - 2.0 * std::pow(p(1),1)) : 0 );
       */

      /*
       // Fluid Benchmark
       return ( (p(0) == 0) && (p(1) <= 0.41) ? -mean_inflow_velocity *
       (4.0/0.1681) *
       (std::pow(p(1), 2) - 0.41 * std::pow(p(1),1)) : 0 );

       */

      //  FSI Benchmark
      if (mytime < 2.0)
        {
          return (
                   (p(0) == 0) && (p(1) <= 0.41) ? -1.5 * mean_inflow_velocity
                   * (1.0 - std::cos(M_PI / 2.0 * mytime)) / 2.0 * (4.0 / 0.1681)
                   * (std::pow(p(1), 2) - 0.41 * std::pow(p(1), 1)) :
                   0);
        }
      else
        {
          return (
                   (p(0) == 0) && (p(1) <= 0.41) ? -1.5 * mean_inflow_velocity
                   * (4.0 / 0.1681)
                   * (std::pow(p(1), 2) - 0.41 * std::pow(p(1), 1)) :
                   0);
        }

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


