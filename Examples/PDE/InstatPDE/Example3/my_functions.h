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

class InitialData: public DOpEWrapper::Function<2>
{
public:
  InitialData(ParameterReader &param_reader) :
    DOpEWrapper::Function<2>()
  {
    param_reader.SetSubsection("Local PDE parameters");
    strike_= param_reader.get_double("strike price");
  }
  virtual double value(const Point<2> &p, const unsigned int component = 0) const;
  virtual void vector_value(const Point<2> &p, Vector<double> &value) const;
  static void declare_params(ParameterReader &param_reader);
private:
  double strike_;

};

/******************************************************/

double InitialData::value(const Point<2> &p, const unsigned int /*component*/) const
{
  double x = p[0];
  double y = p[1];

  return std::max(strike_ - 0.5 * x - 0.5 * y, 0.);
}

/******************************************************/

void InitialData::vector_value(const Point<2> &p, Vector<double> &values) const
{
  for (unsigned int c = 0; c < this->n_components; ++c)
    values(c) = InitialData::value(p, c);
}

/******************************************************/

void InitialData::declare_params(ParameterReader &param_reader)
{
  param_reader.SetSubsection("Local PDE parameters");
  param_reader.declare_entry("strike price", "0.0", Patterns::Double(0));
}
