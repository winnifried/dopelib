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

class NonHomoDirichletData : public DOpEWrapper::Function<2>
{
public:
  NonHomoDirichletData (ParameterReader &param_reader) : DOpEWrapper::Function<2>(4)
  {
    param_reader.SetSubsection("Problem data parameters");
    dis_step_per_timestep_ = param_reader.get_double ("dis_step_per_timestep");
  }

  virtual double value (const Point<2>   &p,
                        const unsigned int  component = 0) const;

  virtual void vector_value (const Point<2> &p,
                             Vector<double>   &value) const;

  static void declare_params(ParameterReader &param_reader)
  {
    param_reader.SetSubsection("Problem data parameters");
    param_reader.declare_entry("dis_step_per_timestep", "0.0",
                               Patterns::Double(0));
  }

  void SetTime(double t) const
  {
    localtime=t;
  }

private:
  double dis_step_per_timestep_;
  mutable double localtime;


};

/******************************************************/

double
NonHomoDirichletData::value (const Point<2>  &p,
                             const unsigned int component) const
{
  Assert (component < this->n_components,
          ExcIndexRange (component, 0, this->n_components));

  // Miehe tension test
  if (component == 1)
    {

      return ( ((p(1) == 0.0) )
               ?
               (-1.0) * localtime *dis_step_per_timestep_ : 0 );

    }
  if (component == 0)
    {
	   
      return 0.0;
    }
  return 0;
}

/******************************************************/

void
NonHomoDirichletData::vector_value (const Point<2> &p,
                                    Vector<double>   &values) const
{
  for (unsigned int c=0; c<this->n_components; ++c)
    values (c) = NonHomoDirichletData::value (p, c);
}



/******************************************************/

class InitialData: public DOpEWrapper::Function<2>
{
public:
  InitialData() :
    DOpEWrapper::Function<2>(4)
  {

  }
  virtual double value(const Point<2> &p, const unsigned int component = 0) const;
  virtual void vector_value(const Point<2> &p, Vector<double> &value) const;

private:


};

/******************************************************/

double InitialData::value(const Point<2> &/*p*/, const unsigned int component) const
{

  if (component == 2)
    return 1.0;
  else
    return 0.0;
}

/******************************************************/

void InitialData::vector_value(const Point<2> &p, Vector<double> &values) const
{
  for (unsigned int c = 0; c < this->n_components; ++c)
    values(c) = InitialData::value(p, c);
}

