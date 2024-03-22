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

#ifndef PROBLEM_DATA_H
#define PROBLEM_DATA_H

/******************************************************/

class InitialData: public DOpEWrapper::Function<2>
{
public:
  InitialData(double meshsize, bool interpolate) :
    DOpEWrapper::Function<2>(4)
  {
    h_ = meshsize;
    interpolate_ = interpolate;
  }
  virtual double value(const Point<2> &p, const unsigned int component = 0) const override;
  virtual void vector_value(const Point<2> &p, Vector<double> &value) const override;
  void SetParams(double d)
  {
    h_ = d;
  }

private:
  double h_;
  bool interpolate_;

};

/******************************************************/

double InitialData::value(const Point<2> &p, const unsigned int component) const
{
  // Set phase-field to "1" since domain is unbroken at the beginning
  if (component == 2)
    {
      if (fabs(p[0]) <= 1. && fabs(p[1]) <= h_)
        return 0.0;
      if (interpolate_)
        {
          if (fabs(p[0]) <= 1. && fabs(p[1]) <= 2.*h_)
            {
              //Interpolate linearly  between 0 und 1
              return (fabs(p[1])-h_)/h_;
            }
          if (fabs(p[0]) <= 1.+h_ && fabs(p[1]) <= h_)
            {
              //Interpolate linearly between 0 und 1
              return (fabs(p[0])-1.)/h_;
            }

          if (fabs(p[0]) <= 1.+h_ && fabs(p[1]) <= 2.*h_)
            {
              //Interpolate bilinearly from 0 und 1
              double x = (fabs(p[0])-1.)/h_; //in [0,1]
              double y = (fabs(p[1])-h_)/h_; //in [0,1]
              //Bilineare function g auf [0,1] mit Werten
              //g(0,0) = 0, g(0,1) = g(1,0) = g(1,1) = 1
              return 1.-(1.-x)*(1.-y);
            }
        }
      return 1.0;
    }
  else
    return 0.0;
}


/******************************************************/

void InitialData::vector_value(const Point<2> &p, Vector<double> &values) const
{
  for (unsigned int c = 0; c < this->n_components; ++c)
    values(c) = InitialData::value(p, c);
}


//////////////////prescribed pressure

double driving_pressure(const Point<2> &p, int c, double d)
{
  if (c == 0)
    {
      return 1.e-3;
    }
  else if (c == 1)
    {
      double x = p[0]-1;
      double y = p[1];
      return 1./(M_PI*sqrt(d))*exp(-x*x/sqrt(d))*exp(-y*y/sqrt(d));
    }
  else if (c == 2)
    {
      double x = p[0]-1;
      double y = p[1];
      return 1./(M_PI*2*d)*exp(-x*x/(2*d))*exp(-y*y/(2*d));
    }
  else if (c == 3)
    {
      double x = p[0]-0.5;
      double y = p[1];
      double scale = 1.;
      if (fabs(y) < 0.5)
        scale = 1.;
      else if (fabs(y) < 1.5)
        scale = 2*(fabs(y)-1.5)*(fabs(y)-1.5)*(fabs(y)+0.);
      else
        scale = 0.;

      if (x <= 0.)
        if (x > -1.)
          return -1.e-3*2*(x+1)*(x+1)*(x-0.5)*scale;
        else
          return 0.;
      else if (x < 1)
        return 1.e-3*scale;
      else if (1 <= x && x < 2 )
        return 1.e-3*2*(x-2)*(x-2)*(x-0.5)*scale;
      else
        return 0.;
    }
  else if (c == 4)
    {
      double x = p[0];
      double y = p[1];
      double scale = 1.;
      if (fabs(y) < 0.5)
        scale = 1.;
      else if (fabs(y) < 1.5)
        scale = 2*(fabs(y)-1.5)*(fabs(y)-1.5)*(fabs(y)+0.);
      else
        scale = 0.;

      if (fabs(x) <= 1.)
        return 1.e-3*scale;
      else if (fabs(x) <= 2.)
        return 1.e-3*2*(fabs(x)-2)*(fabs(x)-2)*(fabs(x)-0.5)*scale;
      else
        return 0.;
    }
  else
    {
      std::cout<<"Unknown pressure case "<<c<<std::endl;
      abort();
    }
}
void driving_pressure_gradient(const Point<2> &p, Tensor<1,2> &grad, int c, double d)
{
  if (c == 0)
    {
      grad[0] = grad[1] = 0.;
    }
  else if (c == 1)
    {
      double x = p[0]-1;
      double y = p[1];
      grad[0] = -2.*x/(M_PI*d)*exp(-x*x/sqrt(d))*exp(-y*y/sqrt(d));
      grad[1] = -2.*y/(M_PI*d)*exp(-x*x/sqrt(d))*exp(-y*y/sqrt(d));
    }
  else if (c == 2)
    {
      double x = p[0]-1;
      double y = p[1];
      grad[0] = -2.*x/(M_PI*4*d*d)*exp(-x*x/(2*d))*exp(-y*y/(2*d));
      grad[1] = -2.*y/(M_PI*4*d*d)*exp(-x*x/(2*d))*exp(-y*y/(2*d));
    }
  else if (c == 3)
    {
      double x = p[0]-0.5;
      double y = p[1];
      grad[1] = 0.;
      grad[0] = 0.;
      double scale = 1.;
      if (fabs(y) < 0.5)
        {
          grad[1] = 0.;
          scale = 1.;
        }
      else if (fabs(y) < 1.5)
        {
          double signy = 1.;
          if (y < 0)
            {
              signy=-1.;
            }
          grad[1] = signy*2*(2*(fabs(y)-1.5)*(fabs(y)+0.) + (fabs(y)-1.5)*(fabs(y)-1.5));
          scale = 2*(fabs(y)-1.5)*(fabs(y)-1.5)*(fabs(y)+0.);
        }
      else
        {
          grad[1] = 0.;
          scale = 0.;
        }

      if (x <= 0.)
        {
          if (x > -1.)
            {
              grad[0] = -1.e-3*2*(2.*(x+1)*(x-0.5)+(x+1)*(x+1))*scale;
              grad[1] *= -1.e-3*2*(x+1)*(x+1)*(x-0.5);
            }
          else
            {
              grad[1] = 0.;
            }
        }
      else
        {
          if ( x < 1)
            {
              grad[1] *= 1.e-3;
            }
          else if (1<= x && x < 2)
            {
              grad[0] = 1.e-3*2*(2*(x-2)*(x-0.5)+(x-2)*(x-2))*scale;
              grad[1] *= 1.e-3*2*(x-2)*(x-2)*(x-0.5);
            }
          else
            grad[1] = 0.;
        }
    }
  else if (c == 4)
    {
      double x = p[0];
      double y = p[1];
      grad[1] = 0.;
      grad[0] = 0.;
      double scale = 1.;
      if (fabs(y) < 0.5)
        {
          grad[1] = 0.;
          scale = 1.;
        }
      else if (fabs(y) < 1.5)
        {
          double signy = 1.;
          if (y < 0)
            {
              signy=-1.;
            }
          grad[1] = signy*2*(2*(fabs(y)-1.5)*(fabs(y)+0.) + (fabs(y)-1.5)*(fabs(y)-1.5));
          scale = 2*(fabs(y)-1.5)*(fabs(y)-1.5)*(fabs(y)+0.);
        }
      else
        {
          grad[1] = 0.;
          scale = 0.;
        }

      if (fabs(x) <= 1.)
        {
          grad[0] = 0.;
          grad[1] *= 1.e-3;
        }
      else if (fabs(x) <= 2.)
        {
          double signx = 1.;
          if (x < 0)
            {
              signx=-1.;
            }
          grad[0] = signx*1.e-3*2*(2*(fabs(x)-2)*(fabs(x)-0.5)+(fabs(x)-2)*(fabs(x)-2))*scale;
          grad[1] *= 1.e-3*2*(fabs(x)-2)*(fabs(x)-2)*(fabs(x)-0.5);
        }
      else
        {
          grad[0] = 0.;
          grad[1] = 0.;
        }
    }
  else
    {
      std::cout<<"Unknown pressure case "<<c<<std::endl;
      abort();
    }
}

#endif
