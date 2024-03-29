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

template <int dim>
class ExactSolution : public Function<dim>
{
public :
  ExactSolution() : Function<dim> (dim + 1) {}

  virtual double value (const Point<dim> &p,
                        unsigned int component) const override;

  virtual void vector_value (const Point<dim> &p,
                             Vector<double> &values) const override;

  virtual Tensor<1, dim> gradient (const Point<dim> &p,
                                   const unsigned int component) const override;
private:
  const double lambda = 0.3;
};

/* ---------------------------------------------------------- */
template <int dim>
double
ExactSolution<dim>::value(const Point<dim> &p,
                          const unsigned int component) const
{
  double return_value = 0;
  const double x = p[0];
  const double y = p[1];

  if (component == 0)
    return_value = ( 200*x*x*(1-x)*(1-x)*y*(1-y)*(1-2*y) );

  if (component == 1)
    return_value = ( -200*y*y*(1-y)*(1-y)*x*(1-x)*(1-2*x) );

  if (component == 2)
    return_value = -( 10*(x-0.5)*(x-0.5)*(x-0.5)*y*y +
                      (1-x)*(1-x)*(1-x)*(y-0.5)*(y-0.5)*(y-0.5) -(1./8.) );

  return (return_value);
}

/* ---------------------------------------------------------- */
template <int dim>
void
ExactSolution<dim>::vector_value (const Point<dim> &p,
                                  Vector<double> &values) const
{
  for ( unsigned int c = 0; c <this->n_components; ++c)
    values(c) = ExactSolution<dim>::value(p,c);
}

/* ---------------------------------------------------------- */
template <int dim>
Tensor<1, dim>
ExactSolution<dim>::gradient(const Point<dim> &p,
                             const unsigned int component) const
{

  Tensor<1, dim> return_value;

  const double x = p[0];
  const double y = p[1];

  if (component == 0)
    {
      return_value[0] = 400*x*y*(1-y)*(1-2*y)*(1-x)*(1-2*x);
      return_value[1] = 200*x*x*(1-x)*(1-x)*(6*y*y - 6*y + 1);
    }

  if (component == 1)
    {
      return_value[0] = 200*y*y*(1-y)*(1-y)*(-6*x*x + 6*x -1);
      return_value[1] = 400*x*y*(1-x)*(2*x-1)*(1-y)*(1-2*y);
    }

  if (component == 2)
    {
      return_value[0] = -( 30*y*y*(x-0.5)*(x-0.5) - 3*(y-0.5)*(y-0.5)*(y-0.5)*(1-x)*(1-x) );
      return_value[1] = -( 20*y*(x-0.5)*(x-0.5)*(x-0.5) + 3*(1-x)*(1-x)*(1-x)*(y-0.5)*(y-0.5));
    }

  return (return_value);

}

/* ---------------------------------------------------------- */
template<int dim>
class BoundaryValues : public DOpEWrapper::Function<dim>
{
public:
  BoundaryValues () : DOpEWrapper::Function<dim>(dim + 1) {}

  virtual double value (const Point<dim> &p,
                        unsigned int component) const override;

  virtual void vector_value (const Point<dim> &p,
                             Vector<double> &values) const override;

private:
  const double lambda = 0.3;
};

/* ---------------------------------------------------------- */
template <int dim>
double
BoundaryValues<dim>::value (const Point<dim> &p,
                            const unsigned int component) const
{
  double return_value = 0;

  const double x = p[0];
  const double y = p[1];

  if (component == 0)
    return_value = ( 200*x*x*(1-x)*(1-x)*y*(1-y)*(1-2*y) );

  if (component == 1)
    return_value = ( 200*y*y*(1-y)*(1-y)*x*(1-x)*(2*x-1) );

  if (component == 2)
    return_value = -( 10*(x-0.5)*(x-0.5)*(x-0.5)*y*y +
                      (1-x)*(1-x)*(1-x)*(y-0.5)*(y-0.5)*(y-0.5) - ( 1./8.) );

  return (return_value);
}

/* ---------------------------------------------------------- */
template <int dim>
void
BoundaryValues<dim>::vector_value (const Point<dim> &p,
                                   Vector<double> &values) const
{
  for ( unsigned int c = 0; c < this->n_components; ++c)
    values(c) = BoundaryValues<dim>::value(p,c);
}

/* ---------------------------------------------------------- */

