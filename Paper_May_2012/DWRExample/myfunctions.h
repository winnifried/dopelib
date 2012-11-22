/*
 * myfunctions.h
 *
 *  Created on: Apr 10, 2012
 *      Author: cgoll
 */

#ifndef MYFUNCTIONS_H_
#define MYFUNCTIONS_H_

using namespace std;
using namespace dealii;
using namespace DOpE;

#include "function_wrapper.h"
#include "parameterreader.h"

class OneExtension : public DOpEWrapper::Function<2>
{
  public:
    OneExtension(/*ParameterReader &param_reader*/)
        : DOpEWrapper::Function<2>(3), _center(0.2, 0.2), _radius(0.05)
    {

    }

    virtual double
    value(const Point<2> &p, const unsigned int component = 0) const
    {
      double erg = 0;
      if (component == 0)
      {
        if (std::fabs(p.distance(_center) - _radius) < 1e-12)
          return 1;
        else
          return 0;
      }
      return erg;
    }

//    virtual void
//    vector_value(const Point<2> &p, Vector<double> &value) const
//    {
//
//    }
  private:
    const Point<2> _center;
    const double _radius;
};

/*************************************************************************************************/

class BoundaryParabel : public DOpEWrapper::Function<2>
{
  public:
    BoundaryParabel(ParameterReader &param_reader)
        : DOpEWrapper::Function<2>(3)
    {
      param_reader.SetSubsection("My functions parameters");
      _mean_inflow_velocity = param_reader.get_double("mean_inflow_velocity");
    }

    virtual double
    value(const Point<2> &p, const unsigned int component = 0) const;

    virtual void
    vector_value(const Point<2> &p, Vector<double> &value) const;

    static void
    declare_params(ParameterReader &param_reader)
    {
      param_reader.SetSubsection("My functions parameters");
      param_reader.declare_entry("mean_inflow_velocity", "0.0",
          Patterns::Double(0));
    }

  private:
    double _mean_inflow_velocity;

};

/******************************************************/

double
BoundaryParabel::value(const Point<2> &p, const unsigned int component) const
{
  Assert(component < this->n_components,
      ExcIndexRange(component, 0, this->n_components));

  //double _mean_inflow_velocity = 1.5;

  if (component == 0)
  {

    // Benchmark: BFAC 2D-1, 2D-2
    return (
        (p(0) == 0) && (p(1) <= 0.41) ?
            -_mean_inflow_velocity * (4.0 / 0.1681)
                * (std::pow(p(1), 2) - 0.41 * std::pow(p(1), 1)) :
            0);

//    // Benchmark: BFAC 2D-3
//    return (
//        (p(0) == 0) && (p(1) <= 0.41) ? -_mean_inflow_velocity * (4.0 / 0.1681)
//            * (std::pow(p(1), 2) - 0.41 * std::pow(p(1), 1)) :
//            0);

    /*
     // Channel problem
     return   ( (p(0) == -6.0) && (p(1) <= 2.0)  ? - _mean_inflow_velocity*
     (std::pow(p(1), 2) - 2.0 * std::pow(p(1),1)) : 0 );
     */

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

#endif /* MYFUNCTIONS_H_ */
