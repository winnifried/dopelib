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

#include <deal.II/base/numbers.h>

namespace DOpE
{
  class ExactSolution : public DOpEWrapper::Function<2>
  {
    public:
      ExactSolution()
          : DOpEWrapper::Function<2>(1)
      {
      }

      virtual double
      value(const Point<2> &p, const unsigned int component = 0) const;

      virtual double
      laplacian(const Point<2> & p, const unsigned int component = 0) const;

  };

  /******************************************************/

  double
  ExactSolution::value(const Point<2> &p, const unsigned int component) const
  {
    Assert(component < this->n_components,
        ExcIndexRange(component, 0, this->n_components));

    const double x = p[0];
    const double y = p[1];
    const double pi = numbers::PI;
    double erg = 0;
    switch (component)
    {
      case 0:
        erg = sin(pi / (x * x + y * y));
        break;
      default:
        erg = -123123123.;
        break;
    }
    return erg;
  }

  /******************************************************/

  double
  ExactSolution::laplacian(const Point<2> &p,
      const unsigned int component) const
  {
    Assert(component < this->n_components,
        ExcIndexRange(component, 0, this->n_components));

    const double x = p[0];
    const double y = p[1];
    const double pi = numbers::PI;
    const double x2 = x * x;
    const double x4 = x2 * x2;
    const double y2 = y * y;
    const double y4 = y2 * y2;
    double erg = 0;
    switch (component)
    {
      case 0:
        erg = -2 * pi
            * (2 * pi * x2 * sin(pi / (x2 + y2))
                + (-3 * x4 - 2 * x2 * y2 + y4) * cos(pi / (x2 + y2)))
            / (std::pow(x2 + y2, 4.))
            - 2 * pi
                * (2 * pi * y2 * sin(pi / (x2 + y2))
                    + (-3 * y4 - 2 * x2 * y2 + x4) * cos(pi / (x2 + y2)))
                / (std::pow(x2 + y2, 4.));
        break;
      default:
        erg = -123123123.;
        break;
    }
    return erg;
  }

}

#endif /* MYFUNCTIONS_H_ */
