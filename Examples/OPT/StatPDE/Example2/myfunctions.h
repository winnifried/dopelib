/*
 * myfunctions.h
 *
 *  Created on: 05.05.2012
 *      Author: cgoll
 */

#ifndef _MYFUNCTIONS_H_
#define _MYFUNCTIONS_H_

#include <base/function.h>

namespace DOpE
{
  class ExactU : public DOpEWrapper::Function<2>
  {
    public:
      ExactU() :
          DOpEWrapper::Function<2>(2)
      {
      }

      virtual double
      value(const Point<2> &p, const unsigned int component = 0) const
      {
        assert(component<2);
        const double x = p[0];
        const double y = p[1];
        double r;
        switch (component)
        {
        case 0:
          r = std::sin(M_PI * x) * (std::sin(M_PI * y) + 0.5 * std::sin(2 * M_PI * y));
          break;
        case 1:
          r = std::sin(2*M_PI * x) * std::sin(2*M_PI * y);
          break;
        default:
          r=-999999999999;
          break;
        }

        return r;
      }
  };
}

#endif /* MYFUNCTIONS_H_ */
