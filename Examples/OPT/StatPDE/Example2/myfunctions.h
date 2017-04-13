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

#ifndef MYFUNCTIONS_H_
#define MYFUNCTIONS_H_

#include <deal.II/base/function.h>

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
