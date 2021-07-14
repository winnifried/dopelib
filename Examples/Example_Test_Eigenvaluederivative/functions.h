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

#ifndef FUNCTIONS_H_
#define FUNCTIONS_H_

using namespace std;
using namespace dealii;

#include <deal.II/base/numbers.h>
#include <deal.II/base/function.h>
#include <wrapper/function_wrapper.h>

namespace local
{
  class
    Q_Control : public Function<2>
  {
  public:
    Q_Control() : Function<2>(2){}

    double value(const Point<2> &p, const unsigned int component /* = 0*/ ) const
    {

    	const double x = p[0];
    	const double y = p[1];



    	if(component ==0){
    	  return cos(10./180.*M_PI)*x - sin(10./180.*M_PI)*y-x;// 0.0001*x;//  ??3*x-x;//
      }else if(component == 1){
    		return sin(10./180.*M_PI)*x + cos(10./180.*M_PI)*y-y;//0.0001*y;//
    }

      return 0;


   }

    };
}

#endif
