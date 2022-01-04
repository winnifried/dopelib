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

  /******************************************************/
   // Hier alle Funktionen, die in Bezug zu DF benötigt werden

   	Tensor<2, 2> deformation_tensor_(Tensor<2, 2> qgrads) {
   		Tensor<2, 2> DF_;
   		DF_[0][0] = qgrads[0][0] + 1;
   		DF_[1][1] = qgrads[1][1] + 1;
   		DF_[1][0] = qgrads[1][0];
   		DF_[0][1] = qgrads[0][1];

   		return DF_;
   	}
   	double determinante_(Tensor<2, 2> tensor) {
   		return tensor[0][0] * tensor[1][1] - tensor[1][0] * tensor[0][1];
   	}
   	Tensor<2, 2> adjunkte_(Tensor<2, 2> tensor) {
   		Tensor<2, 2> adjunkte;
   		adjunkte[0][0] = tensor[1][1];
   		adjunkte[1][1] = tensor[0][0];
   		adjunkte[1][0] = -tensor[1][0];
   		adjunkte[0][1] = -tensor[0][1];

   		return adjunkte;
   	}

	Tensor<2,2> transpose_(Tensor<2, 2> tensor){
   		Tensor <2,2> transposed_tensor;
   		transposed_tensor[0][0] = tensor[0][0];
   		transposed_tensor[1][1] = tensor[1][1];
   		transposed_tensor[1][0] = tensor[0][1];
   		transposed_tensor[0][1] = tensor[1][0];

   		return transposed_tensor;
   	}

   	Tensor<2, 2> adjunkte_transposed_(Tensor<2, 2> tensor) {
   		Tensor<2, 2> adjunkte_transposed;
   		adjunkte_transposed =transpose_(adjunkte_(tensor));

   		return adjunkte_transposed;
   	}

   	Tensor<2, 2> inverse_(Tensor<2, 2> tensor) {
   		Tensor<2, 2> inverse = (1/determinante_(tensor))*adjunkte_(tensor);

   		return inverse;
   	}

   	Tensor<2, 2> inverse_transpose_(Tensor<2, 2> tensor) {
   		Tensor<2, 2> tensor_inv_t = transpose_(inverse_(tensor));

   		return tensor_inv_t;
   	}



}

#endif
