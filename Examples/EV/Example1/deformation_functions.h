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

#ifndef DEFORMATION_FUNCTIONS_H_
#define DEFORMATION_FUNCTIONS_H_

using namespace std;
using namespace dealii;

#include <deal.II/base/numbers.h>
#include <deal.II/base/function.h>
#include <wrapper/function_wrapper.h>

namespace deformation_functions
{
  /*********************************************************/
   // all functions used for calculations with DF

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


    // all functions used for calculations with DFdq
   double determinante_DF_dq_(Tensor<2, 2> qgrads_,Tensor<2, 2> grad_phi_v_i){
	   Tensor<2, 2> DF_ = deformation_tensor_(qgrads_);
	   double det_dq = (DF_[0][0]*grad_phi_v_i[1][1]
			+ DF_[1][1]*grad_phi_v_i[0][0]
			-DF_[1][0]*grad_phi_v_i[0][1]
			-DF_[0][1]*grad_phi_v_i[1][0] );

	   return det_dq;
   }

   double determinante_DF_inv_dq_(Tensor<2, 2> qgrads_,Tensor<2, 2> grad_phi_v_i){
	   Tensor<2, 2>  DF = deformation_tensor_(qgrads_);
	   double detDF = determinante_(DF);
	   double detDFdq = determinante_DF_dq_(qgrads_,grad_phi_v_i);
   	   return -(1/detDF)*(1/detDF)*detDFdq;
      }


   Tensor<2, 2> DF_inv_T_dq_(Tensor<2, 2> qgrads_,Tensor<2, 2> grad_phi_v_i){
	   Tensor<2, 2>  DF = deformation_tensor_(qgrads_);
	   	double detDF = determinante_(DF);
	   	double detDFINVdq = determinante_DF_inv_dq_(qgrads_,grad_phi_v_i);

   		return ((1/detDF)*adjunkte_transposed_(grad_phi_v_i)+detDFINVdq*adjunkte_transposed_(DF));
   }


   Tensor<2, 2> DF_inv_dq_(Tensor<2, 2> qgrads_,Tensor<2, 2> grad_phi_v_i){
  	   Tensor<2, 2>  DF = deformation_tensor_(qgrads_);
  	   	double detDF = determinante_(DF);
  	   	double detDFINVdq = determinante_DF_inv_dq_(qgrads_,grad_phi_v_i);

  	   	return  ((1/detDF)*adjunkte_(grad_phi_v_i)+detDFINVdq*adjunkte_(DF));

     }


























}

#endif
