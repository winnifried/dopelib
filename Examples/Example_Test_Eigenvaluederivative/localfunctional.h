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

#ifndef LOCALFunctional_
#define LOCALFunctional_

#include <interfaces/functionalinterface.h>
#include "functions.h"

using namespace std;
using namespace dealii;
using namespace DOpE;
using namespace local;

template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
  template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
  template<int, int> class DH, typename VECTOR, int dopedim, int dealdim =
  dopedim>
class LocalFunctional : public FunctionalInterface<EDC, FDC, DH, VECTOR,
  dopedim, dealdim>
{
public:
  LocalFunctional(ParameterReader &param_reader, double alpha)
  {
    alpha_ = alpha;
    param_reader.SetSubsection("localfunctional parameters");
    target_eigenvalue_    = param_reader.get_double("target_eigenvalue");
  }

  double
  ElementValue(const EDC<DH, VECTOR, dealdim> &edc)
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
    edc.GetFEValuesState();
    unsigned int n_q_points = edc.GetNQPoints();

    qvalues_.resize(n_q_points, Vector<double>(2));
    qgrads_.resize(n_q_points, vector<Tensor<1, dealdim> >(2));

//    control_initial_iteration.resize(n_q_points, Vector<double>(2));

    edc.GetValuesControl("control", qvalues_);
    edc.GetGradsControl("control", qgrads_);
//    edc.GetValuesControl("control_counter", control_initial_iteration);
//    if(control_initial_iteration[0][0] != 0 ){
//       qgrads_old_.resize(n_q_points, vector<Tensor<1, dealdim> >(2));
//       edc.GetGradsControl("q_previous", qgrads_old_);
//   }
    Tensor<2, 2> qgrads;
    Tensor<2, 2> qgrads_old;
    Tensor<2, 2> DF;
    double detDF;
    double r = 0.;

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
    	DF.clear();
    	qgrads.clear();
    	qgrads_old.clear();
		qgrads[0][0] = qgrads_[q_point][0][0];
		qgrads[1][1] = qgrads_[q_point][1][1];
		qgrads[1][0] = qgrads_[q_point][1][0];
		qgrads[0][1] = qgrads_[q_point][0][1];

//		if(control_initial_iteration[0][0]!= 0){
//		    	qgrads_old[0][0] = qgrads_old_[q_point][0][0];
//		    	qgrads_old[1][1] = qgrads_old_[q_point][1][1];
//		    	qgrads_old[1][0] = qgrads_old_[q_point][1][0];
//		    	qgrads_old[0][1] = qgrads_old_[q_point][0][1];
//		}

		DF = deformation_tensor_(qgrads);
		detDF = determinante_(DF);
		if(detDF < 0.001){
			r = 10.e20;
		} else{
		         r += 0.5 * alpha_* (qvalues_[q_point][0] * qvalues_[q_point][0] + qvalues_[q_point][1] * qvalues_[q_point][1])
             * state_fe_values.JxW(q_point);

//		if(control_initial_iteration[0][0] == 0){
			r += 0.5 * alpha_  * (scalar_product(qgrads[0],qgrads[0])+scalar_product(qgrads[1],qgrads[1])) * state_fe_values.JxW(q_point);
//		}else{
//			r += 0.5 * alpha_  * (scalar_product((qgrads[0] - qgrads_old_[q_point][0] ),(qgrads[0]-qgrads_old_[q_point][0]))+scalar_product((qgrads[1]-qgrads_old_[q_point][1]),(qgrads[1]-qgrads_old_[q_point][1]))) * state_fe_values.JxW(q_point);
//       }
		}
      }
    return r;
  }

  double AlgebraicValue(const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
                         const std::map<std::string, const VECTOR *> &/*domain_values*/, double eigenvalue)
   {
     assert(this->GetProblemType() == "cost_functional");

     return 0.5*(eigenvalue- target_eigenvalue_)*(eigenvalue - target_eigenvalue_);

   }

  double AlgebraicValue_U(const EDC<DH, VECTOR, dealdim> &edc,
          dealii::Vector<double> &local_vector, double scale, double eigenvalue)
   {

     return eigenvalue - target_eigenvalue_; //TODO Rueckgabe Void
   }

  void
  ElementValue_Q(const EDC<DH, VECTOR, dealdim> &edc,
                 dealii::Vector<double> &local_vector, double scale)
  {
    const DOpEWrapper::FEValues<dealdim> &control_fe_values =
      edc.GetFEValuesControl();
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();

    qvalues_.resize(n_q_points, Vector<double>(2));
    qgrads_.resize(n_q_points, vector<Tensor<1, dealdim> >(2));

//    control_initial_iteration.resize(n_q_points, Vector<double>(2));

    edc.GetValuesControl("control", qvalues_);
    edc.GetGradsControl("control", qgrads_);

//   edc.GetValuesControl("control_counter", control_initial_iteration);
//    if(control_initial_iteration[0][0] != 0 ){
//    	 qgrads_old_.resize(n_q_points, vector<Tensor<1, dealdim> >(2));
//    	 edc.GetGradsControl("q_previous", qgrads_old_);
//    }

    Tensor<2, 2> qgrads;
    Tensor<2, 2> qgrads_old;
    Tensor<2, 2> DF;
    double detDF;
    double detDFdq;

	vector<Tensor<1, dealdim> > phi_q(n_dofs_per_element);
	vector<Tensor<dealdim, dealdim> > grad_phi_v(n_dofs_per_element);
	const FEValuesExtractors::Vector dv(0);
    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
    {
    	DF.clear();
    	qgrads.clear();
    	qgrads_old.clear();

    	qgrads[0][0] = qgrads_[q_point][0][0];
    	qgrads[1][1] = qgrads_[q_point][1][1];
    	qgrads[1][0] = qgrads_[q_point][1][0];
    	qgrads[0][1] = qgrads_[q_point][0][1];

//    	if(control_initial_iteration[0][0] == 1){
//
//    	qgrads_old[0][0] = qgrads_old_[q_point][0][0];
//    	qgrads_old[1][1] = qgrads_old_[q_point][1][1];
//    	qgrads_old[1][0] = qgrads_old_[q_point][1][0];
//    	qgrads_old[0][1] = qgrads_old_[q_point][0][1];
//    	}

    	DF = deformation_tensor_(qgrads);
    	detDF = determinante_(DF);

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
        	phi_q[i]=  control_fe_values[dv].value(i,q_point);
        	grad_phi_v[i]=control_fe_values[dv].gradient(i,q_point);

        	detDFdq =  ((qgrads[0][0]+1)*grad_phi_v[i][1][1]
					+ (qgrads[1][1]+1)*grad_phi_v[i][0][0]
					-qgrads[1][0]*grad_phi_v[i][0][1]
					-qgrads[0][1]*grad_phi_v[i][1][0] );


        	local_vector(i) +=  scale * alpha_ *(qvalues_[q_point][0]*phi_q[i][0]+qvalues_[q_point][1]*phi_q[i][1])
        	        	              * state_fe_values.JxW(q_point);

//        	if(control_initial_iteration[0][0] == 0){
            	local_vector(i) += scale * alpha_  *(scalar_product(grad_phi_v[i][0],qgrads[0])+scalar_product(grad_phi_v[i][1],qgrads[1])) * state_fe_values.JxW(q_point);

//        	}else{
////        		std::cout << " detDF_alt = " << determinante_(deformation_tensor_(qgrads_old))<< std::endl;
//            	local_vector(i) += scale * alpha_  *(scalar_product(grad_phi_v[i][0],qgrads[0]-qgrads_old[0])+scalar_product(grad_phi_v[i][1],qgrads[1]-qgrads_old[1])) * state_fe_values.JxW(q_point);
//
//        	}
          }
      }

  }


  UpdateFlags
  GetUpdateFlags() const
  {
    return update_values | update_gradients | update_quadrature_points | update_JxW_values;
  }

  string
  GetType() const
  {
    return "domain algebraic";
  }

  string
  GetName() const
  {
    return "cost functional";
  }

  static void
  declare_params(ParameterReader &param_reader)
  {
    param_reader.SetSubsection("localfunctional parameters");
    param_reader.declare_entry("target_eigenvalue", "1.5", Patterns::Double(0));
  }

private:
  vector<Vector<double> > uvalues_;
  vector<Vector<double> > qvalues_;
//  vector<Vector<double> > qvalues_old_;
//  vector<Vector<double> > control_initial_iteration;
  vector<Vector< double > >lambda_target_value_;

  vector<Vector<double> > funcgradvalues_;

  vector<vector<Tensor<1, dealdim> > > qgrads_;
//  vector<vector<Tensor<1, dealdim> > > qgrads_old_;

  double alpha_ ;
  double target_eigenvalue_;



};
#endif
