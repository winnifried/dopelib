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

using namespace std;
using namespace dealii;
using namespace DOpE;

template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
  template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
  template<int, int> class DH, typename VECTOR, int dopedim, int dealdim =
  dopedim>
class LocalFunctional : public FunctionalInterface<EDC, FDC, DH, VECTOR,
  dopedim, dealdim>
{
public:
  LocalFunctional(double alpha)
  {
    alpha_ = alpha;
  }

  double
  ElementValue(const EDC<DH, VECTOR, dealdim> &edc)
  {
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
    edc.GetFEValuesState();
    unsigned int n_q_points = edc.GetNQPoints();

//    uvalues_.resize(n_q_points, Vector<double>(3));
//    evalues_.resize(n_q_points, Vector<double>(3));
    qvalues_.resize(n_q_points, Vector<double>(2));
    qgrads_.resize(n_q_points, vector<Tensor<1, dealdim> >(2));

    edc.GetValuesControl("control", qvalues_);
    edc.GetGradsControl("control", qgrads_);
//    edc.GetValuesState("state", uvalues_);
//    edc.GetValuesState("eigenvalue", evalues_);

    Tensor<2, 2> qgrads;
    Tensor<2, 2> DF;
    double detDF;
//    Tensor<1, 2> u;
//    double ev =0;
//    double fval = 3.;
    double r = 0.;


    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
//    	u.clear();
    	DF.clear();
//    	ev = evalues_[q_point][0];
//    	std::cout << ev << std::endl;
    	qgrads.clear();
		qgrads[0][0] = qgrads_[q_point][0][0];
		qgrads[1][1] = qgrads_[q_point][1][1];
		qgrads[1][0] = qgrads_[q_point][1][0];
		qgrads[0][1] = qgrads_[q_point][0][1];

//		u[0] = uvalues_[q_point][1];
//		u[1] = uvalues_[q_point][2];


		DF = calc_DF(qgrads);
		detDF = calc_detDF(DF);

//		 r +=  0.5*(ev- fval)*(ev - fval)*(u[0]*u[0]+u[1]*u[1])* state_fe_values.JxW(q_point);

         r += 0.5 * alpha_ * (qvalues_[q_point][0] * qvalues_[q_point][0] + qvalues_[q_point][1] * qvalues_[q_point][1])
             * state_fe_values.JxW(q_point);


         //qgradient*qgradient für beide Komponenten

//         r += 0.5 * alpha_  * (scalar_product(qgrads[0],qgrads[0])+scalar_product(qgrads[1],qgrads[1])) * state_fe_values.JxW(q_point);
////////DF
////////         //  log(detDF)
//         r +=  0.5 *alpha_ * std::log(detDF)* std::log(detDF) * state_fe_values.JxW(q_point);
      }


    return r;

  }
  double AlgebraicValue(const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
                         const std::map<std::string, const VECTOR *> &/*domain_values*/, double eigenvalue)
   {
     assert(this->GetProblemType() == "cost_functional");
//std::cout << eigenvalue << std::endl;
     double fval = 3.; //TODO übergeben
     return 0.5*(eigenvalue- fval)*(eigenvalue - fval);
//    return 0.5*(vals->second->operator[](0) - fval)*(vals->second->operator[](0) - fval);

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
    edc.GetValuesControl("control", qvalues_);
    edc.GetGradsControl("control", qgrads_);

    Tensor<2, 2> qgrads;
    Tensor<2, 2> DFdq;
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
    	qgrads[0][0] = qgrads_[q_point][0][0];
    	qgrads[1][1] = qgrads_[q_point][1][1];
    	qgrads[1][0] = qgrads_[q_point][1][0];
    	qgrads[0][1] = qgrads_[q_point][0][1];

    	DFdq.clear();

    	DF = calc_DF(qgrads);
    	detDF = calc_detDF(DF);

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
        	phi_q[i]=  control_fe_values[dv].value(i,q_point);
        	grad_phi_v[i]=control_fe_values[dv].gradient(i,q_point);

        	DFdq = calc_DF(grad_phi_v[i]);
        	detDFdq =  -(1/detDF)*(1/detDF)*
					((qgrads[0][0]+1)*grad_phi_v[i][1][1]
					+ (qgrads[1][1]+1)*grad_phi_v[i][0][0]
					-qgrads[1][0]*grad_phi_v[i][0][1]
					-qgrads[0][1]*grad_phi_v[i][1][0] );


        	local_vector(i) +=  scale * alpha_ * (qvalues_[q_point][0]*phi_q[i][0]+qvalues_[q_point][1]*phi_q[i][1])
        	        	              * control_fe_values.JxW(q_point);
////
//        	local_vector(i) += scale * alpha_  *(scalar_product(grad_phi_v[i][0],qgrads[0])+scalar_product(grad_phi_v[i][1],qgrads[1])) * state_fe_values.JxW(q_point);
////
//        	local_vector(i) += scale *alpha_  *std::log(detDF)* std::log(std::abs(detDFdq)) * state_fe_values.JxW(q_point);
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

private:
  vector<Vector<double> > uvalues_;
  vector<Vector<double> > evalues_;
  vector<Vector<double> > qvalues_;
  vector<Vector<double> > funcgradvalues_;

  vector<vector<Tensor<1, dealdim> > > qgrads_;
  double alpha_;

  // ---------------------------------------------------------------
  // Hier alle Funktionen, die in Bezug zu DF benötigt werden
  	Tensor<2, 2> calc_DF(Tensor<2, 2> qgrads) {
  		Tensor<2, 2> DF_;
  		DF_[0][0] = qgrads[0][0] + 1;
  		DF_[1][1] = qgrads[1][1] + 1;
  		DF_[1][0] = qgrads[1][0];
  		DF_[0][1] = qgrads[0][1];

  		return DF_;
  	}
  	double calc_detDF(Tensor<2, 2> DF) {

  		return DF[0][0] * DF[1][1] - DF[1][0] * DF[0][1];
  	}
};
#endif
