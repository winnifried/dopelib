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

#ifndef LOCALPDE_
#define LOCALPDE_

#include <interfaces/pdeinterface.h>
#include "functions.h"

using namespace std;
using namespace dealii;
using namespace DOpE;
using namespace local;

template<
		template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
		template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
		template<int, int> class DH, typename VECTOR, int dealdim>
class LocalPDE: public PDEInterface<EDC, FDC, DH, VECTOR, dealdim> {
public:

	static void declare_params(ParameterReader &param_reader) {
	}

	LocalPDE(ParameterReader&/*param_reader*/) :
			state_block_component_(2, 0), control_block_component_(2, 0) {
			control_block_component_[0] = 0;
			control_block_component_[1] = 0;
			state_block_component_[1] = 1;
	}

	/**********************************************************************************************************/
	/**********************************************************************************************************/
	  void
	  ControlElementEquation(const EDC<DH, VECTOR, dealdim> &edc,
	                         dealii::Vector<double> &local_vector, double scale)
	  {
	    const DOpEWrapper::FEValues<dealdim> &control_fe_values =
	      edc.GetFEValuesControl();
	    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
	    unsigned int n_q_points = edc.GetNQPoints();
	    {
	      assert(this->problem_type_ == "eigenvaluegradient");

	      funcgradvalues_.resize(n_q_points, Vector<double>(2));
	      edc.GetValuesControl("last_newton_solution", funcgradvalues_);
	    }
		const FEValuesExtractors::Vector controlextractor(0);
		vector<Tensor<1, dealdim> > psi_q(n_dofs_per_element);
		Tensor<1, 2> funcgradval;
	    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
	      {
	    	funcgradval.clear();
	    	funcgradval[0] = funcgradvalues_[q_point][0];
	    	funcgradval[1] = funcgradvalues_[q_point][1];


			for (unsigned int i = 0; i < n_dofs_per_element; i++) {
				psi_q[i] = control_fe_values[controlextractor].value(i,q_point);

				local_vector(i) += scale * scalar_product(funcgradval, psi_q[i]) * control_fe_values.JxW(q_point);
			}
	      }
	  }

	/**********************************************************************************************************/
	/**********************************************************************************************************/

	  void
	  ControlElementMatrix(const EDC<DH, VECTOR, dealdim> &edc,
	                       FullMatrix<double> &local_matrix, double scale)
	  {
	    const DOpEWrapper::FEValues<dealdim> &control_fe_values =
	      edc.GetFEValuesControl();
	    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
	    unsigned int n_q_points = edc.GetNQPoints();
	    const FEValuesExtractors::Vector controlextractor(0);
	    vector<Tensor<1, dealdim> > psi_q(n_dofs_per_element);
	    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
	      {
	    	for (unsigned int i = 0; i < n_dofs_per_element; i++){
	    		psi_q[i] = control_fe_values[controlextractor].value(i,q_point);
	    	}

	        for (unsigned int i = 0; i < n_dofs_per_element; i++)
	          {
	            for (unsigned int j = 0; j < n_dofs_per_element; j++)
	              {

	                local_matrix(i, j) += scale *  psi_q[i] * psi_q[j]
	                                    * control_fe_values.JxW(q_point);
	              }
	          }
	      }
	  }
	/*********************************************************************************************************/
	/**********************************************************************************************************/

	void ElementMatrix(const EDC<DH, VECTOR, dealdim> &edc,
			FullMatrix<double> &local_matrix, double scale,
			double /*scale_ico*/) {
		const DOpEWrapper::FEValues<dealdim> &state_fe_values =
				edc.GetFEValuesState();
		const unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
		const unsigned int n_q_points = edc.GetNQPoints();

		uvalues_.resize(n_q_points, Vector<double>(3));
		ugrads_.resize(n_q_points, vector<Tensor<1, dealdim> >(3));
		qgrads_.resize(n_q_points, vector<Tensor<1, dealdim> >(2));

		edc.GetGradsControl("control", qgrads_);

		vector<Tensor<1, dealdim> > phi_grads_u(n_dofs_per_element);
		vector<Tensor<1, dealdim> > phi_u(n_dofs_per_element);
		vector<Tensor<2, dealdim> > psi_grads_q(n_dofs_per_element);
		vector<typename internal::CurlType<dealdim>::type> phi_curl_u(
				n_dofs_per_element);

		const FEValuesExtractors::Scalar psi(0);
		const FEValuesExtractors::Vector E(1);
		const FEValuesExtractors::Vector controlextractor(0);

		Tensor<2, 2> qgrads;
		Tensor<2, 2> DF;
		Tensor<2, 2> DF_Inverse;
		Tensor<2, 2> DF_Inverse_T;
		double detDF;

		for (unsigned int q_point = 0; q_point < n_q_points; q_point++) {
			qgrads.clear();
			DF.clear();
			DF_Inverse.clear();
			DF_Inverse_T.clear();

			qgrads[0][0] = qgrads_[q_point][0][0];
			qgrads[1][1] = qgrads_[q_point][1][1];
			qgrads[1][0] = qgrads_[q_point][1][0];
			qgrads[0][1] = qgrads_[q_point][0][1];

			DF = deformation_tensor_(qgrads);
			detDF = determinante_(DF);
			DF_Inverse = inverse_(DF);
			DF_Inverse_T = inverse_transpose_(DF);

			for (unsigned int i = 0; i < n_dofs_per_element; i++) {
				phi_grads_u[i] = state_fe_values[psi].gradient(i, q_point);
				phi_curl_u[i] = state_fe_values[E].curl(i, q_point);
				phi_u[i] = state_fe_values[E].value(i, q_point);
			}

			for (unsigned int i = 0; i < n_dofs_per_element; i++) {
				for (unsigned int j = 0; j < n_dofs_per_element; j++) {


					local_matrix(i, j) += scale *((1/detDF)*phi_curl_u[j]*(1/detDF)*phi_curl_u[i])
											* state_fe_values.JxW(q_point);

					local_matrix(i, j) += scale* scalar_product(phi_grads_u[j]*DF_Inverse,
							DF_Inverse_T*phi_u[i])* state_fe_values.JxW(q_point);

					local_matrix(i, j) += scale* scalar_product(DF_Inverse_T*phi_u[j], phi_grads_u[i]*DF_Inverse)
																		* state_fe_values.JxW(q_point);



				}
			}
		}
	}

	/**********************************************************************************************************/
	/**********************************************************************************************************/
	void ElementMassMatrix(const EDC<DH, VECTOR, dealdim> &edc,
			FullMatrix<double> &local_matrix, double scale,
			double /*scale_ico*/) {
		const DOpEWrapper::FEValues<dealdim> &state_fe_values =
				edc.GetFEValuesState();
		const unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
		const unsigned int n_q_points = edc.GetNQPoints();

		uvalues_.resize(n_q_points, Vector<double>(3));
		ugrads_.resize(n_q_points, vector<Tensor<1, dealdim> >(3));
		qgrads_.resize(n_q_points, vector<Tensor<1, dealdim> >(2));

		edc.GetGradsControl("control", qgrads_);

		vector<Tensor<1, dealdim> > phi_grads_u(n_dofs_per_element);
		vector<Tensor<1, dealdim> > phi_u(n_dofs_per_element);
		vector<typename internal::CurlType<dealdim>::type> phi_curl_u(
				n_dofs_per_element);
		const FEValuesExtractors::Scalar psi(0);
		const FEValuesExtractors::Vector E(1);

		Tensor<2, 2> qgrads;
		Tensor<2, 2> DF;
		Tensor<2, 2> DF_Inverse;
		Tensor<2, 2> DF_Inverse_T;
		for (unsigned int q_point = 0; q_point < n_q_points; q_point++) {
			DF_Inverse.clear();
			DF_Inverse_T.clear();
			DF.clear();
			qgrads.clear();

			qgrads[0][0] = qgrads_[q_point][0][0];
			qgrads[1][1] = qgrads_[q_point][1][1];
			qgrads[1][0] = qgrads_[q_point][1][0];
			qgrads[0][1] = qgrads_[q_point][0][1];
			DF = deformation_tensor_(qgrads);
			DF_Inverse = inverse_(DF);
			DF_Inverse_T = inverse_transpose_(DF);

			for (unsigned int i = 0; i < n_dofs_per_element; i++) {
				phi_curl_u[i] = state_fe_values[E].curl(i, q_point);
				phi_u[i] = state_fe_values[E].value(i, q_point);
			}

			for (unsigned int i = 0; i < n_dofs_per_element; i++) {
				for (unsigned int j = 0; j < n_dofs_per_element; j++) {

					local_matrix(i, j) += scale* scalar_product(DF_Inverse_T*phi_u[j],DF_Inverse_T*phi_u[i])
																* state_fe_values.JxW(q_point);
				}
			}
		}
	}

	/**********************************************************************************************************/

	void ElementEquation_Q(const EDC<DH, VECTOR, dealdim> & edc,
			dealii::Vector<double> &local_vector, double scale, double /*scale_ico*/ ) {
		const DOpEWrapper::FEValues<dealdim> &control_fe_values =
				edc.GetFEValuesControl();
		const DOpEWrapper::FEValues<dealdim> &state_fe_values =
						edc.GetFEValuesState();
		unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
		unsigned int n_q_points = edc.GetNQPoints();

		assert(this->problem_type_ == "eigenvaluegradient");

		uvalues_.resize(n_q_points, Vector<double>(3));
		zvalues_.resize(n_q_points, Vector<double>(3));
		qgrads_.resize(n_q_points, vector<Tensor<1, dealdim> >(2));
		ugrads_.resize(n_q_points, vector<Tensor<1, dealdim> >(3));
		zgrads_.resize(n_q_points, vector<Tensor<1, dealdim> >(3));

		edc.GetGradsControl("control", qgrads_);
		edc.GetValuesState("state", uvalues_);
		edc.GetGradsState("state", ugrads_);
		edc.GetValuesState("adjoint", zvalues_);
		edc.GetGradsState("adjoint", zgrads_);

//		 qvalues_old_.resize(n_q_points, Vector<double>(2));
//		 edc.GetValuesControl("q_previous", qvalues_old_);

		vector<Tensor<dealdim, dealdim> > grad_phi_v(n_dofs_per_element);
		const FEValuesExtractors::Vector dv(0);

		Tensor<1, 2> u;
		Tensor<1, 2> uDF_Inv_T;
		Tensor<1, 2> grad_u;
		double curl_u = 0;

		Tensor<1, 2> z;
		Tensor<1, 2> zDF_Inv_T;
		Tensor<1, 2> grad_z;
		double curl_z = 0;

		Tensor<2, 2> qgrads;
		Tensor<2, 2> DF;
		Tensor<2, 2> DF_Inverse_T;
		Tensor<2, 2> DF_Inverse;
		double detDF;
		Tensor<2, 2> DFdq;
		Tensor<2, 2> DF_Inverse_Tdq;
		Tensor<2, 2> DF_Inversedq;
		double detDFINVdq;

		for (unsigned int q_point = 0; q_point < n_q_points; q_point++) {
		u.clear();
		uDF_Inv_T.clear();
		z.clear();
		zDF_Inv_T.clear();
		grad_u.clear();
		grad_z.clear();

		qgrads.clear();
		DF.clear();
		DF_Inverse_T.clear();
		DF_Inverse_Tdq.clear();
		DF_Inverse.clear();
		DF_Inversedq.clear();

		qgrads[0][0] = qgrads_[q_point][0][0];
		qgrads[1][1] = qgrads_[q_point][1][1];
		qgrads[1][0] = qgrads_[q_point][1][0];
		qgrads[0][1] = qgrads_[q_point][0][1];

		DF = deformation_tensor_(qgrads);
		DF_Inverse_T = inverse_transpose_(DF);
		DF_Inverse = inverse_(DF);
		detDF = determinante_(DF);

		u[0] = uvalues_[q_point][1];
		u[1] = uvalues_[q_point][2];
		z[0] = zvalues_[q_point][1];
		z[1] = zvalues_[q_point][2];

		grad_u[0] = ugrads_[q_point][0][0];
		grad_u[1] = ugrads_[q_point][0][1];
		grad_z[0] = zgrads_[q_point][0][0];
		grad_z[1] = zgrads_[q_point][0][1];

		curl_u =  ugrads_[q_point][2][0] - ugrads_[q_point][1][1];
		curl_z =  zgrads_[q_point][2][0] - zgrads_[q_point][1][1];

		uDF_Inv_T = DF_Inverse_T*u;
		zDF_Inv_T = DF_Inverse_T*z;

		for (unsigned int i = 0; i < n_dofs_per_element; i++) {
			grad_phi_v[i]=control_fe_values[dv].gradient(i,q_point);

			detDFINVdq = -(1/detDF)*(1/detDF)*
								((qgrads[0][0]+1)*grad_phi_v[i][1][1]
								+ (qgrads[1][1]+1)*grad_phi_v[i][0][0]
								-qgrads[1][0]*grad_phi_v[i][0][1]
								-qgrads[0][1]*grad_phi_v[i][1][0] );

			DF_Inverse_Tdq = -((1/detDF)*adjunkte_transposed_(grad_phi_v[i])+detDFINVdq*adjunkte_transposed_(DF));
			DF_Inversedq = -((1/detDF)*adjunkte_(grad_phi_v[i])+detDFINVdq*adjunkte_(DF));


			//Curl Term
			local_vector(i) += scale*detDFINVdq*curl_z * (1/detDF)*curl_u
					* state_fe_values.JxW(q_point);

			//1. Grad Term
			local_vector(i) += scale*
					(scalar_product(grad_u*DF_Inverse, DF_Inverse_Tdq*z))
					* state_fe_values.JxW(q_point);

			//2. Grad Term
			local_vector(i) += scale*(scalar_product(DF_Inverse_Tdq*z, grad_u*DF_Inverse)
				)* state_fe_values.JxW(q_point);


//			//Curl Term
//			local_vector(i) += scale*2*detDFINVdq*curl_z * (1/detDF)*curl_u
//				* state_fe_values.JxW(q_point);
//
//			//1. Grad Term
//			local_vector(i) += scale*
//				(scalar_product(grad_u*DF_Inverse, DF_Inverse_Tdq*z)+scalar_product(grad_z*DF_Inversedq, DF_Inverse_T*u))
//					* state_fe_values.JxW(q_point);
//
//			//2. Grad Term
//			local_vector(i) += scale*(scalar_product(DF_Inverse_Tdq*z, grad_u*DF_Inverse)+scalar_product(DF_Inverse_T*u, grad_z*DF_Inversedq)
//					)* state_fe_values.JxW(q_point);

		}

		}

	}

	/**********************************************************************************************************/
	/**********************************************************************************************************/

	void ElementMassEquation_Q(const EDC<DH, VECTOR, dealdim> & edc,
			dealii::Vector<double> &local_vector, double scale, double eigenvalue) {

			const DOpEWrapper::FEValues<dealdim> &control_fe_values =
					edc.GetFEValuesControl();
			const DOpEWrapper::FEValues<dealdim> &state_fe_values =
								edc.GetFEValuesState();

			unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
			unsigned int n_q_points = edc.GetNQPoints();

			assert(this->problem_type_ == "eigenvaluegradient");

			uvalues_.resize(n_q_points, Vector<double>(3));
			zvalues_.resize(n_q_points, Vector<double>(3));
			qgrads_.resize(n_q_points, vector<Tensor<1, dealdim> >(2));
			ugrads_.resize(n_q_points, vector<Tensor<1, dealdim> >(3));
			zgrads_.resize(n_q_points, vector<Tensor<1, dealdim> >(3));

			edc.GetValuesState("state", uvalues_);
			edc.GetValuesState("adjoint", zvalues_);
			edc.GetGradsControl("control", qgrads_);
			edc.GetGradsState("state", ugrads_);
			edc.GetGradsState("adjoint", zgrads_);;

			vector<Tensor<dealdim, dealdim> > grad_phi_v(n_dofs_per_element);
			const FEValuesExtractors::Vector dv(0);

			Tensor<1, 2> u;
			Tensor<1, 2> z;
			Tensor<1, 2> uDF_Inv_T;
			Tensor<1, 2> zDF_Inv_T;

			double curl_u = 0;
			double curl_z = 0;

			double detDF = 0;
			double detDFINVdq =0;
			Tensor<2, 2> DF;
			Tensor<2, 2> DFdq;
			Tensor<2, 2> DF_Inverse;
			Tensor<2, 2> DF_Inversedq;
			Tensor<2, 2> DF_Inverse_T;
			Tensor<2, 2> DF_Inverse_Tdq;
			Tensor<2, 2> qgrads;

			for (unsigned int q_point = 0; q_point < n_q_points; q_point++) {
				uDF_Inv_T.clear();
				zDF_Inv_T.clear();
				z.clear();
				u.clear();

				DF.clear();
				DFdq.clear();

				DF_Inverse.clear();
				DF_Inversedq.clear();


				DF_Inverse_T.clear();
				DF_Inverse_Tdq.clear();
				qgrads.clear();

				qgrads[0][0] = qgrads_[q_point][0][0];
				qgrads[1][1] = qgrads_[q_point][1][1];
				qgrads[1][0] = qgrads_[q_point][1][0];
				qgrads[0][1] = qgrads_[q_point][0][1];

				DF = deformation_tensor_(qgrads);
				detDF = determinante_(DF);
				DF_Inverse = inverse_(DF);
				DF_Inverse_T = inverse_transpose_(DF);

				u[0] = uvalues_[q_point][1];
				u[1] = uvalues_[q_point][2];
				z[0] = zvalues_[q_point][1];
				z[1] = zvalues_[q_point][2];

				curl_u =  ugrads_[q_point][2][0] - ugrads_[q_point][1][1];
				curl_z =  zgrads_[q_point][2][0] - zgrads_[q_point][1][1];

				uDF_Inv_T = DF_Inverse_T*u;
				zDF_Inv_T = (DF_Inverse_T*z);

				for (unsigned int i = 0; i < n_dofs_per_element; i++) {
					grad_phi_v[i]=control_fe_values[dv].gradient(i,q_point);

					detDFINVdq = -(1/detDF)*(1/detDF)*
										(qgrads[0][0]*grad_phi_v[i][1][1]
										+ qgrads[1][1]*grad_phi_v[i][0][0]
										-qgrads[1][0]*grad_phi_v[i][0][1]
										-qgrads[0][1]*grad_phi_v[i][1][0]
										+grad_phi_v[i][1][1]
										+grad_phi_v[i][0][0] );

					DF_Inversedq = -((1/detDF)*adjunkte_(grad_phi_v[i])+detDFINVdq*adjunkte_(DF));
					DF_Inverse_Tdq =  -((1/detDF)* adjunkte_transposed_(grad_phi_v[i])+detDFINVdq*adjunkte_transposed_(DF));



//					local_vector(i) += scale*4*eigenvalue*(
//						scalar_product(uDF_Inv_T ,DF_Inverse_Tdq*z))
//				* state_fe_values.JxW(q_point);

//////				 Normbedingung
//					local_vector(i) += scale*eigenvalue*((scalar_product(uDF_Inv_T,DF_Inverse_Tdq *z)
//								)* state_fe_values.JxW(q_point));

				}
		}
	}

	/**********************************************************************************************************/
	/**********************************************************************************************************/

	UpdateFlags GetFaceUpdateFlags() const {
		return update_values | update_gradients | update_normal_vectors
				| update_quadrature_points | update_JxW_values;
	}

	UpdateFlags GetUpdateFlags() const {
		return update_values | update_gradients | update_quadrature_points | update_JxW_values;
	}

	unsigned int GetStateNBlocks() const {
		return 2;
	}

unsigned int GetControlNBlocks() const {
		return 1;
	}

	vector<unsigned int>&
	GetStateBlockComponent() {
		return state_block_component_;
	}
	const vector<unsigned int>&
	GetStateBlockComponent() const {
		return state_block_component_;
	}
	vector<unsigned int>&
	GetControlBlockComponent() {
		return control_block_component_;
	}
	const vector<unsigned int>&
	GetControlBlockComponent() const {
		return control_block_component_;
	}


protected:

private:
	vector<Vector<double> > uvalues_;
	vector<Vector<double> > duvalues_;
	vector<Vector<double> > zvalues_;
	vector<Vector<double> >  eigValues_;
	vector<Vector<double> > qvalues_;
	//	vector<Vector<double> > qvalues_old_;
	vector<Vector<double> > funcgradvalues_;

	vector<vector<Tensor<1, dealdim> > > ugrads_;
	vector<vector<Tensor<1, dealdim> > > dugrads_;
	vector<vector<Tensor<1, dealdim> > > zgrads_;
	vector<vector<Tensor<1, dealdim> > > qgrads_;


	vector<unsigned int> control_block_component_;
	vector<unsigned int> state_block_component_;

};

#endif
