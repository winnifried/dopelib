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

using namespace std;
using namespace dealii;
using namespace DOpE;

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
			state_block_component_[1] = 1;
	}

	// TODO für alle Gleichungen/Matrizen: Richtige Übergabe für Control q bzw grad_q!!

	/**********************************************************************************************************/
	/**********************************************************************************************************/

	void ElementEquation(const EDC<DH, VECTOR, dealdim> &edc,
			dealii::Vector<double> &local_vector, double scale,
			double /*scale_ico*/) {

		const DOpEWrapper::FEValues<dealdim> &state_fe_values =
				edc.GetFEValuesState();

		const unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
		const unsigned int n_q_points = edc.GetNQPoints();

		assert(this->problem_type_ == "state");

		uvalues_.resize(n_q_points, Vector<double>(3));
		ugrads_.resize(n_q_points, vector<Tensor<1, dealdim> >(3));
		qvalues_.resize(n_q_points, Vector<double>(2));
		qgrads_.resize(n_q_points, vector<Tensor<1, dealdim> >(2));

		edc.GetValuesState("last_newton_solution", uvalues_);
		edc.GetGradsState("last_newton_solution", ugrads_);

		edc.GetValuesControl("control", qvalues_);
//		edc.GetGradsControl("control", qgrads_);

		vector<Tensor<1, dealdim> > phi_grads_u(n_dofs_per_element);
		vector<Tensor<1, dealdim> > phi_u(n_dofs_per_element);
		vector<typename internal::CurlType<dealdim>::type> phi_curl_u(
				n_dofs_per_element);

		const FEValuesExtractors::Scalar psi(0);
		const FEValuesExtractors::Vector E(1);

		Tensor<1, 2> grad_phi;
		Tensor<2, 2> grad_u;
		Tensor<1, 2> u;
		double curl_u;
		Tensor<1, 2> q;
		Tensor<2, 2> DF;
		Tensor<2, 2> DF_Inverse;
		double detDF;
		Tensor<2, 2> DF_Inverse_T;
		Tensor<1,2> qvalues;
		Tensor<2, 2> qgrads;

		for (unsigned int q_point = 0; q_point < n_q_points; q_point++) {
			u.clear();
			q.clear();
			grad_phi.clear();
			grad_u.clear();
			qgrads.clear();
			DF.clear();
			DF_Inverse.clear();
			DF_Inverse_T.clear();

			qvalues[0] = qvalues_[q_point][0];
			qvalues[1] = qvalues_[q_point][1];
			qgrads[0][0] = qvalues_[q_point][0]; //qgrads_[q_point][0][0];
			qgrads[1][1] = qvalues_[q_point][0]; //qgrads_[q_point][1][1];
			qgrads[1][0] = qvalues_[q_point][1]; //qgrads_[q_point][1][0];
			qgrads[0][1] = qvalues_[q_point][1]; //qgrads_[q_point][0][1];

			DF = calc_DF(qgrads);
			detDF = calc_detDF(DF);
			DF_Inverse = calc_invDF(DF);
			DF_Inverse_T = calc_invDFTranspose(DF);

			for (unsigned int i = 0; i < 2; i++) {
				grad_u[0][i] = ugrads_[q_point][1][i];
				grad_u[1][i] = ugrads_[q_point][2][i];
				u[i] = uvalues_[q_point][i];
				q[i] = qvalues_[q_point][i];
			}
			curl_u = grad_u[1][0] - grad_u[0][1];
			for (unsigned int i = 0; i < n_dofs_per_element; i++) {
				phi_grads_u[i] = state_fe_values[psi].gradient(i, q_point);
				phi_curl_u[i] = state_fe_values[E].curl(i, q_point);
				phi_u[i] = state_fe_values[E].value(i, q_point);

				local_vector(i) += scale * ((1 / detDF)* phi_curl_u[i][0] *(1 / detDF)
						* curl_u )* state_fe_values.JxW(q_point);
				local_vector(i) += scale * scalar_product(phi_u[i] * detDF,
								DF_Inverse_T * grad_u[i] * DF_Inverse) * state_fe_values.JxW(q_point);
				local_vector(i) += scale * scalar_product(DF_Inverse_T * phi_grads_u[i] * DF_Inverse,
								u * detDF) * state_fe_values.JxW(q_point);
			}
		}
	}

	/**********************************************************************************************************/
	/**********************************************************************************************************/

	void ElementMassEquation(const EDC<DH, VECTOR, dealdim> &edc,
			dealii::Vector<double> &local_vector, double scale,
			double /*scale_ico*/) {
		const DOpEWrapper::FEValues<dealdim> &state_fe_values =
				edc.GetFEValuesState();
		const unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
		const unsigned int n_q_points = edc.GetNQPoints();

		assert(this->problem_type_ == "state");

		uvalues_.resize(n_q_points, Vector<double>(3));
		qvalues_.resize(n_q_points, Vector<double>(2));

		edc.GetValuesState("state", uvalues_);
		edc.GetValuesControl("control", qvalues_);

		vector<Tensor<1, dealdim> > phi_u(n_dofs_per_element);
		const FEValuesExtractors::Vector E(1);

		Tensor<1, 2> u;
		Tensor<1, 2> qvalues;
		Tensor<2, 2> qgrads;
		Tensor<2, 2> DF;
		Tensor<2, 2> DF_Inverse;
		Tensor<2, 2> DF_Inverse_T;
		for (unsigned int q_point = 0; q_point < n_q_points; q_point++) {
			qgrads.clear();
			DF.clear();
			DF_Inverse.clear();
			DF_Inverse_T.clear();

			qgrads[0][0] = qvalues_[q_point][0]; //qgrads_[q_point][0][0];
			qgrads[1][1] = qvalues_[q_point][0]; //qgrads_[q_point][1][1];
			qgrads[1][0] = qvalues_[q_point][1]; //qgrads_[q_point][1][0];
			qgrads[0][1] = qvalues_[q_point][1]; //qgrads_[q_point][0][1];

			DF = calc_DF(qgrads);
			DF_Inverse = calc_invDF(DF);
			DF_Inverse_T = calc_invDFTranspose(DF);

			for (unsigned int i = 0; i < 2; i++) {
				u[i] = uvalues_[q_point][i];
			}
			for (unsigned int i = 0; i < n_dofs_per_element; i++) {
				phi_u[i] = state_fe_values[E].value(i, q_point);
				local_vector(i) += scale
						* scalar_product(DF_Inverse_T * phi_u[i],
								DF_Inverse_T * u) * state_fe_values.JxW(q_point);
			}
		}
	}

	/**********************************************************************************************************/
	/**********************************************************************************************************/

	void ElementMatrix(const EDC<DH, VECTOR, dealdim> &edc,
			FullMatrix<double> &local_matrix, double scale,
			double /*scale_ico*/) {
		const DOpEWrapper::FEValues<dealdim> &state_fe_values =
				edc.GetFEValuesState();
		const DOpEWrapper::FEValues<dealdim> &control_fe_values =
				edc.GetFEValuesControl();
		const unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
		const unsigned int n_q_points = edc.GetNQPoints();

		assert(this->problem_type_ ==  "eigenvaluestate" );

		uvalues_.resize(n_q_points, Vector<double>(3));
		ugrads_.resize(n_q_points, vector<Tensor<1, dealdim> >(3));

		qvalues_.resize(n_q_points, Vector<double>(2));
		qgrads_.resize(n_q_points, vector<Tensor<1, dealdim> >(2));

		edc.GetValuesControl("control", qvalues_);
		edc.GetGradsControl("control", qgrads_);

		vector<Tensor<1, dealdim> > phi_grads_u(n_dofs_per_element);
		vector<Tensor<1, dealdim> > phi_u(n_dofs_per_element);
		vector<Tensor<1, dealdim> > psi_q(n_dofs_per_element);
		vector<Tensor<2, dealdim> > psi_grads_q(n_dofs_per_element);
		vector<typename internal::CurlType<dealdim>::type> phi_curl_u(
				n_dofs_per_element);

		const FEValuesExtractors::Scalar psi(0);
		const FEValuesExtractors::Vector E(1);
		const FEValuesExtractors::Vector controlextractor(0);

		Tensor<1,2> qvalues;
		Tensor<2, 2> qgrads;
		Tensor<2, 2> DF;
		Tensor<2, 2> DF_Inverse;
		double detDF;
		Tensor<2, 2> DF_Inverse_T;
		for (unsigned int q_point = 0; q_point < n_q_points; q_point++) {
			qgrads.clear();
			DF.clear();
			DF_Inverse.clear();
			DF_Inverse_T.clear();
			qvalues[0] = qvalues_[q_point][0];
			qvalues[1] = qvalues_[q_point][1];
			qgrads[0][0] = qvalues_[q_point][0];//qgrads_[q_point][0][0];
			qgrads[1][1] = qvalues_[q_point][0];//qgrads_[q_point][1][1];
			qgrads[1][0] = qvalues_[q_point][1];//qgrads_[q_point][1][0];
			qgrads[0][1] = qvalues_[q_point][1];//qgrads_[q_point][0][1];

//			std::cout <<  qvalues[0] << std::endl;
//			std::cout << qvalues[1] << std::endl;
//			std::cout << "######qgrads#################"<< std::endl;
//			std::cout << qgrads[0][0] << std::endl;
//			std::cout <<  qgrads[1][1]<< std::endl;
//			std::cout <<  qgrads[1][0]<< std::endl;
//			std::cout << qgrads[0][1]<< std::endl;
			DF = calc_DF(qgrads);
//			std::cout << "#######DF################"<< std::endl;
//			std::cout <<DF[0][0] << std::endl;
//			std::cout <<  DF[1][1]<< std::endl;
//			std::cout <<  DF[1][0]<< std::endl;
//			std::cout << DF[0][1]<< std::endl;

			detDF = calc_detDF(DF);
			DF_Inverse = calc_invDF(DF);
			DF_Inverse_T = calc_invDFTranspose(DF);

			for (unsigned int i = 0; i < n_dofs_per_element; i++) {
//				psi_q[i] = control_fe_values[controlextractor].value(i,q_point);
				psi_grads_q[i] = control_fe_values[controlextractor].gradient(i,q_point);
				phi_grads_u[i] = state_fe_values[psi].gradient(i, q_point);
				phi_curl_u[i] = state_fe_values[E].curl(i, q_point);
				phi_u[i] = state_fe_values[E].value(i, q_point);
			}

			for (unsigned int i = 0; i < n_dofs_per_element; i++) {
				for (unsigned int j = 0; j < n_dofs_per_element; j++) {
					local_matrix(i, j) += scale * (1/detDF)*phi_curl_u[j][0]
							* (1/detDF)*phi_curl_u[i][0]
							* state_fe_values.JxW(q_point);
					local_matrix(i, j) += scale
							* scalar_product(phi_u[j]*detDF,DF_Inverse_T*
									phi_grads_u[i]*DF_Inverse)
							* state_fe_values.JxW(q_point);
					local_matrix(i, j) += scale
									* scalar_product(DF_Inverse_T*phi_grads_u[j]*DF_Inverse,
											phi_u[i]*detDF)
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
//		const DOpEWrapper::FEValues<dealdim> &control_fe_values =
//				edc.GetFEValuesControl();
		const unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
		const unsigned int n_q_points = edc.GetNQPoints();

		assert(this->problem_type_ == "eigenvaluestate");

		uvalues_.resize(n_q_points, Vector<double>(3));
		ugrads_.resize(n_q_points, vector<Tensor<1, dealdim> >(3));

		qvalues_.resize(n_q_points, Vector<double>(2));
		qgrads_.resize(n_q_points, vector<Tensor<1, dealdim> >(2));

		edc.GetGradsControl("control", qgrads_);
		edc.GetValuesControl("control", qvalues_);

		vector<Tensor<1, dealdim> > phi_grads_u(n_dofs_per_element);
		vector<Tensor<1, dealdim> > phi_u(n_dofs_per_element);
		vector<typename internal::CurlType<dealdim>::type> phi_curl_u(
				n_dofs_per_element);
		const FEValuesExtractors::Scalar psi(0);
		const FEValuesExtractors::Vector E(1);

		Tensor<1,2> qvalues;
		Tensor<2, 2> qgrads;
		Tensor<2, 2> DF;
		Tensor<2, 2> DF_Inverse;
		double detDF;
		Tensor<2, 2> DF_Inverse_T;
		for (unsigned int q_point = 0; q_point < n_q_points; q_point++) {
			DF_Inverse.clear();
			DF_Inverse_T.clear();
			DF.clear();
			qgrads.clear();
			qgrads[0][0] = qvalues_[q_point][0];//qgrads_[q_point][0][0];
			qgrads[1][1] = qvalues_[q_point][0];//qgrads_[q_point][1][1];
			qgrads[1][0] = qvalues_[q_point][1];//qgrads_[q_point][1][0];
			qgrads[0][1] = qvalues_[q_point][1];//qgrads_[q_point][0][1];
			DF = calc_DF(qgrads);
			detDF = calc_detDF(DF);
			DF_Inverse = calc_invDF(DF);
			DF_Inverse_T = calc_invDFTranspose(DF);

			for (unsigned int i = 0; i < n_dofs_per_element; i++) {
				phi_curl_u[i] = state_fe_values[E].curl(i, q_point);
				phi_u[i] = state_fe_values[E].value(i, q_point);
			}

			for (unsigned int i = 0; i < n_dofs_per_element; i++) {
				for (unsigned int j = 0; j < n_dofs_per_element; j++) {

					local_matrix(i, j) += scale
							* scalar_product(DF_Inverse_T*phi_u[j], DF_Inverse_T*phi_u[i])
							* state_fe_values.JxW(q_point);
				}
			}
		}
	}

	/**********************************************************************************************************/
	/**********************************************************************************************************/

	void ElementEquation_U(const EDC<DH, VECTOR, dealdim>& edc,
		dealii::Vector<double> &local_vector, double scale,
			double /*scale_ico*/) {
			const DOpEWrapper::FEValues<dealdim> &state_fe_values =
					edc.GetFEValuesState();
		unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
		unsigned int n_q_points = edc.GetNQPoints();

		assert(this->problem_type_ == /*"eigenvalueadjoint"*/"eigenvalueadjoint");

		zvalues_.resize(n_q_points, Vector<double>(3));
		qvalues_.resize(n_q_points, Vector<double>(2));
		qgrads_.resize(n_q_points, vector<Tensor<1, dealdim> >(2));

		edc.GetValuesControl("control", qvalues_);
		edc.GetGradsControl("control", qgrads_);
		edc.GetValuesState("last_newton_solution"/*"state"*/, zvalues_);
		edc.GetGradsState("last_newton_solution"/*"state"*/, zgrads_);

		const FEValuesExtractors::Vector E(1);
		const FEValuesExtractors::Scalar psi(0);


		vector<Tensor<1, dealdim> > phi_grads_u(n_dofs_per_element);
		vector<Tensor<1, dealdim> > phi_u(n_dofs_per_element);
		vector<typename internal::CurlType<dealdim>::type> phi_curl_u(
				n_dofs_per_element);

		Tensor<2, 2> grad_z;
		Tensor<1, 2> z;
		double curl_z;
		Tensor<1, 2> q;
		Tensor<2, 2> DF;
		Tensor<2, 2> DF_Inverse;
		double detDF;
		Tensor<2, 2> DF_Inverse_T;
		Tensor<1,2> qvalues;
		Tensor<2, 2> qgrads;

		for (unsigned int q_point = 0; q_point < n_q_points; q_point++) {
		z.clear();
		q.clear();
		grad_z.clear();
		qgrads.clear();
		DF.clear();
		DF_Inverse.clear();
		DF_Inverse_T.clear();

		qvalues[0] = qvalues_[q_point][0];
		qvalues[1] = qvalues_[q_point][1];
		qgrads[0][0] = qvalues_[q_point][0]; //qgrads_[q_point][0][0];
		qgrads[1][1] = qvalues_[q_point][0]; //qgrads_[q_point][1][1];
		qgrads[1][0] = qvalues_[q_point][1]; //qgrads_[q_point][1][0];
		qgrads[0][1] = qvalues_[q_point][1]; //qgrads_[q_point][0][1];

		DF = calc_DF(qgrads);
		detDF = calc_detDF(DF);
		DF_Inverse = calc_invDF(DF);
		DF_Inverse_T = calc_invDFTranspose(DF);

		for (unsigned int i = 0; i < 2; i++) {
			grad_z[0][i] = zgrads_[q_point][1][i];
			grad_z[1][i] = zgrads_[q_point][2][i];
			z[i] = zvalues_[q_point][i];
			q[i] = qvalues_[q_point][i];
		}
			curl_z = grad_z[1][0] - grad_z[0][1];
			for (unsigned int i = 0; i < n_dofs_per_element; i++) {
				phi_grads_u[i] = state_fe_values[psi].gradient(i, q_point);
				phi_curl_u[i] = state_fe_values[E].curl(i, q_point);
				phi_u[i] = state_fe_values[E].value(i, q_point);

				local_vector(i) += scale * ((1 / detDF)* phi_curl_u[i][0] *(1 / detDF)
						* curl_z )* state_fe_values.JxW(q_point);
				local_vector(i) += scale * scalar_product(phi_u[i] * detDF,
						DF_Inverse_T * grad_z[i] * DF_Inverse) * state_fe_values.JxW(q_point);
				local_vector(i) += scale * scalar_product(DF_Inverse_T * phi_grads_u[i] * DF_Inverse,
						z * detDF) * state_fe_values.JxW(q_point);
					}
				}
	}

	/**********************************************************************************************************/
	/**********************************************************************************************************/

	void ElementMassEquation_U(const EDC<DH, VECTOR, dealdim>& edc,
			dealii::Vector<double> &local_vector, double scale,
				double /*scale_ico*/) {

		const DOpEWrapper::FEValues	<dealdim> &state_fe_values =
				edc.GetFEValuesState();
		unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
		unsigned int n_q_points = edc.GetNQPoints();

		assert(this->problem_type_ == "eigenvalueadjoint");

		zvalues_.resize(n_q_points, Vector<double>(3));
		qvalues_.resize(n_q_points, Vector<double>(2));
		qgrads_.resize(n_q_points, vector<Tensor<1, dealdim> >(2));

		const FEValuesExtractors::Vector E(1);
		vector<Tensor<1, dealdim> > phi_u(n_dofs_per_element);

		edc.GetValuesControl("control", qvalues_);
		edc.GetGradsControl("control", qgrads_);
		edc.GetValuesState(/*"last_newton_solution"*/"state", zvalues_);

		Tensor<2, 2> qgrads;
		Tensor<2, 2> DF;
		Tensor<2, 2> DF_Inverse_T;
		Tensor<1, 2> z;
		for (unsigned int q_point = 0; q_point < n_q_points; q_point++) {
			qgrads.clear();
			DF.clear();
			DF_Inverse_T.clear();
			z.clear();

			qgrads[0][0] = qvalues_[q_point][0]; //qgrads_[q_point][0][0];
			qgrads[1][1] = qvalues_[q_point][0]; //qgrads_[q_point][1][1];
			qgrads[1][0] = qvalues_[q_point][1]; //qgrads_[q_point][1][0];
			qgrads[0][1] = qvalues_[q_point][1]; //qgrads_[q_point][0][1];

			DF = calc_DF(qgrads);
			DF_Inverse_T = calc_invDFTranspose(DF);
			for (unsigned int i = 0; i < 2; i++) {
				z[i] = zvalues_[q_point][i];
			}

			for (unsigned int i = 0; i < n_dofs_per_element; i++) {
				phi_u[i] = state_fe_values[E].value(i, q_point);

				local_vector(i) += scale *scalar_product(DF_Inverse_T*phi_u[i], DF_Inverse_T*z)
						* state_fe_values.JxW(q_point);
			}
		}
	}

	/**********************************************************************************************************/
	/**********************************************************************************************************/

	void ElementEquation_Q(const EDC<DH, VECTOR, dealdim> & edc,
			dealii::Vector<double> &local_vector, double scale,
			double /*scale_ico*/) {
		const DOpEWrapper::FEValues<dealdim> &control_fe_values =
				edc.GetFEValuesControl();
		const DOpEWrapper::FEValues<dealdim> &state_fe_values =
						edc.GetFEValuesState();
		unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
		unsigned int n_q_points = edc.GetNQPoints();

		assert(this->problem_type_ == "gradient");

		uvalues_.resize(n_q_points, Vector<double>(3));
		zvalues_.resize(n_q_points, Vector<double>(3));
		qvalues_.resize(n_q_points, Vector<double>(2));
		qgrads_.resize(n_q_points, vector<Tensor<1, dealdim> >(2));
		dqvalues_.resize(n_q_points, Vector<double>(2));
		dqgrads_.resize(n_q_points, vector<Tensor<1, dealdim> >(2));

		edc.GetValuesControl("control", qvalues_);
		edc.GetGradsControl("control", qgrads_);
		edc.GetValuesControl("gradient", dqvalues_);
		edc.GetGradsControl("gradient", dqgrads_);
		edc.GetValuesState("state", uvalues_);
		edc.GetValuesState("adjoint", zvalues_);

		vector<Tensor<1, dealdim> > phi_u(n_dofs_per_element);
		const FEValuesExtractors::Scalar psi(0);
		const FEValuesExtractors::Vector E(1);

		Tensor<1, 2> u;
		double curl_u;
		Tensor<2, 2> grad_u;
		Tensor<1, 2> z;
		double curl_z;
		Tensor<2, 2> grad_z;

		Tensor<1, 2> qvalues;
		Tensor<2, 2> qgrads;
		Tensor<1, 2> dqvalues;
		Tensor<2, 2> dqgrads;
		Tensor<2, 2> DF;
		Tensor<2, 2> DF_Inverse_T;
		Tensor<2, 2> DF_Inverse;
		double detDF;
		Tensor<2, 2> DFdq;
		Tensor<2, 2> DF_Inverse_Tdq;
		Tensor<2, 2> DF_Inversedq;
		double detDFdq;
		for (unsigned int q_point = 0; q_point < n_q_points; q_point++) {
		u.clear();
		z.clear();
		grad_u.clear();
		grad_z.clear();

		qgrads.clear();
		dqgrads.clear();
		DF.clear();
		DF_Inverse_T.clear();
		DFdq.clear();
		DF_Inverse_Tdq.clear();

		qgrads[0][0] = qvalues_[q_point][0]; //qgrads_[q_point][0][0];
		qgrads[1][1] = qvalues_[q_point][0]; //qgrads_[q_point][1][1];
		qgrads[1][0] = qvalues_[q_point][1]; //qgrads_[q_point][1][0];
		qgrads[0][1] = qvalues_[q_point][1]; //qgrads_[q_point][0][1];

		dqgrads[0][0] = dqvalues_[q_point][0]; //qgrads_[q_point][0][0];
		dqgrads[1][1] = dqvalues_[q_point][0]; //qgrads_[q_point][1][1];
		dqgrads[1][0] = dqvalues_[q_point][1]; //qgrads_[q_point][1][0];
		dqgrads[0][1] = dqvalues_[q_point][1]; //qgrads_[q_point][0][1];


		DF = calc_DF(qgrads);
		DF_Inverse = calc_invDF(DF);
		DF_Inverse_T = calc_invDFTranspose(DF);
		detDF = calc_detDF(DF);

		DFdq = calc_DF(dqgrads);
		DF_Inversedq = calc_invDF(DFdq);
		DF_Inverse_Tdq = calc_invDFTranspose(DFdq);
		detDFdq=calc_detDF(DFdq);

		for (unsigned int i = 0; i < 2; i++) {
			u[i] = uvalues_[q_point][i];
			z[i] = zvalues_[q_point][i];

			grad_u[0][i] = ugrads_[q_point][1][i];
			grad_u[1][i] = ugrads_[q_point][2][i];

			grad_z[0][i] = zgrads_[q_point][1][i];
			grad_z[1][i] = zgrads_[q_point][2][i];

		}
		curl_u = grad_u[1][0] - grad_u[0][1];
		curl_z = grad_z[1][0] - grad_z[0][1];

		for (unsigned int i = 0; i < n_dofs_per_element; i++) {
			phi_u[i] = state_fe_values[E].value(i, q_point);
			local_vector(i) += scale*((1/detDFdq)*curl_z*(1/detDF)*curl_u
					+ (1/detDF)*curl_z * (1/detDFdq)*curl_u)
				* control_fe_values.JxW(q_point);

			local_vector(i) += scale*(scalar_product(z*detDFdq, DF_Inverse_T*grad_u[i]*DF_Inverse)
				 + scalar_product(z*detDF, DF_Inverse_Tdq*grad_u[i]*DF_Inverse)
				+ scalar_product(z*detDF, DF_Inverse_T*grad_u[i]*DF_Inversedq)
			)* control_fe_values.JxW(q_point);

			local_vector(i) += scale*(scalar_product(DF_Inverse_Tdq*grad_z[i]*DF_Inverse, u*detDF)
					+ scalar_product(DF_Inverse_T*grad_z[i]*DF_Inversedq, u * detDF)
					+ scalar_product(DF_Inverse_T*grad_z[i]*DF_Inverse, u * detDFdq)
			)* control_fe_values.JxW(q_point);
			}
		}

	}

	/**********************************************************************************************************/
	/**********************************************************************************************************/

	void ElementMassEquation_Q(const EDC<DH, VECTOR, dealdim> & edc,
			dealii::Vector<double> &local_vector, double scale,
			double /*scale_ico*/) {
			const DOpEWrapper::FEValues<dealdim> &control_fe_values =
					edc.GetFEValuesControl();
			const DOpEWrapper::FEValues<dealdim> &state_fe_values =
							edc.GetFEValuesState();
			unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
			unsigned int n_q_points = edc.GetNQPoints();

			assert(this->problem_type_ == "gradient");

			uvalues_.resize(n_q_points, Vector<double>(3));
			zvalues_.resize(n_q_points, Vector<double>(3));
			qvalues_.resize(n_q_points, Vector<double>(2));
			qgrads_.resize(n_q_points, vector<Tensor<1, dealdim> >(2));
			dqvalues_.resize(n_q_points, Vector<double>(2));
			dqgrads_.resize(n_q_points, vector<Tensor<1, dealdim> >(2));

			edc.GetValuesControl("control", qvalues_);
			edc.GetGradsControl("control", qgrads_);
			edc.GetValuesControl("gradient", dqvalues_); //TODO..das muss weg... Richtungsableitung, nicht der Gradient..
			edc.GetGradsControl("gradient", dqgrads_);
			edc.GetValuesState("state", uvalues_);
			edc.GetValuesState("adjoint", zvalues_);

			vector<Tensor<1, dealdim> > phi_u(n_dofs_per_element);
			vector<Tensor<dealdim, dealdim> > phi_q(n_dofs_per_element);

			const FEValuesExtractors::Scalar psi(0);
			const FEValuesExtractors::Vector E(1);

			Tensor<1, 2> u;
			Tensor<1, 2> z;
			Tensor<1, 2> qvalues;
			Tensor<2, 2> qgrads;
			Tensor<1, 2> dqvalues;
			Tensor<2, 2> dqgrads;
			Tensor<2, 2> DF;
			Tensor<2, 2> DF_Inverse_T;
			Tensor<2, 2> DFdq;
			Tensor<2, 2> DF_Inverse_Tdq;
			for (unsigned int q_point = 0; q_point < n_q_points; q_point++) {
			u.clear();
			z.clear();
			qgrads.clear();
			dqgrads.clear();
			DF.clear();
			DF_Inverse_T.clear();
			DFdq.clear();
			DF_Inverse_Tdq.clear();

			qgrads[0][0] = qvalues_[q_point][0]; //qgrads_[q_point][0][0];
			qgrads[1][1] = qvalues_[q_point][0]; //qgrads_[q_point][1][1];
			qgrads[1][0] = qvalues_[q_point][1]; //qgrads_[q_point][1][0];
			qgrads[0][1] = qvalues_[q_point][1]; //qgrads_[q_point][0][1];

			dqgrads[0][0] = dqvalues_[q_point][0]; //qgrads_[q_point][0][0];
			dqgrads[1][1] = dqvalues_[q_point][0]; //qgrads_[q_point][1][1];
			dqgrads[1][0] = dqvalues_[q_point][1]; //qgrads_[q_point][1][0];
			dqgrads[0][1] = dqvalues_[q_point][1]; //qgrads_[q_point][0][1];


			DF = calc_DF(qgrads);
			DF_Inverse_T = calc_invDFTranspose(DF);

			DFdq = calc_DF(dqgrads);
			DF_Inverse_Tdq = calc_invDFTranspose(DFdq); //TODO.. DF_inverseTDq ist Richtungsableitung.. also mit Phi_q "füttern.."

			for (unsigned int i = 0; i < 2; i++) {
				u[i] = uvalues_[q_point][i];
				z[i] = zvalues_[q_point][i];
			}
			for (unsigned int i = 0; i < n_dofs_per_element; i++) {
				phi_u[i] = state_fe_values[E].value(i, q_point);
				local_vector(i) += scale* (
						scalar_product(DF_Inverse_Tdq * z,	DF_Inverse_T * u)
						+scalar_product(DF_Inverse_T * z, DF_Inverse_Tdq * u))
					* control_fe_values.JxW(q_point);
						}
					}

	}

	/**********************************************************************************************************/
	/**********************************************************************************************************/

	UpdateFlags GetFaceUpdateFlags() const {
		return update_values | update_gradients | update_normal_vectors
				| update_quadrature_points;
	}

	UpdateFlags GetUpdateFlags() const {
		return update_values | update_gradients | update_quadrature_points;
	}

	unsigned int GetStateNBlocks() const {
		return 2;
	}

unsigned int GetControlNBlocks() const {
		return 1;
	}

	std::vector<unsigned int>&
	GetStateBlockComponent() {
		return state_block_component_;
	}
	const std::vector<unsigned int>&
	GetStateBlockComponent() const {
		return state_block_component_;
	}
	std::vector<unsigned int>&
	GetControlBlockComponent() {
		return control_block_component_;
	}
	const std::vector<unsigned int>&
	GetControlBlockComponent() const {
		return control_block_component_;
	}

protected:

private:
	vector<Vector<double> > uvalues_;
	vector<Vector<double> > zvalues_;

	vector<vector<Tensor<1, dealdim> > > ugrads_;
	vector<vector<Tensor<1, dealdim> > > zgrads_;
	vector<Vector<double> > qvalues_;
	vector<Vector<double> > dqvalues_;
	vector<vector<Tensor<1, dealdim> > > qgrads_;
	vector<vector<Tensor<1, dealdim> > > dqgrads_;

	vector<unsigned int> control_block_component_;
	vector<unsigned int> state_block_component_;

//TODO Hier alle Funktionen, die in Bezug zu DF benötigt werden
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
	Tensor<2, 2> calc_adjDF(Tensor<2, 2> DF) {
		Tensor<2, 2> adjDF_;
		adjDF_[0][0] = DF[1][1];
		adjDF_[1][1] = DF[0][0];
		adjDF_[1][0] = -DF[0][1];
		adjDF_[0][1] = -DF[1][0];

		return adjDF_;
	}
	Tensor<2, 2> calc_invDF(Tensor<2, 2> DF) {
		Tensor<2, 2> adjDF_ = calc_adjDF(DF);
		double detF_ = calc_detDF(DF);
		Tensor<2, 2> invDF_;
		invDF_[0][0] = 1 / detF_ * adjDF_[0][0];
		invDF_[1][1] = 1 / detF_ * adjDF_[1][1];
		invDF_[1][0] = 1 / detF_ * adjDF_[1][0];
		invDF_[0][1] = 1 / detF_ * adjDF_[0][1];

		return invDF_;
	}
	Tensor<2, 2> calc_invDFTranspose(Tensor<2, 2> DF) {
		Tensor<2, 2> invDF_ = calc_invDF(DF);
		Tensor<2, 2> invDFTranspose_;
		invDFTranspose_[0][0] = invDF_[0][0];
		invDFTranspose_[1][1] = invDF_[1][1];
		invDFTranspose_[1][0] = invDF_[0][1];
		invDFTranspose_[0][1] = invDF_[1][0];

		return invDFTranspose_;
	}

};

#endif
