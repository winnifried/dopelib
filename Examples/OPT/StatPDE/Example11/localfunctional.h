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

#if DEAL_II_VERSION_GTE(9, 3, 0)

template<
        template<bool DH, typename VECTOR, int dealdim> class EDC,
        template<bool DH, typename VECTOR, int dealdim> class FDC,
        bool DH, typename VECTOR, int dopedim, int dealdim =
        dopedim>
class LocalFunctional : public FunctionalInterface<EDC, FDC, DH, VECTOR,
        dopedim, dealdim>
#else
    template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
  template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
  template<int, int> class DH, typename VECTOR, int dopedim, int dealdim =
  dopedim>
class LocalFunctional : public FunctionalInterface<EDC, FDC, DH, VECTOR,
  dopedim, dealdim>
#endif
{
public:
    explicit LocalFunctional(double alpha, double beta, double lower_bound_control,
    							double upper_bound_control) :
									alpha_(alpha), beta_(beta), lowerbound_(lower_bound_control),
									upperbound_(upper_bound_control)
	{
    }

    double
    ElementValue(const EDC<DH, VECTOR, dealdim> &edc) override
    {

        const DOpEWrapper::FEValues<dealdim> &state_fe_values =
                edc.GetFEValuesState();
        unsigned int n_q_points = edc.GetNQPoints();

        vector<Vector<double> > qvalues_(n_q_points, Vector<double>(3));
        vector<Vector<double> > qdvalues_(n_q_points, Vector<double>(3));
        vector<Vector<double> > uvalues_(n_q_points, Vector<double>(2));
        vector<Vector<double> > udvalues_(n_q_points, Vector<double>(2));

        {
            edc.GetValuesControl("control", qvalues_);
            edc.GetValuesControl("qdvalues", qdvalues_);
            edc.GetValuesState("state", uvalues_);
            edc.GetValuesState("udvalues", udvalues_);
        }

        // Initialize variables of the appropriate size for all necessary getters
        Tensor<2, 2> LBMatrix;
        Tensor<2, 2> UBMatrix;
        Tensor<2, 2> qvals;
        Tensor<2, 2> qdvals;

        SetBoundControl(LBMatrix, lowerbound_);
        SetBoundControl(UBMatrix, upperbound_);

        double r = 0.;
        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
            GetControlMatrix(qvals, qvalues_, q_point);
            GetControlMatrix(qdvals, qdvalues_, q_point);

            r += 0.5 * (uvalues_[q_point][0] - udvalues_[q_point][0])
        			* (uvalues_[q_point][0] - udvalues_[q_point][0])
        			* state_fe_values.JxW(q_point);

            r += 0.5 * alpha_
        			* ((qvals[0][0] - qdvals[0][0]) * (qvals[0][0] - qdvals[0][0]) +
        				(qvals[0][1] - qdvals[0][1]) * (qvals[0][1] - qdvals[0][1]) +
        				(qvals[1][0] - qdvals[1][0]) * (qvals[1][0] - qdvals[1][0]) +
        				(qvals[1][1] - qdvals[1][1]) * (qvals[1][1] - qdvals[1][1]))
        			* state_fe_values.JxW(q_point);

            // barrier upper bound
            r -= detPenalty(qvals - LBMatrix, state_fe_values, q_point);
            // barrier lower bound
            r -= detPenalty(qvals - UBMatrix, state_fe_values, q_point);

            if(trace(qvals - LBMatrix) < 0 || trace(UBMatrix - qvals) < 0 )
            {
                r-= log(0);
            }

        }
        return r;
    }

    void
    ElementValue_U(const EDC<DH, VECTOR, dealdim> &edc,
                   dealii::Vector<double> &local_vector, double scale) override
    {
        const DOpEWrapper::FEValues<dealdim> &state_fe_values =
                edc.GetFEValuesState();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();

        vector<Vector<double> > uvalues_(n_q_points, Vector<double>(2));
        vector<Vector<double> > udvalues_(n_q_points, Vector<double>(2));

        {
            edc.GetValuesState("state", uvalues_);
            edc.GetValuesState("udvalues", udvalues_);
        }
        const FEValuesExtractors::Scalar pde(0);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
            for (unsigned int i = 0; i < n_dofs_per_element; i++)
            {
                local_vector(i) += scale
                                   * (uvalues_[q_point][0] - udvalues_[q_point][0])
                                   * state_fe_values[pde].value(i, q_point)
                                   * state_fe_values.JxW(q_point);
            }
        }
    }

    void
    ElementValue_Q(const EDC<DH, VECTOR, dealdim> &edc,
                   dealii::Vector<double> &local_vector, double scale) override
    {
        const DOpEWrapper::FEValues<dealdim> &control_fe_values =
                edc.GetFEValuesControl();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();

        vector<Vector<double> > qvalues_(n_q_points, Vector<double>(3));
        vector<Vector<double> > qdvalues_(n_q_points, Vector<double>(3));

        {
            edc.GetValuesControl("control", qvalues_);
            edc.GetValuesControl("qdvalues", qdvalues_);
        }
        // Declare variables of the appropriate size for all necessary getters
        Tensor<2, 2> LBMatrix;
        Tensor<2, 2> UBMatrix;
        Tensor<2, 2> controlTestMatrix;
        Tensor<2, 2> qvals;
        Tensor<2, 2> qdvals;

        SetBoundControl(LBMatrix, lowerbound_);
        SetBoundControl(UBMatrix, upperbound_);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++) {
            GetControlMatrix(qvals, qvalues_, q_point);
            GetControlMatrix(qdvals, qdvalues_, q_point);
            for (unsigned int i = 0; i < n_dofs_per_element; i++) {
                GetControlTestMatrix(controlTestMatrix, control_fe_values, i, q_point);

                local_vector(i) += scale * alpha_
                                   * ((qvals[0][0] - qdvals[0][0]) * controlTestMatrix[0][0]
                                      + (qvals[0][1] - qdvals[0][1]) * controlTestMatrix[0][1]
                                      + (qvals[1][0] - qdvals[1][0]) * controlTestMatrix[1][0]
                                      + (qvals[1][1] - qdvals[1][1]) * controlTestMatrix[1][1])
                                   * control_fe_values.JxW(q_point);

                // barrier lower bound
                local_vector(i) -= scale * detPenaltyDeriv(qvals - LBMatrix, controlTestMatrix, control_fe_values, q_point);
                // barrier upper bound
                local_vector(i) -= scale * detPenaltyDeriv(qvals - UBMatrix, controlTestMatrix, control_fe_values, q_point);

            }
        }
    }

    void
    ElementValue_UU(const EDC<DH, VECTOR, dealdim> &edc,
                    dealii::Vector<double> &local_vector, double scale) override
    {
        const DOpEWrapper::FEValues<dealdim> &state_fe_values =
                edc.GetFEValuesState();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();

        vector<Vector<double> > duvalues_(n_q_points, Vector<double>(2));

        {
            edc.GetValuesState("tangent", duvalues_);
        }

        const FEValuesExtractors::Scalar pde(0);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
            for (unsigned int i = 0; i < n_dofs_per_element; i++)
            {
                local_vector(i) += scale * duvalues_[q_point][0]
                                   * state_fe_values[pde].value(i, q_point)
                                   * state_fe_values.JxW(q_point);
            }
        }
    }

    void
    ElementValue_QU(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                    dealii::Vector<double> &/*local_vector*/, double /*scale*/) override
    {
    }

    void
    ElementValue_UQ(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                    dealii::Vector<double> &/*local_vector*/, double /*scale*/) override
    {
    }

    void
    ElementValue_QQ(const EDC<DH, VECTOR, dealdim> &edc,
                    dealii::Vector<double> &local_vector, double scale) override
    {
        const DOpEWrapper::FEValues<dealdim> &control_fe_values =
                edc.GetFEValuesControl();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();

        vector<Vector<double> > qvalues_(n_q_points, Vector<double>(3));
        vector<Vector<double> > dqvalues_(n_q_points, Vector<double>(3));

        {
            edc.GetValuesControl("control", qvalues_);
            edc.GetValuesControl("dq", dqvalues_);
        }
        // Declare variables of the appropriate size for all necessary getters
        Tensor<2, 2> LBMatrix;
        Tensor<2, 2> UBMatrix;
        Tensor<2, 2> controlTestMatrix;
        Tensor<2, 2> qvals;
        Tensor<2, 2> dqvals;

        SetBoundControl(LBMatrix, lowerbound_);
        SetBoundControl(UBMatrix, upperbound_);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
            GetControlMatrix(qvals, qvalues_, q_point);
            GetControlMatrix(dqvals, dqvalues_, q_point);
            for (unsigned int i = 0; i < n_dofs_per_element; i++)
            {
                GetControlTestMatrix(controlTestMatrix, control_fe_values, i, q_point);

                local_vector(i) += scale * alpha_
                                   * (dqvals[0][0] * controlTestMatrix[0][0]
                                      + dqvals[0][1] * controlTestMatrix[0][1]
                                      + dqvals[1][0] * controlTestMatrix[1][0]
                                      + dqvals[1][1] * controlTestMatrix[1][1])
                                   * control_fe_values.JxW(q_point);

                local_vector(i) -= scale * detPenaltySecondDeriv(qvals - LBMatrix, controlTestMatrix, dqvals, control_fe_values, q_point);
                local_vector(i) -= scale * detPenaltySecondDeriv(qvals - UBMatrix, controlTestMatrix, dqvals, control_fe_values, q_point);

            }
        }
    }

    [[nodiscard]] UpdateFlags
    GetUpdateFlags() const override
    {
        return update_values | update_quadrature_points;
    }

    [[nodiscard]] string
    GetType() const override
    {
        return "domain";
    }

    [[nodiscard]] string
    GetName() const override
    {
        return "cost functional";
    }

    static void GetControlMatrix(Tensor<2, 2> &ControlMatrix, const vector<Vector<double> > &val, unsigned int q_point)
	{
        ControlMatrix[0][0] = val[q_point][0];
        ControlMatrix[0][1] = val[q_point][1];
        ControlMatrix[1][0] = val[q_point][1];
        ControlMatrix[1][1] = val[q_point][2];
    }

    static void
    GetControlTestMatrix(Tensor<2, 2> &TestMatrix, const DOpEWrapper::FEValues<dealdim> &control_fe_values, int i,
                         int q_point)
	{
        const FEValuesExtractors::Scalar firstEntry(0);
        const FEValuesExtractors::Scalar secondEntry(1);
        const FEValuesExtractors::Scalar thirdEntry(2);
        TestMatrix[0][0] = control_fe_values[firstEntry].value(i, q_point);
        TestMatrix[0][1] = control_fe_values[secondEntry].value(i, q_point);
        TestMatrix[1][0] = control_fe_values[secondEntry].value(i, q_point);
        TestMatrix[1][1] = control_fe_values[thirdEntry].value(i, q_point);
    }

    static void SetBoundControl(Tensor<2, 2> &ControlBound, double a)
	{
        ControlBound[0][0] = a;
        ControlBound[0][1] = 0;
        ControlBound[1][0] = 0;
        ControlBound[1][1] = a;
    }

    static double DetDerivDirection(Tensor<2, 2> A, Tensor<2, 2> &Direction)
	{
        return A[1][1] * Direction[0][0]
               - A[1][0] * Direction[0][1]
               - A[0][1] * Direction[1][0]
               + A[0][0] * Direction[1][1];
    }

    static double DetSecondDerivDirection(Tensor<2, 2> /*A*/, Tensor<2, 2> &Direction1, Tensor<2, 2> &Direction2)
	{
        return Direction2[1][1] * Direction1[0][0]
               - Direction2[1][0] * Direction1[0][1]
               - Direction2[0][1] * Direction1[1][0]
               + Direction2[0][0] * Direction1[1][1];
    }

    double detPenalty(const Tensor<2, 2> &A, const DOpEWrapper::FEValues<dealdim> &fe_values, int q_point)
	{
        return beta_
    			* log(determinant(A))
    			* fe_values.JxW(q_point);
    }

    double detPenaltyDeriv(const Tensor<2, 2> &A, Tensor<2, 2> &Direction, const DOpEWrapper::FEValues<dealdim> &fe_values,
    						int q_point)
	{
        return beta_
    			* (DetDerivDirection(A, Direction) / determinant(A))
    			* fe_values.JxW(q_point);
    }

    double detPenaltySecondDeriv(const Tensor<2, 2> &A, Tensor<2, 2> &Direction1, Tensor<2, 2> &Direction2,
    								const DOpEWrapper::FEValues<dealdim> &fe_values, int q_point)
	{
        return beta_
    			* (((DetSecondDerivDirection(A, Direction1, Direction2) * determinant(A))
					- (DetDerivDirection(A, Direction1) * DetDerivDirection(A, Direction2)))
                    / (determinant(A) * determinant(A)))
    			* fe_values.JxW(q_point);
    }

    double trPenalty(const Tensor<2, 2> &A, const DOpEWrapper::FEValues<dealdim> &fe_values, int q_point)
	{
        return beta_
    			* log(trace(A))
    			* fe_values.JxW(q_point);
    }

    double trPenaltyDeriv(const Tensor<2, 2> &A, const Tensor<2, 2> &Direction, const DOpEWrapper::FEValues<dealdim> &fe_values,
    						int q_point)
	{
        return beta_
    			* (trace(Direction) / trace(A))
    			* fe_values.JxW(q_point);
    }

    double trPenaltySecondDeriv(const Tensor<2, 2>& A, const Tensor<2, 2> &Direction1, const Tensor<2, 2> &Direction2,
    							const DOpEWrapper::FEValues<dealdim> &fe_values, int q_point)
	{
        return beta_
    			* -1 * ((trace(Direction1) * trace(Direction2)) / (trace(A) * trace(A)))
    			* fe_values.JxW(q_point);
    }

private:
    double alpha_;
    double beta_;
    double lowerbound_;
    double upperbound_;
};

#endif