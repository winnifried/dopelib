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

#include <limits>
#include <interfaces/pdeinterface.h>
#include <deal.II/base/numbers.h>
#include "functions.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

/***********************************************************************************************/
#if DEAL_II_VERSION_GTE(9, 3, 0)

template<
        template<bool DH, typename VECTOR, int dealdim> class EDC,
        template<bool DH, typename VECTOR, int dealdim> class FDC,
        bool DH, typename VECTOR, int dealdim>
class LocalPDE : public PDEInterface<EDC, FDC, DH, VECTOR, dealdim>
#else
    template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
  template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
  template<int, int> class DH, typename VECTOR, int dealdim>
class LocalPDE : public PDEInterface<EDC, FDC, DH, VECTOR, dealdim>
#endif
{
public:
    explicit LocalPDE(double obst_scale) :
            state_block_component_(2, 0), control_block_component_(3, 0), obst_scale_(obst_scale) {
        control_block_component_[1] = 1;
        control_block_component_[2] = 2;
        state_block_component_[1] = 1;
    }

    void
    ElementEquation(const EDC<DH, VECTOR, dealdim> &edc,
                    dealii::Vector<double> &local_vector, double scale,
                    double/*scale_ico*/) override
    {
    	// Initialize FE Values, Number of Degrees of Freedom and Quadrature Points
        const DOpEWrapper::FEValues<dealdim> &state_fe_values = edc.GetFEValuesState();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();

        // Set Index of State/Adjoint Components
        const FEValuesExtractors::Scalar pde(componentIndex_PDE);
        const FEValuesExtractors::Scalar mult(componentIndex_Mult);

    	// Declare Problem Variables
        vector<Vector<double> > qvalues_(n_q_points, Vector<double>(3));
        vector<Vector<double> > uvalues_(n_q_points, Vector<double>(2));
        vector<Vector<double> > obstacle_(n_q_points, Vector<double>(2));
        vector<vector<Tensor<1, dealdim> > > ugrads_(n_q_points, vector<Tensor<1, dealdim> >(2));

        // Initialize Variables
        {
            assert(this->problem_type_ == "state");
            edc.GetValuesControl("control", qvalues_);
            edc.GetValuesState("last_newton_solution", uvalues_);
            edc.GetValuesState("obstacle", obstacle_);
            edc.GetGradsState("last_newton_solution", ugrads_);
        }

    	// Declare Values: State Components, Control Components
        Tensor<1, 2> grad_u;
        Tensor<2, 2> qvals;
        double u;
        double mult_u;

        // Iterate over all quadrature points
        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {

            // Disassemble variables into the appropriate components
            GetGradPDE(grad_u, ugrads_, q_point);
            GetControlMatrix(qvals, qvalues_, q_point);
            GetComponentState(u, uvalues_, q_point);
            GetComponentMult(mult_u, uvalues_, q_point);

            // Iterate over all degrees of freedom
            for (unsigned int i = 0; i < n_dofs_per_element; i++)
            {
            	// Initialize Test Function Components
                const Tensor<1, 2> phi_i_grads_v = state_fe_values[pde].gradient(i, q_point);

                // Main part of the PDE with coefficient control
                local_vector(i) += scale
                                   * qvals
                                   * grad_u
                                   * phi_i_grads_v
                                   * state_fe_values.JxW(q_point);

                // Obstacle: The second equation is in effect when looking at vertices
                if (fabs(state_fe_values[mult].value(i, q_point) - 1.) < std::numeric_limits<double>::epsilon())
                {
                    // Get the number of neighboring elements to accurately determine the weight in each computation step
                    unsigned int n_neig = edc.GetNNeighbourElementsOfVertex(state_fe_values.quadrature_point(q_point));
                    double const weight = 1. / n_neig;

                    // Check if we are on an inner point or on the boundary
                    if (n_neig == 4)
                    {
                        // Equation for multiplier
                        local_vector(i) += scale
                    						* weight
                    						* (mult_u - std::max(0., mult_u - obst_scale_ * (obstacle_[q_point][0] - u)))
											* state_fe_values[mult].value(i, q_point);

                        // Add the multiplier to the state equation by finding the corresponding basis functions
                        for (unsigned int j = 0; j < n_dofs_per_element; j++)
                        {
                            if (fabs(state_fe_values[pde].value(j, q_point) - 1.) <
                                std::numeric_limits<double>::epsilon())
                            {
                                local_vector(j) += scale
                            						* weight
                            						* mult_u;
                            }
                        }
                    }
                	else
                	{
                        // Boundary
                        local_vector(i) += scale
                							* mult_u;
                    }
                }
            }
        }
    }

    void
    ElementMatrix(const EDC<DH, VECTOR, dealdim> &edc, FullMatrix<double> &local_matrix, double scale,
                  double /*scale_ico*/) override
    {
        // Initialize FE Values, Number of Degrees of Freedom and Quadrature Points
        const DOpEWrapper::FEValues<dealdim> &state_fe_values = edc.GetFEValuesState();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();

    	// Set Index of State/Adjoint Components
        const FEValuesExtractors::Scalar pde(componentIndex_PDE);
        const FEValuesExtractors::Scalar mult(componentIndex_Mult);

    	// Declare Problem Variables
        vector<Vector<double> > qvalues_(n_q_points, Vector<double>(3));
        vector<Vector<double> > uvalues_(n_q_points, Vector<double>(2));
        vector<Vector<double> > obstacle_(n_q_points, Vector<double>(2));

        // Initialize Variables
    	if (this->problem_type_ == "state")
        {
            edc.GetValuesControl("control", qvalues_);
            edc.GetValuesState("last_newton_solution", uvalues_);
            edc.GetValuesState("obstacle", obstacle_);
        }
    	else
    	{
    		edc.GetValuesControl("control", qvalues_);
    		edc.GetValuesState("state", uvalues_);
    		edc.GetValuesState("obstacle", obstacle_);
    	}

    	// Declare Values: State and Control
        Tensor<2, 2> qvals;
        double u;
        double mult_u;
    	// Declare Values: Test Function
        std::vector<Tensor<1, 2> > phi_grads_v(n_dofs_per_element);
        std::vector<double> phi_vals(n_dofs_per_element);

        // Iterate over all quadrature points
        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
        	// Prefill vectors of test functions for all degrees of freedom
        	// as we will need arbitrary combinations
            for (unsigned int k = 0; k < n_dofs_per_element; k++)
            {
                phi_grads_v[k] = state_fe_values[pde].gradient(k, q_point);
                phi_vals[k] = state_fe_values[pde].value(k, q_point);
            }

        	// Initialize Values: Control
            GetControlMatrix(qvals, qvalues_, q_point);
        	// Initialize Values: State Components
            GetComponentState(u, uvalues_, q_point);
            GetComponentMult(mult_u, uvalues_, q_point);

            // Iterate over all degrees of freedom row index
            for (unsigned int i = 0; i < n_dofs_per_element; i++)
            {
                // Iterate over all degrees of freedom column index
                for (unsigned int j = 0; j < n_dofs_per_element; j++)
                {
                    //Main part of the PDE
                    local_matrix(i, j) += scale
                                          * qvals
                                          * phi_grads_v[j]
                                          * phi_grads_v[i]
                                          * state_fe_values.JxW(q_point);

                    // Obstacle: Check if we are in a vertex for either index
                    if ((fabs(state_fe_values[mult].value(i, q_point) - 1.) < std::numeric_limits<double>::epsilon()) ||
                        (fabs(state_fe_values[mult].value(j, q_point) - 1.) < std::numeric_limits<double>::epsilon()))
                    {
                        //Weight to account for multiplicity when running over multiple meshes.
                        unsigned int n_neig = edc.GetNNeighbourElementsOfVertex(
                                state_fe_values.quadrature_point(q_point));
                        double const weight = 1. / n_neig;
                        // Check if we are in an inner point or on the boundary
                        if (n_neig == 4)
                        {
                            // Check if the obstacle constraint is active
                            if ((mult_u - obst_scale_ * (obstacle_[q_point][0] - u)) < 0.)
                            {
                                local_matrix(i, j) += scale
                            							* weight
                            							* state_fe_values[mult].value(i, q_point)
														* state_fe_values[mult].value(j, q_point);
                            }
                        	else
                        	{
                                local_matrix(i, j) -= scale
                        								* weight
                        								* obst_scale_
                        								* state_fe_values[pde].value(j, q_point)
                        								* state_fe_values[mult].value(i, q_point);
                            }
                            //From \lambda_j\phi_i in the first equation
                            //No need to check for the correct j, since otherwise
                            //the testfunction is zero in a vertex!
                            local_matrix(i, j) += scale
                        							* weight
                        							* state_fe_values[pde].value(i, q_point)
													* state_fe_values[mult].value(j, q_point);
                        }
                    	else
                    	{
                            // Boundary
                            local_matrix(i, j) += scale
                    								* state_fe_values[mult].value(i, q_point)
													* state_fe_values[mult].value(j, q_point);
                        }
                    }
                }
            }
        }
    }


    void ElementEquation_U(const EDC<DH, VECTOR, dealdim> &edc, dealii::Vector<double> &local_vector, double scale,
                           double /*scale_ico*/) override
    {
        // Get finite element information
        const DOpEWrapper::FEValues<dealdim> &state_fe_values = edc.GetFEValuesState();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();

    	// Set Index of State/Adjoint Components
        const FEValuesExtractors::Scalar pde(componentIndex_PDE);
        const FEValuesExtractors::Scalar mult(componentIndex_Mult);

    	// Declare Problem Variables
        vector<vector<Tensor<1, dealdim> > > zgrads_(n_q_points, std::vector<Tensor<1, dealdim> >(2));
        vector<Vector<double> > qvalues_(n_q_points, Vector<double>(3));
        vector<Vector<double> > zvalues_(n_q_points, Vector<double>(2));
        vector<Vector<double> > uvalues_(n_q_points, Vector<double>(2));
        vector<Vector<double> > obstacle_(n_q_points, Vector<double>(2));

    	// Initialize Problem Variables
        {
            assert(this->problem_type_ == "adjoint");
            edc.GetGradsState("last_newton_solution", zgrads_);
            edc.GetValuesControl("control", qvalues_);
            edc.GetValuesState("last_newton_solution", zvalues_);
            edc.GetValuesState("state", uvalues_);
            edc.GetValuesState("obstacle", obstacle_);
        }

    	// Declare Values: State, Adjoint and Control Components
        Tensor<1, 2> grad_z;
        Tensor<2, 2> qvals;
        double u;
        double mult_u;
        double z;
        double mult_z;

        // Iterate over all quadrature points
        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
            // Disassemble variables into the appropriate components
            GetGradPDE(grad_z, zgrads_, q_point);
            GetControlMatrix(qvals, qvalues_, q_point);
            GetComponentState(u, uvalues_, q_point);
            GetComponentMult(mult_u, uvalues_, q_point);
            GetComponentState(z, zvalues_, q_point);
            GetComponentMult(mult_z, zvalues_, q_point);

            // Iterate over all degrees of freedom
            for (unsigned int i = 0; i < n_dofs_per_element; i++)
            {
                // Get Test Functions
                const Tensor<1, 2> phi_i_grads_v = state_fe_values[pde].gradient(i, q_point);
                // Main part of the PDE
                local_vector(i) += scale
                                   * qvals
                                   * phi_i_grads_v
                                   * grad_z
                                   * state_fe_values.JxW(q_point);

                // Obstacle: Check if we are in a vertex
                if (fabs(state_fe_values[mult].value(i, q_point) - 1.) < std::numeric_limits<double>::epsilon())
                {
                    //Weight to account for multiplicity when running over multiple elements.
                    unsigned int n_neig = edc.GetNNeighbourElementsOfVertex(state_fe_values.quadrature_point(q_point));
                    double const weight = 1. / n_neig;
                    // Check if we are in an inner point or on the boundary
                    if (n_neig == 4)
                    {
                        // Check if the obstacle constraint is active
                        if ((mult_u - obst_scale_ * (obstacle_[q_point][0] - u)) < 0.)
                        {
                            local_vector(i) += scale
                        						* weight
                        						* mult_z
                        						* state_fe_values[mult].value(i, q_point);
                        }
                    	else
                    	{
                            for (unsigned int j = 0; j < n_dofs_per_element; j++)
                            {
                                if (fabs(state_fe_values[pde].value(j, q_point) - 1.) <
                                    std::numeric_limits<double>::epsilon())
                                {
                                    local_vector(j) -= scale
                                						* weight
                                						* mult_z
                                						* obst_scale_
                                						* state_fe_values[pde].value(j, q_point);
                                }
                            }
                        }
                        local_vector(i) += scale
                    						* weight
                    						* z
                    						* state_fe_values[mult].value(i, q_point);
                    }
                	else
                	{
                        // Boundary
                        local_vector(i) += scale
                							* mult_z
                							* state_fe_values[mult].value(i, q_point);
                    }
                }
            }
        }
    }

    void
    ElementEquation_UT(const EDC<DH, VECTOR, dealdim> &edc,
                       dealii::Vector<double> &local_vector, double scale,
                       double /*scale_ico*/) override
    {
        // Initialize FE Values, Number of Degrees of Freedom and Quadrature Points
        const DOpEWrapper::FEValues<dealdim> &state_fe_values = edc.GetFEValuesState();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();

        // We need to extract the appropriate parts of the state finite element in each step
        const FEValuesExtractors::Scalar pde(componentIndex_PDE);
        const FEValuesExtractors::Scalar mult(componentIndex_Mult);

    	// Declare Problem Variables
        vector<vector<Tensor<1, dealdim> > > dugrads_(n_q_points, std::vector<Tensor<1, dealdim> >(2));
        vector<Vector<double> > qvalues_(n_q_points, Vector<double>(3));
        vector<Vector<double> > duvalues_(n_q_points, Vector<double>(2));
        vector<Vector<double> > uvalues_(n_q_points, Vector<double>(2));
        vector<Vector<double> > obstacle_(n_q_points, Vector<double>(2));

        // Initialize Variables
        {
            assert(this->problem_type_ == "tangent");
            edc.GetValuesControl("control", qvalues_);
            edc.GetGradsState("last_newton_solution", dugrads_);
            edc.GetValuesState("last_newton_solution", duvalues_);
            edc.GetValuesState("state", uvalues_);
            edc.GetValuesState("obstacle", obstacle_);
        }

    	// Declare Values: State, Adjoint and Control Components
        Tensor<1, 2> grad_du;
        Tensor<2, 2> qvals;
        double u;
        double mult_u;
        double du;
        double mult_du;

        // Iterate over all quadrature points
        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
            // Disassemble variables into the appropriate components
            GetGradPDE(grad_du, dugrads_, q_point);
            GetControlMatrix(qvals, qvalues_, q_point);
            GetComponentState(u, uvalues_, q_point);
            GetComponentMult(mult_u, uvalues_, q_point);
            GetComponentState(du, duvalues_, q_point);
            GetComponentMult(mult_du, duvalues_, q_point);

            // Iterate over all degrees of freedom
            for (unsigned int i = 0; i < n_dofs_per_element; i++)
            {
            	// Get Test Functions
                const Tensor<1, 2> phi_i_grads_v = state_fe_values[pde].gradient(i, q_point);
                // Main part of the PDE
                local_vector(i) += scale
                                   * qvals
                                   * grad_du
                                   * phi_i_grads_v
                                   * state_fe_values.JxW(q_point);

                // Obstacle: Check if we are in a vertex
                if (fabs(state_fe_values[mult].value(i, q_point) - 1.) < std::numeric_limits<double>::epsilon())
                {

                    // Weight to account for multiplicity when running over multiple elements.
                    unsigned int n_neig = edc.GetNNeighbourElementsOfVertex(state_fe_values.quadrature_point(q_point));
                    double const weight = 1. / n_neig;


                    // Check if we are in an inner point or on the boundary
                    if (n_neig == 4)
                    {
                        // Check if the obstacle constraint is active
                        if ((mult_u - obst_scale_ * (obstacle_[q_point][0] - u)) < 0.)
                        {
                            local_vector(i) += scale
                        						* weight
                        						* mult_du
                        						* state_fe_values[mult].value(i, q_point);
                        }
                    	else
                    	{
                            local_vector(i) -= scale
                    							* weight
                    							* obst_scale_
                    							* du
                    							* state_fe_values[mult].value(i, q_point);
                    	}
                    	// We need to find all degrees of freedom that contain the Lagrange multiplier phi_i
                    	for (unsigned int j = 0; j < n_dofs_per_element; j++)
                    	{
                    		if (fabs(state_fe_values[pde].value(j, q_point) - 1.) < std::numeric_limits<double>::epsilon())
                    		{
                    			local_vector(j) += scale
                    								* weight
                    								* mult_du
                    								* state_fe_values[pde].value(j, q_point);
                            }
                        }
                    }
                	else
                	{
                        // Boundary
                        local_vector(i) += scale
                							* mult_du
                							* state_fe_values[mult].value(i, q_point);
                    }
                }
            }
        }
    }

    void
    ElementEquation_UTT(const EDC<DH, VECTOR, dealdim> &edc,
                        dealii::Vector<double> &local_vector, double scale,
                        double /*scale_ico*/) override
    {
        // Initialize FE Values, Number of Degrees of Freedom and Quadrature Points
        const DOpEWrapper::FEValues<dealdim> &state_fe_values = edc.GetFEValuesState();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();

        // We need to extract the appropriate parts of the state finite element in each step
        const FEValuesExtractors::Scalar pde(componentIndex_PDE);
        const FEValuesExtractors::Scalar mult(componentIndex_Mult);

    	// Declare Problem Variables
        vector<Vector<double> > qvalues_(n_q_points, Vector<double>(3));
        vector<Vector<double> > dzvalues_(n_q_points, Vector<double>(2));
        vector<Vector<double> > uvalues_(n_q_points, Vector<double>(2));
        vector<Vector<double> > obstacle_(n_q_points, Vector<double>(2));
        vector<vector<Tensor<1, dealdim> > > dzgrads_(n_q_points, std::vector<Tensor<1, dealdim> >(2));

        // Initialize Variables
        {
            assert(this->problem_type_ == "adjoint_hessian");
            edc.GetValuesControl("control", qvalues_);
            edc.GetGradsState("last_newton_solution", dzgrads_);
            edc.GetValuesState("last_newton_solution", dzvalues_);
            edc.GetValuesState("state", uvalues_);
            edc.GetValuesState("obstacle", obstacle_);
        }

    	// Declare Values: State, Adjoint and Control Components
        Tensor<1, 2> grad_dz;
        Tensor<2, 2> qvals;
        double u;
        double mult_u;
        double dz;
        double mult_dz;

        // Iterate over all quadrature points
        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
            // Get Values: Control and State Components
            GetGradPDE(grad_dz, dzgrads_, q_point);
            GetControlMatrix(qvals, qvalues_, q_point);
            GetComponentState(u, uvalues_, q_point);
            GetComponentMult(mult_u, uvalues_, q_point);
            GetComponentState(dz, dzvalues_, q_point);
            GetComponentMult(mult_dz, dzvalues_, q_point);

            // Iterate over all degrees of freedom
            for (unsigned int i = 0; i < n_dofs_per_element; i++)
            {
                const Tensor<1, 2> phi_i_grads_v = state_fe_values[pde].gradient(i, q_point);
                // Main part of the PDE
                local_vector(i) += scale
                                   * qvals
                                   * phi_i_grads_v
                                   * grad_dz
                                   * state_fe_values.JxW(q_point);

                // Obstacle: Check if we are in a vertex
                if (fabs(state_fe_values[mult].value(i, q_point) - 1.) < std::numeric_limits<double>::epsilon())
                {
                    // Weight to account for multiplicity when running over multiple elements.
                    unsigned int n_neig = edc.GetNNeighbourElementsOfVertex(state_fe_values.quadrature_point(q_point));
                    double const weight = 1. / n_neig;
                    // Check if we are in an inner point or on the boundary
                    if (n_neig == 4)
                    {
                        // Check if the obstacle constraint is active
                        if ((mult_u - obst_scale_ * (obstacle_[q_point][0] - u)) < 0.)
                        {
                            local_vector(i) += scale
                        						* weight
                        						* mult_dz
                        						* state_fe_values[mult].value(i, q_point);
                        }
                    	else
                    	{
                            for (unsigned int j = 0; j < n_dofs_per_element; j++)
                            {
                                if (fabs(state_fe_values[pde].value(j, q_point) - 1.) <
                                    std::numeric_limits<double>::epsilon())
                                {
                                    local_vector(j) -= scale
                                						* weight
                                						* obst_scale_
                                						* mult_dz
                                						* state_fe_values[pde].value(j, q_point);
                                }
                            }
                        }
                        local_vector(i) += scale
                    						* weight
                    						* dz
                    						* state_fe_values[mult].value(i, q_point);
                    }
                	else
                	{
                        // Boundary
                        local_vector(i) += scale
                							* mult_dz
                							* state_fe_values[mult].value(i, q_point);
                    }
                }
            }
        }
    }

    void
    ElementEquation_Q(const EDC<DH, VECTOR, dealdim> &edc,
                      dealii::Vector<double> &local_vector, double scale,
                      double /*scale_ico*/) override
    {
        // Initialize FE Values, Number of Degrees of Freedom and Quadrature Points
        const DOpEWrapper::FEValues<dealdim> &control_fe_values =
                edc.GetFEValuesControl();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();

    	// Declare Problem Variables
        vector<vector<Tensor<1, dealdim> > > zgrads_(n_q_points, std::vector<Tensor<1, dealdim> >(2));
        vector<vector<Tensor<1, dealdim> > > ugrads_(n_q_points, std::vector<Tensor<1, dealdim> >(2));

        // Initialize Variables
        {
            assert(this->problem_type_ == "gradient");
            edc.GetGradsState("adjoint", zgrads_);
            edc.GetGradsState("state", ugrads_);
        }

    	// Declare Values: Control and State Components
        Tensor<2, 2> controlTestMatrix;
        Tensor<1, 2> grad_u;
        Tensor<1, 2> grad_z;

    	//Iterate over all quadrature points
        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
        	// Get Values: State Components
            GetGradPDE(grad_u, ugrads_, q_point);
            GetGradPDE(grad_z, zgrads_, q_point);

        	// Iterate over the degrees of freedom
            for (unsigned int i = 0; i < n_dofs_per_element; i++)
            {
            	// Get Values: Control Test Function
                GetControlTestMatrix(controlTestMatrix, control_fe_values, i, q_point);
            	// Main part of the PDE
                local_vector(i) += scale
                                   * controlTestMatrix
                                   * grad_u
                                   * grad_z
                                   * control_fe_values.JxW(q_point);
            }
        }
    }

    void
    ElementEquation_QT(const EDC<DH, VECTOR, dealdim> &edc,
                       dealii::Vector<double> &local_vector, double scale,
                       double /*scale_ico*/) override
    {
		// Initialize FE Values, Number of Degrees of Freedom and Quadrature Points
        const DOpEWrapper::FEValues<dealdim> &state_fe_values = edc.GetFEValuesState();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();

    	// Set Index of State/Adjoint Components
        const FEValuesExtractors::Scalar pde(0);

    	// Declare Problem Variables
        vector<Vector<double> > dqvalues_(n_q_points, Vector<double>(3));
        vector<vector<Tensor<1, dealdim> > > ugrads_(n_q_points, std::vector<Tensor<1, dealdim> >(2));

        // Initialize Variables
        {
            assert(this->problem_type_ == "tangent");
            edc.GetValuesControl("dq", dqvalues_);
            edc.GetGradsState("state", ugrads_);
        }

    	// Declare Values: Control and State Components
        Tensor<1, 2> grad_u;
        Tensor<2, 2> dqvals;

    	// Iterate over all quadrature points
        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
        	// Get Values: Control and State Components
            GetControlMatrix(dqvals, dqvalues_, q_point);
            GetGradPDE(grad_u, ugrads_, q_point);
        	// Iterate over the degrees of freedom
            for (unsigned int i = 0; i < n_dofs_per_element; i++)
            {
            	// Main part of the PDE
                local_vector(i) += scale
                                   * dqvals
                                   * state_fe_values[pde].gradient(i, q_point)
                                   * grad_u
                                   * state_fe_values.JxW(q_point);
            }
        }
    }

    void
    ElementEquation_QTT(const EDC<DH, VECTOR, dealdim> &edc,
                        dealii::Vector<double> &local_vector, double scale,
                        double /*scale_ico*/) override
    {
		// Initialize FE Values, Number of Degrees of Freedom and Quadrature Points
        const DOpEWrapper::FEValues<dealdim> &control_fe_values =
                edc.GetFEValuesControl();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();

    	// Declare Problem Variables
        vector<vector<Tensor<1, dealdim> > > dzgrads_(n_q_points, std::vector<Tensor<1, dealdim> >(2));
        vector<vector<Tensor<1, dealdim> > > ugrads_(n_q_points, std::vector<Tensor<1, dealdim> >(2));

    	// Initialize Problem Variables
        {
            assert(this->problem_type_ == "hessian");
            edc.GetGradsState("adjoint_hessian", dzgrads_);
            edc.GetGradsState("state", ugrads_);
        }

    	// Declare Values: Control and State Components
        Tensor<2, 2> controlTestMatrix;
        Tensor<1, 2> grad_u;
        Tensor<1, 2> grad_dz;

    	// Iterate over all quadrature points
        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
            // Get Values: Control and State Components
            GetGradPDE(grad_u, ugrads_, q_point);
            GetGradPDE(grad_dz, dzgrads_, q_point);
        	// Iterate over the degrees of freedom
            for (unsigned int i = 0; i < n_dofs_per_element; i++)
            {
            	// Get Values: Controltest
                GetControlTestMatrix(controlTestMatrix, control_fe_values, i, q_point);
            	// Main part of the PDE
                local_vector(i) += scale
                                   * controlTestMatrix
                                   * grad_u
                                   * grad_dz
                                   * control_fe_values.JxW(q_point);
            }
        }
    }

    void
    ElementEquation_UU(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                       dealii::Vector<double> &/*local_vector*/, double /*scale*/,
                       double /*scale_ico*/) override
    {
        assert(this->problem_type_ == "adjoint_hessian");
    }

    void
    ElementEquation_QU(const EDC<DH, VECTOR, dealdim> &edc,
                       dealii::Vector<double> &local_vector, double scale,
                       double /*scale_ico*/) override
    {
        // Initialize FE Values, Number of Degrees of Freedom and Quadrature Points
        const DOpEWrapper::FEValues<dealdim> &state_fe_values =
                edc.GetFEValuesState();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();

    	// Set Index of State/Adjoint Components
        const FEValuesExtractors::Scalar pde(0);

    	// Declare Problem Variables
        vector<vector<Tensor<1, dealdim> > > zgrads_(n_q_points, std::vector<Tensor<1, dealdim> >(2));
        vector<Vector<double> > dqvalues_(n_q_points, Vector<double>(3));

        // Initialize Variables
        {
            assert(this->problem_type_ == "adjoint_hessian");
            edc.GetGradsState("adjoint", zgrads_);
            edc.GetValuesControl("dq", dqvalues_);
        }

    	// Declare Values: State and Control Components
        Tensor<1, 2> grad_z;
        Tensor<2, 2> dqvals;

    	// Iterate over all quadrature points
        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
            // Get Values: Control and State Components
            GetGradPDE(grad_z, zgrads_, q_point);
            GetControlMatrix(dqvals, dqvalues_, q_point);

            // Iterate over all degrees of freedom
            for (unsigned int i = 0; i < n_dofs_per_element; i++)
            {
                // Main part of the PDE
                local_vector(i) += scale
                                   * dqvals
                                   * state_fe_values[pde].gradient(i, q_point)
                                   * grad_z
                                   * state_fe_values.JxW(q_point);
            }
        }
    }

    void
    ElementEquation_UQ(const EDC<DH, VECTOR, dealdim> &edc,
                       dealii::Vector<double> &local_vector, double scale,
                       double /*scale_ico*/) override
    {
        // Initialize FE Values, Number of Degrees of Freedom and Quadrature Points
        const DOpEWrapper::FEValues<dealdim> &control_fe_values =
                edc.GetFEValuesControl();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();

    	// Declare Problem Variables
        vector<vector<Tensor<1, dealdim> > > zgrads_(n_q_points, std::vector<Tensor<1, dealdim> >(2));
        vector<vector<Tensor<1, dealdim> > > dugrads_(n_q_points, std::vector<Tensor<1, dealdim> >(2));

        // Initialize Variables
        {
            assert(this->problem_type_ == "hessian");
            edc.GetGradsState("adjoint", zgrads_);
            edc.GetGradsState("tangent", dugrads_);
        }

    	// Declare Values: State and Control Components
        Tensor<1, 2> grad_z;
        Tensor<1, 2> grad_du;
        Tensor<2, 2> controlTestMatrix;

    	// Iterate over all quadrature points
        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
            // Get Values: Control and State Components
            GetGradPDE(grad_du, dugrads_, q_point);
            GetGradPDE(grad_z, zgrads_, q_point);

        	// Iterate over all degrees of freedom
            for (unsigned int i = 0; i < n_dofs_per_element; i++)
            {
				// Get Values: Control Test Function
                GetControlTestMatrix(controlTestMatrix, control_fe_values, i, q_point);

            	// Main part of the PDE
                local_vector(i) += scale
                                   * controlTestMatrix
                                   * grad_du
                                   * grad_z
                                   * control_fe_values.JxW(q_point);
            }
        }
    }

    void
    ElementEquation_QQ(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                       dealii::Vector<double> &/*local_vector*/, double /*scale*/,
                       double /*scale_ico*/) override
    {
        assert(this->problem_type_ == "hessian");
    }

    void
    ElementRightHandSide(const EDC<DH, VECTOR, dealdim> &edc,
                         dealii::Vector<double> &local_vector, double scale) override
    {
        // Initialize FE Values, Number of Degrees of Freedom and Quadrature Points
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();
        const DOpEWrapper::FEValues<dealdim> &state_fe_values = edc.GetFEValuesState();

    	// Set Index of State/Adjoint Components
        const FEValuesExtractors::Scalar pde(0);

    	// Declare Problem Variables
        vector<double> fvalues_(n_q_points);

    	// Iterate over all quadrature points
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
        	// Get Values: Right Hand Side
            fvalues_[q_point] = local::rhs(state_fe_values.quadrature_point(q_point));

        	// Iterate over all degrees of freedom
            for (unsigned int i = 0; i < n_dofs_per_element; i++)
            {
                // Main part of the PDE
                local_vector(i) += scale
									* fvalues_[q_point]
            						* state_fe_values[pde].value(i, q_point)
									* state_fe_values.JxW(q_point);
            }
        }
    }

    void
    ControlElementEquation(const EDC<DH, VECTOR, dealdim> &edc,
                           dealii::Vector<double> &local_vector, double scale) override
    {
        // Initialize FE Values, Number of Degrees of Freedom and Quadrature Points
        const DOpEWrapper::FEValues<dealdim> &control_fe_values =
                edc.GetFEValuesControl();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();

    	// Declare Problem Variables
        vector<Vector<double> > funcgradvalues_(n_q_points, Vector<double>(3));

    	// Initialize Problem Variables
        {
            assert((this->problem_type_ == "gradient") || (this->problem_type_ == "hessian"));
            edc.GetValuesControl("last_newton_solution", funcgradvalues_);
        }

    	// Declare Values: State and Control Components
        Tensor<2, 2> funcgradqvals;
        Tensor<2, 2> controlTestMatrix;

    	// Iterate over all quadrature points
        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
        	// Get Values: Control Components
            GetControlMatrix(funcgradqvals, funcgradvalues_, q_point);
        	// Iterate over all degrees of freedom
            for (unsigned int i = 0; i < n_dofs_per_element; i++)
            {
            	// Get Values: Control Test Function
                GetControlTestMatrix(controlTestMatrix, control_fe_values, i, q_point);

                local_vector(i) += scale
                                   * (funcgradqvals[0][0] * controlTestMatrix[0][0]
                                      + funcgradqvals[0][1] * controlTestMatrix[0][1]
                                      + funcgradqvals[1][0] * controlTestMatrix[1][0]
                                      + funcgradqvals[1][1] * controlTestMatrix[1][1])
                                   * control_fe_values.JxW(q_point);
            }
        }
    }

    void
    ControlElementMatrix(const EDC<DH, VECTOR, dealdim> &edc,
                         FullMatrix<double> &local_matrix, double scale) override
    {
        // Initialize FE Values, Number of Degrees of Freedom and Quadrature Points
        const DOpEWrapper::FEValues<dealdim> &control_fe_values =
                edc.GetFEValuesControl();
        unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
        unsigned int n_q_points = edc.GetNQPoints();

    	// Declare Problem Variables
        Tensor<2, 2> controlTestMatrix_i;
        Tensor<2, 2> controlTestMatrix_j;

    	// Iterate over all quadrature points
        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
        	// Iterate over all degrees of freedom
            for (unsigned int i = 0; i < n_dofs_per_element; i++)
            {
            	// Get Values: Control Test Function
                GetControlTestMatrix(controlTestMatrix_i, control_fe_values, i, q_point);
        		// Iterate over all degrees of freedom
                for (unsigned int j = 0; j < n_dofs_per_element; j++)
                {
                	// Get Values: Control Test Function
                    GetControlTestMatrix(controlTestMatrix_j, control_fe_values, j, q_point);

                	// ControlElementMatrix
                    local_matrix(i, j) += scale
                                          * (controlTestMatrix_j[0][0] * controlTestMatrix_i[0][0]
                                             + controlTestMatrix_j[0][1] * controlTestMatrix_i[0][1]
                                             + controlTestMatrix_j[1][0] * controlTestMatrix_i[1][0]
                                             + controlTestMatrix_j[1][1] * controlTestMatrix_i[1][1])
                                          * control_fe_values.JxW(q_point);
                }
            }
        }
    }

    void FaceAuxRhs(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                    dealii::Vector<double> &/*local_vector*/,
                    double /*scale*/) override
    {
    }

    void BoundaryAuxRhs(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                        dealii::Vector<double> &/*local_vector*/,
                        double /*scale*/) override
    {
    }

    [[nodiscard]] UpdateFlags
    GetUpdateFlags() const override
    {
        if ((this->problem_type_ == "adjoint")
            || (this->problem_type_ == "state")
            || (this->problem_type_ == "tangent")
            || (this->problem_type_ == "adjoint_hessian")
            || (this->problem_type_ == "hessian")
            || (this->problem_type_ == "gradient"))
        	return update_values | update_gradients | update_quadrature_points;
        else
            throw DOpEException("Unknown Problem Type " + this->problem_type_, "LocalPDE::GetUpdateFlags");
    }

    [[nodiscard]] UpdateFlags
    GetFaceUpdateFlags() const override
    {
        return update_values | update_gradients | update_normal_vectors
               | update_quadrature_points;
    }

    [[nodiscard]] unsigned int
    GetStateNBlocks() const override
    {
        return 2;
    }

    [[nodiscard]] unsigned int
    GetControlNBlocks() const override
    {
        return 3;
    }

    std::vector<unsigned int> &
    GetStateBlockComponent() override
    {
        return state_block_component_;
    }

    [[nodiscard]] const std::vector<unsigned int> &
    GetStateBlockComponent() const override
    {
        return state_block_component_;
    }

    std::vector<unsigned int> &
    GetControlBlockComponent() override
    {
        return control_block_component_;
    }

    [[nodiscard]] const std::vector<unsigned int> &
    GetControlBlockComponent() const override
    {
        return control_block_component_;
    }

    [[nodiscard]] bool
    HasFaces() const override
    {
        return false;
    }

    [[nodiscard]] bool
    HasInterfaces() const override
    {
        return false;
    }

    [[nodiscard]] bool
    HasVertices() const override
    {
        return true;
    }


private:
    vector<unsigned int> state_block_component_;
    vector<unsigned int> control_block_component_;

    double obst_scale_;

    const int componentIndex_PDE = 0;
    const int componentIndex_Mult = 1;


    void GetComponentState(double &u, const vector<Vector<double> > &val, unsigned int q_point) const
    {
        u = val[q_point][componentIndex_PDE];
    }

    void GetComponentMult(double &pde_mult, const vector<Vector<double> > &val, unsigned int q_point) const
    {
        pde_mult = val[q_point][componentIndex_Mult];
    }

    void GetGradPDE(Tensor<1, 2> &GradPDE, const vector<std::vector<Tensor<1, dealdim> > > &grad, unsigned int q_point)
	{
        GradPDE[0] = grad[q_point][componentIndex_PDE][0];
        GradPDE[1] = grad[q_point][componentIndex_PDE][1];
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

};
//**********************************************************************************

#endif