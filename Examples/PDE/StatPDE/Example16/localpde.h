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

#ifndef LOCALPDE_H_
#define LOCALPDE_H_

#include<limits>

#include <interfaces/pdeinterface.h>
#include <deal.II/base/numbers.h>
#include "functions.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

/***********************************************************************************************/
template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
  template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
  template<int, int> class DH, typename VECTOR, int dealdim>
class LocalPDELaplace : public PDEInterface<EDC, FDC, DH, VECTOR, dealdim>
{
public:
  
  LocalPDELaplace() :
    state_block_component_(2, 0)
  {
    state_block_component_[1] =1;
  }

  void
  ElementEquation(const EDC<DH, VECTOR, dealdim> &edc,
                  dealii::Vector<double> &local_vector, double scale,
                  double/*scale_ico*/)
  {
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();

    assert(this->problem_type_ == "state");

    ugrads_.resize(n_q_points, std::vector<Tensor<1, dealdim> >(2));
    uvalues_.resize(n_q_points, Vector<double>(2));
    obstacle_.resize(n_q_points, Vector<double>(2));
    edc.GetGradsState("last_newton_solution", ugrads_);
    edc.GetValuesState("last_newton_solution", uvalues_);
    edc.GetValuesState("obstacle", obstacle_);

    const FEValuesExtractors::Scalar pde(0);
    const FEValuesExtractors::Scalar mult(1);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
    {
      Tensor<1, 2> vgrads;
      vgrads.clear();
      vgrads[0] = ugrads_[q_point][0][0];
      vgrads[1] = ugrads_[q_point][0][1];
      
      for (unsigned int i = 0; i < n_dofs_per_element; i++)
      {
	const Tensor<1, 2> phi_i_grads_v =
	  state_fe_values[pde].gradient(i, q_point);

	//The acual PDE operator, i.e. -\Delta u
	local_vector(i) += scale
	  * vgrads * phi_i_grads_v
	  * state_fe_values.JxW(q_point);
	//Second equation, only in vertices, so we check whether the lambda test function
	// is one (i.e. we are in a vertex)
	//Notice that we don't need to multiply with a testfunction as
	//its value is known to be one!
	if(fabs(state_fe_values[mult].value(i,q_point) - 1.) < std::numeric_limits<double>::epsilon())
	{
	  //Weight to account for multiplicity when running over multiple elements.
	  unsigned int n_neig = edc.GetNNeighbourElementsOfVertex(state_fe_values.quadrature_point(q_point));
	  double weight = 1./n_neig;
	  if(n_neig == 4)
	  {
	    //Equation for multiplier, i.e. lambda - max(0,\lambda-(u(x_i)-chi(x_i)))
	    local_vector(i) += scale * weight* (uvalues_[q_point][1]
						- std::max(0.,uvalues_[q_point][1]-(uvalues_[q_point][0]-obstacle_[q_point][0])));
	    //Add Multiplier to the state equation
	    //To do that, we need to find out to which basis function j for the first equation
	    //the current index i of the multiplier belongs.
	    //This is easy, since it must be the one index j where phi^j(x_i) = 1
	    for(unsigned int j = 0; j < n_dofs_per_element; j++)
	    {
	      if(fabs(state_fe_values[pde].value(j,q_point) - 1.) < std::numeric_limits<double>::epsilon())
	      {
		local_vector(j) -= scale * weight* uvalues_[q_point][1];
	      }
	    }
	  }
	  else //Boundary or hanging node (no weight, so it works if hanging)
	  {
	     local_vector(i) += scale * uvalues_[q_point][1];
	  }
	}
      }
    }
  }

  void
  ElementMatrix(const EDC<DH, VECTOR, dealdim> &edc,
                FullMatrix<double> &local_matrix, double scale,
                double/*scale_ico*/)
  {
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();
    //unsigned int material_id = edc.GetMaterialId();
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();

    const FEValuesExtractors::Scalar pde(0);
    const FEValuesExtractors::Scalar mult(1);

    std::vector<Tensor<1, 2> > phi_grads_v(n_dofs_per_element);
    std::vector<double> phi_vals(n_dofs_per_element);

    uvalues_.resize(n_q_points, Vector<double>(2));
    obstacle_.resize(n_q_points, Vector<double>(2));
    
    if(this->problem_type_ == "state")
      edc.GetValuesState("last_newton_solution", uvalues_);
    else
      edc.GetValuesState("state", uvalues_);
    edc.GetValuesState("obstacle", obstacle_);
    
    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
        for (unsigned int k = 0; k < n_dofs_per_element; k++)
          {
            phi_grads_v[k] = state_fe_values[pde].gradient(k, q_point);
	    phi_vals[k] = state_fe_values[pde].value(k, q_point);
          }

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            for (unsigned int j = 0; j < n_dofs_per_element; j++)
              {
		//Matrix for -\Delta
                local_matrix(i, j) += scale * (
		   phi_grads_v[j] * phi_grads_v[i]
		  )* state_fe_values.JxW(q_point);

		//Second equation, only in vertices, so we check whether one of the 
		//lambda test function
		// is one (i.e. we are in a vertex)
		if(
		  (fabs(state_fe_values[mult].value(i,q_point) - 1.) < std::numeric_limits<double>::epsilon())
		  ||
		  (fabs(state_fe_values[mult].value(j,q_point) - 1.) < std::numeric_limits<double>::epsilon())
		  )
		{
		  //Weight to account for multiplicity when running over multiple meshes.
		  unsigned int n_neig = edc.GetNNeighbourElementsOfVertex(state_fe_values.quadrature_point(q_point));
		  double weight = 1./n_neig;

		  if(n_neig == 4)
		  {
		    //Derivative is different if the \max in the complementarity function
		    //is 0 or lambda-(u-\chi) > 0
		    //max = 0 
		    if( (uvalues_[q_point][1]-(uvalues_[q_point][0]-obstacle_[q_point][0])) <= 0. )
		    {
		      local_matrix(i, j) += scale * weight* state_fe_values[mult].value(i,q_point)
			*state_fe_values[mult].value(j,q_point);
		    }
		    else //max > 0
		    {
		      //From Complementarity
		      local_matrix(i, j) += scale * weight* state_fe_values[pde].value(j,q_point)
			*state_fe_values[mult].value(i,q_point);
		    }
		    //From \lambda_j\phi_i in the first equation
		    //No need to check for the correct j, since otherwise
		    //the testfuncction is zero in a vertex!
		    local_matrix(i, j) -= scale * weight* state_fe_values[pde].value(i,q_point)
		      *state_fe_values[mult].value(j,q_point);
		  }
		  else //Boundary or hanging node no weight so it works when hanging
		  {
		    local_matrix(i, j) += scale *  state_fe_values[mult].value(i,q_point)
		      *state_fe_values[mult].value(j,q_point);
		  }
		}
              }
          }
      }
  }

  void
  ElementRightHandSide(const EDC<DH, VECTOR, dealdim> &edc,
                       dealii::Vector<double> &local_vector, double scale)
  {
    assert(this->problem_type_ == "state");
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
    unsigned int n_q_points = edc.GetNQPoints();
    const DOpEWrapper::FEValues<dealdim> &state_fe_values =
      edc.GetFEValuesState();

    fvalues_.resize(n_q_points);
    const FEValuesExtractors::Scalar pde(0);

    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
      {
        fvalues_[q_point] = local::rhs(
                              state_fe_values.quadrature_point(q_point));

        for (unsigned int i = 0; i < n_dofs_per_element; i++)
          {
            local_vector(i) += scale * fvalues_[q_point]
                               * state_fe_values[pde].value(i, q_point)
                               * state_fe_values.JxW(q_point);
          }
      } //endfor qpoint
  }
  
  UpdateFlags
  GetUpdateFlags() const
  {
    return update_values | update_gradients 
           | update_quadrature_points;
  }

  UpdateFlags
  GetFaceUpdateFlags() const
  {
    return update_values | update_gradients | update_normal_vectors
           | update_quadrature_points;
  }

  unsigned int
  GetStateNBlocks() const
  {
    return 2;
  }
  std::vector<unsigned int> &
  GetStateBlockComponent()
  {
    return state_block_component_;
  }
  const std::vector<unsigned int> &
  GetStateBlockComponent() const
  {
    return state_block_component_;
  }
  bool
  HasFaces() const
  {
    return false;
  }
  bool
  HasInterfaces() const
  {
    return false;
  }
  bool
  HasVertices() const
  {
    return true;
  }
private:

  vector<double> fvalues_;
  vector<Vector<double> > PI_h_z_;
  vector<Vector<double> > lap_u_;
  vector<Vector<double> > obstacle_;

  vector<std::vector<Tensor<1, dealdim> > > ugrads_;
  vector<std::vector<Tensor<1, dealdim> > > ugrads_nbr_;
  vector<Vector<double> > uvalues_;
  vector<Vector<double> > auxvalues_;

  vector<unsigned int> state_block_component_;

}
;
//**********************************************************************************

#endif

