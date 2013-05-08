/**
*
* Copyright (C) 2012 by the DOpElib authors
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

#ifndef _LOCALPDE_
#define _LOCALPDE_

#include "pdeinterface.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

template<template<int, int> class DH, typename VECTOR, int dealdim>
  class LocalPDE : public PDEInterface<CellDataContainer,FaceDataContainer,DH, VECTOR,dealdim>
  {
  public:
  LocalPDE() : _control_block_components(5,0), _state_block_components(2,0)
      {
	_state_block_components[1]= 1;
	_control_block_components[1]= 1;
	_control_block_components[2]= 2;
 	_control_block_components[3]= 3;
      	_control_block_components[4]= 4;
      }

      void CellEquation(const CellDataContainer<DH, VECTOR, dealdim>& cdc,
			dealii::Vector<double> &local_cell_vector, double scale, double /*scale_ico*/)
    {
      const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
      unsigned int n_q_points = cdc.GetNQPoints();
      {
	//Reading data
	assert(this->_problem_type == "state");
	_ugrads.resize(n_q_points,vector<Tensor<1,dealdim> >(2));
	_fvalues.resize(n_q_points,Vector<double>(2));
	//Geting u
	cdc.GetGradsState("last_newton_solution",_ugrads);
      }
      const FEValuesExtractors::Scalar comp_0 (0);
      const FEValuesExtractors::Scalar comp_1 (1);

      for(unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
	_fvalues[q_point](0) = 20.*M_PI*M_PI*( sin(M_PI * state_fe_values.quadrature_point(q_point)(0)) *
					      sin(M_PI * state_fe_values.quadrature_point(q_point)(1)));
	_fvalues[q_point](1) = 1.;

	for(unsigned int i = 0; i < n_dofs_per_cell; i++)
	{
	  local_cell_vector(i) += scale *(_ugrads[q_point][0] * state_fe_values[comp_0].gradient (i, q_point)
					  +  _ugrads[q_point][1] * state_fe_values[comp_1].gradient (i, q_point)
					  -_fvalues[q_point](0)*state_fe_values[comp_0].value (i, q_point)
					  -_fvalues[q_point](1)*state_fe_values[comp_1].value (i, q_point))
	    * state_fe_values.JxW(q_point);
	}
      }
    }

    void CellEquation_U(const CellDataContainer<DH, VECTOR, dealdim>& cdc,
			dealii::Vector<double> &local_cell_vector, double scale, double /*scale_ico*/)
    {
      const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
      unsigned int n_q_points = cdc.GetNQPoints();
      {
	assert(this->_problem_type == "adjoint");
	_zgrads.resize(n_q_points,vector<Tensor<1,dealdim> >(2));
	//We don't need u so we don't search for state
	cdc.GetGradsState("last_newton_solution",_zgrads);
      }

      const FEValuesExtractors::Scalar comp_0 (0);
      const FEValuesExtractors::Scalar comp_1 (1);
      for(unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
	for(unsigned int i = 0; i < n_dofs_per_cell; i++)
	{
	  local_cell_vector(i) += scale *(_zgrads[q_point][0] * state_fe_values[comp_0].gradient (i, q_point)
					  +  _zgrads[q_point][1] * state_fe_values[comp_1].gradient (i, q_point))
	    * state_fe_values.JxW(q_point);
	}
      }
    }

    void CellEquation_UT(const CellDataContainer<DH, VECTOR, dealdim>& cdc,
			 dealii::Vector<double> &local_cell_vector, double scale, double /*scale_ico*/)
    {
      const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
      unsigned int n_q_points = cdc.GetNQPoints();
      {
        assert(this->_problem_type == "tangent");
	_dugrads.resize(n_q_points,vector<Tensor<1,dealdim> >(2));
	cdc.GetGradsState("last_newton_solution",_dugrads);
      }
      const FEValuesExtractors::Scalar comp_0 (0);
      const FEValuesExtractors::Scalar comp_1 (1);
      for(unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
	for(unsigned int i = 0; i < n_dofs_per_cell; i++)
	{
	  local_cell_vector(i) += scale *(_dugrads[q_point][0] * state_fe_values[comp_0].gradient (i, q_point)
					  +  _dugrads[q_point][1] * state_fe_values[comp_1].gradient (i, q_point))
	    * state_fe_values.JxW(q_point);
	}
      }
    }

    void CellEquation_UTT(const CellDataContainer<DH, VECTOR, dealdim>& cdc,
			  dealii::Vector<double> &local_cell_vector, double scale, double /*scale_ico*/)
    {
      const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
      unsigned int n_q_points = cdc.GetNQPoints();
      {
        assert(this->_problem_type == "adjoint_hessian");
	_dzgrads.resize(n_q_points,vector<Tensor<1,dealdim> >(2));
	cdc.GetGradsState("last_newton_solution",_dzgrads);
      }
      const FEValuesExtractors::Scalar comp_0 (0);
      const FEValuesExtractors::Scalar comp_1 (1);
      for(unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
	for(unsigned int i = 0; i < n_dofs_per_cell; i++)
	{
	  local_cell_vector(i) += scale *(_dzgrads[q_point][0] * state_fe_values[comp_0].gradient (i, q_point)
					  +  _dzgrads[q_point][1] * state_fe_values[comp_1].gradient (i, q_point))
	    * state_fe_values.JxW(q_point);
	}
      }
    }

    void CellEquation_Q(const CellDataContainer<DH, VECTOR, dealdim>& cdc __attribute__((unused)),
			dealii::Vector<double> &local_cell_vector __attribute__((unused)), double scale __attribute__((unused)), double /*scale_ico*/)
    {

    }

    void CellEquation_QT(const CellDataContainer<DH, VECTOR, dealdim>& cdc __attribute__((unused)),
			 dealii::Vector<double> &local_cell_vector __attribute__((unused)), double scale __attribute__((unused)), double /*scale_ico*/)
    {

    }

    void CellEquation_QTT(const CellDataContainer<DH, VECTOR, dealdim>& cdc __attribute__((unused)),
			  dealii::Vector<double> &local_cell_vector __attribute__((unused)), double scale __attribute__((unused)), double /*scale_ico*/)
    {

    }

    void CellEquation_UU(const CellDataContainer<DH, VECTOR, dealdim>& cdc __attribute__((unused)),
			 dealii::Vector<double> &local_cell_vector __attribute__((unused)), double scale __attribute__((unused)), double /*scale_ico*/)
    {
      assert(this->_problem_type == "adjoint_hessian");
    }
    void CellEquation_QU(const CellDataContainer<DH, VECTOR, dealdim>& cdc __attribute__((unused)),
			 dealii::Vector<double> &local_cell_vector __attribute__((unused)), double scale __attribute__((unused)), double /*scale_ico*/)
    {
      assert(this->_problem_type == "adjoint_hessian");
    }
    void CellEquation_UQ(const CellDataContainer<DH, VECTOR, dealdim>& cdc __attribute__((unused)),
			 dealii::Vector<double> &local_cell_vector __attribute__((unused)), double scale __attribute__((unused)), double /*scale_ico*/)
    {
      assert(this->_problem_type == "hessian");
    }
    void CellEquation_QQ(const CellDataContainer<DH, VECTOR, dealdim>& cdc __attribute__((unused)),
			 dealii::Vector<double> &local_cell_vector __attribute__((unused)), double scale __attribute__((unused)), double /*scale_ico*/)
    {
      assert(this->_problem_type == "hessian");
    }

    void CellRightHandSide(const CellDataContainer<DH, VECTOR, dealdim>& cdc __attribute__((unused)),
			   dealii::Vector<double> &local_cell_vector __attribute__((unused)), double scale __attribute__((unused)))
    {
	assert(this->_problem_type == "state");
    }

    void CellMatrix(const CellDataContainer<DH, VECTOR, dealdim>& cdc,
		    FullMatrix<double> &local_entry_matrix, double scale, double /*scale_ico*/)
    {
      const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
      unsigned int n_q_points = cdc.GetNQPoints();

      const FEValuesExtractors::Scalar comp_0 (0);
      const FEValuesExtractors::Scalar comp_1 (1);

      for(unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
	for(unsigned int i = 0; i < n_dofs_per_cell; i++)
	{
	  for(unsigned int j = 0; j < n_dofs_per_cell; j++)
	  {
	    local_entry_matrix(i,j) += scale * 
	      state_fe_values[comp_0].gradient (i, q_point)*state_fe_values[comp_0].gradient (j, q_point)
	      * state_fe_values.JxW(q_point);

	    local_entry_matrix(i,j) += scale * 
	      state_fe_values[comp_1].gradient (i, q_point)*state_fe_values[comp_1].gradient (j, q_point)
	      * state_fe_values.JxW(q_point);
	  }
	}
      }
    }

    void ControlCellEquation(const CellDataContainer<DH, VECTOR, dealdim>& cdc,
			     dealii::Vector<double> &local_cell_vector, double scale)
    {
      {
 	assert((this->_problem_type == "gradient")||(this->_problem_type == "hessian"));
 	_funcgradvalues.reinit(GetControlNBlocks());
 	cdc.GetParamValues("last_newton_solution",_funcgradvalues);
      }

//      for(unsigned int q_point = 0; q_point < n_q_points; q_point++)
//      {
	for(unsigned int i = 0; i < GetControlNBlocks(); i++)
 	{
 	  local_cell_vector(i) += scale * _funcgradvalues(i);
 	}
//      }
    }

    void ControlCellMatrix(const CellDataContainer<DH, VECTOR, dealdim>& cdc __attribute__((unused)),
			   FullMatrix<double> &local_entry_matrix)
    {
//        for(unsigned int q_point = 0; q_point < n_q_points; q_point++)
//        {
	  for(unsigned int i = 0; i < GetControlNBlocks(); i++)
	  {
	    local_entry_matrix(i,i) += 1.;
	  }
//        }
    }

    UpdateFlags GetUpdateFlags() const
    {
      return update_values | update_gradients | update_quadrature_points;
    }
    UpdateFlags GetFaceUpdateFlags() const
    {
      return update_values | update_quadrature_points;
    }
    unsigned int GetControlNBlocks() const{ return 5;}
    unsigned int GetStateNBlocks() const{ return 2;}
    std::vector<unsigned int>& GetControlBlockComponent(){ return _control_block_components; }
    const std::vector<unsigned int>& GetControlBlockComponent() const{ return _control_block_components; }
    std::vector<unsigned int>& GetStateBlockComponent(){ return _state_block_components; }
    const std::vector<unsigned int>& GetStateBlockComponent() const{ return _state_block_components; }

  private:
    Vector<double> _qvalues;
    Vector<double> _dqvalues;
    Vector<double> _funcgradvalues;
    vector<Vector<double> > _fvalues;
    vector<Vector<double> > _zvalues;
    vector<Vector<double> > _dzvalues;

    vector<vector<Tensor<1,dealdim> > > _ugrads;
    vector<vector<Tensor<1,dealdim> > > _zgrads;
    vector<vector<Tensor<1,dealdim> > > _dugrads;
    vector<vector<Tensor<1,dealdim> > > _dzgrads;
    vector<unsigned int> _control_block_components;
    vector<unsigned int> _state_block_components;
  };
#endif
