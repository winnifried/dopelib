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
#include "celldatacontainer.h"
#include "facedatacontainer.h"


using namespace std;
using namespace dealii;
using namespace DOpE;

template<typename VECTOR, int dopedim, int dealdim>
  class LocalPDE : public PDEInterface<CellDataContainer,FaceDataContainer,dealii::DoFHandler, VECTOR, dealdim>
  {
  public:

    static void declare_params(ParameterReader &param_reader)
    {
      param_reader.SetSubsection("Local PDE parameters");
      param_reader.declare_entry("density_fluid", "1.0",
				 Patterns::Double(0));
      param_reader.declare_entry("viscosity", "1.0",
				 Patterns::Double(0));
    }

  LocalPDE(ParameterReader &param_reader) : _state_block_components(3,0)
      {
	_alpha = 1.e-3;
	_state_block_components[2]= 1;

	param_reader.SetSubsection("Local PDE parameters");
	_density_fluid = param_reader.get_double ("density_fluid");
	_viscosity = param_reader.get_double ("viscosity");
      }


    void CellEquation(const CellDataContainer<dealii::DoFHandler, VECTOR, dealdim>& cdc,
                      dealii::Vector<double> &local_cell_vector,
                      double scale, double scale_ico)
    {
      assert(this->_problem_type == "state");
      const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
      unsigned int n_q_points = cdc.GetNQPoints();
      //unsigned int material_id = cdc.GetMaterialId();

      _uvalues.resize(n_q_points,Vector<double>(3));
      _ugrads.resize(n_q_points,vector<Tensor<1,2> >(3));

      cdc.GetValuesState("last_newton_solution",_uvalues);
      cdc.GetGradsState("last_newton_solution",_ugrads);

      const FEValuesExtractors::Vector velocities (0);
      const FEValuesExtractors::Scalar pressure (2);

      for (unsigned int q_point=0;q_point<n_q_points;q_point++)
	{
	  Tensor<2,dealdim> fluid_pressure;
	  fluid_pressure.clear();
	  fluid_pressure[0][0] =  -_uvalues[q_point](2);
	  fluid_pressure[1][1] =  -_uvalues[q_point](2);

	  double incompressibility = _ugrads[q_point][0][0] + _ugrads[q_point][1][1];

	  Tensor<2,2> vgrads;
	  vgrads.clear();
	  vgrads[0][0] = _ugrads[q_point][0][0];
	  vgrads[0][1] = _ugrads[q_point][0][1];
	  vgrads[1][0] = _ugrads[q_point][1][0];
	  vgrads[1][1] = _ugrads[q_point][1][1];

	  Tensor<1,2> v;
	  v.clear();
	  v[0] = _uvalues[q_point](0);
	  v[1] = _uvalues[q_point](1);

	  Tensor<1,2> convection_fluid = vgrads * v;

	  for (unsigned int i=0;i<n_dofs_per_cell;i++)
	    {
	      const Tensor<1,2> phi_i_v = state_fe_values[velocities].value (i, q_point);
	      const Tensor<2,2> phi_i_grads_v = state_fe_values[velocities].gradient (i, q_point);
	      const double phi_i_p = state_fe_values[pressure].value (i, q_point);

	      local_cell_vector(i) +=  scale * (convection_fluid * phi_i_v
						+ _viscosity * scalar_product(vgrads + transpose(vgrads),phi_i_grads_v)
						)
		* state_fe_values.JxW(q_point);

	      local_cell_vector(i) +=  scale_ico 
		* (scalar_product(fluid_pressure, phi_i_grads_v) + incompressibility * phi_i_p)
		* state_fe_values.JxW(q_point);

	    }
	}

    }

    void CellMatrix(const CellDataContainer<dealii::DoFHandler, VECTOR, dealdim>& cdc,
		    FullMatrix<double> &local_entry_matrix, double scale, double scale_ico)
    {
      const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
      unsigned int n_q_points = cdc.GetNQPoints();
      //unsigned int material_id = cdc.GetMaterialId();

      const FEValuesExtractors::Vector velocities (0);
      const FEValuesExtractors::Scalar pressure (2);

      _uvalues.resize(n_q_points,Vector<double>(3));
      _ugrads.resize(n_q_points,vector<Tensor<1,2> >(3));

      cdc.GetValuesState("last_newton_solution",_uvalues);
      cdc.GetGradsState("last_newton_solution",_ugrads);

      std::vector<Tensor<1,2> >     phi_v (n_dofs_per_cell);
      std::vector<Tensor<2,2> >     phi_grads_v (n_dofs_per_cell);
      std::vector<double>           phi_p (n_dofs_per_cell);
      std::vector<double>           div_phi_v (n_dofs_per_cell);

      for(unsigned int q_point = 0; q_point < n_q_points; q_point++)
	{
	   for(unsigned int k = 0; k < n_dofs_per_cell; k++)
	     {
	       phi_v[k]       = state_fe_values[velocities].value (k, q_point);
	       phi_grads_v[k] = state_fe_values[velocities].gradient (k, q_point);
	       phi_p[k]       = state_fe_values[pressure].value (k, q_point);
	       div_phi_v[k]   = state_fe_values[velocities].divergence (k, q_point);
	     }



	   Tensor<2,2> vgrads;
	   vgrads.clear();
	   vgrads[0][0] = _ugrads[q_point][0][0];
	   vgrads[0][1] = _ugrads[q_point][0][1];
	   vgrads[1][0] = _ugrads[q_point][1][0];
	   vgrads[1][1] = _ugrads[q_point][1][1];

	   Tensor<1,2> v;
	   v[0] = _uvalues[q_point](0);
	   v[1] = _uvalues[q_point](1);

	  for(unsigned int i = 0; i < n_dofs_per_cell; i++)
	    {
	      Tensor<2,dealdim> fluid_pressure_LinP;
	      fluid_pressure_LinP.clear();
	      fluid_pressure_LinP[0][0] = -phi_p[i];
	      fluid_pressure_LinP[1][1] = -phi_p[i];
	      
	      Tensor<1,2> convection_fluid_LinV = phi_grads_v[i] * v + vgrads * phi_v[i];

	      for(unsigned int j = 0; j < n_dofs_per_cell; j++)
		{
		  local_entry_matrix(j,i) += scale * (convection_fluid_LinV * phi_v[j]
					      + _viscosity * scalar_product(phi_grads_v[i] + transpose(phi_grads_v[i]), phi_grads_v[j])
					      )
		    * state_fe_values.JxW(q_point);

		  local_entry_matrix(j,i) += scale_ico * (scalar_product(fluid_pressure_LinP, phi_grads_v[j])
					      + (phi_grads_v[i][0][0] + phi_grads_v[i][1][1]) * phi_p[j]
					      )
		    * state_fe_values.JxW(q_point);
		}
	    }
	}

    }


    void CellRightHandSide(const CellDataContainer<dealii::DoFHandler, VECTOR, dealdim>& /*cdc*/,
                           dealii::Vector<double> &local_cell_vector __attribute__((unused)),
                           double scale __attribute__((unused)))
    {
      assert(this->_problem_type == "state");
    }


    void CellTimeEquationExplicit (const CellDataContainer<dealii::DoFHandler, VECTOR, dealdim>& /*cdc*/,
                           dealii::Vector<double> &local_cell_vector __attribute__((unused)),
				   double scale __attribute__((unused)))
    {
      assert(this->_problem_type == "state");
    }



 void CellTimeEquation(const CellDataContainer<dealii::DoFHandler, VECTOR, dealdim>& cdc,
                           dealii::Vector<double> &local_cell_vector __attribute__((unused)),
		       double scale __attribute__((unused)))
    {
      assert(this->_problem_type == "state");

      const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
      unsigned int n_q_points = cdc.GetNQPoints();

      _uvalues.resize(n_q_points,Vector<double>(3));

      cdc.GetValuesState("last_newton_solution",_uvalues);

      const FEValuesExtractors::Vector velocities (0);

      for (unsigned int q_point=0;q_point<n_q_points;q_point++)
	{
	  Tensor<1,2> v;
	  v[0] = _uvalues[q_point](0);
	  v[1] = _uvalues[q_point](1);

	  for (unsigned int i=0;i<n_dofs_per_cell;i++)
	    {
	      const Tensor<1,2> phi_i_v = state_fe_values[velocities].value (i, q_point);

	      local_cell_vector(i) +=  scale * (v * phi_i_v) * state_fe_values.JxW(q_point);
	    }
	}


    }


 void CellTimeMatrixExplicit(const CellDataContainer<dealii::DoFHandler, VECTOR, dealdim>& /*cdc*/,
			     FullMatrix<double> &/*local_entry_matrix*/)
    {
      assert(this->_problem_type == "state");
    }


  void CellTimeMatrix(const CellDataContainer<dealii::DoFHandler, VECTOR, dealdim>& cdc,
		      FullMatrix<double> &local_entry_matrix)
  {
    assert(this->_problem_type == "state");
    
    const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
    unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
    unsigned int n_q_points = cdc.GetNQPoints();

      const FEValuesExtractors::Vector velocities (0);

      std::vector<Tensor<1,2> >     phi_v (n_dofs_per_cell);

      for(unsigned int q_point = 0; q_point < n_q_points; q_point++)
	{
	   for(unsigned int k = 0; k < n_dofs_per_cell; k++)
	     {
	       phi_v[k] = state_fe_values[velocities].value (k, q_point);
	     }

	   for(unsigned int i = 0; i < n_dofs_per_cell; i++)
	     {
	       for(unsigned int j = 0; j < n_dofs_per_cell; j++)
		 {
		   local_entry_matrix(j,i) += (phi_v[i] * phi_v[j]) * state_fe_values.JxW(q_point);
		 }
	     }
	}

    }


  // Values for boundary integrals
    void BoundaryEquation (const FaceDataContainer<dealii::DoFHandler, VECTOR, dealdim>& fdc,
			   dealii::Vector<double> &local_cell_vector,
			   double scale, double /*scale_ico*/)
    {

      assert(this->_problem_type == "state");

      const auto & state_fe_face_values = fdc.GetFEFaceValuesState();
      unsigned int n_dofs_per_cell = fdc.GetNDoFsPerCell();
      unsigned int n_q_points = fdc.GetNQPoints();
      unsigned int color = fdc.GetBoundaryIndicator();


      // do-nothing applied on outflow boundary
      if (color == 1)
	{
	  _ufacegrads.resize(n_q_points,vector<Tensor<1,2> >(3));

	  fdc.GetFaceGradsState("last_newton_solution",_ufacegrads);

	  const FEValuesExtractors::Vector velocities (0);

	  for (unsigned int q_point=0;q_point<n_q_points;q_point++)
	    {
	      Tensor<2,2> vgrads;
	      vgrads.clear();
	      vgrads[0][0] = _ufacegrads[q_point][0][0];
	      vgrads[0][1] = _ufacegrads[q_point][0][1];
	      vgrads[1][0] = _ufacegrads[q_point][1][0];
	      vgrads[1][1] = _ufacegrads[q_point][1][1];

	      for (unsigned int i=0;i<n_dofs_per_cell;i++)
		{
		  const Tensor<1,2> phi_i_v = state_fe_face_values[velocities].value (i, q_point);

		  const Tensor<1,2> neumann_value
		    = _viscosity * (transpose(vgrads) * state_fe_face_values.normal_vector(q_point));

		  local_cell_vector(i) -= scale * neumann_value * phi_i_v  * state_fe_face_values.JxW(q_point);
		}
	    }
	}

    }

    void BoundaryMatrix (const FaceDataContainer<dealii::DoFHandler, VECTOR, dealdim>& fdc,
			 dealii::FullMatrix<double> &local_entry_matrix, double /*scale_ico*/, double /*scale_ico*/)
    {
      assert(this->_problem_type == "state");
      
      const auto & state_fe_face_values = fdc.GetFEFaceValuesState();
      unsigned int n_dofs_per_cell = fdc.GetNDoFsPerCell();
      unsigned int n_q_points = fdc.GetNQPoints();
      unsigned int color = fdc.GetBoundaryIndicator();

      // do-nothing applied on outflow boundary
      if (color == 1)
	{
	  const FEValuesExtractors::Vector velocities (0);

	  for (unsigned int q_point=0;q_point<n_q_points;q_point++)
	    {
	      for(unsigned int i = 0; i < n_dofs_per_cell; i++)
		{
		  const Tensor<2,2> phi_j_grads_v = state_fe_face_values[velocities].gradient (i, q_point);
		  const Tensor<1,2> neumann_value
		    = _viscosity * (transpose(phi_j_grads_v) * state_fe_face_values.normal_vector(q_point));

		  for(unsigned int j = 0; j < n_dofs_per_cell; j++)
		    {
		      const Tensor<1,2> phi_i_v = state_fe_face_values[velocities].value (j, q_point);

		      local_entry_matrix(j,i) -=  neumann_value *  phi_i_v  * state_fe_face_values.JxW(q_point);
		    }
		}
	    }
	}
    }

    void BoundaryRightHandSide (const FaceDataContainer<dealii::DoFHandler, VECTOR, dealdim>&,
				dealii::Vector<double> &local_cell_vector __attribute__((unused)),
				double scale __attribute__((unused)))
    {
      assert(this->_problem_type == "state");
    }




    UpdateFlags GetUpdateFlags() const
    {
      if((this->_problem_type == "adjoint") || (this->_problem_type == "state")
	 || (this->_problem_type == "tangent")|| (this->_problem_type == "adjoint_hessian")||(this->_problem_type == "hessian"))
	return update_values | update_gradients  | update_quadrature_points;
      else if((this->_problem_type == "gradient"))
	return update_values | update_quadrature_points;
      else
	throw DOpEException("Unknown Problem Type "+this->_problem_type ,"LocalPDE::GetUpdateFlags");
    }

    UpdateFlags GetFaceUpdateFlags() const
    {
      if((this->_problem_type == "adjoint") || (this->_problem_type == "state")
	 || (this->_problem_type == "tangent")|| (this->_problem_type == "adjoint_hessian")||(this->_problem_type == "hessian"))
	return update_values | update_gradients  | update_normal_vectors | update_quadrature_points;
      else if((this->_problem_type == "gradient"))
	return update_values | update_quadrature_points;
      else
	throw DOpEException("Unknown Problem Type "+this->_problem_type ,"LocalPDE::GetUpdateFlags");
    }




    unsigned int GetControlNBlocks() const
    { return 1;}

    unsigned int GetStateNBlocks() const
    {
      return 2;
    }

    std::vector<unsigned int>& GetControlBlockComponent(){ return _block_components; }
    const std::vector<unsigned int>& GetControlBlockComponent() const{ return _block_components; }
    std::vector<unsigned int>& GetStateBlockComponent(){ return _state_block_components; }
    const std::vector<unsigned int>& GetStateBlockComponent() const{ return _state_block_components; }


  private:
    vector<double> _qvalues;
    vector<double> _dqvalues;
    vector<double> _funcgradvalues;
    vector<Vector<double> > _fvalues;
    vector<Vector<double> > _uvalues;

    vector<vector<Tensor<1,dealdim> > > _ugrads;
    vector<double> _zvalues;
    vector<Tensor<1,dealdim> > _zgrads;
    vector<double> _duvalues;
    vector<Tensor<1,dealdim> > _dugrads;
    vector<double> _dzvalues;
    vector<Tensor<1,dealdim> > _dzgrads;

    // face values
    vector<double> _qfacevalues;
    vector<Vector<double> > _ffacevalues;
    vector<Vector<double> > _ufacevalues;
    vector<vector<Tensor<1,dealdim> > > _ufacegrads;


    vector<unsigned int> _state_block_components;
    vector<unsigned int> _block_components;
    double _alpha, _cell_diameter;

    double _density_fluid, _viscosity;

  };
#endif
