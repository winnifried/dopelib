#ifndef _LOCALPDE_
#define _LOCALPDE_

#include "pdeinterface.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

template<typename VECTOR, int dopedim, int dealdim>
  class LocalPDE : public PDEInterface<CellDataContainer,FaceDataContainer,dealii::DoFHandler<dealdim>, VECTOR, dopedim,dealdim>
  {
  public:

    static void declare_params(ParameterReader &param_reader)
    {
      param_reader.SetSubsection("Local PDE parameters");
      param_reader.declare_entry("density_fluid", "0.0",
				 Patterns::Double(0));
      param_reader.declare_entry("viscosity", "0.0",
				 Patterns::Double(0));
      param_reader.declare_entry("alpha_u", "0.0",
				 Patterns::Double(0));
      param_reader.declare_entry("mu", "0.0",
				 Patterns::Double(0));
      param_reader.declare_entry("poisson_ratio_nu", "0.0",
				 Patterns::Double(0));
    }


  LocalPDE(ParameterReader &param_reader) : _control_block_components(2,0), _state_block_components(5,0)
      {
	// control block components
	_control_block_components[0]= 0;
	_control_block_components[1]= 1;

	// state block components
	_state_block_components[2]= 1;  // pressure
	_state_block_components[3]= 2;  // displacement
	_state_block_components[4]= 2;

	param_reader.SetSubsection("Local PDE parameters");
	_density_fluid = param_reader.get_double ("density_fluid");
	_viscosity = param_reader.get_double ("viscosity");
	_alpha_u = param_reader.get_double ("alpha_u");

	_lame_coefficient_mu = param_reader.get_double ("mu");
	_poisson_ratio_nu = param_reader.get_double ("poisson_ratio_nu");
	_lame_coefficient_lambda =  (2 * _poisson_ratio_nu * _lame_coefficient_mu)/
	  (1.0 - 2 * _poisson_ratio_nu);
      }


  // Domain values for cells
    void CellEquation(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
		      dealii::Vector<double> &local_cell_vector, double scale, double /*scale_ico*/)
    {
      const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
      unsigned int n_q_points = cdc.GetNQPoints();

      assert(this->_problem_type == "state");

      _uvalues.resize(n_q_points,Vector<double>(5));
      _ugrads.resize(n_q_points,vector<Tensor<1,2> >(5));

      // Getting state values
      cdc.GetValuesState("last_newton_solution",_uvalues);
      cdc.GetGradsState("last_newton_solution",_ugrads);


      const FEValuesExtractors::Vector velocities (0);
      const FEValuesExtractors::Scalar pressure (2);
      const FEValuesExtractors::Vector displacements (3);

      for(unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
	Tensor<2,2> v_grads;
	 v_grads.clear();
	 v_grads[0][0] = _ugrads[q_point][0][0];
	 v_grads[0][1] = _ugrads[q_point][0][1];
	 v_grads[1][0] = _ugrads[q_point][1][0];
	 v_grads[1][1] = _ugrads[q_point][1][1];

	Tensor<1,2> v;
	 v.clear();
	 v[0] = _uvalues[q_point](0);
	 v[1] = _uvalues[q_point](1);

	 //double v_press = _uvalues[q_point](2);
	double v_incompressibility = v_grads[0][0] +  v_grads[1][1];

	Tensor<1,2> convection_fluid = v_grads * v;

	Tensor<2,dealdim> fluid_pressure;
	fluid_pressure.clear();
	fluid_pressure[0][0] =  -_uvalues[q_point](2);
	fluid_pressure[1][1] =  -_uvalues[q_point](2);

	Tensor<2,2> u_grad;
	 u_grad.clear();
	 u_grad[0][0] = _ugrads[q_point][3][0];
	 u_grad[0][1] = _ugrads[q_point][3][1];
	 u_grad[1][0] = _ugrads[q_point][4][0];
	 u_grad[1][1] = _ugrads[q_point][4][1];

	for(unsigned int i = 0; i < n_dofs_per_cell; i++)
	{
	  const Tensor<1,2> phi_i_v = state_fe_values[velocities].value (i, q_point);
	  const Tensor<2,2> phi_i_grads_v = state_fe_values[velocities].gradient (i, q_point);
	  const double phi_i_p = state_fe_values[pressure].value (i, q_point);
	  const Tensor<2,2> phi_i_grads_u = state_fe_values[displacements].gradient (i, q_point);

	  local_cell_vector(i) += scale *
	    (
	     scalar_product(fluid_pressure, phi_i_grads_v) +
	     _viscosity * scalar_product(v_grads + transpose(v_grads),phi_i_grads_v) +
	     convection_fluid * phi_i_v +
	     v_incompressibility * phi_i_p +
	     + _alpha_u * scalar_product(u_grad, phi_i_grads_u)
	     )
	    * state_fe_values.JxW(q_point);
	}
      }


    }


    void CellMatrix (const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
		     dealii::FullMatrix<double> &local_entry_matrix, double scale, double /*scale_ico*/)
    {
      const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
      unsigned int n_q_points = cdc.GetNQPoints();

      _uvalues.resize(n_q_points,Vector<double>(5));
      _ugrads.resize(n_q_points,vector<Tensor<1,2> >(5));

      // Getting previous Newton solutions via "last_newton_solution"
      // for the nonlinear convection term for CellEquation
      // (PDE). In contrast the equations for
      // "adjoint", "tangent", etc. need the "state" values
      // for the linearized convection term.
      if (this->_problem_type == "state")
	{
	  cdc.GetValuesState("last_newton_solution",_uvalues);
	  cdc.GetGradsState("last_newton_solution",_ugrads);
	}
      else
	{
	  cdc.GetValuesState("state",_uvalues);
	  cdc.GetGradsState("state",_ugrads);
	}

      const FEValuesExtractors::Vector velocities (0);
      const FEValuesExtractors::Scalar pressure (2);
      const FEValuesExtractors::Vector displacements (3);

      for(unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
	Tensor<2,2> v_grads;
	 v_grads.clear();
	 v_grads[0][0] = _ugrads[q_point][0][0];
	 v_grads[0][1] = _ugrads[q_point][0][1];
	 v_grads[1][0] = _ugrads[q_point][1][0];
	 v_grads[1][1] = _ugrads[q_point][1][1];

	Tensor<1,2> v;
	v.clear();
	v[0] = _uvalues[q_point](0);
	v[1] = _uvalues[q_point](1);

	//double v_press = _uvalues[q_point](2);
	//double v_incompressibility = v_grads[0][0] +  v_grads[1][1];

	Tensor<2,2> u_grads;
	 u_grads.clear();
	 u_grads[0][0] = _ugrads[q_point][3][0];
	 u_grads[0][1] = _ugrads[q_point][3][1];
	 u_grads[1][0] = _ugrads[q_point][4][0];
	 u_grads[1][1] = _ugrads[q_point][4][1];

	 for(unsigned int j = 0; j < n_dofs_per_cell; j++)
	   {
	     const Tensor<1,2> phi_j_v = state_fe_values[velocities].value (j, q_point);
	     const Tensor<2,2> phi_j_grads_v = state_fe_values[velocities].gradient (j, q_point);
	     const double phi_j_p = state_fe_values[pressure].value (j, q_point);
	     const Tensor<2,2> phi_j_grads_u = state_fe_values[displacements].gradient (j, q_point);

	     Tensor<2,dealdim> fluid_pressure_LinP;
	     fluid_pressure_LinP.clear();
	     fluid_pressure_LinP[0][0] = -phi_j_p;
	     fluid_pressure_LinP[1][1] = -phi_j_p;

	     for(unsigned int i = 0; i < n_dofs_per_cell; i++)
	       {
		 const Tensor<1,2> phi_i_v = state_fe_values[velocities].value (i, q_point);
		 const Tensor<2,2> phi_i_grads_v = state_fe_values[velocities].gradient (i, q_point);
		 const double phi_i_p = state_fe_values[pressure].value (i, q_point);
		 const Tensor<2,2> phi_i_grads_u = state_fe_values[displacements].gradient (i, q_point);


		 local_entry_matrix(i,j) += scale *
		   (scalar_product(fluid_pressure_LinP, phi_i_grads_v)
		    + _viscosity *
		    scalar_product(phi_j_grads_v + transpose(phi_j_grads_v), phi_i_grads_v)
		    + (phi_j_grads_v * v + v_grads * phi_j_v) * phi_i_v
		    + (phi_j_grads_v[0][0] + phi_j_grads_v[1][1]) * phi_i_p
		    + _alpha_u * scalar_product(phi_j_grads_u, phi_i_grads_u)
		    )
		   * state_fe_values.JxW(q_point);
	       }
	   }
      }
    }


    void CellEquation_U(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
			dealii::Vector<double> &local_cell_vector, double scale, double /*scale_ico*/)
    {
      const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
      unsigned int n_q_points = cdc.GetNQPoints();
      
      assert(this->_problem_type == "adjoint");
      
      _zvalues.resize(n_q_points,Vector<double>(5));
      _zgrads.resize(n_q_points,vector<Tensor<1,2> >(5));
      
      cdc.GetValuesState("last_newton_solution",_zvalues);
      cdc.GetGradsState("last_newton_solution",_zgrads);
      
      _z_state_values.resize(n_q_points,Vector<double>(5));
      _z_state_grads.resize(n_q_points,vector<Tensor<1,2> >(5));

      cdc.GetValuesState("state",_z_state_values);
      cdc.GetGradsState("state",_z_state_grads);

      const FEValuesExtractors::Vector velocities (0);
      const FEValuesExtractors::Scalar pressure (2);
      const FEValuesExtractors::Vector displacements (3);
      
      
      for(unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
	Tensor<2,2> zv_grads;
	zv_grads.clear();
	zv_grads[0][0] = _zgrads[q_point][0][0];
	zv_grads[0][1] = _zgrads[q_point][0][1];
	zv_grads[1][0] = _zgrads[q_point][1][0];
	zv_grads[1][1] = _zgrads[q_point][1][1];
	
	Tensor<1,2> zv;
	zv.clear();
	zv[0] = _zvalues[q_point](0);
	zv[1] = _zvalues[q_point](1);
	
	// state values which contains
	// solution from previous Newton step
	// Necessary for fluid convection term
	Tensor<2,2> zv_state_grads;
	zv_state_grads.clear();
	zv_state_grads[0][0] = _z_state_grads[q_point][0][0];
	zv_state_grads[0][1] = _z_state_grads[q_point][0][1];
	zv_state_grads[1][0] = _z_state_grads[q_point][1][0];
	zv_state_grads[1][1] = _z_state_grads[q_point][1][1];
	
	Tensor<1,2> zv_state;
	zv_state.clear();
	zv_state[0] = _z_state_values[q_point](0);
	zv_state[1] = _z_state_values[q_point](1);
	
	double zp = _zvalues[q_point](2);
	
	Tensor<2,2> zu_grads;
	zu_grads.clear();
	zu_grads[0][0] = _zgrads[q_point][3][0];
	zu_grads[0][1] = _zgrads[q_point][3][1];
	zu_grads[1][0] = _zgrads[q_point][4][0];
	zu_grads[1][1] = _zgrads[q_point][4][1];
	
	
	for(unsigned int i = 0; i < n_dofs_per_cell; i++)
	{
	  const Tensor<1,2> phi_i_v = state_fe_values[velocities].value (i, q_point);
	  const Tensor<2,2> phi_i_grads_v = state_fe_values[velocities].gradient (i, q_point);
	  const double phi_i_p = state_fe_values[pressure].value (i, q_point);
	  const Tensor<2,2> phi_i_grads_u = state_fe_values[displacements].gradient (i, q_point);
	  
	  Tensor<2,dealdim> fluid_pressure_phi_i;
	  fluid_pressure_phi_i.clear();
	  fluid_pressure_phi_i[0][0] =  -phi_i_p;
	  fluid_pressure_phi_i[1][1] =  -phi_i_p;
	  
	  
	  local_cell_vector(i) +=  scale *
	    (scalar_product(fluid_pressure_phi_i, zv_grads)
	     + _viscosity *
	     scalar_product(phi_i_grads_v + transpose(phi_i_grads_v), zv_grads)
	     + (phi_i_grads_v * zv_state + zv_state_grads * phi_i_v) * zv
	     + (phi_i_grads_v[0][0] + phi_i_grads_v[1][1]) * zp
	     + _alpha_u * scalar_product(phi_i_grads_u, zu_grads)
	      )
	    * state_fe_values.JxW(q_point);
	  
	}
      }
    }
    
    
    void CellEquation_UT(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
			 dealii::Vector<double> &local_cell_vector, double scale, double /*scale_ico*/)
    {
      const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
      unsigned int n_q_points = cdc.GetNQPoints();
      
      assert(this->_problem_type == "tangent");

      _duvalues.resize(n_q_points,Vector<double>(5));
      _dugrads.resize(n_q_points,vector<Tensor<1,2> >(5));

      cdc.GetValuesState("last_newton_solution",_duvalues);
      cdc.GetGradsState("last_newton_solution",_dugrads);

      _du_state_values.resize(n_q_points,Vector<double>(5));
      _du_state_grads.resize(n_q_points,vector<Tensor<1,2> >(5));

      cdc.GetValuesState("state",_du_state_values);
      cdc.GetGradsState("state",_du_state_grads);

      const FEValuesExtractors::Vector velocities (0);
      const FEValuesExtractors::Scalar pressure (2);
      const FEValuesExtractors::Vector displacements (3);


      for(unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
	Tensor<2,dealdim> du_pI;
	du_pI.clear();
	du_pI[0][0] =  -_duvalues[q_point](2);
	du_pI[1][1] =  -_duvalues[q_point](2);

	Tensor<2,2> duv_grads;
	duv_grads.clear();
	duv_grads[0][0] = _dugrads[q_point][0][0];
	duv_grads[0][1] = _dugrads[q_point][0][1];
	duv_grads[1][0] = _dugrads[q_point][1][0];
	duv_grads[1][1] = _dugrads[q_point][1][1];

	Tensor<1,2> duv;
	duv.clear();
	duv[0] = _duvalues[q_point](0);
	duv[1] = _duvalues[q_point](1);

	//double dup = _duvalues[q_point](2);
	double duv_incompressibility = duv_grads[0][0] +  duv_grads[1][1];

	Tensor<2,2> duu_grads;
	duu_grads.clear();
	duu_grads[0][0] = _dugrads[q_point][3][0];
	duu_grads[0][1] = _dugrads[q_point][3][1];
	duu_grads[1][0] = _dugrads[q_point][4][0];
	duu_grads[1][1] = _dugrads[q_point][4][1];

	// state values which contains
	// solution from previous Newton step
	// Necessary for fluid convection term
	Tensor<2,2> duv_state_grads;
	duv_state_grads.clear();
	duv_state_grads[0][0] = _du_state_grads[q_point][0][0];
	duv_state_grads[0][1] = _du_state_grads[q_point][0][1];
	duv_state_grads[1][0] = _du_state_grads[q_point][1][0];
	duv_state_grads[1][1] = _du_state_grads[q_point][1][1];

	Tensor<1,2> duv_state;
	duv_state.clear();
	duv_state[0] = _du_state_values[q_point](0);
	duv_state[1] = _du_state_values[q_point](1);

	for(unsigned int i = 0; i < n_dofs_per_cell; i++)
	{
	  const Tensor<1,2> phi_i_v = state_fe_values[velocities].value (i, q_point);
	  const Tensor<2,2> phi_i_grads_v = state_fe_values[velocities].gradient (i, q_point);
	  const double phi_i_p = state_fe_values[pressure].value (i, q_point);
	  const Tensor<2,2> phi_i_grads_u = state_fe_values[displacements].gradient (i, q_point);

	  local_cell_vector(i) += scale *
	    (
	     scalar_product(du_pI, phi_i_grads_v)
	     + _viscosity * scalar_product(duv_grads + transpose(duv_grads),phi_i_grads_v)
	     + ( duv_grads * duv_state +  duv_state_grads * duv) * phi_i_v
	     + duv_incompressibility * phi_i_p
	     + _alpha_u * scalar_product(duu_grads, phi_i_grads_u)
	     )
	    * state_fe_values.JxW(q_point);
	}
      }


    }


    void CellEquation_UTT(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
			  dealii::Vector<double> &local_cell_vector, double scale, double /*scale_ico*/)
    {
      const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
      unsigned int n_q_points = cdc.GetNQPoints();
      
      assert(this->_problem_type == "adjoint_hessian");

      _dzvalues.resize(n_q_points,Vector<double>(5));
      _dzgrads.resize(n_q_points,vector<Tensor<1,2> >(5));

      cdc.GetValuesState("last_newton_solution",_dzvalues);
      cdc.GetGradsState("last_newton_solution",_dzgrads);

      _dz_state_values.resize(n_q_points,Vector<double>(5));
      _dz_state_grads.resize(n_q_points,vector<Tensor<1,2> >(5));

      cdc.GetValuesState("state",_dz_state_values);
      cdc.GetGradsState("state",_dz_state_grads);

      const FEValuesExtractors::Vector velocities (0);
      const FEValuesExtractors::Scalar pressure (2);
      const FEValuesExtractors::Vector displacements (3);


      for(unsigned int q_point = 0; q_point < n_q_points; q_point++)
	{
	  Tensor<2,2> dzv_grads;
	  dzv_grads.clear();
	  dzv_grads[0][0] = _dzgrads[q_point][0][0];
	  dzv_grads[0][1] = _dzgrads[q_point][0][1];
	  dzv_grads[1][0] = _dzgrads[q_point][1][0];
	  dzv_grads[1][1] = _dzgrads[q_point][1][1];

	  Tensor<1,2> dzv;
	  dzv.clear();
	  dzv[0] = _dzvalues[q_point](0);
	  dzv[1] = _dzvalues[q_point](1);

	  // state values which contains
	  // solution from previous Newton step
	  // Necessary for fluid convection term
	  Tensor<2,2> dzv_state_grads;
	  dzv_state_grads.clear();
	  dzv_state_grads[0][0] = _dz_state_grads[q_point][0][0];
	  dzv_state_grads[0][1] = _dz_state_grads[q_point][0][1];
	  dzv_state_grads[1][0] = _dz_state_grads[q_point][1][0];
	  dzv_state_grads[1][1] = _dz_state_grads[q_point][1][1];

	  Tensor<1,2> dzv_state;
	  dzv_state.clear();
	  dzv_state[0] = _dz_state_values[q_point](0);
	  dzv_state[1] = _dz_state_values[q_point](1);

	  double dzp = _dzvalues[q_point](2);

	  Tensor<2,2> dzu_grads;
	  dzu_grads.clear();
	  dzu_grads[0][0] = _dzgrads[q_point][3][0];
	  dzu_grads[0][1] = _dzgrads[q_point][3][1];
	  dzu_grads[1][0] = _dzgrads[q_point][4][0];
	  dzu_grads[1][1] = _dzgrads[q_point][4][1];


	  for(unsigned int i = 0; i < n_dofs_per_cell; i++)
	    {
	      const Tensor<1,2> phi_i_v = state_fe_values[velocities].value (i, q_point);
	      const Tensor<2,2> phi_i_grads_v = state_fe_values[velocities].gradient (i, q_point);
	      const double phi_i_p = state_fe_values[pressure].value (i, q_point);
	      const Tensor<2,2> phi_i_grads_u = state_fe_values[displacements].gradient (i, q_point);

	      Tensor<2,dealdim> fluid_pressure_phi_i;
	      fluid_pressure_phi_i.clear();
	      fluid_pressure_phi_i[0][0] =  -phi_i_p;
	      fluid_pressure_phi_i[1][1] =  -phi_i_p;


	      local_cell_vector(i) +=  scale *
		  (scalar_product(fluid_pressure_phi_i, dzv_grads)
		   + _viscosity *
		   scalar_product(phi_i_grads_v + transpose(phi_i_grads_v), dzv_grads)
		   + (phi_i_grads_v * dzv_state + dzv_state_grads * phi_i_v) * dzv
		   + (phi_i_grads_v[0][0] + phi_i_grads_v[1][1]) * dzp
		   + _alpha_u * scalar_product(phi_i_grads_u, dzu_grads)
		   )
		* state_fe_values.JxW(q_point);
	    }
	}
    }

    void CellEquation_UU(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
			 dealii::Vector<double> &local_cell_vector, double scale, double /*scale_ico*/)
    {
      const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
      unsigned int n_q_points = cdc.GetNQPoints();

      assert(this->_problem_type == "adjoint_hessian");

      _zvalues.resize(n_q_points,Vector<double>(5));
      _zgrads.resize(n_q_points,vector<Tensor<1,2> >(5));

      cdc.GetValuesState("adjoint",_zvalues);
      cdc.GetGradsState("adjoint",_zgrads);

      _du_state_values.resize(n_q_points,Vector<double>(5));
      _du_state_grads.resize(n_q_points,vector<Tensor<1,2> >(5));

      cdc.GetValuesState("tangent",_du_state_values);
      cdc.GetGradsState("tangent",_du_state_grads);

      const FEValuesExtractors::Vector velocities (0);
      const FEValuesExtractors::Scalar pressure (2);
      const FEValuesExtractors::Vector displacements (3);


      for(unsigned int q_point = 0; q_point < n_q_points; q_point++)
	{
	  Tensor<2,2> zv_grads;
	  zv_grads.clear();
	  zv_grads[0][0] = _zgrads[q_point][0][0];
	  zv_grads[0][1] = _zgrads[q_point][0][1];
	  zv_grads[1][0] = _zgrads[q_point][1][0];
	  zv_grads[1][1] = _zgrads[q_point][1][1];

	  Tensor<1,2> zv;
	  zv.clear();
	  zv[0] = _zvalues[q_point](0);
	  zv[1] = _zvalues[q_point](1);

	  // state values which contains
	  // solution from previous Newton step
	  // Necessary for fluid convection term
	  Tensor<2,2> duv_state_grads;
	  duv_state_grads.clear();
	  duv_state_grads[0][0] = _du_state_grads[q_point][0][0];
	  duv_state_grads[0][1] = _du_state_grads[q_point][0][1];
	  duv_state_grads[1][0] = _du_state_grads[q_point][1][0];
	  duv_state_grads[1][1] = _du_state_grads[q_point][1][1];

	  Tensor<1,2> duv_state;
	  duv_state.clear();
	  duv_state[0] = _du_state_values[q_point](0);
	  duv_state[1] = _du_state_values[q_point](1);


	  for(unsigned int i = 0; i < n_dofs_per_cell; i++)
	    {
	      const Tensor<1,2> phi_i_v = state_fe_values[velocities].value (i, q_point);
	      const Tensor<2,2> phi_i_grads_v = state_fe_values[velocities].gradient (i, q_point);

	      local_cell_vector(i) +=  scale *
		((phi_i_grads_v * duv_state + duv_state_grads * phi_i_v) * zv
		   )
		* state_fe_values.JxW(q_point);
	    }
	}



    }


    // Look for BoundaryEquationQ
    void CellEquation_Q(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc __attribute__((unused)),
			dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/, double /*scale_ico*/)
    {
      assert(this->_problem_type == "gradient");
    }


    void CellEquation_QT(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc __attribute__((unused)),
			 dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/, double /*scale_ico*/)
    {
      assert(this->_problem_type == "tangent");
    }

    void CellEquation_QTT(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc __attribute__((unused)),
			  dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/, double /*scale_ico*/)
    {
      assert(this->_problem_type == "hessian");
    }


    void CellEquation_QU(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc __attribute__((unused)),
			 dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/, double /*scale_ico*/)
    {
      assert(this->_problem_type == "adjoint_hessian");
    }
    void CellEquation_UQ(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc __attribute__((unused)),
			 dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/, double /*scale_ico*/)
    {
      assert(this->_problem_type == "hessian");
    }
    void CellEquation_QQ(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc __attribute__((unused)),
			 dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/, double /*scale_ico*/)
    {
      assert(this->_problem_type == "hessian");
    }


  void CellRightHandSide(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc __attribute__((unused)),
			 dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/)
  {
      assert(this->_problem_type == "state");
  }



  // Values for Boundary integrals
  void BoundaryEquation (const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
			 dealii::Vector<double> &local_cell_vector, double scale, double /*scale_ico*/)
  {
    const auto & state_fe_face_values = fdc.GetFEFaceValuesState();
    unsigned int n_dofs_per_cell = fdc.GetNDoFsPerCell();
    unsigned int n_q_points = fdc.GetNQPoints();
    unsigned int color = fdc.GetBoundaryIndicator();
    assert(this->_problem_type == "state");

    // do-nothing condition applied at outflow boundary due symmetric part of
    // fluid's stress tensor
    if (color == 1)
      {
	_uboundarygrads.resize(n_q_points,vector<Tensor<1,2> >(5));

	fdc.GetFaceGradsState("last_newton_solution",_uboundarygrads);

	const FEValuesExtractors::Vector velocities (0);
	const FEValuesExtractors::Scalar pressure (2);

	for (unsigned int q_point=0;q_point<n_q_points;q_point++)
	  {
	    Tensor<2,2> v_grad;
	    v_grad.clear();
	    v_grad[0][0] = _uboundarygrads[q_point][0][0];
	    v_grad[0][1] = _uboundarygrads[q_point][0][1];
	    v_grad[1][0] = _uboundarygrads[q_point][1][0];
	    v_grad[1][1] = _uboundarygrads[q_point][1][1];

	    const Tensor<2,2> do_nothing
	      = _density_fluid * _viscosity *  transpose(v_grad);

	    const Tensor<1,2> neumann_value
	      = do_nothing* state_fe_face_values.normal_vector(q_point);

	    for (unsigned int i=0;i<n_dofs_per_cell;i++)
	      {
		const Tensor<1,2> phi_i_v = state_fe_face_values[velocities].value (i, q_point);

		local_cell_vector(i) -= scale * neumann_value * phi_i_v * state_fe_face_values.JxW(q_point);
	      }
	  }
      }

    // Get Param Values for the Control
    // They are initialized in main.cc
    _qvalues.reinit(2);
    fdc.GetParamValues("control",_qvalues);

    // control value for the upper part: \Gamma_q0
    if (color == 50)
      {
	const FEValuesExtractors::Vector velocities (0);

	for (unsigned int q_point=0;q_point<n_q_points;q_point++)
	  {
	    for (unsigned int i=0;i<n_dofs_per_cell;i++)
	      {
		const Tensor<1,2> phi_i_v = state_fe_face_values[velocities].value (i, q_point);

		local_cell_vector(i) -= scale *
		  _qvalues(0) * state_fe_face_values.normal_vector(q_point) *
		  phi_i_v * state_fe_face_values.JxW(q_point);
	      }

	  }
       }

    // control value for the lower part: \Gamma_q1
      if (color == 51)
       {
	const FEValuesExtractors::Vector velocities (0);

	for (unsigned int q_point=0;q_point<n_q_points;q_point++)
	  {
	    for (unsigned int i=0;i<n_dofs_per_cell;i++)
	      {
		const Tensor<1,2> phi_i_v = state_fe_face_values[velocities].value (i, q_point);

		local_cell_vector(i) -= scale *
		  _qvalues(1) * state_fe_face_values.normal_vector(q_point) *
		  phi_i_v * state_fe_face_values.JxW(q_point);
	      }

	  }
       }
  }


  void BoundaryMatrix (const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
		       dealii::FullMatrix<double> &local_entry_matrix, double /*scale*/, double /*scale_ico*/)
  {
    const auto & state_fe_face_values = fdc.GetFEFaceValuesState();
    unsigned int n_dofs_per_cell = fdc.GetNDoFsPerCell();
    unsigned int n_q_points = fdc.GetNQPoints();
    unsigned int color = fdc.GetBoundaryIndicator();

    // do-nothing applied on outflow boundary
    if (color == 1)
      {
	_uboundarygrads.resize(n_q_points,vector<Tensor<1,2> >(5));

	if (this->_problem_type == "state")
	  fdc.GetFaceGradsState("last_newton_solution",_uboundarygrads);
	else
	  fdc.GetFaceGradsState("state",_uboundarygrads);

	const FEValuesExtractors::Vector velocities (0);

	for (unsigned int q_point=0;q_point<n_q_points;q_point++)
	  {
	    Tensor<2,2> v_grad;
	    v_grad[0][0] = _uboundarygrads[q_point][0][0];
	    v_grad[0][1] = _uboundarygrads[q_point][0][1];
	    v_grad[1][0] = _uboundarygrads[q_point][1][0];
	    v_grad[1][1] = _uboundarygrads[q_point][1][1];

	    for(unsigned int i = 0; i < n_dofs_per_cell; i++)
	      {
		const Tensor<1,2> phi_i_v = state_fe_face_values[velocities].value (i, q_point);

		  for(unsigned int j = 0; j < n_dofs_per_cell; j++)
		    {
		      const Tensor<2,2> phi_j_grads_v = state_fe_face_values[velocities].gradient (j, q_point);

		      // do-nothing
		      Tensor<2,2> do_nothing_LinAll;
		      do_nothing_LinAll = _density_fluid * _viscosity * transpose(phi_j_grads_v);

		      const Tensor<1,2> neumann_value
			= do_nothing_LinAll * state_fe_face_values.normal_vector(q_point);

		      local_entry_matrix(i,j) -=  neumann_value * phi_i_v * state_fe_face_values.JxW(q_point);
		    }
	      }
	  }
      }



  }


  void BoundaryRightHandSide (const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc __attribute__((unused)),
			      dealii::Vector<double> &local_cell_vector __attribute__((unused)), double scale __attribute__((unused)))
  {
    assert(this->_problem_type == "state");
  }




  void BoundaryEquation_Q (const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
			   dealii::Vector<double> &local_cell_vector,
			   double scale, double /*scale_ico*/)
  {
    const auto & state_fe_face_values = fdc.GetFEFaceValuesState();
    unsigned int n_q_points = fdc.GetNQPoints();
    unsigned int color = fdc.GetBoundaryIndicator();

    assert(this->_problem_type == "gradient");

    _zboundaryvalues.resize(n_q_points,Vector<double>(5));

    fdc.GetFaceValuesState("adjoint",_zboundaryvalues);

    // control values for the upper and lower part
    if (color == 50)
      {
	for (unsigned int q_point=0;q_point<n_q_points;q_point++)
	  {
	    Tensor<1,2> zvboundary;
	    zvboundary.clear();
	    zvboundary[0] = _zboundaryvalues[q_point](0);
	    zvboundary[1] = _zboundaryvalues[q_point](1);

	    local_cell_vector(0) -= scale *
	      1.0 * state_fe_face_values.normal_vector(q_point) *
	      zvboundary *state_fe_face_values.JxW(q_point);
	  }
      }
    if (color == 51)
      {
	for (unsigned int q_point=0;q_point<n_q_points;q_point++)
	  {
	    Tensor<1,2> zvboundary;
	    zvboundary.clear();
	    zvboundary[0] = _zboundaryvalues[q_point](0);
	    zvboundary[1] = _zboundaryvalues[q_point](1);

	    local_cell_vector(1) -= scale *
	      1.0 * state_fe_face_values.normal_vector(q_point) *
	      zvboundary *state_fe_face_values.JxW(q_point);

	  }
      }
  }

 void BoundaryEquation_QT (const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
			   dealii::Vector<double> &local_cell_vector,
			   double scale, double /*scale_ico*/)
 {
   const auto & state_fe_face_values = fdc.GetFEFaceValuesState();
   unsigned int n_dofs_per_cell = fdc.GetNDoFsPerCell();
   unsigned int n_q_points = fdc.GetNQPoints();
   unsigned int color = fdc.GetBoundaryIndicator();
   assert(this->_problem_type == "tangent");

    _dqvalues.reinit(2);
    fdc.GetParamValues("dq",_dqvalues);

    if (color == 50)
      {
	const FEValuesExtractors::Vector velocities (0);

	for (unsigned int q_point=0;q_point<n_q_points;q_point++)
	  {
	    for (unsigned int i=0;i<n_dofs_per_cell;i++)
	      {
		const Tensor<1,2> phi_i_v = state_fe_face_values[velocities].value (i, q_point);
		local_cell_vector(i) -= 1.0 * scale * _dqvalues(0) *
		  phi_i_v * state_fe_face_values.normal_vector(q_point) *
		  state_fe_face_values.JxW(q_point);
	      }

	  }
      }

    if (color == 51)
      {
	const FEValuesExtractors::Vector velocities (0);

	for (unsigned int q_point=0;q_point<n_q_points;q_point++)
	  {
	    for (unsigned int i=0;i<n_dofs_per_cell;i++)
	      {
		const Tensor<1,2> phi_i_v = state_fe_face_values[velocities].value (i, q_point);
		local_cell_vector(i) -= 1.0 * scale * _dqvalues(1) *
		     phi_i_v * state_fe_face_values.normal_vector(q_point) *
		  state_fe_face_values.JxW(q_point);
	      }

	  }
      }
   }


 void BoundaryEquation_QTT (const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
			    dealii::Vector<double> &local_cell_vector,
			    double scale, double /*scale_ico*/)
 {
   const auto & state_fe_face_values = fdc.GetFEFaceValuesState();
   unsigned int n_q_points = fdc.GetNQPoints();
   unsigned int color = fdc.GetBoundaryIndicator();

    assert(this->_problem_type == "hessian");

   _dzboundaryvalues.resize(n_q_points,Vector<double>(5));

   fdc.GetFaceValuesState("adjoint_hessian",_dzboundaryvalues);

   // control values for both parts
   if (color == 50)
     {
       for (unsigned int q_point=0;q_point<n_q_points;q_point++)
	 {
	   Tensor<1,2> dzvboundary;
	   dzvboundary.clear();
	   dzvboundary[0] = _dzboundaryvalues[q_point](0);
	   dzvboundary[1] = _dzboundaryvalues[q_point](1);

	   local_cell_vector(0) -= scale *
	     1.0 * state_fe_face_values.normal_vector(q_point) *
	     dzvboundary *state_fe_face_values.JxW(q_point);

	 }
     }
   if (color == 51)
     {
       for (unsigned int q_point=0;q_point<n_q_points;q_point++)
	 {
	   Tensor<1,2> dzvboundary;
	   dzvboundary.clear();
	   dzvboundary[0] = _dzboundaryvalues[q_point](0);
	   dzvboundary[1] = _dzboundaryvalues[q_point](1);

	   local_cell_vector(1) -= scale *
	     1.0 * state_fe_face_values.normal_vector(q_point) *
	     dzvboundary *state_fe_face_values.JxW(q_point);
	 }
     }
 }

 // do-nothing condition at boundary /Gamma_1
 void BoundaryEquation_U (const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
			  dealii::Vector<double> &local_cell_vector, double scale, double /*scale_ico*/)
 {
   const auto & state_fe_face_values = fdc.GetFEFaceValuesState();
   unsigned int n_dofs_per_cell = fdc.GetNDoFsPerCell();
   unsigned int n_q_points = fdc.GetNQPoints();
   unsigned int color = fdc.GetBoundaryIndicator();
   
   assert(this->_problem_type == "adjoint");

    // do-nothing applied on outflow boundary due symmetric part of
    // fluid's stress tensor
    if (color == 1)
      {
	_zboundaryvalues.resize(n_q_points,Vector<double>(5));

	fdc.GetFaceValuesState("last_newton_solution",_zboundaryvalues);

	const FEValuesExtractors::Vector velocities (0);

	for (unsigned int q_point=0;q_point<n_q_points;q_point++)
	  {
	    Tensor<1,2> zvboundary;
	    zvboundary.clear();
	    zvboundary[0] = _zboundaryvalues[q_point](0);
	    zvboundary[1] = _zboundaryvalues[q_point](1);

	    for (unsigned int i=0;i<n_dofs_per_cell;i++)
	      {
		const Tensor<2,2> phi_i_grads_v = state_fe_face_values[velocities].gradient (i, q_point);

		local_cell_vector(i) -= scale *
		  _density_fluid * _viscosity * transpose(phi_i_grads_v) *
		  state_fe_face_values.normal_vector(q_point) *
		  zvboundary *
		  state_fe_face_values.JxW(q_point);
	      }
	  }
      }
  }


 void BoundaryEquation_UT (const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
			   dealii::Vector<double> &local_cell_vector,
			   double scale, double /*scale_ico*/)
 {
   const auto & state_fe_face_values = fdc.GetFEFaceValuesState();
   unsigned int n_dofs_per_cell = fdc.GetNDoFsPerCell();
   unsigned int n_q_points = fdc.GetNQPoints();
   unsigned int color = fdc.GetBoundaryIndicator();

    assert(this->_problem_type == "tangent");

    // do-nothing applied on outflow boundary due symmetric part of
    // fluid's stress tensor
    if (color == 1)
      {
	_duboundarygrads.resize(n_q_points,vector<Tensor<1,2> >(5));

	fdc.GetFaceGradsState("last_newton_solution",_duboundarygrads);

	const FEValuesExtractors::Vector velocities (0);

	for (unsigned int q_point=0;q_point<n_q_points;q_point++)
	  {
	    Tensor<2,2> duv_grad;
	    duv_grad[0][0] = _duboundarygrads[q_point][0][0];
	    duv_grad[0][1] = _duboundarygrads[q_point][0][1];
	    duv_grad[1][0] = _duboundarygrads[q_point][1][0];
	    duv_grad[1][1] = _duboundarygrads[q_point][1][1];
	    for (unsigned int i=0;i<n_dofs_per_cell;i++)
	      {
		const Tensor<1,2> phi_i_v = state_fe_face_values[velocities].value (i, q_point);

		local_cell_vector(i) -= scale *
		  _density_fluid * _viscosity * transpose(duv_grad) *
		  state_fe_face_values.normal_vector(q_point) *
		  phi_i_v *
		  state_fe_face_values.JxW(q_point);
	      }
	  }
      }
  }

 void BoundaryEquation_UTT (const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
			    dealii::Vector<double> &local_cell_vector,
			    double scale, double /*scale_ico*/)
 {
   const auto & state_fe_face_values = fdc.GetFEFaceValuesState();
   unsigned int n_dofs_per_cell = fdc.GetNDoFsPerCell();
   unsigned int n_q_points = fdc.GetNQPoints();
   unsigned int color = fdc.GetBoundaryIndicator();
   
    assert(this->_problem_type == "adjoint_hessian");

    // do-nothing applied on outflow boundary due symmetric part of
    // fluid's stress tensor
    if (color == 1)
      {
	_dzboundaryvalues.resize(n_q_points,Vector<double>(5));

	fdc.GetFaceValuesState("last_newton_solution",_dzboundaryvalues);


	const FEValuesExtractors::Vector velocities (0);

	for (unsigned int q_point=0;q_point<n_q_points;q_point++)
	  {
	    Tensor<1,2> dzvboundary;
	    dzvboundary.clear();
	    dzvboundary[0] = _dzboundaryvalues[q_point](0);
	    dzvboundary[1] = _dzboundaryvalues[q_point](1);

	    for (unsigned int i=0;i<n_dofs_per_cell;i++)
	      {
		const Tensor<2,2> phi_i_grads_v = state_fe_face_values[velocities].gradient (i, q_point);

		local_cell_vector(i) -= scale *
		  _density_fluid * _viscosity * transpose(phi_i_grads_v) *
		  state_fe_face_values.normal_vector(q_point) *
		  dzvboundary *
		  state_fe_face_values.JxW(q_point);
	      }
	  }
      }
  }

 void BoundaryEquation_UU (const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc __attribute__((unused)),
			   dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/, double /*scale_ico*/)
 {
   assert(this->_problem_type == "adjoint_hessian");
 }
 
 void BoundaryEquation_QU (const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc __attribute__((unused)),
			   dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/, double /*scale_ico*/)
 {
   assert(this->_problem_type == "adjoint_hessian");
 }
 
 void BoundaryEquation_UQ (const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc __attribute__((unused)),
			   dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/, double /*scale_ico*/)
 {
   
 }
 
 void BoundaryEquation_QQ (const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc __attribute__((unused)),
			   dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/, double /*scale_ico*/)
 {
   
 }


 void ControlCellEquation(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc __attribute__((unused)),
   dealii::Vector<double> &local_cell_vector, double scale)
 {
   {
     assert((this->_problem_type == "gradient")||(this->_problem_type == "hessian"));
     _funcgradvalues.reinit(local_cell_vector.size());
     cdc.GetParamValues("last_newton_solution",_funcgradvalues);
   }
   
   for(unsigned int i = 0; i < local_cell_vector.size(); i++)
   {
     local_cell_vector(i) += scale * _funcgradvalues(i);
   }
 }
 
 void ControlCellMatrix(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc __attribute__((unused)),
   FullMatrix<double> &local_entry_matrix)
 {
   assert(local_entry_matrix.m() == local_entry_matrix.n());
   for(unsigned int i = 0; i < local_entry_matrix.m(); i++)
   {
     local_entry_matrix(i,i) += 1.;
   }
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
	return update_values | update_quadrature_points | update_normal_vectors;
      else
	throw DOpEException("Unknown Problem Type "+this->_problem_type ,"LocalPDE::GetUpdateFlags");
    }



    unsigned int GetControlNBlocks() const
    { return 2;}

    unsigned int GetStateNBlocks() const
    {
      return 3;
    }

    std::vector<unsigned int>& GetControlBlockComponent(){ return _control_block_components; }
    const std::vector<unsigned int>& GetControlBlockComponent() const{ return _control_block_components; }
    std::vector<unsigned int>& GetStateBlockComponent(){ return _state_block_components; }
    const std::vector<unsigned int>& GetStateBlockComponent() const{ return _state_block_components; }


  protected:

  private:
     Vector<double> _qvalues;
     Vector<double> _dqvalues;

     Vector<double>  _funcgradvalues;
     vector<Vector<double> > _fvalues;

     vector<Vector<double> > _uvalues;
     vector<vector<Tensor<1,dealdim> > > _ugrads;

     vector<Vector<double> > _zvalues;
     vector<vector<Tensor<1,dealdim> > > _zgrads;
     vector<Vector<double> > _z_state_values;
     vector<vector<Tensor<1,dealdim> > > _z_state_grads;

     vector<Vector<double> > _duvalues;
     vector<vector<Tensor<1,dealdim> > > _dugrads;
     vector<Vector<double> > _du_state_values;
     vector<vector<Tensor<1,dealdim> > > _du_state_grads;

     vector<Vector<double> > _dzvalues;
     vector<vector<Tensor<1,dealdim> > > _dzgrads;
     vector<Vector<double> > _dz_state_values;
     vector<vector<Tensor<1,dealdim> > > _dz_state_grads;

     // boundary values
     vector<Vector<double> > _qboundaryvalues;
     vector<Vector<double> > _fboundaryvalues;
     vector<Vector<double> > _uboundaryvalues;

     vector<Vector<double> > _zboundaryvalues;
     vector<Vector<double> > _dzboundaryvalues;

     vector<vector<Tensor<1,dealdim> > > _uboundarygrads;
     vector<vector<Tensor<1,dealdim> > > _duboundarygrads;


     vector<unsigned int> _control_block_components;
     vector<unsigned int> _state_block_components;

     double _cell_diameter;

     // Fluid- and material variables
     double _density_fluid, _viscosity, _alpha_u, _lame_coefficient_mu,
       _poisson_ratio_nu, _lame_coefficient_lambda;

  };
#endif


