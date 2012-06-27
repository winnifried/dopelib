#ifndef _LOCALPDE_
#define _LOCALPDE_

#include "pdeinterface.h"
#include "ale_transformations.h"

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
      param_reader.declare_entry("density_structure", "0.0",
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
	_state_block_components[2]= 1;  // displacement x
	_state_block_components[3]= 1;  // displacement y
	_state_block_components[4]= 2;  // pressure

	param_reader.SetSubsection("Local PDE parameters");
	_density_fluid = param_reader.get_double ("density_fluid");
	_density_structure = param_reader.get_double ("density_structure");
	_viscosity = param_reader.get_double ("viscosity");
	_alpha_u = param_reader.get_double ("alpha_u");
	
	_lame_coefficient_mu = param_reader.get_double ("mu");
	_poisson_ratio_nu = param_reader.get_double ("poisson_ratio_nu");
	_lame_coefficient_lambda =  (2 * _poisson_ratio_nu * _lame_coefficient_mu)/
	  (1.0 - 2 * _poisson_ratio_nu);	      	
      }

 bool HasFaces() const
  {
    // should be true
    return true;
  }
 
    
  // Domain values for cells     
    void CellEquation(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
		      dealii::Vector<double> &local_cell_vector, double scale, double /*scale_ico*/)
    {
      const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
      unsigned int n_q_points = cdc.GetNQPoints();
      unsigned int material_id = cdc.GetMaterialId(); 
      double cell_diameter = cdc.GetCellDiameter();

      assert(this->_problem_type == "state"); 
    
      _uvalues.resize(n_q_points,Vector<double>(5));
      _ugrads.resize(n_q_points,vector<Tensor<1,2> >(5));

      // Getting state values
      cdc.GetValuesState("last_newton_solution",_uvalues);
      cdc.GetGradsState("last_newton_solution",_ugrads);
      
      
      const FEValuesExtractors::Vector velocities (0);
      const FEValuesExtractors::Vector displacements (2);
      const FEValuesExtractors::Scalar pressure (4);
	 
      const Tensor<2,dealdim> Identity = ALE_Transformations
	::get_Identity<dealdim> ();

      // fluid
      if (material_id == 0)
	{
	  
	  for(unsigned int q_point = 0; q_point < n_q_points; q_point++)
	    {
	      const Tensor<2,dealdim> pI = ALE_Transformations
		::get_pI<dealdim> (q_point, _uvalues);

	      const Tensor<1,dealdim> v = ALE_Transformations
		::get_v<dealdim> (q_point, _uvalues);
	      
	      const Tensor<2,dealdim> grad_v = ALE_Transformations 
		::get_grad_v<dealdim> (q_point, _ugrads);
	      	      
	      const Tensor<2,dealdim> grad_v_T = ALE_Transformations
		::get_grad_v_T<dealdim> (grad_v);
	      
	      const Tensor<1,dealdim> u = ALE_Transformations
		::get_u<dealdim> (q_point, _uvalues); 

	      const Tensor<2,dealdim> grad_u = ALE_Transformations 
		::get_grad_u<dealdim> (q_point, _ugrads);
	      
	      const Tensor<2,dealdim> F = ALE_Transformations
		::get_F<dealdim> (q_point, _ugrads);	       	     
	      
	      const Tensor<2,dealdim> F_Inverse = ALE_Transformations
		::get_F_Inverse<dealdim> (F);
	      
	      const Tensor<2,dealdim> F_Inverse_T = ALE_Transformations
		::get_F_Inverse_T<dealdim> (F_Inverse);
	      
	      const double J = ALE_Transformations
		::get_J<dealdim> (F);


	      const Tensor<2,dealdim> sigma_ALE = NSE_in_ALE
		::get_stress_fluid_except_pressure_ALE<dealdim> (_density_fluid, _viscosity, 
							     grad_v, grad_v_T, F_Inverse, F_Inverse_T );
	      
	      const Tensor<2,dealdim> stress_fluid = (J * sigma_ALE * F_Inverse_T);
	      
	      const Tensor<1,dealdim> convection_fluid = _density_fluid * J * (grad_v * F_Inverse * v);
	      	          	      	    
	      const Tensor<2,dealdim> fluid_pressure = (-pI * J * F_Inverse_T); 
	      
	      const double incompressiblity_fluid = NSE_in_ALE
		::get_Incompressibility_ALE<dealdim> (q_point, _ugrads);

	      
	      for(unsigned int i = 0; i < n_dofs_per_cell; i++)
		{
		  const Tensor<1,2> phi_i_v = state_fe_values[velocities].value (i, q_point);
		  const Tensor<2,2> phi_i_grads_v = state_fe_values[velocities].gradient (i, q_point);
		  const double phi_i_p = state_fe_values[pressure].value (i, q_point);
		  const Tensor<2,2> phi_i_grads_u = state_fe_values[displacements].gradient (i, q_point);
		  
		  local_cell_vector(i) += scale * 
		    (
		     convection_fluid * phi_i_v  
		     + scalar_product(fluid_pressure, phi_i_grads_v)	
		     + scalar_product(stress_fluid, phi_i_grads_v)
		     + incompressiblity_fluid * phi_i_p 
		     + _alpha_u * cell_diameter * cell_diameter * scalar_product(grad_u, phi_i_grads_u)
		     )
		    * state_fe_values.JxW(q_point);
		}
	    }
	  


	}  // end material_id == 0
      else if (material_id == 1)
	{
	  // structure, STVK
	  for(unsigned int q_point = 0; q_point < n_q_points; q_point++)
	    {
	      const Tensor<1,2> v = ALE_Transformations
		::get_v<2> (q_point, _uvalues);

	      const Tensor<2,2> grad_v = ALE_Transformations
		::get_grad_v<2> (q_point, _ugrads);
	      	      
	      const Tensor<1,2> u = ALE_Transformations
		::get_u<2> (q_point, _uvalues);
	      
	      const Tensor<2,2> F = ALE_Transformations
		::get_F<2> (q_point, _ugrads);
	      
	      const Tensor<2,2> F_T = ALE_Transformations
		::get_F_T<2> (F);
	      	  
	      const Tensor<2,2> F_Inverse = ALE_Transformations
		::get_F_Inverse<2> (F);
	      
	      const Tensor<2,2> F_Inverse_T = ALE_Transformations
		::get_F_Inverse_T<2> (F_Inverse);
	      
	      const double J = ALE_Transformations
		::get_J<2> (F);
	      
	      const Tensor<2,2> E = Structure_Terms_in_ALE
		::get_E<2> (F_T, F, Identity);
	      
	      const double tr_E = Structure_Terms_in_ALE
		::get_tr_E<2> (E);


	      // STVK
	      const Tensor<2,2> sigma_structure_ALE = 
		(F *(_lame_coefficient_lambda * tr_E * Identity + 2 * _lame_coefficient_mu * E)); 
							    
							  
							   
						      
	      
	      const Tensor<2,2> stress_term  = (J * sigma_structure_ALE * F_Inverse_T);
 
	      for(unsigned int i = 0; i < n_dofs_per_cell; i++)
		{
		  const Tensor<1,2> phi_i_v = state_fe_values[velocities].value (i, q_point);
		  const Tensor<2,2> phi_i_grads_v = state_fe_values[velocities].gradient (i, q_point);
		  const double phi_i_p = state_fe_values[pressure].value (i, q_point);
		  const Tensor<1,2> phi_i_u = state_fe_values[displacements].value (i, q_point);
		  const Tensor<2,2> phi_i_grads_u = state_fe_values[displacements].gradient (i, q_point);
		  
		  local_cell_vector(i) += scale * 
		    (scalar_product(sigma_structure_ALE,phi_i_grads_v)
		     - _density_structure * v * phi_i_u
		     +_uvalues[q_point](4) * phi_i_p		     
		     )
		    * state_fe_values.JxW(q_point);
		}
	    }
	}  // end material_id == 1
      
    }
    
    
    void CellMatrix (const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
		     dealii::FullMatrix<double> &local_entry_matrix, double scale, double /*scale_ico*/)
    { 
      const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
      unsigned int n_q_points = cdc.GetNQPoints();
      unsigned int material_id = cdc.GetMaterialId(); 
      double cell_diameter = cdc.GetCellDiameter();
  
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
      const FEValuesExtractors::Vector displacements (2);
      const FEValuesExtractors::Scalar pressure (4);

      std::vector<Tensor<1,2> >     phi_v (n_dofs_per_cell);  
      std::vector<Tensor<2,2> >     phi_grads_v (n_dofs_per_cell);  
      std::vector<Tensor<1,2> >     phi_u (n_dofs_per_cell);  
      std::vector<Tensor<2,2> >     phi_grads_u (n_dofs_per_cell);  
      std::vector<double>           phi_p (n_dofs_per_cell);   
    
      const Tensor<2,dealdim> Identity = ALE_Transformations
	::get_Identity<dealdim> ();

	
      if (material_id == 0)
	{

	  
	  for(unsigned int q_point = 0; q_point < n_q_points; q_point++)
	    {
	       for(unsigned int k = 0; k < n_dofs_per_cell; k++)
		{	
		  phi_p[k]       = state_fe_values[pressure].value (k, q_point);
		  phi_v[k]       = state_fe_values[velocities].value (k, q_point);
		  phi_grads_v[k] = state_fe_values[velocities].gradient (k, q_point);			      			 
		  phi_u[k]       = state_fe_values[displacements].value (k, q_point);
		  phi_grads_u[k] = state_fe_values[displacements].gradient (k, q_point);		
		}
	      
	         const Tensor<2,dealdim> pI = ALE_Transformations::get_pI<dealdim> (q_point, _uvalues);
	      const Tensor<1,dealdim> v = ALE_Transformations::get_v<dealdim> (q_point, _uvalues);
	   
	      
	      const Tensor<1,dealdim> u = ALE_Transformations::get_u<dealdim> (q_point,_uvalues);
	   
	      const Tensor<2,dealdim> grad_v = ALE_Transformations::get_grad_v<dealdim> (q_point, _ugrads);	
	      
	      const Tensor<2,dealdim> grad_v_T = ALE_Transformations::get_grad_v_T<dealdim> (grad_v);		
	      const Tensor<2,dealdim> F = ALE_Transformations::get_F<dealdim> (q_point, _ugrads);
	      const Tensor<2,dealdim> F_Inverse = ALE_Transformations::get_F_Inverse<dealdim> (F);
	      
	      const Tensor<2,dealdim> F_Inverse_T = ALE_Transformations::get_F_Inverse_T<dealdim> (F_Inverse);
	      const double J = ALE_Transformations::get_J<dealdim> (F);

	      const Tensor<2,dealdim> sigma_ALE = NSE_in_ALE
		::get_stress_fluid_ALE<dealdim> (_density_fluid, _viscosity, pI,
						 grad_v, grad_v_T, F_Inverse, F_Inverse_T );

	      for(unsigned int j = 0; j < n_dofs_per_cell; j++)
		{
		   const Tensor<2,dealdim> pI_LinP = ALE_Transformations::get_pI_LinP<dealdim> (phi_p[j]);
		  const Tensor<2,dealdim> grad_v_LinV = ALE_Transformations::get_grad_v_LinV<dealdim> (phi_grads_v[j]);
		  const double J_LinU =  ALE_Transformations::get_J_LinU<dealdim> (q_point, _ugrads, phi_grads_u[j]);
		  
		  const double J_Inverse_LinU = ALE_Transformations::get_J_Inverse_LinU<dealdim> (J, J_LinU);
		  const Tensor<2,dealdim> J_F_Inverse_T_LinU = ALE_Transformations::get_J_F_Inverse_T_LinU<dealdim> (phi_grads_u[j]);
		  const Tensor<2,dealdim> F_Inverse_LinU = ALE_Transformations::get_F_Inverse_LinU (phi_grads_u[j],J,
												J_LinU,q_point,_ugrads
												);
		  
		  const Tensor<2,dealdim>  stress_fluid_ALE_1st_term_LinAll =  NSE_in_ALE			
		    ::get_stress_fluid_ALE_1st_term_LinAll_short<dealdim> (pI, F_Inverse_T,J_F_Inverse_T_LinU,	
									   pI_LinP,J);
	
		  const double incompressibility_ALE_LinAll = NSE_in_ALE
		    ::get_Incompressibility_ALE_LinAll<dealdim> (phi_grads_v[j],phi_grads_u[j], q_point,
							     _ugrads); 
	
		  const Tensor<2,dealdim> stress_fluid_ALE_2nd_term_LinAll = NSE_in_ALE
		    ::get_stress_fluid_ALE_2nd_term_LinAll_short (J_F_Inverse_T_LinU,sigma_ALE,						      
								  grad_v, grad_v_LinV, F_Inverse,	       
								  F_Inverse_LinU, J, _viscosity, _density_fluid );     
		  
		  const Tensor<1,dealdim> convection_fluid_LinAll_short = NSE_in_ALE
		    ::get_Convection_LinAll_short<dealdim> (phi_grads_v[j],phi_v[j], J,J_LinU,
							    F_Inverse, F_Inverse_LinU,v, grad_v, _density_fluid);	
		  	
		  
		  for(unsigned int i = 0; i < n_dofs_per_cell; i++)
		    {
	     
		      local_entry_matrix(i,j) += 
			(
			 convection_fluid_LinAll_short * phi_v[i] 					      	      
			 + scalar_product(stress_fluid_ALE_2nd_term_LinAll, phi_grads_v[i])
			 + scalar_product(stress_fluid_ALE_1st_term_LinAll, phi_grads_v[i])
			 + incompressibility_ALE_LinAll *  phi_p[i]
			 + _alpha_u * cell_diameter * cell_diameter 						   			 
			 * scalar_product(phi_grads_u[j], phi_grads_u[i])
			 )
			* state_fe_values.JxW(q_point);
	
		    }
		}
	    }
	  



	} // end material_id ==0
      else if (material_id == 1)
	{
	  // structure, STVK
	  for(unsigned int q_point = 0; q_point < n_q_points; q_point++)
	    {
	      for(unsigned int k = 0; k < n_dofs_per_cell; k++)
		{	
		  phi_p[k]       = state_fe_values[pressure].value (k, q_point);
		  phi_v[k]       = state_fe_values[velocities].value (k, q_point);
		  phi_grads_v[k] = state_fe_values[velocities].gradient (k, q_point);			      			 
		  phi_u[k]       = state_fe_values[displacements].value (k, q_point);
		  phi_grads_u[k] = state_fe_values[displacements].gradient (k, q_point);		
		}
	      
	      const Tensor<2,dealdim> F = ALE_Transformations
		::get_F<dealdim> (q_point, _ugrads);
	     
	      const Tensor<2,dealdim> F_Inverse = ALE_Transformations
		::get_F_Inverse<dealdim> (F);
	      
	      const Tensor<2,dealdim> F_Inverse_T = ALE_Transformations
		::get_F_Inverse_T<dealdim> (F_Inverse);
	      
	      const Tensor<2,dealdim> F_T = ALE_Transformations
		::get_F_T<dealdim> (F);
	      
	      const double J = ALE_Transformations
		::get_J<dealdim> (F);
	      	     
	      const Tensor<2,dealdim> E = Structure_Terms_in_ALE 
		::get_E<dealdim> (F_T, F, Identity);
	      
	      const double tr_E = Structure_Terms_in_ALE
		::get_tr_E<dealdim> (E);


	      for(unsigned int j = 0; j < n_dofs_per_cell; j++)
		{

		  const double J_LinU = ALE_Transformations
		    ::get_J_LinU<dealdim> (q_point, _ugrads,
				       phi_grads_u[j]);
		  
		  const Tensor<2,dealdim> F_LinU = ALE_Transformations		  
		    ::get_F_LinU<dealdim> (phi_grads_u[j]);
		  
		  const Tensor<2,dealdim> F_Inverse_LinU = ALE_Transformations
		    ::get_F_Inverse_LinU<dealdim> (phi_grads_u[j],
					       J, J_LinU, q_point,
					       _ugrads);
		  
		  const Tensor<2,dealdim> F_Inverse_T_LinU = transpose(F_Inverse_LinU);
		  
		  const Tensor<2,dealdim> J_F_Inverse_T_LinU = ALE_Transformations
		    ::get_J_F_Inverse_T_LinU<dealdim> (phi_grads_u[j]);
		  
		  // STVK: Green_Lagrange strain tensor derivatives
		  const Tensor<2,dealdim> E_LinU = 0.5 * (transpose(F_LinU) * F + transpose(F) * F_LinU);
		  
		  const double tr_E_LinU = Structure_Terms_in_ALE
		    ::get_tr_E_LinU<dealdim> (q_point,_ugrads, phi_grads_u[j]);
		  
		  
		  // STVK
		  // piola-kirchhoff stress structure STVK linearized in all directions 
		  // J * (1/J*F*(lambda*tr_E*I + 2*mu*E)*F^T) * F^{-T} --> Linearization
		  Tensor<2,dealdim> piola_kirchhoff_stress_structure_STVK_LinALL;
		  piola_kirchhoff_stress_structure_STVK_LinALL = _lame_coefficient_lambda * 
		    (F_LinU * tr_E * Identity + F * tr_E_LinU * Identity) 
		    + 2 * _lame_coefficient_mu * (F_LinU * E + F * E_LinU);
		  
		  for(unsigned int i = 0; i < n_dofs_per_cell; i++)
		    {	      
		      local_entry_matrix(i,j) += 
			(scalar_product(piola_kirchhoff_stress_structure_STVK_LinALL,phi_grads_v[i])
			 - _density_structure * phi_v[j] * phi_u[i]
			 + phi_p[j] * phi_p[i]
			 )
			* state_fe_values.JxW(q_point);
		    }
		}
	    }
	} // end material_id ==1
      
      
    }
 
    
    void CellEquation_U(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
			dealii::Vector<double> &local_cell_vector, double scale, double /*scale_ico*/)
    {
      const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
      unsigned int n_q_points = cdc.GetNQPoints();
      unsigned int material_id = cdc.GetMaterialId(); 
      double cell_diameter = cdc.GetCellDiameter();
 
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
	const FEValuesExtractors::Vector displacements (2);
       	const FEValuesExtractors::Scalar pressure (4);

	std::vector<Tensor<1,2> >     phi_v (n_dofs_per_cell);  
	std::vector<Tensor<2,2> >     phi_grads_v (n_dofs_per_cell);  
	std::vector<Tensor<1,2> >     phi_u (n_dofs_per_cell);  
	std::vector<Tensor<2,2> >     phi_grads_u (n_dofs_per_cell);  
	std::vector<double>           phi_p (n_dofs_per_cell);   
	
	const Tensor<2,dealdim> Identity = ALE_Transformations
	  ::get_Identity<dealdim> ();

	
	// fluid fsi
	if (material_id == 0)
	  {
	    
	    for(unsigned int q_point = 0; q_point < n_q_points; q_point++)
	      {
		for(unsigned int k = 0; k < n_dofs_per_cell; k++)
		  {	
		    phi_p[k]       = state_fe_values[pressure].value (k, q_point);
		    phi_v[k]       = state_fe_values[velocities].value (k, q_point);
		    phi_grads_v[k] = state_fe_values[velocities].gradient (k, q_point);			      			 
		    phi_u[k]       = state_fe_values[displacements].value (k, q_point);
		    phi_grads_u[k] = state_fe_values[displacements].gradient (k, q_point);		
		  }

		// adjoint values and grads			
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
		
		double zp = _zvalues[q_point](4);

		Tensor<2,2> zu_grads;
		zu_grads.clear();
		zu_grads[0][0] = _zgrads[q_point][2][0];
		zu_grads[0][1] = _zgrads[q_point][2][1];
		zu_grads[1][0] = _zgrads[q_point][3][0];
		zu_grads[1][1] = _zgrads[q_point][3][1];
	
	
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
				
		Tensor<2,2> zpI_state;
		zpI_state.clear();
		zpI_state[0][0] = _z_state_values[q_point](4);
		zpI_state[0][1] = 0.0;
		zpI_state[1][0] = 0.0;
		zpI_state[1][1] = _z_state_values[q_point](4);

		
		// state values and grads
		const Tensor<2,dealdim> F = ALE_Transformations
		  ::get_F<dealdim> (q_point, _z_state_grads);
		
		const Tensor<2,dealdim> F_Inverse = ALE_Transformations
		  ::get_F_Inverse<dealdim> (F);
		
		const Tensor<2,dealdim> F_Inverse_T = ALE_Transformations
		  ::get_F_Inverse_T<dealdim> (F_Inverse);
		
		const Tensor<2,dealdim> F_T = ALE_Transformations
		  ::get_F_T<dealdim> (F);
		
		const double J = ALE_Transformations
		  ::get_J<dealdim> (F);
		
		const Tensor<2,dealdim> sigma_ALE = NSE_in_ALE
		  ::get_stress_fluid_ALE<dealdim> (_density_fluid, _viscosity, zpI_state,
						   zv_state_grads, transpose(zv_state_grads), F_Inverse, F_Inverse_T );
		
		for(unsigned int j = 0; j < n_dofs_per_cell; j++)
		  {	
		    const Tensor<2,dealdim> pI_LinP = ALE_Transformations
		      ::get_pI_LinP<dealdim> (phi_p[j]);
		  
		    const Tensor<2,dealdim> grad_v_LinV = ALE_Transformations
		      ::get_grad_v_LinV<dealdim> (phi_grads_v[j]);
		  
		    const double J_LinU =  ALE_Transformations
		      ::get_J_LinU<dealdim> (q_point, _z_state_grads, phi_grads_u[j]);
		  
		    const double J_Inverse_LinU = ALE_Transformations
		      ::get_J_Inverse_LinU<dealdim> (J, J_LinU);

		    const Tensor<2,dealdim> J_F_Inverse_T_LinU = ALE_Transformations
		      ::get_J_F_Inverse_T_LinU<dealdim> (phi_grads_u[j]);
		  
		    const Tensor<2,dealdim> F_Inverse_LinU = ALE_Transformations
		      ::get_F_Inverse_LinU (phi_grads_u[j],J, J_LinU,q_point,_z_state_grads);
					  
	    
		    // four main equations
		    const Tensor<2,dealdim>  stress_fluid_ALE_1st_term_LinAll =  NSE_in_ALE			
		      ::get_stress_fluid_ALE_1st_term_LinAll_short<dealdim> (zpI_state, F_Inverse_T,J_F_Inverse_T_LinU,	
									     pI_LinP,J);
		    
		    const double incompressibility_ALE_LinAll = NSE_in_ALE
		      ::get_Incompressibility_ALE_LinAll<dealdim> (phi_grads_v[j],phi_grads_u[j], q_point,
								   _z_state_grads); 
		    
		    const Tensor<2,dealdim> stress_fluid_ALE_2nd_term_LinAll = NSE_in_ALE
		      ::get_stress_fluid_ALE_2nd_term_LinAll_short (J_F_Inverse_T_LinU, sigma_ALE,						      
								    zv_state_grads, grad_v_LinV, F_Inverse,	       
								    F_Inverse_LinU, J, _viscosity, _density_fluid );     
		    
		    const Tensor<1,dealdim> convection_fluid_LinAll_short = NSE_in_ALE
		      ::get_Convection_LinAll_short<dealdim> (phi_grads_v[j],phi_v[j], J,J_LinU,
							      F_Inverse, F_Inverse_LinU, zv_state, zv_state_grads, _density_fluid);

		    
		    local_cell_vector(j) +=  scale * 
		      (
		       convection_fluid_LinAll_short * zv
		       + scalar_product(stress_fluid_ALE_2nd_term_LinAll, zv_grads)
		       + scalar_product(stress_fluid_ALE_1st_term_LinAll, zv_grads) 
		       + incompressibility_ALE_LinAll *  zp
		       + _alpha_u * cell_diameter * cell_diameter
		       * scalar_product(phi_grads_u[j], zu_grads)
		       )
		      * state_fe_values.JxW(q_point);
		    
		  }
	      }
	    
 
	  }  // end material_id == 0
	else if (material_id == 1)
	  {
	    
	    for(unsigned int q_point = 0; q_point < n_q_points; q_point++)
	      {
		for(unsigned int k = 0; k < n_dofs_per_cell; k++)
		  {	
		    phi_p[k]       = state_fe_values[pressure].value (k, q_point);
		    phi_v[k]       = state_fe_values[velocities].value (k, q_point);
		    phi_grads_v[k] = state_fe_values[velocities].gradient (k, q_point);			      			 
		    phi_u[k]       = state_fe_values[displacements].value (k, q_point);
		    phi_grads_u[k] = state_fe_values[displacements].gradient (k, q_point);		
		  }
	      
		// adjoint values and grads
		Tensor<2,2> zv_grads;
		zv_grads.clear();
		zv_grads[0][0] = _zgrads[q_point][0][0];
		zv_grads[0][1] = _zgrads[q_point][0][1];
		zv_grads[1][0] = _zgrads[q_point][1][0];
		zv_grads[1][1] = _zgrads[q_point][1][1];
		
	
		Tensor<1,2> zu;
		zu.clear();
		zu[0] = _zvalues[q_point](2);
		zu[1] = _zvalues[q_point](3);
      
		double zp = _zvalues[q_point](4);
	

		// state values and grads
		const Tensor<2,dealdim> F = ALE_Transformations
		::get_F<dealdim> (q_point, _z_state_grads);
	     
	      const Tensor<2,dealdim> F_Inverse = ALE_Transformations
		::get_F_Inverse<dealdim> (F);
	      
	      const Tensor<2,dealdim> F_Inverse_T = ALE_Transformations
		::get_F_Inverse_T<dealdim> (F_Inverse);
	      
	      const Tensor<2,dealdim> F_T = ALE_Transformations
		::get_F_T<dealdim> (F);
	      
	      const double J = ALE_Transformations
		::get_J<dealdim> (F);
	      	     
	      const Tensor<2,dealdim> E = Structure_Terms_in_ALE 
		::get_E<dealdim> (F_T, F, Identity);
	      
	      const double tr_E = Structure_Terms_in_ALE
		::get_tr_E<dealdim> (E);


		for(unsigned int j = 0; j < n_dofs_per_cell; j++)
		  {
		 
		    //_z_state_values
		    //_z_state_grads
		    const double J_LinU = ALE_Transformations
		      ::get_J_LinU<dealdim> (q_point, _z_state_grads,
					     phi_grads_u[j]);
		    
		    const Tensor<2,dealdim> F_LinU = ALE_Transformations		  
		      ::get_F_LinU<dealdim> (phi_grads_u[j]);
		    
		    const Tensor<2,dealdim> F_Inverse_LinU = ALE_Transformations
		      ::get_F_Inverse_LinU<dealdim> (phi_grads_u[j],
						     J, J_LinU, q_point,
						     _z_state_grads);
		    
		    const Tensor<2,dealdim> F_Inverse_T_LinU = transpose(F_Inverse_LinU);
		    
		    const Tensor<2,dealdim> J_F_Inverse_T_LinU = ALE_Transformations
		      ::get_J_F_Inverse_T_LinU<dealdim> (phi_grads_u[j]);
		    
		    // STVK: Green_Lagrange strain tensor derivatives
		    const Tensor<2,dealdim> E_LinU = 0.5 * (transpose(F_LinU) * F + transpose(F) * F_LinU);
		    
		    const double tr_E_LinU = Structure_Terms_in_ALE
		      ::get_tr_E_LinU<dealdim> (q_point,_z_state_grads, phi_grads_u[j]);
		    

		    // STVK
		    // piola-kirchhoff stress structure STVK linearized in all directions 
		    // J * (1/J*F*(lambda*tr_E*I + 2*mu*E)*F^T) * F^{-T} --> Linearization
		    Tensor<2,dealdim> piola_kirchhoff_stress_structure_STVK_LinALL;
		    piola_kirchhoff_stress_structure_STVK_LinALL = _lame_coefficient_lambda * 
		      (F_LinU * tr_E * Identity + F * tr_E_LinU * Identity) 
		      + 2 * _lame_coefficient_mu * (F_LinU * E + F * E_LinU);

		    
		    local_cell_vector(j) +=  scale * 
		      (scalar_product(piola_kirchhoff_stress_structure_STVK_LinALL,zv_grads)				      
		       - _density_structure * phi_v[j] * zu
		       + phi_p[j] * zp	      
		       )
		      * state_fe_values.JxW(q_point);
		    
		  }
	      }
	  }  // end material_id == 1
	
    }
    
    
 void CellEquation_UT(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
			 dealii::Vector<double> &local_cell_vector, double scale, double /*scale_ico*/)
    {
      const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
      unsigned int n_q_points = cdc.GetNQPoints();
      unsigned int material_id = cdc.GetMaterialId(); 
      double cell_diameter = cdc.GetCellDiameter();

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
      const FEValuesExtractors::Vector displacements (2);
      const FEValuesExtractors::Scalar pressure (4);

      const Tensor<2,dealdim> Identity = ALE_Transformations
	::get_Identity<dealdim> ();



      if (material_id == 0)
	{
	  
	  for(unsigned int q_point = 0; q_point < n_q_points; q_point++)
	    {
	      Tensor<2,dealdim> du_pI;
	      du_pI.clear();
	      du_pI[0][0] =  -_duvalues[q_point](4);
	      du_pI[1][1] =  -_duvalues[q_point](4);
	      
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
	      
	      double dup = _duvalues[q_point](4);
	      double duv_incompressibility = duv_grads[0][0] +  duv_grads[1][1];

	      Tensor<2,2> dupI;
	      dupI.clear();
	      dupI[0][0] = _duvalues[q_point](4);
	      dupI[0][1] = 0.0;
	      dupI[1][0] = 0.0;
	      dupI[1][1] = _duvalues[q_point](4);

	      
	      Tensor<2,2> duu_grads;
	      duu_grads.clear();
	      duu_grads[0][0] = _dugrads[q_point][2][0];
	      duu_grads[0][1] = _dugrads[q_point][2][1];
	      duu_grads[1][0] = _dugrads[q_point][3][0];
	      duu_grads[1][1] = _dugrads[q_point][3][1];
	      
	      // state values which contains 
	      // solution from previous Newton step
	      // Necessary for fluid convection term
	      Tensor<2,2> dupI_state;
	      dupI_state.clear();
	      dupI_state[0][0] = _du_state_values[q_point](4);
	      dupI_state[0][1] = 0.0;
	      dupI_state[1][0] = 0.0;
	      dupI_state[1][1] = _du_state_values[q_point](4);

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
	      

	       // get state values
	      const Tensor<2,dealdim> F = ALE_Transformations
		::get_F<dealdim> (q_point, _du_state_grads);
	     
	      const Tensor<2,dealdim> F_Inverse = ALE_Transformations
		::get_F_Inverse<dealdim> (F);
	      
	      const Tensor<2,dealdim> F_Inverse_T = ALE_Transformations
		::get_F_Inverse_T<dealdim> (F_Inverse);
	      
	      const Tensor<2,dealdim> F_T = ALE_Transformations
		::get_F_T<dealdim> (F);
	      
	      const double J = ALE_Transformations
		::get_J<dealdim> (F);
	      	     
	      const Tensor<2,dealdim> E = Structure_Terms_in_ALE 
		::get_E<dealdim> (F_T, F, Identity);
	      
	      const double tr_E = Structure_Terms_in_ALE
		::get_tr_E<dealdim> (E);

	      // sigma in ALE for fluid
	      const Tensor<2,dealdim> sigma_ALE = NSE_in_ALE
		::get_stress_fluid_ALE<dealdim> (_density_fluid, _viscosity, dupI_state,
						 duv_state_grads, transpose(duv_state_grads), F_Inverse, F_Inverse_T );


	      // linearizations
	      const double J_LinU = ALE_Transformations
		::get_J_LinU<dealdim> (q_point, _du_state_grads,
				       duu_grads);
		  
	      const Tensor<2,dealdim> F_LinU = ALE_Transformations		  
		::get_F_LinU<dealdim> (duu_grads);
	      
	      const Tensor<2,dealdim> F_Inverse_LinU = ALE_Transformations
		::get_F_Inverse_LinU<dealdim> (duu_grads,
					       J, J_LinU, q_point,
					       _du_state_grads);
	      
	      const Tensor<2,dealdim> F_Inverse_T_LinU = transpose(F_Inverse_LinU);
	      
	      const Tensor<2,dealdim> J_F_Inverse_T_LinU = ALE_Transformations
		::get_J_F_Inverse_T_LinU<dealdim> (duu_grads);
	      

	      // four main equations
	      const Tensor<2,dealdim>  stress_fluid_ALE_1st_term_LinAll =  NSE_in_ALE			
		::get_stress_fluid_ALE_1st_term_LinAll_short<dealdim> (dupI_state, F_Inverse_T,J_F_Inverse_T_LinU,	
								       dupI,J);
	      
	      const double incompressibility_ALE_LinAll = NSE_in_ALE
		::get_Incompressibility_ALE_LinAll<dealdim> (duv_grads, duu_grads, q_point,
							     _du_state_grads); 
	      
	      const Tensor<2,dealdim> stress_fluid_ALE_2nd_term_LinAll = NSE_in_ALE
		::get_stress_fluid_ALE_2nd_term_LinAll_short (J_F_Inverse_T_LinU, sigma_ALE,						      
							      duv_state_grads, duv_grads, F_Inverse,	       
							      F_Inverse_LinU, J, _viscosity, _density_fluid );     
	      
	      const Tensor<1,dealdim> convection_fluid_LinAll_short = NSE_in_ALE
		::get_Convection_LinAll_short<dealdim> (duv_grads, duv, J,J_LinU,
							F_Inverse, F_Inverse_LinU, duv_state, duv_state_grads, _density_fluid);
	      
		   


	      for(unsigned int i = 0; i < n_dofs_per_cell; i++)
		{
		  const Tensor<1,2> phi_i_v = state_fe_values[velocities].value (i, q_point);
		  const Tensor<2,2> phi_i_grads_v = state_fe_values[velocities].gradient (i, q_point);
		  const double phi_i_p = state_fe_values[pressure].value (i, q_point);
		  const Tensor<2,2> phi_i_grads_u = state_fe_values[displacements].gradient (i, q_point);
		  
		    local_cell_vector(i) += scale * 
		    (
		     convection_fluid_LinAll_short * phi_i_v
		     + scalar_product(stress_fluid_ALE_2nd_term_LinAll, phi_i_grads_v)
		     + scalar_product(stress_fluid_ALE_1st_term_LinAll, phi_i_grads_v) 
		     + incompressibility_ALE_LinAll * phi_i_p 
		     + _alpha_u * cell_diameter * cell_diameter 
		     * scalar_product(duu_grads, phi_i_grads_u)
		     )
		    * state_fe_values.JxW(q_point);


		}
	    }
	  
	} // material_id ==0
      else if (material_id == 1)
	{
	  
	  for(unsigned int q_point = 0; q_point < n_q_points; q_point++)
	    {
	      Tensor<1,2> duv;
	      duv.clear();
	      duv[0] = _duvalues[q_point](0);
	      duv[1] = _duvalues[q_point](1);

	      
	      
	     
	      double dup = _duvalues[q_point](4);
	      	      
	      Tensor<2,2> duu_grads;
	      duu_grads.clear();
	      duu_grads[0][0] = _dugrads[q_point][2][0];
	      duu_grads[0][1] = _dugrads[q_point][2][1];
	      duu_grads[1][0] = _dugrads[q_point][3][0];
	      duu_grads[1][1] = _dugrads[q_point][3][1];
	    

	      // get state values
	      const Tensor<2,dealdim> F = ALE_Transformations
		::get_F<dealdim> (q_point, _du_state_grads);
	     
	      const Tensor<2,dealdim> F_Inverse = ALE_Transformations
		::get_F_Inverse<dealdim> (F);
	      
	      const Tensor<2,dealdim> F_Inverse_T = ALE_Transformations
		::get_F_Inverse_T<dealdim> (F_Inverse);
	      
	      const Tensor<2,dealdim> F_T = ALE_Transformations
		::get_F_T<dealdim> (F);
	      
	      const double J = ALE_Transformations
		::get_J<dealdim> (F);
	      	     
	      const Tensor<2,dealdim> E = Structure_Terms_in_ALE 
		::get_E<dealdim> (F_T, F, Identity);
	      
	      const double tr_E = Structure_Terms_in_ALE
		::get_tr_E<dealdim> (E);



	      // linearizations
	      const double J_LinU = ALE_Transformations
		    ::get_J_LinU<dealdim> (q_point, _du_state_grads,
				       duu_grads);
		  
		  const Tensor<2,dealdim> F_LinU = ALE_Transformations		  
		    ::get_F_LinU<dealdim> (duu_grads);
		  
		  const Tensor<2,dealdim> F_Inverse_LinU = ALE_Transformations
		    ::get_F_Inverse_LinU<dealdim> (duu_grads,
					       J, J_LinU, q_point,
					       _du_state_grads);
		  
		  const Tensor<2,dealdim> F_Inverse_T_LinU = transpose(F_Inverse_LinU);
		  
		  const Tensor<2,dealdim> J_F_Inverse_T_LinU = ALE_Transformations
		    ::get_J_F_Inverse_T_LinU<dealdim> (duu_grads);
		  
		  // STVK: Green_Lagrange strain tensor derivatives
		  const Tensor<2,dealdim> E_LinU = 0.5 * (transpose(F_LinU) * F + transpose(F) * F_LinU);
		  
		  const double tr_E_LinU = Structure_Terms_in_ALE
		    ::get_tr_E_LinU<dealdim> (q_point,_du_state_grads, duu_grads);




	      // STVK
		  // piola-kirchhoff stress structure STVK linearized in all directions 
		  // J * (1/J*F*(lambda*tr_E*I + 2*mu*E)*F^T) * F^{-T} --> Linearization
		  Tensor<2,dealdim> piola_kirchhoff_stress_structure_STVK_LinALL;
		  piola_kirchhoff_stress_structure_STVK_LinALL = _lame_coefficient_lambda * 
		    (F_LinU * tr_E * Identity + F * tr_E_LinU * Identity) 
		    + 2 * _lame_coefficient_mu * (F_LinU * E + F * E_LinU);


	      for(unsigned int i = 0; i < n_dofs_per_cell; i++)
		{
		  const Tensor<2,2> phi_i_grads_v = state_fe_values[velocities].gradient (i, q_point);
		  const double phi_i_p = state_fe_values[pressure].value (i, q_point);

		  const Tensor<1,2> phi_i_u = state_fe_values[displacements].value (i, q_point);		  
		  		  
		  local_cell_vector(i) += scale * 
		    (scalar_product(piola_kirchhoff_stress_structure_STVK_LinALL,
				      phi_i_grads_v)
		     - _density_structure * duv * phi_i_u 	
		     + dup * phi_i_p		     
		     )
		    * state_fe_values.JxW(q_point);
		}
	    }
	  
	} // material_id ==1
      
    }
 
 
 void CellEquation_UTT(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
			  dealii::Vector<double> &local_cell_vector, double scale, double /*scale_ico*/)
 {  
      const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
      unsigned int n_q_points = cdc.GetNQPoints();
      unsigned int material_id = cdc.GetMaterialId(); 
      double cell_diameter = cdc.GetCellDiameter();
 
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
      const FEValuesExtractors::Vector displacements (2);
      const FEValuesExtractors::Scalar pressure (4);
      
      std::vector<Tensor<1,2> >     phi_v (n_dofs_per_cell);  
      std::vector<Tensor<2,2> >     phi_grads_v (n_dofs_per_cell);  
      std::vector<Tensor<1,2> >     phi_u (n_dofs_per_cell);  
      std::vector<Tensor<2,2> >     phi_grads_u (n_dofs_per_cell);  
      std::vector<double>           phi_p (n_dofs_per_cell);   
      
      const Tensor<2,dealdim> Identity = ALE_Transformations
	::get_Identity<dealdim> ();
      
      // fluid
      if (material_id == 0)
	{
	  
	  for(unsigned int q_point = 0; q_point < n_q_points; q_point++)
	    {
	      	for(unsigned int k = 0; k < n_dofs_per_cell; k++)
		  {	
		    phi_p[k]       = state_fe_values[pressure].value (k, q_point);
		    phi_v[k]       = state_fe_values[velocities].value (k, q_point);
		    phi_grads_v[k] = state_fe_values[velocities].gradient (k, q_point);			      			 
		    phi_u[k]       = state_fe_values[displacements].value (k, q_point);
		    phi_grads_u[k] = state_fe_values[displacements].gradient (k, q_point);		
		  }

		// adjoint values and grads			
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
		
		double dzp = _dzvalues[q_point](4);

		Tensor<2,2> dzu_grads;
		dzu_grads.clear();
		dzu_grads[0][0] = _dzgrads[q_point][2][0];
		dzu_grads[0][1] = _dzgrads[q_point][2][1];
		dzu_grads[1][0] = _dzgrads[q_point][3][0];
		dzu_grads[1][1] = _dzgrads[q_point][3][1];
	
	
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
				
		Tensor<2,2> dzpI_state;
		dzpI_state.clear();
		dzpI_state[0][0] = _dz_state_values[q_point](4);
		dzpI_state[0][1] = 0.0;
		dzpI_state[1][0] = 0.0;
		dzpI_state[1][1] = _dz_state_values[q_point](4);

		
		// state values and grads
		const Tensor<2,dealdim> F = ALE_Transformations
		  ::get_F<dealdim> (q_point, _dz_state_grads);
		
		const Tensor<2,dealdim> F_Inverse = ALE_Transformations
		  ::get_F_Inverse<dealdim> (F);
		
		const Tensor<2,dealdim> F_Inverse_T = ALE_Transformations
		  ::get_F_Inverse_T<dealdim> (F_Inverse);
		
		const Tensor<2,dealdim> F_T = ALE_Transformations
		  ::get_F_T<dealdim> (F);
		
		const double J = ALE_Transformations
		  ::get_J<dealdim> (F);
		
		const Tensor<2,dealdim> sigma_ALE = NSE_in_ALE
		  ::get_stress_fluid_ALE<dealdim> (_density_fluid, _viscosity, dzpI_state,
						   dzv_state_grads, transpose(dzv_state_grads), F_Inverse, F_Inverse_T );



	   	for(unsigned int j = 0; j < n_dofs_per_cell; j++)
		  {	
		    const Tensor<2,dealdim> pI_LinP = ALE_Transformations
		      ::get_pI_LinP<dealdim> (phi_p[j]);
		  
		    const Tensor<2,dealdim> grad_v_LinV = ALE_Transformations
		      ::get_grad_v_LinV<dealdim> (phi_grads_v[j]);
		  
		    const double J_LinU =  ALE_Transformations
		      ::get_J_LinU<dealdim> (q_point, _dz_state_grads, phi_grads_u[j]);
		  
		    const double J_Inverse_LinU = ALE_Transformations
		      ::get_J_Inverse_LinU<dealdim> (J, J_LinU);

		    const Tensor<2,dealdim> J_F_Inverse_T_LinU = ALE_Transformations
		      ::get_J_F_Inverse_T_LinU<dealdim> (phi_grads_u[j]);
		  
		    const Tensor<2,dealdim> F_Inverse_LinU = ALE_Transformations
		      ::get_F_Inverse_LinU (phi_grads_u[j],J, J_LinU,q_point,_dz_state_grads);
					  
	    
		    // four main equations
		    const Tensor<2,dealdim>  stress_fluid_ALE_1st_term_LinAll =  NSE_in_ALE			
		      ::get_stress_fluid_ALE_1st_term_LinAll_short<dealdim> (dzpI_state, F_Inverse_T,J_F_Inverse_T_LinU,	
									     pI_LinP,J);
		    
		    const double incompressibility_ALE_LinAll = NSE_in_ALE
		      ::get_Incompressibility_ALE_LinAll<dealdim> (phi_grads_v[j],phi_grads_u[j], q_point,
								   _dz_state_grads); 
		    
		    const Tensor<2,dealdim> stress_fluid_ALE_2nd_term_LinAll = NSE_in_ALE
		      ::get_stress_fluid_ALE_2nd_term_LinAll_short (J_F_Inverse_T_LinU, sigma_ALE,						      
								    dzv_state_grads, grad_v_LinV, F_Inverse,	       
								    F_Inverse_LinU, J, _viscosity, _density_fluid );     
		    
		    const Tensor<1,dealdim> convection_fluid_LinAll_short = NSE_in_ALE
		      ::get_Convection_LinAll_short<dealdim> (phi_grads_v[j],phi_v[j], J,J_LinU,
							      F_Inverse, F_Inverse_LinU, dzv_state, dzv_state_grads, _density_fluid);

	

		    local_cell_vector(j) +=  scale * 
		      (
		       convection_fluid_LinAll_short * dzv
		       + scalar_product(stress_fluid_ALE_2nd_term_LinAll, dzv_grads)
		       + scalar_product(stress_fluid_ALE_1st_term_LinAll, dzv_grads) 
		       + incompressibility_ALE_LinAll *  dzp
		       + _alpha_u * cell_diameter * cell_diameter
		       * scalar_product(phi_grads_u[j], dzu_grads)
		       )
		      * state_fe_values.JxW(q_point);
		  	     
		}
	    }  
	} // material_id == 0
      else if (material_id == 1)
	{
	  
	  for(unsigned int q_point = 0; q_point < n_q_points; q_point++)
	    {
	      	for(unsigned int k = 0; k < n_dofs_per_cell; k++)
		  {	
		    phi_p[k]       = state_fe_values[pressure].value (k, q_point);
		    phi_v[k]       = state_fe_values[velocities].value (k, q_point);
		    phi_grads_v[k] = state_fe_values[velocities].gradient (k, q_point);			      			 
		    phi_u[k]       = state_fe_values[displacements].value (k, q_point);
		    phi_grads_u[k] = state_fe_values[displacements].gradient (k, q_point);		
		  }
	      
			// adjoint values and grads
		Tensor<2,2> dzv_grads;
		dzv_grads.clear();
		dzv_grads[0][0] = _dzgrads[q_point][0][0];
		dzv_grads[0][1] = _dzgrads[q_point][0][1];
		dzv_grads[1][0] = _dzgrads[q_point][1][0];
		dzv_grads[1][1] = _dzgrads[q_point][1][1];
		
	
		Tensor<1,2> dzu;
		dzu.clear();
		dzu[0] = _dzvalues[q_point](2);
		dzu[1] = _dzvalues[q_point](3);
      
		double dzp = _dzvalues[q_point](4);
	

		// state values and grads
		const Tensor<2,dealdim> F = ALE_Transformations
		::get_F<dealdim> (q_point, _dz_state_grads);
	     
	      const Tensor<2,dealdim> F_Inverse = ALE_Transformations
		::get_F_Inverse<dealdim> (F);
	      
	      const Tensor<2,dealdim> F_Inverse_T = ALE_Transformations
		::get_F_Inverse_T<dealdim> (F_Inverse);
	      
	      const Tensor<2,dealdim> F_T = ALE_Transformations
		::get_F_T<dealdim> (F);
	      
	      const double J = ALE_Transformations
		::get_J<dealdim> (F);
	      	     
	      const Tensor<2,dealdim> E = Structure_Terms_in_ALE 
		::get_E<dealdim> (F_T, F, Identity);
	      
	      const double tr_E = Structure_Terms_in_ALE
		::get_tr_E<dealdim> (E);

	     	for(unsigned int j = 0; j < n_dofs_per_cell; j++)
		  {

		     //_z_state_values
		    //_z_state_grads
		    const double J_LinU = ALE_Transformations
		      ::get_J_LinU<dealdim> (q_point, _dz_state_grads,
					     phi_grads_u[j]);
		    
		    const Tensor<2,dealdim> F_LinU = ALE_Transformations		  
		      ::get_F_LinU<dealdim> (phi_grads_u[j]);
		    
		    const Tensor<2,dealdim> F_Inverse_LinU = ALE_Transformations
		      ::get_F_Inverse_LinU<dealdim> (phi_grads_u[j],
						     J, J_LinU, q_point,
						     _dz_state_grads);
		    
		    const Tensor<2,dealdim> F_Inverse_T_LinU = transpose(F_Inverse_LinU);
		    
		    const Tensor<2,dealdim> J_F_Inverse_T_LinU = ALE_Transformations
		      ::get_J_F_Inverse_T_LinU<dealdim> (phi_grads_u[j]);
		    
		    // STVK: Green_Lagrange strain tensor derivatives
		    const Tensor<2,dealdim> E_LinU = 0.5 * (transpose(F_LinU) * F + transpose(F) * F_LinU);
		    
		    const double tr_E_LinU = Structure_Terms_in_ALE
		      ::get_tr_E_LinU<dealdim> (q_point,_dz_state_grads, phi_grads_u[j]);
		    

		    // STVK
		    // piola-kirchhoff stress structure STVK linearized in all directions 
		    // J * (1/J*F*(lambda*tr_E*I + 2*mu*E)*F^T) * F^{-T} --> Linearization
		    Tensor<2,dealdim> piola_kirchhoff_stress_structure_STVK_LinALL;
		    piola_kirchhoff_stress_structure_STVK_LinALL = _lame_coefficient_lambda * 
		      (F_LinU * tr_E * Identity + F * tr_E_LinU * Identity) 
		      + 2 * _lame_coefficient_mu * (F_LinU * E + F * E_LinU);

		
		  local_cell_vector(j) +=  scale * 
		    (scalar_product(piola_kirchhoff_stress_structure_STVK_LinALL,dzv_grads)				      
		     - _density_structure * phi_v[j] * dzu
		     + phi_p[j] * dzp
		     )
		    * state_fe_values.JxW(q_point);	     
		}
	    }  
	} // material_id == 1
      
      
      
    }
    
    void CellEquation_UU(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
			 dealii::Vector<double> &local_cell_vector, double scale, double /*scale_ico*/)
    {
      const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
      unsigned int n_q_points = cdc.GetNQPoints();
      unsigned int material_id = cdc.GetMaterialId(); 
      double cell_diameter = cdc.GetCellDiameter();

      assert(this->_problem_type == "adjoint_hessian");
          
      _zvalues.resize(n_q_points,Vector<double>(5));
      _zgrads.resize(n_q_points,vector<Tensor<1,2> >(5));
      
      cdc.GetValuesState("adjoint",_zvalues);
      cdc.GetGradsState("adjoint",_zgrads);
      
      _du_tangent_values.resize(n_q_points,Vector<double>(5));
      _du_tangent_grads.resize(n_q_points,vector<Tensor<1,2> >(5));
      
      cdc.GetValuesState("tangent",_du_tangent_values);
      cdc.GetGradsState("tangent",_du_tangent_grads);

      _du_state_values.resize(n_q_points,Vector<double>(5));
      _du_state_grads.resize(n_q_points,vector<Tensor<1,2> >(5));
      
      cdc.GetValuesState("state",_du_state_values);
      cdc.GetGradsState("state",_du_state_grads);

      
      const FEValuesExtractors::Vector velocities (0);
      const FEValuesExtractors::Vector displacements (2);
      const FEValuesExtractors::Scalar pressure (4);
      
      std::vector<Tensor<1,2> >     phi_v (n_dofs_per_cell);  
      std::vector<Tensor<2,2> >     phi_grads_v (n_dofs_per_cell);  
      std::vector<Tensor<1,2> >     phi_u (n_dofs_per_cell);  
      std::vector<Tensor<2,2> >     phi_grads_u (n_dofs_per_cell);  
      std::vector<double>           phi_p (n_dofs_per_cell);   
      
      const Tensor<2,dealdim> Identity = ALE_Transformations
	::get_Identity<dealdim> ();

      if (material_id == 0)
	{
	  
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
	      Tensor<2,2> duv_tangent_grads;
	      duv_tangent_grads.clear();
	      duv_tangent_grads[0][0] = _du_tangent_grads[q_point][0][0];
	      duv_tangent_grads[0][1] = _du_tangent_grads[q_point][0][1];
	      duv_tangent_grads[1][0] = _du_tangent_grads[q_point][1][0];
	      duv_tangent_grads[1][1] = _du_tangent_grads[q_point][1][1];
	      
	      Tensor<1,2> duv_tangent;
	      duv_tangent.clear();
	      duv_tangent[0] = _du_tangent_values[q_point](0);
	      duv_tangent[1] = _du_tangent_values[q_point](1);
	      
	      
	      for(unsigned int i = 0; i < n_dofs_per_cell; i++)
		{
		  const Tensor<1,2> phi_i_v = state_fe_values[velocities].value (i, q_point);
		  const Tensor<2,2> phi_i_grads_v = state_fe_values[velocities].gradient (i, q_point);
		  
		  local_cell_vector(i) +=  scale * 
		    (_density_fluid * (phi_i_grads_v * duv_tangent + duv_tangent_grads * phi_i_v) * zv 
		     )
		    * state_fe_values.JxW(q_point);	     
		}
	    }    
	} // end material_id == 0
      if (material_id == 1)
	{
	
	    
	  for(unsigned int q_point = 0; q_point < n_q_points; q_point++)
	    {
	     	for(unsigned int k = 0; k < n_dofs_per_cell; k++)
		  {	
		    phi_p[k]       = state_fe_values[pressure].value (k, q_point);
		    phi_v[k]       = state_fe_values[velocities].value (k, q_point);
		    phi_grads_v[k] = state_fe_values[velocities].gradient (k, q_point);			      			 
		    phi_u[k]       = state_fe_values[displacements].value (k, q_point);
		    phi_grads_u[k] = state_fe_values[displacements].gradient (k, q_point);		
		  }
	      
		// adjoint values and grads
		Tensor<2,2> zv_grads;
		zv_grads.clear();
		zv_grads[0][0] = _zgrads[q_point][0][0];
		zv_grads[0][1] = _zgrads[q_point][0][1];
		zv_grads[1][0] = _zgrads[q_point][1][0];
		zv_grads[1][1] = _zgrads[q_point][1][1];
		
	
		// tangent values and grads
		const Tensor<2,dealdim> F = ALE_Transformations
		::get_F<dealdim> (q_point, _du_tangent_grads);
	     
	      const Tensor<2,dealdim> F_Inverse = ALE_Transformations
		::get_F_Inverse<dealdim> (F);
	      
	      const Tensor<2,dealdim> F_Inverse_T = ALE_Transformations
		::get_F_Inverse_T<dealdim> (F_Inverse);
	      
	      const Tensor<2,dealdim> F_T = ALE_Transformations
		::get_F_T<dealdim> (F);
	      
	      const double J = ALE_Transformations
		::get_J<dealdim> (F);
	      	     
	      const Tensor<2,dealdim> E = Structure_Terms_in_ALE 
		::get_E<dealdim> (F_T, F, Identity);
	      
	      const double tr_E = Structure_Terms_in_ALE
		::get_tr_E<dealdim> (E);


	      // state values and grads
	      const Tensor<2,dealdim> F_state = ALE_Transformations
		::get_F<dealdim> (q_point, _du_state_grads);
	     
	      const Tensor<2,dealdim> F_Inverse_state = ALE_Transformations
		::get_F_Inverse<dealdim> (F);
	      
	      const Tensor<2,dealdim> F_Inverse_T_state = ALE_Transformations
		::get_F_Inverse_T<dealdim> (F_Inverse);
	      
	      const Tensor<2,dealdim> F_T_state = ALE_Transformations
		::get_F_T<dealdim> (F);

	      const Tensor<2,dealdim> F_LinU_state = ALE_Transformations
		::get_F_LinU_state<dealdim> (q_point, _du_state_grads);


	      for(unsigned int j = 0; j < n_dofs_per_cell; j++)
		{
		  //_du_state_values
		    //_z_state_grads
		    const double J_LinU = ALE_Transformations
		      ::get_J_LinU<dealdim> (q_point, _du_tangent_grads,
					     phi_grads_u[j]);
		    
		    const Tensor<2,dealdim> F_LinU = ALE_Transformations		  
		      ::get_F_LinU<dealdim> (phi_grads_u[j]);
		    
		    const Tensor<2,dealdim> F_Inverse_LinU = ALE_Transformations
		      ::get_F_Inverse_LinU<dealdim> (phi_grads_u[j],
						     J, J_LinU, q_point,
						     _du_tangent_grads);
		    
		    const Tensor<2,dealdim> F_Inverse_T_LinU = transpose(F_Inverse_LinU);
		    
		    const Tensor<2,dealdim> J_F_Inverse_T_LinU = ALE_Transformations
		      ::get_J_F_Inverse_T_LinU<dealdim> (phi_grads_u[j]);
		    
		    // STVK: Green_Lagrange strain tensor derivatives
		    const Tensor<2,dealdim> E_LinU = 0.5 * (transpose(F_LinU) * F + transpose(F) * F_LinU);

		    const Tensor<2,dealdim> E_LinU_state = 0.5 * (transpose(F_LinU_state) * F + transpose(F) * F_LinU_state);
		    

		    // I'm not so sure!!!!!
		    const double tr_E_LinU = Structure_Terms_in_ALE
		      ::get_tr_E_LinU<dealdim> (q_point,_du_tangent_grads, phi_grads_u[j]);
		

		    // 2nd derivatives for tr_E and E
		    const double tr_E_LinW_LinU = Structure_Terms_in_ALE
		      ::get_tr_E_LinU<dealdim> (q_point,_du_state_grads, phi_grads_u[j]);
		   
		    const Tensor<2,dealdim> E_LinW_LinU 
		      = 0.5 * (transpose(F_LinU) * F_LinU_state + transpose(F_LinU_state) * F_LinU);
		   
		    // STVK: 2nd derivative
		    Tensor<2,dealdim> piola_kirchhoff_stress_structure_STVK_LinALL_LinALL;
		    piola_kirchhoff_stress_structure_STVK_LinALL_LinALL = _lame_coefficient_lambda * 
		      (tr_E_LinW_LinU * F + 2*tr_E_LinU * F_LinU) 
		      + 2 * _lame_coefficient_mu * (F_LinU * E_LinU_state + F_LinU_state * E_LinU + F * E_LinW_LinU);

	   
		    // STVK
		    // piola-kirchhoff stress structure STVK linearized in all directions 
		    // J * (1/J*F*(lambda*tr_E*I + 2*mu*E)*F^T) * F^{-T} --> Linearization
		    Tensor<2,dealdim> piola_kirchhoff_stress_structure_STVK_LinALL;
		    piola_kirchhoff_stress_structure_STVK_LinALL = _lame_coefficient_lambda * 
		      (F_LinU * tr_E * Identity + F * tr_E_LinU * Identity) 
		      + 2 * _lame_coefficient_mu * (F_LinU * E + F * E_LinU);

		    // 0.0 ist besser, da Deformationen sehr klein
		  local_cell_vector(j) +=  0.0 * scale * 
		    scalar_product(piola_kirchhoff_stress_structure_STVK_LinALL,zv_grads)				    		     
		    * state_fe_values.JxW(q_point);	     
		}
	    }  

	} // end material_id == 1
      
      
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
	const FEValuesExtractors::Scalar pressure (4);
	
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
		
		local_cell_vector(i) -= 1.0 * scale * neumann_value * phi_i_v * state_fe_face_values.JxW(q_point);
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
		      
		      local_entry_matrix(i,j) -=  1.0 * neumann_value * phi_i_v * state_fe_face_values.JxW(q_point);
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
	      zvboundary * state_fe_face_values.JxW(q_point);
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
	      zvboundary * state_fe_face_values.JxW(q_point);
	    
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
	
		local_cell_vector(i) -= 1.0 * scale * 
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
	
		local_cell_vector(i) -= 1.0 * scale * 
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
	
		local_cell_vector(i) -= 1.0 * scale * 
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
 


///// Hier FaceEquation einfuegen

  void FaceEquation (const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
			 dealii::Vector<double> &local_cell_vector, double scale, double /*scale_ico*/)
  {

  }
  void FaceMatrix (const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
		       dealii::FullMatrix<double> &local_entry_matrix, double /*scale*/, double /*scale_ico*/)
  {
  }
  void FaceRightHandSide (const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc __attribute__((unused)),
			      dealii::Vector<double> &local_cell_vector __attribute__((unused)), double scale __attribute__((unused)))
  {
    assert(this->_problem_type == "state");
  }
 
  void FaceEquation_Q (const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
			   dealii::Vector<double> &local_cell_vector,
			   double scale, double /*scale_ico*/)
  {
  }

 void FaceEquation_QT (const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
			   dealii::Vector<double> &local_cell_vector,
			   double scale, double /*scale_ico*/)
 {
}

 void FaceEquation_QTT (const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
			    dealii::Vector<double> &local_cell_vector,
			    double scale, double /*scale_ico*/)
 { 
 }

 void FaceEquation_U (const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
			  dealii::Vector<double> &local_cell_vector, double scale, double /*scale_ico*/)
 {
 }

  void FaceEquation_UT (const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
			   dealii::Vector<double> &local_cell_vector,
			   double scale, double /*scale_ico*/)
  {
  }

 void FaceEquation_UTT (const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
			    dealii::Vector<double> &local_cell_vector,
			    double scale, double /*scale_ico*/)
  {
  }

void FaceEquation_UU (const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc __attribute__((unused)),
			   dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/, double /*scale_ico*/) 
  {
    assert(this->_problem_type == "adjoint_hessian");	
  }
 
void FaceEquation_QU (const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc __attribute__((unused)),
			   dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/, double /*scale_ico*/)
  {
    assert(this->_problem_type == "adjoint_hessian");	
  }
 
void FaceEquation_UQ (const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc __attribute__((unused)),
			   dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/, double /*scale_ico*/)
  {
    
  }
 
void FaceEquation_QQ (const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc __attribute__((unused)),
			   dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/, double /*scale_ico*/)
  {
    
  }

///////// Hier Face zuende

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


     // for CellEquation_UU
     vector<Vector<double> > _du_tangent_values;
     vector<vector<Tensor<1,dealdim> > > _du_tangent_grads;
     
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
     
     
     vector<unsigned int> _state_block_components;
     vector<unsigned int> _control_block_components;
     
     double _cell_diameter;
     
     // Fluid- and material variables
     double _density_fluid, _density_structure, _viscosity, _alpha_u, _lame_coefficient_mu, 
       _poisson_ratio_nu, _lame_coefficient_lambda;
     
  };
#endif

