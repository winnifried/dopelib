#ifndef _LOCALPDE_
#define _LOCALPDE_

#include "pdeinterface.h"
#include "ale_transformations.h"
#include "celldatacontainer.h"
#include "facedatacontainer.h"


using namespace std;
using namespace dealii;
using namespace DOpE;

template<typename VECTOR, int dopedim, int dealdim>
  class LocalPDE : public PDEInterface<CellDataContainer,FaceDataContainer,dealii::DoFHandler<dealdim>, VECTOR,dopedim,dealdim>
  {
  public:

     static void declare_params(ParameterReader &param_reader)
    {
      param_reader.SetSubsection("Local PDE parameters");
      param_reader.declare_entry("density_fluid", "1.0",
				 Patterns::Double(0));
      param_reader.declare_entry("density_structure", "1.0",
				 Patterns::Double(0));
      param_reader.declare_entry("viscosity", "1.0",
				 Patterns::Double(0));
      param_reader.declare_entry("alpha_u", "1.0",
				 Patterns::Double(0));
      param_reader.declare_entry("mu", "1.0",
				 Patterns::Double(0));
      param_reader.declare_entry("poisson_ratio_nu", "0.4",
				 Patterns::Double(0));
    }

  LocalPDE(ParameterReader &param_reader) : _state_block_components(5,0)
      {
	_alpha = 1.e-3;
	_state_block_components[2]= 1;
	_state_block_components[3]= 1;
	_state_block_components[4]= 2;

	param_reader.SetSubsection("Local PDE parameters");
	density_fluid = param_reader.get_double ("density_fluid");
	density_structure = param_reader.get_double ("density_structure");
	viscosity = param_reader.get_double ("viscosity");
	alpha_u = param_reader.get_double ("alpha_u");
	lame_coefficient_mu = param_reader.get_double ("mu");
	poisson_ratio_nu = param_reader.get_double ("poisson_ratio_nu");

	lame_coefficient_lambda =  (2 * poisson_ratio_nu * lame_coefficient_mu)/
	  (1.0 - 2 * poisson_ratio_nu);

	// variables for mesh-dependent alpha_u
	structure_continuation_E = 1.0e-6;
	scale_distance_to_interface = 5.0e-6;
	scale_d_exp = 100.0;

	// only for pseudo-STVK
	structure_continuation_nus = -0.5;

      }

     // necessary to compute face integrals
     bool HasFaces() const
     {
       return true;
     }

//
//     // CellEquation contains all implicit terms,
//     // e.g. pressure, incompressibility of the fluid,
//     // where no "theta" time-discretization should be applied
//     void CellEquationOnlyImplicit (const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
//				   dealii::Vector<double> &local_cell_vector,
//				   double scale, double scale_ico)
//    {
//      assert(this->_problem_type == "state");
//
//      const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
//      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
//      unsigned int n_q_points = cdc.GetNQPoints();
//      unsigned int material_id = cdc.GetMaterialId();
//      double cell_diameter = cdc.GetCellDiameter();
//
//        // old Newton step solution values and gradients
//      _uvalues.resize(n_q_points,Vector<double>(5));
//      _ugrads.resize(n_q_points,vector<Tensor<1,2> >(5));
//
//      cdc.GetValuesState("last_newton_solution",_uvalues);
//      cdc.GetGradsState("last_newton_solution",_ugrads);
//
//      // last timestep solution values and gradients
//      _last_timestep_uvalues.resize(n_q_points,Vector<double>(5));
//      _last_timestep_ugrads.resize(n_q_points,vector<Tensor<1,2> >(5));
//
//      cdc.GetValuesState("last_time_solution",_last_timestep_uvalues);
//      cdc.GetGradsState("last_time_solution",_last_timestep_ugrads);
//
//      const FEValuesExtractors::Vector velocities (0);
//      const FEValuesExtractors::Vector displacements (2);
//      const FEValuesExtractors::Scalar pressure (4);
//
//      const Tensor<2,dealdim> Identity = ALE_Transformations
//	::get_Identity<dealdim> ();
//
//      if (material_id == 0)
//	{
//	  for (unsigned int q_point=0;q_point<n_q_points;q_point++)
//	    {
//	      // variable for mesh-dependent alpha_u
//	      distance_to_interface = std::sqrt((state_fe_values.quadrature_point(q_point)(0)-0.6)*
//						(state_fe_values.quadrature_point(q_point)(0)-0.6)+
//						(state_fe_values.quadrature_point(q_point)(1)-0.2)*
//						(state_fe_values.quadrature_point(q_point)(1)-0.2));
//
//	      // mesh-dependent alpha_u. Here the variable has name tmp_structure_continuation_E.
//	      tmp_structure_continuation_E = structure_continuation_E
//		+ scale_distance_to_interface*std::exp(-scale_d_exp*std::abs(distance_to_interface));
//
//	      // for elastic structure continuation with pseudo-STVK
//	        structure_continuation_mu = tmp_structure_continuation_E/(2.0*(1.0+structure_continuation_nus));
//
//		structure_continuation_lambda =  (2 * structure_continuation_nus * structure_continuation_mu)/
//		     (1.0 - 2 * structure_continuation_nus);
//
//	      const Tensor<2,dealdim> pI = ALE_Transformations::get_pI<dealdim> (q_point, _uvalues);
//	      const Tensor<1,dealdim> v = ALE_Transformations::get_v<dealdim> (q_point, _uvalues);
//
//	      const Tensor<2,dealdim> grad_v = ALE_Transformations
//		::get_grad_v<dealdim> (q_point, _ugrads);
//
//	      const Tensor<2,dealdim> grad_v_T = ALE_Transformations
//		::get_grad_v_T<dealdim> (grad_v);
//
//	      const Tensor<2,dealdim> grad_u = ALE_Transformations
//		::get_grad_u<dealdim> (q_point, _ugrads);
//
//	      const Tensor<2,dealdim> F = ALE_Transformations
//		::get_F<dealdim> (q_point, _ugrads);
//
//	      const Tensor<2,dealdim> F_Inverse = ALE_Transformations
//		::get_F_Inverse<dealdim> (F);
//
//	      const Tensor<2,dealdim> F_Inverse_T = ALE_Transformations
//		::get_F_Inverse_T<dealdim> (F_Inverse);
//
//	      const double J = ALE_Transformations
//		::get_J<dealdim> (F);
//
//	      const Tensor<2,dealdim> fluid_pressure = (-pI * J * F_Inverse_T);
//
//	      const double incompressiblity_fluid = NSE_in_ALE
//		::get_Incompressibility_ALE<dealdim> (q_point, _ugrads);
//
//
//	      // pseudo-STVK
//	      const Tensor<2,dealdim> linear_E = Structure_Terms_in_ALE
//		::get_linear_E<dealdim> (q_point, _ugrads);
//	      const double tr_linear_E = Structure_Terms_in_ALE
//		::get_tr_E<dealdim> (linear_E);
//
//	      const Tensor<2,dealdim> sigma_structure_continuation_ALE
//		= (structure_continuation_lambda *
//		   tr_linear_E * Identity +
//		   2 * structure_continuation_mu *
//		   linear_E);
//
//
//
//	      for (unsigned int i=0;i<n_dofs_per_cell;i++)
//		{
//		  // Fluid
//		      const Tensor<1,dealdim> phi_i_v = state_fe_values[velocities].value (i, q_point);
//		      const Tensor<2,dealdim> phi_i_grads_v = state_fe_values[velocities].gradient (i, q_point);
//
//		      local_cell_vector(i) +=  scale * (scalar_product(fluid_pressure, phi_i_grads_v)
//							)
//			* state_fe_values.JxW(q_point);
//
//		      const Tensor<1,dealdim> phi_i_u = state_fe_values[displacements].value (i, q_point);
//		      const Tensor<2,dealdim> phi_i_grads_u = state_fe_values[displacements].gradient (i, q_point);
//
//
////		      local_cell_vector(i) += scale * (//tmp_structure_continuation_E
////						       alpha_u
////						       * cell_diameter * cell_diameter
////						       * scalar_product(grad_u, phi_i_grads_u)
////						       ) * state_fe_values.JxW(q_point)
////						       + scale_ico * () * state_fe_values.JxW(q_point);
//
//
//
//
//		      const double phi_i_p = state_fe_values[pressure].value (i, q_point);
//
//		      local_cell_vector(i) += scale * (incompressiblity_fluid * phi_i_p) * state_fe_values.JxW(q_point);
//
//		}
//	    }
//	}
//      else if (material_id == 1)
//	{
//	  for (unsigned int q_point=0;q_point<n_q_points;q_point++)
//	    {
//	      for (unsigned int i=0;i<n_dofs_per_cell;i++)
//		{
//		  // Structure, STVK, implicit
//		      const double phi_i_p = state_fe_values[pressure].value (i, q_point);
//
//		      local_cell_vector(i) += scale * (_uvalues[q_point](dealdim+dealdim) * phi_i_p) * state_fe_values.JxW(q_point);
//
//
//		}
//	    }
//	}
//    }

     // CellEquationExplicit contains all "normal" terms which
     // can be treated by full "theta" time-discretization
     void CellEquation(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
				   dealii::Vector<double> &local_cell_vector,
				   double scale, double scale_ico)
    {
      assert(this->_problem_type == "state");

      const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
      unsigned int n_q_points = cdc.GetNQPoints();
      unsigned int material_id = cdc.GetMaterialId();
      double cell_diameter = cdc.GetCellDiameter();

      // old Newton step solution values and gradients
      _uvalues.resize(n_q_points,Vector<double>(5));
      _ugrads.resize(n_q_points,vector<Tensor<1,2> >(5));

      cdc.GetValuesState("last_newton_solution",_uvalues);
      cdc.GetGradsState("last_newton_solution",_ugrads);

      // old timestep solution values and gradients
      _last_timestep_uvalues.resize(n_q_points,Vector<double>(5));
      _last_timestep_ugrads.resize(n_q_points,vector<Tensor<1,2> >(5));

      cdc.GetValuesState("last_time_solution",_last_timestep_uvalues);
      cdc.GetGradsState("last_time_solution",_last_timestep_ugrads);

      const FEValuesExtractors::Vector velocities (0);
      const FEValuesExtractors::Vector displacements (2);
      const FEValuesExtractors::Scalar pressure (4);

      const Tensor<2,dealdim> Identity = ALE_Transformations
	::get_Identity<dealdim> ();


      if (material_id == 0)
	{
	  for (unsigned int q_point=0;q_point<n_q_points;q_point++)
	    {

	      const Tensor<1,dealdim> v = ALE_Transformations
		::get_v<dealdim> (q_point, _uvalues);

	      const Tensor<2,dealdim> grad_v = ALE_Transformations
		::get_grad_v<dealdim> (q_point, _ugrads);

	      const Tensor<2,dealdim> grad_v_T = ALE_Transformations
		::get_grad_v_T<dealdim> (grad_v);

//	      const Tensor<1,dealdim> u = ALE_Transformations
//		::get_u<dealdim> (q_point, _uvalues);

	      const Tensor<2,dealdim> F = ALE_Transformations
		::get_F<dealdim> (q_point, _ugrads);

	      const Tensor<2,dealdim> F_Inverse = ALE_Transformations
		::get_F_Inverse<dealdim> (F);

	      const Tensor<2,dealdim> F_Inverse_T = ALE_Transformations
		::get_F_Inverse_T<dealdim> (F_Inverse);

	      const double J = ALE_Transformations::get_J<dealdim>(F);

              const Tensor<2, dealdim> grad_u =
                  ALE_Transformations::get_grad_u<dealdim>(q_point, _ugrads);

              const Tensor<2, dealdim> pI =
                  ALE_Transformations::get_pI<dealdim>(q_point, _uvalues);

              const Tensor<2, dealdim> sigma_ALE =
                  NSE_in_ALE::get_stress_fluid_except_pressure_ALE<dealdim>(
                      density_fluid, viscosity, grad_v, grad_v_T, F_Inverse,
                      F_Inverse_T);

              const Tensor<2, dealdim> stress_fluid = (J * sigma_ALE
                  * F_Inverse_T);

              const Tensor<1, dealdim> convection_fluid = density_fluid * J
                  * (grad_v * F_Inverse * v);

              const Tensor<2, dealdim> fluid_pressure = (-pI * J * F_Inverse_T);

              const double incompressiblity_fluid =
                  NSE_in_ALE::get_Incompressibility_ALE<dealdim>(q_point,
                      _ugrads);


	      for (unsigned int i=0;i<n_dofs_per_cell;i++)
		{
		  // Fluid, explicit
		  const Tensor<1,dealdim> phi_i_v = state_fe_values[velocities].value (i, q_point);
		  const Tensor<2,dealdim> phi_i_grads_v = state_fe_values[velocities].gradient (i, q_point);
		  const Tensor<2,dealdim> phi_i_grads_u = state_fe_values[displacements].gradient (i, q_point);
                  const double phi_i_p = state_fe_values[pressure].value(i,
                      q_point);

		  local_cell_vector(i) += scale * (convection_fluid * phi_i_v
                      + scalar_product(stress_fluid, phi_i_grads_v))
                      * state_fe_values.JxW(q_point);

                  local_cell_vector(i) += scale_ico * (//tmp_structure_continuation_E
                      alpha_u * cell_diameter * cell_diameter * scalar_product(
                          grad_u, phi_i_grads_u))
                      * state_fe_values.JxW(q_point);

                  local_cell_vector(i) += scale_ico * (scalar_product(
                      fluid_pressure, phi_i_grads_v)) * state_fe_values.JxW(
                      q_point);

                  local_cell_vector(i) += scale_ico * (incompressiblity_fluid
                      * phi_i_p) * state_fe_values.JxW(q_point);

		}
	    }
	}
      else if (material_id == 1)
	{
	  for (unsigned int q_point=0;q_point<n_q_points;q_point++)
	    {

	      const Tensor<1,dealdim> v = ALE_Transformations
		::get_v<dealdim> (q_point, _uvalues);

//	      const Tensor<2,dealdim> grad_v = ALE_Transformations
//		::get_grad_v<dealdim> (q_point, _ugrads);

//	      const Tensor<1,dealdim> u = ALE_Transformations
//		::get_u<dealdim> (q_point, _uvalues);

	      const Tensor<2,dealdim> F = ALE_Transformations
		::get_F<dealdim> (q_point, _ugrads);

	      const Tensor<2,dealdim> F_T = ALE_Transformations
		::get_F_T<dealdim> (F);

	      const Tensor<2,dealdim> F_Inverse = ALE_Transformations
		::get_F_Inverse<dealdim> (F);

	      const Tensor<2,dealdim> F_Inverse_T = ALE_Transformations
		::get_F_Inverse_T<dealdim> (F_Inverse);

	      const double J = ALE_Transformations
		::get_J<dealdim> (F);

	      const Tensor<2,dealdim> E = Structure_Terms_in_ALE
		::get_E<dealdim> (F_T, F, Identity);

	      const double tr_E = Structure_Terms_in_ALE
		::get_tr_E<dealdim> (E);

	      // STVK
	      const Tensor<2,dealdim> sigma_structure_ALE = (1.0/J *
							 F * (lame_coefficient_lambda *
							      tr_E * Identity +
							      2 * lame_coefficient_mu *
							      E)
							 * F_T);

	      const Tensor<2,dealdim> stress_term  = (J * sigma_structure_ALE * F_Inverse_T);

	      for (unsigned int i = 0; i < n_dofs_per_cell; i++)
                {
                  // Structure, STVK, explicit

//                  const Tensor<1, dealdim> phi_i_v =
//                      state_fe_values[velocities].value(i, q_point);
                  const Tensor<2, dealdim> phi_i_grads_v =
                      state_fe_values[velocities].gradient(i, q_point);

                  const Tensor<1, dealdim> phi_i_u =
                      state_fe_values[displacements].value(i, q_point);
//                  const Tensor<2, dealdim> phi_i_grads_u =
//                      state_fe_values[displacements].gradient(i, q_point);

                  const double phi_i_p = state_fe_values[pressure].value(i,
                      q_point);
                  local_cell_vector(i) += scale * scalar_product(stress_term,
                      phi_i_grads_v) * state_fe_values.JxW(q_point);
                  // with common condition
                  local_cell_vector(i) += scale * (-v * phi_i_u)
                      * state_fe_values.JxW(q_point);

                  local_cell_vector(i) += scale_ico * (_uvalues[q_point](
                      dealdim + dealdim) * phi_i_p) * state_fe_values.JxW(
                      q_point);

		}
	    }
	}
    }

//     void CellMatrixOnlyImplicit(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
//				FullMatrix<double> &local_entry_matrix)
//    {
//      const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
//      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
//      unsigned int n_q_points = cdc.GetNQPoints();
//      unsigned int material_id = cdc.GetMaterialId();
//      double cell_diameter = cdc.GetCellDiameter();
//
//       // old Newton step solution values and gradients
//      _uvalues.resize(n_q_points,Vector<double>(5));
//      _ugrads.resize(n_q_points,vector<Tensor<1,2> >(5));
//
//      cdc.GetValuesState("last_newton_solution",_uvalues);
//      cdc.GetGradsState("last_newton_solution",_ugrads);
//
//      // old timestep solution values and gradients
//      _last_timestep_uvalues.resize(n_q_points,Vector<double>(5));
//      _last_timestep_ugrads.resize(n_q_points,vector<Tensor<1,2> >(5));
//
//      cdc.GetValuesState("last_time_solution",_last_timestep_uvalues);
//      cdc.GetGradsState("last_time_solution",_last_timestep_ugrads);
//
//      const FEValuesExtractors::Vector velocities (0);
//      const FEValuesExtractors::Vector displacements (2);
//      const FEValuesExtractors::Scalar pressure (4);
//
//      std::vector<Tensor<1,2> >     phi_v (n_dofs_per_cell);
//      std::vector<Tensor<2,2> >     phi_grads_v (n_dofs_per_cell);
//      std::vector<Tensor<1,2> >     phi_u (n_dofs_per_cell);
//      std::vector<Tensor<2,2> >     phi_grads_u (n_dofs_per_cell);
//      std::vector<double>           phi_p (n_dofs_per_cell);
//
//      const Tensor<2,dealdim> Identity = ALE_Transformations
//	::get_Identity<dealdim> ();
//
//      if (material_id == 0)
//	{
//	  for(unsigned int q_point = 0; q_point < n_q_points; q_point++)
//	    {
//	      // variable for mesh-dependent alpha_u
//	      distance_to_interface = std::sqrt((state_fe_values.quadrature_point(q_point)(0)-0.6)*
//						(state_fe_values.quadrature_point(q_point)(0)-0.6)+
//						(state_fe_values.quadrature_point(q_point)(1)-0.2)*
//						(state_fe_values.quadrature_point(q_point)(1)-0.2));
//
//	      // mesh-dependent alpha_u. Here the variable has name tmp_structure_continuation_E.
//	      tmp_structure_continuation_E = structure_continuation_E
//		+ scale_distance_to_interface*std::exp(-scale_d_exp*std::abs(distance_to_interface));
//	      // for elastic structure continuation with pseudo-STVK
//	        structure_continuation_mu = tmp_structure_continuation_E/(2.0*(1.0+structure_continuation_nus));
//
//		structure_continuation_lambda =  (2 * structure_continuation_nus * structure_continuation_mu)/
//		     (1.0 - 2 * structure_continuation_nus);
//
//	      for(unsigned int k = 0; k < n_dofs_per_cell; k++)
//		{
//		  phi_v[k]       = state_fe_values[velocities].value (k, q_point);
//		  phi_grads_v[k] = state_fe_values[velocities].gradient (k, q_point);
//		  phi_p[k]       = state_fe_values[pressure].value (k, q_point);
//		  phi_u[k]       = state_fe_values[displacements].value (k, q_point);
//		  phi_grads_u[k] = state_fe_values[displacements].gradient (k, q_point);
//		}
//
//	      const Tensor<2,dealdim> pI = ALE_Transformations::get_pI<dealdim> (q_point, _uvalues);
//	      const Tensor<1,dealdim> v = ALE_Transformations::get_v<dealdim> (q_point, _uvalues);
//
//
//	      const Tensor<1,dealdim> u = ALE_Transformations::get_u<dealdim> (q_point,_uvalues);
//
//	      const Tensor<2,dealdim> grad_v = ALE_Transformations::get_grad_v<dealdim> (q_point, _ugrads);
//
//	      const Tensor<2,dealdim> grad_v_T = ALE_Transformations::get_grad_v_T<dealdim> (grad_v);
//	      const Tensor<2,dealdim> F = ALE_Transformations::get_F<dealdim> (q_point, _ugrads);
//	      const Tensor<2,dealdim> F_Inverse = ALE_Transformations::get_F_Inverse<dealdim> (F);
//
//	      const Tensor<2,dealdim> F_Inverse_T = ALE_Transformations::get_F_Inverse_T<dealdim> (F_Inverse);
//	      const double J = ALE_Transformations::get_J<dealdim> (F);
//
//	      for(unsigned int i = 0; i < n_dofs_per_cell; i++)
//		{
//		  const Tensor<2,dealdim> pI_LinP = ALE_Transformations::get_pI_LinP<dealdim> (phi_p[i]);
//		  const Tensor<2,dealdim> grad_v_LinV = ALE_Transformations::get_grad_v_LinV<dealdim> (phi_grads_v[i]);
//		  const double J_LinU =  ALE_Transformations::get_J_LinU<dealdim> (q_point, _ugrads, phi_grads_u[i]);
//
//		  const double J_Inverse_LinU = ALE_Transformations::get_J_Inverse_LinU<dealdim> (J, J_LinU);
//		  const Tensor<2,dealdim> J_F_Inverse_T_LinU = ALE_Transformations::get_J_F_Inverse_T_LinU<dealdim> (phi_grads_u[i]);
//		  const Tensor<2,dealdim> F_Inverse_LinU = ALE_Transformations::get_F_Inverse_LinU (phi_grads_u[i],J,
//												J_LinU,q_point,_ugrads
//												);
//
//		  const Tensor<2,dealdim>  stress_fluid_ALE_1st_term_LinAll =  NSE_in_ALE
//		    ::get_stress_fluid_ALE_1st_term_LinAll_short<dealdim> (pI, F_Inverse_T,J_F_Inverse_T_LinU,
//									   pI_LinP,J);
//
//		  const double incompressibility_ALE_LinAll = NSE_in_ALE
//		    ::get_Incompressibility_ALE_LinAll<dealdim> (phi_grads_v[i],phi_grads_u[i], q_point,
//							     _ugrads);
//
//		  // pseudo-STVK: structure extension (instead of harmonic)
//		  const Tensor<2,dealdim> linear_E_LinU = 0.5 * (phi_grads_u[i] + transpose(phi_grads_u[i]));
//		  const Tensor<2,dealdim> sigma_structure_continuation_ALE_LinAll = (structure_continuation_lambda *
//										 trace(linear_E_LinU) * Identity +
//										 2 * structure_continuation_mu *
//										 linear_E_LinU);
//
//		  for(unsigned int j=0; j<n_dofs_per_cell; j++)
//		    {
//		      // Fluid, implicit
//		      local_entry_matrix(j,i) += (scalar_product(stress_fluid_ALE_1st_term_LinAll, phi_grads_v[j])
//						  )
//			* state_fe_values.JxW(q_point);
//
//		      // 1) structure extension with common condition: harmonic
//		      local_entry_matrix(j,i) += (//tmp_structure_continuation_E
//		      				   alpha_u
//		        				  * cell_diameter * cell_diameter
//		        				  * scalar_product(phi_grads_u[i], phi_grads_u[j]) )
//		        	* state_fe_values.JxW(q_point);
//
//
//		      // 2) structure extension with Nitsche trick
//		//local_entry_matrix(j,i) += (tmp_structure_continuation_E
//		//				  //alpha_u
//		//				  * scalar_product(phi_grads_u[i], phi_grads_u[j]) )
//		//	* state_fe_values.JxW(q_point);
//
//		       // 3) structure extension with pseudo-STVK
//	//	      local_entry_matrix(j,i) +=
//	//		cell_diameter * cell_diameter
//	//		* scalar_product(sigma_structure_continuation_ALE_LinAll, phi_grads_u[j])
//	//		* state_fe_values.JxW(q_point);
//	//
//
//
//		      local_entry_matrix(j,i) += (incompressibility_ALE_LinAll *  phi_p[j])
//			* state_fe_values.JxW(q_point);
//
//
//		    }
//		}
//	    }
//	}
//      else if (material_id == 1)
//	{
//	  for(unsigned int q_point = 0; q_point < n_q_points; q_point++)
//	    {
//	      for(unsigned int k = 0; k < n_dofs_per_cell; k++)
//		{
//		  phi_p[k]       = state_fe_values[pressure].value (k, q_point);
//		}
//
//	      for(unsigned int i=0; i<n_dofs_per_cell; i++)
//		{
//		  for(unsigned int j=0; j<n_dofs_per_cell; j++)
//		    {
//		      // Structure, STVK, implicit
//		      local_entry_matrix(j,i) += (phi_p[i] * phi_p[j])
//			* state_fe_values.JxW(q_point);
//
//		    }
//		}
//	    }
//	}
//    }


    void CellMatrix(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
		    FullMatrix<double> &local_entry_matrix, double scale, double scale_ico)
    {
      const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
      unsigned int n_q_points = cdc.GetNQPoints();
      unsigned int material_id = cdc.GetMaterialId();
      double cell_diameter = cdc.GetCellDiameter();
      
      // old Newton step solution values and gradients
      _uvalues.resize(n_q_points,Vector<double>(5));
      _ugrads.resize(n_q_points,vector<Tensor<1,2> >(5));

      cdc.GetValuesState("last_newton_solution",_uvalues);
      cdc.GetGradsState("last_newton_solution",_ugrads);

      // old timestep solution values and gradients
      _last_timestep_uvalues.resize(n_q_points,Vector<double>(5));
      _last_timestep_ugrads.resize(n_q_points,vector<Tensor<1,2> >(5));

      cdc.GetValuesState("last_time_solution",_last_timestep_uvalues);
      cdc.GetGradsState("last_time_solution",_last_timestep_ugrads);

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
		  phi_v[k]       = state_fe_values[velocities].value (k, q_point);
		  phi_grads_v[k] = state_fe_values[velocities].gradient (k, q_point);
		  phi_p[k]       = state_fe_values[pressure].value (k, q_point);
		  phi_u[k]       = state_fe_values[displacements].value (k, q_point);
		  phi_grads_u[k] = state_fe_values[displacements].gradient (k, q_point);
		}

	      const Tensor<2,dealdim> pI = ALE_Transformations
		::get_pI<dealdim> (q_point, _uvalues);

	      const Tensor<1,dealdim> v = ALE_Transformations
		::get_v<dealdim> (q_point, _uvalues);

//	      const Tensor<1,dealdim> u = ALE_Transformations
//		::get_u<dealdim> (q_point,_uvalues);

	      const Tensor<2,dealdim> grad_v = ALE_Transformations
		::get_grad_v<dealdim> (q_point, _ugrads);

	      const Tensor<2,dealdim> grad_v_T = ALE_Transformations
		::get_grad_v_T<dealdim> (grad_v);

	      const Tensor<2,dealdim> F = ALE_Transformations
		::get_F<dealdim> (q_point, _ugrads);

	      const Tensor<2,dealdim> F_Inverse = ALE_Transformations
		::get_F_Inverse<dealdim> (F);

	      const Tensor<2,dealdim> F_Inverse_T = ALE_Transformations
		::get_F_Inverse_T<dealdim> (F_Inverse);

	      const double J = ALE_Transformations
		::get_J<dealdim> (F);

	      const Tensor<2,dealdim> sigma_ALE = NSE_in_ALE
		::get_stress_fluid_ALE<dealdim> (density_fluid, viscosity, pI,
						 grad_v, grad_v_T, F_Inverse, F_Inverse_T );


	      for(unsigned int i = 0; i < n_dofs_per_cell; i++)
		{
		  const Tensor<2, dealdim> pI_LinP =
                      ALE_Transformations::get_pI_LinP<dealdim>(phi_p[i]);

                  const Tensor<2, dealdim> grad_v_LinV =
                      ALE_Transformations::get_grad_v_LinV<dealdim>(
                          phi_grads_v[i]);

                  const double J_LinU =
                      ALE_Transformations::get_J_LinU<dealdim>(q_point,
                          _ugrads, phi_grads_u[i]);

//                  const double J_Inverse_LinU =
//                      ALE_Transformations::get_J_Inverse_LinU<dealdim>(J,
//                          J_LinU);

                  const Tensor<2, dealdim> J_F_Inverse_T_LinU =
                      ALE_Transformations::get_J_F_Inverse_T_LinU<dealdim>(
                          phi_grads_u[i]);

                  const Tensor<2, dealdim> F_Inverse_LinU =
                      ALE_Transformations::get_F_Inverse_LinU(phi_grads_u[i],
                          J, J_LinU, q_point, _ugrads);

                  const Tensor<2, dealdim> stress_fluid_ALE_2nd_term_LinAll =
                      NSE_in_ALE::get_stress_fluid_ALE_2nd_term_LinAll_short(
                          J_F_Inverse_T_LinU, sigma_ALE, grad_v, grad_v_LinV,
                          F_Inverse, F_Inverse_LinU, J, viscosity,
                          density_fluid);

                  const Tensor<1, dealdim> convection_fluid_LinAll_short =
                      NSE_in_ALE::get_Convection_LinAll_short<dealdim>(
                          phi_grads_v[i], phi_v[i], J, J_LinU, F_Inverse,
                          F_Inverse_LinU, v, grad_v, density_fluid);

                  const Tensor<2, dealdim> stress_fluid_ALE_1st_term_LinAll =
                      NSE_in_ALE::get_stress_fluid_ALE_1st_term_LinAll_short<
                          dealdim>(pI, F_Inverse_T, J_F_Inverse_T_LinU,
                          pI_LinP, J);

                  const double incompressibility_ALE_LinAll =
                      NSE_in_ALE::get_Incompressibility_ALE_LinAll<dealdim>(
                          phi_grads_v[i], phi_grads_u[i], q_point, _ugrads);

                  // pseudo-STVK: structure extension (instead of harmonic)
                  const Tensor<2, dealdim> linear_E_LinU = 0.5
                      * (phi_grads_u[i] + transpose(phi_grads_u[i]));
//                  const Tensor<2, dealdim>
//                      sigma_structure_continuation_ALE_LinAll =
//                          (structure_continuation_lambda * trace(linear_E_LinU)
//                              * Identity + 2 * structure_continuation_mu
//                              * linear_E_LinU);


		  for(unsigned int j=0; j<n_dofs_per_cell; j++)
		    {
		      // Fluid, explicit
                      local_entry_matrix(j, i) += scale
                          * (convection_fluid_LinAll_short * phi_v[j]
                              + scalar_product(
                                  stress_fluid_ALE_2nd_term_LinAll,
                                  phi_grads_v[j])) * state_fe_values.JxW(
                          q_point);

                      local_entry_matrix(j, i) += scale_ico * (//tmp_structure_continuation_E
                          alpha_u * cell_diameter * cell_diameter
                              * scalar_product(phi_grads_u[i], phi_grads_u[j]))
                          * state_fe_values.JxW(q_point);

                      local_entry_matrix(j, i) += scale_ico * scalar_product(
                          stress_fluid_ALE_1st_term_LinAll, phi_grads_v[j])
                          * state_fe_values.JxW(q_point);

                      local_entry_matrix(j, i) += scale_ico
                          * (incompressibility_ALE_LinAll * phi_p[j])
                          * state_fe_values.JxW(q_point);


		    }
		}
	    }
	}  // end material_id = 0
      else if (material_id == 1)
	{
	  for(unsigned int q_point = 0; q_point < n_q_points; q_point++)
	    {
	      for(unsigned int k = 0; k < n_dofs_per_cell; k++)
		{
		  phi_v[k]       = state_fe_values[velocities].value (k, q_point);
		  phi_grads_v[k] = state_fe_values[velocities].gradient (k, q_point);
		  phi_u[k]       = state_fe_values[displacements].value (k, q_point);
		  phi_grads_u[k] = state_fe_values[displacements].gradient (k, q_point);
		  phi_p[k]       = state_fe_values[pressure].value (k, q_point);
		}

	      const Tensor<2,dealdim> F = ALE_Transformations
		::get_F<dealdim> (q_point, _ugrads);

	      const Tensor<2,dealdim> F_Inverse = ALE_Transformations
		::get_F_Inverse<dealdim> (F);

//	      const Tensor<2,dealdim> F_Inverse_T = ALE_Transformations
//		::get_F_Inverse_T<dealdim> (F_Inverse);

	      const Tensor<2,dealdim> F_T = ALE_Transformations
		::get_F_T<dealdim> (F);

	      const double J = ALE_Transformations
		::get_J<dealdim> (F);

	      const Tensor<2,dealdim> E = Structure_Terms_in_ALE
		::get_E<dealdim> (F_T, F, Identity);

	      const double tr_E = Structure_Terms_in_ALE
		::get_tr_E<dealdim> (E);

	      for(unsigned int i=0; i<n_dofs_per_cell; i++)
		{

		  const double J_LinU = ALE_Transformations
		    ::get_J_LinU<dealdim> (q_point, _ugrads,
				       phi_grads_u[i]);

		  const Tensor<2,dealdim> F_LinU = ALE_Transformations
		    ::get_F_LinU<dealdim> (phi_grads_u[i]);

		  const Tensor<2,dealdim> F_Inverse_LinU = ALE_Transformations
		    ::get_F_Inverse_LinU<dealdim> (phi_grads_u[i],
					       J, J_LinU, q_point,
					       _ugrads);

//		  const Tensor<2,dealdim> F_Inverse_T_LinU = transpose(F_Inverse_LinU);
//
//		  const Tensor<2,dealdim> J_F_Inverse_T_LinU = ALE_Transformations
//		    ::get_J_F_Inverse_T_LinU<dealdim> (phi_grads_u[i]);

		  // STVK: Green_Lagrange strain tensor derivatives
		  const Tensor<2,dealdim> E_LinU = 0.5 * (transpose(F_LinU) * F + transpose(F) * F_LinU);

		  const double tr_E_LinU = Structure_Terms_in_ALE
		    ::get_tr_E_LinU<dealdim> (q_point,_ugrads, phi_grads_u[i]);


		  // STVK
		  // piola-kirchhoff stress structure STVK linearized in all directions
		  // J * (1/J*F*(lambda*tr_E*I + 2*mu*E)*F^T) * F^{-T} --> Linearization
		  Tensor<2,dealdim> piola_kirchhoff_stress_structure_STVK_LinALL;
		  piola_kirchhoff_stress_structure_STVK_LinALL = lame_coefficient_lambda *
		    (F_LinU * tr_E * Identity + F * tr_E_LinU * Identity)
		    + 2 * lame_coefficient_mu * (F_LinU * E + F * E_LinU);

		  for(unsigned int j=0; j<n_dofs_per_cell; j++)
		    {
		      // STVK, explicit
                      local_entry_matrix(j, i) += scale * (scalar_product(
                          piola_kirchhoff_stress_structure_STVK_LinALL,
                          phi_grads_v[j])) * state_fe_values.JxW(q_point);

                      // with common condition
                      local_entry_matrix(j, i) += scale
                          * (-phi_v[i] * phi_u[j]) * state_fe_values.JxW(
                          q_point);

                      local_entry_matrix(j, i) += scale_ico * (phi_p[i]
                          * phi_p[j]) * state_fe_values.JxW(q_point);
		    }
		}
	    }
	}  // end material_id = 1
    }


    void CellRightHandSide(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& /*cdc*/,
                           dealii::Vector<double> &local_cell_vector __attribute__((unused)),
                           double scale __attribute__((unused)))
    {
      assert(this->_problem_type == "state");
    }


    void CellTimeEquation (const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& /*cdc*/,
                           dealii::Vector<double> &local_cell_vector __attribute__((unused)),
				   double scale __attribute__((unused)))
    {
      assert(this->_problem_type == "state");
    }

  void CellTimeEquationExplicit (const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
                           dealii::Vector<double> &local_cell_vector __attribute__((unused)),
				   double scale __attribute__((unused)))
    {
      assert(this->_problem_type == "state");

      const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
      unsigned int n_q_points = cdc.GetNQPoints();
      unsigned int material_id = cdc.GetMaterialId(); 

      // old Newton step solution values and gradients
      _uvalues.resize(n_q_points,Vector<double>(5));
      _ugrads.resize(n_q_points,vector<Tensor<1,2> >(5));

      cdc.GetValuesState("last_newton_solution",_uvalues);
      cdc.GetGradsState("last_newton_solution",_ugrads);

      // old timestep solution values and gradients
      _last_timestep_uvalues.resize(n_q_points,Vector<double>(5));
      _last_timestep_ugrads.resize(n_q_points,vector<Tensor<1,2> >(5));

      cdc.GetValuesState("last_time_solution",_last_timestep_uvalues);
      cdc.GetGradsState("last_time_solution",_last_timestep_ugrads);

      const FEValuesExtractors::Vector velocities (0);
      const FEValuesExtractors::Vector displacements (2);

      if (material_id == 0)
	{
	  for (unsigned int q_point=0;q_point<n_q_points;q_point++)
	    {
	      const Tensor<1,dealdim> v = ALE_Transformations
		::get_v<dealdim> (q_point, _uvalues);

	      const Tensor<2,dealdim> F = ALE_Transformations
		::get_F<dealdim> (q_point, _ugrads);

	      const double J = ALE_Transformations
		::get_J<dealdim> (F);

	      const Tensor<1,dealdim> last_timestep_v = ALE_Transformations
		::get_v<dealdim> (q_point, _last_timestep_uvalues);

	      const Tensor<2,dealdim> last_timestep_F = ALE_Transformations
		::get_F<dealdim> (q_point, _last_timestep_ugrads);

	      const double last_timestep_J = ALE_Transformations
		::get_J<dealdim> (last_timestep_F);

	      const Tensor<2,dealdim> grad_v = ALE_Transformations
		::get_grad_v<dealdim> (q_point, _ugrads);

//	      const Tensor<2,dealdim> grad_u = ALE_Transformations
//		::get_grad_u<dealdim> (q_point, _ugrads);

//	      const Tensor<2,dealdim> grad_v_T = ALE_Transformations
//		::get_grad_v_T<dealdim> (grad_v);

	      const Tensor<1,dealdim> u = ALE_Transformations
		::get_u<dealdim> (q_point, _uvalues);



	      const Tensor<2,dealdim> F_Inverse = ALE_Transformations
		::get_F_Inverse<dealdim> (F);

	      const Tensor<1,dealdim> last_timestep_u = ALE_Transformations
		::get_u<dealdim> (q_point, _last_timestep_uvalues);

	      // convection term with u
	      const Tensor<1,dealdim> convection_fluid_with_u = density_fluid * J * (grad_v * F_Inverse * u);

	      // convection term with old timestep u
	      const Tensor<1,dealdim> convection_fluid_with_last_timestep_u = density_fluid * J * (grad_v * F_Inverse * last_timestep_u);

	      for (unsigned int i=0;i<n_dofs_per_cell;i++)
		{
		  // Fluid, CellTimeEquation, explicit
		  const Tensor<1,2> phi_i_v = state_fe_values[velocities].value (i, q_point);
		  local_cell_vector(i) +=  scale * (density_fluid * (J + last_timestep_J)/2.0 * (v - last_timestep_v)  * phi_i_v
						    - convection_fluid_with_u * phi_i_v
						    + convection_fluid_with_last_timestep_u * phi_i_v
						    ) * state_fe_values.JxW(q_point);


		}
	    }
	}
      else  if (material_id == 1)
	{
	  for (unsigned int q_point=0;q_point<n_q_points;q_point++)
	    {
	      const Tensor<1,dealdim> v = ALE_Transformations::get_v<dealdim> (q_point, _uvalues);
	      const Tensor<1,dealdim> u = ALE_Transformations::get_u<dealdim> (q_point, _uvalues);

	      const Tensor<2,dealdim> F = ALE_Transformations::get_F<dealdim> (q_point, _ugrads);
	      const double J = ALE_Transformations::get_J<dealdim> (F);

	      const Tensor<1,dealdim> last_timestep_v = ALE_Transformations::get_v<dealdim> (q_point, _last_timestep_uvalues);
	      const Tensor<2,dealdim> last_timestep_F = ALE_Transformations::get_F<dealdim> (q_point, _last_timestep_ugrads);
	      const double last_timestep_J = ALE_Transformations::get_J<dealdim> (last_timestep_F);

	      const Tensor<1,dealdim> last_timestep_u = ALE_Transformations::get_u<dealdim> (q_point, _last_timestep_uvalues);

	      for (unsigned int i=0;i<n_dofs_per_cell;i++)
		{
		  // Structure, CellTimeEquation, explicit
		  const Tensor<1,2> phi_i_v = state_fe_values[velocities].value (i, q_point);
		  const Tensor<1,2> phi_i_u = state_fe_values[displacements].value (i, q_point);
		  local_cell_vector(i) +=  scale * (density_structure * (J + last_timestep_J)/2.0 * (v - last_timestep_v) * phi_i_v
						    + (u - last_timestep_u) * phi_i_u
						    ) * state_fe_values.JxW(q_point);
		}
	    }
	}
    }


  void CellTimeMatrix(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& /*cdc*/,
			FullMatrix<double> &/*local_entry_matrix*/)
    {
      assert(this->_problem_type == "state");
    }

    void CellTimeMatrixExplicit(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
				FullMatrix<double> &local_entry_matrix)
    {
      assert(this->_problem_type == "state");

      const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
      unsigned int n_q_points = cdc.GetNQPoints();
      unsigned int material_id = cdc.GetMaterialId(); 
      //double cell_diameter = cdc.GetCellDiameter();


      // old Newton step solution values and gradients
      _uvalues.resize(n_q_points,Vector<double>(5));
      _ugrads.resize(n_q_points,vector<Tensor<1,2> >(5));

      cdc.GetValuesState("last_newton_solution",_uvalues);
      cdc.GetGradsState("last_newton_solution",_ugrads);

      // old timestep solution values and gradients
      _last_timestep_uvalues.resize(n_q_points,Vector<double>(5));
      _last_timestep_ugrads.resize(n_q_points,vector<Tensor<1,2> >(5));

      cdc.GetValuesState("last_time_solution",_last_timestep_uvalues);
      cdc.GetGradsState("last_time_solution",_last_timestep_ugrads);


      const FEValuesExtractors::Vector velocities (0);
      const FEValuesExtractors::Vector displacements (2);

      std::vector<Tensor<1,dealdim> >     phi_v (n_dofs_per_cell);
      std::vector<Tensor<2,dealdim> >     phi_grads_v (n_dofs_per_cell);
      std::vector<Tensor<1,dealdim> >     phi_u (n_dofs_per_cell);
      std::vector<Tensor<2,dealdim> >     phi_grads_u (n_dofs_per_cell);

      if (material_id == 0)
	{
	  for(unsigned int q_point = 0; q_point < n_q_points; q_point++)
	    {
	      for(unsigned int k = 0; k < n_dofs_per_cell; k++)
		{
		  phi_v[k]         = state_fe_values[velocities].value (k, q_point);
		  phi_grads_v[k]   = state_fe_values[velocities].gradient (k, q_point);
		  phi_u[k]         = state_fe_values[displacements].value (k, q_point);
		  phi_grads_u[k]   = state_fe_values[displacements].gradient (k, q_point);
		}

	      const Tensor<1,dealdim> v = ALE_Transformations::get_v<dealdim> (q_point, _uvalues);
	      const Tensor<2,dealdim> F = ALE_Transformations::get_F<dealdim> (q_point, _ugrads);
	      const Tensor<2,dealdim> F_Inverse = ALE_Transformations::get_F_Inverse<dealdim> (F);
	      const double J = ALE_Transformations::get_J<dealdim> (F);

	      const Tensor<1,dealdim> last_timestep_v = ALE_Transformations::get_v<dealdim> (q_point, _last_timestep_uvalues);
	      const Tensor<2,dealdim> last_timestep_F = ALE_Transformations::get_F<dealdim> (q_point, _last_timestep_ugrads);
	      const double last_timestep_J = ALE_Transformations::get_J<dealdim> (last_timestep_F);

	      const Tensor<1,dealdim> u = ALE_Transformations::get_u<dealdim> (q_point, _uvalues);
	      const Tensor<1,dealdim> last_timestep_u = ALE_Transformations::get_u<dealdim> (q_point, _last_timestep_uvalues);

	      for(unsigned int i = 0; i < n_dofs_per_cell; i++)
		{
		  const Tensor<2,dealdim> grad_v = ALE_Transformations::get_grad_v<dealdim> (q_point, _ugrads);
		  const double J_LinU =  ALE_Transformations
		    ::get_J_LinU<dealdim> (q_point, _ugrads, phi_grads_u[i]);

		  const Tensor<2,dealdim> F_Inverse_LinU = ALE_Transformations
		    ::get_F_Inverse_LinU (phi_grads_u[i],J,
					  J_LinU,q_point,_ugrads
					  );


		  const Tensor<1,dealdim> accelaration_term_LinAll = NSE_in_ALE
		    ::get_accelaration_term_LinAll (phi_v[i],
						    v, last_timestep_v,
						    J_LinU,
						    J, last_timestep_J,
						    density_fluid);

		  const Tensor<1,dealdim> convection_fluid_u_LinAll_short =  NSE_in_ALE
		    ::get_Convection_u_LinAll_short<dealdim> (phi_grads_v[i],phi_u[i], J,J_LinU,
							      F_Inverse,F_Inverse_LinU, u, grad_v, density_fluid);

		  const Tensor<1,dealdim> convection_fluid_u_old_LinAll_short = NSE_in_ALE
		    ::get_Convection_u_old_LinAll_short<dealdim> (phi_grads_v[i],J, J_LinU, F_Inverse,
								  F_Inverse_LinU,last_timestep_u, grad_v,  density_fluid);

		  for(unsigned int j = 0; j < n_dofs_per_cell; j++)
		    {
		      // Fluid, CellTimeMatrix, explicit
		      local_entry_matrix(j,i) += (accelaration_term_LinAll * phi_v[j]
						  - convection_fluid_u_LinAll_short * phi_v[j]
						  + convection_fluid_u_old_LinAll_short * phi_v[j]
						  ) * state_fe_values.JxW(q_point);

		    }
		}
	    }
	}
      else if (material_id == 1)
	{
	  for(unsigned int q_point = 0; q_point < n_q_points; q_point++)
	    {
	      for(unsigned int k = 0; k < n_dofs_per_cell; k++)
		{
		  phi_v[k]         = state_fe_values[velocities].value (k, q_point);
		  phi_u[k]         = state_fe_values[displacements].value (k, q_point);
		  phi_grads_u[k]   = state_fe_values[displacements].gradient (k, q_point);
		}

	      const Tensor<1,dealdim> v = ALE_Transformations::get_v<dealdim> (q_point, _uvalues);
	      const Tensor<2,dealdim> F = ALE_Transformations::get_F<dealdim> (q_point, _ugrads);
	      const double J = ALE_Transformations::get_J<dealdim> (F);

	      const Tensor<1,dealdim> last_timestep_v = ALE_Transformations::get_v<dealdim> (q_point, _last_timestep_uvalues);
	      const Tensor<2,dealdim> last_timestep_F = ALE_Transformations::get_F<dealdim> (q_point, _last_timestep_ugrads);
	      const double last_timestep_J = ALE_Transformations::get_J<dealdim> (last_timestep_F);

	      for(unsigned int i = 0; i < n_dofs_per_cell; i++)
		{

		  const double J_LinU =  ALE_Transformations
		    ::get_J_LinU<dealdim> (q_point, _ugrads, phi_grads_u[i]);

		  const Tensor<1,dealdim> accelaration_term_LinAll = NSE_in_ALE
		    ::get_accelaration_term_LinAll (phi_v[i],
						    v, last_timestep_v,
						    J_LinU,
						    J, last_timestep_J,
						    density_structure);


		  for(unsigned int j = 0; j < n_dofs_per_cell; j++)
		    {
		      // Structure, CellTimeMatrix, explicit
		      local_entry_matrix(j,i) += (accelaration_term_LinAll * phi_v[j]
						  + phi_u[i] * phi_u[j]
						  ) * state_fe_values.JxW(q_point);

		    }
		}
	    }
	}

    }


     void FaceEquation (const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
			dealii::Vector<double> &local_cell_vector,
			double scale)
     {
       assert(this->_problem_type == "state");

       const DOpEWrapper::FEFaceValues<dealdim> & state_fe_face_values = fdc.GetFEFaceValuesState();
       unsigned int n_dofs_per_cell = fdc.GetNDoFsPerCell();
       unsigned int n_q_points = fdc.GetNQPoints();
       //unsigned int color = fdc.GetBoundaryIndicator();
       unsigned int material_id = fdc.GetMaterialId();
       unsigned int material_id_neighbor = fdc.GetNbrMaterialId();
       //bool at_boundary = fdc.GetIsAtBoundary();
       double cell_diameter = fdc.GetCellDiameter();

	if (material_id != material_id_neighbor)
	  {
	      vector<Vector<double> > _ufacevalues;
	      vector<vector<Tensor<1,dealdim> > > _ufacegrads;

	      _ufacevalues.resize(n_q_points,Vector<double>(5));
	      _ufacegrads.resize(n_q_points,vector<Tensor<1,2> >(5));

	      fdc.GetFaceValuesState("last_newton_solution",_ufacevalues);
	      fdc.GetFaceGradsState("last_newton_solution",_ufacegrads);

	      const FEValuesExtractors::Vector displacements (2);


	      for (unsigned int q_point=0;q_point<n_q_points;q_point++)
		{
		  // variable for mesh-dependent alpha_u
		  distance_to_interface = std::sqrt((state_fe_face_values.quadrature_point(q_point)(0)-0.6)*
						    (state_fe_face_values.quadrature_point(q_point)(0)-0.6)+
						    (state_fe_face_values.quadrature_point(q_point)(1)-0.2)*
						    (state_fe_face_values.quadrature_point(q_point)(1)-0.2));

		  // mesh-dependent alpha_u. Here the variable has name tmp_structure_continuation_E.
		  tmp_structure_continuation_E = structure_continuation_E
		    + scale_distance_to_interface*std::exp(-scale_d_exp*std::abs(distance_to_interface));

		  const Tensor<2,dealdim> grad_u = ALE_Transformations
		    ::get_grad_u<dealdim> (q_point, _ufacegrads);

		  const Tensor<1,dealdim> neumann_value
		    = (grad_u * state_fe_face_values.normal_vector(q_point));

//		  const Tensor<1,dealdim> v = ALE_Transformations
//		    ::get_v<dealdim> (q_point, _ufacevalues);

		  for (unsigned int i=0;i<n_dofs_per_cell;i++)
		    {
		      const Tensor<1,2> phi_i_u = state_fe_face_values[displacements].value (i, q_point);

		      local_cell_vector(i) -= 0.0 * (scale
						     * tmp_structure_continuation_E
						     //* alpha_u
						     // not to add when using Nitsche Trick
						     * cell_diameter * cell_diameter
						     * neumann_value * phi_i_u)  * state_fe_face_values.JxW(q_point);

		      // with Nitsche trick
		//local_cell_vector(i) -= 1.0 * (scale * 1000.0/cell_diameter * v * phi_i_u)
		//	* state_fe_face_values.JxW(q_point);

		    }
		}
	  }
     }

     void FaceMatrix (const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
		      dealii::FullMatrix<double> &local_entry_matrix)
     {
       assert(this->_problem_type == "state");

       const DOpEWrapper::FEFaceValues<dealdim> & state_fe_face_values = fdc.GetFEFaceValuesState();
       unsigned int n_dofs_per_cell = fdc.GetNDoFsPerCell();
       unsigned int n_q_points = fdc.GetNQPoints();
       //unsigned int color = fdc.GetBoundaryIndicator();
       unsigned int material_id = fdc.GetMaterialId();
       unsigned int material_id_neighbor = fdc.GetNbrMaterialId();
       //bool at_boundary = fdc.GetIsAtBoundary();
       double cell_diameter = fdc.GetCellDiameter(); 

	if (material_id != material_id_neighbor)
	  {
	    std::vector<Tensor<1,dealdim> >     phi_v (n_dofs_per_cell);
	    std::vector<Tensor<1,dealdim> >     phi_u (n_dofs_per_cell);
	    std::vector<Tensor<2,dealdim> >     phi_grads_u (n_dofs_per_cell);

	    const FEValuesExtractors::Vector velocities (0);
	    const FEValuesExtractors::Vector displacements (2);

	    for (unsigned int q_point=0;q_point<n_q_points;q_point++)
	      {
		// variable for mesh-dependent alpha_u
		distance_to_interface = std::sqrt((state_fe_face_values.quadrature_point(q_point)(0)-0.6)*
						  (state_fe_face_values.quadrature_point(q_point)(0)-0.6)+
						  (state_fe_face_values.quadrature_point(q_point)(1)-0.2)*
						  (state_fe_face_values.quadrature_point(q_point)(1)-0.2));

		// mesh-dependent alpha_u. Here the variable has name tmp_structure_continuation_E.
		tmp_structure_continuation_E = structure_continuation_E
		  + scale_distance_to_interface*std::exp(-scale_d_exp*std::abs(distance_to_interface));

		for (unsigned int k=0; k<n_dofs_per_cell; ++k)
		  {
		    phi_v[k]         = state_fe_face_values[velocities].value (k, q_point);
		    phi_u[k]         = state_fe_face_values[displacements].value (k, q_point);
		    phi_grads_u[k]   = state_fe_face_values[displacements].gradient (k, q_point);
		  }
		 for(unsigned int i = 0; i < n_dofs_per_cell; i++)
		   {
		     const Tensor<1,dealdim> neumann_value
		       = (phi_grads_u[i] * state_fe_face_values.normal_vector(q_point));

		     for(unsigned int j = 0; j < n_dofs_per_cell; j++)
		       {
			 // Fluid
			 local_entry_matrix(j,i) -=  0.0 * (tmp_structure_continuation_E
							    //alpha_u
							    // not to add when using Nitsche Trick
							    * cell_diameter * cell_diameter
							    * neumann_value *  phi_u[j])  * state_fe_face_values.JxW(q_point);

			 // with Nitsche trick
			//local_entry_matrix(j,i) -=  1.0 * (1000.0/cell_diameter
			//				    * phi_v[i] * phi_u[j])* state_fe_face_values.JxW(q_point);


		       }
		   }


		}

	  }
     }


     void FaceRightHandSide (const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& /*fdc*/,
			     dealii::Vector<double> &/*local_cell_vector*/,
			     double /*scale*/)
     {
       //assert(this->_problem_type == "state");
     }


    // Values for boundary integrals
    void BoundaryEquation (const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
			   dealii::Vector<double> &local_cell_vector,
			   double scale)
    {

      assert(this->_problem_type == "state");

      const DOpEWrapper::FEFaceValues<dealdim> & state_fe_face_values = fdc.GetFEFaceValuesState();
      unsigned int n_dofs_per_cell = fdc.GetNDoFsPerCell();
      unsigned int n_q_points = fdc.GetNQPoints();
      unsigned int color = fdc.GetBoundaryIndicator();

      // do-nothing applied on outflow boundary
      if (color == 1)
	{
	  // old Newton step face_solution values and gradients
	  _ufacevalues.resize(n_q_points,Vector<double>(5));
	  _ufacegrads.resize(n_q_points,vector<Tensor<1,2> >(5));

	  fdc.GetFaceValuesState("last_newton_solution",_ufacevalues);
	  fdc.GetFaceGradsState("last_newton_solution",_ufacegrads);

	  const FEValuesExtractors::Vector velocities (0);

	  for (unsigned int q_point=0;q_point<n_q_points;q_point++)
	    {
	      const Tensor<2,dealdim> grad_v = ALE_Transformations
		::get_grad_v<dealdim> (q_point, _ufacegrads);

	      const Tensor<2,dealdim> F = ALE_Transformations
		::get_F<dealdim> (q_point, _ufacegrads);

	      const Tensor<2,dealdim> F_Inverse = ALE_Transformations
		::get_F_Inverse<dealdim> (F);

	      const Tensor<2,dealdim> F_Inverse_T = ALE_Transformations
		::get_F_Inverse_T<dealdim> (F_Inverse);

	      const double J = ALE_Transformations
		::get_J<dealdim> (F);

	      const Tensor<2,dealdim> sigma_ALE_tilde = (density_fluid * viscosity * F_Inverse_T * transpose(grad_v));

	      // Neumann boundary integral
	      const Tensor<2,dealdim> stress_fluid_transposed_part = (J * sigma_ALE_tilde * F_Inverse_T);

	      const Tensor<1,dealdim> neumann_value
		= (stress_fluid_transposed_part * state_fe_face_values.normal_vector(q_point));

	      for (unsigned int i=0;i<n_dofs_per_cell;i++)
		{
		  const Tensor<1,2> phi_i_v = state_fe_face_values[velocities].value (i, q_point);

		  local_cell_vector(i) -= scale * neumann_value * phi_i_v  * state_fe_face_values.JxW(q_point);
		}
	    }
	}

    }

    void BoundaryMatrix (const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
			 dealii::FullMatrix<double> &local_entry_matrix)
    {
      assert(this->_problem_type == "state");

      const DOpEWrapper::FEFaceValues<dealdim> & state_fe_face_values = fdc.GetFEFaceValuesState();
      unsigned int n_dofs_per_cell = fdc.GetNDoFsPerCell();
      unsigned int n_q_points = fdc.GetNQPoints();
      unsigned int color = fdc.GetBoundaryIndicator();

      // do-nothing applied on outflow boundary
      if (color == 1)
	{
	  // old Newton step face_solution values and gradients
	  _ufacevalues.resize(n_q_points,Vector<double>(5));
	  _ufacegrads.resize(n_q_points,vector<Tensor<1,2> >(5));

	  fdc.GetFaceValuesState("last_newton_solution",_ufacevalues);
	  fdc.GetFaceGradsState("last_newton_solution",_ufacegrads);

	  std::vector<Tensor<1,dealdim> >     phi_v (n_dofs_per_cell);
	  std::vector<Tensor<2,dealdim> >     phi_grads_v (n_dofs_per_cell);
	  std::vector<Tensor<2,dealdim> >     phi_grads_u (n_dofs_per_cell);

	  const FEValuesExtractors::Vector velocities (0);
	  const FEValuesExtractors::Vector displacements (2);

	  for (unsigned int q_point=0;q_point<n_q_points;q_point++)
	    {
	      for (unsigned int k=0; k<n_dofs_per_cell; ++k)
		{
		  phi_v[k]         = state_fe_face_values[velocities].value (k, q_point);
		  phi_grads_v[k]   = state_fe_face_values[velocities].gradient (k, q_point);
		  phi_grads_u[k]   = state_fe_face_values[displacements].gradient (k, q_point);
		}

//	      const Tensor<2,dealdim> pI = ALE_Transformations
//		::get_pI<dealdim> (q_point, _ufacevalues);

//	      const Tensor<1,dealdim> v = ALE_Transformations
//		::get_v<dealdim> (q_point, _ufacevalues);

	      const Tensor<2,dealdim>  grad_v = ALE_Transformations
		::get_grad_v<dealdim> (q_point, _ufacegrads);

//	      const Tensor<2,dealdim> grad_v_T = ALE_Transformations
//		::get_grad_v_T<dealdim> (grad_v);

	      const Tensor<2,dealdim> F = ALE_Transformations
		::get_F<dealdim> (q_point, _ufacegrads);

	      const Tensor<2,dealdim> F_Inverse = ALE_Transformations
		::get_F_Inverse<dealdim> (F);

//	      const Tensor<2,dealdim> F_Inverse_T = ALE_Transformations
//		::get_F_Inverse_T<dealdim> (F_Inverse);

	      const double J = ALE_Transformations
		::get_J<dealdim> (F);

	      for(unsigned int i = 0; i < n_dofs_per_cell; i++)
		{
		  const Tensor<2,dealdim> grad_v_LinV = ALE_Transformations
		    ::get_grad_v_LinV<dealdim> (phi_grads_v[i]);

		  const double J_LinU = ALE_Transformations
		    ::get_J_LinU<dealdim> (q_point, _ufacegrads, phi_grads_u[i]);

//		  const double J_Inverse_LinU = ALE_Transformations
//		    ::get_J_Inverse_LinU<dealdim> (J, J_LinU);

		  const Tensor<2,dealdim> J_F_Inverse_T_LinU = ALE_Transformations
		    ::get_J_F_Inverse_T_LinU<dealdim> (phi_grads_u[i]);

		  const Tensor<2,dealdim> F_Inverse_LinU = ALE_Transformations
		    ::get_F_Inverse_LinU (phi_grads_u[i],J,J_LinU,q_point,_ufacegrads);

		  const Tensor<2,dealdim> stress_fluid_ALE_3rd_term_LinAll =  NSE_in_ALE
		    ::get_stress_fluid_ALE_3rd_term_LinAll_short<dealdim> (F_Inverse, F_Inverse_LinU,
									   grad_v,grad_v_LinV,
									   viscosity,density_fluid,
									   J,J_F_Inverse_T_LinU);

		  const Tensor<1,dealdim> neumann_value
		    = (stress_fluid_ALE_3rd_term_LinAll * state_fe_face_values.normal_vector(q_point));

		  for(unsigned int j = 0; j < n_dofs_per_cell; j++)
		    {
		      // Fluid
		      local_entry_matrix(j,i) -=  neumann_value *  phi_v[j]  * state_fe_face_values.JxW(q_point);
		    }
		}
	    }
	}
    }

    void BoundaryRightHandSide (const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& /*fdc*/,
				dealii::Vector<double> &/*local_cell_vector*/,
				double /*scale*/)
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
      return 3;
    }

    std::vector<unsigned int>& GetControlBlockComponent(){ return _block_components; }
    const std::vector<unsigned int>& GetControlBlockComponent() const{ return _block_components; }
    std::vector<unsigned int>& GetStateBlockComponent(){ return _state_block_components; }
    const std::vector<unsigned int>& GetStateBlockComponent() const{ return _state_block_components; }

  protected:
    inline void GetValues(const DOpEWrapper::FEValues<dealdim>& fe_values,
			  const map<string, const BlockVector<double>* >& domain_values,string name,
			  vector<Vector<double> >& values)
    {
      typename map<string, const BlockVector<double>* >::const_iterator it = domain_values.find(name);
      if(it == domain_values.end())
	{
	  throw DOpEException("Did not find " + name,"LocalPDE::GetValues");
	}
      fe_values.get_function_values(*(it->second),values);
    }

    inline void GetGrads(const DOpEWrapper::FEValues<dealdim>& fe_values,
			 const map<string, const BlockVector<double>* >& domain_values,string name,
			 vector<vector<Tensor<1,dealdim> > >& values)
    {
      typename map<string, const BlockVector<double>* >::const_iterator it = domain_values.find(name);
      if(it == domain_values.end())
	{
	  throw DOpEException("Did not find " + name,"LocalPDE::GetGrads");
	}
      fe_values.get_function_grads(*(it->second),values);
    }


    inline void GetFaceValues(const DOpEWrapper::FEFaceValues<dealdim>& fe_face_values,
			      const map<string, const BlockVector<double>* >& domain_values, string name,
			      vector<Vector<double> >& values)
    {
      typename map<string, const BlockVector<double>* >::const_iterator it = domain_values.find(name);
      if(it == domain_values.end())
	{
	  throw DOpEException("Did not find " + name,"LocalPDE::GetValues");
	}
      fe_face_values.get_function_values(*(it->second),values);
    }

    inline void GetFaceGrads(const DOpEWrapper::FEFaceValues<dealdim>& fe_face_values,
			     const map<string, const BlockVector<double>* >& domain_values, string name,
			     vector<vector<Tensor<1,dealdim> > >& values)
    {
      typename map<string, const BlockVector<double>* >::const_iterator it = domain_values.find(name);
      if(it == domain_values.end())
	{
	  throw DOpEException("Did not find " + name,"LocalPDE::GetGrads");
	}
      fe_face_values.get_function_grads(*(it->second),values);
    }


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

    //last timestep solution values
    vector<Vector<double> > _last_timestep_uvalues;
    vector<vector<Tensor<1,dealdim> > > _last_timestep_ugrads;

    // face values
    vector<double> _qfacevalues;
    vector<Vector<double> > _ffacevalues;
    vector<Vector<double> > _ufacevalues;
    vector<vector<Tensor<1,dealdim> > > _ufacegrads;

    vector<Vector<double> > _last_timestep_ufacevalues;
    vector<vector<Tensor<1,dealdim> > > _last_timestep_ufacegrads;


    vector<unsigned int> _state_block_components;
    vector<unsigned int> _block_components;
    double _alpha, _cell_diameter;

    // material variables
    double density_fluid, density_structure, viscosity, alpha_u,
      lame_coefficient_mu, poisson_ratio_nu, lame_coefficient_lambda;

    // variables for mesh-dependent alpha_u
    double structure_continuation_E;
    double scale_distance_to_interface;
    double scale_d_exp;
    double tmp_structure_continuation_E, distance_to_interface;
    double structure_continuation_mu, structure_continuation_lambda,
      structure_continuation_nus;


  };
#endif
