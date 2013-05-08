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

#ifndef _LOCALFunctionalS_
#define _LOCALFunctionalS_

#include "pdeinterface.h"
//#include "functionalinterface.h"

using namespace std;
using namespace dealii;
using namespace DOpE;


/****************************************************************************************/

template<typename VECTOR, int dopedim, int dealdim>
  class LocalPointFunctionalPressure : public FunctionalInterface<CellDataContainer,FaceDataContainer,dealii::DoFHandler, VECTOR, dopedim,dealdim>
  {
  public:

    double PointValue(const DOpEWrapper::DoFHandler<dopedim, dealii::DoFHandler > & /*control_dof_handler*/,
		    const DOpEWrapper::DoFHandler<dealdim, dealii::DoFHandler > & state_dof_handler,
		      const std::map<std::string, const dealii::Vector<double>* > &/*param_values*/,
		    const std::map<std::string, const VECTOR* > &domain_values)
  {
    Point<2> p1(0.15,0.2);
    Point<2> p2(0.25,0.2);

    typename map<string, const VECTOR* >::const_iterator it = domain_values.find("state");
    Vector<double> tmp_vector(5);


    VectorTools::point_value (state_dof_handler, *(it->second), p1, tmp_vector);
    double p1_value = tmp_vector(2);
    tmp_vector = 0;
    VectorTools::point_value (state_dof_handler, *(it->second), p2, tmp_vector);
    double p2_value = tmp_vector(2);

    // pressure analysis
    return (p1_value - p2_value);

  }

  string GetType() const
  {
    return "point";
  }
  string GetName() const
  {
    return "Pressure difference";
  }

  };

/****************************************************************************************/
template<typename VECTOR, int dopedim, int dealdim>
  class LocalPointFunctionalDeflectionX : public FunctionalInterface<CellDataContainer,FaceDataContainer,dealii::DoFHandler, VECTOR, dopedim,dealdim>
  {
  public:

    double PointValue(const DOpEWrapper::DoFHandler<dopedim, dealii::DoFHandler > & /*control_dof_handler*/,
			      const DOpEWrapper::DoFHandler<dealdim, dealii::DoFHandler > &state_dof_handler,
		      const std::map<std::string, const dealii::Vector<double>* > &/*param_values*/,
			      const std::map<std::string, const VECTOR* > &domain_values)
  {
    Point<2> p1(0.6,0.2);

    typename map<string, const VECTOR* >::const_iterator it = domain_values.find("state");
    Vector<double> tmp_vector(5);


    VectorTools::point_value (state_dof_handler, *(it->second), p1, tmp_vector);
    double x = tmp_vector(3);

    // Deflection X
    return x;

  }

  string GetType() const
  {
    return "point";
  }
  string GetName() const
  {
    return "Deflection_X";
  }

  };

/****************************************************************************************/
template<typename VECTOR, int dopedim, int dealdim>
  class LocalPointFunctionalDeflectionY : public FunctionalInterface<CellDataContainer,FaceDataContainer,dealii::DoFHandler, VECTOR, dopedim,dealdim>
  {
  public:

    double PointValue(const DOpEWrapper::DoFHandler<dopedim, dealii::DoFHandler > & /*control_dof_handler*/,
		    const DOpEWrapper::DoFHandler<dealdim, dealii::DoFHandler > &state_dof_handler,
		      const std::map<std::string, const dealii::Vector<double>* > &/*param_values*/,
		    const std::map<std::string, const VECTOR* > &domain_values)
  {
    Point<2> p1(0.6,0.2);

    typename map<string, const VECTOR* >::const_iterator it = domain_values.find("state");
    Vector<double> tmp_vector(5);

    VectorTools::point_value (state_dof_handler, *(it->second), p1, tmp_vector);
    double y = tmp_vector(4);

    // Delfection Y
    return y;

  }

  string GetType() const
  {
    return "point";
  }
  string GetName() const
  {
    return "Deflection_Y";
  }

  };

// drag
/****************************************************************************************/
template<typename VECTOR, int dopedim, int dealdim>
  class LocalBoundaryFaceFunctionalDrag : public FunctionalInterface<CellDataContainer,FaceDataContainer,dealii::DoFHandler, VECTOR, dopedim,dealdim>
  {
  public:
     static void declare_params(ParameterReader &param_reader)
    {
      param_reader.SetSubsection("Local PDE parameters");
      param_reader.declare_entry("density_fluid", "0.0",
				 Patterns::Double(0));
      param_reader.declare_entry("viscosity", "0.0",
				 Patterns::Double(0));
      param_reader.declare_entry("mu", "0.0",
				 Patterns::Double(0));
      param_reader.declare_entry("poisson_ratio_nu", "0.0",
				 Patterns::Double(0));

    }

     LocalBoundaryFaceFunctionalDrag (ParameterReader &param_reader)
       {
	 param_reader.SetSubsection("Local PDE parameters");
	 _density_fluid = param_reader.get_double ("density_fluid");
	 _viscosity = param_reader.get_double ("viscosity");
	 _lame_coefficient_mu = param_reader.get_double ("mu");
	 _poisson_ratio_nu = param_reader.get_double ("poisson_ratio_nu");

	 _lame_coefficient_lambda =  (2 * _poisson_ratio_nu * _lame_coefficient_mu)/
	   (1.0 - 2 * _poisson_ratio_nu);
       }

     bool HasFaces() const
      {
        return false;
      }


     // compute drag value around cylinder
     double BoundaryValue(const FaceDataContainer<dealii::DoFHandler, VECTOR, dealdim>& fdc)
     {
       const auto & state_fe_face_values = fdc.GetFEFaceValuesState();
       unsigned int n_q_points = fdc.GetNQPoints();
       unsigned int color = fdc.GetBoundaryIndicator();

       Tensor<1,2> drag_lift_value;
       drag_lift_value.clear();
       if (color == 80)
	 {
	  vector<Vector<double> > _ufacevalues;
	  vector<vector<Tensor<1,dealdim> > > _ufacegrads;

	  _ufacevalues.resize(n_q_points,Vector<double>(5));
	  _ufacegrads.resize(n_q_points,vector<Tensor<1,2> >(5));

	  fdc.GetFaceValuesState("state",_ufacevalues);
	  fdc.GetFaceGradsState("state",_ufacegrads);

	  for (unsigned int q_point=0;q_point<n_q_points;q_point++)
	    {
	       Tensor<2,2> pI;
	       pI[0][0] = _ufacevalues[q_point](2);
	       pI[1][1] = _ufacevalues[q_point](2);

	       Tensor<1,2> v;
	       v.clear();
	       v[0] = _ufacevalues[q_point](0);
	       v[1] = _ufacevalues[q_point](1);

	       Tensor<2,2> grad_v;
	       grad_v[0][0] = _ufacegrads[q_point][0][0];
	       grad_v[0][1] = _ufacegrads[q_point][0][1];
	       grad_v[1][0] = _ufacegrads[q_point][1][0];
	       grad_v[1][1] = _ufacegrads[q_point][1][1];

	       Tensor<2,2> F;
	       F[0][0] = 1.0 +  _ufacegrads[q_point][3][0];
	       F[0][1] = _ufacegrads[q_point][3][1];
	       F[1][0] = _ufacegrads[q_point][4][0];
	       F[1][1] = 1.0 + _ufacegrads[q_point][4][1];

	       Tensor<2,2> F_Inverse;
	       F_Inverse = invert (F);

	       Tensor<2,2> F_T;
	       F_T = transpose (F);

	       Tensor<2,2> F_Inverse_T;
	       F_Inverse_T = transpose (F_Inverse);

	       double J;
	       J = determinant (F);


	       // constitutive stress tensors for fluid
	       Tensor<2,2> cauchy_stress_fluid;
	       cauchy_stress_fluid = J * (-pI + _density_fluid * _viscosity *
					  (grad_v * F_Inverse + F_Inverse_T * transpose(grad_v))) * F_Inverse_T;



	      drag_lift_value -=  cauchy_stress_fluid
		* state_fe_face_values.normal_vector(q_point) *
		state_fe_face_values.JxW(q_point);
	    }
	 }
         return 500 * drag_lift_value[0];  // for fluid problems
     }



     UpdateFlags GetFaceUpdateFlags() const
     {
       return update_values | update_quadrature_points |
	 update_gradients | update_normal_vectors;
     }

     string GetType() const
     {
       return "boundary";
     }
     string GetName() const
     {
       return "Drag";
     }

  private:
  double _density_fluid,_viscosity,_lame_coefficient_mu,
    _poisson_ratio_nu, _lame_coefficient_lambda;

  };


 // lift
/****************************************************************************************/
template<typename VECTOR, int dopedim, int dealdim>
  class LocalBoundaryFaceFunctionalLift : public FunctionalInterface<CellDataContainer,FaceDataContainer,dealii::DoFHandler, VECTOR, dopedim,dealdim>
  {
  public:
     static void declare_params(ParameterReader &param_reader)
    {
      param_reader.SetSubsection("Local PDE parameters");
      param_reader.declare_entry("density_fluid", "0.0",
				 Patterns::Double(0));
      param_reader.declare_entry("viscosity", "0.0",
				 Patterns::Double(0));
      param_reader.declare_entry("mu", "0.0",
				 Patterns::Double(0));
      param_reader.declare_entry("poisson_ratio_nu", "0.0",
				 Patterns::Double(0));

    }

     LocalBoundaryFaceFunctionalLift (ParameterReader &param_reader)
       {
	 param_reader.SetSubsection("Local PDE parameters");
	 _density_fluid = param_reader.get_double ("density_fluid");
	 _viscosity = param_reader.get_double ("viscosity");
	 _lame_coefficient_mu = param_reader.get_double ("mu");
	 _poisson_ratio_nu = param_reader.get_double ("poisson_ratio_nu");

	 _lame_coefficient_lambda =  (2 * _poisson_ratio_nu * _lame_coefficient_mu)/
	   (1.0 - 2 * _poisson_ratio_nu);
       }

     bool HasFaces() const
      {
        return false;
      }


     // compute drag value around cylinder
     double BoundaryValue(const FaceDataContainer<dealii::DoFHandler, VECTOR, dealdim>& fdc)
     {
       const auto & state_fe_face_values = fdc.GetFEFaceValuesState();
       
       unsigned int n_q_points = fdc.GetNQPoints();
       unsigned int color = fdc.GetBoundaryIndicator();
       
       Tensor<1,2> drag_lift_value;
       drag_lift_value.clear();
       if (color == 80)
	 {
	  vector<Vector<double> > _ufacevalues;
	  vector<vector<Tensor<1,dealdim> > > _ufacegrads;

	  _ufacevalues.resize(n_q_points,Vector<double>(5));
	  _ufacegrads.resize(n_q_points,vector<Tensor<1,2> >(5));

	  fdc.GetFaceValuesState("state",_ufacevalues);
	  fdc.GetFaceGradsState("state",_ufacegrads);

	  for (unsigned int q_point=0;q_point<n_q_points;q_point++)
	    {
	       Tensor<2,2> pI;
	       pI[0][0] = _ufacevalues[q_point](2);
	       pI[1][1] = _ufacevalues[q_point](2);

	       Tensor<1,2> v;
	       v.clear();
	       v[0] = _ufacevalues[q_point](0);
	       v[1] = _ufacevalues[q_point](1);

	       Tensor<2,2> grad_v;
	       grad_v[0][0] = _ufacegrads[q_point][0][0];
	       grad_v[0][1] = _ufacegrads[q_point][0][1];
	       grad_v[1][0] = _ufacegrads[q_point][1][0];
	       grad_v[1][1] = _ufacegrads[q_point][1][1];

	       Tensor<2,2> F;
	       F[0][0] = 1.0 +  _ufacegrads[q_point][3][0];
	       F[0][1] = _ufacegrads[q_point][3][1];
	       F[1][0] = _ufacegrads[q_point][4][0];
	       F[1][1] = 1.0 + _ufacegrads[q_point][4][1];

	       Tensor<2,2> F_Inverse;
	       F_Inverse = invert (F);

	       Tensor<2,2> F_T;
	       F_T = transpose (F);

	       Tensor<2,2> F_Inverse_T;
	       F_Inverse_T = transpose (F_Inverse);

	       double J;
	       J = determinant (F);


	       // constitutive stress tensors for fluid
	       Tensor<2,2> cauchy_stress_fluid;
	       cauchy_stress_fluid = J * (-pI + _density_fluid * _viscosity *
					  (grad_v * F_Inverse + F_Inverse_T * transpose(grad_v))) * F_Inverse_T;



	      drag_lift_value -=  cauchy_stress_fluid
		* state_fe_face_values.normal_vector(q_point) *
		state_fe_face_values.JxW(q_point);
	    }

	 }
       return 500 * drag_lift_value[1];  // for fluid problems
     }

     /*
       // compute drag value at interface
   double FaceValue(const DOpEWrapper::FEFaceValues<dopedim> &control_fe_face_values,
 		      const auto &state_fe_face_values,
 		      const std::map<std::string, const dealii::Vector<double>* > &param_values,
 		      const std::map<std::string, const VECTOR* > &domain_values,
 		      unsigned int n_q_points,
 		      unsigned int material_id,
 		      unsigned int material_id_neighbor,
 		      double cell_diameter,
 		      bool at_boundary)
   {
     Tensor<1,2> drag_lift_value;
     drag_lift_value.clear();
     if (material_id == 0)
       {

 	    if ((material_id != material_id_neighbor) && (!at_boundary))
 	      {
		//	std::cout << at_boundary << std::endl;


 		vector<Vector<double> > _ufacevalues;
 		vector<vector<Tensor<1,dealdim> > > _ufacegrads;

 		_ufacevalues.resize(n_q_points,Vector<double>(5));
 		_ufacegrads.resize(n_q_points,vector<Tensor<1,2> >(5));

 		GetFaceValues(state_fe_face_values,domain_values,"state",_ufacevalues);
 		GetFaceGrads(state_fe_face_values,domain_values,"state",_ufacegrads);

 		for (unsigned int q_point=0;q_point<n_q_points;q_point++)
 		  {
 		    Tensor<2,2> pI;
 		    pI[0][0] = _ufacevalues[q_point](2);
 		    pI[1][1] = _ufacevalues[q_point](2);

 		    Tensor<1,2> v;
 		    v.clear();
 		    v[0] = _ufacevalues[q_point](0);
 		    v[1] = _ufacevalues[q_point](1);

 		    Tensor<2,2> grad_v;
 		    grad_v[0][0] = _ufacegrads[q_point][0][0];
 		    grad_v[0][1] = _ufacegrads[q_point][0][1];
 		    grad_v[1][0] = _ufacegrads[q_point][1][0];
 		    grad_v[1][1] = _ufacegrads[q_point][1][1];

 		    Tensor<2,2> F;
 		    F[0][0] = 1.0 +  _ufacegrads[q_point][3][0];
 		    F[0][1] = _ufacegrads[q_point][3][1];
 		    F[1][0] = _ufacegrads[q_point][4][0];
 		    F[1][1] = 1.0 + _ufacegrads[q_point][4][1];

 		    Tensor<2,2> F_Inverse;
 		    F_Inverse = invert (F);

 		    Tensor<2,2> F_T;
 		    F_T = transpose (F);

 		    Tensor<2,2> F_Inverse_T;
 		    F_Inverse_T = transpose (F_Inverse);

 		    double J;
 		    J = determinant (F);


 		    // constitutive stress tensors for fluid
 		    Tensor<2,2> cauchy_stress_fluid;
 		    cauchy_stress_fluid = J * (-pI + _density_fluid * _viscosity *
 					       (grad_v * F_Inverse + F_Inverse_T * transpose(grad_v))) * F_Inverse_T;



 		    drag_lift_value -=  cauchy_stress_fluid
 		      * state_fe_face_values.normal_vector(q_point) *
 		      state_fe_face_values.JxW(q_point);

 		  }
 	      }
 	  }
      return drag_lift_value[1];
   }

     */

     UpdateFlags GetFaceUpdateFlags() const
     {
       return update_values | update_quadrature_points |
	 update_gradients | update_normal_vectors;
     }

     string GetType() const
     {
       //return "boundary face";
       return "boundary";
     }
     string GetName() const
     {
       return "Lift";
     }

 
  private:
  double _density_fluid,_viscosity,_lame_coefficient_mu,
    _poisson_ratio_nu, _lame_coefficient_lambda;

  };


#endif
