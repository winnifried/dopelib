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

using namespace std;
using namespace dealii;
using namespace DOpE;


// x-displacement in (90,0)
/****************************************************************************************/

template<typename VECTOR, int dealdim>
  class LocalPointFunctionalDisp_1 : public FunctionalInterface<CellDataContainer,FaceDataContainer,dealii::DoFHandler<dealdim>, VECTOR,dealdim>
  {

  public:

    double PointValue(const DOpEWrapper::DoFHandler<dealdim, dealii::DoFHandler<dealdim> > & control_dof_handler __attribute__((unused)),
		    const DOpEWrapper::DoFHandler<dealdim, dealii::DoFHandler<dealdim> > &state_dof_handler,
		    const std::map<std::string, const dealii::Vector<double>* > &param_values __attribute__((unused)),
		    const std::map<std::string, const VECTOR* > &domain_values)
  {
    Point<2> p1(90,0);

    typename map<string, const VECTOR* >::const_iterator it = domain_values.find("state");
    Vector<double> tmp_vector(2);

    VectorTools::point_value (state_dof_handler, *(it->second), p1, tmp_vector);
    double u1 = tmp_vector(0);

    return u1;
  }
  string GetType() const
  {
    return "point";
  }
  string GetName() const
  {
    return "x-displacement_in_(90,0)";
  }

  };


// y-displacement in (100,100)
/****************************************************************************************/
template<typename VECTOR, int dealdim>
  class LocalPointFunctionalDisp_2 : public FunctionalInterface<CellDataContainer,FaceDataContainer,dealii::DoFHandler<dealdim>,VECTOR,dealdim>
  {

  public:

  double PointValue(const DOpEWrapper::DoFHandler<dealdim, dealii::DoFHandler<dealdim> > & control_dof_handler __attribute__((unused)),
			      const DOpEWrapper::DoFHandler<dealdim, dealii::DoFHandler<dealdim> > &state_dof_handler,
			      const std::map<std::string, const dealii::Vector<double>* > &param_values __attribute__((unused)),
			      const std::map<std::string, const VECTOR* > &domain_values)
  {
    Point<dealdim> p1(100,100);

    typename map<string, const VECTOR* >::const_iterator it = domain_values.find("state");
    Vector<double> tmp_vector(2);


    VectorTools::point_value (state_dof_handler, *(it->second), p1, tmp_vector);
    double u2 = tmp_vector(1);

    return u2;
  }

  string GetType() const
  {
    return "point";
  }
  string GetName() const
  {
    return "y-displacement_in_(100,100)";
  }

  };


// x-displacement in (0,100)
/****************************************************************************************/
template<typename VECTOR, int dealdim>
  class LocalPointFunctionalDisp_3 : public FunctionalInterface<CellDataContainer,FaceDataContainer,dealii::DoFHandler<dealdim>,VECTOR,dealdim>
  {

  public:

  double PointValue(const DOpEWrapper::DoFHandler<dealdim, dealii::DoFHandler<dealdim> > & control_dof_handler __attribute__((unused)),
		    const DOpEWrapper::DoFHandler<dealdim, dealii::DoFHandler<dealdim> > &state_dof_handler,
		    const std::map<std::string, const dealii::Vector<double>* > &param_values __attribute__((unused)),
		    const std::map<std::string, const VECTOR* > &domain_values)
  {
    Point<2> p1(0,100);

    typename map<string, const VECTOR* >::const_iterator it = domain_values.find("state");
    Vector<double> tmp_vector(2);

    VectorTools::point_value (state_dof_handler, *(it->second), p1, tmp_vector);
    double u1 = tmp_vector(0);

    return u1;
  }

  string GetType() const
  {
    return "point";
  }
  string GetName() const
  {
    return "x-displacement_in_(0,100)";
  }

  };


//yy-stress in (90,0)
/****************************************************************************************/

template<typename VECTOR, int dealdim>
  class LocalDomainFunctionalStress : public FunctionalInterface<CellDataContainer,FaceDataContainer,dealii::DoFHandler<dealdim>, VECTOR,dealdim>
  {
  private:

  public:

      double Value(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc)
      {
	const DOpEWrapper::FEValues<dealdim> &state_fe_values = cdc.GetFEValuesState();
	unsigned int n_q_points = cdc.GetNQPoints();

	double yy_stress = 0.;

	vector<vector<Tensor<1,2> > > _ugrads;

	_ugrads.resize(n_q_points,vector<Tensor<1,2> >(2));

	cdc.GetGradsState("state",_ugrads);

	const double mu = 80193.800283;
	const double kappa = 271131.389455;
	const double lambda = 110743.788889;

	for (unsigned int q_point=0;q_point<n_q_points;q_point++)
	{
	  Tensor<2,2> vgrads;
	  vgrads.clear();
	  vgrads[0][0] = _ugrads[q_point][0][0];
	  vgrads[0][1] = _ugrads[q_point][0][1];
	  vgrads[1][0] = _ugrads[q_point][1][0];
	  vgrads[1][1] = _ugrads[q_point][1][1];

	  Tensor<2,2> realgrads;
	  realgrads.clear();
	  realgrads[0][0] = kappa * vgrads[0][0] + lambda * vgrads[1][1];
	  realgrads[0][1] = mu * vgrads[0][1] + mu * vgrads[1][0];
	  realgrads[1][0] = mu * vgrads[0][1] + mu * vgrads[1][0];
	  realgrads[1][1] = kappa * vgrads[1][1] + lambda * vgrads[0][0];

	  if(std::abs(state_fe_values.quadrature_point(q_point)(0) - 90) < 1.e-10 && std::abs(state_fe_values.quadrature_point(q_point)(1)) < 1.e-10)
	    {
	      yy_stress = realgrads[1][1];
	    }

	}
	return yy_stress;
      }

      UpdateFlags GetUpdateFlags() const
      {
	return update_values | update_quadrature_points |
	  update_gradients;
      }

      string GetType() const
      {
	return "domain";
      }
      string GetName() const
      {
	return "yy-stress_in_(90,0)";
      }

  protected:
      inline void GetValues(const DOpEWrapper::FEValues<dealdim>& fe_values,
			    const map<string, const VECTOR* >& domain_values, string name,
			    vector<Vector<double> >& values)
      {
	typename map<string, const VECTOR* >::const_iterator it = domain_values.find(name);
	if(it == domain_values.end())
	{
	  throw DOpEException("Did not find " + name,"LocalPDE::GetValues");
	}
	fe_values.get_function_values(*(it->second),values);
      }

      inline void GetGrads(const DOpEWrapper::FEValues<dealdim>& fe_values,
			   const map<string, const VECTOR* >& domain_values, string name,
			   vector<vector<Tensor<1,dealdim> > >& values)
      {
	typename map<string, const VECTOR* >::const_iterator it = domain_values.find(name);
	if(it == domain_values.end())
	{
	  throw DOpEException("Did not find " + name,"LocalPDE::GetGrads");
	}
	fe_values.get_function_grads(*(it->second),values);
      }
  };

// y-displacement-integral on upper boundary
/****************************************************************************************/

template<typename VECTOR, int dealdim>
  class LocalBoundaryFaceFunctionalUpBd : public FunctionalInterface<CellDataContainer,FaceDataContainer,dealii::DoFHandler<dealdim>, VECTOR,dealdim>
  {
  private:



  public:

    bool HasFaces() const
    {
      return false;
    }

    // compute y-displacement-integral
    double BoundaryValue(const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc)
    {
      unsigned int color = fdc.GetBoundaryIndicator();
      const auto &state_fe_face_values = fdc.GetFEFaceValuesState();
      unsigned int n_q_points = fdc.GetNQPoints();

      double integral = 0;
      if (color == 3)
      {

	vector<Vector<double> > _ufacevalues;

	_ufacevalues.resize(n_q_points,Vector<double>(2));

	fdc.GetFaceValuesState("state",_ufacevalues);

	for (unsigned int q_point=0;q_point<n_q_points;q_point++)
	{
	  double y_displacement = _ufacevalues[q_point](1);

	  integral += y_displacement *
		state_fe_face_values.JxW(q_point);
	}
      }
      return integral;
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
      return "y-displacement-integral_on_upper_boundary";
    }

 
  };


#endif
