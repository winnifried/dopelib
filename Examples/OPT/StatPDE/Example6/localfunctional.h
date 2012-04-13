#ifndef _LOCALFunctional_
#define _LOCALFunctional_

#include "functionalinterface.h"
#include "myfunctions.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

template<template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC, template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC, typename DOFHANDLER, typename VECTOR,int dopedim, int dealdim>
  class LocalFunctional : public FunctionalInterface<CDC,FDC,DOFHANDLER,VECTOR,dopedim,dealdim>
  {
  public:
    LocalFunctional() 
    {
    }
    
    double BoundaryValue(const FDC<DOFHANDLER, VECTOR, dealdim>& fdc)
    {
      const auto & state_fe_face_values = fdc.GetFEFaceValuesState();
      unsigned int n_q_points = fdc.GetNQPoints();
      unsigned int color = fdc.GetBoundaryIndicator();
      {
	//Reading data
	_uvalues.resize(n_q_points,Vector<double>(2));
 
	fdc.GetFaceValuesState("state",_uvalues);
	_fvalues.resize(2);
      }
      double ret = 0.;
      
      if(color == 3)
      {
	for (unsigned int q_point=0;q_point<n_q_points; ++q_point)
	{
	  MyFunctions::Forces(_fvalues,state_fe_face_values.quadrature_point(q_point)(0), state_fe_face_values.quadrature_point(q_point)(1));
	  ret += (_fvalues[0] * _uvalues[q_point](0) +_fvalues[1] * _uvalues[q_point](1)) * state_fe_face_values.JxW(q_point);
	}
      }
      return ret;
    }
    
    void BoundaryValue_U(const FDC<DOFHANDLER, VECTOR, dealdim>& fdc,
			 dealii::Vector<double> &local_cell_vector, 
			 double scale)
    {
      const auto & state_fe_face_values = fdc.GetFEFaceValuesState();
      unsigned int n_dofs_per_cell = fdc.GetNDoFsPerCell();
      unsigned int n_q_points = fdc.GetNQPoints();
      unsigned int color = fdc.GetBoundaryIndicator();
      {
	_fvalues.resize(2);
      }
      
      if(color == 3)
      {
	const FEValuesExtractors::Scalar comp_0 (0);
	const FEValuesExtractors::Scalar comp_1 (1);
	
	for (unsigned int q_point=0;q_point<n_q_points; ++q_point)
	{
	  MyFunctions::Forces(_fvalues,state_fe_face_values.quadrature_point(q_point)(0), state_fe_face_values.quadrature_point(q_point)(1));
	  for(unsigned int i = 0; i < n_dofs_per_cell; i++)
	  {
	    local_cell_vector(i) += scale * (_fvalues[0] * state_fe_face_values[comp_0].value (i, q_point)
					     +_fvalues[1] * state_fe_face_values[comp_1].value (i, q_point)) * state_fe_face_values.JxW(q_point);
	  }
	}
      }
      
    }

     void BoundaryValue_Q(const FDC<DOFHANDLER, VECTOR, dealdim>& fdc __attribute__((unused)),
			  dealii::Vector<double> &local_cell_vector __attribute__((unused)), 
			  double scale __attribute__((unused)))
     {
     }
     
    void BoundaryValue_UU(const FDC<DOFHANDLER, VECTOR, dealdim>& fdc __attribute__((unused)),
			  dealii::Vector<double> &local_cell_vector __attribute__((unused)), 
			  double scale __attribute__((unused)))
    {
    }
     
    void BoundaryValue_QU(const FDC<DOFHANDLER, VECTOR, dealdim>& fdc __attribute__((unused)),
			  dealii::Vector<double> &local_cell_vector __attribute__((unused)), 
			  double scale __attribute__((unused)))
    {
    }
    
    void BoundaryValue_UQ(const FDC<DOFHANDLER, VECTOR, dealdim>& fdc __attribute__((unused)),
			  dealii::Vector<double> &local_cell_vector __attribute__((unused)), 
			  double scale __attribute__((unused)))
    {
    }
     
    void BoundaryValue_QQ(const FDC<DOFHANDLER, VECTOR, dealdim>& fdc __attribute__((unused)),
			  dealii::Vector<double> &local_cell_vector __attribute__((unused)), 
			  double scale __attribute__((unused)))
    {
    }

    void Value_U(const CDC<DOFHANDLER, VECTOR, dealdim>& cdc __attribute__((unused)),
		 dealii::Vector<double> &local_cell_vector __attribute__((unused)), 
		 double scale __attribute__((unused)))
    {
    }
    void Value_Q(const CDC<DOFHANDLER, VECTOR, dealdim>& cdc __attribute__((unused)),
		 dealii::Vector<double> &local_cell_vector __attribute__((unused)), 
		 double scale __attribute__((unused)))
    {
    }

    UpdateFlags GetFaceUpdateFlags() const
    {
      return update_values | update_quadrature_points;
    }
    
    string GetType() const
    {
      return "boundary";
    }
    
        string GetName() const
    {
	  return "cost functional";
	}

  private:
    vector<double> _fvalues;
    vector<Vector<double> > _uvalues;
  };
#endif
