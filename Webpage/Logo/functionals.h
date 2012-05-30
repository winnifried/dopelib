#ifndef _LOCALFunctionalS_
#define _LOCALFunctionalS_

#include "pdeinterface.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

template<typename VECTOR, int dopedim, int dealdim>
  class LocalValueFunctional : public FunctionalInterface<CellDataContainer,FaceDataContainer,dealii::DoFHandler<dealdim>, VECTOR,dopedim,dealdim>
  {
  public:
    LocalValueFunctional()
    { 
      _alpha = 1.;
      _b = 0.06;
    }

    double Value(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc)
    {
      const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
      unsigned int n_q_points = cdc.GetNQPoints();
      {
	_uvalues.resize(n_q_points,Vector<double> (2));
	cdc.GetValuesState("state",_uvalues);	
      }

      double r = 0.;
      double x = 0.;
      double y = 0.;
      for(unsigned int q_point=0; q_point<n_q_points; q_point++)
      {
	x = state_fe_values.quadrature_point(q_point)(0);
	y = state_fe_values.quadrature_point(q_point)(1);
	r += std::max(0.,_uvalues[q_point](0)-upper_bound(x,y)) * state_fe_values.JxW(q_point);
      }
      return r;
    }

    UpdateFlags GetUpdateFlags() const
    {
      return update_values | update_quadrature_points;
    }

    string GetType() const
    {
      return "domain";
    }
    string GetName() const
    {
      return "L1-Violation";
    }

  private:    
    double upper_bound(double x, double y)
    {
      return _b;
    }

    vector<Vector<double> > _uvalues;
    double _alpha;
    double _b;
  };

template<typename VECTOR, int dopedim, int dealdim>
  class LocalValueFunctional2 : public FunctionalInterface<CellDataContainer,FaceDataContainer,dealii::DoFHandler<dealdim>, VECTOR,dopedim,dealdim>
  {
  public:
    LocalValueFunctional2()
    {
      _alpha = 1.;
      _b = 0.06;
    }

    double Value(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc)
    {
      const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
      unsigned int n_q_points = cdc.GetNQPoints();
      {
	_uvalues.resize(n_q_points,Vector<double> (2));
	cdc.GetValuesState("state",_uvalues);
      }

      double r = 0.;
      double x = 0.;
      double y = 0.;
      for(unsigned int q_point=0; q_point<n_q_points; q_point++)
      {
	x = state_fe_values.quadrature_point(q_point)(0);
	y = state_fe_values.quadrature_point(q_point)(1);
	r += std::max(0.,_uvalues[q_point](0)-upper_bound(x,y)) *std::max(0.,_uvalues[q_point](0)-upper_bound(x,y)) * state_fe_values.JxW(q_point);
      }
      return r;
    }

    UpdateFlags GetUpdateFlags() const
    {
      return update_values | update_quadrature_points;
    }

    string GetType() const
    {
      return "domain";
    }
    string GetName() const
    {
      return "L2-Violation";
    }

  private:
    double upper_bound(double x, double y)
    {
      return _b;
    }
    vector<Vector<double> > _uvalues;
    double _alpha;
    double _b;
  };
/****************************************************************************************/

template<typename VECTOR, int dopedim, int dealdim>
  class LocalPointFunctional : public FunctionalInterface<CellDataContainer,FaceDataContainer,dealii::DoFHandler<dealdim>, VECTOR, dopedim,dealdim>
  {
  public:
    LocalPointFunctional()
    {	
      _x = 0.5;
      _b = 0.06;
    }
    double PointValue(const DOpEWrapper::DoFHandler<dopedim> & control_dof_handler __attribute__((unused)),
		      const DOpEWrapper::DoFHandler<dealdim> &state_dof_handler,
		      const std::map<std::string, const dealii::Vector<double>* > &param_values __attribute__((unused)),
		      const std::map<std::string, const VECTOR* > &domain_values)
    {
    double _alpha = 1.;
    Point<2> p(_x,0.5);
    
    typename map<string, const BlockVector<double>* >::const_iterator it = domain_values.find("state");
    Vector<double> tmp_vector(2);

    VectorTools::point_value (state_dof_handler, *(it->second), p, tmp_vector);
    double lb = upper_bound(_x,0.5);
    return  max(0.,tmp_vector(0)-lb);
    }

  string GetType() const
  {
    return "point";
  }
  string GetName() const
  {
    return "PointViolation";
  }

  private:
  double upper_bound(double x, double y)
  {
      return _b;
  }
  double _b,_x;
      
  };

#endif
