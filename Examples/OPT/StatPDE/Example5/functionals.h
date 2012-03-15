#ifndef _LOCALFunctionalS_
#define _LOCALFunctionalS_

#include "pdeinterface.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

template<typename VECTOR, int dopedim, int dealdim>
  class QErrorFunctional : public FunctionalInterface<Multimesh_CellDataContainer,Multimesh_FaceDataContainer,dealii::DoFHandler<dealdim>, VECTOR,dopedim,dealdim>
  {
  public:
    QErrorFunctional()
    {
    }

    double Value(const Multimesh_CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc)
    {
      const DOpEWrapper::FEValues<dealdim> & fe_values = cdc.GetFEValuesControl();
      unsigned int n_q_points = cdc.GetNQPoints();
      {
	_uvalues.resize(n_q_points);
	_fvalues.resize(n_q_points);
	cdc.GetValuesControl("control",_uvalues);
      }
      double alpha = 1.e-3;

      double r = 0.;
      for(unsigned int q_point=0; q_point<n_q_points; q_point++)
      {
	_fvalues[q_point] = 1./alpha*sin(M_PI * fe_values.quadrature_point(q_point)(0)) *
				sin(2 * M_PI *fe_values.quadrature_point(q_point)(1));
       r += (_uvalues[q_point]- _fvalues[q_point]) * (_uvalues[q_point]- _fvalues[q_point]) *fe_values.JxW(q_point);
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
      return "QError";
    }

  private:
    vector<double> _uvalues;
    vector<double> _fvalues;
  };

/****************************************************************************************/

template<typename VECTOR, int dopedim, int dealdim>
  class UErrorFunctional : public FunctionalInterface<Multimesh_CellDataContainer,Multimesh_FaceDataContainer,dealii::DoFHandler<dealdim>, VECTOR,dopedim,dealdim>
  {
  public:
    UErrorFunctional()
    {
    }

    double Value(const Multimesh_CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc)
    {
      const DOpEWrapper::FEValues<dealdim> & fe_values = cdc.GetFEValuesState();
      unsigned int n_q_points = cdc.GetNQPoints();
      {
	_uvalues.resize(n_q_points);
	_fvalues.resize(n_q_points);
	cdc.GetValuesState("state",_uvalues);
      }

      double r = 0.;
      for(unsigned int q_point=0; q_point<n_q_points; q_point++)
      {
	_fvalues[q_point] =  sin(4*M_PI * fe_values.quadrature_point(q_point)(0)) *
				sin(2 * M_PI *fe_values.quadrature_point(q_point)(1));
       r += (_uvalues[q_point]- _fvalues[q_point]) * (_uvalues[q_point]- _fvalues[q_point]) *fe_values.JxW(q_point);
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
      return "UError";
    }

  private:
    vector<double> _uvalues;
    vector<double> _fvalues;
  };

/****************************************************************************************/template<typename VECTOR, int dopedim, int dealdim>
  class LocalMeanValueFunctional : public FunctionalInterface<Multimesh_CellDataContainer,Multimesh_FaceDataContainer,dealii::DoFHandler<dealdim>, VECTOR,dopedim,dealdim>
  {
  public:
    LocalMeanValueFunctional()
    {
    }

    double Value(const Multimesh_CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc)
    {
      const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
      unsigned int n_q_points = cdc.GetNQPoints();
      {
	_uvalues.resize(n_q_points);
	cdc.GetValuesState("state",_uvalues);
      }

      double r = 0.;
      for(unsigned int q_point=0; q_point<n_q_points; q_point++)
      {
	r += fabs(_uvalues[q_point]) * state_fe_values.JxW(q_point);
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
      return "L1-Norm";
    }

  private:
    vector<double> _uvalues;
  };

/****************************************************************************************/

template<typename VECTOR, int dopedim, int dealdim>
  class LocalPointFunctional : public FunctionalInterface<Multimesh_CellDataContainer,Multimesh_FaceDataContainer,dealii::DoFHandler<dealdim>, VECTOR, dopedim,dealdim>
  {
  public:

  double PointValue(const DOpEWrapper::DoFHandler<dopedim, dealii::DoFHandler<dealdim> > & control_dof_handler __attribute__((unused)),
		    const DOpEWrapper::DoFHandler<dealdim, dealii::DoFHandler<dealdim> > &state_dof_handler,
		    const std::map<std::string, const dealii::Vector<double>* > &param_values __attribute__((unused)),
		    const std::map<std::string, const VECTOR* > &domain_values)
  {
    Point<2> p(0.125,0.75);

    typename map<string, const BlockVector<double>* >::const_iterator it = domain_values.find("state");
    Vector<double> tmp_vector(1);

    VectorTools::point_value (state_dof_handler, *(it->second), p, tmp_vector);

    return  tmp_vector(0);
  }

  string GetType() const
  {
    return "point";
  }
  string GetName() const
  {
    return "PointValue";
  }

  };

#endif
