#ifndef _LOCALFunctionalS_
#define _LOCALFunctionalS_

#include "pdeinterface.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

template<typename VECTOR, int dopedim, int dealdim>
  class LocalMeanValueFunctional : public FunctionalInterface<CellDataContainer,FaceDataContainer,dealii::DoFHandler<dealdim>, VECTOR,dopedim,dealdim>
  {
  public:
    LocalMeanValueFunctional()
    {
    }

    double Value(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc)
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
  class LocalPointFunctional : public FunctionalInterface<CellDataContainer,FaceDataContainer,dealii::DoFHandler<dealdim>, VECTOR, dopedim,dealdim>
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
