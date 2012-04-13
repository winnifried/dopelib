#ifndef _LOCALFunctional_
#define _LOCALFunctional_

#include "functionalinterface.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

template<typename VECTOR, int dopedim, int dealdim>
  class LocalFunctional : public FunctionalInterface<CellDataContainer,FaceDataContainer,dealii::DoFHandler<dealdim>, VECTOR, dopedim,dealdim>
  {
  public:
    LocalFunctional()
    {
      _alpha = 1.e-3;
    }

    double Value(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc)
    {
      const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
      unsigned int n_q_points = cdc.GetNQPoints();

      {
	_qvalues.resize(n_q_points);
	_fvalues.resize(n_q_points);
	_uvalues.resize(n_q_points);

	cdc.GetValuesControl("control",_qvalues);
  	cdc.GetValuesState("state",_uvalues);
      }

      double r = 0.;

      for(unsigned int q_point = 0; q_point< n_q_points; q_point++)
      {
	_fvalues[q_point] = ( 1.*sin(4 * M_PI * state_fe_values.quadrature_point(q_point)(0)) +
			      5.*M_PI*M_PI*sin(M_PI * state_fe_values.quadrature_point(q_point)(0))) *
	  sin(2 * M_PI * state_fe_values.quadrature_point(q_point)(1));

	r += 0.5* (_uvalues[q_point]-_fvalues[q_point])*(_uvalues[q_point]-_fvalues[q_point])
	  * state_fe_values.JxW(q_point);
	r += 0.5*_alpha* (_qvalues[q_point]*_qvalues[q_point])
	  * state_fe_values.JxW(q_point);
      }
      return r;
    }

    void Value_U(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
		 dealii::Vector<double> &local_cell_vector,
		 double scale)
    {
      const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
      unsigned int n_q_points = cdc.GetNQPoints();
     {
	_fvalues.resize(n_q_points);
	_uvalues.resize(n_q_points);

	cdc.GetValuesState("state",_uvalues);
      }

      for(unsigned int q_point = 0; q_point< n_q_points; q_point++)
      {
	_fvalues[q_point] = ( 1.*sin(4 * M_PI * state_fe_values.quadrature_point(q_point)(0)) +
			      5.*M_PI*M_PI*sin(M_PI * state_fe_values.quadrature_point(q_point)(0))) *
	  sin(2 * M_PI * state_fe_values.quadrature_point(q_point)(1));
	for(unsigned int i=0; i < n_dofs_per_cell; i++)
	{
	  local_cell_vector(i) += scale*(_uvalues[q_point]-_fvalues[q_point])*state_fe_values.shape_value (i, q_point)
	    * state_fe_values.JxW(q_point);
	}
      }
    }

    void Value_Q(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
		 dealii::Vector<double> &local_cell_vector,
		 double scale)
    {
      const DOpEWrapper::FEValues<dealdim> & control_fe_values = cdc.GetFEValuesControl();
      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
      unsigned int n_q_points = cdc.GetNQPoints();
      {
	_qvalues.resize(n_q_points);

	cdc.GetValuesControl("control",_qvalues);
      }

      for(unsigned int q_point = 0; q_point< n_q_points; q_point++)
      {
	for(unsigned int i=0; i < n_dofs_per_cell; i++)
	{
	  local_cell_vector(i) += scale*_alpha*(_qvalues[q_point]*control_fe_values.shape_value (i, q_point))
	    * control_fe_values.JxW(q_point);
	}
      }
    }

    void Value_UU(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
		  dealii::Vector<double> &local_cell_vector,
		  double scale)
    {
      const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
      unsigned int n_q_points = cdc.GetNQPoints();
      {
	_duvalues.resize(n_q_points);
	cdc.GetValuesState("tangent",_duvalues);
      }

      for(unsigned int q_point = 0; q_point< n_q_points; q_point++)
      {
	for(unsigned int i=0; i < n_dofs_per_cell; i++)
	{
	  local_cell_vector(i) += scale*_duvalues[q_point]*state_fe_values.shape_value (i, q_point)
	    * state_fe_values.JxW(q_point);
	}
      }
    }

    void Value_QU(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc __attribute__((unused)),
		  dealii::Vector<double> &local_cell_vector __attribute__((unused)),
		  double scale __attribute__((unused)))
    {
    }

    void Value_UQ(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc __attribute__((unused)),
		  dealii::Vector<double> &local_cell_vector __attribute__((unused)),
		  double scale __attribute__((unused)))
    {
    }

    void Value_QQ(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
		  dealii::Vector<double> &local_cell_vector,
		  double scale)
    {
      const DOpEWrapper::FEValues<dealdim> & control_fe_values = cdc.GetFEValuesControl();
      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
      unsigned int n_q_points = cdc.GetNQPoints();
      {
	_dqvalues.resize(n_q_points);
	cdc.GetValuesControl("dq",_dqvalues);
      }

      for(unsigned int q_point = 0; q_point< n_q_points; q_point++)
      {
	for(unsigned int i=0; i < n_dofs_per_cell; i++)
	{
	  local_cell_vector(i) += scale*_alpha*(_dqvalues[q_point]*control_fe_values.shape_value (i, q_point))
	    * control_fe_values.JxW(q_point);
	}
      }
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
	  return "cost functional";
	}

  private:
    vector<double> _qvalues;
    vector<double> _fvalues;
    vector<double> _uvalues;
    vector<double> _duvalues;
    vector<double> _dqvalues;
    double _alpha;
  };
#endif
