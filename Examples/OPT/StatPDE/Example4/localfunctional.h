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
	_alpha = 10.;
      }

      double Value(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc)
      {
	const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
	unsigned int n_q_points = cdc.GetNQPoints();
	{
	  _qvalues.reinit(5);
	  _fvalues.resize(n_q_points,Vector<double>(2));
	  _uvalues.resize(n_q_points,Vector<double>(2));

	  cdc.GetParamValues("control",_qvalues);
	  cdc.GetValuesState("state",_uvalues);
	}
	double r = 0.;

	for(unsigned int q_point = 0; q_point< n_q_points; q_point++)
	{
	  _fvalues[q_point](0) = ( sin(M_PI * state_fe_values.quadrature_point(q_point)(0)) *
				   sin(M_PI * state_fe_values.quadrature_point(q_point)(1)))*( state_fe_values.quadrature_point(q_point)(0));
	  _fvalues[q_point](1) = ( state_fe_values.quadrature_point(q_point)(0));


	  r += 0.5* (_uvalues[q_point](0)-_fvalues[q_point](0))*(_uvalues[q_point](0)-_fvalues[q_point](0))
	    * state_fe_values.JxW(q_point);
	  r += 0.5* (_uvalues[q_point](1)-_fvalues[q_point](1))*(_uvalues[q_point](1)-_fvalues[q_point](1))
	    * state_fe_values.JxW(q_point);

	  r+= _alpha*0.5*(_qvalues(0)*_qvalues(0)+_qvalues(1)*_qvalues(1)+_qvalues(2)*_qvalues(2)+_qvalues(3)*_qvalues(3)+_qvalues(4)*_qvalues(4))* state_fe_values.JxW(q_point);

	}
      return r;
    }

    void Value_U(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
		 dealii::Vector<double> &local_cell_vector, double scale)
    {
      const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
      unsigned int n_q_points = cdc.GetNQPoints();
      {
	_fvalues.resize(n_q_points,Vector<double>(2));
	_uvalues.resize(n_q_points,Vector<double>(2));

	cdc.GetValuesState("state",_uvalues);
      }

      const FEValuesExtractors::Scalar comp_0 (0);
      const FEValuesExtractors::Scalar comp_1 (1);

      for(unsigned int q_point = 0; q_point< n_q_points; q_point++)
      {
	_fvalues[q_point](0) = ( sin(M_PI * state_fe_values.quadrature_point(q_point)(0)) *
				 (sin(M_PI * state_fe_values.quadrature_point(q_point)(1)) +
				  0.5*sin(2 * M_PI * state_fe_values.quadrature_point(q_point)(1))))*( state_fe_values.quadrature_point(q_point)(0));
	_fvalues[q_point](1) = ( state_fe_values.quadrature_point(q_point)(0));


	for(unsigned int i=0; i < n_dofs_per_cell; i++)
	{
	  local_cell_vector(i) += scale*(_uvalues[q_point](0)-_fvalues[q_point](0))
	    *state_fe_values[comp_0].value (i, q_point) * state_fe_values.JxW(q_point);
	  local_cell_vector(i) += scale*(_uvalues[q_point](1)-_fvalues[q_point](1))
	    *state_fe_values[comp_1].value (i, q_point) * state_fe_values.JxW(q_point);
	}
      }
    }

    void Value_Q(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
		 dealii::Vector<double> &local_cell_vector, double scale)
    {
      const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
      unsigned int n_q_points = cdc.GetNQPoints();      
      {
     	_qvalues.reinit(local_cell_vector.size());

     	cdc.GetParamValues("control",_qvalues);
      }

      for(unsigned int q_point = 0; q_point< n_q_points; q_point++)
      {
     	for(unsigned int i=0; i < local_cell_vector.size(); i++)
     	{
     	  local_cell_vector(i) += scale*_alpha*(_qvalues(i)) * state_fe_values.JxW(q_point);
     	}
      }
    }

    void Value_UU(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
		  dealii::Vector<double> &local_cell_vector, double scale)
    {
      const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
      unsigned int n_q_points = cdc.GetNQPoints();
      {
	_duvalues.resize(n_q_points,Vector<double>(2));
	cdc.GetValuesState("tangent",_duvalues);
      }

      const FEValuesExtractors::Scalar comp_0 (0);
      const FEValuesExtractors::Scalar comp_1 (1);

      for(unsigned int q_point = 0; q_point< n_q_points; q_point++)
      {
	for(unsigned int i=0; i < n_dofs_per_cell; i++)
	{
	  local_cell_vector(i) += scale*_duvalues[q_point](0)*state_fe_values[comp_0].value (i, q_point)
	    * state_fe_values.JxW(q_point);
	  local_cell_vector(i) += scale*_duvalues[q_point](1)*state_fe_values[comp_1].value (i, q_point)
	    * state_fe_values.JxW(q_point);
	}
      }
    }

    void Value_QU(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc __attribute__((unused)),
		  dealii::Vector<double> &local_cell_vector __attribute__((unused)), double scale __attribute__((unused)))
    {
    }

    void Value_UQ(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc __attribute__((unused)),
		  dealii::Vector<double> &local_cell_vector __attribute__((unused)), double scale __attribute__((unused)))
    {
    }

    void Value_QQ(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
		  dealii::Vector<double> &local_cell_vector, double scale)
    {
      const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
      //unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
      unsigned int n_q_points = cdc.GetNQPoints();
      {
      	_dqvalues.reinit(local_cell_vector.size());
      	cdc.GetParamValues("dq",_dqvalues);
      }

      for(unsigned int q_point = 0; q_point< n_q_points; q_point++)
      {
      	for(unsigned int i=0; i < local_cell_vector.size(); i++)
      	{
      	  local_cell_vector(i) += scale*_alpha* _dqvalues(i) * state_fe_values.JxW(q_point);
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
    Vector<double> _qvalues;
    Vector<double> _dqvalues;
    vector<Vector<double> > _fvalues;
    vector<Vector<double> > _uvalues;
    vector<Vector<double> > _duvalues;
    double _alpha;
  };
#endif
