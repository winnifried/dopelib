#ifndef _LOCALFunctional_
#define _LOCALFunctional_

//#include "pdeinterface.h"
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

  // include NeedTime
    void SetTime(double t) const
    {
      _time = t;
    }


    bool NeedTime() const
    {
      if(fabs(_time-1.0)< 1.e-13)
	return true;
      return false;
    }


    double Value(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>&)
    {
      return 0.0;
    }


    void Value_U(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>&,
                 dealii::Vector<double> &local_cell_vector __attribute__((unused)),
                 double scale __attribute__((unused)))
    {

    }

    void Value_Q(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>&,
                 dealii::Vector<double> &local_cell_vector __attribute__((unused)),
                 double scale __attribute__((unused)))
    {

    }

    void Value_UU(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>&,
                  dealii::Vector<double> &local_cell_vector __attribute__((unused)),
                  double scale __attribute__((unused)))
    {

    }

    void Value_QU(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>&,
                  dealii::Vector<double> &local_cell_vector __attribute__((unused)),
                  double scale __attribute__((unused)))
    {
    }

    void Value_UQ(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>&,
                  dealii::Vector<double> &local_cell_vector __attribute__((unused)),
                  double scale __attribute__((unused)))
    {
    }

    void Value_QQ(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>&,
                  dealii::Vector<double> &local_cell_vector __attribute__((unused)),
                  double scale __attribute__((unused)))
    {

    }




    UpdateFlags GetUpdateFlags() const
    {
      return update_values | update_quadrature_points;
    }

    string GetType() const
    {
      return "domain timelocal";
    }


  protected:
    inline void GetValues(const DOpEWrapper::FEValues<dealdim>& fe_values,  const map<string, const VECTOR* >& domain_values,string name, vector<double>& values)
    {
      typename map<string, const VECTOR* >::const_iterator it = domain_values.find(name);
      if(it == domain_values.end())
	{
	  throw DOpEException("Did not find " + name,"LocalPDE::GetValues");
	}
      fe_values.get_function_values(*(it->second),values);
    }

    inline void GetGrads(const DOpEWrapper::FEValues<dealdim>& fe_values,  const map<string, const VECTOR* >& domain_values,string name, vector<Tensor<1,dealdim> >& values)
    {
      typename map<string, const VECTOR* >::const_iterator it = domain_values.find(name);
      if(it == domain_values.end())
	{
	  throw DOpEException("Did not find " + name,"LocalPDE::GetGrads");
	}
      fe_values.get_function_grads(*(it->second),values);
    }





  private:
    vector<double> _qvalues;
    vector<double> _fvalues;
    vector<double> _uvalues;
    vector<double> _duvalues;
    vector<double> _dqvalues;
    double _alpha;

    mutable double _time;

  };
#endif
