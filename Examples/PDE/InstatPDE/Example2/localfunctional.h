#ifndef _LOCALFunctional_
#define _LOCALFunctional_

#include "pdeinterface.h"

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

  void SetTime(double t) const
    {
      _time = t;
    }


   bool NeedTime() const
    {
      if(fabs(_time-1.)< 1.e-13)
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
      return "domain timedistributed";
    }
    
    string GetName() const
    {
	  return "dummy functional";
	}
  private:
    Vector<double> _qvalues;
    vector<Vector<double> > _fvalues;
    vector<Vector<double> > _uvalues;
    vector<Vector<double> > _duvalues;
    double _alpha;
    mutable double _time;
  };
#endif
