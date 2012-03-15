#ifndef _LOCALFunctionalS_
#define _LOCALFunctionalS_

#include "pdeinterface.h"
#include "celldatacontainer.h"
#include "facedatacontainer.h"

using namespace std;
using namespace dealii;
using namespace DOpE;


/****************************************************************************************/
template<typename VECTOR, int dealdim>
  class LocalPointFunctionalX : public FunctionalInterface<CellDataContainer,FaceDataContainer,dealii::DoFHandler<dealdim>, VECTOR,dealdim>
  {
  public:

    double PointValue(const DOpEWrapper::DoFHandler<dealdim, dealii::DoFHandler<dealdim> > & /*control_dof_handler*/,
			      const DOpEWrapper::DoFHandler<dealdim, dealii::DoFHandler<dealdim> > &state_dof_handler,
		      const std::map<std::string, const dealii::Vector<double>* > &/*param_values*/,
			      const std::map<std::string, const VECTOR* > &domain_values)
  {
    Point<3> p1(0.5, 0.5, 0.5);

    typename map<string, const VECTOR* >::const_iterator it = domain_values.find("state");
    Vector<double> tmp_vector(3);

    VectorTools::point_value (state_dof_handler, *(it->second), p1, tmp_vector);
    double x = tmp_vector(0);

    return x;

  }

  string GetType() const
  {
    return "point";
  }
  string GetName() const
  {
    return "Point value in X";
  }

  };

#endif
