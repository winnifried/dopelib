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
  class LocalPointFunctionalX : public FunctionalInterface<CellDataContainer,FaceDataContainer,dealii::DoFHandler<dealdim>, VECTOR, dealdim>
  {
  public:

    double PointValue(const DOpEWrapper::DoFHandler<dealdim, dealii::DoFHandler<dealdim> > & /*control_dof_handler*/,
			      const DOpEWrapper::DoFHandler<dealdim, dealii::DoFHandler<dealdim> > &state_dof_handler,
		      const std::map<std::string, const dealii::Vector<double>* > &/*param_values*/,
			      const std::map<std::string, const VECTOR* > &domain_values)
  {
    Point<2> p1(2.0,1.0);

    typename map<string, const VECTOR* >::const_iterator it = domain_values.find("state");
    Vector<double> tmp_vector(3);


    VectorTools::point_value (state_dof_handler, *(it->second), p1, tmp_vector);
    double x = tmp_vector(0);

    // x-velocity
    return x;

  }

  string GetType() const
  {
    return "point";
  }
  string GetName() const
  {
    return "Velocity in X";
  }

  };


// drag
/****************************************************************************************/
template<typename VECTOR, int dealdim>
  class LocalBoundaryFluxFunctional : public FunctionalInterface<CellDataContainer,FaceDataContainer,dealii::DoFHandler<dealdim>, VECTOR, dealdim>
  {
  public:
    bool HasFaces() const
      {
        return false;
      }


     // compute drag value around cylinder
     double BoundaryValue(const  FaceDataContainer<dealii::DoFHandler<2>, VECTOR, dealdim>& fdc)
     {
       unsigned int color = fdc.GetBoundaryIndicator();
       const DOpEWrapper::FEFaceValues<dealdim> &state_fe_face_values = fdc.GetFEFaceValuesState();
         unsigned int n_q_points = fdc.GetNQPoints();
       double flux = 0.0;
       if (color == 1)
	 {

	  vector<Vector<double> > _ufacevalues;

	  _ufacevalues.resize(n_q_points,Vector<double>(3));

	  fdc.GetFaceValuesState("state", _ufacevalues);

	  for (unsigned int q_point=0;q_point<n_q_points;q_point++)
	    {
	       Tensor<1,2> v;
	       v.clear();
	       v[0] = _ufacevalues[q_point](0);
	       v[1] = _ufacevalues[q_point](1);

	      flux +=  v * state_fe_face_values.normal_vector(q_point) *
		state_fe_face_values.JxW(q_point);
	    }

	 }
       return flux;

     }



     UpdateFlags GetFaceUpdateFlags() const
     {
       return update_values | update_quadrature_points |  update_normal_vectors;
     }

     string GetType() const
     {
       return "boundary";
     }
     string GetName() const
     {
       return "Flux";
     }

  private:

  };

#endif
