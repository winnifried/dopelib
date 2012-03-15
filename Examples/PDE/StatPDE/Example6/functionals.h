/*
 * functionals.h
 *
 *  Created on: Aug 22, 2011
 *      Author: cgoll
 */

#ifndef FUNCTIONALS_H_
#define FUNCTIONALS_H_

#include "functionalinterface.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

// massflux
/****************************************************************************************/
template<typename VECTOR, typename FACEDATACONTAINER, int dealdim>
class LocalBoundaryFunctionalMassFlux: public FunctionalInterface<CellDataContainer,FaceDataContainer,dealii::DoFHandler<dealdim>, VECTOR,
    dealdim>
{
	public:
		LocalBoundaryFunctionalMassFlux()
		{
		}

		// compute drag value around cylinder
		double BoundaryValue(const FACEDATACONTAINER& fdc)
		{
			unsigned int n_q_points = fdc.GetNQPoints();
			unsigned int color = fdc.GetBoundaryIndicator();

			double mass_flux_stokes=0;

			if (color == 1)
			{
				vector<Vector<double> > _ufacevalues;

				_ufacevalues.resize(n_q_points, Vector<double>(2));

				fdc.GetFaceValuesState("state", _ufacevalues);

				for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
				{
					Tensor < 1, 2 > v;
					v.clear();
					v[0] = _ufacevalues[q_point](0);
					v[1] = _ufacevalues[q_point](1);
					//die normale ist hier einfach der x-vektor, d.h. v*n=v*(1,0) = v[0]

					mass_flux_stokes += v
					    * fdc.GetFEFaceValuesState().normal_vector(q_point)
					    * fdc.GetFEFaceValuesState().JxW(q_point);
				}

			}
			return mass_flux_stokes;
		}

		//Achtung, hier kein gradient update
		UpdateFlags GetFaceUpdateFlags() const
		{
			return update_values | update_quadrature_points
			    | update_normal_vectors;
		}

		string GetType() const
		{
			return "boundary";
		}
		string GetName() const
		{
			return "Outflow";
		}

	private:
		int _outflow_fluid_boundary_color;
};
#endif /* FUNCTIONALS_H_ */
