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
  class LocalFaceFunctional : public FunctionalInterface<CellDataContainer,
      FaceDataContainer, dealii::DoFHandler<dealdim>, VECTOR, dealdim>
  {
    public:
      LocalFaceFunctional()
      {
      }

      double
      FaceValue(const FACEDATACONTAINER& fdc)
      {
        unsigned int n_q_points = fdc.GetNQPoints();
        unsigned int material_id = fdc.GetMaterialId();
        unsigned int material_id_neighbor = fdc.GetNbrMaterialId();

        double mean = 0;

        if (material_id == 1)
        {
          if (material_id_neighbor == 2)
          {
            vector<double> _ufacevalues;

            _ufacevalues.resize(n_q_points);

            fdc.GetFaceValuesState("state", _ufacevalues);

            for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
            {
              double v;

              v = _ufacevalues[q_point];

              mean += 1. / (1.5) * v * fdc.GetFEFaceValuesState().JxW(q_point);
            }
          }
        }
        return mean;
      }

      void
      FaceValue_U(const FACEDATACONTAINER& fdc,
          dealii::Vector<double> &local_cell_vector, double scale)
      {
        unsigned int n_q_points = fdc.GetNQPoints();
        unsigned int material_id = fdc.GetMaterialId();
        unsigned int material_id_neighbor = fdc.GetNbrMaterialId();

        double mean = 0;

        if (material_id == 1)
        {
          if (material_id_neighbor == 2)
          {
            for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
            {
              for (unsigned int i = 0; i < fdc.GetNDoFsPerCell(); ++i)
              {
                local_cell_vector(i) += scale * 1. / (1.5)
                    * fdc.GetFEFaceValuesState().shape_value(i, q_point)
                    * fdc.GetFEFaceValuesState().JxW(q_point);
              }
            }
          }
        }
      }

      //Achtung, hier kein gradient update
      UpdateFlags
      GetFaceUpdateFlags() const
      {
        return update_values | update_quadrature_points | update_normal_vectors;
      }

      string
      GetType() const
      {
        return "face";
      }

      bool HasFaces() const
      {
        return true;
      }

      string
      GetName() const
      {
        return "Local Mean value";
      }

    private:
      int _outflow_fluid_boundary_color;
  };
#endif /* FUNCTIONALS_H_ */
