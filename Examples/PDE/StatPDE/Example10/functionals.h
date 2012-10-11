#ifndef _LOCALFunctionalS_
#define _LOCALFunctionalS_

#include "pdeinterface.h"
#include "celldatacontainer.h"
#include "facedatacontainer.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

template<typename VECTOR, typename FACEDATACONTAINER, int dealdim>
  class BoundaryFunctional : public FunctionalInterface<CellDataContainer,
      FaceDataContainer, dealii::DoFHandler<dealdim>, VECTOR, dealdim>
  {
    public:
      BoundaryFunctional()
      {
      }

      double
      BoundaryValue(const FACEDATACONTAINER& fdc)
      {
        unsigned int n_q_points = fdc.GetNQPoints();
        unsigned int boundary_ind = fdc.GetBoundaryIndicator();

        double cw = 0;

        _ufacevalues.resize(n_q_points);

        fdc.GetFaceValuesState("state", _ufacevalues);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          double p = _ufacevalues[q_point];

          cw += 1. * fdc.GetFEFaceValuesState().JxW(q_point);
        }

        return cw/2;
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
        return "boundary";
      }

      bool
      HasFaces() const
      {
        return true;
      }

      string
      GetName() const
      {
        return "BoundaryPI";
      }

    private:
      vector<double> _ufacevalues;

  };

#endif
