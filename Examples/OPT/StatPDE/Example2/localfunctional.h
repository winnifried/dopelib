#ifndef _LOCALFunctional_
#define _LOCALFunctional_

#include <numerics/vectors.h>
#include <base/quadrature.h>
#include <fe/fe_values.h>
#include <grid/grid_tools.h>
#include <fe/mapping_q1.h>

#include "functionalinterface.h"
#include "myfunctions.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

template<typename VECTOR, int dopedim, int dealdim>
  class LocalFunctional : public FunctionalInterface<CellDataContainer,
      FaceDataContainer, dealii::DoFHandler<dealdim>, VECTOR, dopedim, dealdim>
  {
    public:
      LocalFunctional()
      {
        _alpha = 0.;
        _eval_points.resize(3);
        //for q0
        _eval_points[0][0] = 0.5;
        _eval_points[0][1] = 0.5;
        //for q1
        _eval_points[1][0] = 0.5;
        _eval_points[1][1] = 0.25;
        //for q2
        _eval_points[2][0] = 0.25;
        _eval_points[2][1] = 0.25;
      }

      double
      Value(
          const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc)
      {
        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            cdc.GetFEValuesState();
        unsigned int n_q_points = cdc.GetNQPoints();
        {
          _qvalues.reinit(3);
          cdc.GetParamValues("control", _qvalues);
        }
        double r = 0.;

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          r += _alpha * 0.5
              * (_qvalues(0) * _qvalues(0) + _qvalues(1) * _qvalues(1)
                  + _qvalues(2) * _qvalues(2)) * state_fe_values.JxW(q_point);
        }

        return r;
      }

      double
      PointValue(
          const DOpEWrapper::DoFHandler<dopedim, dealii::DoFHandler<dealdim> > &/*control_dof_handler*/,
          const DOpEWrapper::DoFHandler<dealdim, dealii::DoFHandler<dealdim> > &state_dof_handler,
          const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
          const std::map<std::string, const VECTOR*> &domain_values)
      {
        double r = 0;
        //now we extract the solution u
        typename std::map<std::string, const VECTOR*>::const_iterator it =
            domain_values.find("state");
        if (it == domain_values.end())
        {
          throw DOpEException("Did not find state",
              "localfunctional::PointValue");
        }
        VECTOR U(*(it->second));

        //J[i] = (u_h - \overline{u})(x_i)
        std::vector<Vector<double> > J(3, Vector<double>(2));

        for (unsigned int i = 0; i < _eval_points.size(); i++)
        {
          VectorTools::point_value(state_dof_handler, U, _eval_points[i], J[i]);
          Vector<double> u_ex(2);
          _exact_u.vector_value(_eval_points[i], u_ex);
          J[i].add(-1., u_ex);
          r += std::pow(J[i].l2_norm(), 2);
        }

        r *= 0.5;
        return r;
      }

      void
      Value_U(
	const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& /*cdc*/,
	dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/)
      {
      }

      virtual void
      PointValue_U(
          const DOpEWrapper::DoFHandler<dopedim, dealii::DoFHandler<dealdim> > &/*control_dof_handler*/,
          const DOpEWrapper::DoFHandler<dealdim, dealii::DoFHandler<dealdim> > &state_dof_handler,
          const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
          const std::map<std::string, const VECTOR*> &domain_values,
          VECTOR& rhs, double scale)
      {
        VECTOR rhs_tmp_0, rhs_tmp_1;
        //lets extract the solution u
        typename std::map<std::string, const VECTOR*>::const_iterator it =
            domain_values.find("state");
        if (it == domain_values.end())
        {
          throw DOpEException("Did not find state",
              "localfunctional::PointValue_U");
        }
        VECTOR U(*(it->second));

        //as obove, J[i] = (u_h - \overline{u})(x_i)
        std::vector<Vector<double> > J(3, Vector<double>(2));
        rhs.reinit(U);
        rhs_tmp_0.reinit(U);
        rhs_tmp_1.reinit(U);

        for (unsigned int i = 0; i < _eval_points.size(); i++)
        {
          VectorTools::point_value(state_dof_handler, U, _eval_points[i], J[i]);

          Vector<double> u_ex(2);
          _exact_u.vector_value(_eval_points[i], u_ex);
          J[i].add(-1., u_ex);

          create_point_source(state_dof_handler.GetDEALDoFHandler(),
              _eval_points[i], 0, rhs_tmp_0);
          rhs_tmp_0 *= J[i][0];
          rhs.add(rhs_tmp_0);
          create_point_source(state_dof_handler.GetDEALDoFHandler(),
              _eval_points[i], 1, rhs_tmp_1);
          rhs_tmp_1 *= J[i][1];
          rhs.add(rhs_tmp_1);
        }

        rhs *= scale;
      }

      virtual void
      PointValue_Q(
          const DOpEWrapper::DoFHandler<dopedim, dealii::DoFHandler<dealdim>> &/*control_dof_handler*/,
          const DOpEWrapper::DoFHandler<dealdim, dealii::DoFHandler<dealdim>> &,
          const std::map<std::string, const dealii::Vector<double>*> &,
          const std::map<std::string, const VECTOR*> &,
          VECTOR&, double)
      {
      }

      void
      Value_Q(
          const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
          dealii::Vector<double> &local_cell_vector, double scale)
      {
        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            cdc.GetFEValuesState();
        unsigned int n_q_points = cdc.GetNQPoints();
        {
          _qvalues.reinit(3);

          cdc.GetParamValues("control", _qvalues);
        }

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          for (unsigned int i = 0; i < local_cell_vector.size(); i++)
          {
            local_cell_vector(i) += scale * _alpha * (_qvalues(i))
                * state_fe_values.JxW(q_point);
          }
        }
      }

      void
      Value_UU(
          const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>&,
          dealii::Vector<double> &, double )
      {
      }

      virtual void
      PointValue_UU(
          const DOpEWrapper::DoFHandler<dopedim, dealii::DoFHandler<dealdim> > &/*control_dof_handler*/,
          const DOpEWrapper::DoFHandler<dealdim, dealii::DoFHandler<dealdim> > &state_dof_handler,
          const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
          const std::map<std::string, const VECTOR*> &domain_values,
          VECTOR& rhs, double scale)
      {
        VECTOR rhs_tmp_0, rhs_tmp_1;
        typename std::map<std::string, const VECTOR*>::const_iterator it =
            domain_values.find("tangent");
        if (it == domain_values.end())
        {
          throw DOpEException("Did not find tangent",
              "localfunctional::PointValue_UU");
        }

        VECTOR DU(*(it->second));
        std::vector<Vector<double> > J(3, Vector<double>(2));
        rhs.reinit(DU);
        rhs_tmp_0.reinit(DU);
        rhs_tmp_1.reinit(DU);

        for (unsigned int i = 0; i < _eval_points.size(); i++)
        {
          VectorTools::point_value(state_dof_handler, DU, _eval_points[i],
              J[i]);

          create_point_source(state_dof_handler.GetDEALDoFHandler(),
              _eval_points[i], 0, rhs_tmp_0);
          rhs_tmp_0 *= J[i][0];
          rhs.add(rhs_tmp_0);
          create_point_source(state_dof_handler.GetDEALDoFHandler(),
              _eval_points[i], 1, rhs_tmp_1);
          rhs_tmp_1 *= J[i][1];
          rhs.add(rhs_tmp_1);
        }

        rhs *= scale;
      }

      virtual void
      PointValue_QU(
          const DOpEWrapper::DoFHandler<dopedim, dealii::DoFHandler<dealdim> > &/*control_dof_handler*/,
          const DOpEWrapper::DoFHandler<dealdim, dealii::DoFHandler<dealdim> > &/*state_dof_handler*/,
          const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
          const std::map<std::string, const VECTOR*> &/*domain_values*/,
          VECTOR& /*rhs*/, double /*scale*/)
      {

      }

      virtual void
      PointValue_UQ(
          const DOpEWrapper::DoFHandler<dopedim, dealii::DoFHandler<dealdim> > &/*control_dof_handler*/,
          const DOpEWrapper::DoFHandler<dealdim, dealii::DoFHandler<dealdim> > &/*state_dof_handler*/,
          const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
          const std::map<std::string, const VECTOR*> &/*domain_values*/,
          VECTOR& /*rhs*/, double /*scale*/)
      {

      }

      virtual void
      PointValue_QQ(
          const DOpEWrapper::DoFHandler<dopedim, dealii::DoFHandler<dealdim> > &/*control_dof_handler*/,
          const DOpEWrapper::DoFHandler<dealdim, dealii::DoFHandler<dealdim> > &/*state_dof_handler*/,
          const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
          const std::map<std::string, const VECTOR*> &/*domain_values*/,
          VECTOR& /*rhs*/, double /*scale*/)
      {

      }

      void
      Value_QU(
          const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc __attribute__((unused)),
          dealii::Vector<double> &local_cell_vector __attribute__((unused)),
          double scale __attribute__((unused)))
      {
      }

      void
      Value_UQ(
          const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc __attribute__((unused)),
          dealii::Vector<double> &local_cell_vector __attribute__((unused)),
          double scale __attribute__((unused)))
      {
      }

      void
      Value_QQ(
          const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
          dealii::Vector<double> &local_cell_vector, double scale)
      {
        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            cdc.GetFEValuesState();
        unsigned int n_q_points = cdc.GetNQPoints();
        {
          _dqvalues.reinit(3);
          cdc.GetParamValues("dq", _dqvalues);
        }

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          for (unsigned int i = 0; i < local_cell_vector.size(); i++)
          {
            local_cell_vector(i) += scale * _alpha * _dqvalues(i)
                * state_fe_values.JxW(q_point);
          }
        }
      }

      UpdateFlags
      GetUpdateFlags() const
      {
        return update_values | update_quadrature_points;
      }

      string
      GetType() const
      {
        return "point domain";
      }

      string
      GetName() const
      {
        return "cost functional";
      }

    private:
      void
      create_point_source(const dealii::DoFHandler<dealdim>& dof_handler,
          const Point<dealdim> point, const unsigned int component,
          VECTOR& rhs_vector)
      {
        Assert(rhs_vector.size() == dof_handler.n_dofs(),
            ExcDimensionMismatch(rhs_vector.size(), dof_handler.n_dofs()));

        rhs_vector = 0;

        std::pair<typename DoFHandler<dealdim>::active_cell_iterator,
            Point<dealdim> > cell_point =
            GridTools::find_active_cell_around_point(
                StaticMappingQ1<dealdim>::mapping, dof_handler, point);

        Quadrature<dealdim> q(
            GeometryInfo<dealdim>::project_to_unit_cell(cell_point.second));

        FEValues<dealdim> fe_values(dof_handler.get_fe(), q,
            UpdateFlags(update_values));
        fe_values.reinit(cell_point.first);

        const unsigned int dofs_per_cell = dof_handler.get_fe().dofs_per_cell;

        std::vector<unsigned int> local_dof_indices(dofs_per_cell);
        cell_point.first->get_dof_indices(local_dof_indices);

        for (unsigned int i = 0; i < dofs_per_cell; i++)
          rhs_vector(local_dof_indices[i]) = fe_values.shape_value_component(i,
              0, component);
      }
      vector<Point<2> > _eval_points;
      Vector<double> _qvalues;
      Vector<double> _dqvalues;
      vector<Vector<double> > _fvalues;
      vector<Vector<double> > _uvalues;
      vector<Vector<double> > _duvalues;
      double _alpha;
      ExactU _exact_u;
  };
#endif
