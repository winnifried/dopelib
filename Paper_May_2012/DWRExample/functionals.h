#ifndef _LOCALFunctionalS_
#define _LOCALFunctionalS_

#include "pdeinterface.h"
#include "helper.h"
#include <dofs/dof_tools.h>
#include <base/types.h>
//#include "functionalinterface.h"

using namespace std;
using namespace dealii;
using namespace DOpE;
/****************************************************************************************/

template<typename VECTOR, int dealdim>
  class LocalPointFunctionalPressure : public FunctionalInterface<
      CellDataContainer, FaceDataContainer, dealii::DoFHandler<dealdim>, VECTOR,
      dealdim>
  {
    public:
      bool
      NeedTime() const
      {
        return false;
      }

      double
      PointValue(
          const DOpEWrapper::DoFHandler<dealdim, dealii::DoFHandler<dealdim> > & /*control_dof_handler*/,
          const DOpEWrapper::DoFHandler<dealdim, dealii::DoFHandler<dealdim> > & state_dof_handler,
          const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
          const std::map<std::string, const VECTOR*> &domain_values)
      {

        Point<2> p1(0.15, 0.2);
        Point<2> p2(0.25, 0.2);

        typename map<string, const VECTOR*>::const_iterator it =
            domain_values.find("state");
        Vector<double> tmp_vector(3);

        VectorTools::point_value(state_dof_handler, *(it->second), p1,
            tmp_vector);
        double p1_value = tmp_vector(2);
        tmp_vector = 0;
        VectorTools::point_value(state_dof_handler, *(it->second), p2,
            tmp_vector);
        double p2_value = tmp_vector(2);

        // pressure analysis
        return (p1_value - p2_value);

      }

      string
      GetType() const
      {
        return "point";
        // 1) point domain boundary face
        // 2) timelocal timedistributed
      }
      string
      GetName() const
      {
        return "Pressure_difference";
      }

  };

/****************************************************************************************/

template<typename VECTOR, int dealdim>
  class LocalBoundaryFunctionalDrag : public FunctionalInterface<
      CellDataContainer, FaceDataContainer, dealii::DoFHandler<dealdim>, VECTOR,
      dealdim>
  {
    private:
      double _density_fluid, _viscosity;
      double _drag_lift_constant;

    public:
      static void
      declare_params(ParameterReader &param_reader)
      {
        param_reader.SetSubsection("Local PDE parameters");
        param_reader.declare_entry("density_fluid", "1.0", Patterns::Double(0));
        param_reader.declare_entry("viscosity", "1.0", Patterns::Double(0));
        param_reader.declare_entry("drag_lift_constant", "1.0",
            Patterns::Double(0));
      }

      LocalBoundaryFunctionalDrag(ParameterReader &param_reader)
      {
        param_reader.SetSubsection("Local PDE parameters");
        _density_fluid = param_reader.get_double("density_fluid");
        _viscosity = param_reader.get_double("viscosity");
        _drag_lift_constant = param_reader.get_double("drag_lift_constant");
      }

      bool
      NeedTime() const
      {
        return false;
      }

      double
      BoundaryValue(
          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc)
      {
        unsigned int color = fdc.GetBoundaryIndicator();
        const auto &state_fe_face_values = fdc.GetFEFaceValuesState();
        unsigned int n_q_points = fdc.GetNQPoints();

        if (color == 80)
        {
          Tensor<1, 2> drag_lift_value;
          //double drag_lift_constant = 20; // 2D-1: 500 , 2D-2: 20

          vector<Vector<double> > _ufacevalues;
          vector<vector<Tensor<1, dealdim> > > _ufacegrads;

          _ufacevalues.resize(n_q_points, Vector<double>(3));
          _ufacegrads.resize(n_q_points, vector<Tensor<1, 2> >(3));

          fdc.GetFaceValuesState("state", _ufacevalues);
          fdc.GetFaceGradsState("state", _ufacegrads);

          const FEValuesExtractors::Vector velocities(0);
          const FEValuesExtractors::Scalar pressure(2);

          for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            Tensor<2, 2> fluid_pressure;
            fluid_pressure.clear();
            fluid_pressure[0][0] = -_ufacevalues[q_point](2);
            fluid_pressure[1][1] = -_ufacevalues[q_point](2);

            Tensor<2, 2> vgrads;
            vgrads.clear();
            vgrads[0][0] = _ufacegrads[q_point][0][0];
            vgrads[0][1] = _ufacegrads[q_point][0][1];
            vgrads[1][0] = _ufacegrads[q_point][1][0];
            vgrads[1][1] = _ufacegrads[q_point][1][1];

            drag_lift_value -= (fluid_pressure
                + _density_fluid * _viscosity * (vgrads))
                * state_fe_face_values.normal_vector(q_point)
                * state_fe_face_values.JxW(q_point);
          }

          drag_lift_value *= _drag_lift_constant;

          return drag_lift_value[0];
        }
        return 0.;
      }

      virtual void
      FaceValue_U(
          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
          dealii::Vector<double> &local_cell_vector, double scale)
      {

      }

      virtual void
      BoundaryValue_U(
          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
          dealii::Vector<double> &local_cell_vector, double scale)
      {
        unsigned int color = fdc.GetBoundaryIndicator();
        const auto &state_fe_values = fdc.GetFEFaceValuesState();
        unsigned int n_q_points = fdc.GetNQPoints();

        unsigned int n_dofs_per_cell = fdc.GetNDoFsPerCell();

        if (color == 80)
        {
          const FEValuesExtractors::Vector velocities(0);
          const FEValuesExtractors::Scalar pressure(2);

          for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            for (unsigned int k = 0; k < n_dofs_per_cell; k++)
            {
              Tensor<2, 2> fluid_pressure;
              fluid_pressure.clear();
              fluid_pressure[0][0] = -state_fe_values[pressure].value(k,
                  q_point);
              fluid_pressure[1][1] = -state_fe_values[pressure].value(k,
                  q_point);

              Tensor<2, 2> vgrads;
              vgrads.clear();
              vgrads[0][0] =
                  state_fe_values[velocities].gradient(k, q_point)[0][0];
              vgrads[0][1] =
                  state_fe_values[velocities].gradient(k, q_point)[0][1];
              vgrads[1][0] =
                  state_fe_values[velocities].gradient(k, q_point)[1][0];
              vgrads[1][1] =
                  state_fe_values[velocities].gradient(k, q_point)[1][1];

              local_cell_vector[k] += (-1. * _drag_lift_constant * scale
                  * (fluid_pressure + _density_fluid * _viscosity * (vgrads))
                  * state_fe_values.normal_vector(q_point)
                  * state_fe_values.JxW(q_point))[0];
            }
          }
        }
      }

      UpdateFlags
      GetFaceUpdateFlags() const
      {
        return update_values | update_quadrature_points | update_gradients
            | update_normal_vectors;
      }

      string
      GetType() const
      {
        return "boundary";
        // 1) point domain boundary face
        // 2) timelocal timedistributed
      }
      string
      GetName() const
      {
        return "Drag";
      }
  };

/****************************************************************************************/

template<typename VECTOR, int dealdim>
  class LocalCellFunctionalDrag : public FunctionalInterface<CellDataContainer,
      FaceDataContainer, dealii::DoFHandler<dealdim>, VECTOR, dealdim>
  {
    public:
      static void
      declare_params(ParameterReader &param_reader)
      {
        param_reader.SetSubsection("Local PDE parameters");
        param_reader.declare_entry("density_fluid", "1.0", Patterns::Double(0));
        param_reader.declare_entry("viscosity", "1.0", Patterns::Double(0));
        param_reader.declare_entry("drag_lift_constant", "1.0",
            Patterns::Double(0));
      }

      LocalCellFunctionalDrag(ParameterReader &param_reader)
          : _alpha(1. / 12.), _beta(1. / 6.)
      {
        param_reader.SetSubsection("Local PDE parameters");
        _density_fluid = param_reader.get_double("density_fluid");
        _viscosity = param_reader.get_double("viscosity");
        _drag_lift_constant = param_reader.get_double("drag_lift_constant");
      }

      bool
      NeedTime() const
      {
        return false;
      }

      template<class DH>
        void
        PreparePhiD(DH& dope_dh,
            const std::vector<unsigned int>& dofs_per_block)
        {

          auto& dof_handler = dope_dh.GetStateDoFHandler().GetDEALDoFHandler();
          std::vector<bool> boundary_dofs(dof_handler.n_dofs());
          std::vector<bool> component_mask(3, false);
          component_mask[0] = true;
//          std::set<unsigned int> boundary_indicators;
          std::set<types::boundary_id_t> boundary_indicators;
          boundary_indicators.insert(80);
          dealii::DoFTools::extract_boundary_dofs(dof_handler, component_mask,
              boundary_dofs, boundary_indicators);

          DOpEHelper::ReSizeVector(dof_handler.n_dofs(), dofs_per_block,
              _phi_d);
          for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
            if (boundary_dofs[i] == true)
              _phi_d(i) = 1;
        }

      double
      Value(
          const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc)
      {
        double erg = 0;
        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            cdc.GetFEValuesState();
        unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
        unsigned int n_q_points = cdc.GetNQPoints();
        const double h = cdc.GetCellDiameter();
        //        assert(this->_problem_type == "state");

        _uvalues.resize(n_q_points, Vector<double>(3));
        _ugrads.resize(n_q_points, vector<Tensor<1, 2> >(3));
        _phidgrads.resize(n_q_points, vector<Tensor<1, 2> >(3));
        _lap_u.resize(n_q_points, Vector<double>(3));
        _phidvalues.resize(n_q_points, Vector<double>(3));

        cdc.GetLaplaciansState("state", _lap_u);
        cdc.GetValuesState("state", _uvalues);
        cdc.GetGradsState("state", _ugrads);
        cdc.GetValuesState("phid", _phidvalues);
        cdc.GetGradsState("phid", _phidgrads);

        const double max_v = GetMaxU(_uvalues);

        const double delta = _alpha
            / (_viscosity * pow(h, -2.) + _beta * max_v * 1. / h);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          Tensor<2, 2> vgrads;
          vgrads.clear();
          vgrads[0][0] = _ugrads[q_point][0][0];
          vgrads[0][1] = _ugrads[q_point][0][1];
          vgrads[1][0] = _ugrads[q_point][1][0];
          vgrads[1][1] = _ugrads[q_point][1][1];

          Tensor<1, 2> pgrad;
          pgrad.clear();
          pgrad[0] = _ugrads[q_point][2][0];
          pgrad[1] = _ugrads[q_point][2][1];

          Tensor<1, 2> v;
          v.clear();
          v[0] = _uvalues[q_point](0);
          v[1] = _uvalues[q_point](1);

          Tensor<1, 2> convection_fluid = vgrads * v; //unterschied ob v*vgrads? Ja!
          double press = _uvalues[q_point](2);
          double incompressibility = vgrads[0][0] + vgrads[1][1];

          double res_0 = -_viscosity * _lap_u[q_point][0] + v * vgrads[0]
              + pgrad[0];
          double res_1 = -_viscosity * _lap_u[q_point][1] + v * vgrads[1]
              + pgrad[1];

          Tensor<2, 2> phid_grads_v;
          phid_grads_v[0][0] = _phidgrads[q_point][0][0];
          phid_grads_v[0][1] = _phidgrads[q_point][0][1];
          phid_grads_v[1][0] = _phidgrads[q_point][1][0];
          phid_grads_v[1][1] = _phidgrads[q_point][1][1];

          const Tensor<1, 2> phid_grads_p = _phidgrads[q_point][2];

          const double phid_p = _phidvalues[q_point][2]; //should be 0

          Tensor<1, 2> phid_v;
          phid_v[0] = _phidvalues[q_point][0];
          phid_v[1] = _phidvalues[q_point][1];

          const double div_phid_v = phid_grads_v[0][0] + phid_grads_v[1][1];

          Tensor<1, 2> Su = v * phid_grads_v + phid_grads_p;

          erg += (_viscosity * scalar_product(vgrads, phid_grads_v)
              + convection_fluid * phid_v - press * div_phid_v
              + incompressibility * phid_p) * state_fe_values.JxW(q_point);

          erg += delta
              * (res_0 * Su[0] + res_1 * Su[1] + incompressibility * div_phid_v) //unklar, ob hier '-'
              * state_fe_values.JxW(q_point);

        }
        erg *= -1. * _drag_lift_constant;
        return erg;
      }

      void
      Value_U(
          const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
          dealii::Vector<double> &local_cell_vector, double scale)
      {
        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            cdc.GetFEValuesState();
        unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
        unsigned int n_q_points = cdc.GetNQPoints();
        const double h = cdc.GetCellDiameter();
        //        assert(this->_problem_type == "state");

        _uvalues.resize(n_q_points, Vector<double>(3));
        _ugrads.resize(n_q_points, vector<Tensor<1, 2> >(3));
        _phidgrads.resize(n_q_points, vector<Tensor<1, 2> >(3));
        _lap_u.resize(n_q_points, Vector<double>(3));
        _phidvalues.resize(n_q_points, Vector<double>(3));

        cdc.GetLaplaciansState("state", _lap_u);
        cdc.GetValuesState("state", _uvalues);
        cdc.GetGradsState("state", _ugrads);
        cdc.GetValuesState("phid", _phidvalues);
        cdc.GetGradsState("phid", _phidgrads);

        const double max_v = GetMaxU(_uvalues);

        const double delta = _alpha
            / (_viscosity * pow(h, -2.) + _beta * max_v * 1. / h);

        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(2);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          Tensor<2, 2> vgrads;
          vgrads.clear();
          vgrads[0][0] = _ugrads[q_point][0][0];
          vgrads[0][1] = _ugrads[q_point][0][1];
          vgrads[1][0] = _ugrads[q_point][1][0];
          vgrads[1][1] = _ugrads[q_point][1][1];

          Tensor<1, 2> pgrad;
          pgrad.clear();
          pgrad[0] = _ugrads[q_point][2][0];
          pgrad[1] = _ugrads[q_point][2][1];

          Tensor<1, 2> v;
          v.clear();
          v[0] = _uvalues[q_point](0);
          v[1] = _uvalues[q_point](1);

          double press = _uvalues[q_point](2);
          double incompressibility = vgrads[0][0] + vgrads[1][1];

          double res_0 = -_viscosity * _lap_u[q_point][0] + v * vgrads[0]
              + pgrad[0];
          double res_1 = -_viscosity * _lap_u[q_point][1] + v * vgrads[1]
              + pgrad[1];

          Tensor<2, 2> phid_grads_v;
          phid_grads_v[0][0] = _phidgrads[q_point][0][0];
          phid_grads_v[0][1] = _phidgrads[q_point][0][1];
          phid_grads_v[1][0] = _phidgrads[q_point][1][0];
          phid_grads_v[1][1] = _phidgrads[q_point][1][1];

          const Tensor<1, 2> phid_grads_p = _phidgrads[q_point][2];

          const double phid_p = _phidvalues[q_point][2];

          Tensor<1, 2> phid_v;
          phid_v[0] = _phidvalues[q_point][0];
          phid_v[1] = _phidvalues[q_point][1];

          const double div_phid_v = phid_grads_v[0][0] + phid_grads_v[1][1];

          Tensor<1, 2> Su = v * phid_grads_v + phid_grads_p;

          for (unsigned int i = 0; i < n_dofs_per_cell; i++)
          {
            const Tensor<2, 2> phi_i_grads_v =
                state_fe_values[velocities].gradient(i, q_point);

            const Tensor<1, 2> phi_i_grad_p =
                state_fe_values[pressure].gradient(i, q_point);

            const double phi_i_p = state_fe_values[pressure].value(i, q_point);
            const Tensor<1, 2> phi_i_v = state_fe_values[velocities].value(i,
                q_point);
            const double div_phi_i_v = state_fe_values[velocities].divergence(i,
                q_point);
            Tensor<1, 2> phi_i_lap_v;
            phi_i_lap_v[0] =
                state_fe_values[velocities].hessian(i, q_point)[0][0][0]
                    + state_fe_values[velocities].hessian(i, q_point)[0][1][1];
            phi_i_lap_v[1] =
                state_fe_values[velocities].hessian(i, q_point)[1][0][0]
                    + state_fe_values[velocities].hessian(i, q_point)[1][1][1];

            double res_0 = -_viscosity * _lap_u[q_point][0] + v * vgrads[0]/*hier ok, da beides Tensor<1,dim>*/
            + pgrad[0];
            double res_1 = -_viscosity * _lap_u[q_point][1] + v * vgrads[1]
                + pgrad[1];

            double Dres_0 = -_viscosity * phi_i_lap_v[0] + phi_i_v * vgrads[0]
                + v * phi_i_grads_v[0] + phi_i_grad_p[0];
            double Dres_1 = -_viscosity * phi_i_lap_v[1] + phi_i_v * vgrads[1]
                + v * phi_i_grads_v[1] + phi_i_grad_p[1];

            Tensor<1, 2> Su = v * phid_grads_v + phid_grads_p;
            Tensor<1, 2> DSu = phi_i_v * phid_grads_v;

            local_cell_vector(i) -= scale * _drag_lift_constant
                * (_viscosity * scalar_product(phi_i_grads_v, phid_grads_v)
                    + (vgrads * phi_i_v + phi_i_grads_v * v) * phid_v
                    - phi_i_p * div_phid_v + div_phi_i_v * phid_p)
                * state_fe_values.JxW(q_point);

            local_cell_vector(i) += scale * delta
                * (res_0 * DSu[0] + Dres_0 * Su[0] + Dres_1 * Su[1]
                    + res_1 * DSu[1] + div_phi_i_v * div_phid_v) //unklar, ob hier '-'
                * state_fe_values.JxW(q_point);
          }
        }
      }

      UpdateFlags
      GetUpdateFlags() const
      {
        return update_values | update_quadrature_points | update_gradients
            | update_hessians;
      }

      string
      GetType() const
      {
        return "domain";
        // 1) point domain boundary face
        // 2) timelocal timedistributed
      }
      string
      GetName() const
      {
        return "DomainDrag";
      }

      const VECTOR*
      GetPhiD()
      {
        return &_phi_d;
      }
    private:
      double
      GetMaxU(vector<Vector<double> > vec) const
      {
        double max = 0;
        for (unsigned int i = 0; i < vec.size(); i++)
        {
          for (unsigned int j = 0; j < dealdim; j++)
          {
            if (std::fabs(vec[i][j]) > max)
              max = std::fabs(vec[i][j]);
          }
        }
        return max;
      }

      const double _alpha, _beta;

      double _density_fluid, _viscosity;
      double _drag_lift_constant;

      vector<vector<Tensor<1, dealdim> > > _ugrads;
      vector<Vector<double> > _lap_u;
      vector<Vector<double> > _uvalues;
      vector<Vector<double> > _phidvalues;
      vector<vector<Tensor<1, dealdim> > > _phidgrads;

      VECTOR _phi_d;
  };

/****************************************************************************************/

template<typename VECTOR, int dealdim>
  class LocalBoundaryFunctionalLift : public FunctionalInterface<
      CellDataContainer, FaceDataContainer, dealii::DoFHandler<dealdim>, VECTOR,
      dealdim>
  {
    private:
      mutable double time;
      double _density_fluid, _viscosity;
      double _drag_lift_constant;

    public:
      static void
      declare_params(ParameterReader &param_reader)
      {
        param_reader.SetSubsection("Local PDE parameters");
        param_reader.declare_entry("density_fluid", "1.0", Patterns::Double(0));
        param_reader.declare_entry("viscosity", "1.0", Patterns::Double(0));
        param_reader.declare_entry("drag_lift_constant", "1.0",
            Patterns::Double(0));
      }

      LocalBoundaryFunctionalLift(ParameterReader &param_reader)
      {
        param_reader.SetSubsection("Local PDE parameters");
        _density_fluid = param_reader.get_double("density_fluid");
        _viscosity = param_reader.get_double("viscosity");
        _drag_lift_constant = param_reader.get_double("drag_lift_constant");
      }

      bool
      NeedTime() const
      {
        return false;
      }

      double
      BoundaryValue(
          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc)
      {
        unsigned int color = fdc.GetBoundaryIndicator();
        const auto &state_fe_face_values = fdc.GetFEFaceValuesState();
        unsigned int n_q_points = fdc.GetNQPoints();

        if (color == 80)
        {
          Tensor<1, 2> drag_lift_value;
          //  double drag_lift_constant = 20;// 2D-1: 500 , 2D-2: 20

          vector<Vector<double> > _ufacevalues;
          vector<vector<Tensor<1, dealdim> > > _ufacegrads;

          _ufacevalues.resize(n_q_points, Vector<double>(3));
          _ufacegrads.resize(n_q_points, vector<Tensor<1, 2> >(3));

          fdc.GetFaceValuesState("state", _ufacevalues);
          fdc.GetFaceGradsState("state", _ufacegrads);

          const FEValuesExtractors::Vector velocities(0);
          const FEValuesExtractors::Scalar pressure(2);

          for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            Tensor<2, 2> fluid_pressure;
            fluid_pressure.clear();
            fluid_pressure[0][0] = -_ufacevalues[q_point](2);
            fluid_pressure[1][1] = -_ufacevalues[q_point](2);

            Tensor<2, 2> vgrads;
            vgrads.clear();
            vgrads[0][0] = _ufacegrads[q_point][0][0];
            vgrads[0][1] = _ufacegrads[q_point][0][1];
            vgrads[1][0] = _ufacegrads[q_point][1][0];
            vgrads[1][1] = _ufacegrads[q_point][1][1];

            drag_lift_value -= (fluid_pressure
                + _density_fluid * _viscosity * (vgrads))
                * state_fe_face_values.normal_vector(q_point)
                * state_fe_face_values.JxW(q_point);
          }

          drag_lift_value *= _drag_lift_constant;

          return drag_lift_value[1];
        }
        return 0.;
      }

      UpdateFlags
      GetFaceUpdateFlags() const
      {
        return update_values | update_quadrature_points | update_gradients
            | update_normal_vectors;
      }

      string
      GetType() const
      {
        return "boundary";
        // 1) point domain boundary face
        // 2) timelocal timedistributed
      }
      string
      GetName() const
      {
        return "Lift";
      }
  };

#endif
