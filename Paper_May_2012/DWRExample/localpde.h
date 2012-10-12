#ifndef _LocalPDEStokes_
#define _LocalPDEStokes_

#include "pdeinterface.h"
#include "celldatacontainer.h"
#include "facedatacontainer.h"
#include "myfunctions.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

template<typename VECTOR, int dealdim>
  class LocalPDENStokes : public PDEInterface<CellDataContainer,
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

      LocalPDENStokes(ParameterReader &param_reader)
          : _state_block_components(3, 0)
      {
        _state_block_components[2] = 1;

        param_reader.SetSubsection("Local PDE parameters");
        _density_fluid = param_reader.get_double("density_fluid");
        _viscosity = param_reader.get_double("viscosity");
        _drag_lift_constant = param_reader.get_double("drag_lift_constant");
      }

      // Domain values for cells
      void
      CellEquation(
          const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
          dealii::Vector<double> &local_cell_vector, double scale,
          double /*scale_ico*/)
      {
        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            cdc.GetFEValuesState();
        unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
        unsigned int n_q_points = cdc.GetNQPoints();
//        assert(this->_problem_type == "state");

        _uvalues.resize(n_q_points, Vector<double>(3));
        _ugrads.resize(n_q_points, vector<Tensor<1, 2> >(3));

        cdc.GetValuesState("last_newton_solution", _uvalues);
        cdc.GetGradsState("last_newton_solution", _ugrads);

        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(2);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          Tensor<2, 2> vgrads;
          vgrads[0][0] = _ugrads[q_point][0][0];
          vgrads[0][1] = _ugrads[q_point][0][1];
          vgrads[1][0] = _ugrads[q_point][1][0];
          vgrads[1][1] = _ugrads[q_point][1][1];

          Tensor<1, 2> v;
          v.clear();
          v[0] = _uvalues[q_point](0);
          v[1] = _uvalues[q_point](1);

          Tensor<1, 2> convection_fluid = vgrads * v;
          double press = _uvalues[q_point](2);
          double incompressibility = vgrads[0][0] + vgrads[1][1];

          for (unsigned int i = 0; i < n_dofs_per_cell; i++)
          {
            const Tensor<2, 2> phi_i_grads_v =
                state_fe_values[velocities].gradient(i, q_point);
            const double phi_i_p = state_fe_values[pressure].value(i, q_point);
            const Tensor<1, 2> phi_i_v = state_fe_values[velocities].value(i,
                q_point);
            const double div_phi_v = state_fe_values[velocities].divergence(i,
                q_point);

            local_cell_vector(i) += scale
                * (_viscosity * scalar_product(vgrads, phi_i_grads_v)
                    + convection_fluid * phi_i_v - press * div_phi_v
                    + incompressibility * phi_i_p)
                * state_fe_values.JxW(q_point);

          }
        }

      }
      void
      CellEquation_U(
          const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
          dealii::Vector<double> &local_cell_vector, double scale, double)
      {
        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            cdc.GetFEValuesState();
        unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
        unsigned int n_q_points = cdc.GetNQPoints();
        //        assert(this->_problem_type == "state");

        _uvalues.resize(n_q_points, Vector<double>(3));
        _ugrads.resize(n_q_points, vector<Tensor<1, 2> >(3));

        _zvalues.resize(n_q_points, Vector<double>(3));
        _zgrads.resize(n_q_points, vector<Tensor<1, 2> >(3));

        cdc.GetValuesState("state", _uvalues);
        cdc.GetGradsState("state", _ugrads);

        cdc.GetValuesState("last_newton_solution", _zvalues);
        cdc.GetGradsState("last_newton_solution", _zgrads);

        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(2);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          Tensor<2, 2> vgrads;
          vgrads[0][0] = _ugrads[q_point][0][0];
          vgrads[0][1] = _ugrads[q_point][0][1];
          vgrads[1][0] = _ugrads[q_point][1][0];
          vgrads[1][1] = _ugrads[q_point][1][1];

          Tensor<2, 2> zgrads;
          zgrads[0][0] = _zgrads[q_point][0][0];
          zgrads[0][1] = _zgrads[q_point][0][1];
          zgrads[1][0] = _zgrads[q_point][1][0];
          zgrads[1][1] = _zgrads[q_point][1][1];

          Tensor<1, 2> v;
          v.clear();
          v[0] = _uvalues[q_point](0);
          v[1] = _uvalues[q_point](1);

          Tensor<1, 2> z;
          z.clear();
          z[0] = _zvalues[q_point](0);
          z[1] = _zvalues[q_point](1);

          double dual_press = _zvalues[q_point](2);
          double incompressibility = zgrads[0][0] + zgrads[1][1];

          for (unsigned int i = 0; i < n_dofs_per_cell; i++)
          {
            const Tensor<2, 2> phi_i_grads_z =
                state_fe_values[velocities].gradient(i, q_point);
            const double phi_i_q = state_fe_values[pressure].value(i, q_point);
            const Tensor<1, 2> phi_i_z = state_fe_values[velocities].value(i,
                q_point);
            const double div_phi_z = state_fe_values[velocities].divergence(i,
                q_point);

            local_cell_vector(i) += scale
                * (_viscosity * scalar_product(zgrads, phi_i_grads_z)
                    + (vgrads * phi_i_z) * z + (phi_i_grads_z * v) * z
                    + dual_press * div_phi_z - incompressibility * phi_i_q)
                * state_fe_values.JxW(q_point);

          }
        }
      }

//      void
//      CellMatrix(
//          const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
//          FullMatrix<double> &local_entry_matrix, double scale,
//          double /*scale_ico*/)
//      {
//
//        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
//            cdc.GetFEValuesState();
//        unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
//        unsigned int n_q_points = cdc.GetNQPoints();
//
//        const FEValuesExtractors::Vector velocities(0);
//        const FEValuesExtractors::Scalar pressure(2);
//
//        _uvalues.resize(n_q_points, Vector<double>(3));
//        _ugrads.resize(n_q_points, vector<Tensor<1, 2> >(3));
//
//        cdc.GetValuesState("last_newton_solution", _uvalues);
//        cdc.GetGradsState("last_newton_solution", _ugrads);
//
//        std::vector<Tensor<1, 2> > phi_v(n_dofs_per_cell);
//        std::vector<Tensor<2, 2> > phi_grads_v(n_dofs_per_cell);
//        std::vector<double> phi_p(n_dofs_per_cell);
//        std::vector<double> div_phi_v(n_dofs_per_cell);
//
//        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
//        {
//          for (unsigned int k = 0; k < n_dofs_per_cell; k++)
//          {
//            phi_v[k] = state_fe_values[velocities].value(k, q_point);
//            phi_grads_v[k] = state_fe_values[velocities].gradient(k, q_point);
//            phi_p[k] = state_fe_values[pressure].value(k, q_point);
//            div_phi_v[k] = state_fe_values[velocities].divergence(k, q_point);
//          }
//
//          Tensor<2, 2> vgrads;
//          vgrads.clear();
//          vgrads[0][0] = _ugrads[q_point][0][0];
//          vgrads[0][1] = _ugrads[q_point][0][1];
//          vgrads[1][0] = _ugrads[q_point][1][0];
//          vgrads[1][1] = _ugrads[q_point][1][1];
//
//          Tensor<1, 2> v;
//          v[0] = _uvalues[q_point](0);
//          v[1] = _uvalues[q_point](1);
//
//          for (unsigned int i = 0; i < n_dofs_per_cell; i++)
//          {
//            for (unsigned int j = 0; j < n_dofs_per_cell; j++)
//            {
//              Tensor<1, 2> convection_fluid_LinV = phi_grads_v[j] * v
//                  + vgrads * phi_v[j];
//
//              local_entry_matrix(i, j) +=
//                  scale
//                      * (_viscosity
//                          * scalar_product(phi_grads_v[j], phi_grads_v[i])
//                          + convection_fluid_LinV * phi_v[i]
//                          - phi_p[j] * div_phi_v[i]
//                          + (phi_grads_v[j][0][0] + phi_grads_v[j][1][1])
//                              * phi_p[i]) * state_fe_values.JxW(q_point);
//            }
//          }
//        }
//
//      }
      void
      CellMatrix(
          const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
          FullMatrix<double> &local_entry_matrix, double scale,
          double scale_ico)
      {
        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            cdc.GetFEValuesState();
        unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
        unsigned int n_q_points = cdc.GetNQPoints();
        //unsigned int material_id = cdc.GetMaterialId();

        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(2);

        _uvalues.resize(n_q_points, Vector<double>(3));
        _ugrads.resize(n_q_points, vector<Tensor<1, 2> >(3));

        cdc.GetValuesState("last_newton_solution", _uvalues);
        cdc.GetGradsState("last_newton_solution", _ugrads);

        std::vector<Tensor<1, 2> > phi_v(n_dofs_per_cell);
        std::vector<Tensor<2, 2> > phi_grads_v(n_dofs_per_cell);
        std::vector<double> phi_p(n_dofs_per_cell);
        std::vector<double> div_phi_v(n_dofs_per_cell);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          for (unsigned int k = 0; k < n_dofs_per_cell; k++)
          {
            phi_v[k] = state_fe_values[velocities].value(k, q_point);
            phi_grads_v[k] = state_fe_values[velocities].gradient(k, q_point);
            phi_p[k] = state_fe_values[pressure].value(k, q_point);
            div_phi_v[k] = state_fe_values[velocities].divergence(k, q_point);
          }

          Tensor<2, 2> vgrads;
          vgrads.clear();
          vgrads[0][0] = _ugrads[q_point][0][0];
          vgrads[0][1] = _ugrads[q_point][0][1];
          vgrads[1][0] = _ugrads[q_point][1][0];
          vgrads[1][1] = _ugrads[q_point][1][1];

          Tensor<1, 2> v;
          v[0] = _uvalues[q_point](0);
          v[1] = _uvalues[q_point](1);

          for (unsigned int i = 0; i < n_dofs_per_cell; i++)
          {
            Tensor<2, dealdim> fluid_pressure_LinP;
            fluid_pressure_LinP.clear();
            fluid_pressure_LinP[0][0] = -phi_p[i];
            fluid_pressure_LinP[1][1] = -phi_p[i];

            Tensor<1, 2> convection_fluid_LinV = phi_grads_v[i] * v
                + vgrads * phi_v[i];

            for (unsigned int j = 0; j < n_dofs_per_cell; j++)
            {
              local_entry_matrix(j, i) += scale
                  * (convection_fluid_LinV * phi_v[j]
                      + _viscosity
                          * scalar_product(phi_grads_v[i], phi_grads_v[j]))
                  * state_fe_values.JxW(q_point);

              local_entry_matrix(j, i) +=
                  scale
                      * (scalar_product(fluid_pressure_LinP, phi_grads_v[j])
                          + (phi_grads_v[i][0][0] + phi_grads_v[i][1][1])
                              * phi_p[j]) * state_fe_values.JxW(q_point);
            }
          }
        }

      }

      void
      CellMatrix_T(
          const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
          FullMatrix<double> &local_entry_matrix, double scale, double)
      {

        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            cdc.GetFEValuesState();
        unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
        unsigned int n_q_points = cdc.GetNQPoints();

        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(2);

        _uvalues.resize(n_q_points, Vector<double>(3));
        _ugrads.resize(n_q_points, vector<Tensor<1, 2> >(3));

        cdc.GetValuesState("state", _uvalues);
        cdc.GetGradsState("state", _ugrads);

        std::vector<Tensor<1, 2> > phi_z(n_dofs_per_cell);
        std::vector<Tensor<2, 2> > phi_grads_z(n_dofs_per_cell);
        std::vector<double> phi_q(n_dofs_per_cell);
        std::vector<double> div_phi_z(n_dofs_per_cell);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          for (unsigned int k = 0; k < n_dofs_per_cell; k++)
          {
            phi_z[k] = state_fe_values[velocities].value(k, q_point);
            phi_grads_z[k] = state_fe_values[velocities].gradient(k, q_point);
            phi_q[k] = state_fe_values[pressure].value(k, q_point);
            div_phi_z[k] = state_fe_values[velocities].divergence(k, q_point);
          }

          Tensor<2, 2> vgrads;
          vgrads.clear();
          vgrads[0][0] = _ugrads[q_point][0][0];
          vgrads[0][1] = _ugrads[q_point][0][1];
          vgrads[1][0] = _ugrads[q_point][1][0];
          vgrads[1][1] = _ugrads[q_point][1][1];

          Tensor<1, 2> v;
          v[0] = _uvalues[q_point](0);
          v[1] = _uvalues[q_point](1);

          for (unsigned int i = 0; i < n_dofs_per_cell; i++)
          {
            Tensor<1, 2> convection_fluid_LinV = phi_grads_z[i] * v
                + vgrads * phi_z[i];
            for (unsigned int j = 0; j < n_dofs_per_cell; j++)
            {
              local_entry_matrix(i, j) += scale
                  * (_viscosity * scalar_product(phi_grads_z[j], phi_grads_z[i])
                      + convection_fluid_LinV * phi_z[j]
                      + phi_q[j] * div_phi_z[i] - div_phi_z[j] * phi_q[i])
                  * state_fe_values.JxW(q_point);
            }
          }
        }
      }

      void
      CellRightHandSide(
          const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
          dealii::Vector<double> &local_cell_vector, double scale)
      {

      }

      void
      FaceEquation(
          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>&,
          dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/,
          double)
      {
      }

      void
      FaceEquation_U(
          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>&,
          dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/,
          double)
      {
      }

      void
      FaceMatrix_T(
          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>&,
          dealii::FullMatrix<double> &/*local_entry_matrix*/, double, double)
      {
      }

      void
      FaceMatrix(
          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>&,
          dealii::FullMatrix<double> &/*local_entry_matrix*/, double, double)
      {
      }

      // Values for boundary integrals
      void
      BoundaryEquation(
          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
          dealii::Vector<double> &local_cell_vector, double scale,
          double /*scale_ico*/)
      {

      }

      // Values for boundary integrals
      void
      BoundaryEquation_U(
          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
          dealii::Vector<double> &local_cell_vector, double scale,
          double /*scale_ico*/)
      {

      }

      void
      BoundaryRightHandSide(
          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>&,
          dealii::Vector<double> &local_cell_vector __attribute__((unused)),
          double scale __attribute__((unused)))
      {

      }
      void
      BoundaryMatrix(
          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
          dealii::FullMatrix<double> &local_entry_matrix, double /*scale_ico*/,
          double /*scale_ico*/)
      {

      }

      void
      FaceRightHandSide(
          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
          dealii::Vector<double> &local_rhs, double scale)
      {

      }

      void
      StrongBoundaryResidual(
          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc_w,
          double& sum, double scale)
      {
        const unsigned int n_q_points = fdc.GetNQPoints();
        const unsigned int color = fdc.GetBoundaryIndicator();

        _uvalues.resize(n_q_points, Vector<double>(3));
        _ugrads.resize(n_q_points, vector<Tensor<1, dealdim> >(3));
        _PI_h_z.resize(n_q_points);

        fdc.GetFaceGradsState("state", _ugrads);
        fdc.GetFaceValuesState("state", _uvalues);

        fdc_w.GetFaceValuesState("weight_for_primal_residual", _PI_h_z);

        if (color == 1)
          for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            sum -= scale //normale ist hier (1,0)
                * (_viscosity
                    * (_ugrads[q_point][0][0] * _PI_h_z[q_point][0]
                        + _ugrads[q_point][1][0] * _PI_h_z[q_point][1])
                    - _uvalues[q_point][2] * _PI_h_z[q_point][0])
                * fdc.GetFEFaceValuesState().JxW(q_point);
          }

      }

      void
      StrongCellResidual(
          const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
          const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc_w,
          double& sum, double scale, double)
      {

        unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
        unsigned int n_q_points = cdc.GetNQPoints();
        const DOpEWrapper::FEValues<dealdim> &state_fe_values =
            cdc.GetFEValuesState();
        const DOpEWrapper::FEValues<dealdim> &state_fe_values_weight =
            cdc_w.GetFEValuesState();

        _fvalues.resize(n_q_points, Vector<double>(3));

        _PI_h_z.resize(n_q_points, Vector<double>(3));
        _lap_u.resize(n_q_points, Vector<double>(3));
        _ugrads.resize(n_q_points, vector<Tensor<1, dealdim> >(3));
        _uvalues.resize(n_q_points, Vector<double>(3));

        cdc.GetLaplaciansState("state", _lap_u);
        cdc.GetGradsState("state", _ugrads);
        cdc.GetValuesState("state", _uvalues);

        cdc_w.GetValuesState("weight_for_primal_residual", _PI_h_z);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {

          Tensor<2, 2> vgrads;
          vgrads[0][0] = _ugrads[q_point][0][0];
          vgrads[0][1] = _ugrads[q_point][0][1];
          vgrads[1][0] = _ugrads[q_point][1][0];
          vgrads[1][1] = _ugrads[q_point][1][1];

          Tensor<1, 2> v;
          v.clear();
          v[0] = _uvalues[q_point](0);
          v[1] = _uvalues[q_point](1);

          Tensor<1, 2> convection_fluid = vgrads * v;

          const double divu = _ugrads[q_point][0][0] + _ugrads[q_point][1][1];
          //f - (Starke form der Gleichung!)
          double res_0 = _viscosity * _lap_u[q_point][0] - convection_fluid[0]
              - _ugrads[q_point][2][0];
          double res_1 = _viscosity * _lap_u[q_point][1] - convection_fluid[1]
              - _ugrads[q_point][2][1];

          sum += scale
              * (res_0 * _PI_h_z[q_point][0] + res_1 * _PI_h_z[q_point][1]
                  - divu * _PI_h_z[q_point][2]) * state_fe_values.JxW(q_point);
        }
      }

      void
      StrongFaceResidual(
          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc_w,
          double& sum, double scale)
      {
        const unsigned int n_q_points = fdc.GetNQPoints();
        const unsigned int material_id = fdc.GetMaterialId();
        const unsigned int material_id_nbr = fdc.GetNbrMaterialId();

        _ugrads.resize(n_q_points, vector<Tensor<1, dealdim> >(3));
        _ugrads_nbr.resize(n_q_points, vector<Tensor<1, dealdim> >(3));

        _PI_h_z.resize(n_q_points);

        fdc.GetFaceGradsState("state", _ugrads);
        fdc.GetNbrFaceGradsState("state", _ugrads_nbr);

        fdc_w.GetFaceValuesState("weight_for_primal_residual", _PI_h_z);
        vector<Vector<double> > jump(n_q_points, Vector<double>(2));

        for (unsigned int q = 0; q < n_q_points; q++)
        { //Druck wird vernachlaessigt da Druckapproximation stetig
          jump[q][0] = _viscosity
              * ((_ugrads_nbr[q][0][0] - _ugrads[q][0][0])
                  * fdc.GetFEFaceValuesState().normal_vector(q)[0]
                  + (_ugrads_nbr[q][0][1] - _ugrads[q][0][1])
                      * fdc.GetFEFaceValuesState().normal_vector(q)[1]);

          jump[q][1] = _viscosity
              * ((_ugrads_nbr[q][1][0] - _ugrads[q][1][0])
                  * fdc.GetFEFaceValuesState().normal_vector(q)[0]
                  + (_ugrads_nbr[q][1][1] - _ugrads[q][1][1])
                      * fdc.GetFEFaceValuesState().normal_vector(q)[1]);
        }

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          sum += scale
              * (jump[q_point][0] * _PI_h_z[q_point][0]
                  + jump[q_point][1] * _PI_h_z[q_point][1])
              * fdc.GetFEFaceValuesState().JxW(q_point);
        }

      }

      void
      StrongBoundaryResidual_U(
          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc_w,
          double& sum, double scale)
      {
        const unsigned int n_q_points = fdc.GetNQPoints();
        const unsigned int color = fdc.GetBoundaryIndicator();

        _zgrads.resize(n_q_points, vector<Tensor<1, dealdim> >(3));
        _zvalues.resize(n_q_points, Vector<double>(3));

        _PI_h_u_grads.resize(n_q_points, vector<Tensor<1, dealdim> >(3));

        fdc.GetFaceGradsState("adjoint_for_ee", _zgrads);
        fdc.GetFaceValuesState("adjoint_for_ee", _zvalues);

        fdc_w.GetFaceGradsState("weight_for_dual_residual", _PI_h_u_grads);

        if (color == 1)
        {
          _uvalues.resize(n_q_points, Vector<double>(3));
          fdc.GetFaceValuesState("state", _uvalues);

          for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            //negative Wert des Randintegrals wird addiert.
            sum -= scale
                * (_viscosity //erinnerung: n = (1,0)
                    * (_zgrads[q_point][0][0] * _PI_h_u[q_point][0]
                        + _zgrads[q_point][1][0] * _PI_h_u[q_point][1])
                    + _uvalues[q_point][0]/* =v*n */
                        * (_zvalues[q_point][0] * _PI_h_u[q_point][0]
                            + _zvalues[q_point][1] * _PI_h_u[q_point][1])
                    + _zvalues[q_point][2] * _PI_h_u[q_point][0])
                * fdc.GetFEFaceValuesState().JxW(q_point);
          }
        }
        else if (color == 80)
          for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            Tensor<2, 2> fluid_pressure;
            fluid_pressure.clear();
            fluid_pressure[0][0] = -_PI_h_u[q_point](2);
            fluid_pressure[1][1] = -_PI_h_u[q_point](2);

            Tensor<2, 2> vgrads;
            vgrads.clear();
            vgrads[0][0] = _PI_h_u_grads[q_point][0][0];
            vgrads[0][1] = _PI_h_u_grads[q_point][0][1];
            vgrads[1][0] = _PI_h_u_grads[q_point][1][0];
            vgrads[1][1] = _PI_h_u_grads[q_point][1][1];

            sum -= _drag_lift_constant
                * ((fluid_pressure + _density_fluid * _viscosity * (vgrads))
                    * fdc.GetFEFaceValuesState().normal_vector(q_point)
                    * fdc.GetFEFaceValuesState().JxW(q_point))[0];
          }

      }

      void
      StrongCellResidual_U(
          const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
          const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc_w,
          double& sum, double scale)
      {

        unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
        unsigned int n_q_points = cdc.GetNQPoints();
        const DOpEWrapper::FEValues<dealdim> &state_fe_values =
            cdc.GetFEValuesState();
        const DOpEWrapper::FEValues<dealdim> &state_fe_values_weight =
            cdc_w.GetFEValuesState();

        _fvalues.resize(n_q_points, Vector<double>(3));

        _PI_h_u.resize(n_q_points, Vector<double>(3));
        _lap_z.resize(n_q_points, Vector<double>(3));
        _uvalues.resize(n_q_points, Vector<double>(3));
        _zvalues.resize(n_q_points, Vector<double>(3));
        _ugrads.resize(n_q_points, vector<Tensor<1, dealdim> >(3));
        _zgrads.resize(n_q_points, vector<Tensor<1, dealdim> >(3));

        cdc.GetLaplaciansState("adjoint_for_ee", _lap_z);
        cdc.GetGradsState("adjoint_for_ee", _zgrads);
        cdc.GetValuesState("adjoint_for_ee", _zvalues);
        cdc.GetValuesState("state", _uvalues);
        cdc.GetGradsState("state", _ugrads);

        cdc_w.GetValuesState("weight_for_dual_residual", _PI_h_u);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          Tensor<1, dealdim> u;
          u[0] = _uvalues[q_point][0];
          u[1] = _uvalues[q_point][1];

          Tensor<1, dealdim> z;
          z[0] = _zvalues[q_point][0];
          z[1] = _zvalues[q_point][1];

          Tensor<2, dealdim> gradu_t;
          gradu_t[0][0] = _ugrads[q_point][0][0];
          gradu_t[0][1] = _ugrads[q_point][1][0];
          gradu_t[1][0] = _ugrads[q_point][0][1];
          gradu_t[1][1] = _ugrads[q_point][1][1];

          Tensor<2, dealdim> gradz;
          gradz[0][0] = _zgrads[q_point][0][0];
          gradz[0][1] = _zgrads[q_point][0][1];
          gradz[1][0] = _zgrads[q_point][1][0];
          gradz[1][1] = _zgrads[q_point][1][1];

          const double divz = _zgrads[q_point][0][0] + _zgrads[q_point][1][1];
          const double divu = _ugrads[q_point][0][0] + _ugrads[q_point][1][1];

          double res_0 = _viscosity * _lap_z[q_point][0]
              + divu * _zvalues[q_point][0] + u * gradz[0] - gradu_t[0] * z
              + _zgrads[q_point][2][0];
          double res_1 = _viscosity * _lap_z[q_point][1]
              + divu * _zvalues[q_point][1] + u * gradz[1] - gradu_t[1] * z
              + _zgrads[q_point][2][1];

          sum += scale
              * (res_0 * _PI_h_u[q_point][0] + res_1 * _PI_h_u[q_point][1]
                  + divz * _PI_h_u[q_point][2]) * state_fe_values.JxW(q_point);
        }
      }

      void
      StrongFaceResidual_U(
          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc_w,
          double& sum, double scale)
      {
        const unsigned int n_q_points = fdc.GetNQPoints();
        const unsigned int material_id = fdc.GetMaterialId();
        const unsigned int material_id_nbr = fdc.GetNbrMaterialId();

        _zgrads.resize(n_q_points, vector<Tensor<1, dealdim> >(3));
        _zgrads_nbr.resize(n_q_points, vector<Tensor<1, dealdim> >(3));
        _PI_h_u.resize(n_q_points);

        fdc.GetFaceGradsState("adjoint_for_ee", _zgrads);
        fdc.GetNbrFaceGradsState("adjoint_for_ee", _zgrads_nbr);

        fdc_w.GetFaceValuesState("weight_for_dual_residual", _PI_h_u);
        vector<Vector<double> > jump(n_q_points, Vector<double>(2));

        for (unsigned int q = 0; q < n_q_points; q++)
        {
          jump[q][0] = _viscosity * (_zgrads_nbr[q][0][0] - _zgrads[q][0][0])
              * fdc.GetFEFaceValuesState().normal_vector(q)[0]
              + _viscosity * (_zgrads_nbr[q][0][1] - _zgrads[q][0][1])
                  * fdc.GetFEFaceValuesState().normal_vector(q)[1];

          jump[q][1] = _viscosity * (_zgrads_nbr[q][1][0] - _zgrads[q][1][0])
              * fdc.GetFEFaceValuesState().normal_vector(q)[0]
              + _viscosity * (_zgrads_nbr[q][1][1] - _zgrads[q][1][1])
                  * fdc.GetFEFaceValuesState().normal_vector(q)[1];
        }

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          sum += scale
              * (jump[q_point][0] * _PI_h_u[q_point][0]
                  + jump[q_point][1] * _PI_h_u[q_point][1])
              * fdc.GetFEFaceValuesState().JxW(q_point);
        }

      }

      UpdateFlags
      GetUpdateFlags() const
      {
        return update_values | update_gradients | update_hessians
            | update_quadrature_points;
      }

      UpdateFlags
      GetFaceUpdateFlags() const
      {
        return update_values | update_gradients | update_normal_vectors
            | update_quadrature_points;
      }

      unsigned int
      GetStateNBlocks() const
      {
        return 2;
      }
      std::vector<unsigned int>&
      GetStateBlockComponent()
      {
        return _state_block_components;
      }
      const std::vector<unsigned int>&
      GetStateBlockComponent() const
      {
        return _state_block_components;
      }

      virtual bool
      HasFaces() const
      {
        return true;
      }

    protected:
      vector<Vector<double> > _fvalues;

      vector<Vector<double> > _uvalues;
      vector<vector<Tensor<1, dealdim> > > _ugrads;
      vector<vector<Tensor<1, dealdim> > > _ugrads_nbr;
      vector<Vector<double> > _lap_u;

      vector<Vector<double> > _zvalues;
      vector<vector<Tensor<1, dealdim> > > _zgrads;
      vector<vector<Tensor<1, dealdim> > > _zgrads_nbr;
      vector<Vector<double> > _lap_z;
      vector<vector<Tensor<1, dealdim> > > _PI_h_zgrads, _PI_h_u_grads;
      vector<Vector<double> > _PI_h_z, _PI_h_u;

      // face values
      vector<vector<Tensor<1, dealdim> > > _ufacegrads;

      vector<unsigned int> _state_block_components;

      double _density_fluid, _viscosity;
      double _drag_lift_constant;
  }
  ;
/***************************************************************************************/

template<typename VECTOR, int dealdim>
  class LocalPDENStokesStab : public LocalPDENStokes<VECTOR, dealdim>
  {

    public:
      static void
      declare_params(ParameterReader &param_reader)
      {
        LocalPDENStokes<VECTOR, dealdim>::declare_params(param_reader);
      }

      LocalPDENStokesStab(ParameterReader &param_reader)
          : LocalPDENStokes<VECTOR, dealdim>(param_reader), _alpha(1. / 12.), _beta(
              1. / 6.)
      {
      }

      // Domain values for cells
      void
      CellEquation(
          const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
          dealii::Vector<double> &local_cell_vector, double scale,
          double /*scale_ico*/)
      {
        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            cdc.GetFEValuesState();
        unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
        unsigned int n_q_points = cdc.GetNQPoints();
        const double h = cdc.GetCellDiameter();
        //        assert(this->_problem_type == "state");

        LocalPDENStokes<VECTOR, dealdim>::_uvalues.resize(n_q_points,
            Vector<double>(3));
        LocalPDENStokes<VECTOR, dealdim>::_ugrads.resize(n_q_points,
            vector<Tensor<1, 2> >(3));
        LocalPDENStokes<VECTOR, dealdim>::_lap_u.resize(n_q_points,
            Vector<double>(3));

        cdc.GetLaplaciansState("last_newton_solution",
            LocalPDENStokes<VECTOR, dealdim>::_lap_u);
        cdc.GetValuesState("last_newton_solution",
            LocalPDENStokes<VECTOR, dealdim>::_uvalues);
        cdc.GetGradsState("last_newton_solution",
            LocalPDENStokes<VECTOR, dealdim>::_ugrads);

        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(2);

        const double max_v = GetMaxU(
            LocalPDENStokes<VECTOR, dealdim>::_uvalues);

        const double delta = _alpha
            / (LocalPDENStokes<VECTOR, dealdim>::_viscosity * pow(h, -2.)
                + _beta * max_v * 1. / h);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          Tensor<2, 2> vgrads;
          vgrads.clear();
          vgrads[0][0] =
              LocalPDENStokes<VECTOR, dealdim>::_ugrads[q_point][0][0];
          vgrads[0][1] =
              LocalPDENStokes<VECTOR, dealdim>::_ugrads[q_point][0][1];
          vgrads[1][0] =
              LocalPDENStokes<VECTOR, dealdim>::_ugrads[q_point][1][0];
          vgrads[1][1] =
              LocalPDENStokes<VECTOR, dealdim>::_ugrads[q_point][1][1];

          Tensor<1, 2> pgrad;
          pgrad.clear();
          pgrad[0] = LocalPDENStokes<VECTOR, dealdim>::_ugrads[q_point][2][0];
          pgrad[1] = LocalPDENStokes<VECTOR, dealdim>::_ugrads[q_point][2][1];

          Tensor<1, 2> v;
          v.clear();
          v[0] = LocalPDENStokes<VECTOR, dealdim>::_uvalues[q_point](0);
          v[1] = LocalPDENStokes<VECTOR, dealdim>::_uvalues[q_point](1);

          Tensor<1, 2> convection_fluid = vgrads * v; //unterschied ob v*vgrads?
          double press = LocalPDENStokes<VECTOR, dealdim>::_uvalues[q_point](2);
          double incompressibility = vgrads[0][0] + vgrads[1][1];

          double res_0 = -LocalPDENStokes<VECTOR, dealdim>::_viscosity
              * LocalPDENStokes<VECTOR, dealdim>::_lap_u[q_point][0]
              + v * vgrads[0] + pgrad[0];
          double res_1 = -LocalPDENStokes<VECTOR, dealdim>::_viscosity
              * LocalPDENStokes<VECTOR, dealdim>::_lap_u[q_point][1]
              + v * vgrads[1] + pgrad[1];

          for (unsigned int i = 0; i < n_dofs_per_cell; i++)
          {
            const Tensor<2, 2> phi_i_grads_v =
                state_fe_values[velocities].gradient(i, q_point);

            const Tensor<1, 2> phi_i_grads_p =
                state_fe_values[pressure].gradient(i, q_point);

            const double phi_i_p = state_fe_values[pressure].value(i, q_point);

            const Tensor<1, 2> phi_i_v = state_fe_values[velocities].value(i,
                q_point);

            const double div_phi_v = state_fe_values[velocities].divergence(i,
                q_point);

            Tensor<1, 2> Su = phi_i_grads_v * v + phi_i_grads_p;

            local_cell_vector(i) += scale
                * (LocalPDENStokes<VECTOR, dealdim>::_viscosity
                    * scalar_product(vgrads, phi_i_grads_v)
                    + convection_fluid * phi_i_v - press * div_phi_v
                    + incompressibility * phi_i_p)
                * state_fe_values.JxW(q_point);

            local_cell_vector(i) +=
                scale * delta
                    * (res_0 * Su[0] + res_1 * Su[1]
                        + incompressibility * div_phi_v) //unklar, ob hier '-'
                    * state_fe_values.JxW(q_point);
          }
        }

      }

      void
      CellEquation_U(
          const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
          dealii::Vector<double> &local_cell_vector, double scale, double)
      {
        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            cdc.GetFEValuesState();
        unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
        unsigned int n_q_points = cdc.GetNQPoints();
        double h = cdc.GetCellDiameter();
        //        assert(this->_problem_type == "state");

        LocalPDENStokes<VECTOR, dealdim>::_uvalues.resize(n_q_points,
            Vector<double>(3));
        LocalPDENStokes<VECTOR, dealdim>::_ugrads.resize(n_q_points,
            vector<Tensor<1, 2> >(3));

        LocalPDENStokes<VECTOR, dealdim>::_zvalues.resize(n_q_points,
            Vector<double>(3));
        LocalPDENStokes<VECTOR, dealdim>::_zgrads.resize(n_q_points,
            vector<Tensor<1, 2> >(3));

        cdc.GetValuesState("state", LocalPDENStokes<VECTOR, dealdim>::_uvalues);
        cdc.GetGradsState("state", LocalPDENStokes<VECTOR, dealdim>::_ugrads);

        cdc.GetValuesState("last_newton_solution",
            LocalPDENStokes<VECTOR, dealdim>::_zvalues);
        cdc.GetGradsState("last_newton_solution",
            LocalPDENStokes<VECTOR, dealdim>::_zgrads);

        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(2);

        const double max_v = GetMaxU(
            LocalPDENStokes<VECTOR, dealdim>::_uvalues);

        const double delta = _alpha
            / ((LocalPDENStokes<VECTOR, dealdim>::_viscosity * pow(h, -2.)
                + _beta * max_v * 1. / h));

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          Tensor<2, 2> vgrads;
          vgrads[0][0] =
              LocalPDENStokes<VECTOR, dealdim>::_ugrads[q_point][0][0];
          vgrads[0][1] =
              LocalPDENStokes<VECTOR, dealdim>::_ugrads[q_point][0][1];
          vgrads[1][0] =
              LocalPDENStokes<VECTOR, dealdim>::_ugrads[q_point][1][0];
          vgrads[1][1] =
              LocalPDENStokes<VECTOR, dealdim>::_ugrads[q_point][1][1];

          Tensor<2, 2> zgrads;
          zgrads[0][0] =
              LocalPDENStokes<VECTOR, dealdim>::_zgrads[q_point][0][0];
          zgrads[0][1] =
              LocalPDENStokes<VECTOR, dealdim>::_zgrads[q_point][0][1];
          zgrads[1][0] =
              LocalPDENStokes<VECTOR, dealdim>::_zgrads[q_point][1][0];
          zgrads[1][1] =
              LocalPDENStokes<VECTOR, dealdim>::_zgrads[q_point][1][1];

          Tensor<1, 2> qgrads;
          qgrads = LocalPDENStokes<VECTOR, dealdim>::_zgrads[q_point][2];

          Tensor<1, 2> v;
          v.clear();
          v[0] = LocalPDENStokes<VECTOR, dealdim>::_uvalues[q_point](0);
          v[1] = LocalPDENStokes<VECTOR, dealdim>::_uvalues[q_point](1);

          Tensor<1, 2> z;
          z.clear();
          z[0] = LocalPDENStokes<VECTOR, dealdim>::_zvalues[q_point](0);
          z[1] = LocalPDENStokes<VECTOR, dealdim>::_zvalues[q_point](1);

          double dual_press =
              LocalPDENStokes<VECTOR, dealdim>::_zvalues[q_point](2);
          double incompressibility = zgrads[0][0] + zgrads[1][1];

          for (unsigned int i = 0; i < n_dofs_per_cell; i++)
          {
            const Tensor<2, 2> phi_i_grads_z =
                state_fe_values[velocities].gradient(i, q_point);
            const Tensor<1, 2> phi_i_grads_q =
                state_fe_values[pressure].gradient(i, q_point);

            const double phi_i_q = state_fe_values[pressure].value(i, q_point);
            const Tensor<1, 2> phi_i_z = state_fe_values[velocities].value(i,
                q_point);
            const double div_phi_z = state_fe_values[velocities].divergence(i,
                q_point);

            Tensor<1, 3> Su_phi;
            Su_phi[0] = phi_i_grads_z[0] * v + phi_i_grads_q[0];
            Su_phi[1] = phi_i_grads_z[1] * v + phi_i_grads_q[1];
            Su_phi[2] = phi_i_grads_z[0][0] + phi_i_grads_z[1][1];

            Tensor<1, 3> Su_z;
            Su_z[0] = zgrads[0] * v + qgrads[0];
            Su_z[1] = zgrads[1] * v + qgrads[1];
            Su_z[2] = zgrads[0][0] + zgrads[1][1];

            local_cell_vector(i) += scale
                * (LocalPDENStokes<VECTOR, dealdim>::_viscosity
                    * scalar_product(zgrads, phi_i_grads_z)
                    + (phi_i_grads_z * v) * z + (vgrads * phi_i_z) * z
                    + dual_press * div_phi_z - incompressibility * phi_i_q)
                * state_fe_values.JxW(q_point);

            local_cell_vector(i) += scale * delta * (Su_z * Su_phi)
                * state_fe_values.JxW(q_point);

          }
        }
      }

      void
      CellMatrix(
          const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
          FullMatrix<double> &local_entry_matrix, double scale,
          double /*scale_ico*/)
      {
        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            cdc.GetFEValuesState();
        unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
        unsigned int n_q_points = cdc.GetNQPoints();
        const double h = cdc.GetCellDiameter();

        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(2);

        LocalPDENStokes<VECTOR, dealdim>::_uvalues.resize(n_q_points,
            Vector<double>(3));
        LocalPDENStokes<VECTOR, dealdim>::_ugrads.resize(n_q_points,
            vector<Tensor<1, 2> >(3));
        LocalPDENStokes<VECTOR, dealdim>::_lap_u.resize(n_q_points,
            Vector<double>(3));

        cdc.GetLaplaciansState("last_newton_solution",
            LocalPDENStokes<VECTOR, dealdim>::_lap_u);
        cdc.GetValuesState("last_newton_solution",
            LocalPDENStokes<VECTOR, dealdim>::_uvalues);
        cdc.GetGradsState("last_newton_solution",
            LocalPDENStokes<VECTOR, dealdim>::_ugrads);

        std::vector<Tensor<1, 2> > phi_v(n_dofs_per_cell);
        std::vector<Tensor<2, 2> > phi_grads_v(n_dofs_per_cell);
        std::vector<Tensor<1, 2> > phi_grads_p(n_dofs_per_cell);
        std::vector<double> phi_p(n_dofs_per_cell);
        std::vector<double> div_phi_v(n_dofs_per_cell);
        std::vector<Tensor<1, 2> > phi_lap_v(n_dofs_per_cell);

        const double max_v = GetMaxU(
            LocalPDENStokes<VECTOR, dealdim>::_uvalues);

        const double delta = _alpha
            / (LocalPDENStokes<VECTOR, dealdim>::_viscosity * pow(h, -2.)
                + _beta * max_v * 1. / h);
//        const double delta = std::min(
//            pow(h, 2.) / LocalPDENStokes<VECTOR, dealdim>::_viscosity,
//            h / max_v);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          for (unsigned int k = 0; k < n_dofs_per_cell; k++)
          {
            phi_v[k] = state_fe_values[velocities].value(k, q_point);
            phi_grads_v[k] = state_fe_values[velocities].gradient(k, q_point);
            phi_p[k] = state_fe_values[pressure].value(k, q_point);
            phi_grads_p[k] = state_fe_values[pressure].gradient(k, q_point);
            div_phi_v[k] = state_fe_values[velocities].divergence(k, q_point);
            phi_lap_v[k][0] =
                state_fe_values[velocities].hessian(k, q_point)[0][0][0]
                    + state_fe_values[velocities].hessian(k, q_point)[0][1][1];
            phi_lap_v[k][1] =
                state_fe_values[velocities].hessian(k, q_point)[1][0][0]
                    + state_fe_values[velocities].hessian(k, q_point)[1][1][1];
          }

          Tensor<2, 2> vgrads;
          vgrads.clear();
          vgrads[0][0] =
              LocalPDENStokes<VECTOR, dealdim>::_ugrads[q_point][0][0];
          vgrads[0][1] =
              LocalPDENStokes<VECTOR, dealdim>::_ugrads[q_point][0][1];
          vgrads[1][0] =
              LocalPDENStokes<VECTOR, dealdim>::_ugrads[q_point][1][0];
          vgrads[1][1] =
              LocalPDENStokes<VECTOR, dealdim>::_ugrads[q_point][1][1];

          Tensor<1, 2> pgrad;
          pgrad.clear();
          pgrad[0] = LocalPDENStokes<VECTOR, dealdim>::_ugrads[q_point][2][0];
          pgrad[1] = LocalPDENStokes<VECTOR, dealdim>::_ugrads[q_point][2][1];

          Tensor<1, 2> v;
          v.clear();
          v[0] = LocalPDENStokes<VECTOR, dealdim>::_uvalues[q_point](0);
          v[1] = LocalPDENStokes<VECTOR, dealdim>::_uvalues[q_point](1);

          Tensor<1, 2> res;
          res.clear();
          res[0] = -LocalPDENStokes<VECTOR, dealdim>::_viscosity
              * LocalPDENStokes<VECTOR, dealdim>::_lap_u[q_point][0]
              + v * vgrads[0] + pgrad[0];
          res[1] = -LocalPDENStokes<VECTOR, dealdim>::_viscosity
              * LocalPDENStokes<VECTOR, dealdim>::_lap_u[q_point][1]
              + v * vgrads[1] + pgrad[1];

          for (unsigned int i = 0; i < n_dofs_per_cell; i++)
          {
            for (unsigned int j = 0; j < n_dofs_per_cell; j++)
            {
              Tensor<1, 2> convection_fluid_LinV = phi_grads_v[j] * v
                  + vgrads * phi_v[j];

              //stokes part
              local_entry_matrix(i, j) += scale
                  * (LocalPDENStokes<VECTOR, dealdim>::_viscosity
                      * scalar_product(phi_grads_v[j], phi_grads_v[i])
                      + convection_fluid_LinV * phi_v[i]
                      - phi_p[j] * div_phi_v[i] + div_phi_v[j] * phi_p[i])
                  * state_fe_values.JxW(q_point);

              //A'*S
              local_entry_matrix(i, j) += scale * delta
                  * ((-LocalPDENStokes<VECTOR, dealdim>::_viscosity
                      * phi_lap_v[j] + phi_grads_v[j] * v + phi_v[j] * vgrads
                      + phi_grads_p[j]) * (phi_grads_v[i] * v + phi_grads_p[i])
                      + div_phi_v[j] * div_phi_v[i])
                  * state_fe_values.JxW(q_point);
              //A*S'
              local_entry_matrix(i, j) += scale * delta
                  * (res * (phi_grads_v[i] * phi_v[j]))
                  * state_fe_values.JxW(q_point);

            }
          }
        }

      }

      void
      CellMatrix_T(
          const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
          FullMatrix<double> &local_entry_matrix, double scale, double)
      {

        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            cdc.GetFEValuesState();
        unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
        unsigned int n_q_points = cdc.GetNQPoints();
        const double h = cdc.GetCellDiameter();

        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(2);

        LocalPDENStokes<VECTOR, dealdim>::_uvalues.resize(n_q_points,
            Vector<double>(3));
        LocalPDENStokes<VECTOR, dealdim>::_ugrads.resize(n_q_points,
            vector<Tensor<1, 2> >(3));

        cdc.GetValuesState("state", LocalPDENStokes<VECTOR, dealdim>::_uvalues);
        cdc.GetGradsState("state", LocalPDENStokes<VECTOR, dealdim>::_ugrads);

        std::vector<Tensor<1, 2> > phi_z(n_dofs_per_cell);
        std::vector<Tensor<2, 2> > phi_grads_z(n_dofs_per_cell);
        std::vector<Tensor<1, 2> > phi_grads_q(n_dofs_per_cell);
        std::vector<double> phi_q(n_dofs_per_cell);
        std::vector<double> div_phi_z(n_dofs_per_cell);

        const double max_v = GetMaxU(
            LocalPDENStokes<VECTOR, dealdim>::_uvalues);

        const double delta = _alpha
            / (LocalPDENStokes<VECTOR, dealdim>::_viscosity * pow(h, -2.)
                + _beta * max_v * 1. / h);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {

          for (unsigned int k = 0; k < n_dofs_per_cell; k++)
          {
            phi_z[k] = state_fe_values[velocities].value(k, q_point);
            phi_grads_z[k] = state_fe_values[velocities].gradient(k, q_point);
            phi_grads_q[k] = state_fe_values[pressure].gradient(k, q_point);
            phi_q[k] = state_fe_values[pressure].value(k, q_point);
            div_phi_z[k] = state_fe_values[velocities].divergence(k, q_point);
          }

          Tensor<2, 2> vgrads;
          vgrads.clear();
          vgrads[0][0] =
              LocalPDENStokes<VECTOR, dealdim>::_ugrads[q_point][0][0];
          vgrads[0][1] =
              LocalPDENStokes<VECTOR, dealdim>::_ugrads[q_point][0][1];
          vgrads[1][0] =
              LocalPDENStokes<VECTOR, dealdim>::_ugrads[q_point][1][0];
          vgrads[1][1] =
              LocalPDENStokes<VECTOR, dealdim>::_ugrads[q_point][1][1];

          Tensor<1, 2> v;
          v[0] = LocalPDENStokes<VECTOR, dealdim>::_uvalues[q_point](0);
          v[1] = LocalPDENStokes<VECTOR, dealdim>::_uvalues[q_point](1);

          for (unsigned int i = 0; i < n_dofs_per_cell; i++)
          {
            for (unsigned int j = 0; j < n_dofs_per_cell; j++)
            {
              local_entry_matrix(i, j) += scale
                  * (LocalPDENStokes<VECTOR, dealdim>::_viscosity
                      * scalar_product(phi_grads_z[j], phi_grads_z[i])
                      + (phi_grads_z[i] * v) * phi_z[j]
                      + (vgrads * phi_z[i]) * phi_z[j] + phi_q[j] * div_phi_z[i]
                      - div_phi_z[j] * phi_q[i]) * state_fe_values.JxW(q_point);

              local_entry_matrix(i, j) += scale * delta
                  * ((phi_grads_z[i] * v + phi_grads_q[i])
                      * (phi_grads_z[j] * v + phi_grads_q[j])
                      + div_phi_z[i] * div_phi_z[j])
                  * state_fe_values.JxW(q_point);

            }
          }
        }
      }

      void
      StrongBoundaryResidual_U(
          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc_w,
          double& sum, double scale)
      {
        const unsigned int n_q_points = fdc.GetNQPoints();
        const unsigned int color = fdc.GetBoundaryIndicator();

        LocalPDENStokes<VECTOR, dealdim>::_zgrads.resize(n_q_points,
            vector<Tensor<1, dealdim> >(3));
        LocalPDENStokes<VECTOR, dealdim>::_zvalues.resize(n_q_points,
            Vector<double>(3));

        LocalPDENStokes<VECTOR, dealdim>::_PI_h_u_grads.resize(n_q_points,
            vector<Tensor<1, dealdim> >(3));

        fdc.GetFaceGradsState("adjoint_for_ee",
            LocalPDENStokes<VECTOR, dealdim>::_zgrads);
        fdc.GetFaceValuesState("adjoint_for_ee",
            LocalPDENStokes<VECTOR, dealdim>::_zvalues);

        fdc_w.GetFaceGradsState("weight_for_dual_residual",
            LocalPDENStokes<VECTOR, dealdim>::_PI_h_u_grads);

        if (color == 1)
        {
          LocalPDENStokes<VECTOR, dealdim>::_uvalues.resize(n_q_points,
              Vector<double>(3));
          fdc.GetFaceValuesState("state",
              LocalPDENStokes<VECTOR, dealdim>::_uvalues);

          for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {

            //negative Wert des Randintegrals wird addiert.
            sum -=
                scale
                    * (LocalPDENStokes<VECTOR, dealdim>::_viscosity
                        * (LocalPDENStokes<VECTOR, dealdim>::_zgrads[q_point][0][0]
                            * LocalPDENStokes<VECTOR, dealdim>::_PI_h_u[q_point][0]
                            + LocalPDENStokes<VECTOR, dealdim>::_zgrads[q_point][1][0]
                                * LocalPDENStokes<VECTOR, dealdim>::_PI_h_u[q_point][1])
                        + LocalPDENStokes<VECTOR, dealdim>::_uvalues[q_point][0]/* =v*n */
                            * (LocalPDENStokes<VECTOR, dealdim>::_zvalues[q_point][0]
                                * LocalPDENStokes<VECTOR, dealdim>::_PI_h_u[q_point][0]
                                + LocalPDENStokes<VECTOR, dealdim>::_zvalues[q_point][1]
                                    * LocalPDENStokes<VECTOR, dealdim>::_PI_h_u[q_point][1])
                        - LocalPDENStokes<VECTOR, dealdim>::_zvalues[q_point][2]
                            * LocalPDENStokes<VECTOR, dealdim>::_PI_h_u[q_point][0])
                    * fdc.GetFEFaceValuesState().JxW(q_point);
          }
        }
//            else if (color == 80)
//              for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
//              {
//                Tensor<2, 2> fluid_pressure;
//                fluid_pressure.clear();
//                fluid_pressure[0][0] = -_PI_h_u[q_point](2);
//                fluid_pressure[1][1] = -_PI_h_u[q_point](2);
//
//                Tensor<2, 2> vgrads;
//                vgrads.clear();
//                vgrads[0][0] = _PI_h_u_grads[q_point][0][0];
//                vgrads[0][1] = _PI_h_u_grads[q_point][0][1];
//                vgrads[1][0] = _PI_h_u_grads[q_point][1][0];
//                vgrads[1][1] = _PI_h_u_grads[q_point][1][1];
//
//                sum -= _drag_lift_constant
//                    * ((fluid_pressure + _density_fluid * _viscosity * (vgrads))
//                        * fdc.GetFEFaceValuesState().normal_vector(q_point)
//                        * fdc.GetFEFaceValuesState().JxW(q_point))[0];
//              }

      }

      void
      StrongCellResidual_U(
          const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
          const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc_w,
          double& sum, double scale)
      {

        unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
        unsigned int n_q_points = cdc.GetNQPoints();
        const DOpEWrapper::FEValues<dealdim> &state_fe_values =
            cdc.GetFEValuesState();
        const DOpEWrapper::FEValues<dealdim> &state_fe_values_weight =
            cdc_w.GetFEValuesState();

        LocalPDENStokes<VECTOR, dealdim>::_fvalues.resize(n_q_points,
            Vector<double>(3));

        LocalPDENStokes<VECTOR, dealdim>::_PI_h_u.resize(n_q_points,
            Vector<double>(3));
        LocalPDENStokes<VECTOR, dealdim>::_lap_z.resize(n_q_points,
            Vector<double>(3));
        LocalPDENStokes<VECTOR, dealdim>::_uvalues.resize(n_q_points,
            Vector<double>(3));
        LocalPDENStokes<VECTOR, dealdim>::_zvalues.resize(n_q_points,
            Vector<double>(3));
        LocalPDENStokes<VECTOR, dealdim>::_ugrads.resize(n_q_points,
            vector<Tensor<1, dealdim> >(3));
        LocalPDENStokes<VECTOR, dealdim>::_zgrads.resize(n_q_points,
            vector<Tensor<1, dealdim> >(3));

        cdc.GetLaplaciansState("adjoint_for_ee",
            LocalPDENStokes<VECTOR, dealdim>::_lap_z);
        cdc.GetGradsState("adjoint_for_ee",
            LocalPDENStokes<VECTOR, dealdim>::_zgrads);
        cdc.GetValuesState("adjoint_for_ee",
            LocalPDENStokes<VECTOR, dealdim>::_zvalues);
        cdc.GetValuesState("state", LocalPDENStokes<VECTOR, dealdim>::_uvalues);
        cdc.GetGradsState("state", LocalPDENStokes<VECTOR, dealdim>::_ugrads);

        cdc_w.GetValuesState("weight_for_dual_residual",
            LocalPDENStokes<VECTOR, dealdim>::_PI_h_u);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          Tensor<1, dealdim> u;
          u[0] = LocalPDENStokes<VECTOR, dealdim>::_uvalues[q_point][0];
          u[1] = LocalPDENStokes<VECTOR, dealdim>::_uvalues[q_point][1];

          Tensor<1, dealdim> z;
          z[0] = LocalPDENStokes<VECTOR, dealdim>::_zvalues[q_point][0];
          z[1] = LocalPDENStokes<VECTOR, dealdim>::_zvalues[q_point][1];

          Tensor<2, dealdim> gradu_t;
          gradu_t[0][0] =
              LocalPDENStokes<VECTOR, dealdim>::_ugrads[q_point][0][0];
          gradu_t[0][1] =
              LocalPDENStokes<VECTOR, dealdim>::_ugrads[q_point][1][0];
          gradu_t[1][0] =
              LocalPDENStokes<VECTOR, dealdim>::_ugrads[q_point][0][1];
          gradu_t[1][1] =
              LocalPDENStokes<VECTOR, dealdim>::_ugrads[q_point][1][1];

          Tensor<2, dealdim> gradz;
          gradz[0][0] =
              LocalPDENStokes<VECTOR, dealdim>::_zgrads[q_point][0][0];
          gradz[0][1] =
              LocalPDENStokes<VECTOR, dealdim>::_zgrads[q_point][0][1];
          gradz[1][0] =
              LocalPDENStokes<VECTOR, dealdim>::_zgrads[q_point][1][0];
          gradz[1][1] =
              LocalPDENStokes<VECTOR, dealdim>::_zgrads[q_point][1][1];

          const double divz =
              LocalPDENStokes<VECTOR, dealdim>::_zgrads[q_point][0][0]
                  + LocalPDENStokes<VECTOR, dealdim>::_zgrads[q_point][1][1];
          const double divu =
              LocalPDENStokes<VECTOR, dealdim>::_ugrads[q_point][0][0]
                  + LocalPDENStokes<VECTOR, dealdim>::_ugrads[q_point][1][1];

          double res_0 = LocalPDENStokes<VECTOR, dealdim>::_viscosity
              * LocalPDENStokes<VECTOR, dealdim>::_lap_z[q_point][0]
              + divu * LocalPDENStokes<VECTOR, dealdim>::_zvalues[q_point][0]
              + u * gradz[0] - gradu_t[0] * z
              + LocalPDENStokes<VECTOR, dealdim>::_zgrads[q_point][2][0];
          double res_1 = LocalPDENStokes<VECTOR, dealdim>::_viscosity
              * LocalPDENStokes<VECTOR, dealdim>::_lap_z[q_point][1]
              + divu * LocalPDENStokes<VECTOR, dealdim>::_zvalues[q_point][1]
              + u * gradz[1] - gradu_t[1] * z
              + LocalPDENStokes<VECTOR, dealdim>::_zgrads[q_point][2][1];

          sum +=
              scale
                  * (res_0
                      * LocalPDENStokes<VECTOR, dealdim>::_PI_h_u[q_point][0]
                      + res_1
                          * LocalPDENStokes<VECTOR, dealdim>::_PI_h_u[q_point][1]
                      - divz
                          * LocalPDENStokes<VECTOR, dealdim>::_PI_h_u[q_point][2])
                  * state_fe_values.JxW(q_point);
        }
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
  };

/***************************************************************************************/

template<typename VECTOR, int dealdim>
  class LocalPDEStokes : public PDEInterface<CellDataContainer,
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

      LocalPDEStokes(ParameterReader &param_reader)
          : _state_block_components(3, 0)
      {
        _state_block_components[2] = 1;

        param_reader.SetSubsection("Local PDE parameters");
        _density_fluid = param_reader.get_double("density_fluid");
        _viscosity = param_reader.get_double("viscosity");
        _drag_lift_constant = param_reader.get_double("drag_lift_constant");
      }

      // Domain values for cells
      void
      CellEquation(
          const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
          dealii::Vector<double> &local_cell_vector, double scale,
          double /*scale_ico*/)
      {
        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            cdc.GetFEValuesState();
        unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
        unsigned int n_q_points = cdc.GetNQPoints();
        //        assert(this->_problem_type == "state");

        _uvalues.resize(n_q_points, Vector<double>(3));
        _ugrads.resize(n_q_points, vector<Tensor<1, 2> >(3));

        cdc.GetValuesState("last_newton_solution", _uvalues);
        cdc.GetGradsState("last_newton_solution", _ugrads);

        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(2);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          Tensor<2, 2> vgrads;
          vgrads[0][0] = _ugrads[q_point][0][0];
          vgrads[0][1] = _ugrads[q_point][0][1];
          vgrads[1][0] = _ugrads[q_point][1][0];
          vgrads[1][1] = _ugrads[q_point][1][1];

          Tensor<1, 2> v;
          v.clear();
          v[0] = _uvalues[q_point](0);
          v[1] = _uvalues[q_point](1);

          Tensor<1, 2> convection_fluid = vgrads * v;
          double press = _uvalues[q_point](2);
          double incompressibility = vgrads[0][0] + vgrads[1][1];

          for (unsigned int i = 0; i < n_dofs_per_cell; i++)
          {
            const Tensor<2, 2> phi_i_grads_v =
                state_fe_values[velocities].gradient(i, q_point);
            const double phi_i_p = state_fe_values[pressure].value(i, q_point);
            const Tensor<1, 2> phi_i_v = state_fe_values[velocities].value(i,
                q_point);
            const double div_phi_v = state_fe_values[velocities].divergence(i,
                q_point);

            local_cell_vector(i) += scale
                * (_viscosity * scalar_product(vgrads, phi_i_grads_v)
                /*+ convection_fluid * phi_i_v*/- press * div_phi_v
                    + incompressibility * phi_i_p)
                * state_fe_values.JxW(q_point);

          }
        }

      }
      void
      CellEquation_U(
          const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
          dealii::Vector<double> &local_cell_vector, double scale, double)
      {
        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            cdc.GetFEValuesState();
        unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
        unsigned int n_q_points = cdc.GetNQPoints();
        //        assert(this->_problem_type == "state");

        _uvalues.resize(n_q_points, Vector<double>(3));
        _ugrads.resize(n_q_points, vector<Tensor<1, 2> >(3));

        _zvalues.resize(n_q_points, Vector<double>(3));
        _zgrads.resize(n_q_points, vector<Tensor<1, 2> >(3));

        cdc.GetValuesState("state", _uvalues);
        cdc.GetGradsState("state", _ugrads);

        cdc.GetValuesState("last_newton_solution", _zvalues);
        cdc.GetGradsState("last_newton_solution", _zgrads);

        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(2);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          Tensor<2, 2> vgrads;
          vgrads[0][0] = _ugrads[q_point][0][0];
          vgrads[0][1] = _ugrads[q_point][0][1];
          vgrads[1][0] = _ugrads[q_point][1][0];
          vgrads[1][1] = _ugrads[q_point][1][1];

          Tensor<2, 2> zgrads;
          zgrads[0][0] = _zgrads[q_point][0][0];
          zgrads[0][1] = _zgrads[q_point][0][1];
          zgrads[1][0] = _zgrads[q_point][1][0];
          zgrads[1][1] = _zgrads[q_point][1][1];

          Tensor<1, 2> v;
          v.clear();
          v[0] = _uvalues[q_point](0);
          v[1] = _uvalues[q_point](1);

          Tensor<1, 2> z;
          z.clear();
          z[0] = _zvalues[q_point](0);
          z[1] = _zvalues[q_point](1);

          double dual_press = _zvalues[q_point](2);
          double incompressibility = zgrads[0][0] + zgrads[1][1];

          for (unsigned int i = 0; i < n_dofs_per_cell; i++)
          {
            const Tensor<2, 2> phi_i_grads_z =
                state_fe_values[velocities].gradient(i, q_point);
            const double phi_i_q = state_fe_values[pressure].value(i, q_point);
            const Tensor<1, 2> phi_i_z = state_fe_values[velocities].value(i,
                q_point);
            const double div_phi_z = state_fe_values[velocities].divergence(i,
                q_point);

            local_cell_vector(i) += scale
                * (_viscosity * scalar_product(zgrads, phi_i_grads_z)
                /*+ (vgrads * phi_i_z) * z + (phi_i_grads_z * v) * z*/
                + dual_press * div_phi_z - incompressibility * phi_i_q)
                * state_fe_values.JxW(q_point);

          }
        }
      }

      void
      CellMatrix(
          const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
          FullMatrix<double> &local_entry_matrix, double scale,
          double scale_ico)
      {
        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            cdc.GetFEValuesState();
        unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
        unsigned int n_q_points = cdc.GetNQPoints();
        //unsigned int material_id = cdc.GetMaterialId();

        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(2);

        _uvalues.resize(n_q_points, Vector<double>(3));
        _ugrads.resize(n_q_points, vector<Tensor<1, 2> >(3));

        cdc.GetValuesState("last_newton_solution", _uvalues);
        cdc.GetGradsState("last_newton_solution", _ugrads);

        std::vector<Tensor<1, 2> > phi_v(n_dofs_per_cell);
        std::vector<Tensor<2, 2> > phi_grads_v(n_dofs_per_cell);
        std::vector<double> phi_p(n_dofs_per_cell);
        std::vector<double> div_phi_v(n_dofs_per_cell);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          for (unsigned int k = 0; k < n_dofs_per_cell; k++)
          {
            phi_v[k] = state_fe_values[velocities].value(k, q_point);
            phi_grads_v[k] = state_fe_values[velocities].gradient(k, q_point);
            phi_p[k] = state_fe_values[pressure].value(k, q_point);
            div_phi_v[k] = state_fe_values[velocities].divergence(k, q_point);
          }

          Tensor<2, 2> vgrads;
          vgrads.clear();
          vgrads[0][0] = _ugrads[q_point][0][0];
          vgrads[0][1] = _ugrads[q_point][0][1];
          vgrads[1][0] = _ugrads[q_point][1][0];
          vgrads[1][1] = _ugrads[q_point][1][1];

          Tensor<1, 2> v;
          v[0] = _uvalues[q_point](0);
          v[1] = _uvalues[q_point](1);

          for (unsigned int i = 0; i < n_dofs_per_cell; i++)
          {
            Tensor<2, dealdim> fluid_pressure_LinP;
            fluid_pressure_LinP.clear();
            fluid_pressure_LinP[0][0] = -phi_p[i];
            fluid_pressure_LinP[1][1] = -phi_p[i];

            Tensor<1, 2> convection_fluid_LinV = phi_grads_v[i] * v
                + vgrads * phi_v[i];

            for (unsigned int j = 0; j < n_dofs_per_cell; j++)
            {
              local_entry_matrix(j, i) += scale * (/*convection_fluid_LinV * phi_v[j]
               +*/_viscosity * scalar_product(phi_grads_v[i], phi_grads_v[j]))
                  * state_fe_values.JxW(q_point);

              local_entry_matrix(j, i) +=
                  scale
                      * (scalar_product(fluid_pressure_LinP, phi_grads_v[j])
                          + (phi_grads_v[i][0][0] + phi_grads_v[i][1][1])
                              * phi_p[j]) * state_fe_values.JxW(q_point);
            }
          }
        }

      }

      void
      CellMatrix_T(
          const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
          FullMatrix<double> &local_entry_matrix, double scale, double)
      {

        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            cdc.GetFEValuesState();
        unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
        unsigned int n_q_points = cdc.GetNQPoints();

        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(2);

        _uvalues.resize(n_q_points, Vector<double>(3));
        _ugrads.resize(n_q_points, vector<Tensor<1, 2> >(3));

        cdc.GetValuesState("state", _uvalues);
        cdc.GetGradsState("state", _ugrads);

        std::vector<Tensor<1, 2> > phi_z(n_dofs_per_cell);
        std::vector<Tensor<2, 2> > phi_grads_z(n_dofs_per_cell);
        std::vector<double> phi_q(n_dofs_per_cell);
        std::vector<double> div_phi_z(n_dofs_per_cell);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          for (unsigned int k = 0; k < n_dofs_per_cell; k++)
          {
            phi_z[k] = state_fe_values[velocities].value(k, q_point);
            phi_grads_z[k] = state_fe_values[velocities].gradient(k, q_point);
            phi_q[k] = state_fe_values[pressure].value(k, q_point);
            div_phi_z[k] = state_fe_values[velocities].divergence(k, q_point);
          }

          Tensor<2, 2> vgrads;
          vgrads.clear();
          vgrads[0][0] = _ugrads[q_point][0][0];
          vgrads[0][1] = _ugrads[q_point][0][1];
          vgrads[1][0] = _ugrads[q_point][1][0];
          vgrads[1][1] = _ugrads[q_point][1][1];

          Tensor<1, 2> v;
          v[0] = _uvalues[q_point](0);
          v[1] = _uvalues[q_point](1);

          for (unsigned int i = 0; i < n_dofs_per_cell; i++)
          {
            Tensor<1, 2> convection_fluid_LinV = phi_grads_z[i] * v
                + vgrads * phi_z[i];
            for (unsigned int j = 0; j < n_dofs_per_cell; j++)
            {
              local_entry_matrix(i, j) += scale
                  * (_viscosity * scalar_product(phi_grads_z[j], phi_grads_z[i])
                  /*+ convection_fluid_LinV * phi_z[j]*/
                  + phi_q[j] * div_phi_z[i] - div_phi_z[j] * phi_q[i])
                  * state_fe_values.JxW(q_point);
            }
          }
        }
      }

      void
      CellRightHandSide(
          const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
          dealii::Vector<double> &local_cell_vector, double scale)
      {
//        unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
//        unsigned int n_q_points = cdc.GetNQPoints();
//        const DOpEWrapper::FEValues<dealdim> &state_fe_values =
//            cdc.GetFEValuesState();
//        _fvalues.resize(n_q_points);
//        std::vector<Tensor<1, 2> > phi_v(n_dofs_per_cell);
//        const FEValuesExtractors::Vector velocities(0);
//
//        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
//        {
//          //          const double x = state_fe_values.quadrature_point(q_point)(0);
//          //          const double y = state_fe_values.quadrature_point(q_point)(1);
//          //          const double pi = numbers::PI;
//          _fvalues[q_point][0] = 0;
//          _fvalues[q_point][1] = pi * pi * pi * cos(pi * x) * (y * y - y)
//              - 4 * pi * cos(pi * x);
////             _fvalues[q_point][0] = 1;
////             _fvalues[q_point][1] = 0;
//
//          for (unsigned int k = 0; k < n_dofs_per_cell; k++)
//          {
//            phi_v[k] = state_fe_values[velocities].value(k, q_point);
//          }
//          for (unsigned int i = 0; i < n_dofs_per_cell; i++)
//          {
//            local_cell_vector(i) += scale * _coeff_rhs
//                * contract(_fvalues[q_point], phi_v[i])
//                * state_fe_values.JxW(q_point);
//          }
//        }
      }

      void
      FaceEquation(
          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>&,
          dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/,
          double)
      {
      }

      void
      FaceEquation_U(
          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>&,
          dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/,
          double)
      {
      }

      void
      FaceMatrix_T(
          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>&,
          dealii::FullMatrix<double> &/*local_entry_matrix*/, double, double)
      {
      }

      void
      FaceMatrix(
          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>&,
          dealii::FullMatrix<double> &/*local_entry_matrix*/, double, double)
      {
      }

      // Values for boundary integrals
      void
      BoundaryEquation(
          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
          dealii::Vector<double> &local_cell_vector, double scale,
          double /*scale_ico*/)
      {

      }

      // Values for boundary integrals
      void
      BoundaryEquation_U(
          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
          dealii::Vector<double> &local_cell_vector, double scale,
          double /*scale_ico*/)
      {

      }

      void
      BoundaryRightHandSide(
          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>&,
          dealii::Vector<double> &local_cell_vector __attribute__((unused)),
          double scale __attribute__((unused)))
      {

      }
      void
      BoundaryMatrix(
          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
          dealii::FullMatrix<double> &local_entry_matrix, double /*scale_ico*/,
          double /*scale_ico*/)
      {

      }

      void
      FaceRightHandSide(
          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
          dealii::Vector<double> &local_rhs, double scale)
      {

      }

      void
      StrongBoundaryResidual(
          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc_w,
          double& sum, double scale)
      {
        const unsigned int n_q_points = fdc.GetNQPoints();
        const unsigned int color = fdc.GetBoundaryIndicator();

        _uvalues.resize(n_q_points, Vector<double>(3));
        _ugrads.resize(n_q_points, vector<Tensor<1, dealdim> >(3));
        _PI_h_z.resize(n_q_points);

        fdc.GetFaceGradsState("state", _ugrads);
        fdc.GetFaceValuesState("state", _uvalues);

        fdc_w.GetFaceValuesState("weight_for_primal_residual", _PI_h_z);

        //normale ist hier (1,0)
        Vector<double> res(3);

        if (color == 1)
          for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {

            res[0] = -(_viscosity * _ugrads[q_point][0][0]
                - _uvalues[q_point][2]);
            res[1] = -(_viscosity * _ugrads[q_point][1][0]);
            res[2] = 0;
            sum += scale * (res * _PI_h_z[q_point])
                * fdc.GetFEFaceValuesState().JxW(q_point);
          }

      }

      void
      StrongCellResidual(
          const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
          const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc_w,
          double& sum, double scale, double)
      {

        unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
        unsigned int n_q_points = cdc.GetNQPoints();
        const DOpEWrapper::FEValues<dealdim> &state_fe_values =
            cdc.GetFEValuesState();
        const DOpEWrapper::FEValues<dealdim> &state_fe_values_weight =
            cdc_w.GetFEValuesState();

        _fvalues.resize(n_q_points, Vector<double>(3));

        _PI_h_z.resize(n_q_points, Vector<double>(3));
        _lap_u.resize(n_q_points, Vector<double>(3));
        _ugrads.resize(n_q_points, vector<Tensor<1, dealdim> >(3));
        _uvalues.resize(n_q_points, Vector<double>(3));

        cdc.GetLaplaciansState("state", _lap_u);
        cdc.GetGradsState("state", _ugrads);
        cdc.GetValuesState("state", _uvalues);

        cdc_w.GetValuesState("weight_for_primal_residual", _PI_h_z);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {

          Tensor<2, 2> vgrads;
          vgrads[0][0] = _ugrads[q_point][0][0];
          vgrads[0][1] = _ugrads[q_point][0][1];
          vgrads[1][0] = _ugrads[q_point][1][0];
          vgrads[1][1] = _ugrads[q_point][1][1];

          Tensor<1, 2> v;
          v.clear();
          v[0] = _uvalues[q_point](0);
          v[1] = _uvalues[q_point](1);

          Tensor<1, 2> convection_fluid = vgrads * v;

          const double divu = _ugrads[q_point][0][0] + _ugrads[q_point][1][1];
          //f - (Starke form der Gleichung!)
          double res_0 = _viscosity * _lap_u[q_point][0]/* - convection_fluid[0]*/
          - _ugrads[q_point][2][0];
          double res_1 = _viscosity * _lap_u[q_point][1] /*- convection_fluid[1]*/
          - _ugrads[q_point][2][1];

          sum += scale
              * (res_0 * _PI_h_z[q_point][0] + res_1 * _PI_h_z[q_point][1]
                  - divu * _PI_h_z[q_point][2]) * state_fe_values.JxW(q_point);
        }
      }

      void
      StrongFaceResidual(
          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc_w,
          double& sum, double scale)
      {
        const unsigned int n_q_points = fdc.GetNQPoints();
        const unsigned int material_id = fdc.GetMaterialId();
        const unsigned int material_id_nbr = fdc.GetNbrMaterialId();

        _ugrads.resize(n_q_points, vector<Tensor<1, dealdim> >(3));
        _ugrads_nbr.resize(n_q_points, vector<Tensor<1, dealdim> >(3));

        _PI_h_z.resize(n_q_points);

        fdc.GetFaceGradsState("state", _ugrads);
        fdc.GetNbrFaceGradsState("state", _ugrads_nbr);

        fdc_w.GetFaceValuesState("weight_for_primal_residual", _PI_h_z);
        vector<Vector<double> > jump(n_q_points, Vector<double>(2));

        for (unsigned int q = 0; q < n_q_points; q++)
        {             //Druck wird vernachlaessigt da Druckapproximation stetig
          jump[q][0] = _viscosity
              * ((_ugrads_nbr[q][0][0] - _ugrads[q][0][0])
                  * fdc.GetFEFaceValuesState().normal_vector(q)[0]
                  + (_ugrads_nbr[q][0][1] - _ugrads[q][0][1])
                      * fdc.GetFEFaceValuesState().normal_vector(q)[1]);

          jump[q][1] = _viscosity
              * ((_ugrads_nbr[q][1][0] - _ugrads[q][1][0])
                  * fdc.GetFEFaceValuesState().normal_vector(q)[0]
                  + (_ugrads_nbr[q][1][1] - _ugrads[q][1][1])
                      * fdc.GetFEFaceValuesState().normal_vector(q)[1]);
        }

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          sum += scale
              * (jump[q_point][0] * _PI_h_z[q_point][0]
                  + jump[q_point][1] * _PI_h_z[q_point][1])
              * fdc.GetFEFaceValuesState().JxW(q_point);
        }

      }

      void
      StrongBoundaryResidual_U(
          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc_w,
          double& sum, double scale)
      {
        const unsigned int n_q_points = fdc.GetNQPoints();
        const unsigned int color = fdc.GetBoundaryIndicator();

        _zgrads.resize(n_q_points, vector<Tensor<1, dealdim> >(3));
        _zvalues.resize(n_q_points, Vector<double>(3));

        _PI_h_u_grads.resize(n_q_points, vector<Tensor<1, dealdim> >(3));

        fdc.GetFaceGradsState("adjoint_for_ee", _zgrads);
        fdc.GetFaceValuesState("adjoint_for_ee", _zvalues);

        fdc_w.GetFaceGradsState("weight_for_dual_residual", _PI_h_u_grads);

        if (color == 1)
        {
          _uvalues.resize(n_q_points, Vector<double>(3));
          fdc.GetFaceValuesState("state", _uvalues);

          for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            //negative Wert des Randintegrals wird addiert.
            sum -= scale
                * (_viscosity             //erinnerung: n = (1,0)
                    * (_zgrads[q_point][0][0] * _PI_h_u[q_point][0]
                        + _zgrads[q_point][1][0] * _PI_h_u[q_point][1])
                /* =v*n *//*+ _uvalues[q_point][0]
                 * (_zvalues[q_point][0] * _PI_h_u[q_point][0]
                 + _zvalues[q_point][1] * _PI_h_u[q_point][1])*/
                + _zvalues[q_point][2] * _PI_h_u[q_point][0])
                * fdc.GetFEFaceValuesState().JxW(q_point);
          }
        }
        else if (color == 80)
          for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            Tensor<2, 2> fluid_pressure;
            fluid_pressure.clear();
            fluid_pressure[0][0] = -_PI_h_u[q_point](2);
            fluid_pressure[1][1] = -_PI_h_u[q_point](2);

            Tensor<2, 2> vgrads;
            vgrads.clear();
            vgrads[0][0] = _PI_h_u_grads[q_point][0][0];
            vgrads[0][1] = _PI_h_u_grads[q_point][0][1];
            vgrads[1][0] = _PI_h_u_grads[q_point][1][0];
            vgrads[1][1] = _PI_h_u_grads[q_point][1][1];

            sum -= _drag_lift_constant
                * ((fluid_pressure + _density_fluid * _viscosity * (vgrads))
                    * fdc.GetFEFaceValuesState().normal_vector(q_point)
                    * fdc.GetFEFaceValuesState().JxW(q_point))[0];
          }

      }

      void
      StrongCellResidual_U(
          const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
          const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc_w,
          double& sum, double scale)
      {

        unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
        unsigned int n_q_points = cdc.GetNQPoints();
        const DOpEWrapper::FEValues<dealdim> &state_fe_values =
            cdc.GetFEValuesState();
        const DOpEWrapper::FEValues<dealdim> &state_fe_values_weight =
            cdc_w.GetFEValuesState();

        _fvalues.resize(n_q_points, Vector<double>(3));

        _PI_h_u.resize(n_q_points, Vector<double>(3));
        _lap_z.resize(n_q_points, Vector<double>(3));
        _uvalues.resize(n_q_points, Vector<double>(3));
        _zvalues.resize(n_q_points, Vector<double>(3));
        _ugrads.resize(n_q_points, vector<Tensor<1, dealdim> >(3));
        _zgrads.resize(n_q_points, vector<Tensor<1, dealdim> >(3));

        cdc.GetLaplaciansState("adjoint_for_ee", _lap_z);
        cdc.GetGradsState("adjoint_for_ee", _zgrads);
        cdc.GetValuesState("adjoint_for_ee", _zvalues);
        cdc.GetValuesState("state", _uvalues);
        cdc.GetGradsState("state", _ugrads);

        cdc_w.GetValuesState("weight_for_dual_residual", _PI_h_u);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          Tensor<1, dealdim> u;
          u[0] = _uvalues[q_point][0];
          u[1] = _uvalues[q_point][1];

          Tensor<1, dealdim> z;
          z[0] = _zvalues[q_point][0];
          z[1] = _zvalues[q_point][1];

          Tensor<2, dealdim> gradu_t;
          gradu_t[0][0] = _ugrads[q_point][0][0];
          gradu_t[0][1] = _ugrads[q_point][1][0];
          gradu_t[1][0] = _ugrads[q_point][0][1];
          gradu_t[1][1] = _ugrads[q_point][1][1];

          Tensor<2, dealdim> gradz;
          gradz[0][0] = _zgrads[q_point][0][0];
          gradz[0][1] = _zgrads[q_point][0][1];
          gradz[1][0] = _zgrads[q_point][1][0];
          gradz[1][1] = _zgrads[q_point][1][1];

          const double divz = _zgrads[q_point][0][0] + _zgrads[q_point][1][1];
          const double divu = _ugrads[q_point][0][0] + _ugrads[q_point][1][1];

          double res_0 = _viscosity * _lap_z[q_point][0]
          /*+ divu * _zvalues[q_point][0] + u * gradz[0] - gradu_t[0] * z*/
          + _zgrads[q_point][2][0];
          double res_1 = _viscosity * _lap_z[q_point][1]
          /*+ divu * _zvalues[q_point][1] + u * gradz[1] - gradu_t[1] * z*/
          + _zgrads[q_point][2][1];

          sum += scale
              * (res_0 * _PI_h_u[q_point][0] + res_1 * _PI_h_u[q_point][1]
                  + divz * _PI_h_u[q_point][2]) * state_fe_values.JxW(q_point);
        }
      }

      void
      StrongFaceResidual_U(
          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc_w,
          double& sum, double scale)
      {
        const unsigned int n_q_points = fdc.GetNQPoints();
        const unsigned int material_id = fdc.GetMaterialId();
        const unsigned int material_id_nbr = fdc.GetNbrMaterialId();

        _zgrads.resize(n_q_points, vector<Tensor<1, dealdim> >(3));
        _zgrads_nbr.resize(n_q_points, vector<Tensor<1, dealdim> >(3));
        _PI_h_u.resize(n_q_points);

        fdc.GetFaceGradsState("adjoint_for_ee", _zgrads);
        fdc.GetNbrFaceGradsState("adjoint_for_ee", _zgrads_nbr);

        fdc_w.GetFaceValuesState("weight_for_dual_residual", _PI_h_u);
        vector<Vector<double> > jump(n_q_points, Vector<double>(2));

        for (unsigned int q = 0; q < n_q_points; q++)
        {
          jump[q][0] = _viscosity * (_zgrads_nbr[q][0][0] - _zgrads[q][0][0])
              * fdc.GetFEFaceValuesState().normal_vector(q)[0]
              + _viscosity * (_zgrads_nbr[q][0][1] - _zgrads[q][0][1])
                  * fdc.GetFEFaceValuesState().normal_vector(q)[1];

          jump[q][1] = _viscosity * (_zgrads_nbr[q][1][0] - _zgrads[q][1][0])
              * fdc.GetFEFaceValuesState().normal_vector(q)[0]
              + _viscosity * (_zgrads_nbr[q][1][1] - _zgrads[q][1][1])
                  * fdc.GetFEFaceValuesState().normal_vector(q)[1];
        }

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          sum += scale
              * (jump[q_point][0] * _PI_h_u[q_point][0]
                  + jump[q_point][1] * _PI_h_u[q_point][1])
              * fdc.GetFEFaceValuesState().JxW(q_point);
        }

      }

      ///////////////////////////////////////////////////////////////////
//
//      // Domain values for cells
//      void
//      CellEquation(
//          const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
//          dealii::Vector<double> &local_cell_vector, double scale,
//          double /*scale_ico*/)
//      {
//        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
//            cdc.GetFEValuesState();
//        unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
//        unsigned int n_q_points = cdc.GetNQPoints();
//        //        assert(this->_problem_type == "state");
//
//        _uvalues.resize(n_q_points, Vector<double>(3));
//        _ugrads.resize(n_q_points, vector<Tensor<1, 2> >(3));
//
//        cdc.GetValuesState("last_newton_solution", _uvalues);
//        cdc.GetGradsState("last_newton_solution", _ugrads);
//
//        const FEValuesExtractors::Vector velocities(0);
//        const FEValuesExtractors::Scalar pressure(2);
//
//        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
//        {
//          Tensor<2, 2> vgrads;
//          vgrads.clear();
//          vgrads[0][0] = _ugrads[q_point][0][0];
//          vgrads[0][1] = _ugrads[q_point][0][1];
//          vgrads[1][0] = _ugrads[q_point][1][0];
//          vgrads[1][1] = _ugrads[q_point][1][1];
//
//          double press = _uvalues[q_point](2);
//          double incompressibility = vgrads[0][0] + vgrads[1][1];
//
//          for (unsigned int i = 0; i < n_dofs_per_cell; i++)
//          {
//            const Tensor<2, 2> phi_i_grads_v =
//                state_fe_values[velocities].gradient(i, q_point);
//            const double phi_i_p = state_fe_values[pressure].value(i, q_point);
//            const double div_phi_v = state_fe_values[velocities].divergence(i,
//                q_point);
//
//            local_cell_vector(i) += scale
//                * (_viscosity * scalar_product(vgrads, phi_i_grads_v)
//                    - press * div_phi_v - incompressibility * phi_i_p)
//                * state_fe_values.JxW(q_point);
//
//          }
//        }
//
//      }
//      void
//      CellEquation_U(
//          const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
//          dealii::Vector<double> &local_cell_vector, double scale, double)
//      {
//        CellEquation(cdc, local_cell_vector, scale, 1.);
//      }
//
//      void
//      CellMatrix(
//          const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
//          FullMatrix<double> &local_entry_matrix, double scale,
//          double /*scale_ico*/)
//      {
//
//        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
//            cdc.GetFEValuesState();
//        unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
//        unsigned int n_q_points = cdc.GetNQPoints();
//
//        const FEValuesExtractors::Vector velocities(0);
//        const FEValuesExtractors::Scalar pressure(2);
//
//        std::vector<Tensor<1, 2> > phi_v(n_dofs_per_cell);
//        std::vector<Tensor<2, 2> > phi_grads_v(n_dofs_per_cell);
//        std::vector<double> phi_p(n_dofs_per_cell);
//        std::vector<double> div_phi_v(n_dofs_per_cell);
//
//        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
//        {
//          for (unsigned int k = 0; k < n_dofs_per_cell; k++)
//          {
//            phi_v[k] = state_fe_values[velocities].value(k, q_point);
//            phi_grads_v[k] = state_fe_values[velocities].gradient(k, q_point);
//            phi_p[k] = state_fe_values[pressure].value(k, q_point);
//            div_phi_v[k] = state_fe_values[velocities].divergence(k, q_point);
//          }
//
//          for (unsigned int i = 0; i < n_dofs_per_cell; i++)
//          {
//            for (unsigned int j = 0; j < n_dofs_per_cell; j++)
//            {
//              local_entry_matrix(i, j) +=
//                  scale
//                      * (_viscosity
//                          * scalar_product(phi_grads_v[j], phi_grads_v[i])
//                          - phi_p[j] * div_phi_v[i]
//                          - (phi_grads_v[j][0][0] + phi_grads_v[j][1][1])
//                              * phi_p[i]) * state_fe_values.JxW(q_point);
//            }
//          }
//        }
//
//      }
//
//      void
//      CellMatrix_T(
//          const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
//          FullMatrix<double> &local_entry_matrix, double scale, double)
//      {
//        CellMatrix(cdc, local_entry_matrix, scale, 1.);
//      }
//
//      void
//      CellRightHandSide(
//          const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
//          dealii::Vector<double> &local_cell_vector, double scale)
//      {
//
//      }
//
//      void
//      FaceEquation(
//          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>&,
//          dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/,
//          double)
//      {
//      }
//
//      void
//      FaceEquation_U(
//          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>&,
//          dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/,
//          double)
//      {
//      }
//
//      void
//      FaceMatrix_T(
//          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>&,
//          dealii::FullMatrix<double> &/*local_entry_matrix*/, double, double)
//      {
//      }
//
//      void
//      FaceMatrix(
//          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>&,
//          dealii::FullMatrix<double> &/*local_entry_matrix*/, double, double)
//      {
//      }
//
//      // Values for boundary integrals
//      void
//      BoundaryEquation(
//          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
//          dealii::Vector<double> &local_cell_vector, double scale,
//          double /*scale_ico*/)
//      {
//
//      }
//
//      // Values for boundary integrals
//      void
//      BoundaryEquation_U(
//          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
//          dealii::Vector<double> &local_cell_vector, double scale,
//          double /*scale_ico*/)
//      {
//
//      }
//
//      void
//      BoundaryRightHandSide(
//          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>&,
//          dealii::Vector<double> &local_cell_vector __attribute__((unused)),
//          double scale __attribute__((unused)))
//      {
//
//      }
//      void
//      BoundaryMatrix(
//          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
//          dealii::FullMatrix<double> &local_entry_matrix, double /*scale_ico*/,
//          double /*scale_ico*/)
//      {
//
//      }
//
//      void
//      FaceRightHandSide(
//          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
//          dealii::Vector<double> &local_rhs, double scale)
//      {
//
//      }
//
//      void
//      StrongBoundaryResidual(
//          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
//          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc_w,
//          double& sum, double scale)
//      {
//        const unsigned int n_q_points = fdc.GetNQPoints();
//        const unsigned int color = fdc.GetBoundaryIndicator();
//
//        _uvalues.resize(n_q_points, Vector<double>(3));
//        _ugrads.resize(n_q_points, vector<Tensor<1, dealdim> >(3));
//        _PI_h_z.resize(n_q_points);
//
//        fdc.GetFaceGradsState("state", _ugrads);
//        fdc.GetFaceValuesState("state", _uvalues);
//
//        fdc_w.GetFaceValuesState("weight_for_primal_residual", _PI_h_z);
//
//        if (color == 1)
//          for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
//          {
//            sum -= scale
//                * (_viscosity
//                    * (_ugrads[q_point][0][0] * _PI_h_z[q_point][0]
//                        + _ugrads[q_point][1][0] * _PI_h_z[q_point][1])
//                    - _uvalues[q_point][2] * _PI_h_z[q_point][0])
//                * fdc.GetFEFaceValuesState().JxW(q_point);
//          }
//
//      }
//
//      void
//      StrongCellResidual(
//          const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
//          const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc_w,
//          double& sum, double scale, double)
//      {
//
//        unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
//        unsigned int n_q_points = cdc.GetNQPoints();
//        const DOpEWrapper::FEValues<dealdim> &state_fe_values =
//            cdc.GetFEValuesState();
//        const DOpEWrapper::FEValues<dealdim> &state_fe_values_weight =
//            cdc_w.GetFEValuesState();
//
//        _fvalues.resize(n_q_points, Vector<double>(3));
//
//        _PI_h_z.resize(n_q_points, Vector<double>(3));
//        _lap_u.resize(n_q_points, Vector<double>(3));
//        _ugrads.resize(n_q_points, vector<Tensor<1, dealdim> >(3));
//
//        cdc.GetLaplaciansState("state", _lap_u);
//        cdc.GetGradsState("state", _ugrads);
//
//        cdc_w.GetValuesState("weight_for_primal_residual", _PI_h_z);
//
//        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
//        {
//          const double divu = _ugrads[q_point][0][0] + _ugrads[q_point][1][1];
//
//          double res_0 = _viscosity * _lap_u[q_point][0]
//              - _ugrads[q_point][2][0];
//          double res_1 = _viscosity * _lap_u[q_point][1]
//              - _ugrads[q_point][2][1];
//
//          sum += scale
//              * (res_0 * _PI_h_z[q_point][0] + res_1 * _PI_h_z[q_point][1]
//                  + divu * _PI_h_z[q_point][2]) * state_fe_values.JxW(q_point);
//        }
//      }
//
//      void
//      StrongFaceResidual(
//          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
//          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc_w,
//          double& sum, double scale)
//      {
//        const unsigned int n_q_points = fdc.GetNQPoints();
//        const unsigned int material_id = fdc.GetMaterialId();
//        const unsigned int material_id_nbr = fdc.GetNbrMaterialId();
//
//        _ugrads.resize(n_q_points, vector<Tensor<1, dealdim> >(3));
//        _ugrads_nbr.resize(n_q_points, vector<Tensor<1, dealdim> >(3));
//        _PI_h_z.resize(n_q_points);
//
//        fdc.GetFaceGradsState("state", _ugrads);
//        fdc.GetNbrFaceGradsState("state", _ugrads_nbr);
//
//        fdc_w.GetFaceValuesState("weight_for_primal_residual", _PI_h_z);
//        vector<Vector<double> > jump(n_q_points, Vector<double>(2));
//
//        for (unsigned int q = 0; q < n_q_points; q++)
//        {
//          //just the jump over the velocity, pressure is cont.
//          jump[q][0] = _viscosity
//              * ((_ugrads_nbr[q][0][0] - _ugrads[q][0][0])
//                  * fdc.GetFEFaceValuesState().normal_vector(q)[0]
//                  + (_ugrads_nbr[q][0][1] - _ugrads[q][0][1])
//                      * fdc.GetFEFaceValuesState().normal_vector(q)[1]);
//
//          jump[q][1] = _viscosity
//              * ((_ugrads_nbr[q][1][0] - _ugrads[q][1][0])
//                  * fdc.GetFEFaceValuesState().normal_vector(q)[0]
//                  + (_ugrads_nbr[q][1][1] - _ugrads[q][1][1])
//                      * fdc.GetFEFaceValuesState().normal_vector(q)[1]);
//        }
//
//        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
//        {
//          sum += scale
//              * (jump[q_point][0] * _PI_h_z[q_point][0]
//                  + jump[q_point][1] * _PI_h_z[q_point][1])
//              * fdc.GetFEFaceValuesState().JxW(q_point);
//        }
//
//      }
//
//      void
//      StrongBoundaryResidual_U(
//          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
//          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc_w,
//          double& sum, double scale)
//      {
//        const unsigned int n_q_points = fdc.GetNQPoints();
//        const unsigned int color = fdc.GetBoundaryIndicator();
//
//        _zgrads.resize(n_q_points, vector<Tensor<1, dealdim> >(3));
//        _zvalues.resize(n_q_points, Vector<double>(3));
//
//        _PI_h_u_grads.resize(n_q_points, vector<Tensor<1, dealdim> >(3));
//
//        fdc.GetFaceGradsState("adjoint_for_ee", _zgrads);
//        fdc.GetFaceValuesState("adjoint_for_ee", _zvalues);
//
//        fdc_w.GetFaceGradsState("weight_for_dual_residual", _PI_h_u_grads);
//
//        if (color == 1)
//          for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
//          {
//            //negative Wert des Randintegrals wird addiert. Nur die Ableitung in x-richtung, da color==1 dem rechten Ausflussrand entspricht
//            sum -= scale
//                * (_viscosity
//                    * (_zgrads[q_point][0][0] * _PI_h_u[q_point][0]
//                        + _zgrads[q_point][1][0] * _PI_h_u[q_point][1])
//                    - _zvalues[q_point][2] * _PI_h_u[q_point][0])
//                * fdc.GetFEFaceValuesState().JxW(q_point);
//          }
//        else if (color == 80)
//          for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
//          {
//            Tensor<2, 2> fluid_pressure;
//            fluid_pressure.clear();
//            fluid_pressure[0][0] = -_PI_h_u[q_point](2);
//            fluid_pressure[1][1] = -_PI_h_u[q_point](2);
//
//            Tensor<2, 2> vgrads;
//            vgrads.clear();
//            vgrads[0][0] = _PI_h_u_grads[q_point][0][0];
//            vgrads[0][1] = _PI_h_u_grads[q_point][0][1];
//            vgrads[1][0] = _PI_h_u_grads[q_point][1][0];
//            vgrads[1][1] = _PI_h_u_grads[q_point][1][1];
//
//            //warum hier -?? Weil Drag negatives Vorzeichen besitzt!
//            sum -= _drag_lift_constant
//                * ((fluid_pressure + _density_fluid * _viscosity * (vgrads))
//                    * fdc.GetFEFaceValuesState().normal_vector(q_point)
//                    * fdc.GetFEFaceValuesState().JxW(q_point))[0];
//
//          }
//
//      }
//
//      void
//      StrongCellResidual_U(
//          const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
//          const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc_w,
//          double& sum, double scale)
//      {
//
//        unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
//        unsigned int n_q_points = cdc.GetNQPoints();
//        const DOpEWrapper::FEValues<dealdim> &state_fe_values =
//            cdc.GetFEValuesState();
//        const DOpEWrapper::FEValues<dealdim> &state_fe_values_weight =
//            cdc_w.GetFEValuesState();
//
//        _fvalues.resize(n_q_points, Vector<double>(3));
//
//        _PI_h_u.resize(n_q_points, Vector<double>(3));
//        _lap_z.resize(n_q_points, Vector<double>(3));
//        _zgrads.resize(n_q_points, vector<Tensor<1, dealdim> >(3));
//
//        cdc.GetLaplaciansState("adjoint_for_ee", _lap_z);
//        cdc.GetGradsState("adjoint_for_ee", _zgrads);
//
//        cdc_w.GetValuesState("weight_for_dual_residual", _PI_h_u);
//
//        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
//        {
//          const double divu = _zgrads[q_point][0][0] + _zgrads[q_point][1][1];
//          double res_0 = _viscosity * _lap_z[q_point][0]
//              - _zgrads[q_point][2][0];
//          double res_1 = _viscosity * _lap_z[q_point][1]
//              - _zgrads[q_point][2][1];
//
//          sum += scale
//              * (res_0 * _PI_h_u[q_point][0] + res_1 * _PI_h_u[q_point][1]
//                  + divu * _PI_h_u[q_point][2]) * state_fe_values.JxW(q_point);
//        }
//      }
//
//      void
//      StrongFaceResidual_U(
//          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
//          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc_w,
//          double& sum, double scale)
//      {
//        const unsigned int n_q_points = fdc.GetNQPoints();
//        const unsigned int material_id = fdc.GetMaterialId();
//        const unsigned int material_id_nbr = fdc.GetNbrMaterialId();
//
//        _zgrads.resize(n_q_points, vector<Tensor<1, dealdim> >(3));
//        _zgrads_nbr.resize(n_q_points, vector<Tensor<1, dealdim> >(3));
//        _PI_h_u.resize(n_q_points);
//
//        fdc.GetFaceGradsState("adjoint_for_ee", _zgrads);
//        fdc.GetNbrFaceGradsState("adjoint_for_ee", _zgrads_nbr);
//
//        fdc_w.GetFaceValuesState("weight_for_dual_residual", _PI_h_u);
//        vector<Vector<double> > jump(n_q_points, Vector<double>(2));
//
//        for (unsigned int q = 0; q < n_q_points; q++)
//        {
//          jump[q][0] = _viscosity * (_zgrads_nbr[q][0][0] - _zgrads[q][0][0])
//              * fdc.GetFEFaceValuesState().normal_vector(q)[0]
//              + _viscosity * (_zgrads_nbr[q][0][1] - _zgrads[q][0][1])
//                  * fdc.GetFEFaceValuesState().normal_vector(q)[1];
//
//          jump[q][1] = _viscosity * (_zgrads_nbr[q][1][0] - _zgrads[q][1][0])
//              * fdc.GetFEFaceValuesState().normal_vector(q)[0]
//              + _viscosity * (_zgrads_nbr[q][1][1] - _zgrads[q][1][1])
//                  * fdc.GetFEFaceValuesState().normal_vector(q)[1];
//        }
//
//        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
//        {
//          sum += scale
//              * (jump[q_point][0] * _PI_h_u[q_point][0]
//                  + jump[q_point][1] * _PI_h_u[q_point][1])
//              * fdc.GetFEFaceValuesState().JxW(q_point);
//        }
//
//      }

      UpdateFlags
      GetUpdateFlags() const
      {
        return update_values | update_gradients | update_hessians
            | update_quadrature_points;
      }

      UpdateFlags
      GetFaceUpdateFlags() const
      {
        return update_values | update_gradients | update_normal_vectors
            | update_quadrature_points;
      }

      unsigned int
      GetStateNBlocks() const
      {
        return 2;
      }
      std::vector<unsigned int>&
      GetStateBlockComponent()
      {
        return _state_block_components;
      }
      const std::vector<unsigned int>&
      GetStateBlockComponent() const
      {
        return _state_block_components;
      }

      virtual bool
      HasFaces() const
      {
        return true;
      }

    private:
      vector<Vector<double> > _fvalues;

      vector<Vector<double> > _uvalues;
      vector<vector<Tensor<1, dealdim> > > _ugrads;
      vector<vector<Tensor<1, dealdim> > > _ugrads_nbr;
      vector<Vector<double> > _lap_u;

      vector<Vector<double> > _zvalues;
      vector<vector<Tensor<1, dealdim> > > _zgrads;
      vector<vector<Tensor<1, dealdim> > > _zgrads_nbr;
      vector<Vector<double> > _lap_z;
      vector<vector<Tensor<1, dealdim> > > _PI_h_zgrads, _PI_h_u_grads;
      vector<Vector<double> > _PI_h_z, _PI_h_u;

      // face values
      vector<vector<Tensor<1, dealdim> > > _ufacegrads;

      vector<unsigned int> _state_block_components;

      double _density_fluid, _viscosity;
      double _drag_lift_constant;
  }
  ;
#endif
