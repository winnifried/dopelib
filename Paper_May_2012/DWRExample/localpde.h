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
  class LocalPDEStokes : public PDEInterface<CellDataContainer,
      FaceDataContainer, dealii::DoFHandler<dealdim>, VECTOR, dealdim>
  {
    public:
      static void
      declare_params(ParameterReader &param_reader)
      {
        param_reader.SetSubsection("Local PDE parameters");
        param_reader.declare_entry("viscosity", "1.0", Patterns::Double(0));
      }

      LocalPDEStokes(ParameterReader &param_reader) :
          _state_block_components(3, 0)
      {
        _state_block_components[2] = 1;

        param_reader.SetSubsection("Local PDE parameters");
        _viscosity = param_reader.get_double("viscosity");
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
          vgrads.clear();
          vgrads[0][0] = _ugrads[q_point][0][0];
          vgrads[0][1] = _ugrads[q_point][0][1];
          vgrads[1][0] = _ugrads[q_point][1][0];
          vgrads[1][1] = _ugrads[q_point][1][1];

          double press = _uvalues[q_point](2);
          double incompressibility = vgrads[0][0] + vgrads[1][1];

          for (unsigned int i = 0; i < n_dofs_per_cell; i++)
          {
            const Tensor<2, 2> phi_i_grads_v =
                state_fe_values[velocities].gradient(i, q_point);
            const double phi_i_p = state_fe_values[pressure].value(i, q_point);
            const double div_phi_v = state_fe_values[velocities].divergence(i,
                q_point);

            local_cell_vector(i) += scale
                * (_viscosity * scalar_product(vgrads, phi_i_grads_v)
                    - press * div_phi_v - incompressibility * phi_i_p)
                * state_fe_values.JxW(q_point);

          }
        }

      }
      void
      CellEquation_U(
          const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
          dealii::Vector<double> &local_cell_vector, double scale, double)
      {
        CellEquation(cdc, local_cell_vector, scale, 1.);
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

        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(2);

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

          for (unsigned int i = 0; i < n_dofs_per_cell; i++)
          {
            for (unsigned int j = 0; j < n_dofs_per_cell; j++)
            {
              local_entry_matrix(i, j) +=
                  scale
                      * (_viscosity
                          * scalar_product(phi_grads_v[j], phi_grads_v[i])
                          - phi_p[j] * div_phi_v[i]
                          - (phi_grads_v[j][0][0] + phi_grads_v[j][1][1])
                              * phi_p[i]) * state_fe_values.JxW(q_point);
            }
          }
        }

      }

      void
      CellMatrix_T(
          const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
          FullMatrix<double> &local_entry_matrix, double scale, double)
      {
        CellMatrix(cdc, local_entry_matrix, scale, 1.);
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
        void BoundaryEquation (const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
             dealii::Vector<double> &local_cell_vector,
             double scale, double /*scale_ico*/)
        {

        }

        // Values for boundary integrals
          void BoundaryEquation_U (const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
               dealii::Vector<double> &local_cell_vector,
               double scale, double /*scale_ico*/)
          {

          }

          void BoundaryRightHandSide (const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>&,
              dealii::Vector<double> &local_cell_vector __attribute__((unused)),
              double scale __attribute__((unused)))
          {
            assert(this->_problem_type == "state");
          }
          void BoundaryMatrix (const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
                dealii::FullMatrix<double> &local_entry_matrix, double /*scale_ico*/, double /*scale_ico*/)
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

        _ugrads.resize(n_q_points, vector<Tensor<1, dealdim> >(3));
        _PI_h_z.resize(n_q_points);

        fdc.GetFaceGradsState("state", _ugrads);

        fdc_w.GetFaceValuesState("weight_for_primal_residual", _PI_h_z);

        if (color == 1)
          for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
          {
            //da - \partial_normal = -1 * -\partial_2 = \partial_2
            sum += scale * _viscosity * _ugrads[q_point][0][1]
                * _PI_h_z[q_point][0] * fdc.GetFEFaceValuesState().JxW(q_point);
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

        cdc.GetLaplaciansState("state", _lap_u);
        cdc.GetGradsState("state", _ugrads);

        cdc_w.GetValuesState("weight_for_primal_residual", _PI_h_z);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          const double divu = _ugrads[q_point][0][0] + _ugrads[q_point][1][1];
          double res_0 = _viscosity * _lap_u[q_point][0]
              - _ugrads[q_point][2][0];
          double res_1 = _viscosity * _lap_u[q_point][1]
              - _ugrads[q_point][2][1];

          sum += scale
              * (res_0 * _PI_h_z[q_point][0] + res_1 * _PI_h_z[q_point][1]
                  + divu * _PI_h_z[q_point][2]) * state_fe_values.JxW(q_point);
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
        {
          jump[q][0] = _viscosity * (_ugrads_nbr[q][0][0] - _ugrads[q][0][0])
              * fdc.GetFEFaceValuesState().normal_vector(q)[0]
              + _viscosity * (_ugrads_nbr[q][0][1] - _ugrads[q][0][1])
                  * fdc.GetFEFaceValuesState().normal_vector(q)[1];

          jump[q][1] = _viscosity * (_ugrads_nbr[q][1][0] - _ugrads[q][1][0])
              * fdc.GetFEFaceValuesState().normal_vector(q)[0]
              + _viscosity * (_ugrads_nbr[q][1][1] - _ugrads[q][1][1])
                  * fdc.GetFEFaceValuesState().normal_vector(q)[1];
        }

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          sum += scale
              * (jump[q_point][0] * _PI_h_z[q_point][0]
                  + jump[q_point][1] * _PI_h_z[q_point][1])
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

    private:
      vector<Vector<double> > _fvalues;

      vector<Vector<double> > _uvalues;
      vector<vector<Tensor<1, dealdim> > > _ugrads;
      vector<vector<Tensor<1, dealdim> > > _ugrads_nbr;
      vector<Vector<double> > _lap_u;

      vector<vector<Tensor<1, dealdim> > > _zgrads;
      vector<vector<Tensor<1, dealdim> > > _PI_h_zgrads;
      vector<Vector<double> > _PI_h_z;

      // face values
      vector<vector<Tensor<1, dealdim> > > _ufacegrads;

      vector<unsigned int> _state_block_components;

      double _viscosity;
  }
  ;
#endif
