#ifndef __LOCALPDE
#define __LOCALPDE

#include "pdeinterface.h"
#include "myfunctions.h"
#include <deal.II/base/numbers.h>

using namespace std;
using namespace dealii;
using namespace DOpE;

/***********************************************************************************************/
template<typename VECTOR, int dealdim>
  class LocalPDELaplace : public PDEInterface<CellDataContainer,
      FaceDataContainer, dealii::DoFHandler<dealdim>, VECTOR, dealdim>
  {
    public:
      LocalPDELaplace()
          : _state_block_components(1, 0)
      {
      }

      // Domain values for cells
      void
      CellEquation(
          const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
          dealii::Vector<double> &local_cell_vector, double scale, double)
      {
        unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
        unsigned int n_q_points = cdc.GetNQPoints();
        const DOpEWrapper::FEValues<dealdim> &state_fe_values =
            cdc.GetFEValuesState();

        assert(this->_problem_type == "state");

        _ugrads.resize(n_q_points, Tensor<1, dealdim>());
        cdc.GetGradsState("last_newton_solution", _ugrads);

        const FEValuesExtractors::Scalar velocities(0);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          Tensor<1, 2> vgrads;
          vgrads.clear();
          vgrads[0] = _ugrads[q_point][0];
          vgrads[1] = _ugrads[q_point][1];

          for (unsigned int i = 0; i < n_dofs_per_cell; i++)
          {
            const Tensor<1, 2> phi_i_grads_v =
                state_fe_values[velocities].gradient(i, q_point);

            local_cell_vector(i) += scale * (vgrads * phi_i_grads_v)
                * state_fe_values.JxW(q_point);
          }
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

        _fvalues.resize(n_q_points);

        _PI_h_z.resize(n_q_points);
        _lap_u.resize(n_q_points);
        cdc.GetLaplaciansState("state", _lap_u);
        cdc_w.GetValuesState("weight_for_primal_residual", _PI_h_z);

        const FEValuesExtractors::Scalar velocities(0);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
//          const double x = state_fe_values.quadrature_point(q_point)(0);
//          const double y = state_fe_values.quadrature_point(q_point)(1);

          _fvalues[q_point] = -_ex_sol.laplacian(
              state_fe_values.quadrature_point(q_point));
//          _fvalues[q_point][0] = cos(exp(10 * x)) * y * y * x + sin(y);
//          _fvalues[q_point][1] = cos(exp(10 * y)) * x * x * y + sin(x);
          double res;
          res = _fvalues[q_point] + _lap_u[q_point];

          sum += scale * (res * _PI_h_z[q_point])
              * state_fe_values.JxW(q_point);
        }
      }

      void
      StrongFaceResidual(
          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc_w,
          double& sum, double scale)
      {

        unsigned int n_q_points = fdc.GetNQPoints();
        _ugrads.resize(n_q_points, Tensor<1, dealdim>());
        _ugrads_nbr.resize(n_q_points, Tensor<1, dealdim>());
        _PI_h_z.resize(n_q_points);

        fdc.GetFaceGradsState("state", _ugrads);
        fdc.GetNbrFaceGradsState("state", _ugrads_nbr);
        const auto & facefevalues = fdc_w.GetFEFaceValuesState();
        fdc_w.GetFaceValuesState("weight_for_primal_residual", _PI_h_z);
        vector<double> jump(n_q_points);
        for (unsigned int q = 0; q < n_q_points; q++)
        {
          jump[q] = (_ugrads[q][0] - _ugrads_nbr[q][0])
              * fdc.GetFEFaceValuesState().normal_vector(q)[0]
              + (_ugrads[q][1] - _ugrads_nbr[q][1])
                  * fdc.GetFEFaceValuesState().normal_vector(q)[1];
        }

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          sum += scale * (jump[q_point] * _PI_h_z[q_point])
              * fdc.GetFEFaceValuesState().JxW(q_point);
        }
      }

      void
      StrongBoundaryResidual(
          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc_w,
          double& sum, double /*scale*/)
      {
        sum = 0;
      }

      void
      FaceEquation_U(
          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
          dealii::Vector<double> &local_cell_vector, double scale)
      {

      }

      void
      FaceMatrix(
          const FaceDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& fdc,
          FullMatrix<double> &local_entry_matrix)
      {

      }

      void
      CellEquation_U(
          const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
          dealii::Vector<double> &local_cell_vector, double scale)
      {
        const DOpEWrapper::FEValues<dealdim> & state_fe_values =
            cdc.GetFEValuesState();
        unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
        unsigned int n_q_points = cdc.GetNQPoints();

        assert(this->_problem_type == "adjoint_for_ee");
        _zgrads.resize(n_q_points, Tensor<1, dealdim>());
        //We don't need u so we don't search for state
        cdc.GetGradsState("last_newton_solution", _zgrads);

        const FEValuesExtractors::Scalar velocities(0);
        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          Tensor<1, 2> vgrads;
          vgrads.clear();
          vgrads[0] = _zgrads[q_point][0];
          vgrads[1] = _zgrads[q_point][1];
          for (unsigned int i = 0; i < n_dofs_per_cell; i++)
          {
            const Tensor<1, 2> phi_i_grads_v =
                state_fe_values[velocities].gradient(i, q_point);
            local_cell_vector(i) += scale * vgrads * phi_i_grads_v
                * state_fe_values.JxW(q_point);
          }
        }
      }

      void
      CellMatrix(
          const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
          FullMatrix<double> &local_entry_matrix, double scale, double)
      {
        unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
        unsigned int n_q_points = cdc.GetNQPoints();
        //unsigned int material_id = cdc.GetMaterialId();
        const DOpEWrapper::FEValues<dealdim> &state_fe_values =
            cdc.GetFEValuesState();

        const FEValuesExtractors::Scalar velocities(0);

        std::vector<Tensor<1, 2> > phi_grads_v(n_dofs_per_cell);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          for (unsigned int k = 0; k < n_dofs_per_cell; k++)
          {
            phi_grads_v[k] = state_fe_values[velocities].gradient(k, q_point);
          }

          for (unsigned int i = 0; i < n_dofs_per_cell; i++)
          {
            for (unsigned int j = 0; j < n_dofs_per_cell; j++)
            {

              local_entry_matrix(i, j) += scale * phi_grads_v[j]
                  * phi_grads_v[i] * state_fe_values.JxW(q_point);
            }
          }
        }
      }

      void
      CellMatrix_T(
          const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
          FullMatrix<double> &local_entry_matrix, double scale, double)
      {
        unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
        unsigned int n_q_points = cdc.GetNQPoints();
        //unsigned int material_id = cdc.GetMaterialId();
        const DOpEWrapper::FEValues<dealdim> &state_fe_values =
            cdc.GetFEValuesState();

        const FEValuesExtractors::Scalar velocities(0);

        std::vector<Tensor<1, 2> > phi_grads_v(n_dofs_per_cell);

        for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          for (unsigned int k = 0; k < n_dofs_per_cell; k++)
          {
            phi_grads_v[k] = state_fe_values[velocities].gradient(k, q_point);
          }

          for (unsigned int i = 0; i < n_dofs_per_cell; i++)
          {
            for (unsigned int j = 0; j < n_dofs_per_cell; j++)
            {

              local_entry_matrix(i, j) += scale * phi_grads_v[j]
                  * phi_grads_v[i] * state_fe_values.JxW(q_point);
            }
          }
        }
      }

      void
      CellRightHandSide(
          const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
          dealii::Vector<double> &local_cell_vector, double scale)
      {
        assert(this->_problem_type == "state");
        unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
        unsigned int n_q_points = cdc.GetNQPoints();
        const DOpEWrapper::FEValues<dealdim> &state_fe_values =
            cdc.GetFEValuesState();

        _fvalues.resize(n_q_points);
        const FEValuesExtractors::Scalar velocities(0);

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
          _fvalues[q_point] = -_ex_sol.laplacian(
              state_fe_values.quadrature_point(q_point));

          for (unsigned int i = 0; i < n_dofs_per_cell; i++)
          {
            local_cell_vector(i) += scale * _fvalues[q_point]
                * state_fe_values[velocities].value(i, q_point)
                * state_fe_values.JxW(q_point);
          }
        } //endfor qpoint
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
        return 1;
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
      bool
      HasFaces() const
      {
        return false;
      }
      bool
      HasInterfaces() const
      {
        return false;
      }
    private:

      vector<double> _fvalues;
      vector<double> _PI_h_z;
      vector<double> _lap_u;

      vector<Tensor<1, dealdim> > _ugrads;
      vector<Tensor<1, dealdim> > _ugrads_nbr;

      vector<Tensor<1, dealdim> > _zgrads;

      vector<unsigned int> _state_block_components;

      ExactSolution _ex_sol;
  };
//**********************************************************************************

#endif

