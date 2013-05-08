/**
*
* Copyright (C) 2012 by the DOpElib authors
*
* This file is part of DOpElib
*
* DOpElib is free software: you can redistribute it
* and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either
* version 3 of the License, or (at your option) any later
* version.
*
* DOpElib is distributed in the hope that it will be
* useful, but WITHOUT ANY WARRANTY; without even the implied
* warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
* PURPOSE.  See the GNU General Public License for more
* details.
*
* Please refer to the file LICENSE.TXT included in this distribution
* for further information on this license.
*
**/

#ifndef _LOCALPDE_
#define _LOCALPDE_

#include "pdeinterface.h"
#include "celldatacontainer.h"
#include "facedatacontainer.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

template<typename VECTOR, int dopedim, int dealdim>
  class LocalPDE : public PDEInterface<CellDataContainer,FaceDataContainer,dealii::DoFHandler, VECTOR,
      dopedim, dealdim>
  {
  public:

    static void
    declare_params(ParameterReader &param_reader)
    {
      param_reader.SetSubsection("Local PDE parameters");
      param_reader.declare_entry("interest rate", "0.", Patterns::Double(0));
      param_reader.declare_entry("volatility_1", "0.", Patterns::Double(0));
      param_reader.declare_entry("volatility_2", "0.", Patterns::Double(0));
      param_reader.declare_entry("rho", ".0", Patterns::Double(-1, 1));
      param_reader.declare_entry("strike price", "0.", Patterns::Double(0));
      param_reader.declare_entry("expiration date", "1.0", Patterns::Double(0));
    }

    LocalPDE(ParameterReader &param_reader) :
      _state_block_components(1, 0)
    {
      param_reader.SetSubsection("Local PDE parameters");
      _rate = param_reader.get_double("interest rate");
      _volatility(0) = param_reader.get_double("volatility_1");
      _volatility(1) = param_reader.get_double("volatility_2");
      _rho = param_reader.get_double("rho");
      _strike = param_reader.get_double("strike price");
    }

    // Domain values for cells
    void
    CellEquation(
        const CellDataContainer<dealii::DoFHandler, VECTOR, dealdim>& cdc,
        dealii::Vector<double> &local_cell_vector, double scale, double /*scale_ico*/)
    {
      assert(this->_problem_type == "state");

      const DOpEWrapper::FEValues<dealdim> & state_fe_values =
          cdc.GetFEValuesState();
      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
      unsigned int n_q_points = cdc.GetNQPoints();

      _uvalues.resize(n_q_points);
      _ugrads.resize(n_q_points);

      cdc.GetValuesState("last_newton_solution", _uvalues);
      cdc.GetGradsState("last_newton_solution", _ugrads);

      Tensor<2, 2> CoeffMatrix;
      Tensor<1, 2> CoeffVector;
      const double correlation = 2 * _rho * 1. / (1 + _rho * _rho);

      for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          const double x = state_fe_values.quadrature_point(q_point)[0];
          const double y = state_fe_values.quadrature_point(q_point)[1];

          for (unsigned int i = 0; i < n_dofs_per_cell; i++)
            {
              CoeffMatrix.clear();
              CoeffVector.clear();

              CoeffMatrix[0][0] = _volatility(0) * _volatility(0) * x * x;
              CoeffMatrix[1][0] = _volatility(1) * _volatility(0) * x * y
                  * correlation;
              CoeffMatrix[0][1] = _volatility(0) * _volatility(1) * y * x
                  * correlation;
              CoeffMatrix[1][1] = _volatility(1) * _volatility(1) * y * y;
              CoeffMatrix.operator*=(0.5);

              CoeffVector[0] = x * (_volatility(0) * _volatility(0) + 0.5
                  * correlation * _volatility(0) * _volatility(1) - _rate);
              CoeffVector[1] = y * (_volatility(1) * _volatility(1) + 0.5
                  * correlation * _volatility(0) * _volatility(1) - _rate);

              const double phi_i = state_fe_values.shape_value(i, q_point);
              const Tensor<1, 2> phi_i_grads = state_fe_values.shape_grad(i,
                  q_point);

              local_cell_vector(i) += scale * ((CoeffMatrix * _ugrads[q_point])
                  * phi_i_grads + (CoeffVector * _ugrads[q_point]) * phi_i
                  + _rate * _uvalues[q_point] * phi_i) * state_fe_values.JxW(
                  q_point);

            }
        }
    }


    void
    CellMatrix(
        const CellDataContainer<dealii::DoFHandler, VECTOR, dealdim>& cdc,
        FullMatrix<double> &local_entry_matrix, double scale, double)
    {
      const DOpEWrapper::FEValues<dealdim> & state_fe_values =
          cdc.GetFEValuesState();
      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
      unsigned int n_q_points = cdc.GetNQPoints();

      const FEValuesExtractors::Vector velocities(0);
      const FEValuesExtractors::Scalar pressure(2);

      std::vector<double> phi(n_dofs_per_cell);
      std::vector<Tensor<1, 2> > phi_grads(n_dofs_per_cell);

      Tensor<2, 2> CoeffMatrix;
      Tensor<1, 2> CoeffVector;
      const double correlation = 2 * _rho * 1. / (1 + _rho * _rho);

      for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          for (unsigned int k = 0; k < n_dofs_per_cell; k++)
            {
              phi[k] = state_fe_values.shape_value(k, q_point);
              phi_grads[k] = state_fe_values.shape_grad(k, q_point);
            }

          const double x = state_fe_values.quadrature_point(q_point)[0];
          const double y = state_fe_values.quadrature_point(q_point)[1];

          for (unsigned int i = 0; i < n_dofs_per_cell; i++)
            {
              CoeffMatrix.clear();
              CoeffVector.clear();

              CoeffMatrix[0][0] = _volatility(0) * _volatility(0) * x * x;
              CoeffMatrix[1][0] = _volatility(1) * _volatility(0) * x * y
                  * correlation;
              CoeffMatrix[0][1] = _volatility(0) * _volatility(1) * y * x
                  * correlation;
              CoeffMatrix[1][1] = _volatility(1) * _volatility(1) * y * y;
              CoeffMatrix.operator *=(0.5);

              CoeffVector[0] = x * (_volatility(0) * _volatility(0) + 0.5
                  * correlation * _volatility(0) * _volatility(1) - _rate);
              CoeffVector[1] = y * (_volatility(1) * _volatility(1) + 0.5
                  * correlation * _volatility(0) * _volatility(1) - _rate);

              for (unsigned int j = 0; j < n_dofs_per_cell; j++)
                {
                  local_entry_matrix(i, j) += scale * ((phi_grads[j] * CoeffMatrix)
                      * phi_grads[i] + (CoeffVector * phi_grads[j]) * phi[i]
                      + _rate * phi[j] * phi[i]) * state_fe_values.JxW(q_point);
                }
            }
        }
    }


    void
    CellRightHandSide(
      const CellDataContainer<dealii::DoFHandler, VECTOR, dealdim>& /*cdc*/,
        dealii::Vector<double> &local_cell_vector __attribute__((unused)), double scale __attribute__((unused)))
    {
      assert(this->_problem_type == "state");
      //i.e. f=0
    }

    void
    CellTimeEquationExplicit(
      const CellDataContainer<dealii::DoFHandler, VECTOR, dealdim>& /*cdc*/,
        dealii::Vector<double> &local_cell_vector __attribute__((unused)), double scale __attribute__((unused)))
    {
      assert(this->_problem_type == "state");
    }

    void
    CellTimeEquation(
        const CellDataContainer<dealii::DoFHandler, VECTOR, dealdim>& cdc,
        dealii::Vector<double> &local_cell_vector __attribute__((unused)), double scale __attribute__((unused)))
    {
      assert(this->_problem_type == "state");

      const DOpEWrapper::FEValues<dealdim> & state_fe_values =
          cdc.GetFEValuesState();
      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
      unsigned int n_q_points = cdc.GetNQPoints();

      _uvalues.resize(n_q_points);

      cdc.GetValuesState("last_newton_solution", _uvalues);

      for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          for (unsigned int i = 0; i < n_dofs_per_cell; i++)
            {
              const double phi_i = state_fe_values.shape_value(i, q_point);

              local_cell_vector(i) += scale * (_uvalues[q_point] * phi_i)
                  * state_fe_values.JxW(q_point);
            }
        }
    }

    void
    CellTimeMatrixExplicit(
      const CellDataContainer<dealii::DoFHandler, VECTOR, dealdim>& /*cdc*/,
      FullMatrix<double> &/*local_entry_matrix*/)
    {
      assert(this->_problem_type == "state");
    }

    void
    CellTimeMatrix(
        const CellDataContainer<dealii::DoFHandler, VECTOR, dealdim>& cdc,
        FullMatrix<double> &local_entry_matrix)
    {
      assert(this->_problem_type == "state");

      const DOpEWrapper::FEValues<dealdim> & state_fe_values =
          cdc.GetFEValuesState();
      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
      unsigned int n_q_points = cdc.GetNQPoints();

      std::vector<double> phi(n_dofs_per_cell);

      for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          for (unsigned int k = 0; k < n_dofs_per_cell; k++)
            {
              phi[k] = state_fe_values.shape_value(k, q_point);
            }

          for (unsigned int i = 0; i < n_dofs_per_cell; i++)
            {
              for (unsigned int j = 0; j < n_dofs_per_cell; j++)
                {
                  local_entry_matrix(j, i) += (phi[i] * phi[j])
                      * state_fe_values.JxW(q_point);
                }
            }
        }

    }

    // Values for boundary integrals
    void
    BoundaryEquation(
      const FaceDataContainer<dealii::DoFHandler, VECTOR, dealdim>& /*fdc*/,
      dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/, double /*scale_ico*/)
    {

      assert(this->_problem_type == "state");

    }

    void
    BoundaryRightHandSide(
      const FaceDataContainer<dealii::DoFHandler, VECTOR, dealdim>& /*fdc*/,
      dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/)
    {
      assert(this->_problem_type == "state");
    }

    UpdateFlags
    GetUpdateFlags() const
    {
      if ((this->_problem_type == "adjoint")
          || (this->_problem_type == "state") || (this->_problem_type
          == "tangent") || (this->_problem_type == "adjoint_hessian")
          || (this->_problem_type == "hessian"))
        return update_values | update_gradients | update_quadrature_points;
      else if ((this->_problem_type == "gradient"))
        return update_values | update_quadrature_points;
      else
        throw DOpEException("Unknown Problem Type " + this->_problem_type,
            "LocalPDE::GetUpdateFlags");
    }

    UpdateFlags
    GetFaceUpdateFlags() const
    {
      if ((this->_problem_type == "adjoint")
          || (this->_problem_type == "state") || (this->_problem_type
          == "tangent") || (this->_problem_type == "adjoint_hessian")
          || (this->_problem_type == "hessian"))
        return update_values | update_gradients | update_normal_vectors
            | update_quadrature_points;
      else if ((this->_problem_type == "gradient"))
        return update_values | update_quadrature_points;
      else
        throw DOpEException("Unknown Problem Type " + this->_problem_type,
            "LocalPDE::GetUpdateFlags");
    }

    unsigned int
    GetControlNBlocks() const
    {
      return 1;
    }

    unsigned int
    GetStateNBlocks() const
    {
      return 1;
    }

    std::vector<unsigned int>&
    GetControlBlockComponent()
    {
      return _block_components;
    }
    const std::vector<unsigned int>&
    GetControlBlockComponent() const
    {
      return _block_components;
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
  private:
    vector<double> _qvalues;
    vector<double> _dqvalues;
    vector<double> _funcgradvalues;
    vector<double> _fvalues;
    vector<double> _uvalues;

    vector<Tensor<1, dealdim> > _ugrads;
    vector<double> _zvalues;
    vector<Tensor<1, dealdim> > _zgrads;
    vector<double> _duvalues;
    vector<Tensor<1, dealdim> > _dugrads;
    vector<double> _dzvalues;
    vector<Tensor<1, dealdim> > _dzgrads;

    // face values
    vector<double> _qfacevalues;
    vector<Vector<double> > _ffacevalues;
    vector<Vector<double> > _ufacevalues;
    vector<vector<Tensor<1, dealdim> > > _ufacegrads;

    vector<unsigned int> _state_block_components;
    vector<unsigned int> _block_components;
    //double _alpha, _cell_diameter;

    double _rate, _rho, _strike, _expiration_date, _theta;
    Point<dealdim> _volatility;

  };
#endif
