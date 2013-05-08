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

#include "pdeinterface.h"
#include "dopeexception.h"

#include <iostream>

using namespace dealii;

namespace DOpE
{

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::PDEInterface()
    {

    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::~PDEInterface()
    {

    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::CellEquation(
        const CDC<DH, VECTOR, dealdim>& /*cdc*/,
        dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/,
        double /*scale_ico*/)
    {
      throw DOpEException("Not Implemented", "PDEInterface::CellEquation");
    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::StrongCellResidual(
        const CDC<DH, VECTOR, dealdim>& /*cdc*/,
        const CDC<DH, VECTOR, dealdim>& /*cdc_weight*/, double&,
        double /*scale*/)
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::StrongCellResidual");
    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::CellTimeEquation(
        const CDC<DH, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/)
    {
      throw DOpEException("Not Implemented", "PDEInterface::CellTimeEquation");
    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::CellTimeEquation_U(
        const CDC<DH, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/)
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::CellTimeEquation_U");
    }
  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::CellTimeEquation_UT(
        const CDC<DH, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/)
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::CellTimeEquation_UT");
    }
  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::CellTimeEquation_UTT(
        const CDC<DH, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/)
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::CellTimeEquation_UTT");
    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::CellTimeEquationExplicit(
        const CDC<DH, VECTOR, dealdim>& /*cdc*/,
        dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/)
    {
      //This should be left empty, then one can use the default case *Time* without the 
      //need to implement CellTimeEquationExplicit
    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::CellTimeEquationExplicit_U(
        const CDC<DH, VECTOR, dealdim>& /*cdc*/,
        dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/)
    {
      //This should be left empty, then one can use the default case *Time* without the 
      //need to implement CellTimeEquationExplicit
    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::CellTimeEquationExplicit_UT(
        const CDC<DH, VECTOR, dealdim>& /*cdc*/,
        dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/)
    {
      //This should be left empty, then one can use the default case *Time* without the 
      //need to implement CellTimeEquationExplicit
    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::CellTimeEquationExplicit_UTT(
        const CDC<DH, VECTOR, dealdim>& /*cdc*/,
        dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/)
    {
      //This should be left empty, then one can use the default case *Time* without the 
      //need to implement CellTimeEquationExplicit
    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::CellTimeEquationExplicit_UU(
        const CDC<DH, VECTOR, dealdim>& /*cdc*/,
        dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/)
    {
      //This should be left empty, then one can use the default case *Time* without the 
      //need to implement CellTimeEquationExplicit
    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::CellEquation_U(
        const CDC<DH, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/,
        double /*scale_ico*/)
    {
      throw DOpEException("Not Implemented", "PDEInterface::CellEquation_U");
    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::StrongCellResidual_U(
        const CDC<DH, VECTOR, dealdim>&,
        const CDC<DH, VECTOR, dealdim>& /*cdc_weight*/, double&,
        double /*scale*/)
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::StrongCellResidual_U");
    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::CellEquation_UT(
        const CDC<DH, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/,
        double /*scale_ico*/)
    {
      throw DOpEException("Not Implemented", "PDEInterface::CellEquation_UT");
    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::CellEquation_UTT(
        const CDC<DH, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/,
        double /*scale_ico*/)
    {
      throw DOpEException("Not Implemented", "PDEInterface::CellEquation_UTT");
    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::CellEquation_Q(
        const CDC<DH, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/,
        double /*scale_ico*/)
    {
      throw DOpEException("Not Implemented", "PDEInterface::CellEquation_Q");
    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::CellEquation_QT(
        const CDC<DH, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/,
        double /*scale_ico*/)
    {
      throw DOpEException("Not Implemented", "PDEInterface::CellEquation_QT");
    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::CellEquation_QTT(
        const CDC<DH, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/,
        double /*scale_ico*/)
    {
      throw DOpEException("Not Implemented", "PDEInterface::CellEquation_QTT");
    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::CellEquation_UU(
        const CDC<DH, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/,
        double /*scale_ico*/)
    {
      throw DOpEException("Not Implemented", "PDEInterface::CellEquation_UU");
    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::CellEquation_QU(
        const CDC<DH, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/,
        double /*scale_ico*/)
    {
      throw DOpEException("Not Implemented", "PDEInterface::CellEquation_QU");
    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::CellEquation_UQ(
        const CDC<DH, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/,
        double /*scale_ico*/)
    {
      throw DOpEException("Not Implemented", "PDEInterface::CellEquation_UQ");
    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::CellEquation_QQ(
        const CDC<DH, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/,
        double /*scale_ico*/)
    {
      throw DOpEException("Not Implemented", "PDEInterface::CellEquation_QQ");
    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::ControlCellEquation(
        const CDC<DH, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/)
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::ControlCellEquation");
    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::CellRightHandSide(
        const CDC<DH, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/)
    {
      throw DOpEException("Not Implemented", "PDEInterface::CellRightHandSide");
    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::ControlCellMatrix(
        const CDC<DH, VECTOR, dealdim>&,
        FullMatrix<double> &/*local_entry_matrix*/)
    {
      throw DOpEException("Not Implemented", "PDEInterface::ControlCellMatrix");
    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::StrongCellResidual_Control(
        const CDC<DH, VECTOR, dealdim>&, const CDC<DH, VECTOR, dealdim>&,
        double&, double)
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::StrongCellResidual_Control");
    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::StrongFaceResidual_Control(
        const FDC<DH, VECTOR, dealdim>&, const FDC<DH, VECTOR, dealdim>&,
        double&, double)
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::StrongFaceResidual_Control");
    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::StrongBoundaryResidual_Control(
        const FDC<DH, VECTOR, dealdim>&, const FDC<DH, VECTOR, dealdim>&,
        double&, double)
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::StrongBoundaryResidual_Control");
    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::CellMatrix(
        const CDC<DH, VECTOR, dealdim>&,
        FullMatrix<double> &/*local_entry_matrix*/, double/*scale*/,
        double /*scale_ico*/)
    {
      throw DOpEException("Not Implemented", "PDEInterface::CellMatrix");
    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::CellTimeMatrix(
        const CDC<DH, VECTOR, dealdim>&,
        FullMatrix<double> &/*local_entry_matrix*/)
    {
      throw DOpEException("Not Implemented", "PDEInterface::CellTimeMatrix");
    }
  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::CellTimeMatrix_T(
        const CDC<DH, VECTOR, dealdim>& cdc,
        FullMatrix<double> &local_entry_matrix)
    {
      FullMatrix<double> tmp_mat = local_entry_matrix;
      tmp_mat = 0.;

      //FIXME is this the right behaviour in the instationary case? or what
      //are the correct values for scale and scale_ico?
      CellTimeMatrix(cdc, tmp_mat);
      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();

      for (unsigned int i = 0; i < n_dofs_per_cell; i++)
      {
        for (unsigned int j = 0; j < n_dofs_per_cell; j++)
        {
          local_entry_matrix(j, i) += tmp_mat(i, j);
        }
      }
    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::CellTimeMatrixExplicit(
        const CDC<DH, VECTOR, dealdim>& /*cdc*/,
        FullMatrix<double> &/*local_entry_matrix*/)
    {
      //This should be left empty, then one can use the default case *Time* without the 
      //need to implement CellTimeEquationExplicit
    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::CellTimeMatrixExplicit_T(
        const CDC<DH, VECTOR, dealdim>& cdc,
        FullMatrix<double> &local_entry_matrix)
    {
      FullMatrix<double> tmp_mat = local_entry_matrix;
      tmp_mat = 0.;

      //FIXME is this the right behaviour in the instationary case? or what
      //are the correct values for scale and scale_ico?
      CellTimeMatrixExplicit(cdc, tmp_mat);
      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();

      for (unsigned int i = 0; i < n_dofs_per_cell; i++)
      {
        for (unsigned int j = 0; j < n_dofs_per_cell; j++)
        {
          local_entry_matrix(j, i) += tmp_mat(i, j);
        }
      }
    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::CellMatrix_T(
        const CDC<DH, VECTOR, dealdim>& cdc,
        FullMatrix<double> &local_entry_matrix, double scale, double scale_ico)
    {
      FullMatrix<double> tmp_mat = local_entry_matrix;
      tmp_mat = 0.;

      //FIXME is this the right behaviour in the instationary case? or what
      //are the correct values for scale and scale_ico?
      CellMatrix(cdc, tmp_mat, scale, scale_ico);
      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();

      for (unsigned int i = 0; i < n_dofs_per_cell; i++)
      {
        for (unsigned int j = 0; j < n_dofs_per_cell; j++)
        {
          local_entry_matrix(j, i) += tmp_mat(i, j);
        }
      }
    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::FaceEquation(
        const FDC<DH, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/,
        double /*scale_ico*/)
    {
      throw DOpEException("Not Implemented", "PDEInterface::FaceEquation");
    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::StrongFaceResidual(
        const FDC<DH, VECTOR, dealdim>&,
        const FDC<DH, VECTOR, dealdim>& /*fdc_weight*/, double&,
        double /*scale*/)
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::StrongFaceResidual");
    }

  /********************************************/
  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::FaceEquation_U(
        const FDC<DH, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/,
        double /*scale_ico*/)
    {
      throw DOpEException("Not Implemented", "PDEInterface::FaceEquation_U");
    }

  /********************************************/
  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::StrongFaceResidual_U(
        const FDC<DH, VECTOR, dealdim>&,
        const FDC<DH, VECTOR, dealdim>& /*fdc_weight*/, double&,
        double /*scale*/)
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::StrongFaceResidual_U");
    }

  /********************************************/
  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::FaceEquation_UT(
        const FDC<DH, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/,
        double /*scale_ico*/)
    {
      throw DOpEException("Not Implemented", "PDEInterface::FaceEquation_UT");
    }

  /********************************************/
  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::FaceEquation_UTT(
        const FDC<DH, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/,
        double /*scale_ico*/)
    {
      throw DOpEException("Not Implemented", "PDEInterface::FaceEquation_UTT");
    }

  /********************************************/
  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::FaceEquation_Q(
        const FDC<DH, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/,
        double /*scale_ico*/)
    {
      throw DOpEException("Not Implemented", "PDEInterface::FaceEquation_Q");
    }

  /********************************************/
  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::FaceEquation_QT(
        const FDC<DH, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/,
        double /*scale_ico*/)
    {
      throw DOpEException("Not Implemented", "PDEInterface::FaceEquation_QT");
    }

  /********************************************/
  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::FaceEquation_QTT(
        const FDC<DH, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/,
        double /*scale_ico*/)
    {
      throw DOpEException("Not Implemented", "PDEInterface::FaceEquation_QTT");
    }

  /********************************************/
  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::FaceEquation_UU(
        const FDC<DH, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/,
        double /*scale_ico*/)
    {
      throw DOpEException("Not Implemented", "PDEInterface::FaceEquation_UU");
    }

  /********************************************/
  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::FaceEquation_QU(
        const FDC<DH, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/,
        double /*scale_ico*/)
    {
      throw DOpEException("Not Implemented", "PDEInterface::FaceEquation_QU");
    }

  /********************************************/
  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::FaceEquation_UQ(
        const FDC<DH, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/,
        double /*scale_ico*/)
    {
      throw DOpEException("Not Implemented", "PDEInterface::FaceEquation_UQ");
    }

  /********************************************/
  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::FaceEquation_QQ(
        const FDC<DH, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/,
        double /*scale_ico*/)
    {
      throw DOpEException("Not Implemented", "PDEInterface::FaceEquation_QQ");
    }

  /********************************************/
  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::FaceRightHandSide(
        const FDC<DH, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/)
    {
      throw DOpEException("Not Implemented", "PDEInterface::FaceRightHandSide");
    }

  /********************************************/
  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::FaceMatrix(
        const FDC<DH, VECTOR, dealdim>&,
        FullMatrix<double> &/*local_entry_matrix*/, double /*scale*/,
        double /*scale_ico*/)
    {
      throw DOpEException("Not Implemented", "PDEInterface::FaceMatrix");
    }

  /********************************************/
  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::FaceMatrix_T(
        const FDC<DH, VECTOR, dealdim>& fdc,
        FullMatrix<double> &local_entry_matrix, double scale, double scale_ico)
    {
      FullMatrix<double> tmp_mat = local_entry_matrix;
      tmp_mat = 0.;

      FaceMatrix(fdc, tmp_mat, scale, scale_ico);
      unsigned int n_dofs_per_cell = fdc.GetNDoFsPerCell();

      for (unsigned int i = 0; i < n_dofs_per_cell; i++)
      {
        for (unsigned int j = 0; j < n_dofs_per_cell; j++)
        {
          local_entry_matrix(j, i) += tmp_mat(i, j);
        }
      }
    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::InterfaceEquation(
        const FDC<DH, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/,
        double /*scale_ico*/)
    {
      throw DOpEException("Not Implemented", "PDEInterface::InterfaceEquation");
    }
  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::InterfaceEquation_U(
        const FDC<DH, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/,
        double /*scale_ico*/)
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::InterfaceEquation_U");
    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::InterfaceMatrix(
        const FDC<DH, VECTOR, dealdim>&,
        FullMatrix<double> &/*local_entry_matrix*/, double /*scale*/,
        double /*scale_ico*/)
    {
      throw DOpEException("Not Implemented", "PDEInterface::InterfaceMatrix");
    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::InterfaceMatrix_T(
        const FDC<DH, VECTOR, dealdim>&,
        FullMatrix<double> &/*local_entry_matrix*/, double /*scale*/,
        double /*scale_ico*/)
    {
      throw DOpEException("Not Implemented", "PDEInterface::InterfaceMatrix_T");
    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::BoundaryEquation(
        const FDC<DH, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/,
        double /*scale_ico*/)
    {
      throw DOpEException("Not Implemented", "PDEInterface::BoundaryEquation");
    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::StrongBoundaryResidual(
        const FDC<DH, VECTOR, dealdim>&,
        const FDC<DH, VECTOR, dealdim>& /*Fdc_weight*/, double&,
        double /*scale*/)
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::StrongBoundaryResidual");
    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::BoundaryEquation_U(
        const FDC<DH, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/,
        double /*scale_ico*/)
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::BoundaryEquation_U");
    }

  /********************************************/
  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::StrongBoundaryResidual_U(
        const FDC<DH, VECTOR, dealdim>&,
        const FDC<DH, VECTOR, dealdim>& /*fdc_weight*/, double&,
        double /*scale*/)
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::StrongBoundaryResidual_U");
    }

  /********************************************/
  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::BoundaryEquation_UT(
        const FDC<DH, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/,
        double /*scale_ico*/)
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::BoundaryEquation_UT");
    }

  /********************************************/
  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::BoundaryEquation_UTT(
        const FDC<DH, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/,
        double /*scale_ico*/)
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::BoundaryEquation_UTT");
    }

  /********************************************/
  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::BoundaryEquation_Q(
        const FDC<DH, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/,
        double /*scale_ico*/)
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::BoundaryEquation_Q");
    }

  /********************************************/
  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::BoundaryEquation_QT(
        const FDC<DH, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/,
        double /*scale_ico*/)
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::BoundaryEquation_QT");
    }

  /********************************************/
  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::BoundaryEquation_QTT(
        const FDC<DH, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/,
        double /*scale_ico*/)
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::BoundaryEquation_QTT");
    }

  /********************************************/
  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::BoundaryEquation_UU(
        const FDC<DH, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/,
        double /*scale_ico*/)
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::BoundaryEquation_UU");
    }

  /********************************************/
  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::BoundaryEquation_QU(
        const FDC<DH, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/,
        double /*scale_ico*/)
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::BoundaryEquation_QU");
    }

  /********************************************/
  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::BoundaryEquation_UQ(
        const FDC<DH, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/,
        double /*scale_ico*/)
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::BoundaryEquation_UQ");
    }

  /********************************************/
  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::BoundaryEquation_QQ(
        const FDC<DH, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/,
        double /*scale_ico*/)
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::BoundaryEquation_QQ");
    }

  /********************************************/
  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::BoundaryRightHandSide(
        const FDC<DH, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/)
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::BoundaryRightHandSide");
    }

  /********************************************/
  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::BoundaryMatrix(
        const FDC<DH, VECTOR, dealdim>&,
        FullMatrix<double> &/*local_entry_matrix*/, double /*scale*/,
        double /*scale_ico*/)
    {
      throw DOpEException("Not Implemented", "PDEInterface::BoundaryMatrix");
    }

  /********************************************/
  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::BoundaryMatrix_T(
        const FDC<DH, VECTOR, dealdim>& fdc,
        FullMatrix<double> &local_entry_matrix, double scale, double scale_ico)
    {
      FullMatrix<double> tmp_mat = local_entry_matrix;
      tmp_mat = 0.;

      BoundaryMatrix(fdc, tmp_mat, scale, scale_ico);
      unsigned int n_dofs_per_cell = fdc.GetNDoFsPerCell();

      for (unsigned int i = 0; i < n_dofs_per_cell; i++)
      {
        for (unsigned int j = 0; j < n_dofs_per_cell; j++)
        {
          local_entry_matrix(j, i) += tmp_mat(i, j);
        }
      }

    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    UpdateFlags
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::GetUpdateFlags() const
    {
      return update_default; //no update
    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    UpdateFlags
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::GetFaceUpdateFlags() const
    {
      return update_default; //no update
    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    bool
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::HasFaces() const
    {
      return false;
    }
  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    bool
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::HasInterfaces() const
    {
      return false;
    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    unsigned int
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::GetControlNBlocks() const
    {
      throw DOpEException("Not Implemented", "PDEInterface::GetControlNBlocks");
    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    unsigned int
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::GetStateNBlocks() const
    {
      throw DOpEException("Not Implemented", "PDEInterface::GetStateNBlocks");
    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    std::vector<unsigned int>&
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::GetControlBlockComponent()
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::GetControlBlockComponent");
    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    const std::vector<unsigned int>&
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::GetControlBlockComponent() const
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::GetControlBlockComponent");
    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    std::vector<unsigned int>&
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::GetStateBlockComponent()
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::GetStateBlockComponent");
    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    const std::vector<unsigned int>&
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::GetStateBlockComponent() const
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::GetStateBlockComponent");
    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    unsigned int
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::GetStateNComponents() const
    {
      return this->GetStateBlockComponent().size();
    }

  /********************************************/

  template<
      template<template<int, int> class DH, typename VECTOR, int dealdim> class CDC,
      template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
      template<int, int> class DH, typename VECTOR, int dealdim>
    void
    PDEInterface<CDC, FDC, DH, VECTOR, dealdim>::SetProblemType(
        std::string type)
    {
      _problem_type = type;
    }

/********************************************/

} //Endof namespace
/********************************************/
/********************************************/
template class DOpE::PDEInterface<DOpE::CellDataContainer,
    DOpE::FaceDataContainer, dealii::DoFHandler, dealii::BlockVector<double>,
    deal_II_dimension>;
template class DOpE::PDEInterface<DOpE::CellDataContainer,
    DOpE::FaceDataContainer, dealii::DoFHandler, dealii::Vector<double>,
    deal_II_dimension>;
template class DOpE::PDEInterface<DOpE::Multimesh_CellDataContainer,
    DOpE::Multimesh_FaceDataContainer, dealii::DoFHandler,
    dealii::BlockVector<double>, deal_II_dimension>;
template class DOpE::PDEInterface<DOpE::Multimesh_CellDataContainer,
    DOpE::Multimesh_FaceDataContainer, dealii::DoFHandler,
    dealii::Vector<double>, deal_II_dimension>;
template class DOpE::PDEInterface<DOpE::CellDataContainer,
    DOpE::FaceDataContainer, dealii::hp::DoFHandler,
    dealii::BlockVector<double>, deal_II_dimension>;
template class DOpE::PDEInterface<DOpE::CellDataContainer,
    DOpE::FaceDataContainer, dealii::hp::DoFHandler, dealii::Vector<double>,
    deal_II_dimension>;

