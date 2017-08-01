/**
 *
 * Copyright (C) 2012-2014 by the DOpElib authors
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

#include <interfaces/pdeinterface.h>
#include <include/dopeexception.h>

#include <iostream>

//FIXME: For developement of MG-support, please uncomment.
//#include "../../Examples/Experimental/Example12/mgelementdatacontainer.h"

using namespace dealii;

namespace DOpE
{

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::PDEInterface()
  {

  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::~PDEInterface()
  {

  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::ElementEquation(
    const EDC<DH, VECTOR, dealdim> & /*edc*/,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
    throw DOpEException("Not Implemented", "PDEInterface::ElementEquation");
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::StrongElementResidual(
    const EDC<DH, VECTOR, dealdim> & /*edc*/,
    const EDC<DH, VECTOR, dealdim> & /*edc_weight*/, double &,
    double /*scale*/)
  {
    throw DOpEException("Not Implemented",
                        "PDEInterface::StrongElementResidual");
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::ElementTimeEquation(
    const EDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
    throw DOpEException("Not Implemented", "PDEInterface::ElementTimeEquation");
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::ElementTimeEquation_U(
    const EDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
    throw DOpEException("Not Implemented",
                        "PDEInterface::ElementTimeEquation_U");
  }
  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::ElementTimeEquation_UT(
    const EDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
    throw DOpEException("Not Implemented",
                        "PDEInterface::ElementTimeEquation_UT");
  }
  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::ElementTimeEquation_UTT(
    const EDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
    throw DOpEException("Not Implemented",
                        "PDEInterface::ElementTimeEquation_UTT");
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::ElementTimeEquationExplicit(
    const EDC<DH, VECTOR, dealdim> & /*edc*/,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
    //This should be left empty, then one can use the default case *Time* without the
    //need to implement ElementTimeEquationExplicit
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::ElementTimeEquationExplicit_U(
    const EDC<DH, VECTOR, dealdim> & /*edc*/,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
    //This should be left empty, then one can use the default case *Time* without the
    //need to implement ElementTimeEquationExplicit
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::ElementTimeEquationExplicit_UT(
    const EDC<DH, VECTOR, dealdim> & /*edc*/,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
    //This should be left empty, then one can use the default case *Time* without the
    //need to implement ElementTimeEquationExplicit
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::ElementTimeEquationExplicit_UTT(
    const EDC<DH, VECTOR, dealdim> & /*edc*/,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
    //This should be left empty, then one can use the default case *Time* without the
    //need to implement ElementTimeEquationExplicit
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::ElementTimeEquationExplicit_UU(
    const EDC<DH, VECTOR, dealdim> & /*edc*/,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
    //This should be left empty, then one can use the default case *Time* without the
    //need to implement ElementTimeEquationExplicit
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::ElementEquation_U(
    const EDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
    throw DOpEException("Not Implemented", "PDEInterface::ElementEquation_U");
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::StrongElementResidual_U(
    const EDC<DH, VECTOR, dealdim> &,
    const EDC<DH, VECTOR, dealdim> & /*edc_weight*/, double &,
    double /*scale*/)
  {
    throw DOpEException("Not Implemented",
                        "PDEInterface::StrongElementResidual_U");
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::ElementEquation_UT(
    const EDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
    throw DOpEException("Not Implemented", "PDEInterface::ElementEquation_UT");
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::ElementEquation_UTT(
    const EDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
    throw DOpEException("Not Implemented", "PDEInterface::ElementEquation_UTT");
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::ElementEquation_Q(
    const EDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
    throw DOpEException("Not Implemented", "PDEInterface::ElementEquation_Q");
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::ElementEquation_QT(
    const EDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
    throw DOpEException("Not Implemented", "PDEInterface::ElementEquation_QT");
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::ElementEquation_QTT(
    const EDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
    throw DOpEException("Not Implemented", "PDEInterface::ElementEquation_QTT");
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::ElementEquation_UU(
    const EDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
    throw DOpEException("Not Implemented", "PDEInterface::ElementEquation_UU");
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::ElementEquation_QU(
    const EDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
    throw DOpEException("Not Implemented", "PDEInterface::ElementEquation_QU");
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::ElementEquation_UQ(
    const EDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
    throw DOpEException("Not Implemented", "PDEInterface::ElementEquation_UQ");
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::ElementEquation_QQ(
    const EDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
    throw DOpEException("Not Implemented", "PDEInterface::ElementEquation_QQ");
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::ControlElementEquation(
    const EDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
    throw DOpEException("Not Implemented",
                        "PDEInterface::ControlElementEquation");
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::ControlBoundaryEquation(
    const FDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
    throw DOpEException("Not Implemented",
                        "PDEInterface::ControlBoundaryEquation");
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::ElementRightHandSide(
    const EDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
    throw DOpEException("Not Implemented", "PDEInterface::ElementRightHandSide");
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::ControlElementMatrix(
    const EDC<DH, VECTOR, dealdim> &,
    FullMatrix<double> &/*local_entry_matrix*/, double /*scale*/)
  {
    throw DOpEException("Not Implemented", "PDEInterface::ControlElementMatrix");
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::ControlBoundaryMatrix(
    const FDC<DH, VECTOR, dealdim> &,
    FullMatrix<double> &/*local_entry_matrix*/, double /*scale*/)
  {
    throw DOpEException("Not Implemented", "PDEInterface::ControlBoundaryMatrix");
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::StrongElementResidual_Control(
    const EDC<DH, VECTOR, dealdim> &, const EDC<DH, VECTOR, dealdim> &,
    double &, double)
  {
    throw DOpEException("Not Implemented",
                        "PDEInterface::StrongElementResidual_Control");
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::StrongFaceResidual_Control(
    const FDC<DH, VECTOR, dealdim> &, const FDC<DH, VECTOR, dealdim> &,
    double &, double)
  {
    throw DOpEException("Not Implemented",
                        "PDEInterface::StrongFaceResidual_Control");
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::StrongBoundaryResidual_Control(
    const FDC<DH, VECTOR, dealdim> &, const FDC<DH, VECTOR, dealdim> &,
    double &, double)
  {
    throw DOpEException("Not Implemented",
                        "PDEInterface::StrongBoundaryResidual_Control");
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::ElementMatrix(
    const EDC<DH, VECTOR, dealdim> &,
    FullMatrix<double> &/*local_entry_matrix*/, double/*scale*/,
    double /*scale_ico*/)
  {
    throw DOpEException("Not Implemented", "PDEInterface::ElementMatrix");
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::ElementTimeMatrix(
    const EDC<DH, VECTOR, dealdim> &,
    FullMatrix<double> &/*local_entry_matrix*/)
  {
    throw DOpEException("Not Implemented", "PDEInterface::ElementTimeMatrix");
  }
  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::ElementTimeMatrix_T(
    const EDC<DH, VECTOR, dealdim> &edc,
    FullMatrix<double> &local_entry_matrix)
  {
    FullMatrix<double> tmp_mat = local_entry_matrix;
    tmp_mat = 0.;

    //FIXME is this the right behaviour in the instationary case? or what
    //are the correct values for scale and scale_ico?
    ElementTimeMatrix(edc, tmp_mat);
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();

    for (unsigned int i = 0; i < n_dofs_per_element; i++)
      {
        for (unsigned int j = 0; j < n_dofs_per_element; j++)
          {
            local_entry_matrix(j, i) += tmp_mat(i, j);
          }
      }
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::ElementTimeMatrixExplicit(
    const EDC<DH, VECTOR, dealdim> & /*edc*/,
    FullMatrix<double> &/*local_entry_matrix*/)
  {
    //This should be left empty, then one can use the default case *Time* without the
    //need to implement ElementTimeEquationExplicit
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::ElementTimeMatrixExplicit_T(
    const EDC<DH, VECTOR, dealdim> &edc,
    FullMatrix<double> &local_entry_matrix)
  {
    FullMatrix<double> tmp_mat = local_entry_matrix;
    tmp_mat = 0.;

    //FIXME is this the right behaviour in the instationary case? or what
    //are the correct values for scale and scale_ico?
    ElementTimeMatrixExplicit(edc, tmp_mat);
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();

    for (unsigned int i = 0; i < n_dofs_per_element; i++)
      {
        for (unsigned int j = 0; j < n_dofs_per_element; j++)
          {
            local_entry_matrix(j, i) += tmp_mat(i, j);
          }
      }
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::ElementMatrix_T(
    const EDC<DH, VECTOR, dealdim> &edc,
    FullMatrix<double> &local_entry_matrix, double scale, double scale_ico)
  {
    FullMatrix<double> tmp_mat = local_entry_matrix;
    tmp_mat = 0.;

    //FIXME is this the right behaviour in the instationary case? or what
    //are the correct values for scale and scale_ico?
    ElementMatrix(edc, tmp_mat, scale, scale_ico);
    unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();

    for (unsigned int i = 0; i < n_dofs_per_element; i++)
      {
        for (unsigned int j = 0; j < n_dofs_per_element; j++)
          {
            local_entry_matrix(j, i) += tmp_mat(i, j);
          }
      }
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::FaceEquation(
    const FDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
    throw DOpEException("Not Implemented", "PDEInterface::FaceEquation");
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::StrongFaceResidual(
    const FDC<DH, VECTOR, dealdim> &,
    const FDC<DH, VECTOR, dealdim> & /*fdc_weight*/, double &,
    double /*scale*/)
  {
    throw DOpEException("Not Implemented",
                        "PDEInterface::StrongFaceResidual");
  }

  /********************************************/
  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::FaceEquation_U(
    const FDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
    throw DOpEException("Not Implemented", "PDEInterface::FaceEquation_U");
  }

  /********************************************/
  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::StrongFaceResidual_U(
    const FDC<DH, VECTOR, dealdim> &,
    const FDC<DH, VECTOR, dealdim> & /*fdc_weight*/, double &,
    double /*scale*/)
  {
    throw DOpEException("Not Implemented",
                        "PDEInterface::StrongFaceResidual_U");
  }

  /********************************************/
  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::FaceEquation_UT(
    const FDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
    throw DOpEException("Not Implemented", "PDEInterface::FaceEquation_UT");
  }

  /********************************************/
  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::FaceEquation_UTT(
    const FDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
    throw DOpEException("Not Implemented", "PDEInterface::FaceEquation_UTT");
  }

  /********************************************/
  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::FaceEquation_Q(
    const FDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
    throw DOpEException("Not Implemented", "PDEInterface::FaceEquation_Q");
  }

  /********************************************/
  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::FaceEquation_QT(
    const FDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
    throw DOpEException("Not Implemented", "PDEInterface::FaceEquation_QT");
  }

  /********************************************/
  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::FaceEquation_QTT(
    const FDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
    throw DOpEException("Not Implemented", "PDEInterface::FaceEquation_QTT");
  }

  /********************************************/
  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::FaceEquation_UU(
    const FDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
    throw DOpEException("Not Implemented", "PDEInterface::FaceEquation_UU");
  }

  /********************************************/
  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::FaceEquation_QU(
    const FDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
    throw DOpEException("Not Implemented", "PDEInterface::FaceEquation_QU");
  }

  /********************************************/
  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::FaceEquation_UQ(
    const FDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
    throw DOpEException("Not Implemented", "PDEInterface::FaceEquation_UQ");
  }

  /********************************************/
  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::FaceEquation_QQ(
    const FDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
    throw DOpEException("Not Implemented", "PDEInterface::FaceEquation_QQ");
  }

  /********************************************/
  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::FaceRightHandSide(
    const FDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
    throw DOpEException("Not Implemented", "PDEInterface::FaceRightHandSide");
  }

  /********************************************/
  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::FaceMatrix(
    const FDC<DH, VECTOR, dealdim> &,
    FullMatrix<double> &/*local_entry_matrix*/, double /*scale*/,
    double /*scale_ico*/)
  {
    throw DOpEException("Not Implemented", "PDEInterface::FaceMatrix");
  }

  /********************************************/
  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::FaceMatrix_T(
    const FDC<DH, VECTOR, dealdim> &fdc,
    FullMatrix<double> &local_entry_matrix, double scale, double scale_ico)
  {
    FullMatrix<double> tmp_mat = local_entry_matrix;
    tmp_mat = 0.;

    FaceMatrix(fdc, tmp_mat, scale, scale_ico);
    unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();

    for (unsigned int i = 0; i < n_dofs_per_element; i++)
      {
        for (unsigned int j = 0; j < n_dofs_per_element; j++)
          {
            local_entry_matrix(j, i) += tmp_mat(i, j);
          }
      }
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::InterfaceEquation(
    const FDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
    throw DOpEException("Not Implemented", "PDEInterface::InterfaceEquation");
  }
  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::InterfaceEquation_U(
    const FDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
    throw DOpEException("Not Implemented",
                        "PDEInterface::InterfaceEquation_U");
  }
  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::InterfaceEquation_UT(
    const FDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
    throw DOpEException("Not Implemented",
                        "PDEInterface::InterfaceEquation_UT");
  }
  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::InterfaceEquation_UTT(
    const FDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
    throw DOpEException("Not Implemented",
                        "PDEInterface::InterfaceEquation_UTT");
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::InterfaceMatrix(
    const FDC<DH, VECTOR, dealdim> &,
    FullMatrix<double> &/*local_entry_matrix*/, double /*scale*/,
    double /*scale_ico*/)
  {
    throw DOpEException("Not Implemented", "PDEInterface::InterfaceMatrix");
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::InterfaceMatrix_T(
    const FDC<DH, VECTOR, dealdim> &,
    FullMatrix<double> &/*local_entry_matrix*/, double /*scale*/,
    double /*scale_ico*/)
  {
    throw DOpEException("Not Implemented", "PDEInterface::InterfaceMatrix_T");
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::BoundaryEquation(
    const FDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
    throw DOpEException("Not Implemented", "PDEInterface::BoundaryEquation");
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::StrongBoundaryResidual(
    const FDC<DH, VECTOR, dealdim> &,
    const FDC<DH, VECTOR, dealdim> & /*Fdc_weight*/, double &,
    double /*scale*/)
  {
    throw DOpEException("Not Implemented",
                        "PDEInterface::StrongBoundaryResidual");
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::BoundaryEquation_U(
    const FDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
    throw DOpEException("Not Implemented",
                        "PDEInterface::BoundaryEquation_U");
  }

  /********************************************/
  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::StrongBoundaryResidual_U(
    const FDC<DH, VECTOR, dealdim> &,
    const FDC<DH, VECTOR, dealdim> & /*fdc_weight*/, double &,
    double /*scale*/)
  {
    throw DOpEException("Not Implemented",
                        "PDEInterface::StrongBoundaryResidual_U");
  }

  /********************************************/
  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::BoundaryEquation_UT(
    const FDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
    throw DOpEException("Not Implemented",
                        "PDEInterface::BoundaryEquation_UT");
  }

  /********************************************/
  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::BoundaryEquation_UTT(
    const FDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
    throw DOpEException("Not Implemented",
                        "PDEInterface::BoundaryEquation_UTT");
  }

  /********************************************/
  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::BoundaryEquation_Q(
    const FDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
    throw DOpEException("Not Implemented",
                        "PDEInterface::BoundaryEquation_Q");
  }

  /********************************************/
  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::BoundaryEquation_QT(
    const FDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
    throw DOpEException("Not Implemented",
                        "PDEInterface::BoundaryEquation_QT");
  }

  /********************************************/
  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::BoundaryEquation_QTT(
    const FDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
    throw DOpEException("Not Implemented",
                        "PDEInterface::BoundaryEquation_QTT");
  }

  /********************************************/
  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::BoundaryEquation_UU(
    const FDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
    throw DOpEException("Not Implemented",
                        "PDEInterface::BoundaryEquation_UU");
  }

  /********************************************/
  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::BoundaryEquation_QU(
    const FDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
    throw DOpEException("Not Implemented",
                        "PDEInterface::BoundaryEquation_QU");
  }

  /********************************************/
  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::BoundaryEquation_UQ(
    const FDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
    throw DOpEException("Not Implemented",
                        "PDEInterface::BoundaryEquation_UQ");
  }

  /********************************************/
  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::BoundaryEquation_QQ(
    const FDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/,
    double /*scale_ico*/)
  {
    throw DOpEException("Not Implemented",
                        "PDEInterface::BoundaryEquation_QQ");
  }

  /********************************************/
  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::BoundaryRightHandSide(
    const FDC<DH, VECTOR, dealdim> &,
    dealii::Vector<double> &/*local_vector*/, double /*scale*/)
  {
    throw DOpEException("Not Implemented",
                        "PDEInterface::BoundaryRightHandSide");
  }

  /********************************************/
  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::BoundaryMatrix(
    const FDC<DH, VECTOR, dealdim> &,
    FullMatrix<double> &/*local_entry_matrix*/, double /*scale*/,
    double /*scale_ico*/)
  {
    throw DOpEException("Not Implemented", "PDEInterface::BoundaryMatrix");
  }

  /********************************************/
  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::BoundaryMatrix_T(
    const FDC<DH, VECTOR, dealdim> &fdc,
    FullMatrix<double> &local_entry_matrix, double scale, double scale_ico)
  {
    FullMatrix<double> tmp_mat = local_entry_matrix;
    tmp_mat = 0.;

    BoundaryMatrix(fdc, tmp_mat, scale, scale_ico);
    unsigned int n_dofs_per_element = fdc.GetNDoFsPerElement();

    for (unsigned int i = 0; i < n_dofs_per_element; i++)
      {
        for (unsigned int j = 0; j < n_dofs_per_element; j++)
          {
            local_entry_matrix(j, i) += tmp_mat(i, j);
          }
      }

  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  UpdateFlags
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::GetUpdateFlags() const
  {
    return update_default; //no update
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  UpdateFlags
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::GetFaceUpdateFlags() const
  {
    return update_default; //no update
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  bool
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::HasFaces() const
  {
    return false;
  }
  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  bool
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::HasInterfaces() const
  {
    return false;
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  unsigned int
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::GetControlNBlocks() const
  {
    throw DOpEException("Not Implemented", "PDEInterface::GetControlNBlocks");
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  unsigned int
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::GetStateNBlocks() const
  {
    throw DOpEException("Not Implemented", "PDEInterface::GetStateNBlocks");
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  std::vector<unsigned int> &
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::GetControlBlockComponent()
  {
    throw DOpEException("Not Implemented",
                        "PDEInterface::GetControlBlockComponent");
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  const std::vector<unsigned int> &
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::GetControlBlockComponent() const
  {
    throw DOpEException("Not Implemented",
                        "PDEInterface::GetControlBlockComponent");
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  std::vector<unsigned int> &
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::GetStateBlockComponent()
  {
    throw DOpEException("Not Implemented",
                        "PDEInterface::GetStateBlockComponent");
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  const std::vector<unsigned int> &
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::GetStateBlockComponent() const
  {
    throw DOpEException("Not Implemented",
                        "PDEInterface::GetStateBlockComponent");
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  unsigned int
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::GetStateNComponents() const
  {
    return this->GetStateBlockComponent().size();
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dealdim>
  void
  PDEInterface<EDC, FDC, DH, VECTOR, dealdim>::SetProblemType(
    std::string type)
  {
    problem_type_ = type;
  }

  /********************************************/

} //Endof namespace
/********************************************/
/********************************************/
template class DOpE::PDEInterface<DOpE::ElementDataContainer,
         DOpE::FaceDataContainer, dealii::DoFHandler, dealii::BlockVector<double>,
         deal_II_dimension>;
template class DOpE::PDEInterface<DOpE::ElementDataContainer,
         DOpE::FaceDataContainer, dealii::DoFHandler, dealii::Vector<double>,
         deal_II_dimension>;
template class DOpE::PDEInterface<DOpE::Multimesh_ElementDataContainer,
         DOpE::Multimesh_FaceDataContainer, dealii::DoFHandler,
         dealii::BlockVector<double>, deal_II_dimension>;
template class DOpE::PDEInterface<DOpE::Multimesh_ElementDataContainer,
         DOpE::Multimesh_FaceDataContainer, dealii::DoFHandler,
         dealii::Vector<double>, deal_II_dimension>;
template class DOpE::PDEInterface<DOpE::ElementDataContainer,
         DOpE::FaceDataContainer, dealii::hp::DoFHandler,
         dealii::BlockVector<double>, deal_II_dimension>;
template class DOpE::PDEInterface<DOpE::ElementDataContainer,
         DOpE::FaceDataContainer, dealii::hp::DoFHandler, dealii::Vector<double>,
         deal_II_dimension>;
//FIXME: For developement of MG-support, please uncomment.
//template class DOpE::PDEInterface<DOpE::ElementDataContainer,
//    DOpE::FaceDataContainer, dealii::MGDoFHandler,
//    dealii::BlockVector<double>, deal_II_dimension>;
//template class DOpE::PDEInterface<DOpE::ElementDataContainer,
//    DOpE::FaceDataContainer, dealii::MGDoFHandler,
//    dealii::Vector<double>, deal_II_dimension>;
//template class DOpE::PDEInterface<DOpE::MGElementDataContainer,
//    DOpE::FaceDataContainer, dealii::MGDoFHandler,
//    dealii::BlockVector<double>, deal_II_dimension>;
//template class DOpE::PDEInterface<DOpE::MGElementDataContainer,
//    DOpE::FaceDataContainer, dealii::MGDoFHandler,
//    dealii::Vector<double>, deal_II_dimension>;
template class DOpE::PDEInterface<DOpE::Networks::Network_ElementDataContainer,
         DOpE::Networks::Network_FaceDataContainer, dealii::DoFHandler, dealii::BlockVector<double>,
         deal_II_dimension>;
template class DOpE::PDEInterface<DOpE::Networks::Network_ElementDataContainer,
         DOpE::Networks::Network_FaceDataContainer, dealii::DoFHandler, dealii::Vector<double>,
         deal_II_dimension>;

