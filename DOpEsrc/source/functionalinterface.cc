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

#include <interfaces/functionalinterface.h>
#include <include/dopeexception.h>

#include <iostream>

using namespace dealii;

namespace DOpE
{

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
  FunctionalInterface<EDC, FDC, DH, VECTOR, dopedim, dealdim>::FunctionalInterface()
  {

  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
  FunctionalInterface<EDC, FDC, DH, VECTOR, dopedim, dealdim>::~FunctionalInterface()
  {

  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
  double
  FunctionalInterface<EDC, FDC, DH, VECTOR, dopedim, dealdim>::ElementValue(
    const EDC<DH, VECTOR, dealdim> & /*edc*/)
  {
    throw DOpEException("Not implemented", "FunctionalInterface::ElementValue");
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
  double
  FunctionalInterface<EDC, FDC, DH, VECTOR, dopedim, dealdim>::PointValue(
    const DOpEWrapper::DoFHandler<dopedim, DH> & /*control_dof_handler*/,
    const DOpEWrapper::DoFHandler<dealdim, DH> & /*state_dof_handler*/,
    const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
    const std::map<std::string, const VECTOR *> &/*domain_values*/)
  {
    throw DOpEException("Not implemented", "FunctionalInterface::PointValue");
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
  void
  FunctionalInterface<EDC, FDC, DH, VECTOR, dopedim, dealdim>::PointValue_U(
    const DOpEWrapper::DoFHandler<dopedim, DH> & /*control_dof_handler*/,
    const DOpEWrapper::DoFHandler<dealdim, DH> & /*state_dof_handler*/,
    const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
    const std::map<std::string, const VECTOR *> &/*domain_values*/,
    VECTOR & /*rhs*/, double /*scale*/)
  {
    throw DOpEException("Not implemented",
                        "FunctionalInterface::PointValue_U");
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
  void
  FunctionalInterface<EDC, FDC, DH, VECTOR, dopedim, dealdim>::PointValue_Q(
    const DOpEWrapper::DoFHandler<dopedim, DH> & /*control_dof_handler*/,
    const DOpEWrapper::DoFHandler<dealdim, DH> & /*state_dof_handler*/,
    const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
    const std::map<std::string, const VECTOR *> &/*domain_values*/,
    VECTOR & /*rhs*/, double /*scale*/)
  {
    throw DOpEException("Not implemented",
                        "FunctionalInterface::PointValue_Q");
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
  void
  FunctionalInterface<EDC, FDC, DH, VECTOR, dopedim, dealdim>::PointValue_UU(
    const DOpEWrapper::DoFHandler<dopedim, DH> & /*control_dof_handler*/,
    const DOpEWrapper::DoFHandler<dealdim, DH> & /*state_dof_handler*/,
    const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
    const std::map<std::string, const VECTOR *> &/*domain_values*/,
    VECTOR & /*rhs*/, double /*scale*/)
  {
    throw DOpEException("Not implemented",
                        "FunctionalInterface::PointValue_UU");
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
  void
  FunctionalInterface<EDC, FDC, DH, VECTOR, dopedim, dealdim>::PointValue_QU(
    const DOpEWrapper::DoFHandler<dopedim, DH> & /*control_dof_handler*/,
    const DOpEWrapper::DoFHandler<dealdim, DH> & /*state_dof_handler*/,
    const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
    const std::map<std::string, const VECTOR *> &/*domain_values*/,
    VECTOR & /*rhs*/, double /*scale*/)
  {
    throw DOpEException("Not implemented",
                        "FunctionalInterface::PointValue_QU");
  }
  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
  void
  FunctionalInterface<EDC, FDC, DH, VECTOR, dopedim, dealdim>::PointValue_UQ(
    const DOpEWrapper::DoFHandler<dopedim, DH> & /*control_dof_handler*/,
    const DOpEWrapper::DoFHandler<dealdim, DH> & /*state_dof_handler*/,
    const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
    const std::map<std::string, const VECTOR *> &/*domain_values*/,
    VECTOR & /*rhs*/, double /*scale*/)
  {
    throw DOpEException("Not implemented",
                        "FunctionalInterface::PointValue_UQ");
  }
  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
  void
  FunctionalInterface<EDC, FDC, DH, VECTOR, dopedim, dealdim>::PointValue_QQ(
    const DOpEWrapper::DoFHandler<dopedim, DH> & /*control_dof_handler*/,
    const DOpEWrapper::DoFHandler<dealdim, DH> & /*state_dof_handler*/,
    const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
    const std::map<std::string, const VECTOR *> &/*domain_values*/,
    VECTOR & /*rhs*/, double /*scale*/)
  {
    throw DOpEException("Not implemented",
                        "FunctionalInterface::PointValue_QQ");
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
  void
  FunctionalInterface<EDC, FDC, DH, VECTOR, dopedim, dealdim>::ElementValue_U(
    const EDC<DH, VECTOR, dealdim> & /*edc*/,
    dealii::Vector<double> &/*local_vector*/,
    double /*scale*/)
  {
    throw DOpEException("Not implemented", "FunctionalInterface::ElementValue_U");
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
  void
  FunctionalInterface<EDC, FDC, DH, VECTOR, dopedim, dealdim>::ElementValue_Q(
    const EDC<DH, VECTOR, dealdim> & /*edc*/,
    dealii::Vector<double> &/*local_vector*/,
    double /*scale*/)
  {
    throw DOpEException("Not implemented", "FunctionalInterface::ElementValue_Q");
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
  void
  FunctionalInterface<EDC, FDC, DH, VECTOR, dopedim, dealdim>::ElementValue_UU(
    const EDC<DH, VECTOR, dealdim> & /*edc*/,
    dealii::Vector<double> &/*local_vector*/,
    double /*scale*/)
  {
    throw DOpEException("Not implemented", "FunctionalInterface::ElementValue_UU");
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
  void
  FunctionalInterface<EDC, FDC, DH, VECTOR, dopedim, dealdim>::ElementValue_QU(
    const EDC<DH, VECTOR, dealdim> & /*edc*/,
    dealii::Vector<double> &/*local_vector*/,
    double /*scale*/)
  {
    throw DOpEException("Not implemented", "FunctionalInterface::ElementValue_QU");
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
  void
  FunctionalInterface<EDC, FDC, DH, VECTOR, dopedim, dealdim>::ElementValue_UQ(
    const EDC<DH, VECTOR, dealdim> & /*edc*/,
    dealii::Vector<double> &/*local_vector*/,
    double /*scale*/)
  {
    throw DOpEException("Not implemented", "FunctionalInterface::ElementValue_UQ");
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
  void
  FunctionalInterface<EDC, FDC, DH, VECTOR, dopedim, dealdim>::ElementValue_QQ(
    const EDC<DH, VECTOR, dealdim> & /*edc*/,
    dealii::Vector<double> &/*local_vector*/,
    double /*scale*/)
  {
    throw DOpEException("Not implemented", "FunctionalInterface::ElementValue_QQ");
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
  std::string
  FunctionalInterface<EDC, FDC, DH, VECTOR, dopedim, dealdim>::GetType() const
  {
    throw DOpEException("Not implemented", "FunctionalInterface::GetType");
  }
  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
  std::string
  FunctionalInterface<EDC, FDC, DH, VECTOR, dopedim, dealdim>::GetName() const
  {
    throw DOpEException("Not implemented", "FunctionalInterface::GetName");
  }

  /********************************************/
  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
  double
  FunctionalInterface<EDC, FDC, DH, VECTOR, dopedim, dealdim>::FaceValue(
    const FDC<DH, VECTOR, dealdim> & /*fdc*/)
  {
    throw DOpEException("Not Implemented", "FunctionalInterface::FaceValue");
  }

  /********************************************/
  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
  void
  FunctionalInterface<EDC, FDC, DH, VECTOR, dopedim, dealdim>::FaceValue_U(
    const FDC<DH, VECTOR, dealdim> & /*fdc*/,
    dealii::Vector<double> &/*local_vector*/,
    double /*scale*/)
  {
    throw DOpEException("Not Implemented",
                        "FunctionalInterface::FaceValue_U");
  }

  /********************************************/
  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
  void
  FunctionalInterface<EDC, FDC, DH, VECTOR, dopedim, dealdim>::FaceValue_Q(
    const FDC<DH, VECTOR, dealdim> & /*fdc*/,
    dealii::Vector<double> &/*local_vector*/,
    double /*scale*/)
  {
    throw DOpEException("Not Implemented",
                        "FunctionalInterface::FaceValue_Q");
  }

  /********************************************/
  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
  void
  FunctionalInterface<EDC, FDC, DH, VECTOR, dopedim, dealdim>::FaceValue_UU(
    const FDC<DH, VECTOR, dealdim> & /*fdc*/,
    dealii::Vector<double> &/*local_vector*/,
    double /*scale*/)
  {
    throw DOpEException("Not Implemented",
                        "FunctionalInterface::FaceValue_UU");
  }

  /********************************************/
  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
  void
  FunctionalInterface<EDC, FDC, DH, VECTOR, dopedim, dealdim>::FaceValue_QU(
    const FDC<DH, VECTOR, dealdim> & /*fdc*/,
    dealii::Vector<double> &/*local_vector*/,
    double /*scale*/)
  {
    throw DOpEException("Not Implemented",
                        "FunctionalInterface::FaceValue_QU");
  }

  /********************************************/
  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
  void
  FunctionalInterface<EDC, FDC, DH, VECTOR, dopedim, dealdim>::FaceValue_UQ(
    const FDC<DH, VECTOR, dealdim> & /*fdc*/,
    dealii::Vector<double> &/*local_vector*/,
    double /*scale*/)
  {
    throw DOpEException("Not Implemented",
                        "FunctionalInterface::FaceValue_UQ");
  }

  /********************************************/
  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
  void
  FunctionalInterface<EDC, FDC, DH, VECTOR, dopedim, dealdim>::FaceValue_QQ(
    const FDC<DH, VECTOR, dealdim> & /*fdc*/,
    dealii::Vector<double> &/*local_vector*/,
    double /*scale*/)
  {
    throw DOpEException("Not Implemented",
                        "FunctionalInterface::FaceValue_QQ");
  }

  /********************************************/
  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
  double
  FunctionalInterface<EDC, FDC, DH, VECTOR, dopedim, dealdim>::BoundaryValue(
    const FDC<DH, VECTOR, dealdim> & /*fdc*/)
  {
    throw DOpEException("Not Implemented",
                        "FunctionalInterface::BoundaryValue");
  }

  /********************************************/
  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
  void
  FunctionalInterface<EDC, FDC, DH, VECTOR, dopedim, dealdim>::BoundaryValue_U(
    const FDC<DH, VECTOR, dealdim> & /*fdc*/,
    dealii::Vector<double> &/*local_vector*/,
    double /*scale*/)
  {
    throw DOpEException("Not Implemented",
                        "FunctionalInterface::BoundaryValue_U");
  }

  /********************************************/
  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
  void
  FunctionalInterface<EDC, FDC, DH, VECTOR, dopedim, dealdim>::BoundaryValue_Q(
    const FDC<DH, VECTOR, dealdim> & /*fdc*/,
    dealii::Vector<double> &/*local_vector*/,
    double /*scale*/)
  {
    throw DOpEException("Not Implemented",
                        "FunctionalInterface::BoundaryValue_Q");
  }

  /********************************************/
  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
  void
  FunctionalInterface<EDC, FDC, DH, VECTOR, dopedim, dealdim>::BoundaryValue_UU(
    const FDC<DH, VECTOR, dealdim> & /*fdc*/,
    dealii::Vector<double> &/*local_vector*/,
    double /*scale*/)
  {
    throw DOpEException("Not Implemented",
                        "FunctionalInterface::BoundaryValue_UU");
  }

  /********************************************/
  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
  void
  FunctionalInterface<EDC, FDC, DH, VECTOR, dopedim, dealdim>::BoundaryValue_QU(
    const FDC<DH, VECTOR, dealdim> & /*fdc*/,
    dealii::Vector<double> &/*local_vector*/,
    double /*scale*/)
  {
    throw DOpEException("Not Implemented",
                        "FunctionalInterface::BoundaryValue_QU");
  }

  /********************************************/
  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
  void
  FunctionalInterface<EDC, FDC, DH, VECTOR, dopedim, dealdim>::BoundaryValue_UQ(
    const FDC<DH, VECTOR, dealdim> & /*fdc*/,
    dealii::Vector<double> &/*local_vector*/,
    double /*scale*/)
  {
    throw DOpEException("Not Implemented",
                        "FunctionalInterface::BoundaryValue_UQ");
  }

  /********************************************/
  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
  void
  FunctionalInterface<EDC, FDC, DH, VECTOR, dopedim, dealdim>::BoundaryValue_QQ(
    const FDC<DH, VECTOR, dealdim> & /*fdc*/,
    dealii::Vector<double> &/*local_vector*/,
    double /*scale*/)
  {
    throw DOpEException("Not Implemented",
                        "FunctionalInterface::BoundaryValue_QQ");
  }

  /********************************************/
  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
  double
  FunctionalInterface<EDC, FDC, DH, VECTOR, dopedim, dealdim>::AlgebraicValue(
    const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
    const std::map<std::string, const VECTOR *> &/*domain_values*/)
  {
    throw DOpEException("Not Implemented",
                        "FunctionalInterface::AlgebraicValue");
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
  UpdateFlags
  FunctionalInterface<EDC, FDC, DH, VECTOR, dopedim, dealdim>::GetUpdateFlags() const
  {
    return update_default; //no update
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
  UpdateFlags
  FunctionalInterface<EDC, FDC, DH, VECTOR, dopedim, dealdim>::GetFaceUpdateFlags() const
  {
    return update_default; //no update
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
  void
  FunctionalInterface<EDC, FDC, DH, VECTOR, dopedim, dealdim>::AlgebraicGradient_Q(
    VECTOR & /*gradient*/,
    const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
    const std::map<std::string, const VECTOR *> &/*domain_values*/)
  {
    throw DOpEException("Not Implemented",
                        "FunctionalInterface::AlgebraicGradient_Q");
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
  bool
  FunctionalInterface<EDC, FDC, DH, VECTOR, dopedim, dealdim>::HasFaces() const
  {
    bool retrn;
    //we check if the functional is of type face
    if (GetType().find("face") != std::string::npos)
      retrn = true;
    else
      retrn = false;

    return retrn;
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
  bool
  FunctionalInterface<EDC, FDC, DH, VECTOR, dopedim, dealdim>::HasPoints() const
  {
    bool retrn;
    //we check if the functional is of type point
    if (GetType().find("point") != std::string::npos)
      retrn = true;
    else
      retrn = false;

    return retrn;
  }
  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
  unsigned int
  FunctionalInterface<EDC, FDC, DH, VECTOR, dopedim, dealdim>::NeedPrecomputations() const
  {
    return 0;
  }

  /********************************************/

  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
  bool
  FunctionalInterface<EDC, FDC, DH, VECTOR, dopedim, dealdim>::NeedFinalValue() const
  {
    return false;
  }
  /********************************************/
  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
  void
  FunctionalInterface<EDC, FDC, DH, VECTOR, dopedim, dealdim>::SetProblemType(
    std::string p_type, unsigned int num)
  {
    problem_type_ = p_type;
    problem_num_ = num;
  }

  /********************************************/
  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
  std::string
  FunctionalInterface<EDC, FDC, DH, VECTOR, dopedim, dealdim>::GetProblemType() const
  {
    return problem_type_;
  }
  /********************************************/
  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dopedim, int dealdim>
  unsigned int
  FunctionalInterface<EDC, FDC, DH, VECTOR, dopedim, dealdim>::GetProblemNum() const
  {
    return problem_num_;
  }

} //Endof namespace
/********************************************/
/********************************************/
template class DOpE::FunctionalInterface<DOpE::ElementDataContainer,
         DOpE::FaceDataContainer, dealii::DoFHandler, dealii::Vector<double>,
         dope_dimension, deal_II_dimension>;
template class DOpE::FunctionalInterface<DOpE::ElementDataContainer,
         DOpE::FaceDataContainer, dealii::DoFHandler, dealii::BlockVector<double>,
         dope_dimension, deal_II_dimension>;
template class DOpE::FunctionalInterface<DOpE::Multimesh_ElementDataContainer,
         DOpE::Multimesh_FaceDataContainer, dealii::DoFHandler,
         dealii::Vector<double>, dope_dimension, deal_II_dimension>;
template class DOpE::FunctionalInterface<DOpE::Multimesh_ElementDataContainer,
         DOpE::Multimesh_FaceDataContainer, dealii::DoFHandler,
         dealii::BlockVector<double>, dope_dimension, deal_II_dimension>;
template class DOpE::FunctionalInterface<DOpE::ElementDataContainer,
         DOpE::FaceDataContainer, dealii::hp::DoFHandler, dealii::Vector<double>,
         dope_dimension, deal_II_dimension>;
template class DOpE::FunctionalInterface<DOpE::ElementDataContainer,
         DOpE::FaceDataContainer, dealii::hp::DoFHandler,
         dealii::BlockVector<double>, dope_dimension, deal_II_dimension>;
//template class DOpE::FunctionalInterface<DOpE::ElementDataContainer,
//    DOpE::FaceDataContainer, dealii::MGDoFHandler, dealii::Vector<double>,
//    dope_dimension, deal_II_dimension>;
//template class DOpE::FunctionalInterface<DOpE::ElementDataContainer,
//    DOpE::FaceDataContainer, dealii::MGDoFHandler,
//    dealii::BlockVector<double>, dope_dimension, deal_II_dimension>;
template class DOpE::FunctionalInterface<DOpE::Networks::Network_ElementDataContainer,
         DOpE::Networks::Network_FaceDataContainer, dealii::DoFHandler, dealii::Vector<double>,
         dope_dimension, deal_II_dimension>;
template class DOpE::FunctionalInterface<DOpE::Networks::Network_ElementDataContainer,
         DOpE::Networks::Network_FaceDataContainer, dealii::DoFHandler, dealii::BlockVector<double>,
         dope_dimension, deal_II_dimension>;
