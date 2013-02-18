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


#include "functionalinterface.h"
#include "dopeexception.h"

#include <iostream>

using namespace dealii;

namespace DOpE
{

  /********************************************/

  template<template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC, template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC, typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::FunctionalInterface()
    {

    }

  /********************************************/

  template<template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC, template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC, typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::~FunctionalInterface()
    {

    }

  /********************************************/

  template<template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC, template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC, typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    double
    FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::Value(
        const CDC<DOFHANDLER, VECTOR, dealdim>&)
    {
      throw DOpEException("Not implemented",
          "FunctionalInterface::Value");
    }

  /********************************************/

  template<template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC, template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC, typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    double
    FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::PointValue(
        const DOpEWrapper::DoFHandler<dopedim, DOFHANDLER> & control_dof_handler __attribute__((unused)),
        const DOpEWrapper::DoFHandler<dealdim, DOFHANDLER> & state_dof_handler __attribute__((unused)),
        const std::map<std::string, const dealii::Vector<double>*> &param_values __attribute__((unused)),
        const std::map<std::string, const VECTOR*> &domain_values __attribute__((unused)))
    {
      throw DOpEException("Not implemented",
          "FunctionalInterface::PointValue");
    }

  /********************************************/

  template<template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC, template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC, typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::PointValue_U(
        const DOpEWrapper::DoFHandler<dopedim, DOFHANDLER> & control_dof_handler __attribute__((unused)),
        const DOpEWrapper::DoFHandler<dealdim, DOFHANDLER> & state_dof_handler __attribute__((unused)),
        const std::map<std::string, const dealii::Vector<double>*> &param_values __attribute__((unused)),
        const std::map<std::string, const VECTOR*> &domain_values __attribute__((unused)),
        VECTOR& /*rhs*/,
        double /*scale*/)
    {
      throw DOpEException("Not implemented",
          "FunctionalInterface::PointValue_U");
    }

  /********************************************/

  template<template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC, template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC, typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::PointValue_Q(
        const DOpEWrapper::DoFHandler<dopedim, DOFHANDLER> & control_dof_handler __attribute__((unused)),
        const DOpEWrapper::DoFHandler<dealdim, DOFHANDLER> & state_dof_handler __attribute__((unused)),
        const std::map<std::string, const dealii::Vector<double>*> &param_values __attribute__((unused)),
        const std::map<std::string, const VECTOR*> &domain_values __attribute__((unused)),
        VECTOR& /*rhs*/,
        double /*scale*/)
    {
      throw DOpEException("Not implemented",
          "FunctionalInterface::PointValue_Q");
    }

  /********************************************/

  template<template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC, template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC, typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::PointValue_UU(
        const DOpEWrapper::DoFHandler<dopedim, DOFHANDLER> & control_dof_handler __attribute__((unused)),
        const DOpEWrapper::DoFHandler<dealdim, DOFHANDLER> & state_dof_handler __attribute__((unused)),
        const std::map<std::string, const dealii::Vector<double>*> &param_values __attribute__((unused)),
        const std::map<std::string, const VECTOR*> &domain_values __attribute__((unused)),
        VECTOR& /*rhs*/,
        double /*scale*/)
    {
      throw DOpEException("Not implemented",
          "FunctionalInterface::PointValue_UU");
    }

  /********************************************/

  template<template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC, template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC, typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::PointValue_QU(
        const DOpEWrapper::DoFHandler<dopedim, DOFHANDLER> & control_dof_handler __attribute__((unused)),
        const DOpEWrapper::DoFHandler<dealdim, DOFHANDLER> & state_dof_handler __attribute__((unused)),
        const std::map<std::string, const dealii::Vector<double>*> &param_values __attribute__((unused)),
        const std::map<std::string, const VECTOR*> &domain_values __attribute__((unused)),
        VECTOR& /*rhs*/,
        double /*scale*/)
    {
      throw DOpEException("Not implemented",
          "FunctionalInterface::PointValue_QU");
    }
  /********************************************/

  template<template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC, template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC, typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::PointValue_UQ(
        const DOpEWrapper::DoFHandler<dopedim, DOFHANDLER> & control_dof_handler __attribute__((unused)),
        const DOpEWrapper::DoFHandler<dealdim, DOFHANDLER> & state_dof_handler __attribute__((unused)),
        const std::map<std::string, const dealii::Vector<double>*> &param_values __attribute__((unused)),
        const std::map<std::string, const VECTOR*> &domain_values __attribute__((unused)),
        VECTOR& /*rhs*/,
        double /*scale*/)
    {
      throw DOpEException("Not implemented",
          "FunctionalInterface::PointValue_UQ");
    }
  /********************************************/

  template<template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC, template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC, typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::PointValue_QQ(
        const DOpEWrapper::DoFHandler<dopedim, DOFHANDLER> & control_dof_handler __attribute__((unused)),
        const DOpEWrapper::DoFHandler<dealdim, DOFHANDLER> & state_dof_handler __attribute__((unused)),
        const std::map<std::string, const dealii::Vector<double>*> &param_values __attribute__((unused)),
        const std::map<std::string, const VECTOR*> &domain_values __attribute__((unused)),
        VECTOR& /*rhs*/,
        double /*scale*/)
    {
      throw DOpEException("Not implemented",
          "FunctionalInterface::PointValue_QQ");
    }

  /********************************************/

  template<template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC, template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC, typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::Value_U(
        const CDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &local_cell_vector __attribute__((unused)), double scale __attribute__((unused)))
    {
      throw DOpEException("Not implemented",
          "FunctionalInterface::Value_U");
    }

  /********************************************/

  template<template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC, template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC, typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::Value_Q(
        const CDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &local_cell_vector __attribute__((unused)), double scale __attribute__((unused)))
    {
      throw DOpEException("Not implemented",
          "FunctionalInterface::Value_Q");
    }

  /********************************************/

  template<template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC, template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC, typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::Value_UU(
        const CDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &local_cell_vector __attribute__((unused)), double scale __attribute__((unused)))
    {
      throw DOpEException("Not implemented",
          "FunctionalInterface::Value_UU");
    }

  /********************************************/

  template<template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC, template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC, typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::Value_QU(
        const CDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &local_cell_vector __attribute__((unused)), double scale __attribute__((unused)))
    {
      throw DOpEException("Not implemented",
          "FunctionalInterface::Value_QU");
    }

  /********************************************/

  template<template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC, template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC, typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::Value_UQ(
        const CDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &local_cell_vector __attribute__((unused)), double scale __attribute__((unused)))
    {
      throw DOpEException("Not implemented",
          "FunctionalInterface::Value_UQ");
    }

  /********************************************/

  template<template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC, template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC, typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::Value_QQ(
        const CDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &local_cell_vector __attribute__((unused)), double scale __attribute__((unused)))
    {
      throw DOpEException("Not implemented",
          "FunctionalInterface::Value_QQ");
    }

  /********************************************/

  template<template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC, template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC, typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    std::string
    FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::GetType() const
    {
      throw DOpEException("Not implemented",
          "FunctionalInterface::GetType");
    }
  /********************************************/

  template<template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC, template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC, typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    std::string
    FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::GetName() const
    {
      throw DOpEException("Not implemented",
          "FunctionalInterface::GetName");
    }

  /********************************************/
  template<template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC, template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC, typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    double
    FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::FaceValue(
        const FDC<DOFHANDLER, VECTOR, dealdim>&)
    {
      throw DOpEException("Not Implemented", "FunctionalInterface::FaceValue");
    }

  /********************************************/
  template<template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC, template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC, typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::FaceValue_U(
        const FDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &local_cell_vector __attribute__((unused)), double scale __attribute__((unused)))
    {
      throw DOpEException("Not Implemented", "FunctionalInterface::FaceValue_U");
    }

  /********************************************/
  template<template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC, template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC, typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::FaceValue_Q(
        const FDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &local_cell_vector __attribute__((unused)), double scale __attribute__((unused)))
    {
      throw DOpEException("Not Implemented", "FunctionalInterface::FaceValue_Q");
    }

  /********************************************/
  template<template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC, template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC, typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::FaceValue_UU(
        const FDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &local_cell_vector __attribute__((unused)), double scale __attribute__((unused)))
    {
      throw DOpEException("Not Implemented",
          "FunctionalInterface::FaceValue_UU");
    }

  /********************************************/
  template<template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC, template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC, typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::FaceValue_QU(
        const FDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &local_cell_vector __attribute__((unused)), double scale __attribute__((unused)))
    {
      throw DOpEException("Not Implemented",
          "FunctionalInterface::FaceValue_QU");
    }

  /********************************************/
  template<template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC, template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC, typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::FaceValue_UQ(
        const FDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &local_cell_vector __attribute__((unused)), double scale __attribute__((unused)))
    {
      throw DOpEException("Not Implemented",
          "FunctionalInterface::FaceValue_UQ");
    }

  /********************************************/
  template<template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC, template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC, typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::FaceValue_QQ(
        const FDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &local_cell_vector __attribute__((unused)), double scale __attribute__((unused)))
    {
      throw DOpEException("Not Implemented",
          "FunctionalInterface::FaceValue_QQ");
    }

  /********************************************/
  template<template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC, template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC, typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    double
    FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::BoundaryValue(
        const FDC<DOFHANDLER, VECTOR, dealdim>&)
    {
      throw DOpEException("Not Implemented",
          "FunctionalInterface::BoundaryValue");
    }

  /********************************************/
  template<template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC, template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC, typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::BoundaryValue_U(
        const FDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &local_cell_vector __attribute__((unused)), double scale __attribute__((unused)))
    {
      throw DOpEException("Not Implemented",
          "FunctionalInterface::BoundaryValue_U");
    }

  /********************************************/
  template<template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC, template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC, typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::BoundaryValue_Q(
        const FDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &local_cell_vector __attribute__((unused)), double scale __attribute__((unused)))
    {
      throw DOpEException("Not Implemented",
          "FunctionalInterface::BoundaryValue_Q");
    }

  /********************************************/
  template<template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC, template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC, typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::BoundaryValue_UU(
        const FDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &local_cell_vector __attribute__((unused)), double scale __attribute__((unused)))
    {
      throw DOpEException("Not Implemented",
          "FunctionalInterface::BoundaryValue_UU");
    }

  /********************************************/
  template<template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC, template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC, typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::BoundaryValue_QU(
        const FDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &local_cell_vector __attribute__((unused)), double scale __attribute__((unused)))
    {
      throw DOpEException("Not Implemented",
          "FunctionalInterface::BoundaryValue_QU");
    }

  /********************************************/
  template<template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC, template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC, typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::BoundaryValue_UQ(
        const FDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &local_cell_vector __attribute__((unused)), double scale __attribute__((unused)))
    {
      throw DOpEException("Not Implemented",
          "FunctionalInterface::BoundaryValue_UQ");
    }

  /********************************************/
  template<template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC, template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC, typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::BoundaryValue_QQ(
        const FDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &local_cell_vector __attribute__((unused)), double scale __attribute__((unused)))
    {
      throw DOpEException("Not Implemented",
          "FunctionalInterface::BoundaryValue_QQ");
    }

  /********************************************/
  template<template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC, template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC, typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    double
    FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::AlgebraicValue(
        const std::map<std::string, const dealii::Vector<double>*> &param_values __attribute__((unused)),
        const std::map<std::string, const VECTOR*> &domain_values __attribute__((unused)))
    {
      throw DOpEException("Not Implemented",
          "FunctionalInterface::AlgebraicValue");
    }

  /********************************************/

  template<template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC, template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC, typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    UpdateFlags
    FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::GetUpdateFlags() const
    {
      return update_default; //no update
    }

  /********************************************/

  template<template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC, template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC, typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    UpdateFlags
    FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::GetFaceUpdateFlags() const
    {
      return update_default; //no update
    }

   /********************************************/

  template<template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC, template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC, typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
  void FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::AlgebraicGradient_Q(VECTOR& gradient __attribute__((unused)),
									 const std::map<std::string, const dealii::Vector<double>* > &param_values __attribute__((unused)),
									 const std::map<std::string, const VECTOR* > &domain_values __attribute__((unused)))
  {
    throw DOpEException("Not Implemented","FunctionalInterface::AlgebraicGradient_Q");
  }

  /********************************************/

  template<template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC, template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC, typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    bool
    FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::HasFaces() const
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

  template<template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC, template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC, typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    bool
    FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::HasPoints() const
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
  template<template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC
      , template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC
      , typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::SetProblemType(
        std::string p_type)
    {
      _problem_type = p_type;
    }

  /********************************************/
  template<template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC
      , template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC
      , typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    std::string
    FunctionalInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::GetProblemType() const
    {
      return _problem_type;
    }

}//Endof namespace
/********************************************/
/********************************************/
template class DOpE::FunctionalInterface<DOpE::CellDataContainer, 
					 DOpE::FaceDataContainer,
					 dealii::DoFHandler<deal_II_dimension>,
					 dealii::Vector<double>, dope_dimension, deal_II_dimension>;
template class DOpE::FunctionalInterface<DOpE::CellDataContainer, 
					 DOpE::FaceDataContainer,
					 dealii::DoFHandler<deal_II_dimension>,
					 dealii::BlockVector<double>, dope_dimension, deal_II_dimension>;
template class DOpE::FunctionalInterface<DOpE::Multimesh_CellDataContainer, 
					 DOpE::Multimesh_FaceDataContainer,
					 dealii::DoFHandler<deal_II_dimension>,
					 dealii::Vector<double>, dope_dimension, deal_II_dimension>;
template class DOpE::FunctionalInterface<DOpE::Multimesh_CellDataContainer, 
					 DOpE::Multimesh_FaceDataContainer,
					 dealii::DoFHandler<deal_II_dimension>,
					 dealii::BlockVector<double>, dope_dimension, deal_II_dimension>;
template class DOpE::FunctionalInterface<DOpE::CellDataContainer, 
					 DOpE::FaceDataContainer,
					 dealii::hp::DoFHandler<deal_II_dimension>,
					 dealii::Vector<double>, dope_dimension, deal_II_dimension>;
template class DOpE::FunctionalInterface<DOpE::CellDataContainer, 
					 DOpE::FaceDataContainer,
					 dealii::hp::DoFHandler<deal_II_dimension>,
					 dealii::BlockVector<double>, dope_dimension, deal_II_dimension>;
