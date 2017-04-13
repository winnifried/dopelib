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

#ifndef CONSTRAINTMAKER_H_
#define CONSTRAINTMAKER_H_

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/constraint_matrix.h>

#include <wrapper/dofhandler_wrapper.h>
#include <wrapper/mapping_wrapper.h>
#include <include/parameterreader.h>
#include <include/dopeexception.h>

namespace DOpE
{
  /**
   * This class is an interface which offers the user the possibility
   * to define some DoFConstraints for the state and/or control fe function.
   * To define non-standard constraints, one has to implement these in a
   * derived class of this one, and give then an instantiation
   * to the SpaceTimeHandler (via SetUserDefinedDoFConstraints).
   *
   * The constraints defined by MakeStateDoFConstrains and MakeControlDoFConstraints
   * are computed AFTER hanging_node_constraint is called, so if there are two
   * or more conflicting constraints on a DoF, the constraints coming from
   * hanging nodes win.
   *
   * FIXME: Just homogeneous dof constraints at the moment.
   * If we change distribution from global to local, this should
   * get changed.
   */
  template<template<int, int> class DH, int dopedim, int dealdim = dopedim>
  class UserDefinedDoFConstraints
  {
  public:
    UserDefinedDoFConstraints()
    {
    }
    virtual
    ~UserDefinedDoFConstraints()
    {
    }
    virtual void
    MakeStateDoFConstraints(
      const DOpEWrapper::DoFHandler<dealdim, DH> &dof_handler,
      dealii::ConstraintMatrix &dof_constraints) const;

    virtual void
    MakeControlDoFConstraints(
      const DOpEWrapper::DoFHandler<dopedim, DH> &dof_handler,
      dealii::ConstraintMatrix &dof_constraints) const;

    void
    RegisterMapping(const typename DOpEWrapper::Mapping<dealdim, DH> &mapping)
    {
      mapping_ = &mapping;
    }

  protected:
    const DOpEWrapper::Mapping<dealdim, DH> &
    GetMapping() const
    {
      return *mapping_;
    }
  private:
    const DOpEWrapper::Mapping<dealdim, DH> *mapping_;
  };

  template<template<int, int> class DH, int dopedim, int dealdim>
  void
  UserDefinedDoFConstraints<DH, dopedim, dealdim>::MakeStateDoFConstraints(
    const DOpEWrapper::DoFHandler<dealdim, DH> & /*dof_handler*/,
    dealii::ConstraintMatrix & /*dof_constraints*/) const
  {
    throw DOpEException("Not Implemented.",
                        "UserDefinedDoFConstraints::MakeStateDoFConstraints");
  }

  template<template<int, int> class DH, int dopedim, int dealdim>
  void
  UserDefinedDoFConstraints<DH, dopedim, dealdim>::MakeControlDoFConstraints(
    const DOpEWrapper::DoFHandler<dopedim, DH> & /*dof_handler*/,
    dealii::ConstraintMatrix & /*dof_constraints*/) const
  {
    throw DOpEException("Not Implemented.",
                        "UserDefinedDoFConstraints::MakeControlDoFConstraints");
  }

} //end of namespace
#endif /* CONSTRAINTMAKER_H_ */
