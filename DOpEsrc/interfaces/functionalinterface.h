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

#ifndef FUNCTIONAL_INTERFACE_H_
#define FUNCTIONAL_INTERFACE_H_

#include <map>
#include <string>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>

#include <wrapper/fevalues_wrapper.h>
#include <wrapper/dofhandler_wrapper.h>
#include <container/elementdatacontainer.h>
#include <container/facedatacontainer.h>
#include <container/multimesh_elementdatacontainer.h>
#include <container/multimesh_facedatacontainer.h>
#include <network/network_elementdatacontainer.h>
#include <network/network_facedatacontainer.h>

namespace DOpE
{
  /**
   * A template for an arbitrary Functional J to be used as cost functional for an optimization problem.
   * Or any Functional that should be evaluated. For evaluation only *Value routines are required, but none  of
   * the derivatives thereof.
   */
  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR, int dopedim, int dealdim =
           dopedim>
  class FunctionalInterface
  {
  public:
    FunctionalInterface();
    virtual
    ~FunctionalInterface();

    /**
     * This evaluates the Cost Functional J(q,u) = \int_\Omega j(q(x),u(x)) \dx on a given element T.
     *
     * @param edc     The ElementDataContainer containing all the data necessary to evaluate the functional on
     *                a element.
     */
    virtual double
    ElementValue(const EDC<DH, VECTOR, dealdim> &edc);

    /**
     * This evaluates the Cost Functional J_u'(q,u)(.) = \int_\Omega j_u'(q(x),u(x))(.) \dx on a given element T.
     *
     * @param edc                      The ElementDataContainer containing all the data necessary to evaluate the functional on
     *                                 a element.
     * @param local_vector        A Vector to contain the result. After completion local_vector fulfills
     *                                 local_vector(i) += scale * \int_T j_u'(q(x),u(x))(\phi_i) \dx where \phi_i is
     *                                 the i-th local basis function of the state space.
     * @param scale                    A factor by which the result is scaled.
     */
    virtual void
    ElementValue_U(const EDC<DH, VECTOR, dealdim> &edc,
                   dealii::Vector<double> &local_vector, double scale);

    /**
     * This evaluates the Cost Functional J_q'(q,u)(.) = \int_\Omega j_q'(q(x),u(x))(.) \dx on a given element T.
     *
     * @param edc                      The ElementDataContainer containing all the data necessary to evaluate the functional on
     *                                 a element.
     * @param local_vector        A Vector to contain the result. After completion local_vector fullfils
     *                                 local_vector(i) += scale * \int_T j_q'(q(x),u(x))(\phi_i) \dx where \phi_i is
     *                                 the i-th local basis function of the control space.
     * @param scale                    A factor by which the result is scaled.
     */
    virtual void
    ElementValue_Q(const EDC<DH, VECTOR, dealdim> &edc,
                   dealii::Vector<double> &local_vector, double scale);

    /**
     * This evaluates the Cost Functional J_uu'(q,u)(.,DU) = \int_\Omega j_uu'(q(x),u(x))(.,DU) \dx on a given element T.
     *
     * @param edc                      The ElementDataContainer containing all the data necessary to evaluate the functional on
     *                                 a element.
     * @param local_vector        A Vector to contain the result. After completion local_vector fullfils
     *                                 local_vector(i) += scale * \int_T j_uu'(q(x),u(x))(\phi_i,DU) \dx where \phi_i is
     *                                 the i-th local basis function of the state space.
     * @param scale                    A factor by which the result is scaled.
     */
    virtual void
    ElementValue_UU(const EDC<DH, VECTOR, dealdim> &edc,
                    dealii::Vector<double> &local_vector, double scale);

    /**
     * This evaluates the Cost Functional J_qu'(q,u)(.,DQ) = \int_\Omega j_qu'(q(x),u(x))(.,DQ) \dx on a given element T.
     *
     * @param edc                      The ElementDataContainer containing all the data necessary to evaluate the functional on
     *                                 a element.
     * @param local_vector        A Vector to contain the result. After completion local_vector fullfils
     *                                 local_vector(i) += scale * \int_T j_qu'(q(x),u(x))(\phi_i,DQ) \dx where \phi_i is
     *                                 the i-th local basis function of the state space.
     * @param scale                    A factor by which the result is scaled.
     */
    virtual void
    ElementValue_QU(const EDC<DH, VECTOR, dealdim> &edc,
                    dealii::Vector<double> &local_vector, double scale);

    /**
     * This evaluates the Cost Functional J_uq'(q,u)(.,DU) = \int_\Omega j_uq'(q(x),u(x))(.,DU) \dx on a given element T.
     *
     * @param edc                      The ElementDataContainer containing all the data necessary to evaluate the functional on
     *                                 a element.
     * @param local_vector        A Vector to contain the result. After completion local_vector fullfils
     *                                 local_vector(i) += scale * \int_T j_uq'(q(x),u(x))(\phi_i,DU) \dx where \phi_i is
     *                                 the i-th local basis function of the control space.
     * @param scale                    A factor by which the result is scaled.
     */
    virtual void
    ElementValue_UQ(const EDC<DH, VECTOR, dealdim> &edc,
                    dealii::Vector<double> &local_vector, double scale);

    /**
     * This evaluates the Cost Functional J_qq'(q,u)(.,DQ) = \int_\Omega j_qq'(q(x),u(x))(.,DQ) \dx on a given element T.
     *
     * @param edc                      The ElementDataContainer containing all the data necessary to evaluate the functional on
     *                                 a element.
     * @param local_vector        A Vector to contain the result. After completion local_vector fullfils
     *                                 local_vector(i) += scale * \int_T j_qq'(q(x),u(x))(\phi_i,DQ) \dx where \phi_i is
     *                                 the i-th local basis function of the control space.
     * @param scale                    A factor by which the result is scaled.
     */
    virtual void
    ElementValue_QQ(const EDC<DH, VECTOR, dealdim> &edc,
                    dealii::Vector<double> &local_vector, double scale);

    /**
     * This evaluates the Cost Functional J(q,u) = \sum_i j(q(x_i),u(x_i)). For given points x_i.
     *
     * @param control_dof_handler      The DOpEWrapper::DoFHandler for the control variable.
     * @param state_dof_handler        The DOpEWrapper::DoFHandler for the state variable.
     * @param param_values             A std::map containing parameter data (e.g. non space dependent data). If the control
     *                                 is done by parameters, it is contained in this map at the position "control".
     * @param domain_values            A std::map containing domain data (e.g. nodal vectors for FE-Functions). If the control
     *                                 is distributed, it is contained in this map at the position "control". The state may always
     *                                 be found in this map at the position "state"
     * @return                         A number which is \sum_i j(q(x_i),u(x_i))
     */
    virtual double
    PointValue(
      const DOpEWrapper::DoFHandler<dopedim, DH> &control_dof_handler,
      const DOpEWrapper::DoFHandler<dealdim, DH> &state_dof_handler,
      const std::map<std::string, const dealii::Vector<double>*> &param_values,
      const std::map<std::string, const VECTOR *> &domain_values);

    /**
     * This evaluates the Cost Functional J_u'(q,u)(.) = \sum_i j_u'(q(x_i),u(x_i))(.). For given points x_i.
     *
     * @param control_dof_handler      The DOpEWrapper::DoFHandler for the control variable.
     * @param state_dof_handler        The DOpEWrapper::DoFHandler for the state variable.
     * @param param_values             A std::map containing parameter data (e.g. non space dependent data). If the control
     *                                 is done by parameters, it is contained in this map at the position "control".
     * @param domain_values            A std::map containing domain data (e.g. nodal vectors for FE-Functions). If the control
     *                                 is distributed, it is contained in this map at the position "control". The state may always
     *                                 be found in this map at the position "state"
     * @param                          The complete (!) rhs-vector, i.e. J_u'(q,u)(phi_i) with the i-th basis vector of the state space phi_i.
     * @param scale                    A factor by which the result is scaled.
     */
    virtual void
    PointValue_U(
      const DOpEWrapper::DoFHandler<dopedim, DH> &control_dof_handler,
      const DOpEWrapper::DoFHandler<dealdim, DH> &state_dof_handler,
      const std::map<std::string, const dealii::Vector<double>*> &param_values,
      const std::map<std::string, const VECTOR *> &domain_values,
      VECTOR &rhs, double scale);

    /**
     * This evaluates the Cost Functional J_q'(q,u)(.) = \sum_i j_q'(q(x_i),u(x_i))(.). For given points x_i.
     *
     * @param control_dof_handler      The DOpEWrapper::DoFHandler for the control variable.
     * @param state_dof_handler        The DOpEWrapper::DoFHandler for the state variable.
     * @param param_values             A std::map containing parameter data (e.g. non space dependent data). If the control
     *                                 is done by parameters, it is contained in this map at the position "control".
     * @param domain_values            A std::map containing domain data (e.g. nodal vectors for FE-Functions). If the control
     *                                 is distributed, it is contained in this map at the position "control". The state may always
     *                                 be found in this map at the position "state"
     * @param                          The complete (!) rhs-vector, i.e. J_q'(q,u)(phi_i) with the i-th basis vector of the control space phi_i.
     * @param scale                    A factor by which the result is scaled.
     */
    virtual void
    PointValue_Q(
      const DOpEWrapper::DoFHandler<dopedim, DH> &control_dof_handler,
      const DOpEWrapper::DoFHandler<dealdim, DH> &state_dof_handler,
      const std::map<std::string, const dealii::Vector<double>*> &param_values,
      const std::map<std::string, const VECTOR *> &domain_values,
      VECTOR &rhs, double scale);

    /**
     * This evaluates the Cost Functional J_uu''(q,u)(.,Du) = \sum_i j_uu''(q(x_i),u(x_i))(., Du). For given points x_i.
     *
     * @param control_dof_handler      The DOpEWrapper::DoFHandler for the control variable.
     * @param state_dof_handler        The DOpEWrapper::DoFHandler for the state variable.
     * @param param_values             A std::map containing parameter data (e.g. non space dependent data). If the control
     *                                 is done by parameters, it is contained in this map at the position "control".
     * @param domain_values            A std::map containing domain data (e.g. nodal vectors for FE-Functions). If the control
     *                                 is distributed, it is contained in this map at the position "control". The state may always
     *                                 be found in this map at the position "state"
     * @param                          The complete (!) rhs-vector, i.e. J_uu''(q,u)(phi_i, Du) with the i-th basis vector of the state space phi_i.
     * @param scale                    A factor by which the result is scaled.
     */
    virtual void
    PointValue_UU(
      const DOpEWrapper::DoFHandler<dopedim, DH> &control_dof_handler,
      const DOpEWrapper::DoFHandler<dealdim, DH> &state_dof_handler,
      const std::map<std::string, const dealii::Vector<double>*> &param_values,
      const std::map<std::string, const VECTOR *> &domain_values,
      VECTOR &rhs, double scale);

    /**
     * This evaluates the Cost Functional J_qu''(q,u)(.,Dq) = \sum_i j_qu''(q(x_i),u(x_i))(., Dq). For given points x_i.
     *
     * @param control_dof_handler      The DOpEWrapper::DoFHandler for the control variable.
     * @param state_dof_handler        The DOpEWrapper::DoFHandler for the state variable.
     * @param param_values             A std::map containing parameter data (e.g. non space dependent data). If the control
     *                                 is done by parameters, it is contained in this map at the position "control".
     * @param domain_values            A std::map containing domain data (e.g. nodal vectors for FE-Functions). If the control
     *                                 is distributed, it is contained in this map at the position "control". The state may always
     *                                 be found in this map at the position "state"
     * @param                          The complete (!) rhs-vector, i.e. J_qu''(q,u)(phi_i, Dq) with the i-th basis vector
     *                                  of the state space phi_i.
     * @param scale                    A factor by which the result is scaled.
     */
    virtual void
    PointValue_QU(
      const DOpEWrapper::DoFHandler<dopedim, DH> &control_dof_handler,
      const DOpEWrapper::DoFHandler<dealdim, DH> &state_dof_handler,
      const std::map<std::string, const dealii::Vector<double>*> &param_values,
      const std::map<std::string, const VECTOR *> &domain_values,
      VECTOR &rhs, double scale);

    /**
     * This evaluates the Cost Functional J_uq''(q,u)(.,Du) = \sum_i j_uq''(q(x_i),u(x_i))(., Du). For given points x_i.
     *
     * @param control_dof_handler      The DOpEWrapper::DoFHandler for the control variable.
     * @param state_dof_handler        The DOpEWrapper::DoFHandler for the state variable.
     * @param param_values             A std::map containing parameter data (e.g. non space dependent data). If the control
     *                                 is done by parameters, it is contained in this map at the position "control".
     * @param domain_values            A std::map containing domain data (e.g. nodal vectors for FE-Functions). If the control
     *                                 is distributed, it is contained in this map at the position "control". The state may always
     *                                 be found in this map at the position "state"
     * @param                          The complete (!) rhs-vector, i.e. J_uq''(q,u)(phi_i, Du) with the i-th basis vector of
     *                                 the control space phi_i.
     * @param scale                    A factor by which the result is scaled.
     */
    virtual void
    PointValue_UQ(
      const DOpEWrapper::DoFHandler<dopedim, DH> &control_dof_handler,
      const DOpEWrapper::DoFHandler<dealdim, DH> &state_dof_handler,
      const std::map<std::string, const dealii::Vector<double>*> &param_values,
      const std::map<std::string, const VECTOR *> &domain_values,
      VECTOR &rhs, double scale);

    /**
     * This evaluates the Cost Functional J_qq''(q,u)(.,Dq) = \sum_i j_qq''(q(x_i),u(x_i))(., Dq). For given points x_i.
     *
     * @param control_dof_handler      The DOpEWrapper::DoFHandler for the control variable.
     * @param state_dof_handler        The DOpEWrapper::DoFHandler for the state variable.
     * @param param_values             A std::map containing parameter data (e.g. non space dependent data). If the control
     *                                 is done by parameters, it is contained in this map at the position "control".
     * @param domain_values            A std::map containing domain data (e.g. nodal vectors for FE-Functions). If the control
     *                                 is distributed, it is contained in this map at the position "control". The state may always
     *                                 be found in this map at the position "state"
     * @param                          The complete (!) rhs-vector, i.e. J_qq''(q,u)(phi_i, Dq) with the i-th basis vector of
     *                                 the control space phi_i.
     * @param scale                    A factor by which the result is scaled.
     */
    virtual void
    PointValue_QQ(
      const DOpEWrapper::DoFHandler<dopedim, DH> &control_dof_handler,
      const DOpEWrapper::DoFHandler<dealdim, DH> &state_dof_handler,
      const std::map<std::string, const dealii::Vector<double>*> &param_values,
      const std::map<std::string, const VECTOR *> &domain_values,
      VECTOR &rhs, double scale);

    /**
     * The same as FunctionalInterface::ElementValue only on boundaries.
     *
     * @param fdc                      A FaceDataContainer containing all the information to evaluate
     *                                 the functional on a face.
     * @param local_vector        A Vector to contain the result.
     * @param scale                    A factor by which the result is scaled.
     */
    virtual double
    BoundaryValue(const FDC<DH, VECTOR, dealdim> &fdc);

    /**
     * The same as FunctionalInterface::ElementValue_U only on boundaries.
     *
     * @param fdc                      A FaceDataContainer containing all the information to evaluate
     *                                 the functional on a face.
     * @param local_vector        A Vector to contain the result.
     * @param scale                    A factor by which the result is scaled.
     */
    virtual void
    BoundaryValue_U(const FDC<DH, VECTOR, dealdim> &fdc,
                    dealii::Vector<double> &local_vector, double scale);

    /**
     * The same as FunctionalInterface::ElementValue_Q only on boundaries.
     *
     * @param fdc                      A FaceDataContainer containing all the information to evaluate
     *                                 the functional on a face.
     * @param local_vector        A Vector to contain the result.
     * @param scale                    A factor by which the result is scaled.
     */
    virtual void
    BoundaryValue_Q(const FDC<DH, VECTOR, dealdim> &fdc,
                    dealii::Vector<double> &local_vector, double scale);

    /**
     * The same as FunctionalInterface::ElementValue_UU only on boundaries.
     *
     * @param fdc                      A FaceDataContainer containing all the information to evaluate
     *                                 the functional on a face.
     * @param local_vector        A Vector to contain the result.
     * @param scale                    A factor by which the result is scaled.
     */
    virtual void
    BoundaryValue_UU(const FDC<DH, VECTOR, dealdim> &fdc,
                     dealii::Vector<double> &local_vector, double scale);

    /**
     * The same as FunctionalInterface::ElementValue_QU only on boundaries.
     *
     * @param fdc                      A FaceDataContainer containing all the information to evaluate
     *                                 the functional on a face.
     * @param local_vector        A Vector to contain the result.
     * @param scale                    A factor by which the result is scaled.
     */
    virtual void
    BoundaryValue_QU(const FDC<DH, VECTOR, dealdim> &fdc,
                     dealii::Vector<double> &local_vector, double scale);

    /**
     * The same as FunctionalInterface::ElementValue_UQ only on boundaries.
     *
     * @param fdc                      A FaceDataContainer containing all the information to evaluate
     *                                 the functional on a face.
     * @param local_vector        A Vector to contain the result.
     * @param scale                    A factor by which the result is scaled.
     */
    virtual void
    BoundaryValue_UQ(const FDC<DH, VECTOR, dealdim> &fdc,
                     dealii::Vector<double> &local_vector, double scale);

    /**
     * The same as FunctionalInterface::ElementValue_QQ only on boundaries.
     *
     * @param fdc                      A FaceDataContainer containing all the information to evaluate
     *                                 the functional on a face.
     * @param local_vector        A Vector to contain the result.
     * @param scale                    A factor by which the result is scaled.
     */
    virtual void
    BoundaryValue_QQ(const FDC<DH, VECTOR, dealdim> &fdc,
                     dealii::Vector<double> &local_vector, double scale);

    /**
     * The same as FunctionalInterface::ElementValue only on a faces between elements.
     * This function is only used if FunctionalInterface::HasFaces returns true.
     *
     * @param fdc                      A FaceDataContainer containing all the information to evaluate
     *                                 the functional on a face.
     */
    virtual double
    FaceValue(const FDC<DH, VECTOR, dealdim> &fdc);

    /**
     * The same as FunctionalInterface::ElementValue_U only on a faces between elements.
     * This function is only used if FunctionalInterface::HasFaces returns true.
     *
     * @param fdc                      A FaceDataContainer containing all the information to evaluate
     *                                 the functional on a face.
     * @param local_vector        A Vector to contain the result.
     * @param scale                    A factor by which the result is scaled.
     */
    virtual void
    FaceValue_U(const FDC<DH, VECTOR, dealdim> &fdc,
                dealii::Vector<double> &local_vector, double scale);

    /**
     * The same as FunctionalInterface::ElementValue_Q only on a faces between elements.
     * This function is only used if FunctionalInterface::HasFaces returns true.
     *
     * @param fdc                      A FaceDataContainer containing all the information to evaluate
     *                                 the functional on a face.
     * @param local_vector        A Vector to contain the result.
     * @param scale                    A factor by which the result is scaled.
     */
    virtual void
    FaceValue_Q(const FDC<DH, VECTOR, dealdim> &fdc,
                dealii::Vector<double> &local_vector, double scale);

    /**
     * The same as FunctionalInterface::ElementValue_UU only on a faces between elements.
     * This function is only used if FunctionalInterface::HasFaces returns true.
     *
     * @param fdc                      A FaceDataContainer containing all the information to evaluate
     *                                 the functional on a face.
     * @param local_vector        A Vector to contain the result.
     * @param scale                    A factor by which the result is scaled.
     */
    virtual void
    FaceValue_UU(const FDC<DH, VECTOR, dealdim> &fdc,
                 dealii::Vector<double> &local_vector, double scale);

    /**
     * The same as FunctionalInterface::ElementValue_QU only on a faces between elements.
     * This function is only used if FunctionalInterface::HasFaces returns true.
     *
     * @param fdc                      A FaceDataContainer containing all the information to evaluate
     *                                 the functional on a face.
     * @param local_vector        A Vector to contain the result.
     * @param scale                    A factor by which the result is scaled.
     */
    virtual void
    FaceValue_QU(const FDC<DH, VECTOR, dealdim> &fdc,
                 dealii::Vector<double> &local_vector, double scale);

    /**
     * The same as FunctionalInterface::ElementValue_UQ only on a faces between elements.
     * This function is only used if FunctionalInterface::HasFaces returns true.
     *
     * @param fdc                      A FaceDataContainer containing all the information to evaluate
     *                                 the functional on a face.
     * @param local_vector        A Vector to contain the result.
     * @param scale                    A factor by which the result is scaled.
     */
    virtual void
    FaceValue_UQ(const FDC<DH, VECTOR, dealdim> &fdc,
                 dealii::Vector<double> &local_vector, double scale);

    /**
     * The same as FunctionalInterface::ElementValue_QQ only on a faces between elements.
     * This function is only used if FunctionalInterface::HasFaces returns true.
     *
     * @param fdc                      A FaceDataContainer containing all the information to evaluate
     *                                 the functional on a face.
     * @param local_vector        A Vector to contain the result.
     * @param scale                    A factor by which the result is scaled.
     */
    virtual void
    FaceValue_QQ(const FDC<DH, VECTOR, dealdim> &fdc,
                 dealii::Vector<double> &local_vector, double scale);

    /**
     * Implements a functional that can be computed by the values in some given Vectors or BlockVectors
     */
    virtual double
    AlgebraicValue(
      const std::map<std::string, const dealii::Vector<double>*> &param_values,
      const std::map<std::string, const VECTOR *> &domain_values);
    /**
     * Implements the gradient of a functional that can be computed by the values in some given Vectors or BlockVectors
     */
    virtual void
    AlgebraicGradient_Q(VECTOR &gradient,
                        const std::map<std::string, const dealii::Vector<double>*> &param_values,
                        const std::map<std::string, const VECTOR *> &domain_values);

    /**
     * This function describes what type of Functional is considered
     *
     * @return A string describing the functional, feasible values are "domain", "boundary", "point" or "face"
     *         if it contains domain, or boundary ... parts all combinations of these keywords are feasible.
     *         In time dependent problems use "timelocal" to indicate that
     *         it should only be evaluated at a certain time_point, or "timedistributed" to consider \int_0^T J(t,q(t),u(t))  \dt
     *         only one of the words "timelocal" and "timedistributed" should be considered if not it will be considered to be
     *         "timelocal"
     *
     */
    virtual std::string
    GetType() const;
    /**
     * This function is used to name the Functional, this is helpful to distinguish different Functionals in the output.
     *
     * @return A string. This is the name being displayed next to the computed values.
     */
    virtual std::string
    GetName() const;

    /**
     * This Function is used to determine whether the current time is required by the functional.
     * The Time is assumed to be set prior by FunctionalInterface::SetTime
     *
     * @return A boolean that is true if the functional should be evaluated at the current time point.
     *         The default is true, i.e., we assume that unless stated otherwise the functional
    *         should be evaluated.
     */
    virtual bool
    NeedTime() const
    {
      return true;
    }

    /**
     * Sets the time for the functional. This is required by FunctionalInterface::NeedTime, and if the time is
     * used within the functional to compute its value.
     *
     * @param t           The time that should be set.
     * @param step_size   The size of the time step.
     */
    void
    SetTime(double t, double step_size) const
    {
      time_ = t;
      step_size_ = step_size;
    }

    /**
     * This function tells what dealii::UpdateFlags are required by the functional to be used when initializing the
     * DOpEWrapper::FEValues on an element.
     */
    virtual dealii::UpdateFlags
    GetUpdateFlags() const;

    /**
     * This function tells what dealii::UpdateFlags are required by the functional to be used when initializing the
     * DOpEWrapper::FEFaceValues on a face.
     */
    virtual dealii::UpdateFlags
    GetFaceUpdateFlags() const;

    /**
     * This function determines whether a loop over all faces is required or not.
     *
     * @return Returns whether or not this functional has components on faces between elements.
     *         The default value is determined by the type of the functional, i.e. true if the
     *         signal 'face' is found in GetType(), false otherwise.
     */
    virtual bool
    HasFaces() const;

    /**
     * This function determines whether an evaluation of PointRhs is required or not.
     *
     * @return Returns whether or not this functional needs pointevaluations. The default
     *         value is determined by the type of the functional, i.e. true if the
     *         signal 'face' is found in GetType(), false otherwise.
     */
    virtual bool
    HasPoints() const;

    /**
     * This function determines whether the functional needs multiple evaluation runs.
     * This method needs to return values larger than zero to evaluate functionals like
     * g(\int f(u(x)) dx)
     * The number determines how many evaluation runs are necessary.
     *
     * Its default return value is zero!
     *
     */
    virtual unsigned int
    NeedPrecomputations() const;

    /**
     * This function needs to return true if control in the initial value is done with a
     * functional coupling the control with the final value, e.g., to assert
     * periodicity of the solution. Then the corresponding container
     * will make the final values of the state and tangent available during the
     * calculations of the gradient.
     *
     * Its default return value is false!
     *
     */
    virtual bool
    NeedFinalValue() const;

    void
    SetProblemType(std::string type,unsigned int num);

  protected:
    std::string
    GetProblemType() const;
    unsigned int GetProblemNum() const;
    double GetTime() const
    {
      return time_;
    }
    double GetTimeStepSize() const
    {
      return step_size_;
    }

  private:
    std::string problem_type_;
    unsigned int problem_num_;
    mutable double time_;
    mutable double step_size_;

  };
}

#endif
