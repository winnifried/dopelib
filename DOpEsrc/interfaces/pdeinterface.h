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

#ifndef PDE_INTERFACE_H_
#define PDE_INTERFACE_H_

#include <map>
#include <string>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/base/function.h>

#include <wrapper/fevalues_wrapper.h>
#include <container/elementdatacontainer.h>
#include <container/facedatacontainer.h>
#include <container/multimesh_elementdatacontainer.h>
#include <container/multimesh_facedatacontainer.h>
#include <network/network_elementdatacontainer.h>
#include <network/network_facedatacontainer.h>

namespace DOpE
{

  /**
   * A template providing all evaluations of a PDE that may be used
   * during the solution of a PDE or an optimization with a PDE constraint.
   *
   * Whenever used below:
   *        u  denotes the solution to the PDE
   *        q  denotes a given control (i.e., parameter to the PDE)
   *        z  denotes the adjoint solution
   *        dq denotes a given, fixed, direction in the control space
   *        du denotes the tangent solution to the PDE according to a
   *           given dq
   *        dz denotes an auxilliary adjoint used to caclulate second
   *           derivatives, e.g., the hessian.
   *
   *        \phi and \phi_q denote the basis functions in the state and control
   *           test space
   */
  template<
  template<template<int, int> class DH, typename VECTOR, int dealdim> class EDC,
           template<template<int, int> class DH, typename VECTOR, int dealdim> class FDC,
           template<int, int> class DH, typename VECTOR,  int dealdim>
  class PDEInterface
  {
  public:
    PDEInterface();
    virtual
    ~PDEInterface();

    /******************************************************/

    /**
     * Assuming that the PDE is given in the form a(u;\phi) = f(\phi),
    * this function implements all terms in a(u;\phi) that are
    * represented by intergrals over elements T.
    * Hence, if a(u;\phi) = \sum_T \int_T a_T(u;\phi) + ...
    * then this function needs to implement \int_T a_T(u;\phi)
    * a_T may depend upon any spatial derivatives, but not on temporal
    * derivatives.
    *
    * @param edc                The ElementDataContainer object which provides
    *                           access to all information on the element,
    *                           e.g., test-functions, mesh size,...
    * @param local_vector  The vector containing the integrals
    *                           ordered according to the local number
    *                           of the testfunction.
    * @param scale              A scaling parameter to be used in all
    *                           equations.
    * @param scale_ico          A special scaling parameter to be used
    *                           in certain parts of the equation
    *                           if they need to be treated differently in
    *                           time-stepping schemes, see the PDF-documentation
    *                           for more details.
    *
     */
    virtual void
    ElementEquation(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                    dealii::Vector<double> &/*local_vector*/,
                    double /*scale*/,
                    double /*scale_ico*/);

    /******************************************************/

    /**
     * This function is used for error estimation and should implement
    * the strong form of the residual on an element T.
    *
    * @param edc                The ElementDataContainer object which provides
    *                           access to all information on the element,
    *                           e.g., test-functions, mesh size,...
    * @param edc_wight          The ElementDataContainer for the weight-function,
    *                           e.g., the testfunction by which the
    *                           residual needs to be multiplied
    * @param ret                The value of the integral on the element
    *                           of residual times testfunction.
    * @param scale              A scaling parameter to be used in all
    *                           equations.
    *
     */
    virtual void
    StrongElementResidual(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                          const EDC<DH, VECTOR, dealdim> & /*edc_weight*/,
                          double & /*ret*/,
                          double /*scale*/);

    /******************************************************/

    /**
     * Assuming that the discretization of temporal derivatives by a backward
    * difference, i.e., \partial_t u(t_i) \approx 1/\delta t ( u(t_i) - u(t_{i-1})
    * in several cases the the temporal derivatives of the
    * equation give rise to a spacial integral of the form
    *
    * \int_Omega T(u(t_i); \phi(t_i)) - \int_Omega T(u(t_{i-1}); \phi(t_{i-1}))
    *
    * This equation is used to implement the element contribution
    * \int_T T(u,\phi)
    *
    * @param edc                The ElementDataContainer object which provides
    *                           access to all information on the element,
    *                           e.g., test-functions, mesh size,...
    * @param local_vector  The vector containing the integrals
    *                           ordered according to the local number
    *                           of the testfunction.
    * @param scale              A scaling parameter to be used in all
    *                           equations.
    */
    //Note that the _UU term is not needed, since we assume that ElementTimeEquation is linear!
    virtual void
    ElementTimeEquation(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                        dealii::Vector<double> &/*local_vector*/,
                        double /*scale*/);

    /******************************************************/
    /**
     * Same as ElementTimeEquation, but here the derivative of T with
    * respect to u is considered.
    * Here, the derivative of T in u in a direction du
    * for a fixed test function z
    * is denoted as T'(u;du,z)
    *
    * This equation is used to implement the element contribution
    * \int_T T'(u;\phi,z)
    *
    * @param edc                The ElementDataContainer object which provides
    *                           access to all information on the element,
    *                           e.g., test-functions, mesh size,...
    * @param local_vector  The vector containing the integrals
    *                           ordered according to the local number
    *                           of the testfunction.
    * @param scale              A scaling parameter to be used in all
    *                           equations.
    */
    virtual void
    ElementTimeEquation_U(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                          dealii::Vector<double> &/*local_vector*/,
                          double /*scale*/);

    /******************************************************/
    /**
     * Same as ElementTimeEquation_U, but we exchange the argument for
    * the test function.
    *
    * This equation is used to implement the element contribution
    * \int_T T'(u;du,\phi)
    *
    * @param edc                The ElementDataContainer object which provides
    *                           access to all information on the element,
    *                           e.g., test-functions, mesh size,...
    * @param local_vector  The vector containing the integrals
    *                           ordered according to the local number
    *                           of the testfunction.
    * @param scale              A scaling parameter to be used in all
    *                           equations.
    */
    virtual void
    ElementTimeEquation_UT(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                           dealii::Vector<double> &/*local_vector*/,
                           double /*scale*/);

    /******************************************************/
    /**
     * Same as ElementTimeEquation_UT, but we exchange the argument for
    * the test function.
    *
    * This equation is used to implement the element contribution
    * \int_T T'(u;\phi,dz)
    *
    * Note that this is the same function as in ElementTimeEquation_U,
    * but it is used with an other argument dz instead of z.
    *
    * @param edc                The ElementDataContainer object which provides
    *                           access to all information on the element,
    *                           e.g., test-functions, mesh size,...
    * @param local_vector  The vector containing the integrals
    *                           ordered according to the local number
    *                           of the testfunction.
    * @param scale              A scaling parameter to be used in all
    *                           equations.
    */
    virtual void
    ElementTimeEquation_UTT(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                            dealii::Vector<double> &/*local_vector*/,
                            double /*scale*/);

    /******************************************************/

    /**
     * In certain cases, the assumption of ElementTimeEquation
     * are not meet, i.e., we can not use the
     * same operator T at t_i and t_{i-1}.
     * In these cases instead of ElementTimeEquation this
     * funtion is used, where the user can implement the
     * complete approximation of the temporal derivative.
     *
     * @param edc                The ElementDataContainer object which provides
     *                           access to all information on the element,
     *                           e.g., test-functions, mesh size,...
     * @param local_vector  The vector containing the integrals
     *                           ordered according to the local number
     *                           of the testfunction.
     * @param scale              A scaling parameter to be used in all
     *                           equations.
     */
    virtual void
    ElementTimeEquationExplicit(const EDC<DH, VECTOR, dealdim> & /*edc**/,
                                dealii::Vector<double> &/*local_vector*/,
                                double /*scale*/);
    /******************************************************/
    /**
     * Analog to ElementTimeEquationExplicit, this function is used
     * to replace ElementTimeEquation_U if needed.
     *
     * @param edc                The ElementDataContainer object which provides
     *                           access to all information on the element,
     *                           e.g., test-functions, mesh size,...
     * @param local_vector  The vector containing the integrals
     *                           ordered according to the local number
     *                           of the testfunction.
     * @param scale              A scaling parameter to be used in all
     *                           equations.
     */
    virtual void
    ElementTimeEquationExplicit_U(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                                  dealii::Vector<double> &/*local_vector*/,
                                  double /*scale*/);

    /******************************************************/
    /**
     * Analog to ElementTimeEquationExplicit, this function is used
     * to replace ElementTimeEquation_UT if needed.
     *
     * @param edc                The ElementDataContainer object which provides
     *                           access to all information on the element,
     *                           e.g., test-functions, mesh size,...
     * @param local_vector  The vector containing the integrals
     *                           ordered according to the local number
     *                           of the testfunction.
     * @param scale              A scaling parameter to be used in all
     *                           equations.
     */

    virtual void
    ElementTimeEquationExplicit_UT(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                                   dealii::Vector<double> &/*local_vector*/,
                                   double /*scale*/);

    /******************************************************/
    /**
     * Analog to ElementTimeEquationExplicit, this function is used
     * to replace ElementTimeEquation_UTT if needed.
     *
     * @param edc                The ElementDataContainer object which provides
     *                           access to all information on the element,
     *                           e.g., test-functions, mesh size,...
     * @param local_vector  The vector containing the integrals
     *                           ordered according to the local number
     *                           of the testfunction.
     * @param scale              A scaling parameter to be used in all
     *                           equations.
     */
    virtual void
    ElementTimeEquationExplicit_UTT(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                                    dealii::Vector<double> &/*local_vector*/,
                                    double /*scale*/);

    /******************************************************/
    /**
     * For nonlinear terms involved in the temporal derivative
     * the second derivatives with respect to the state of the
     * time derivative are implemented here.
     *
     * @param edc                The ElementDataContainer object which provides
     *                           access to all information on the element,
     *                           e.g., test-functions, mesh size,...
     * @param local_vector  The vector containing the integrals
     *                           ordered according to the local number
     *                           of the testfunction.
     * @param scale              A scaling parameter to be used in all
     *                           equations.
     */

    virtual void
    ElementTimeEquationExplicit_UU(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                                   dealii::Vector<double> &/*local_vector*/,
                                   double /*scale*/);

    /******************************************************/
    /**
     * This term implements the derivative of ElementEquation
     * with respect to the u argument. I.e., if
     * ElementEquation implements the term
     * int_T a_T(u;\phi) then this method
     * implements \int_T a_T'(u;\phi,z)
     * where \phi denotes the direction to which the derivative
     * is applied
     *
     * @param edc                The ElementDataContainer object which provides
     *                           access to all information on the element,
     *                           e.g., test-functions, mesh size,...
     * @param local_vector  The vector containing the integrals
     *                           ordered according to the local number
     *                           of the testfunction.
     * @param scale              A scaling parameter to be used in all
     *                           equations.
     * @param scale_ico          A special scaling parameter to be used
     *                           in certain parts of the equation
     *                           if they need to be treated differently in
     *                           time-stepping schemes, see the PDF-documentation
     *                           for more details.
     */
    virtual void
    ElementEquation_U(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                      dealii::Vector<double> &/*local_vector*/,
                      double /*scale*/,
                      double /*scale_ico*/);

    /******************************************************/
    /**
     * Similar to the StongElementResidual, this function implements the
     * strong element residual for the adjoint equation.
     *
     * @param edc                The ElementDataContainer object which provides
     *                           access to all information on the element,
     *                           e.g., test-functions, mesh size,...
     * @param edc_weight          The ElementDataContainer for the weight-function,
     *                           e.g., the testfunction by which the
     *                           residual needs to be multiplied
     * @param ret                The value of the integral on the element
     *                           of residual times testfunction.
     * @param scale              A scaling parameter to be used in all
     *                           equations.
     */

    virtual void
    StrongElementResidual_U(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                            const EDC<DH, VECTOR, dealdim> & /*edc_weight*/,
                            double & /*ret*/,
                            double /*scale*/);

    /******************************************************/
    /**
     * This term implements the derivative of ElementEquation
     * with respect to the u argument. I.e., if
     * ElementEquation implements the term
     * int_T a_T(u;\phi) then this method
     * implements \int_T a_T'(u;du,\phi) .
     * In contrast to ElementEquation_U the arguments are exchanged.
     *
     * @param edc                The ElementDataContainer object which provides
     *                           access to all information on the element,
     *                           e.g., test-functions, mesh size,...
     * @param local_vector  The vector containing the integrals
     *                           ordered according to the local number
     *                           of the testfunction.
     * @param scale              A scaling parameter to be used in all
     *                           equations.
     * @param scale_ico          A special scaling parameter to be used
     *                           in certain parts of the equation
     *                           if they need to be treated differently in
     *                           time-stepping schemes, see the PDF-documentation
     *                           for more details.
     */

    virtual void
    ElementEquation_UT(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                       dealii::Vector<double> &/*local_vector*/,
                       double /*scale*/,
                       double /*scale_ico*/);

    /******************************************************/
    /**
     * This term implements the derivative of ElementEquation
     * with respect to the u argument. I.e., if
     * ElementEquation implements the term
     * int_T a_T(u;\phi) then this method
     * implements \int_T a_T'(u;phi,dz) .
     *
     * This implements the same form as ElementEquation_U, but
     * with exchanged functions, i.e., dz instead of z.
     *
     * @param edc                The ElementDataContainer object which provides
     *                           access to all information on the element,
     *                           e.g., test-functions, mesh size,...
     * @param local_vector  The vector containing the integrals
     *                           ordered according to the local number
     *                           of the testfunction.
     * @param scale              A scaling parameter to be used in all
     *                           equations.
     * @param scale_ico          A special scaling parameter to be used
     *                           in certain parts of the equation
     *                           if they need to be treated differently in
     *                           time-stepping schemes, see the PDF-documentation
     *                           for more details.
     */

    virtual void
    ElementEquation_UTT(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                        dealii::Vector<double> &/*local_vector*/,
                        double /*scale*/,
                        double /*scale_ico*/);

    /******************************************************/

    /**
     * Assuming that the element equation a_T depends not
     * only on the state u, but also on a control q, this
     * term implements the derivative of ElementEquation
     * with respect to variations in q.
     * This term implements the derivative of ElementEquation
     * with respect to the u argument. I.e., if
     * ElementEquation implements the term
     * int_T a_T(u,q;\phi) then this method
     * implements \int_T a_T'_q(u,q;\phi_q,z) .
     *
     * @param edc                The ElementDataContainer object which provides
     *                           access to all information on the element,
     *                           e.g., test-functions, mesh size,...
     * @param local_vector  The vector containing the integrals
     *                           ordered according to the local number
     *                           of the testfunction.
     * @param scale              A scaling parameter to be used in all
     *                           equations.
     * @param scale_ico          A special scaling parameter to be used
     *                           in certain parts of the equation
     *                           if they need to be treated differently in
     *                           time-stepping schemes, see the PDF-documentation
     *                           for more details.
     */
    virtual void
    ElementEquation_Q(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                      dealii::Vector<double> &/*local_vector*/,
                      double /*scale*/,
                      double /*scale_ico*/);

    /******************************************************/

    /**
    * Analog to ElementEqution_Q this term implements the derivative
    * of ElementEquation with respect to the control argument.
    * In contrast to ElementEqution_Q the test function
    * is taken from the state space, while the argument for
    * the control variation dq is fixed.
    * ElementEquation implements the term
    * int_T a_T(u,q;\phi) then this method
    * implements \int_T a_T'_q(u,q;dq,\phi) .
    *
    * @param edc                The ElementDataContainer object which provides
    *                           access to all information on the element,
    *                           e.g., test-functions, mesh size,...
    * @param local_vector  The vector containing the integrals
    *                           ordered according to the local number
    *                           of the testfunction.
    * @param scale              A scaling parameter to be used in all
    *                           equations.
    * @param scale_ico          A special scaling parameter to be used
    *                           in certain parts of the equation
    *                           if they need to be treated differently in
    *                           time-stepping schemes, see the PDF-documentation
    *                           for more details.
    */
    virtual void
    ElementEquation_QT(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                       dealii::Vector<double> &/*local_vector*/,
                       double /*scale*/,
                       double scale_ico);

    /******************************************************/

    /**
    * Analog to ElementEqution_Q, the only difference is that
    * the argument z is exchanged by dz
    * int_T a_T(u,q;\phi) then this method
    * implements \int_T a_T'_q(u,q;\phi_q,dz) .
    *
    * @param edc                The ElementDataContainer object which provides
    *                           access to all information on the element,
    *                           e.g., test-functions, mesh size,...
    * @param local_vector  The vector containing the integrals
    *                           ordered according to the local number
    *                           of the testfunction.
    * @param scale              A scaling parameter to be used in all
    *                           equations.
    * @param scale_ico          A special scaling parameter to be used
    *                           in certain parts of the equation
    *                           if they need to be treated differently in
    *                           time-stepping schemes, see the PDF-documentation
    *                           for more details.
    */
    virtual void
    ElementEquation_QTT(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                        dealii::Vector<double> &/*local_vector*/,
                        double /*scale*/,
                        double /*scale_ico*/);

    /******************************************************/

    /**
     * Analog to ElementEquation_U, but now considering second
     * derivatives with respect to u, i.e., we calculate
     * \int_T a_T''_{uu}(u,q;du,\phi,z)
     * where du is the given tangent direction
     *
     * @param edc                The ElementDataContainer object which provides
     *                           access to all information on the element,
     *                           e.g., test-functions, mesh size,...
     * @param local_vector  The vector containing the integrals
     *                           ordered according to the local number
     *                           of the testfunction.
     * @param scale              A scaling parameter to be used in all
     *                           equations.
     * @param scale_ico          A special scaling parameter to be used
     *                           in certain parts of the equation
     *                           if they need to be treated differently in
     *                           time-stepping schemes, see the PDF-documentation
     *                           for more details.
     */

    virtual void
    ElementEquation_UU(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                       dealii::Vector<double> &/*local_vector*/,
                       double /*scale*/,
                       double /*scale_ico*/);

    /******************************************************/

    /**
     * Analog to ElementEquation_U and ElementEquation_Q,
     * but now considering the mixed second
     * derivatives with respect to u and q, i.e., we calculate
     * \int_T a_T''_{qu}(u,q;dq,\phi,z)
     * where dq is a given variation for the control. This means the
     * test function is taken in the state space.
     *
     * @param edc                The ElementDataContainer object which provides
     *                           access to all information on the element,
     *                           e.g., test-functions, mesh size,...
     * @param local_vector  The vector containing the integrals
     *                           ordered according to the local number
     *                           of the testfunction.
     * @param scale              A scaling parameter to be used in all
     *                           equations.
     * @param scale_ico          A special scaling parameter to be used
     *                           in certain parts of the equation
     *                           if they need to be treated differently in
     *                           time-stepping schemes, see the PDF-documentation
     *                           for more details.
     */

    virtual void
    ElementEquation_QU(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                       dealii::Vector<double> &/*local_vector*/,
                       double /*scale*/,
                       double /*scale_ico*/);

    /******************************************************/
    /**
     * Analog to ElementEquation_QU, but with different arguments, i.e., we calculate
     * \int_T a_T''_{uq}(u,q;du,phi_q,z)
     * where du is the given tangent direction. This means the
     * test function is taken in the control space.
     *
     * @param edc                The ElementDataContainer object which provides
     *                           access to all information on the element,
     *                           e.g., test-functions, mesh size,...
     * @param local_vector  The vector containing the integrals
     *                           ordered according to the local number
     *                           of the testfunction.
     * @param scale              A scaling parameter to be used in all
     *                           equations.
     * @param scale_ico          A special scaling parameter to be used
     *                           in certain parts of the equation
     *                           if they need to be treated differently in
     *                           time-stepping schemes, see the PDF-documentation
     *                           for more details.
     */

    virtual void
    ElementEquation_UQ(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                       dealii::Vector<double> &/*local_vector*/,
                       double /*scale*/,
                       double /*scale_ico*/);

    /******************************************************/
    /**
    * Analog to ElementEquation_Q, but now considering second
    * derivatives with respect to q, i.e., we calculate
    * \int_T a_T''_{qq}(u,q;dq,\phi_q,z)
    * where dq is the given direction.
    *
    * @param edc                The ElementDataContainer object which provides
    *                           access to all information on the element,
    *                           e.g., test-functions, mesh size,...
    * @param local_vector  The vector containing the integrals
    *                           ordered according to the local number
    *                           of the testfunction.
    * @param scale              A scaling parameter to be used in all
    *                           equations.
    * @param scale_ico          A special scaling parameter to be used
    *                           in certain parts of the equation
    *                           if they need to be treated differently in
    *                           time-stepping schemes, see the PDF-documentation
    *                           for more details.
    */
    virtual void
    ElementEquation_QQ(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                       dealii::Vector<double> &/*local_vector*/,
                       double /*scale*/,
                       double /*scale_ico*/);

    /******************************************************/
    /**
    * Implements the element integral corresponding to given volume
    * data for the PDE.
    *
    * @param edc                The ElementDataContainer object which provides
    *                           access to all information on the element,
    *                           e.g., test-functions, mesh size,...
    * @param local_vector  The vector containing the integrals
    *                           ordered according to the local number
    *                           of the testfunction.
    * @param scale              A scaling parameter to be used in all
    *                           equations.
    */
    virtual void
    ElementRightHandSide(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                         dealii::Vector<double> &/*local_vector*/,
                         double /*scale*/);

    /******************************************************/

    /**
     * This implements the element integral used to calculate the
     * stiffness matrix for the primal PDE. It corresponds to
     * the evaluation of the matrix entries
     * a_ij = \int_T a_T'(u;\phi_j,\phi_i)
     *
     *
     * @param edc                The ElementDataContainer object which provides
     *                           access to all information on the element,
     *                           e.g., test-functions, mesh size,...
     * @param local_entry_matrix The matrix containing the integrals
     *                           ordered according to the local number
     *                           of the testfunction.
     * @param scale              A scaling parameter to be used in all
     *                           equations.
     * @param scale_ico          A special scaling parameter to be used
     *                           in certain parts of the equation
     *                           if they need to be treated differently in
     *                           time-stepping schemes, see the PDF-documentation
     *                           for more details.
     */
    virtual void
    ElementMatrix(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                  dealii::FullMatrix<double> &/*local_entry_matrix*/,
                  double /*scale*/,
                  double /*scale_ico*/);

    /******************************************************/
    /**
     * This implements the element integral used to calculate the
     * matrix for the primal PDE corresponding to the time derivatives
     * given in ElementTimeEquation.
     *
     * @param edc                The ElementDataContainer object which provides
     *                           access to all information on the element,
     *                           e.g., test-functions, mesh size,...
     * @param local_entry_matrix The matrix containing the integrals
     *                           ordered according to the local number
     *                           of the testfunction.
     */

    virtual void
    ElementTimeMatrix(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                      dealii::FullMatrix<double> &/*local_entry_matrix*/);

    /******************************************************/
    /**
     * The transposed of ElementTimeEquation.
     *
     * @param edc                The ElementDataContainer object which provides
     *                           access to all information on the element,
     *                           e.g., test-functions, mesh size,...
     * @param local_entry_matrix The matrix containing the integrals
     *                           ordered according to the local number
     *                           of the testfunction.
     */
    virtual void
    ElementTimeMatrix_T(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                        dealii::FullMatrix<double> &/*local_entry_matrix*/);

    /******************************************************/
    /**
     * This implements the element integral used to calculate the
     * matrix for the primal PDE corresponding to the time derivatives
     * given in ElementTimeEquationExplicit.
     *
     * @param edc                The ElementDataContainer object which provides
     *                           access to all information on the element,
     *                           e.g., test-functions, mesh size,...
     * @param local_entry_matrix The matrix containing the integrals
     *                           ordered according to the local number
     *                           of the testfunction.
     */

    virtual void
    ElementTimeMatrixExplicit(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                              dealii::FullMatrix<double> &/*local_entry_matrix*/);

    /******************************************************/
    /**
     * The transposed of ElementTimeEquationExplicit.
     *
     * @param edc                The ElementDataContainer object which provides
     *                           access to all information on the element,
     *                           e.g., test-functions, mesh size,...
     * @param local_entry_matrix The matrix containing the integrals
     *                           ordered according to the local number
     *                           of the testfunction.
     */
    virtual void
    ElementTimeMatrixExplicit_T(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                                dealii::FullMatrix<double> &/*local_entry_matrix*/);

    /******************************************************/
    /**
     * This implements the element integral used to calculate the
     * stiffness matrix for the adjoint PDE. It corresponds to
     * the evaluation of the matrix entries
     * a_ji = \int_T a_T'(u;\phi_j,\phi_i)
     *
     * By default, this function calls Element_Matrix and afterwards
     * returns the transposed of the matrix.
     *
     *
     * @param edc                The ElementDataContainer object which provides
     *                           access to all information on the element,
     *                           e.g., test-functions, mesh size,...
     * @param local_entry_vector The matrix containing the integrals
     *                           ordered according to the local number
     *                           of the testfunction.
     * @param scale              A scaling parameter to be used in all
     *                           equations.
     * @param scale_ico          A special scaling parameter to be used
     *                           in certain parts of the equation
     *                           if they need to be treated differently in
     *                           time-stepping schemes, see the PDF-documentation
     *                           for more details.
     */
    virtual void
    ElementMatrix_T(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                    dealii::FullMatrix<double> &/*local_entry_matrix*/,
                    double /*scale*/,
                    double /*scale_ico*/);

    /******************************************************/

    /**
     * This implements the scalar product in the control space.
     * The equation is used to calculate the representation
     * of the cost functional gradient given the derivative
     * of the cost functional
     *
     * @param edc                The ElementDataContainer object which provides
     *                           access to all information on the element,
     *                           e.g., test-functions, mesh size,...
     * @param local_vector  The vector containing the integrals
     *                           ordered according to the local number
     *                           of the testfunction.
     * @param scale              A scaling parameter to be used in all
     *                           equations.
     */
    virtual void
    ControlElementEquation(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                           dealii::Vector<double> &/*local_vector*/,
                           double /*scale*/);

    /******************************************************/
    /**
    * This implements the matrix corresponding to ControlElementEquation
    *
    * @param edc                The ElementDataContainer object which provides
    *                           access to all information on the element,
    *                           e.g., test-functions, mesh size,...
    * @param local_entry_matrix The matrix containing the integrals
    *                           ordered according to the local number
    *                           of the testfunction.
    */
    virtual void
    ControlElementMatrix(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                         dealii::FullMatrix<double> &/*local_entry_matrix*/,
                         double /*scale*/);
    /******************************************************/

    /**
     * This implements the scalar product on the boundary in the control space.
     * The equation is used to calculate the representation
     * of the cost functional gradient given the derivative
     * of the cost functional
     *
     * @param fdc                The FaceDataContainer object which provides
     *                           access to all information on the element,
     *                           e.g., test-functions, mesh size,...
     * @param local_vector  The vector containing the integrals
     *                           ordered according to the local number
     *                           of the testfunction.
     * @param scale              A scaling parameter to be used in all
     *                           equations.
     */
    virtual void
    ControlBoundaryEquation(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                            dealii::Vector<double> &/*local_vector*/,
                            double /*scale*/);

    /******************************************************/
    /**
    * This implements the matrix corresponding to ControlBoundaryEquation
    *
    * @param fdc                The FaceDataContainer object which provides
    *                           access to all information on the element,
    *                           e.g., test-functions, mesh size,...
    * @param local_entry_matrix The matrix containing the integrals
    *                           ordered according to the local number
    *                           of the testfunction.
    */
    virtual void
    ControlBoundaryMatrix(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                          dealii::FullMatrix<double> &/*local_entry_matrix*/,
                          double /*scale*/);
    /******************************************************/

    /**
     * Similar to the StongElementResidual, this function implements the
     * strong element residual for the gradient equation, i.e., j'(q) = 0.
     *
     * @param edc                The ElementDataContainer object which provides
     *                           access to all information on the element,
     *                           e.g., test-functions, mesh size,...
     * @param edc_weight         The ElementDataContainer for the weight-function,
     *                           e.g., the testfunction by which the
     *                           residual needs to be multiplied
     * @param ret                The value of the integral on the element
     *                           of residual times testfunction.
     * @param scale              A scaling parameter to be used in all
     *                           equations.
     */
    virtual void
    StrongElementResidual_Control(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                                  const EDC<DH, VECTOR, dealdim> & /*edc_weight*/,
                                  double & /*ret*/,
                                  double /*scale*/);
    /******************************************************/
    /**
     * Similar to the StongElementResidual, this function implements the
     * strong face residual (i.e., jumps in conormal direction)
     * for the gradient equation, i.e., j'(q) = 0, if present.
     *
     * @param fdc                The FaceDataContainer object which provides
     *                           access to all information on the face,
     *                           e.g., test-functions, mesh size,...
     * @param fdc_weight         The FaceDataContainer for the weight-function,
     *                           e.g., the testfunction by which the
     *                           residual needs to be multiplied
     * @param ret                The value of the integral on the element
     *                           of residual times testfunction.
     * @param scale              A scaling parameter to be used in all
     *                           equations.
     */

    virtual void
    StrongFaceResidual_Control(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                               const FDC<DH, VECTOR, dealdim> & /*fdc_weight*/,
                               double & /*ret*/,
                               double /*scale*/);
    /******************************************************/
    /**
     * Similar to the StongElementResidual, this function implements the
     * strong boundary residual (i.e., jumps in conormal direction)
     * for the gradient equation, i.e., j'(q) = 0, if present.
     *
     * @param fdc                The FaceDataContainer object which provides
     *                           access to all information on the face,
     *                           e.g., test-functions, mesh size,...
     * @param fdc_weight         The FaceDataContainer for the weight-function,
     *                           e.g., the testfunction by which the
     *                           residual needs to be multiplied
     * @param ret                The value of the integral on the element
     *                           of residual times testfunction.
     * @param scale              A scaling parameter to be used in all
     *                           equations.
     */

    virtual void
    StrongBoundaryResidual_Control(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                                   const FDC<DH, VECTOR, dealdim> & /*fdc_weight*/,
                                   double & /*ret*/,
                                   double /*scale*/);

    /******************************************************/
    // Functions for Face Integrals

    /**
     * The following Face... and Boundary... methods
     * implement the analog terms as the corresponding
     * Element... methods, except that now integrals on
     * faces between elements or on the domain boundary
     * are considered.
     *
     */
    virtual void
    FaceEquation(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                 dealii::Vector<double> &/*local_vector*/,
                 double /*scale*/,
                 double /*scale_ico*/);
    /******************************************************/

    virtual void
    StrongFaceResidual(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                       const FDC<DH, VECTOR, dealdim> & /*fdc_weight*/,
                       double & /*ret*/,
                       double /*scale*/);

    /******************************************************/

    virtual void
    FaceEquation_U(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                   dealii::Vector<double> &/*local_vector*/,
                   double /*scale*/,
                   double /*scale_ico*/);

    /******************************************************/

    virtual void
    StrongFaceResidual_U(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                         const FDC<DH, VECTOR, dealdim> & /*fdc_weight*/,
                         double & /*ret*/,
                         double /*scale*/);

    /******************************************************/

    virtual void
    FaceEquation_UT(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                    dealii::Vector<double> &/*local_vector*/,
                    double /*scale*/,
                    double /*scale_ico*/);

    /******************************************************/

    virtual void
    FaceEquation_UTT(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                     dealii::Vector<double> &/*local_vector*/,
                     double /*scale*/,
                     double /*scale_ico*/);

    /******************************************************/

    virtual void
    FaceEquation_Q(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                   dealii::Vector<double> &/*local_vector*/,
                   double /*scale*/,
                   double /*scale_ico*/);

    /******************************************************/

    virtual void
    FaceEquation_QT(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                    dealii::Vector<double> &/*local_vector*/,
                    double /*scale*/,
                    double /*scale_ico*/);

    /******************************************************/

    virtual void
    FaceEquation_QTT(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                     dealii::Vector<double> &/*local_vector*/,
                     double /*scale*/,
                     double /*scale_ico*/);

    /******************************************************/

    virtual void
    FaceEquation_UU(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                    dealii::Vector<double> &/*local_vector*/,
                    double /*scale*/,
                    double /*scale_ico*/);

    /******************************************************/

    virtual void
    FaceEquation_QU(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                    dealii::Vector<double> &/*local_vector*/,
                    double /*scale*/,
                    double /*scale_ico*/);

    /******************************************************/

    virtual void
    FaceEquation_UQ(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                    dealii::Vector<double> &/*local_vector*/,
                    double /*scale*/,
                    double /*scale_ico*/);

    /******************************************************/

    virtual void
    FaceEquation_QQ(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                    dealii::Vector<double> &/*local_vector*/,
                    double /*scale*/,
                    double /*scale_ico*/);

    /******************************************************/

    virtual void
    FaceRightHandSide(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                      dealii::Vector<double> &/*local_vector*/,
                      double /*scale*/);

    /******************************************************/

    /**
     * Documentation in optproblemcontainer.h.
     */
    virtual void
    FaceMatrix(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
               dealii::FullMatrix<double> &/*local_entry_matrix*/,
               double /*scale*/,
               double /*scale_ico*/);

    /******************************************************/

    virtual void
    FaceMatrix_T(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                 dealii::FullMatrix<double> &/*local_entry_matrix*/,
                 double /*scale*/,
                 double /*scale_ico*/);

    /******************************************************/
    //Functions for Interface Integrals

    virtual void
    InterfaceMatrix(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                    dealii::FullMatrix<double> &/*local_entry_matrix*/,
                    double /*scale*/,
                    double /*scale_ico*/);

    /******************************************************/
    //Functions for Interface Integrals
    virtual void
    InterfaceMatrix_T(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                      dealii::FullMatrix<double> &/*local_entry_matrix*/,
                      double /*scale*/,
                      double /*scale_ico*/);

    /******************************************************/
    // Integrals over interfaces (with test functions from
    // an adjacent (but not the same) element.
    // In optimization problems, at present, no control may
    // act in these and the state shoult only appear
    // linearly. Hence the derivatives UU, Q, ... are not
    // availiable for implementation.
    virtual void
    InterfaceEquation(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                      dealii::Vector<double> &/*local_vector*/,
                      double /*scale*/,
                      double /*scale_ico*/);

    /******************************************************/

    virtual void
    InterfaceEquation_U(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                        dealii::Vector<double> &/*local_vector*/,
                        double /*scale*/,
                        double /*scale_ico*/);
    /******************************************************/

    virtual void
    InterfaceEquation_UT(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                         dealii::Vector<double> &/*local_vector*/,
                         double /*scale*/,
                         double /*scale_ico*/);
    /******************************************************/

    virtual void
    InterfaceEquation_UTT(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                          dealii::Vector<double> &/*local_vector*/,
                          double /*scale*/,
                          double /*scale_ico*/);

    /******************************************************/

    // Functions for Boundary Integrals

    virtual void
    BoundaryEquation(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                     dealii::Vector<double> &/*local_vector*/,
                     double /*scale*/,
                     double /*scale_ico*/);

    /******************************************************/

    virtual void
    StrongBoundaryResidual(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                           const FDC<DH, VECTOR, dealdim> & /*fdc_weight*/,
                           double & /*ret*/,
                           double /*scale*/);

    /******************************************************/

    virtual void
    BoundaryEquation_U(const FDC<DH, VECTOR, dealdim> &/*fdc*/,
                       dealii::Vector<double> &/*local_vector*/,
                       double /*scale*/,
                       double /*scale_ico*/);

    /******************************************************/

    virtual void
    StrongBoundaryResidual_U(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                             const FDC<DH, VECTOR, dealdim> & /*fdc_weight*/,
                             double & /*ret*/,
                             double /*scale*/);

    /******************************************************/

    virtual void
    BoundaryEquation_UT(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                        dealii::Vector<double> &/*local_vector*/,
                        double /*scale*/,
                        double /*scale_ico*/);

    /******************************************************/

    virtual void
    BoundaryEquation_UTT(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                         dealii::Vector<double> &/*local_vector*/,
                         double /*scale*/,
                         double /*scale_ico*/);

    /******************************************************/

    virtual void
    BoundaryEquation_Q(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                       dealii::Vector<double> &/*local_vector*/,
                       double /*scale*/,
                       double /*scale_ico*/);

    /******************************************************/

    virtual void
    BoundaryEquation_QT(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                        dealii::Vector<double> &/*local_vector*/,
                        double /*scale*/,
                        double /*scale_ico*/);

    /******************************************************/

    virtual void
    BoundaryEquation_QTT(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                         dealii::Vector<double> &/*local_vector*/,
                         double /*scale*/,
                         double /*scale_ico*/);

    /******************************************************/

    virtual void
    BoundaryEquation_UU(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                        dealii::Vector<double> &/*local_vector*/,
                        double /*scale*/,
                        double /*scale_ico*/);

    /******************************************************/

    virtual void
    BoundaryEquation_QU(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                        dealii::Vector<double> &/*local_vector*/,
                        double /*scale*/,
                        double /*scale_ico*/);

    /******************************************************/

    virtual void
    BoundaryEquation_UQ(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                        dealii::Vector<double> &/*local_vector*/,
                        double /*scale*/,
                        double /*scale_ico*/);

    /******************************************************/

    virtual void
    BoundaryEquation_QQ(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                        dealii::Vector<double> &/*local_vector*/,
                        double /*scale*/,
                        double /*scale_ico*/);

    /******************************************************/

    virtual void
    BoundaryRightHandSide(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                          dealii::Vector<double> &/*local_vector*/,
                          double /*scale*/);

    /******************************************************/

    virtual void
    BoundaryMatrix(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                   dealii::FullMatrix<double> &/*local_entry_matrix*/,
                   double /*scale*/,
                   double /*scale_ico*/);

    /******************************************************/

    virtual void
    BoundaryMatrix_T(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                     dealii::FullMatrix<double> &/*local_entry_matrix*/,
                     double /*scale*/,
                     double /*scale_ico*/);

    /******************************************************/
    /******************************************************/
    /**
    * The following functions Init_... implement the
    * equation used for transfering the given initial
    * values to the finite element space.
    *
    * The initial data may depend on the control, but (obviously)
    * not on the state itself, hence derivatives with respect
    * to the control are considered.
    *
    * Default is componentwise L2 projection.
    *
    **/

    virtual void
    Init_ElementEquation(const EDC<DH, VECTOR, dealdim> &edc,
                         dealii::Vector<double> &local_vector,
                         double scale,
                         double /*scale_ico*/)
    {
      const DOpEWrapper::FEValues<dealdim> &state_fe_values =
        edc.GetFEValuesState();
      unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
      unsigned int n_q_points = edc.GetNQPoints();
      std::vector<dealii::Vector<double> > uvalues;
      uvalues.resize(n_q_points,
                     dealii::Vector<double>(this->GetStateNComponents()));
      edc.GetValuesState("last_newton_solution", uvalues);

      dealii::Vector<double> f_values(
        dealii::Vector<double>(this->GetStateNComponents()));

      for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          for (unsigned int i = 0; i < n_dofs_per_element; i++)
            {
              for (unsigned int comp = 0; comp < this->GetStateNComponents();
                   comp++)
                {
                  local_vector(i) += scale
                                     * (state_fe_values.shape_value_component(i, q_point, comp)
                                        * uvalues[q_point](comp))
                                     * state_fe_values.JxW(q_point);
                }
            }
        } //endfor q_point
    }

    virtual void
    Init_ElementRhs_Q(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                      dealii::Vector<double> &/*local_vector*/, double /*scale*/)
    {

    }
    virtual void
    Init_ElementRhs_QT(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                       dealii::Vector<double> &/*local_vector*/, double /*scale*/)
    {

    }
    virtual void
    Init_ElementRhs_QTT(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                        dealii::Vector<double> &/*local_vector*/, double /*scale*/)
    {

    }
    virtual void
    Init_ElementRhs_QQ(const EDC<DH, VECTOR, dealdim> & /*edc*/,
                       dealii::Vector<double> &/*local_vector*/, double /*scale*/)
    {

    }

    virtual void
    Init_ElementRhs(const dealii::Function<dealdim> *init_values,
                    const EDC<DH, VECTOR, dealdim> &edc,
                    dealii::Vector<double> &local_vector,
                    double scale)
    {
      const DOpEWrapper::FEValues<dealdim> &state_fe_values =
        edc.GetFEValuesState();
      unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
      unsigned int n_q_points = edc.GetNQPoints();

      dealii::Vector<double> f_values(
        dealii::Vector<double>(this->GetStateNComponents()));

      for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          init_values->vector_value(state_fe_values.quadrature_point(q_point),
                                    f_values);

          for (unsigned int i = 0; i < n_dofs_per_element; i++)
            {
              for (unsigned int comp = 0; comp < this->GetStateNComponents();
                   comp++)
                {
                  local_vector(i) += scale
                                     * (f_values(comp)
                                        * state_fe_values.shape_value_component(i, q_point,
                                            comp)) * state_fe_values.JxW(q_point);
                }
            }
        }
    }

    virtual void
    Init_ElementMatrix(const EDC<DH, VECTOR, dealdim> &edc,
                       dealii::FullMatrix<double> &local_entry_matrix,
                       double scale,
                       double /*scale_ico*/)
    {
      const DOpEWrapper::FEValues<dealdim> &state_fe_values =
        edc.GetFEValuesState();
      unsigned int n_dofs_per_element = edc.GetNDoFsPerElement();
      unsigned int n_q_points = edc.GetNQPoints();

      for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
        {
          for (unsigned int i = 0; i < n_dofs_per_element; i++)
            {
              for (unsigned int j = 0; j < n_dofs_per_element; j++)
                {
                  for (unsigned int comp = 0; comp < this->GetStateNComponents();
                       comp++)
                    {
                      local_entry_matrix(i, j) += scale
                                                  * state_fe_values.shape_value_component(i, q_point, comp)
                                                  * state_fe_values.shape_value_component(j, q_point, comp)
                                                  * state_fe_values.JxW(q_point);
                    }
                }
            }
        }
    }
    /*************************************************************/
    /**
     * Returns the update flags needed by the integrator to
     * decide which finite element informations need to
     * be calculated on the next element.
     */
    virtual dealii::UpdateFlags
    GetUpdateFlags() const;
    /**
     * Returns the update flags needed by the integrator to
     * decide which finite element informations need to
     * be calculated on the next face (including on boundaries).
     */
    virtual dealii::UpdateFlags
    GetFaceUpdateFlags() const;

    /**
     * Should return true, if integration on interior faces is
     * required; i.e., in dG implementations.
     *
     * The default is false.
     */
    virtual bool
    HasFaces() const;
    virtual bool
    HasInterfaces() const;

    /******************************************************/

    void
    SetProblemType(std::string type);

    /******************************************************/

    virtual unsigned int
    GetControlNBlocks() const;
    virtual unsigned int
    GetStateNBlocks() const;
    virtual std::vector<unsigned int> &
    GetControlBlockComponent();
    virtual const std::vector<unsigned int> &
    GetControlBlockComponent() const;
    virtual std::vector<unsigned int> &
    GetStateBlockComponent();
    virtual const std::vector<unsigned int> &
    GetStateBlockComponent() const;

    /******************************************************/

    void
    SetTime(double t, double step_size) const
    {
      time_ = t;
      step_size_ = step_size;
    }

    /******************************************************/

    unsigned int
    GetStateNComponents() const;

    /******************************************************/
    /**
     * This function is set by the error estimators in order
     * to allow the calculation of squared norms of the residual
     * as needed for Residual Error estimators as well as
     * the residual itself as needed by the DWR estimators.
     */
    boost::function1<void, double &> ResidualModifier;
    boost::function1<void, dealii::Vector<double>&> VectorResidualModifier;

    /**
           * Given a vector of active element iterators and a facenumber, checks if the face
           * belongs to an 'interface' (i.e. the adjoining elements have different material ids).
     *
     * Can be changed by the user to allow the use of dg-methods (i.e., then this function
     * needs to return true for all input arguments.
           *
           * @template ELEMENTITERATOR   Class of the elementiterator.
           *
           * @param   element            The element in question.
           * @param   face            Local number of the face for which we ask if it is
           *                          at the interface.
           */
    template<typename ELEMENTITERATOR>
    bool
    AtInterface(ELEMENTITERATOR &element, unsigned int face) const
    {
      if (element[0]->neighbor_index(face) != -1)
        if (element[0]->material_id()
            != element[0]->neighbor(face)->material_id())
          return true;
      return false;
    }

    /**
     * Function returning whether the PDE needs to set a special initial
     * value for the newton solver to work.
     * Relevant for stationary PDEs only when the constant zero is not a
     * feasible starting point.
     */
    virtual bool
    NeedInitialState() const
    {
      return false;
    }

    //Functions needed on networks, all default to abort, so that no implementation is required
    //  for normal PDEs
    /**
     * Implements the derivative of the BoundaryEquation with respect to
     * the given left or right values on the pipe.
     *
     * Only needed for calculations on networks.
     *
     */
    virtual void BoundaryEquation_BV(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                                     dealii::Vector<double> &/*local_vector*/,
                                     double /*scale*/,
                                     double /*scale_ico*/)
    {
      abort();
    }
    /**
     * Calculates the local matrix for the coupling between the unknowns and the
     * locals flux values
     *
     * Only needed for calculations on networks.
     *
     */
    virtual void
    BoundaryMatrix_BV(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                      std::vector<bool> & /*present_in_outflow*/,
                      dealii::FullMatrix<double> &/*local_entry_matrix*/,
                      double /*scale*/,
                      double /*scale_ico*/)
    {
      abort();
    }

    /**
     * Evaluates the difference between the outfolw values on the pipe; i.e. those that
     * do not take the given left or right value and the given left or right value.
     * E.g, it returns u-q_l if left boundary is and outflow boundary and u-q_r if right
     * boundary is an outflow boundary.
     *
     *
     * Only needed for calculations on networks.
     *
     * @param fdc          The container for the face information
     * @param local_vector The resulting computation
     *
     *
     */
    virtual void OutflowValues(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                               std::vector<bool> & /*present_in_outflow*/,
                               dealii::Vector<double> &/*local_vector*/,
                               double /*scale*/,
                               double /*scale_ico*/)
    {
      abort();
    }
    /**
     * The (local) matrix coupling the outflow values of the pde and the fluxes.
     *
     * Only needed for calculations on networks.
     *
     */
    virtual void
    OutflowMatrix(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                  std::vector<bool> & /*present_in_outflow*/,
                  dealii::FullMatrix<double> &/*local_entry_matrix*/,
                  double /*scale*/,
                  double /*scale_ico*/)
    {
      abort();
    }

    /**
     * Same as OutflowValues, but for initial conditions.
     *
     *
     * Only needed for calculations on networks.
     *
     * @param fdc          The container for the face information
     * @param local_vector The resulting computation
     *
     *
     */
    virtual void Init_OutflowValues(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                                    std::vector<bool> & /*present_in_outflow*/,
                                    dealii::Vector<double> &/*local_vector*/,
                                    double /*scale*/,
                                    double /*scale_ico*/)
    {
      abort();
    }
    /**
     * Same as OutflowMatrix, but for initial conditions.
     *
     * Only needed for calculations on networks.
     *
     */
    virtual void
    Init_OutflowMatrix(const FDC<DH, VECTOR, dealdim> & /*fdc*/,
                       std::vector<bool> & /*present_in_outflow*/,
                       dealii::FullMatrix<double> &/*local_entry_matrix*/,
                       double /*scale*/,
                       double /*scale_ico*/)
    {
      abort();
    }

    /**
     * Returns the global coupling residual between the individual pipes.
     *
     * Only needed on Networks.
     *
     *
     * @param res  The residual of the coupling condition
     * @param u    The vector in which the residual is to be calculated
     */
    virtual void PipeCouplingResidual(dealii::Vector<double> & /*res*/,
                                      const dealii::Vector<double> & /*u*/,
                                      const std::vector<bool> & /*present_in_outflow*/)
    {
      abort();
    }
    /**
     * Returns the matrix for the (linear) global couplings between the flux variables
     *
     * Only needed on Networks.
     *
     *
     * @param matrix  The matrix to be calculated
     * @param present_in_outflow A vector indicating which flux variables are outflow.
     */
    virtual void CouplingMatrix(dealii::SparseMatrix<double> & /*matrix*/,
                                const std::vector<bool> & /*present_in_outflow*/)
    {
      abort();
    }

    /**
     * Same as PipeCouplingResidual, but for the initial problem
     *
     * @param res  The residual of the coupling condition
     * @param u    The vector in which the residual is to be calculated
     */
    virtual void Init_PipeCouplingResidual(dealii::Vector<double> & /*res*/,
                                           const dealii::Vector<double> & /*u*/,
                                           const std::vector<bool> & /*present_in_outflow*/)
    {
      abort();
    }
    /**
     * Same as CouplingMatrix, but for the initial problem
     *
     * Only needed on Networks.
     *
     *
     * @param matrix  The matrix to be calculated
     * @param present_in_outflow A vector indicating which flux variables are outflow.
     */
    virtual void Init_CouplingMatrix(dealii::SparseMatrix<double> & /*matrix*/,
                                     const std::vector<bool> & /*present_in_outflow*/)
    {
      abort();
    }


  protected:
    std::string problem_type_;
    double GetTime() const
    {
      return time_;
    }
    double GetTimeStepSize() const
    {
      return step_size_;
    }

  private:
    mutable double time_;
    mutable double step_size_;
  };
}

#endif
