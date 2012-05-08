/*
 * DOpEtypes.h
 *
 *  Created on: Mar 13, 2012
 *      Author: cgoll
 */

#ifndef _DOPETYPES_H_
#define _DOPETYPES_H_

#include <string>

namespace DOpE
{
  namespace DOpEtypes
  {
    //TODO typedef for dealii-boundary-type
    // (see changes after version 7.1 in deal.ii).

    /**
     * This enum describes which terms of the error identity
     * should get computed:
     *
     * primal_only    Only the primal-residual-term.
     * dual_only      Only the dual-residual-term.
     * mixed          Compute both, scale them by 0.5
     *                 and build the sum.
     */
    enum EETerms
    {
      primal_only, dual_only, mixed
    };


    /**
     * This enum describes how we compute the weights in
     * the DWR-method, see for instance
     *
     * Bangerth, Rannacher: Adaptive Finite Element Methods
     * for Differential Equations
     *
     * for the explanation of the different states.
     */
    enum WeightComputation
    {
      higher_order_interpolation, higher_order_computation/*Not implemented!*/
    };


    /**
     * This enum describes how we evaluate the residual
     * in the DWR-method.
     *
     * strong_residual    We use the strong form of the residual.
     */
    enum ResidualEvaluation
    {
      strong_residual
    };

    /**
     * An enum describing the type of the control.
     */
    enum ControlType
    {
      undefined, stationary, initial, timedistributed_constant, timdistributed_timedependend
    };

  }
}

#endif /* DOPETYPES_H_ */
