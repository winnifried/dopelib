/*
 * DOpEtypes.h
 *
 *  Created on: Mar 13, 2012
 *      Author: cgoll
 */

#ifndef _DOPETYPES_H_
#define _DOPETYPES_H_

namespace DOpE
{
  namespace DOpEtypes
  {
    //TODO typedef for dealii-boundary-type
    // (see changes after version 7.1 in deal.ii).

    //TODO define an enum for ProblemTypes => use switch instead
    // of the awful if then else things.

    /**
     * This enum describes which terms of the error identity
     * should get computed:
     *
     * primal_only    Only the primal-residual-term.
     * dual_only      Only the dual-residual-term.
     * mixed          Compute both, scale them by 0.5
     *                 and build the sum.
     */
    enum EE_state
    {
      primal_only, dual_only, mixed
    };

    /**
     * This enum describes how we compute the adjoint solution
     * for the error estimation via the DWR-method
     */
    enum AdjointComputation
    {
      higher_order, normal
    };

    /**
     * This enum describes how we compute the weights in
     * the DWR-method, see for instance
     *
     * Bangerth, Rannacher: Adaptive Finite Element Mehtods
     * for Differential Equations
     *
     * for the explanation of the different states.
     */
    enum WeightComputation
    {
      higher_order_interpolation, higher_order_computation
    };
  }

}

#endif /* DOPETYPES_H_ */
