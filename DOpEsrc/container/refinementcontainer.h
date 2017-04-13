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

#ifndef REFINEMENTCONTAINER_H_
#define REFINEMENTCONTAINER_H_

#include <deal.II/lac/vector.h>
#include <basic/dopetypes.h>
#include <limits>

namespace DOpE
{
  /**
   * This base class represents a container which holds the necessary information
   * needed to (locally) refine a grid.
   * This class is used in the RefineSpace method of the MOL-space time handlers.
   * To use local meshrefinement, see the derived classes 'RefineFixedFraction',
   * 'RefineFixedNumber' and 'RefineOptimized'. To use global meshrefinement, one
   * can use this base class, as no special data is needed in the SpaceTimeHandler.
   *
   */
  class RefinementContainer
  {
  public:
    /**
     * Constructor if one wants to use a refinement which does
     * not need any special data apart from the given DOpEtypes::RefinementType
     * (like global refinement). If no DOpEtypes::RefinementType is given, global
     * mesh refinement is assumed.
     * */
    RefinementContainer(DOpEtypes::RefinementType ref_type =
                          DOpEtypes::RefinementType::global);
    virtual
    ~RefinementContainer()
    {
    }

    /**
     * Get functions, self explanatory. Implemented
     * in the derived classes.
     */
    virtual const dealii::Vector<float> &
    GetLocalErrorIndicators() const;
    virtual double
    GetTopFraction() const;
    virtual double
    GetBottomFraction() const;
    //     virtual unsigned int
    //GetMaxNElements() const;
    virtual double
    GetConvergenceOrder() const;

    /**
     * Returns the refinement type for which
     * the RefinementContainer object is constructed,
     * see dopetypes.h
     */
    DOpEtypes::RefinementType
    GetRefType() const;

    /**
     * Specifies if the mesh refinement uses coarsening.
     */
    bool
    UsesCoarsening() const;
  protected:
    bool coarsening_;
  private:
    const dealii::Vector<float> dummy_;
    const DOpEtypes::RefinementType ref_type_;

  };

  /***************************************************************/

  /**
   * Base class for RefinementContainer with local refinement. This
   * class holds the vector of error indicators.
   */
  class LocalRefinement : public RefinementContainer
  {
  public:
    virtual
    ~LocalRefinement()
    {
    }

    virtual const dealii::Vector<float> &
    GetLocalErrorIndicators() const;

  protected:
    /**
     * Protected constructor for use in the derived classes
     */
    LocalRefinement(const dealii::Vector<float> &,
                    DOpEtypes::RefinementType ref_type);
  private:
    /**
     * Constructor made private. Should not get used!
     */
    LocalRefinement();
    const dealii::Vector<float> &indicators_;
  };

  /***************************************************************/

  /**
   * This class holds the information needed for local mesh refinement
   * with the fixed fraction strategy.
   */
  class RefineFixedFraction : public LocalRefinement
  {
  public:
    /**
     * Constructor if one wants to use local refinement with the
     * fixed fraction strategy.
     *
     * @param indicators        A set of positive values, used to guide refinement.
     * @param topfraction       is the fraction of the total estimate which should be refined.
     * @param bottomfraction    is the fraction of the estimate coarsened.
     */
    RefineFixedFraction(const dealii::Vector<float> &indicators,
                        double top_fraction = 0.1, double bottom_fraction = 0.0);

    virtual
    ~ RefineFixedFraction()
    {
    }

    virtual double
    GetTopFraction() const;
    virtual double
    GetBottomFraction() const;
  private:
    const double top_fraction_, bottom_fraction_;
  };

  /***************************************************************/

  /**
   * This class holds the information needed for local mesh refinement
   * with the fixed number strategy.
   */
  class RefineFixedNumber : public LocalRefinement
  {
  public:
    /**
     * Constructor if one wants to use local refinement with the
     * fixed number strategy. This leads to local mesh refinement
     * with a predictable growth of the mesh.
     *
     * @param indicators        A set of positive values, used to guide refinement.
     * @param topfraction        is the fraction of elements to be refined.
     *
     * @param bottomfraction    In a fixed fraction/fixed number strategy,
     *                          wich part should be coarsened.
     */
    RefineFixedNumber(const dealii::Vector<float> &indicators,
                      double top_fraction = 0.1, double bottom_fraction = 0.0);
    virtual
    ~ RefineFixedNumber()
    {
    }

    virtual double
    GetTopFraction() const;
    virtual double
    GetBottomFraction() const;

  private:
    const double top_fraction_, bottom_fraction_;
  };

  /***************************************************************/

  /**
   * This class holds the information needed for local mesh refinement
   * with the optimized refinement strategy.
   */

  class RefineOptimized : public LocalRefinement
  {
  public:
    /**
     * Constructor if one wants to use the optimized refinement strategy.
     *
     * @param indicators        A set of positive values, used to guide refinement.
     * @param convergence_order Convergence order of the functional of interest.
     */
    RefineOptimized(const dealii::Vector<float> &indicators,
                    double convergence_order = 2.);

    virtual
    ~ RefineOptimized()
    {
    }

    virtual double
    GetConvergenceOrder() const;

  private:
    const double convergence_order_;
  };

}

#endif /* RefinementContainer_H_ */
