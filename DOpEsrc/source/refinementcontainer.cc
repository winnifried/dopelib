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

#include <container/refinementcontainer.h>
#include <include/dopeexception.h>
#include <cassert>

namespace DOpE
{

  /***Implementation of RefinementContainer*******************/

  RefinementContainer::RefinementContainer(DOpEtypes::RefinementType ref_type)
    : dummy_(0), ref_type_(ref_type)
  {
    if (ref_type == DOpEtypes::RefinementType::global
        || ref_type == DOpEtypes::RefinementType::finest_of_both)
      coarsening_ = false; //we know that in these cases no coarsening is performed
    else
      coarsening_ = true; // we do not know, coarsening might be a part of the strategy
  }

  /***********************************************************/

  const dealii::Vector<float> &
  RefinementContainer::GetLocalErrorIndicators() const
  {
    throw DOpEException("Not implemented",
                        "RefinementContainer::GetLocalErrorIndicators()");
    return dummy_;
  }

  /***********************************************************/

  double
  RefinementContainer::GetTopFraction() const
  {
    throw DOpEException("Not implemented",
                        "RefinementContainer::GetTopFraction()");
    return 1.0;
  }

  /***********************************************************/

  double
  RefinementContainer::GetBottomFraction() const
  {
    throw DOpEException("Not implemented",
                        "RefinementContainer::GetBottomFraction()");
    return 0.0;
  }

  /***********************************************************/

  double
  RefinementContainer::GetConvergenceOrder() const
  {
    throw DOpEException("Not implemented",
                        "RefinementContainer::GetConvergenceOrder");
    return 2.0;
  }

  /***********************************************************/

  DOpEtypes::RefinementType
  RefinementContainer::GetRefType() const
  {
    return ref_type_;
  }

  /***********************************************************/

  bool
  RefinementContainer::UsesCoarsening() const
  {
    return coarsening_;
  }

  /***********************************************************/
  /****Implementation of LocalRefinement**********************/
  /***********************************************************/

  LocalRefinement::LocalRefinement(const dealii::Vector<float> &indicators,
                                   DOpEtypes::RefinementType ref_type)
    : RefinementContainer(ref_type), indicators_(indicators)
  {
  }

  const dealii::Vector<float> &
  LocalRefinement::GetLocalErrorIndicators() const
  {
    return indicators_;
  }

  /***********************************************************/
  /****Implementation of RefineFixedFraction******************/
  /***********************************************************/

  RefineFixedFraction::RefineFixedFraction(
    const dealii::Vector<float> &indicators, double top_fraction,
    double bottom_fraction)
    : LocalRefinement(indicators, DOpEtypes::RefinementType::fixed_fraction), top_fraction_(
      top_fraction), bottom_fraction_(bottom_fraction)
  {
    assert(top_fraction_<=1. && top_fraction_>=0.);
    assert(bottom_fraction_<=1. && bottom_fraction_>=0.);

    if (bottom_fraction_ == 0.0)
      coarsening_ = false;
    else
      coarsening_ = true;
  }

  double
  RefineFixedFraction::GetTopFraction() const
  {
    return top_fraction_;
  }

  /***********************************************************/

  double
  RefineFixedFraction::GetBottomFraction() const
  {
    return bottom_fraction_;
  }

  /***********************************************************/
  /****Implementation of RefineFixedNumber********************/
  /***********************************************************/

  RefineFixedNumber::RefineFixedNumber(const dealii::Vector<float> &indicators,
                                       double top_fraction, double bottom_fraction)
    : LocalRefinement(indicators, DOpEtypes::RefinementType::fixed_number), top_fraction_(
      top_fraction), bottom_fraction_(bottom_fraction)
  {
    assert(top_fraction_<=1. && top_fraction_>=0.);
    assert(bottom_fraction_<=1. && bottom_fraction_>=0.);

    if (bottom_fraction_ == 0.0)
      coarsening_ = false;
    else
      coarsening_ = true;
  }

  double
  RefineFixedNumber::GetTopFraction() const
  {
    return top_fraction_;
  }

  /***********************************************************/

  double
  RefineFixedNumber::GetBottomFraction() const
  {
    return bottom_fraction_;
  }

  /***********************************************************/
  /****Implementation of RefineOptimized**********************/
  /***********************************************************/

  RefineOptimized::RefineOptimized(const dealii::Vector<float> &indicators,
                                   double convergence_order)
    : LocalRefinement(indicators, DOpEtypes::RefinementType::optimized), convergence_order_(
      convergence_order)
  {
    coarsening_ = false; //the method uses no coarsening.
  }

  /***********************************************************/

  double
  RefineOptimized::GetConvergenceOrder() const
  {
    return convergence_order_;
  }

}
