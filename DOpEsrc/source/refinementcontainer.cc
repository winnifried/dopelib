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
/*
 * refinementcontainer.cc
 *
 *  Created on: Feb 7, 2013
 *      Author: cgoll
 */

#include "refinementcontainer.h"
#include "dopeexception.h"
#include <cassert>

namespace DOpE
{

  /***Implementation of RefinementContainer*******************/

  RefinementContainer::RefinementContainer(DOpEtypes::RefinementType ref_type)
      : _dummy(0), _ref_type(ref_type)
  {
    if (ref_type == DOpEtypes::RefinementType::global
        || ref_type == DOpEtypes::RefinementType::finest_of_both)
      _coarsening = false; //we know that in these cases no coarsening is performed
    else
      _coarsening = true; // we do not know, coarsening might be a part of the strategy
  }

  /***********************************************************/

  const dealii::Vector<float>&
  RefinementContainer::GetLocalErrorIndicators() const
  {
    throw DOpEException("Not implemented",
        "RefinementContainer::GetLocalErrorIndicators()");
    return _dummy;
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

  unsigned int
  RefinementContainer::GetMaxNCells() const
  {
    throw DOpEException("Not implemented", "RefinementContainer::GetMaxNCells");
    return std::numeric_limits<unsigned int>::max();
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
    return _ref_type;
  }

  /***********************************************************/

  bool
  RefinementContainer::UsesCoarsening() const
  {
    return _coarsening;
  }

  /***********************************************************/
  /****Implementation of LocalRefinement**********************/
  /***********************************************************/

  LocalRefinement::LocalRefinement(const dealii::Vector<float>& indicators,
      DOpEtypes::RefinementType ref_type)
      : RefinementContainer(ref_type), _indicators(indicators)
  {
  }

  const dealii::Vector<float>&
  LocalRefinement::GetLocalErrorIndicators() const
  {
    return _indicators;
  }

  /***********************************************************/
  /****Implementation of RefineFixedFraction******************/
  /***********************************************************/

  RefineFixedFraction::RefineFixedFraction(
      const dealii::Vector<float>& indicators, double top_fraction,
      double bottom_fraction, const unsigned int max_n_cells)
      : LocalRefinement(indicators, DOpEtypes::RefinementType::fixed_fraction), _top_fraction(
          top_fraction), _bottom_fraction(bottom_fraction), _max_n_cells(
          max_n_cells)
  {
    assert(_top_fraction<=1. && _top_fraction>=0.);
    assert(_bottom_fraction<=1. && _bottom_fraction>=0.);

    if (_bottom_fraction == 0.0)
      _coarsening = false;
    else
      _coarsening = true;
  }

  double
  RefineFixedFraction::GetTopFraction() const
  {
    return _top_fraction;
  }

  /***********************************************************/

  double
  RefineFixedFraction::GetBottomFraction() const
  {
    return _bottom_fraction;
  }

  /***********************************************************/
  /****Implementation of RefineFixedNumber********************/
  /***********************************************************/

  RefineFixedNumber::RefineFixedNumber(const dealii::Vector<float>& indicators,
      double top_fraction, double bottom_fraction,
      const unsigned int max_n_cells)
      : LocalRefinement(indicators, DOpEtypes::RefinementType::fixed_number), _top_fraction(
          top_fraction), _bottom_fraction(bottom_fraction), _max_n_cells(
          max_n_cells)
  {
    assert(_top_fraction<=1. && _top_fraction>=0.);
    assert(_bottom_fraction<=1. && _bottom_fraction>=0.);

    if (_bottom_fraction == 0.0)
      _coarsening = false;
    else
      _coarsening = true;
  }

  double
  RefineFixedNumber::GetTopFraction() const
  {
    return _top_fraction;
  }

  /***********************************************************/

  double
  RefineFixedNumber::GetBottomFraction() const
  {
    return _bottom_fraction;
  }

  /***********************************************************/
  /****Implementation of RefineOptimized**********************/
  /***********************************************************/

  RefineOptimized::RefineOptimized(const dealii::Vector<float>& indicators,
      double convergence_order)
      : LocalRefinement(indicators, DOpEtypes::RefinementType::optimized), _convergence_order(
          convergence_order)
  {
    _coarsening = false; //the method uses no coarsening.
  }

  /***********************************************************/

  double
  RefineOptimized::GetConvergenceOrder() const
  {
    return _convergence_order;
  }

}
