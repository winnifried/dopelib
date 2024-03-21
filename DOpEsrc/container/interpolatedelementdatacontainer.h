/**
 *
 * Copyright (C) 2012-2018 by the DOpElib authors
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

#ifndef INTERPOLATEDELEMENTDATACONTAINER_H_
#define INTERPOLATEDELEMENTDATACONTAINER_H_

#include <container/elementdatacontainer.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <include/interpolated_fe_values.h>

using namespace dealii;
using namespace std;

namespace DOpE
{
#if DEAL_II_VERSION_GTE(9,3,0)
  template<bool DH, typename VECTOR, int dim>
#else
  template<template<int, int> class DH, typename VECTOR, int dim>
#endif
  class InterpolatedElementDataContainer : public ElementDataContainer<DH, VECTOR, dim>
  {
  public :
    /**
     * Constructor. Initializes the FEValues objects.
     *
     * @template FE                   Type of the finite element in use. Must be compatible with dealii::DofHandler. //TODO Should we fix this?
     * @template SPARSITYPATTERN      The corresponding Sparsitypattern to the class-template VECTOR.
     * @template dopedim              The dimension of the control variable.
     * @template dealdim              The dimension of the state variable.
     *
     * @param selected_component      A FEValuesExtractor::Vector indicating which
     *                                components of FE need to be interpolated
     * @param map                     The mapping for the target element of the interpolation
     * @param fe_interpolate          The finite element to which the interpolation should go
     * @param quad                    Reference to the quadrature-rule which we use at the moment.
     * @param update_flags            The update flags we need to initialize the FEValues obejcts
     * @param sth                     A reference to the SpaceTimeHandler in use.
     * @param element                    A vector of element iterators through which we gain most of the needed information (like
     *                                material_ids, n_dfos, etc.)
     * @param param_values            A std::map containing parameter data (e.g. non space dependent data). If the control
     *                                is done by parameters, it is contained in this map at the position "control".
     * @param domain_values           A std::map containing domain data (e.g. nodal vectors for FE-Functions). If the control
     *                                is distributed, it is contained in this map at the position "control". The state may always
     *                                be found in this map at the position "state"
     * @param need_vertices           A flag indicating if vertex information needs to be prepared
     *
     */
    template<template<int, int> class FE, typename SPARSITYPATTERN, int dopedim, int dealdim>
    InterpolatedElementDataContainer(
      const FEValuesExtractors::Vector selected_component,
      const Mapping<dim>   &map,
      const FiniteElement<dim> &fe_interpolate,
      const Quadrature<dim> &quad, UpdateFlags update_flags,
#if DEAL_II_VERSION_GTE(9,3,0)
      SpaceTimeHandler<FE, false, SPARSITYPATTERN, VECTOR,
#else
      SpaceTimeHandler<FE, dealii::DoFHandler, SPARSITYPATTERN, VECTOR,
#endif
      dopedim, dealdim>& sth,
      const std::vector<
      typename dealii::DoFHandler<dim>::active_cell_iterator>& element,
      const std::map<std::string, const Vector<double>*> &param_values,
      const std::map<std::string, const VECTOR *> &domain_values,
      bool need_vertices) :
      ElementDataContainer<DH, VECTOR, dim>( quad, update_flags, sth, element,
                                             param_values, domain_values, need_vertices),
      interpolated_fe_values_(selected_component, map, sth.GetFESystem("state"),
                              fe_interpolate, quad, update_flags), element_(element)

    {
    }
    /**
     * Constructor. Initializes the FEValues objects, when only the PDE is used.
     *
     * @template FE                   Type of the finite element in use. Must be compatible with dealii::DofHandler. //TODO Should we fix this?
     * @template SPARSITYPATTERN      The corresponding Sparsitypattern to the class-template VECTOR.
     * @template dopedim              The dimension of the control variable.
     * @template dealdim              The dimension of the state variable.
     *
     * @param selected_component      A FEValuesExtractor::Vector indicating which
     *                                components of FE need to be interpolated
     * @param map                     The mapping for the target element of the interpolation
     * @param fe_interpolate          The finite element to which the interpolation should go
     * @param quad                    Reference to the quadrature-rule which we use at the moment.
     * @param update_flags            The update flags we need to initialize the FEValues obejcts
     * @param sth                     A reference to the SpaceTimeHandler in use.
     * @param element                    A vector of element iterators through which we gain most of the needed information (like
     *                                material_ids, n_dfos, etc.)
     * @param param_values            A std::map containing parameter data (e.g. non space dependent data). If the control
     *                                is done by parameters, it is contained in this map at the position "control".
     * @param domain_values           A std::map containing domain data (e.g. nodal vectors for FE-Functions). If the control
     *                                is distributed, it is contained in this map at the position "control". The state may always
     *                                be found in this map at the position "state"
     * @param need_vertices           A flag indicating if vertex information needs to be prepared
     *
     */
    template<template<int, int> class FE, typename SPARSITYPATTERN>
    InterpolatedElementDataContainer(
      const FEValuesExtractors::Vector selected_component,
      const Mapping<dim>    &map,
      const FiniteElement<dim>  &fe_interpolate,
      const Quadrature<dim> &quad, UpdateFlags update_flags,
#if DEAL_II_VERSION_GTE(9,3,0)
      StateSpaceTimeHandler<FE, false, SPARSITYPATTERN,
#else
      StateSpaceTimeHandler<FE, dealii::DoFHandler, SPARSITYPATTERN,
#endif
      VECTOR, dim> &sth,
      const std::vector<
      typename dealii::DoFHandler<dim>::active_cell_iterator> &element,
      const std::map<std::string, const Vector<double>*> &param_values,
      const std::map<std::string, const VECTOR *> &domain_values,
      bool need_vertices) :
      ElementDataContainer<DH, VECTOR, dim>(quad, update_flags, sth, element, param_values, domain_values,
                                            need_vertices),
      interpolated_fe_values_(selected_component, map, sth.GetFESystem("state"),
                              fe_interpolate, quad, update_flags), element_(element)
    {
    }

    void ReInit()
    {
      ElementDataContainer<DH, VECTOR, dim>::ReInit();
      interpolated_fe_values_.reinit(element_[ElementDataContainer<DH, VECTOR, dim>::GetStateIndex()]);
    }

    InterpolatedFEValues<dim>
    GetInterpolatedFEValuesState() const
    {
      return interpolated_fe_values_;
    }

  private:
    InterpolatedFEValues<dim> interpolated_fe_values_;
    const std::vector<typename dealii::DoFHandler<dim>::active_cell_iterator> &element_;

  };
}





#endif

