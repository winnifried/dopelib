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

#ifndef ELEMENTDATACONTAINER_H_
#define ELEMENTDATACONTAINER_H_

#include <basic/spacetimehandler.h>
#include <basic/statespacetimehandler.h>
#include <wrapper/fevalues_wrapper.h>
#include <include/dopeexception.h>
#include <container/elementdatacontainer_internal.h>

#include <sstream>

#include <deal.II/dofs/dof_handler.h>
#if ! DEAL_II_VERSION_GTE(9,3,0)
#include <deal.II/hp/dof_handler.h>
#endif

using namespace dealii;

namespace DOpE
{
#if DEAL_II_VERSION_GTE(9,3,0)
  /**
   * Dummy Template Class, acts as kind of interface.
   * Through template specialization for DH, we
   * distinguish between the 'classic' and the 'hp' case.
   *
   * @template DH         false for normal DoFHandler true for HP.
   * @template VECTOR     Type of the vector we use in our computations
   *                      (i.e. Vector<double> or BlockVector<double>)
   * @template dim        The dimension of the integral we are actually
   *                      interested in.
   */

  template<bool DH, typename VECTOR, int dim>
#else
  /**
   * Dummy Template Class, acts as kind of interface.
   * Through template specialization for DH, we
   * distinguish between the 'classic' and the 'hp' case.
   *
   * @template DH The type of the dealii-dofhandler we use in
   *                      our DOpEWrapper::DoFHandler, at the moment
   *                      DoFHandler and hp::DoFHandler.
   * @template VECTOR     Type of the vector we use in our computations
   *                      (i.e. Vector<double> or BlockVector<double>)
   * @template dim        The dimension of the integral we are actually
   *                      interested in.
   */

  template<template<int, int> class DH, typename VECTOR, int dim>
#endif
  class ElementDataContainer : public edcinternal::ElementDataContainerInternal<
    VECTOR, dim>
  {
  public:
    ElementDataContainer()
    {
      throw (DOpE::DOpEException(
               "Dummy class, this constructor should never get called.",
               "ElementDataContainer<DH , VECTOR, dim>::ElementDataContainer"));
    }
  };

  /**
   * This two classes hold all the information we need in the integrator to
   * integrate something over a element (could be a functional, a PDE, etc.).
   * Of particular importance: This class holds the FEValues objects.
   *
   * @template VECTOR     Type of the vector we use in our computations (i.e. Vector<double> or BlockVector<double>)
   * @template dim        The dimension of the integral we are actually interested in.
   */

  template<typename VECTOR, int dim>
#if DEAL_II_VERSION_GTE(9,3,0)
  class ElementDataContainer<false, VECTOR, dim> : public edcinternal::ElementDataContainerInternal<VECTOR, dim>
#else
  class ElementDataContainer<dealii::DoFHandler, VECTOR, dim> : public edcinternal::ElementDataContainerInternal<VECTOR, dim>
#endif
  {

  public:
    /**
     * Constructor. Initializes the FEValues objects.
     *
     * @template FE                   Type of the finite element in use. Must be compatible with dealii::DofHandler. //TODO Should we fix this?
     * @template SPARSITYPATTERN      The corresponding Sparsitypattern to the class-template VECTOR.
     * @template dopedim              The dimension of the control variable.
     * @template dealdim              The dimension of the state variable.
     *
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
    ElementDataContainer(const Quadrature<dim> &quad,
                         UpdateFlags update_flags,
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
      edcinternal::ElementDataContainerInternal<VECTOR, dim>(param_values,
                                                             domain_values,
                                                             sth, element[0], need_vertices),
      element_(element),
      state_fe_values_(sth.GetMapping(), (sth.GetFESystem("state")), quad, update_flags),
      control_fe_values_(sth.GetMapping(),(sth.GetFESystem("control")), quad, update_flags)
    {
      state_index_ = sth.GetStateIndex();
      if (state_index_ == 1)
        control_index_ = 0;
      else
        control_index_ = 1;
      n_q_points_per_element_ = quad.size();
      n_dofs_per_element_ = element[0]->get_fe().dofs_per_cell;
    }

    /**
     * Constructor. Initializes the FEValues objects. When only a PDE is used.
     *
     * @template FE                   Type of the finite element in use.
     * @template SPARSITYPATTERN      The corresponding Sparsitypattern to the class-template VECTOR.
     *
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
     *
     */
    template<template<int, int> class FE, typename SPARSITYPATTERN>
    ElementDataContainer(const Quadrature<dim> &quad,
                         UpdateFlags update_flags,
#if DEAL_II_VERSION_GTE(9,3,0)
                         StateSpaceTimeHandler<FE, false, SPARSITYPATTERN,
#else
                         StateSpaceTimeHandler<FE, dealii::DoFHandler, SPARSITYPATTERN,
#endif
                         VECTOR, dim>& sth,
                         const std::vector<
                         typename dealii::DoFHandler<dim>::active_cell_iterator>& element,
                         const std::map<std::string, const Vector<double>*> &param_values,
                         const std::map<std::string, const VECTOR *> &domain_values,
                         bool need_vertices) :
      edcinternal::ElementDataContainerInternal<VECTOR, dim>(param_values,
                                                             domain_values,
                                                             sth, element[0], need_vertices),
      element_(element),
      state_fe_values_(sth.GetMapping(), (sth.GetFESystem("state")), quad, update_flags),
      control_fe_values_(sth.GetMapping(),(sth.GetFESystem("state")), quad, update_flags)
    {
      state_index_ = sth.GetStateIndex();
      control_index_ = element.size(); //Make sure they are never used ...
      n_q_points_per_element_ = quad.size();
      n_dofs_per_element_ = element[0]->get_fe().dofs_per_cell;
    }
    ~ElementDataContainer()
    {
    }
    /*********************************************/
    /*
     * This function reinits the FEValues on the actual element. Should
     * be called prior to any of the get-functions.
     */
    inline void
    ReInit();

    /*********************************************/
    /**
     * Get functions to extract data. They all assume that ReInit
     * is executed before calling them. Self explanatory.
     */
    inline unsigned int
    GetNDoFsPerElement() const;
    inline unsigned int
    GetNQPoints() const;
    inline unsigned int
    GetMaterialId() const;
    inline unsigned int
    GetNbrMaterialId(unsigned int face) const;
    inline unsigned int
    GetFaceBoundaryIndicator(unsigned int face) const;
    inline bool
    GetIsAtBoundary() const;
    inline double
    GetElementDiameter() const;
    inline Point<dim>
    GetCenter() const;
    inline const DOpEWrapper::FEValues<dim> &
    GetFEValuesState() const override;
    inline const DOpEWrapper::FEValues<dim> &
    GetFEValuesControl() const override;
  protected:
    /*
     * Helper Functions
     */
    unsigned int
    GetStateIndex() const;
    unsigned int
    GetControlIndex() const;

  private:

    /***********************************************************/
    //"global" member data, part of every instantiation
    unsigned int state_index_;
    unsigned int control_index_;

    const std::vector<typename dealii::DoFHandler<dim>::active_cell_iterator> &element_;
    DOpEWrapper::FEValues<dim> state_fe_values_;
    DOpEWrapper::FEValues<dim> control_fe_values_;

    unsigned int n_q_points_per_element_;
    unsigned int n_dofs_per_element_;
  };


  template<typename VECTOR, int dim>
#if DEAL_II_VERSION_GTE(9,3,0)
  class ElementDataContainer<true, VECTOR, dim> : public edcinternal::ElementDataContainerInternal<VECTOR, dim>
#else
  class ElementDataContainer<dealii::hp::DoFHandler, VECTOR, dim> : public edcinternal::ElementDataContainerInternal<VECTOR, dim>
#endif
  {

  public:
    /**
     * Constructor. Initializes the hp_fe_falues objects.
     *
     * @template FE                   Type of the finite element in use. Must be compatible with hp::DoFHandler. //TODO Should we fix this?
     * @template SPARSITYPATTERN      The corresponding Sparsitypattern to the class-template VECTOR.
     * @template dopedim              The dimension of the control variable.
     * @template dealdim              The dimension of the state variable.
     *
     * @param quad                    Reference to the qcollection-rule which we use at the moment.
     * @param update_flags            The update flags we need to initialize the FEValues obejcts
     * @param sth                     A reference to the SpaceTimeHandler in use.
     * @param element                    A vector of element iterators through which we gain most of the needed information (like
     *                                material_ids, n_dfos, etc.)
     * @param param_values            A std::map containing parameter data (e.g. non space dependent data). If the control
     *                                is done by parameters, it is contained in this map at the position "control".
     * @param domain_values           A std::map containing domain data (e.g. nodal vectors for FE-Functions). If the control
     *                                is distributed, it is contained in this map at the position "control". The state may always
     *                                be found in this map at the position "state"
     *
     */
    template<template<int, int> class FE, typename SPARSITYPATTERN, int dopedim, int dealdim>
    ElementDataContainer(const hp::QCollection<dim> &q_collection,
                         UpdateFlags update_flags,
#if DEAL_II_VERSION_GTE(9,3,0)
                         SpaceTimeHandler<FE, true, SPARSITYPATTERN,
#else
                         SpaceTimeHandler<FE, dealii::hp::DoFHandler, SPARSITYPATTERN,
#endif
                         VECTOR, dopedim, dealdim>& sth,
                         const std::vector<
#if DEAL_II_VERSION_GTE(9,3,0)
                         typename DOpEWrapper::DoFHandler<dim>::active_cell_iterator>& element,
#else
                         typename DOpEWrapper::DoFHandler<dim, dealii::hp::DoFHandler>::active_cell_iterator>& element,
#endif
                         const std::map<std::string, const Vector<double>*> &param_values,
                         const std::map<std::string, const VECTOR *> &domain_values,
                         bool need_vertices) :
      edcinternal::ElementDataContainerInternal<VECTOR, dim>(param_values,
                                                             domain_values,
                                                             sth, element[0], need_vertices),
      element_(element),
      state_hp_fe_values_(sth.GetMapping(), (sth.GetFESystem("state")), q_collection, update_flags),
      control_hp_fe_values_(sth.GetMapping(), (sth.GetFESystem("control")), q_collection, update_flags), q_collection_(q_collection)
    {
      state_index_ = sth.GetStateIndex();
      if (state_index_ == 1)
        control_index_ = 0;
      else
        control_index_ = 1;
    }

    /**
     * Constructor. Initializes the hp_fe_falues objects. For PDE only.
     *
     * @template SPARSITYPATTERN      The corresponding Sparsitypattern to the class-template VECTOR.
     *
     * @param quad                    Reference to the qcollection-rule which we use at the moment.
     * @param update_flags            The update flags we need to initialize the FEValues obejcts
     * @param sth                     A reference to the SpaceTimeHandler in use.
     * @param element                    A vector of element iterators through which we gain most of the needed information (like
     *                                material_ids, n_dfos, etc.)
     * @param param_values            A std::map containing parameter data (e.g. non space dependent data). If the control
     *                                is done by parameters, it is contained in this map at the position "control".
     * @param domain_values           A std::map containing domain data (e.g. nodal vectors for FE-Functions). If the control
     *                                is distributed, it is contained in this map at the position "control". The state may always
     *                                be found in this map at the position "state"
     *
     */
    template<template<int, int> class FE, typename SPARSITYPATTERN>
    ElementDataContainer(const hp::QCollection<dim> &q_collection,
                         UpdateFlags update_flags,
#if DEAL_II_VERSION_GTE(9,3,0)
                         StateSpaceTimeHandler<FE, true, SPARSITYPATTERN,
#else
                         StateSpaceTimeHandler<FE, dealii::hp::DoFHandler, SPARSITYPATTERN,
#endif
                         VECTOR, dim>& sth,
                         const std::vector<
#if DEAL_II_VERSION_GTE(9,3,0)
                         typename DOpEWrapper::DoFHandler<dim>::active_cell_iterator>& element,
#else
                         typename DOpEWrapper::DoFHandler<dim, dealii::hp::DoFHandler>::active_cell_iterator>& element,
#endif
                         const std::map<std::string, const Vector<double>*> &param_values,
                         const std::map<std::string, const VECTOR *> &domain_values,
                         bool need_vertices) :
      edcinternal::ElementDataContainerInternal<VECTOR, dim>(param_values,
                                                             domain_values,
                                                             sth, element[0], need_vertices),
      element_(element),
      state_hp_fe_values_(sth.GetMapping(), (sth.GetFESystem("state")), q_collection, update_flags),
      control_hp_fe_values_(sth.GetMapping(),(sth.GetFESystem("state")), q_collection, update_flags), q_collection_(q_collection)
    {
      state_index_ = sth.GetStateIndex();
      control_index_ = element.size(); //Make sure they are never used ...
    }
    ~ElementDataContainer()
    {
    }

    /*****************************************************************/
    /*
     * This function reinits the hp::FEValues on the actual element. Should
     * be called prior to any of the get-functions.
     */
    inline void
    ReInit();
    /*********************************************/
    /**
     * Get functions to extract data. They all assume that ReInit
     * is executed before calling them. Self explanatory.
     */

    inline unsigned int
    GetNDoFsPerElement() const;
    inline unsigned int
    GetNQPoints() const;
    inline unsigned int
    GetMaterialId() const;
    inline unsigned int
    GetNbrMaterialId(unsigned int face) const;
    inline unsigned int
    GetFaceBoundaryIndicator(unsigned int face) const;
    inline bool
    GetIsAtBoundary() const;
    inline double
    GetElementDiameter() const;
    inline Point<dim>
    GetCenter() const;

    inline const DOpEWrapper::FEValues<dim> &
    GetFEValuesState() const override;
    inline const DOpEWrapper::FEValues<dim> &
    GetFEValuesControl() const override;

  private:
    unsigned int
    GetStateIndex() const;
    unsigned int
    GetControlIndex() const;
    const std::map<std::string, const VECTOR *> &
    GetDomainValues() const;
    /***********************************************************/
    /**
     * Helper Function.
     * Hier koennte man ueber ein Template nachdenken.
     */
    inline void
    GetValues(const DOpEWrapper::FEValues<dim> &fe_values, std::string name,
              std::vector<double> &values) const;
    /***********************************************************/
    /**
     * Helper Function. Vector valued case.
     */
    inline void
    GetValues(const DOpEWrapper::FEValues<dim> &fe_values, std::string name,
              std::vector<Vector<double> > &values) const;

    /***********************************************************/
    /**
     * Helper Function.
     */
    template<int targetdim>
    inline void
    GetGrads(const DOpEWrapper::FEValues<dim> &fe_values,
             std::string name,
             std::vector<Tensor<1, targetdim> > &values) const;
    /***********************************************************/
    /**
     * Helper Function. Vector valued case.
     */
    template<int targetdim>
    inline void
    GetGrads(const DOpEWrapper::FEValues<dim> &fe_values,
             std::string name,
             std::vector<std::vector<Tensor<1, targetdim> > > &values) const;

    /***********************************************************/
    //"global" member data, part of every instantiation
    unsigned int state_index_;
    unsigned int control_index_;

#if DEAL_II_VERSION_GTE(9,3,0)
    const std::vector<typename dealii::DoFHandler<dim>::active_cell_iterator> &element_;
#else
    const std::vector<typename dealii::hp::DoFHandler<dim>::active_cell_iterator> &element_;
#endif
    DOpEWrapper::HpFEValues<dim> state_hp_fe_values_;
    DOpEWrapper::HpFEValues<dim> control_hp_fe_values_;

    const hp::QCollection<dim> &q_collection_;
  };

  /***********************************************************************/
  /************************IMPLEMENTATION for DoFHandler*********************************/
  /***********************************************************************/

  template<typename VECTOR, int dim>
  void
#if DEAL_II_VERSION_GTE(9,3,0)
  DOpE::ElementDataContainer<false, VECTOR, dim>::ReInit()
#else
  DOpE::ElementDataContainer<dealii::DoFHandler, VECTOR, dim>::ReInit()
#endif
  {
    state_fe_values_.reinit(element_[this->GetStateIndex()]);
    //Make sure that the Control must be initialized.
    if (this->GetControlIndex() < element_.size())
      control_fe_values_.reinit(element_[this->GetControlIndex()]);

    edcinternal::ElementDataContainerInternal<
    VECTOR, dim>::ReInit(element_[this->GetStateIndex()]);
  }

  /***********************************************************************/
  template<typename VECTOR, int dim>
  unsigned int
#if DEAL_II_VERSION_GTE(9,3,0)
  ElementDataContainer<false, VECTOR, dim>::GetNDoFsPerElement() const
#else
  ElementDataContainer<dealii::DoFHandler, VECTOR, dim>::GetNDoFsPerElement() const
#endif
  {
    return n_dofs_per_element_;
  }

  /**********************************************/
  template<typename VECTOR, int dim>
  unsigned int
#if DEAL_II_VERSION_GTE(9,3,0)
  ElementDataContainer<false, VECTOR, dim>::GetNQPoints() const
#else
  ElementDataContainer<dealii::DoFHandler, VECTOR, dim>::GetNQPoints() const
#endif
  {
    return n_q_points_per_element_;
  }

  /**********************************************/
  template<typename VECTOR, int dim>
  unsigned int
#if DEAL_II_VERSION_GTE(9,3,0)
  ElementDataContainer<false, VECTOR, dim>::GetMaterialId() const
#else
  ElementDataContainer<dealii::DoFHandler, VECTOR, dim>::GetMaterialId() const
#endif
  {
    return element_[0]->material_id();
  }

  /**********************************************/
  template<typename VECTOR, int dim>
  unsigned int
#if DEAL_II_VERSION_GTE(9,3,0)
  ElementDataContainer<false, VECTOR, dim>::GetNbrMaterialId(
    unsigned int face) const
#else
  ElementDataContainer<dealii::DoFHandler, VECTOR, dim>::GetNbrMaterialId(
    unsigned int face) const
#endif
  {
    if (element_[0]->neighbor_index(face) != -1)
      return element_[0]->neighbor(face)->material_id();
    else
      {
        std::stringstream out;
        out << "There is no neighbor with number " << face;
        throw DOpEException(out.str(),
                            "ElementDataContainer::GetNbrMaterialId");
      }
  }

  /**********************************************/
  template<typename VECTOR, int dim>
  unsigned int
#if DEAL_II_VERSION_GTE(9,3,0)
  ElementDataContainer<false, VECTOR, dim>::GetFaceBoundaryIndicator(
    unsigned int face) const
#else
  ElementDataContainer<dealii::DoFHandler, VECTOR, dim>::GetFaceBoundaryIndicator(
    unsigned int face) const
#endif
  {
    return element_[0]->face(face)->boundary_indicator();
  }

  /**********************************************/
  template<typename VECTOR, int dim>
  bool
#if DEAL_II_VERSION_GTE(9,3,0)
  ElementDataContainer<false, VECTOR, dim>::GetIsAtBoundary() const
#else
  ElementDataContainer<dealii::DoFHandler, VECTOR, dim>::GetIsAtBoundary() const
#endif
  {
    return element_[0]->at_boundary();
  }
  /**********************************************/
  template<typename VECTOR, int dim>
  double
#if DEAL_II_VERSION_GTE(9,3,0)
  ElementDataContainer<false, VECTOR, dim>::GetElementDiameter() const
#else
  ElementDataContainer<dealii::DoFHandler, VECTOR, dim>::GetElementDiameter() const
#endif
  {
    return element_[0]->diameter();
  }
  /**********************************************/
  template<typename VECTOR, int dim>
  Point<dim>
#if DEAL_II_VERSION_GTE(9,3,0)
  ElementDataContainer<false, VECTOR, dim>::GetCenter() const
#else
  ElementDataContainer<dealii::DoFHandler, VECTOR, dim>::GetCenter() const
#endif
  {
    return element_[0]->center();
  }

  /**********************************************/
  template<typename VECTOR, int dim>
  const DOpEWrapper::FEValues<dim> &
#if DEAL_II_VERSION_GTE(9,3,0)
  ElementDataContainer<false, VECTOR, dim>::GetFEValuesState() const
#else
  ElementDataContainer<dealii::DoFHandler, VECTOR, dim>::GetFEValuesState() const
#endif
  {
    return state_fe_values_;
  }

  /**********************************************/
  template<typename VECTOR, int dim>
  const DOpEWrapper::FEValues<dim> &
#if DEAL_II_VERSION_GTE(9,3,0)
  ElementDataContainer<false, VECTOR, dim>::GetFEValuesControl() const
#else
  ElementDataContainer<dealii::DoFHandler, VECTOR, dim>::GetFEValuesControl() const
#endif
  {
    return control_fe_values_;
  }

  /***********************************************************************/

  template<typename VECTOR, int dim>
  unsigned int
#if DEAL_II_VERSION_GTE(9,3,0)
  ElementDataContainer<false, VECTOR, dim>::GetStateIndex() const
#else
  ElementDataContainer<dealii::DoFHandler, VECTOR, dim>::GetStateIndex() const
#endif
  {
    return state_index_;
  }

  /***********************************************************************/

  template<typename VECTOR, int dim>
  unsigned int
#if DEAL_II_VERSION_GTE(9,3,0)
  ElementDataContainer<false, VECTOR, dim>::GetControlIndex() const
#else
  ElementDataContainer<dealii::DoFHandler, VECTOR, dim>::GetControlIndex() const
#endif
  {
    return control_index_;
  }

  /***********************************************************************/
  /************************END*OF*IMPLEMENTATION**************************/
  /***********************************************************************/
  /***********************************************************************/
  /************************IMPLEMENTATION*********************************/
  /***********************************************************************/


  template<typename VECTOR, int dim>
  void
#if DEAL_II_VERSION_GTE(9,3,0)
  ElementDataContainer<true, VECTOR, dim>::ReInit()
#else
  ElementDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::ReInit()
#endif
  {
    state_hp_fe_values_.reinit(element_[this->GetStateIndex()]);
    //Make sure that the Control must be initialized.
    if (this->GetControlIndex() < element_.size())
      control_hp_fe_values_.reinit(element_[this->GetControlIndex()]);

    edcinternal::ElementDataContainerInternal<
    VECTOR, dim>::ReInit(element_[this->GetStateIndex()]);
  }
  /*********************************************/

  template<typename VECTOR, int dim>
  unsigned int
#if DEAL_II_VERSION_GTE(9,3,0)
  ElementDataContainer<true, VECTOR, dim>::GetNDoFsPerElement() const
#else
  ElementDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetNDoFsPerElement() const
#endif
  {
    return element_[0]->get_fe().dofs_per_cell;
  }
  /*********************************************/
  template<typename VECTOR, int dim>
  unsigned int
#if DEAL_II_VERSION_GTE(9,3,0)
  ElementDataContainer<true, VECTOR, dim>::GetNQPoints() const
#else
  ElementDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetNQPoints() const
#endif
  {
    return (q_collection_[element_[0]->active_fe_index()]).size();
  }
  /*********************************************/

  template<typename VECTOR, int dim>
  unsigned int
#if DEAL_II_VERSION_GTE(9,3,0)
  ElementDataContainer<true, VECTOR, dim>::GetMaterialId() const
#else
  ElementDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetMaterialId() const
#endif
  {
    return element_[0]->material_id();
  }
  /*********************************************/

  template<typename VECTOR, int dim>
  unsigned int
#if DEAL_II_VERSION_GTE(9,3,0)
  ElementDataContainer<true, VECTOR, dim>::GetNbrMaterialId(
    unsigned int face) const
#else
  ElementDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetNbrMaterialId(
    unsigned int face) const
#endif
  {
    if (element_[0]->neighbor_index(face) != -1)
      return element_[0]->neighbor(face)->material_id();
    else
      {
        std::stringstream out;
        out << "There is no neighbor with number " << face;
        throw DOpEException(out.str(),
                            "HpElementDataContainer::GetNbrMaterialId");
      }
  }

  /**********************************************/
  template<typename VECTOR, int dim>
  unsigned int
#if DEAL_II_VERSION_GTE(9,3,0)
  ElementDataContainer<true, VECTOR, dim>::GetFaceBoundaryIndicator(
    unsigned int face) const
#else
  ElementDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetFaceBoundaryIndicator(
    unsigned int face) const
#endif
  {
    return element_[0]->face(face)->boundary_indicator();
  }

  /*********************************************/

  template<typename VECTOR, int dim>
  bool
#if DEAL_II_VERSION_GTE(9,3,0)
  ElementDataContainer<true, VECTOR, dim>::GetIsAtBoundary() const
#else
  ElementDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetIsAtBoundary() const
#endif
  {
    return element_[0]->at_boundary();
  }

  /*********************************************/

  template<typename VECTOR, int dim>
  double
#if DEAL_II_VERSION_GTE(9,3,0)
  ElementDataContainer<true, VECTOR, dim>::GetElementDiameter() const
#else
  ElementDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetElementDiameter() const
#endif
  {
    return element_[0]->diameter();
  }

  /**********************************************/
  template<typename VECTOR, int dim>
  Point<dim>
#if DEAL_II_VERSION_GTE(9,3,0)
  ElementDataContainer<true, VECTOR, dim>::GetCenter() const
#else
  ElementDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetCenter() const
#endif
  {
    return element_[0]->center();
  }

  /*********************************************/
  template<typename VECTOR, int dim>
  const DOpEWrapper::FEValues<dim> &
#if DEAL_II_VERSION_GTE(9,3,0)
  ElementDataContainer<true, VECTOR, dim>::GetFEValuesState() const
#else
  ElementDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetFEValuesState() const
#endif
  {
    return static_cast<const DOpEWrapper::FEValues<dim>&>(state_hp_fe_values_.get_present_fe_values());
  }
  /*********************************************/
  template<typename VECTOR, int dim>
  const DOpEWrapper::FEValues<dim> &
#if DEAL_II_VERSION_GTE(9,3,0)
  ElementDataContainer<true, VECTOR, dim>::GetFEValuesControl() const
#else
  ElementDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetFEValuesControl() const
#endif
  {
    return static_cast<const DOpEWrapper::FEValues<dim>&>(control_hp_fe_values_.get_present_fe_values());
  }
  /*********************************************/

  template<typename VECTOR, int dim>
  unsigned int
#if DEAL_II_VERSION_GTE(9,3,0)
  ElementDataContainer<true, VECTOR, dim>::GetStateIndex() const
#else
  ElementDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetStateIndex() const
#endif
  {
    return state_index_;
  }
  /*********************************************/
  template<typename VECTOR, int dim>
  unsigned int
#if DEAL_II_VERSION_GTE(9,3,0)
  ElementDataContainer<true, VECTOR, dim>::GetControlIndex() const
#else
  ElementDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetControlIndex() const
#endif
  {
    return control_index_;
  }
} //end of namespace

#endif /* WORKINGTITLE_H_ */
