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

#ifndef FACEDATACONTAINER_H_
#define FACEDATACONTAINER_H_

#include <basic/spacetimehandler.h>
#include <basic/statespacetimehandler.h>
#include <wrapper/fevalues_wrapper.h>
#include <include/dopeexception.h>
#include <container/facedatacontainer_internal.h>

#include <sstream>

#include <deal.II/dofs/dof_handler.h>
#if ! DEAL_II_VERSION_GTE(9,3,0)
#include <deal.II/hp/dof_handler.h>
#endif

using namespace dealii;

namespace DOpE
{
  /**
   * Dummy Template Class, acts as kind of interface. Through template specialization, we
   * distinguish between the 'classic' and the 'hp' case.
   *
   */

#if DEAL_II_VERSION_GTE(9,3,0)
  template<bool DH, typename VECTOR, int dim>
#else
  template<template<int, int> class DH, typename VECTOR, int dim>
#endif
    class FaceDataContainer : public fdcinternal::FaceDataContainerInternal<
    VECTOR, dim>
  {
  public:
    FaceDataContainer()
    {
      throw (DOpEException(
               "Dummy class, this constructor should never get called.",
               "FaceDataContainer<dealii::DoFHandler , VECTOR, dim>::FaceDataContainer"));
    }
  };

  /**
   * This two classes hold all the information we need in the integrator to
   * integrate something over a face of a element (could be a functional, a PDE, etc.).
   * Of particular importance: This class holds the (Sub)FaceFEValues objects.
   *
   * @template VECTOR     Type of the vector we use in our computations (i.e. Vector<double> or BlockVector<double>)
   * @template dim        1+ the dimension of the integral we are actually interested in.//TODO 1+??
   */

  template<typename VECTOR, int dim>
#if DEAL_II_VERSION_GTE(9,3,0)
    class FaceDataContainer<false, VECTOR, dim> : public fdcinternal::FaceDataContainerInternal<VECTOR, dim>
#else
    class FaceDataContainer<dealii::DoFHandler, VECTOR, dim> : public fdcinternal::FaceDataContainerInternal<VECTOR, dim>
#endif
  {

  public:
    /**
     * Constructor. Initializes the FaceFEValues objects.
     *
     * @template FE                   The type of Finite Element in use here.
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
     * @param need_neighbour          Describes whether we need all the GetNbr (= Get Neighbor) functions.
     *
     */
    template<template<int, int> class FE, typename SPARSITYPATTERN, int dopedim, int dealdim>
    FaceDataContainer(const Quadrature<dim - 1>& quad,
                      UpdateFlags update_flags,
#if DEAL_II_VERSION_GTE(9,3,0)
                      SpaceTimeHandler<FE, false, SPARSITYPATTERN, VECTOR,
#else
                      SpaceTimeHandler<FE, dealii::DoFHandler, SPARSITYPATTERN, VECTOR,
#endif
                      dopedim, dealdim> &sth,
#if DEAL_II_VERSION_GTE(9,3,0)
		      const std::vector<typename DOpEWrapper::DoFHandler<dim>::active_cell_iterator>& element,
#else
		      const std::vector<typename DOpEWrapper::DoFHandler<dim, dealii::DoFHandler>::active_cell_iterator>& element,
#endif
                      const std::map<std::string, const Vector<double>*> &param_values,
                      const std::map<std::string, const VECTOR *> &domain_values,
                      bool need_neighbour) :
      fdcinternal::FaceDataContainerInternal<VECTOR, dim>(param_values, domain_values, need_neighbour), element_(element), state_fe_values_(
                                                            sth.GetMapping(), (sth.GetFESystem("state")), quad,
                                                            update_flags), control_fe_values_(sth.GetMapping(),
                                                                (sth.GetFESystem("control")), quad, update_flags)
    {
      state_index_ = sth.GetStateIndex();
      if (state_index_ == 1)
        control_index_ = 0;
      else
        control_index_ = 1;

      if (need_neighbour) //so we need FEFAcevalues etc. for the neighbour too.
        {
          nbr_control_fe_values_ = new DOpEWrapper::FEFaceValues<dim>(
            sth.GetMapping(), (sth.GetFESystem("control")), quad,
            update_flags);
          control_fe_subface_values_ =
            new DOpEWrapper::FESubfaceValues<dim>(sth.GetMapping(),
                                                  (sth.GetFESystem("control")), quad, update_flags);
        }
      this->PrivateConstructor(quad, update_flags, sth, need_neighbour);
    }
    /**
     * Constructor. Initializes the FaceFEValues objects. For PDE only
     *
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
     *                                be found in this map at the position "state".
     * @param need_neighbour          Describes whether we need all the GetNbr (= Get Neighbor) functions.
     *
     */
    template<template<int, int> class FE, typename SPARSITYPATTERN>
    FaceDataContainer(const Quadrature<dim - 1>& quad,
                      UpdateFlags update_flags,
#if DEAL_II_VERSION_GTE(9,3,0)
                      StateSpaceTimeHandler<FE, false, SPARSITYPATTERN,
#else
                      StateSpaceTimeHandler<FE, dealii::DoFHandler, SPARSITYPATTERN,
#endif
                      VECTOR, dim> &sth,
#if DEAL_II_VERSION_GTE(9,3,0)
		      const std::vector<typename DOpEWrapper::DoFHandler<dim>::active_cell_iterator>& element,
#else
		      const std::vector<typename DOpEWrapper::DoFHandler<dim, dealii::DoFHandler>::active_cell_iterator>& element,
#endif
                      const std::map<std::string, const Vector<double>*> &param_values,
                      const std::map<std::string, const VECTOR *> &domain_values,
                      bool need_neighbour) :
      fdcinternal::FaceDataContainerInternal<VECTOR, dim>(param_values,
                                                          domain_values, need_neighbour), element_(element), state_fe_values_(
                                                            sth.GetMapping(), (sth.GetFESystem("state")), quad,
                                                            update_flags), control_fe_values_(sth.GetMapping(),
                                                                (sth.GetFESystem("state")), quad, update_flags)
    {
      state_index_ = sth.GetStateIndex();
      control_index_ = element.size();
      n_q_points_per_element_ = quad.size();
      n_dofs_per_element_ = element[0]->get_fe().dofs_per_cell;

      if (need_neighbour) //so we need FEFAcevalues for the neighbour too.
        {
          nbr_control_fe_values_ = new DOpEWrapper::FEFaceValues<dim>(
            sth.GetMapping(), (sth.GetFESystem("state")), quad,
            update_flags);
          control_fe_subface_values_ =
            new DOpEWrapper::FESubfaceValues<dim>(sth.GetMapping(),
                                                  (sth.GetFESystem("state")), quad, update_flags);
        }
      this->PrivateConstructor(quad, update_flags, sth, need_neighbour);
    }

    ~FaceDataContainer()
    {
      if (nbr_state_fe_values_ != NULL)
        {
          delete nbr_state_fe_values_;
        }
      if (nbr_control_fe_values_ != NULL)
        {
          delete nbr_control_fe_values_;
        }
      if (state_fe_subface_values_ != NULL)
        {
          delete state_fe_subface_values_;
        }
      if (control_fe_subface_values_ != NULL)
        {
          delete control_fe_subface_values_;
        }
    }
    /*********************************************/
    /*
     * This function reinitializes the FEFaceValues on the actual face. Should
     * be called prior to any of the get-functions.
     *
     * @param face_no     The 'local number' (i.e. from the perspective of the actual element) of the
     *                    actual face.
     */
    inline void
    ReInit(unsigned int face_no);

    /*********************************************/
    /*
     * This function reinits the FESubfaceValues on the actual subface. Should
     * be called prior to any of the get-functions.
     *
     * @param face_no     The 'local number' (i.e. from the perspective of the actual element) of the
     *                    actual face.
     * @param subface_no  The 'local number' (i.e. from the perspective of the actual element) of the
     *                    actual subface.
     */
    inline void
    ReInit(unsigned int face_no, unsigned int subface_no);

    /*********************************************/
    /*
     * This function reinitializes the FE(Sub)FaceValues on the neighbor_element.
     * This should be called prior to any of the get nbr_functions.
     * Assumes that ReInit is called prior to this function.
     */
    inline void
    ReInitNbr();

    /*********************************************/
    /**
     * Get functions to extract data. They all assume that ReInit
     * (resp. ReInitNbr for the GetNbr* functions) is executed
     * before calling them.
     */
    inline unsigned int
    GetNDoFsPerElement() const;
    inline unsigned int
    GetNbrNDoFsPerElement() const;
    inline unsigned int
    GetNQPoints() const;
    inline unsigned int
    GetNbrNQPoints() const;
    inline unsigned int
    GetMaterialId() const;
    inline unsigned int
    GetNbrMaterialId() const;
    inline unsigned int
    GetNbrMaterialId(unsigned int face) const;
    inline bool
    GetIsAtBoundary() const;
    inline double
    GetElementDiameter() const;
    inline Point<dim>
    GetCenter() const;
    inline unsigned int
    GetBoundaryIndicator() const;
    inline const FEFaceValuesBase<dim> &
    GetFEFaceValuesState() const;
    inline const FEFaceValuesBase<dim> &
    GetFEFaceValuesControl() const;

    inline const FEFaceValuesBase<dim> &
    GetNbrFEFaceValuesState() const;
    inline const FEFaceValuesBase<dim> &
    GetNbrFEFaceValuesControl() const;

  protected:
    /*
     * Helper Functions
     */
    unsigned int
    GetStateIndex() const;
    unsigned int
    GetControlIndex() const;

  private:
    /**
     * This function contains common code of the constructors.
     */
    template<class STH>
    void
    PrivateConstructor(const Quadrature<dim - 1>& quad,
                       UpdateFlags update_flags, STH &sth, bool need_neighbour)
    {
      n_q_points_per_element_ = quad.size();
      n_dofs_per_element_ = element_[0]->get_fe().dofs_per_cell;

      if (need_neighbour) //so we need FEFAcevalues etc. for the neighbour too.
        {
          nbr_state_fe_values_ = new DOpEWrapper::FEFaceValues<dim>(
            (sth.GetFESystem("state")), quad, update_flags);
          state_fe_subface_values_ = new DOpEWrapper::FESubfaceValues<dim>(
            (sth.GetFESystem("state")), quad, update_flags);
        }
      else
        {
          nbr_state_fe_values_ = NULL;
          nbr_control_fe_values_ = NULL;
          state_fe_subface_values_ = NULL;
          control_fe_subface_values_ = NULL;
        }
      // These will point to the object (i.e. FaceValues or SubfaceValues) we actually use.
      // With this, we have the same interface to the user independently of the type (i.e. face or subface)
      state_fe_values_ptr_ = NULL;
      control_fe_values_ptr_ = NULL;
      nbr_state_fe_values_ptr_ = NULL;
      nbr_control_fe_values_ptr_ = NULL;
    }
    /***********************************************************/
    //"global" member data, part of every instantiation
    unsigned int state_index_;
    unsigned int control_index_;

#if DEAL_II_VERSION_GTE(9,3,0)
    const std::vector<typename DOpEWrapper::DoFHandler<dim>::active_cell_iterator> & element_;
#else
    const std::vector<typename DOpEWrapper::DoFHandler<dim, dealii::DoFHandler>::active_cell_iterator> & element_;
#endif
    DOpEWrapper::FEFaceValues<dim> state_fe_values_;
    DOpEWrapper::FEFaceValues<dim> control_fe_values_;

    DOpEWrapper::FEFaceValues<dim> *nbr_state_fe_values_ = nullptr;
    DOpEWrapper::FEFaceValues<dim> *nbr_control_fe_values_ = nullptr;

    DOpEWrapper::FESubfaceValues<dim> *state_fe_subface_values_ = nullptr;
    DOpEWrapper::FESubfaceValues<dim> *control_fe_subface_values_ = nullptr;

    dealii::FEFaceValuesBase<dim> *state_fe_values_ptr_ = nullptr;
    dealii::FEFaceValuesBase<dim> *control_fe_values_ptr_ = nullptr;
    dealii::FEFaceValuesBase<dim> *nbr_state_fe_values_ptr_ = nullptr;
    dealii::FEFaceValuesBase<dim> *nbr_control_fe_values_ptr_ = nullptr;

    unsigned int n_q_points_per_element_ = 0;
    unsigned int n_dofs_per_element_ = 0;
  };



  /****************************************************/


  template<typename VECTOR, int dim>
#if DEAL_II_VERSION_GTE(9,3,0)
    class FaceDataContainer<true, VECTOR, dim> : public fdcinternal::FaceDataContainerInternal<VECTOR, dim>
#else
    class FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim> : public fdcinternal::FaceDataContainerInternal<VECTOR, dim>
#endif
  {

  public:
    /**
     * Constructor. Initializes the hp::FaceFEValues objects.
     *
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
     * @param need_neighbour         Describes, whether we need all the GetNbr (= Get Neighbor) functions.
     *
     */
    template<template<int, int> class FE, typename SPARSITYPATTERN, int dopedim, int dealdim>
    FaceDataContainer(
      const hp::QCollection<dim - 1>& q_collection,
      UpdateFlags update_flags,
#if DEAL_II_VERSION_GTE(9,3,0)
      SpaceTimeHandler<FE, true, SPARSITYPATTERN,
      VECTOR, dopedim, dealdim> &sth,
#else
      SpaceTimeHandler<FE, dealii::hp::DoFHandler, SPARSITYPATTERN,
      VECTOR, dopedim, dealdim> &sth,
#endif
      const std::vector<
#if DEAL_II_VERSION_GTE(9,3,0)
      typename DOpEWrapper::DoFHandler<dim>::active_cell_iterator>& element,
#else
      typename DOpEWrapper::DoFHandler<dim, dealii::hp::DoFHandler>::active_cell_iterator>& element,
#endif
      const std::map<std::string, const Vector<double>*> &param_values,
      const std::map<std::string, const VECTOR *> &domain_values,
      bool need_neighbour) :
      fdcinternal::FaceDataContainerInternal<VECTOR, dim>(param_values,
                                                          domain_values, need_neighbour), element_(element), state_hp_fe_values_(
                                                            (sth.GetFESystem("state")), q_collection, update_flags), control_hp_fe_values_(
                                                              (sth.GetFESystem("control")), q_collection, update_flags), q_collection_(
                                                                q_collection)
    {
      state_index_ = sth.GetStateIndex();
      if (state_index_ == 1)
        control_index_ = 0;
      else
        control_index_ = 1;

      if (need_neighbour) //so we need FEFAcevalues for the neighbour too.
        {
          nbr_control_hp_fe_values_ = new DOpEWrapper::HpFEFaceValues<dim>(
            sth.GetMapping(), (sth.GetFESystem("control")), q_collection,
            update_flags);
          control_hp_fe_subface_values_ =
            new DOpEWrapper::HpFESubfaceValues<dim>(sth.GetMapping(),
                                                    (sth.GetFESystem("control")), q_collection, update_flags);

        }
      this->PrivateConstructor(q_collection, update_flags, sth,
                               need_neighbour);
    }

    /**
     * Constructor. Initializes the hp::FaceFEValues objects.
     *
     * @template SPARSITYPATTERN      The corresponding Sparsitypattern to the class-template VECTOR.
     * @template dopedim              The dimension of the control variable.
     * @template dealdim              The dimension of the state variable.
     *
     * @param quad                    Reference to the quadrature-rule which we use at the moment.
     * @param update_flags            The update flags we need to initialize the FEValues obejcts
     * @param sth                     A reference to the StateSpaceTimeHandler in use.
     * @param element                    A vector of element iterators through which we gain most of the needed information (like
     *                                material_ids, n_dfos, etc.)
     * @param param_values            A std::map containing parameter data (e.g. non space dependent data). If the control
     *                                is done by parameters, it is contained in this map at the position "control".
     * @param domain_values           A std::map containing domain data (e.g. nodal vectors for FE-Functions). If the control
     *                                is distributed, it is contained in this map at the position "control". The state may always
     *                                be found in this map at the position "state"
     * @param need_neighbour         Describes, whether we need all the GetNbr (= Get Neighbor) functions.
     *
     */
    template<template<int, int> class FE, typename SPARSITYPATTERN, int dealdim>
    FaceDataContainer(
      const hp::QCollection<dim - 1>& q_collection,
      UpdateFlags update_flags,
#if DEAL_II_VERSION_GTE(9,3,0)
      StateSpaceTimeHandler<FE, true, SPARSITYPATTERN,
#else
      StateSpaceTimeHandler<FE, dealii::hp::DoFHandler, SPARSITYPATTERN,
#endif
      VECTOR, dealdim> &sth,
      const std::vector<
#if DEAL_II_VERSION_GTE(9,3,0)
      typename DOpEWrapper::DoFHandler<dim>::active_cell_iterator>& element,
#else
      typename DOpEWrapper::DoFHandler<dim, dealii::hp::DoFHandler>::active_cell_iterator>& element,
#endif
      const std::map<std::string, const Vector<double>*> &param_values,
      const std::map<std::string, const VECTOR *> &domain_values,
      bool need_neighbour) :
      fdcinternal::FaceDataContainerInternal<VECTOR, dim>(param_values,
                                                          domain_values, need_neighbour), element_(element), state_hp_fe_values_(
                                                            (sth.GetFESystem("state")), q_collection, update_flags), control_hp_fe_values_(
                                                              (sth.GetFESystem("state")), q_collection, update_flags), q_collection_(
                                                                q_collection)
    {
      state_index_ = sth.GetStateIndex();
      if (state_index_ == 1)
        control_index_ = 0;
      else
        control_index_ = 1;

      if (need_neighbour)
        {
          nbr_control_hp_fe_values_ = new DOpEWrapper::HpFEFaceValues<dim>(
            sth.GetMapping(), (sth.GetFESystem("state")), q_collection,
            update_flags);
          control_hp_fe_subface_values_ =
            new DOpEWrapper::HpFESubfaceValues<dim>(sth.GetMapping(),
                                                    (sth.GetFESystem("state")), q_collection, update_flags);
        }
      this->PrivateConstructor(q_collection, update_flags, sth,
                               need_neighbour);
    }

    /**
     * Destructor
     */
    ~FaceDataContainer()
    {
      delete nbr_state_hp_fe_values_;
      nbr_state_hp_fe_values_ = NULL;
      delete nbr_control_hp_fe_values_;
      nbr_control_hp_fe_values_ = NULL;
      delete state_hp_fe_subface_values_;
      state_hp_fe_subface_values_ = NULL;
      delete control_hp_fe_subface_values_;
      control_hp_fe_subface_values_ = NULL;
    }

    /*********************************************/
    /*
     * This function reinits the FEFaceValues on the actual face. Should
     * be called prior to any of the get-functions.
     *
     * @param face_no     The 'local number' (i.e. from the perspective of the actual element) of the
     *                    actual face.
     */
    inline void
    ReInit(unsigned int face_no);

    /*********************************************/
    /*
     * This function reinits the FESubfaceValues on the actual subface. Should
     * be called prior to any of the get-functions.
     *
     * @param face_no     The 'local number' (i.e. from the perspective of the actual element) of the
     *                    actual face.
     * @param subface_no  The 'local number' (i.e. from the perspective of the actual element) of the
     *                    actual subface.
     */
    inline void
    ReInit(unsigned int face_no, unsigned int subface_no);

    /*********************************************/
    /*
     * This function reinits the FEFaceValues on the neighbor_element for the
     * case that the neighbor-element coarser or as fine as the
     * (previously set!) actual element. This should be called prior
     * to any of the get nbr_functions.
     * Assumes that ReInit is called prior to this function.
     */
    inline void
    ReInitNbr();

    /*********************************************/
    /**
     * Get functions to extract data. They all assume that ReInit
     * is executed before calling them.
     */
    inline unsigned int
    GetNDoFsPerElement() const;

    inline unsigned int
    GetNbrNDoFsPerElement() const;
    inline unsigned int
    GetNQPoints() const;
    inline unsigned int
    GetNbrNQPoints() const;
    inline unsigned int
    GetMaterialId() const;
    inline unsigned int
    GetNbrMaterialId() const;
    inline unsigned int
    GetNbrMaterialId(unsigned int face) const;
    inline bool
    GetIsAtBoundary() const;
    inline unsigned int
    GetBoundaryIndicator() const;
    inline double
    GetElementDiameter() const;
    inline Point<dim>
    GetCenter() const;
    inline const FEFaceValuesBase<dim> &
    GetFEFaceValuesState() const;
    inline const FEFaceValuesBase<dim> &
    GetFEFaceValuesControl() const;

    inline const FEFaceValuesBase<dim> &
    GetNbrFEFaceValuesState() const;
    inline const FEFaceValuesBase<dim> &
    GetNbrFEFaceValuesControl() const;

  private:
    inline unsigned int
    GetStateIndex() const;
    inline unsigned int
    GetControlIndex() const;

    /**
     * Contains common code of the constructors.
     */
    template<class STH>
    void
    PrivateConstructor(const hp::QCollection<dim - 1>& q_collection,
                       UpdateFlags update_flags, STH &sth, bool need_neighbour)
    {
      if (need_neighbour) //so we need FEFAcevalues for the neighbour too.
        {
          nbr_state_hp_fe_values_ = new DOpEWrapper::HpFEFaceValues<dim>(
            sth.GetMapping(), (sth.GetFESystem("state")), q_collection,
            update_flags);
          state_hp_fe_subface_values_ = new DOpEWrapper::HpFESubfaceValues<
          dim>(sth.GetMapping(), (sth.GetFESystem("state")),
               q_collection, update_flags);

        }
      else
        {
          nbr_state_hp_fe_values_ = NULL;
          nbr_control_hp_fe_values_ = NULL;
          state_hp_fe_subface_values_ = NULL;
          control_hp_fe_subface_values_ = NULL;
        }

      nbr_state_hp_fe_values_ptr_ = NULL;
      nbr_control_hp_fe_values_ptr_ = NULL;
      state_hp_fe_values_ptr_ = NULL;
      control_hp_fe_values_ptr_ = NULL;
    }
    /***********************************************************/
    //"global" member data, part of every instantiation
    unsigned int state_index_;
    unsigned int control_index_;
    const std::vector<
#if DEAL_II_VERSION_GTE(9,3,0)
      typename DOpEWrapper::DoFHandler<dim>::active_cell_iterator>& element_;
#else
      typename DOpEWrapper::DoFHandler<dim, dealii::hp::DoFHandler>::active_cell_iterator>& element_;
#endif

    DOpEWrapper::HpFEFaceValues<dim> state_hp_fe_values_;
    DOpEWrapper::HpFEFaceValues<dim> control_hp_fe_values_;

    DOpEWrapper::HpFEFaceValues<dim> *nbr_state_hp_fe_values_ = nullptr;
    DOpEWrapper::HpFEFaceValues<dim> *nbr_control_hp_fe_values_ = nullptr;

    DOpEWrapper::HpFESubfaceValues<dim> *state_hp_fe_subface_values_ = nullptr;
    DOpEWrapper::HpFESubfaceValues<dim> *control_hp_fe_subface_values_ = nullptr;

    const dealii::FEFaceValuesBase<dim> *state_hp_fe_values_ptr_ = nullptr;
    const dealii::FEFaceValuesBase<dim> *control_hp_fe_values_ptr_ = nullptr;
    const dealii::FEFaceValuesBase<dim> *nbr_state_hp_fe_values_ptr_ = nullptr;
    const dealii::FEFaceValuesBase<dim> *nbr_control_hp_fe_values_ptr_ = nullptr;

    const hp::QCollection<dim - 1>& q_collection_;
  };

  /***********************************************************************/
  /************************IMPLEMENTATION*for*DoFHandler*********************************/
  /***********************************************************************/

  namespace
  {
//The following will only be used in debug mode; switch of warning of unused functions
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"

#if DEAL_II_VERSION_GTE(9,3,0)
    template<int dim>
#else
    template<int dim, template<int, int> class DH>
#endif
    bool sanity_check(const
#if DEAL_II_VERSION_GTE(9,3,0)
		      typename DOpEWrapper::DoFHandler<dim>::active_cell_iterator &element_,
#else
		      typename DOpEWrapper::DoFHandler<dim, DH>::active_cell_iterator &element_,
#endif
                      unsigned int face,
                      unsigned int subface)
    {
      const auto neighbor_child =
        element_->neighbor_child_on_subface(
          face, subface);

      bool ret = false;
      if (neighbor_child->face(element_->neighbor_of_neighbor(face)) == element_->face(face)->child(subface))
        ret = true;
      return  ret;
    }

#if DEAL_II_VERSION_GTE(9,3,0)
    template<>
    bool sanity_check<1>(const
			 typename DOpEWrapper::DoFHandler<1>::active_cell_iterator &,
			 unsigned int,
			 unsigned int)
    {
      return  true;
    }
#else
    template<>
    bool sanity_check<1,dealii::hp::DoFHandler>(const
                                                typename DOpEWrapper::DoFHandler<1, dealii::hp::DoFHandler>::active_cell_iterator &,
                                                unsigned int,
                                                unsigned int)
    {
      return  true;
    }

    template<>
    bool sanity_check<1,dealii::DoFHandler>(const
                                            typename DOpEWrapper::DoFHandler<1, dealii::DoFHandler>::active_cell_iterator &,
                                            unsigned int,
                                            unsigned int)
    {
      return  true;
    }
#endif
//Reenable warning or unused functions
#pragma GCC diagnostic pop
  }

  template<typename VECTOR, int dim>
  void
#if DEAL_II_VERSION_GTE(9,3,0)
    FaceDataContainer<false, VECTOR, dim>::ReInit(
    unsigned int face_no)
#else
    FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::ReInit(
    unsigned int face_no)
#endif
  {
    this->SetFace(face_no);
    state_fe_values_.reinit(element_[this->GetStateIndex()], face_no);
    state_fe_values_ptr_ = &state_fe_values_;
    //Make sure that the Control must be initialized.
    if (this->GetControlIndex() < element_.size())
      {
        control_fe_values_.reinit(element_[this->GetControlIndex()], face_no);
        control_fe_values_ptr_ = &control_fe_values_;
      }
  }

  /***********************************************************************/

  template<typename VECTOR, int dim>
  void
#if DEAL_II_VERSION_GTE(9,3,0)
  FaceDataContainer<false, VECTOR, dim>::ReInit(
    unsigned int face_no, unsigned int subface_no)
#else
  FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::ReInit(
    unsigned int face_no, unsigned int subface_no)
#endif
  {
    this->SetFace(face_no);
    this->SetSubFace(subface_no);
    state_fe_subface_values_->reinit(element_[this->GetStateIndex()], face_no,
                                     subface_no);
    state_fe_values_ptr_ = state_fe_subface_values_;
    //Make sure that the Control must be initialized.
    if (this->GetControlIndex() < element_.size())
      {
        control_fe_subface_values_->reinit(element_[this->GetControlIndex()],
                                           face_no, this->GetSubFace());
        control_fe_values_ptr_ = control_fe_subface_values_;
      }
  }
  /***********************************************************************/


  template<typename VECTOR, int dim>
  void
#if DEAL_II_VERSION_GTE(9,3,0)
  FaceDataContainer<false, VECTOR, dim>::ReInitNbr()
#else
  FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::ReInitNbr()
#endif
  {
    Assert(this->NeedNeighbour(), ExcInternalError());
    Assert(
      element_[this->GetStateIndex()]->neighbor_index(this->GetFace()) != -1,
      TriaAccessorExceptions::ExcCellNotUsed())

    if (element_[this->GetStateIndex()]->neighbor(this->GetFace())->has_children())
      {
        //if neighbor is more refined
        const auto neighbor_child =
          element_[this->GetStateIndex()]->neighbor_child_on_subface(
            this->GetFace(), this->GetSubFace());

        // some sanity checks: Check, that the face and subface match and that the neighbour child
        // is not more refined.
#if DEAL_II_VERSION_GTE(9,3,0)
	Assert((sanity_check<dim>(element_[this->GetStateIndex()],
				  this->GetFace(),
				  this->GetSubFace()) == true), ExcInternalError());
#else
	Assert((sanity_check<dim, dealii::DoFHandler>(element_[this->GetStateIndex()],
                                                      this->GetFace(),
                                                      this->GetSubFace()) == true), ExcInternalError());
#endif
        Assert(neighbor_child->has_children() == false, ExcInternalError());

        nbr_state_fe_values_->reinit(neighbor_child,
                                     element_[this->GetStateIndex()]->neighbor_of_neighbor(
                                       this->GetFace()));
        nbr_state_fe_values_ptr_ = nbr_state_fe_values_;

        //Make sure that the Control must be initialized.
        if (this->GetControlIndex() < element_.size())
          {
            const auto control_neighbor_child =
              element_[this->GetControlIndex()]->neighbor_child_on_subface(
                this->GetFace(), this->GetSubFace());

            nbr_control_fe_values_->reinit(control_neighbor_child,
                                           element_[this->GetControlIndex()]->neighbor_of_neighbor(
                                             this->GetFace()));
            nbr_control_fe_values_ptr_ = nbr_control_fe_values_;
          }
      }
    else if (element_[this->GetStateIndex()]->neighbor_is_coarser(
               this->GetFace()))
      {
        //if the neighbour is coarser
        Assert(
          element_[this->GetStateIndex()]->neighbor(this->GetFace())->level() == element_[this->GetStateIndex()]->level()-1,
          ExcInternalError());
        const auto neighbor = element_[this->GetStateIndex()]->neighbor(
                                this->GetFace());
        const std::pair<unsigned int, unsigned int> faceno_subfaceno =
          element_[this->GetStateIndex()]->neighbor_of_coarser_neighbor(
            this->GetFace());
        const unsigned int neighbor_face_no = faceno_subfaceno.first,
                           neighbor_subface_no = faceno_subfaceno.second;
        state_fe_subface_values_->reinit(neighbor, neighbor_face_no,
                                         neighbor_subface_no);
        nbr_state_fe_values_ptr_ = state_fe_subface_values_;
        if (this->GetControlIndex() < element_.size())
          {
            const auto control_neighbor =
              element_[this->GetControlIndex()]->neighbor(this->GetFace());
            const std::pair<unsigned int, unsigned int> control_faceno_subfaceno =
              element_[this->GetControlIndex()]->neighbor_of_coarser_neighbor(
                this->GetFace());
            const unsigned int control_neighbor_face_no =
              control_faceno_subfaceno.first, control_neighbor_subface_no =
                control_faceno_subfaceno.second;
            control_fe_subface_values_->reinit(control_neighbor,
                                               control_neighbor_face_no, control_neighbor_subface_no);
            nbr_control_fe_values_ptr_ = control_fe_subface_values_;
          }

      }
    else
      {
        const auto neighbor_state = element_[this->GetStateIndex()]->neighbor(
                                      this->GetFace());
        // neighbor element is as much refined as the
        Assert(neighbor_state->level() == element_[this->GetStateIndex()]->level(),
               ExcInternalError());
        nbr_state_fe_values_->reinit(neighbor_state,
                                     element_[this->GetStateIndex()]->neighbor_of_neighbor(
                                       this->GetFace()));
        nbr_state_fe_values_ptr_ = nbr_state_fe_values_;

        //Make sure that the Control must be initialized.
        if (this->GetControlIndex() < element_.size())
          {
            nbr_control_fe_values_->reinit(
              element_[this->GetControlIndex()]->neighbor(this->GetFace()),
              element_[this->GetControlIndex()]->neighbor_of_neighbor(
                this->GetFace()));
            nbr_control_fe_values_ptr_ = nbr_control_fe_values_;
          }
      }
  }
  /***********************************************************************/
  template<typename VECTOR, int dim>
  unsigned int
#if DEAL_II_VERSION_GTE(9,3,0)
  FaceDataContainer<false, VECTOR, dim>::GetNDoFsPerElement() const
#else
  FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetNDoFsPerElement() const
#endif
  {
    return n_dofs_per_element_;
  }

  /***********************************************************************/

  template<typename VECTOR, int dim>
  unsigned int
#if DEAL_II_VERSION_GTE(9,3,0)
  FaceDataContainer<false, VECTOR, dim>::GetNbrNDoFsPerElement() const
#else
  FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetNbrNDoFsPerElement() const
#endif
  {
    return n_dofs_per_element_;
  }

  /**********************************************/
  template<typename VECTOR, int dim>
  unsigned int
#if DEAL_II_VERSION_GTE(9,3,0)
  FaceDataContainer<false, VECTOR, dim>::GetNQPoints() const
#else
  FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetNQPoints() const
#endif
  {
    return n_q_points_per_element_;
  }

  /**********************************************/
  template<typename VECTOR, int dim>
  unsigned int
#if DEAL_II_VERSION_GTE(9,3,0)
  FaceDataContainer<false, VECTOR, dim>::GetNbrNQPoints() const
#else
  FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetNbrNQPoints() const
#endif
  {
    return n_q_points_per_element_;
  }

  /**********************************************/
  template<typename VECTOR, int dim>
  unsigned int
#if DEAL_II_VERSION_GTE(9,3,0)
  FaceDataContainer<false, VECTOR, dim>::GetMaterialId() const
#else
  FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetMaterialId() const
#endif
  {
    return element_[0]->material_id();
  }

  /**********************************************/
  template<typename VECTOR, int dim>
  unsigned int
#if DEAL_II_VERSION_GTE(9,3,0)
  FaceDataContainer<false, VECTOR, dim>::GetNbrMaterialId() const
#else
  FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetNbrMaterialId() const
#endif
  {
    return this->GetNbrMaterialId(this->GetFace());
  }

  /**********************************************/
  template<typename VECTOR, int dim>
  unsigned int
#if DEAL_II_VERSION_GTE(9,3,0)
  FaceDataContainer<false, VECTOR, dim>::GetNbrMaterialId(
    unsigned int face) const
#else
  FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetNbrMaterialId(
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
                            "FaceDataContainer::GetNbrMaterialId");
      }
  }

  /**********************************************/
  template<typename VECTOR, int dim>
  bool
#if DEAL_II_VERSION_GTE(9,3,0)
  FaceDataContainer<false, VECTOR, dim>::GetIsAtBoundary() const
#else
  FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetIsAtBoundary() const
#endif
  {
    return element_[0]->face(this->GetFace())->at_boundary();
  }

  /**********************************************/
  template<typename VECTOR, int dim>
  double
#if DEAL_II_VERSION_GTE(9,3,0)
  FaceDataContainer<false, VECTOR, dim>::GetElementDiameter() const
#else
  FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetElementDiameter() const
#endif
  {
//      return element_[0]->face(this->GetFace())->diameter();
    return element_[0]->diameter();
  }
  /**********************************************/
  template<typename VECTOR, int dim>
  Point<dim>
#if DEAL_II_VERSION_GTE(9,3,0)
  FaceDataContainer<false, VECTOR, dim>::GetCenter() const
#else
  FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetCenter() const
#endif
  {
    return element_[0]->face(this->GetFace())->center();
  }

  /**********************************************/

  template<typename VECTOR, int dim>
  unsigned int
#if DEAL_II_VERSION_GTE(9,3,0)
  FaceDataContainer<false, VECTOR, dim>::GetBoundaryIndicator() const
#else
  FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetBoundaryIndicator() const
#endif
  {
    return element_[0]->face(this->GetFace())->boundary_id();
  }

  /**********************************************/
  template<typename VECTOR, int dim>
  const FEFaceValuesBase<dim> &
#if DEAL_II_VERSION_GTE(9,3,0)
  FaceDataContainer<false, VECTOR, dim>::GetFEFaceValuesState() const
#else
  FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetFEFaceValuesState() const
#endif
  {
    return *state_fe_values_ptr_;
  }

  /**********************************************/
  template<typename VECTOR, int dim>
  const FEFaceValuesBase<dim> &
#if DEAL_II_VERSION_GTE(9,3,0)
  FaceDataContainer<false, VECTOR, dim>::GetFEFaceValuesControl() const
#else
  FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetFEFaceValuesControl() const
#endif
  {
    return *control_fe_values_ptr_;
  }
  /**********************************************/
  template<typename VECTOR, int dim>
  const FEFaceValuesBase<dim> &
#if DEAL_II_VERSION_GTE(9,3,0)
  FaceDataContainer<false, VECTOR, dim>::GetNbrFEFaceValuesState() const
#else
  FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetNbrFEFaceValuesState() const
#endif
  {
    return *nbr_state_fe_values_ptr_;
  }

  /**********************************************/
  template<typename VECTOR, int dim>
  const FEFaceValuesBase<dim> &
#if DEAL_II_VERSION_GTE(9,3,0)
  FaceDataContainer<false, VECTOR, dim>::GetNbrFEFaceValuesControl() const
#else
  FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetNbrFEFaceValuesControl() const
#endif
  {
    return *nbr_control_fe_values_ptr_;
  }

  template<typename VECTOR, int dim>
  unsigned int
#if DEAL_II_VERSION_GTE(9,3,0)
  FaceDataContainer<false, VECTOR, dim>::GetStateIndex() const
#else
  FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetStateIndex() const
#endif
  {
    return state_index_;
  }

  /***********************************************************************/

  template<typename VECTOR, int dim>
  unsigned int
#if DEAL_II_VERSION_GTE(9,3,0)
  FaceDataContainer<false, VECTOR, dim>::GetControlIndex() const
#else
  FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetControlIndex() const
#endif
  {
    return control_index_;
  }

  /***********************************************************************/
  /************************END*OF*IMPLEMENTATION**************************/
  /***********************************************************************/

  /***********************************************************************/
  /*****************IMPLEMENTATION for hp::DoFHandler*********************/
  /***********************************************************************/





  template<typename VECTOR, int dim>
  void
#if DEAL_II_VERSION_GTE(9,3,0)
  FaceDataContainer<true, VECTOR, dim>::ReInit(
    unsigned int face_no)
#else
  FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::ReInit(
    unsigned int face_no)
#endif
  {
    this->SetFace(face_no);
    state_hp_fe_values_.reinit(element_[this->GetStateIndex()], face_no);
    state_hp_fe_values_ptr_ = &state_hp_fe_values_.get_present_fe_values();
    //Make sure that the Control must be initialized.
    if (this->GetControlIndex() < element_.size())
      {
        control_hp_fe_values_.reinit(element_[this->GetControlIndex()], face_no);
        control_hp_fe_values_ptr_ =
          &control_hp_fe_values_.get_present_fe_values();
      }
  }

  /***********************************************************************/

  template<typename VECTOR, int dim>
  void
#if DEAL_II_VERSION_GTE(9,3,0)
  FaceDataContainer<true, VECTOR, dim>::ReInit(
    unsigned int face_no, unsigned int subface_no)
#else
  FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::ReInit(
    unsigned int face_no, unsigned int subface_no)
#endif
  {
    this->SetFace(face_no);
    this->SetSubFace(subface_no);
    state_hp_fe_subface_values_->reinit(element_[this->GetStateIndex()],
                                        this->GetFace(), this->GetSubFace());
    state_hp_fe_values_ptr_ =
      &state_hp_fe_subface_values_->get_present_fe_values();
    //Make sure that the Control must be initialized.
    if (this->GetControlIndex() < element_.size())
      {
        control_hp_fe_subface_values_->reinit(element_[this->GetControlIndex()],
                                              this->GetFace(), this->GetSubFace());
        control_hp_fe_values_ptr_ =
          &control_hp_fe_subface_values_->get_present_fe_values();
      }
  }

  /***********************************************************************/
  template<typename VECTOR, int dim>
  void
#if DEAL_II_VERSION_GTE(9,3,0)
  FaceDataContainer<true, VECTOR, dim>::ReInitNbr()
#else
  FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::ReInitNbr()
#endif
  {
    Assert(this->NeedNeighbour(), ExcInternalError());
    Assert(
      element_[this->GetStateIndex()]->neighbor_index(this->GetFace()) != -1,
      TriaAccessorExceptions::ExcCellNotUsed())

    if (element_[this->GetStateIndex()]->neighbor(this->GetFace())->has_children())
      {
        //if neighbor is more refined
        const auto neighbor_child =
          element_[this->GetStateIndex()]->neighbor_child_on_subface(
            this->GetFace(), this->GetSubFace());

        // some sanity checks: Check, that the face and subface match and that the neighbour child
        // is not more refined.
        Assert(
          neighbor_child->face(element_[this->GetStateIndex()]->neighbor_of_neighbor(this->GetFace())) == element_[this->GetStateIndex()]->face(this->GetFace())->child(this->GetSubFace()),
          ExcInternalError());
        Assert(neighbor_child->has_children() == false, ExcInternalError());

        nbr_state_hp_fe_values_->reinit(neighbor_child,
                                        element_[this->GetStateIndex()]->neighbor_of_neighbor(
                                          this->GetFace()));
        nbr_state_hp_fe_values_ptr_ =
          &nbr_state_hp_fe_values_->get_present_fe_values();

        //Make sure that the Control must be initialized.
        if (this->GetControlIndex() < element_.size())
          {
            const auto control_neighbor_child =
              element_[this->GetControlIndex()]->neighbor_child_on_subface(
                this->GetFace(), this->GetSubFace());

            nbr_control_hp_fe_values_->reinit(control_neighbor_child,
                                              element_[this->GetControlIndex()]->neighbor_of_neighbor(
                                                this->GetFace()));
            nbr_control_hp_fe_values_ptr_ =
              &nbr_control_hp_fe_values_->get_present_fe_values();
          }
      }
    else if (element_[this->GetStateIndex()]->neighbor_is_coarser(
               this->GetFace()))
      {
        //if the neighbour is coarser
        Assert(
          element_[this->GetStateIndex()]->neighbor(this->GetFace())->level() == element_[this->GetStateIndex()]->level()-1,
          ExcInternalError());
        const auto neighbor = element_[this->GetStateIndex()]->neighbor(
                                this->GetFace());
        const std::pair<unsigned int, unsigned int> faceno_subfaceno =
          element_[this->GetStateIndex()]->neighbor_of_coarser_neighbor(
            this->GetFace());
        const unsigned int neighbor_face_no = faceno_subfaceno.first,
                           neighbor_subface_no = faceno_subfaceno.second;
        state_hp_fe_subface_values_->reinit(neighbor, neighbor_face_no,
                                            neighbor_subface_no);
        nbr_state_hp_fe_values_ptr_ =
          &state_hp_fe_subface_values_->get_present_fe_values();
        if (this->GetControlIndex() < element_.size())
          {
            const auto control_neighbor =
              element_[this->GetControlIndex()]->neighbor(this->GetFace());
            const std::pair<unsigned int, unsigned int> control_faceno_subfaceno =
              element_[this->GetControlIndex()]->neighbor_of_coarser_neighbor(
                this->GetFace());
            const unsigned int control_neighbor_face_no =
              control_faceno_subfaceno.first, control_neighbor_subface_no =
                control_faceno_subfaceno.second;

            control_hp_fe_subface_values_->reinit(control_neighbor,
                                                  control_neighbor_face_no, control_neighbor_subface_no);
            nbr_control_hp_fe_values_ptr_ =
              &control_hp_fe_subface_values_->get_present_fe_values();
          }
      }
    else
      {
        const auto neighbor_state = element_[this->GetStateIndex()]->neighbor(
                                      this->GetFace());
        // neighbor element is as much refined as the
        Assert(neighbor_state->level() == element_[this->GetStateIndex()]->level(),
               ExcInternalError());
        nbr_state_hp_fe_values_->reinit(neighbor_state,
                                        element_[this->GetStateIndex()]->neighbor_of_neighbor(
                                          this->GetFace()));
        nbr_state_hp_fe_values_ptr_ =
          &nbr_state_hp_fe_values_->get_present_fe_values();

        //Make sure that the Control must be initialized.
        if (this->GetControlIndex() < element_.size())
          {
            nbr_control_hp_fe_values_->reinit(
              element_[this->GetControlIndex()]->neighbor(this->GetFace()),
              element_[this->GetControlIndex()]->neighbor_of_neighbor(
                this->GetFace()));
            nbr_control_hp_fe_values_ptr_ =
              &nbr_control_hp_fe_values_->get_present_fe_values();
          }
      }
  }

  /*********************************************/

  template<typename VECTOR, int dim>
  unsigned int
#if DEAL_II_VERSION_GTE(9,3,0)
  FaceDataContainer<true, VECTOR, dim>::GetNDoFsPerElement() const
#else
  FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetNDoFsPerElement() const
#endif
  {
    return element_[0]->get_fe().dofs_per_cell;
  }

  /*********************************************/

  template<typename VECTOR, int dim>
  unsigned int
#if DEAL_II_VERSION_GTE(9,3,0)
  FaceDataContainer<true, VECTOR, dim>::GetNbrNDoFsPerElement() const
#else
  FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetNbrNDoFsPerElement() const
#endif
  {
    if (element_[0]->neighbor_index(this->GetFace()) != -1)
      return element_[0]->neighbor(this->GetFace())->get_fe().dofs_per_cell;
    else
      {
        std::stringstream out;
        out << "There is no neighbor with number " << this->GetFace();
        throw DOpEException(out.str(),
                            "HpFaceDataContainer::GetNbrNDoFsPerElement");
      }
  }
  /*********************************************/
  template<typename VECTOR, int dim>
  unsigned int
#if DEAL_II_VERSION_GTE(9,3,0)
  FaceDataContainer<true, VECTOR, dim>::GetNQPoints() const
#else
  FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetNQPoints() const
#endif
  {
    return q_collection_[element_[0]->active_fe_index()].size();
  }

  /*********************************************/
  template<typename VECTOR, int dim>
  unsigned int
#if DEAL_II_VERSION_GTE(9,3,0)
  FaceDataContainer<true, VECTOR, dim>::GetNbrNQPoints() const
#else
  FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetNbrNQPoints() const
#endif
  {
    if (element_[0]->neighbor_index(this->GetFace()) != -1)
      return q_collection_[element_[0]->neighbor(this->GetFace())->active_fe_index()].size();
    else
      {
        std::stringstream out;
        out << "There is no neighbor with number " << this->GetFace();
        throw DOpEException(out.str(),
                            "HpFaceDataContainer::GetNbrNQPoints");
      }
  }
  /*********************************************/

  template<typename VECTOR, int dim>
  unsigned int
#if DEAL_II_VERSION_GTE(9,3,0)
  FaceDataContainer<true, VECTOR, dim>::GetMaterialId() const
#else
  FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetMaterialId() const
#endif
  {
    return element_[0]->material_id();
  }
  /*********************************************/
  template<typename VECTOR, int dim>
  unsigned int
#if DEAL_II_VERSION_GTE(9,3,0)
  FaceDataContainer<true, VECTOR, dim>::GetNbrMaterialId() const
#else
  FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetNbrMaterialId() const
#endif
  {
    return this->GetNbrMaterialId(this->GetFace());
  }

  /*********************************************/

  template<typename VECTOR, int dim>
  unsigned int
#if DEAL_II_VERSION_GTE(9,3,0)
  FaceDataContainer<true, VECTOR, dim>::GetNbrMaterialId(
    unsigned int face) const
#else
  FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetNbrMaterialId(
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
                            "HpFaceDataContainer::GetNbrMaterialId");
      }
  }

  /*********************************************/

  template<typename VECTOR, int dim>
  double
#if DEAL_II_VERSION_GTE(9,3,0)
  FaceDataContainer<true, VECTOR, dim>::GetElementDiameter() const
#else
  FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetElementDiameter() const
#endif
  {
//      return element_[0]->face(this->GetFace())->diameter();
    return element_[0]->diameter();
  }
  /**********************************************/
  template<typename VECTOR, int dim>
  Point<dim>
#if DEAL_II_VERSION_GTE(9,3,0)
  FaceDataContainer<true, VECTOR, dim>::GetCenter() const
#else
  FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetCenter() const
#endif
  {
    return element_[0]->face(this->GetFace())->center();
  }

  /**********************************************/

  template<typename VECTOR, int dim>
  bool
#if DEAL_II_VERSION_GTE(9,3,0)
  FaceDataContainer<true, VECTOR, dim>::GetIsAtBoundary() const
#else
  FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetIsAtBoundary() const
#endif
  {
    return element_[0]->face(this->GetFace())->at_boundary();
  }

  /**********************************************/
  template<typename VECTOR, int dim>
  unsigned int
#if DEAL_II_VERSION_GTE(9,3,0)
  FaceDataContainer<true, VECTOR, dim>::GetBoundaryIndicator() const
#else
  FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetBoundaryIndicator() const
#endif
  {
    return element_[0]->face(this->GetFace())->boundary_id();
  }

  /*********************************************/
  template<typename VECTOR, int dim>
  const FEFaceValuesBase<dim> &
#if DEAL_II_VERSION_GTE(9,3,0)
  FaceDataContainer<true, VECTOR, dim>::GetFEFaceValuesState() const
#else
  FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetFEFaceValuesState() const
#endif
  {
    return *state_hp_fe_values_ptr_;
  }
  /*********************************************/
  template<typename VECTOR, int dim>
  const FEFaceValuesBase<dim> &
#if DEAL_II_VERSION_GTE(9,3,0)
  FaceDataContainer<true, VECTOR, dim>::GetFEFaceValuesControl() const
#else
  FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetFEFaceValuesControl() const
#endif
  {
    return *control_hp_fe_values_ptr_;
  }
  /*********************************************/
  template<typename VECTOR, int dim>
  const FEFaceValuesBase<dim> &
#if DEAL_II_VERSION_GTE(9,3,0)
  FaceDataContainer<true, VECTOR, dim>::GetNbrFEFaceValuesState() const
#else
  FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetNbrFEFaceValuesState() const
#endif
  {
    return *nbr_state_hp_fe_values_ptr_;
  }
  /*********************************************/
  template<typename VECTOR, int dim>
  const FEFaceValuesBase<dim> &
#if DEAL_II_VERSION_GTE(9,3,0)
  FaceDataContainer<true, VECTOR, dim>::GetNbrFEFaceValuesControl() const
#else
  FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetNbrFEFaceValuesControl() const
#endif
  {
    return *nbr_control_hp_fe_values_ptr_;
  }

  /**********************************************/
  template<typename VECTOR, int dim>
  unsigned int
#if DEAL_II_VERSION_GTE(9,3,0)
  FaceDataContainer<true, VECTOR, dim>::GetStateIndex() const
#else
  FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetStateIndex() const
#endif
  {
    return state_index_;
  }
  /*********************************************/
  template<typename VECTOR, int dim>
  unsigned int
#if DEAL_II_VERSION_GTE(9,3,0)
  FaceDataContainer<true, VECTOR, dim>::GetControlIndex() const
#else
  FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetControlIndex() const
#endif
  {
    return control_index_;
  }
} //end of namespace

#endif /* FACEDATACONTAINER_H_ */
