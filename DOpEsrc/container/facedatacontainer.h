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

#ifndef FACEDATACONTAINER_H_
#define FACEDATACONTAINER_H_

#include <basic/spacetimehandler.h>
#include <basic/statespacetimehandler.h>
#include <wrapper/fevalues_wrapper.h>
#include <include/dopeexception.h>
#include <container/facedatacontainer_internal.h>

#include <sstream>

#include <deal.II/dofs/dof_handler.h>
//#include <deal.II/multigrid/mg_dof_handler.h>
#include <deal.II/hp/dof_handler.h>

using namespace dealii;

namespace DOpE
{
  /**
   * Dummy Template Class, acts as kind of interface. Through template specialization, we
   * distinguish between the 'classic' and the 'hp' case.
   *
   */

  template<template<int, int> class DH, typename VECTOR, int dim>
  class FaceDataContainer : public fdcinternal::FaceDataContainerInternal<
    VECTOR, dim>
  {
  public:
    FaceDataContainer()
    {
      throw (DOpEException(
               "Dummy class, this constructor should never get called.",
               "ElementDataContainer<dealii::DoFHandler , VECTOR, dim>::ElementDataContainer"));
    }
    ;
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
  class FaceDataContainer<dealii::DoFHandler, VECTOR, dim> : public fdcinternal::FaceDataContainerInternal<
    VECTOR, dim>
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
                      SpaceTimeHandler<FE, dealii::DoFHandler, SPARSITYPATTERN, VECTOR,
                      dopedim, dealdim>& sth,
                      const std::vector<
                      typename DOpEWrapper::DoFHandler<dim, dealii::DoFHandler>::active_cell_iterator>& element,
                      const std::map<std::string, const Vector<double>*> &param_values,
                      const std::map<std::string, const VECTOR *> &domain_values,
                      bool need_neighbour) :
      fdcinternal::FaceDataContainerInternal<VECTOR, dim>(param_values,
                                                          domain_values, need_neighbour), element_(element), state_fe_values_(
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
                      StateSpaceTimeHandler<FE, dealii::DoFHandler, SPARSITYPATTERN,
                      VECTOR, dim>& sth,
                      const std::vector<
                      typename DOpEWrapper::DoFHandler<dim, dealii::DoFHandler>::active_cell_iterator>& element,
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

  private:
    /*
     * Helper Functions
     */
    unsigned int
    GetStateIndex() const;
    unsigned int
    GetControlIndex() const;
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

    const std::vector<
    typename DOpEWrapper::DoFHandler<dim, dealii::DoFHandler>::active_cell_iterator> & element_;
    DOpEWrapper::FEFaceValues<dim> state_fe_values_;
    DOpEWrapper::FEFaceValues<dim> control_fe_values_;

    DOpEWrapper::FEFaceValues<dim> *nbr_state_fe_values_;
    DOpEWrapper::FEFaceValues<dim> *nbr_control_fe_values_;

    DOpEWrapper::FESubfaceValues<dim> *state_fe_subface_values_;
    DOpEWrapper::FESubfaceValues<dim> *control_fe_subface_values_;

    dealii::FEFaceValuesBase<dim> *state_fe_values_ptr_;
    dealii::FEFaceValuesBase<dim> *control_fe_values_ptr_;
    dealii::FEFaceValuesBase<dim> *nbr_state_fe_values_ptr_;
    dealii::FEFaceValuesBase<dim> *nbr_control_fe_values_ptr_;

    unsigned int n_q_points_per_element_;
    unsigned int n_dofs_per_element_;
  };





  /****************************************************/
  /* MGDofHandler */
//
//
//  /**
//   * This two classes hold all the information we need in the integrator to
//   * integrate something over a face of a element (could be a functional, a PDE, etc.).
//   * Of particular importance: This class holds the (Sub)FaceFEValues objects.
//   *
//   * @template VECTOR     Type of the vector we use in our computations (i.e. Vector<double> or BlockVector<double>)
//   * @template dim        1+ the dimension of the integral we are actually interested in.//TODO 1+??
//   */
//
//  template<typename VECTOR, int dim>
//    class FaceDataContainer<dealii::MGDoFHandler, VECTOR, dim> : public fdcinternal::FaceDataContainerInternal<
//        VECTOR, dim>
//    {
//
//      public:
//        /**
//         * Constructor. Initializes the FaceFEValues objects.
//         *
//         * @template FE                   The type of Finite Element in use here.
//         * @template SPARSITYPATTERN      The corresponding Sparsitypattern to the class-template VECTOR.
//         * @template dopedim              The dimension of the control variable.
//         * @template dealdim              The dimension of the state variable.
//         *
//         * @param quad                    Reference to the quadrature-rule which we use at the moment.
//         * @param update_flags            The update flags we need to initialize the FEValues obejcts
//         * @param sth                     A reference to the SpaceTimeHandler in use.
//         * @param element                    A vector of element iterators through which we gain most of the needed information (like
//         *                                material_ids, n_dfos, etc.)
//         * @param param_values            A std::map containing parameter data (e.g. non space dependent data). If the control
//         *                                is done by parameters, it is contained in this map at the position "control".
//         * @param domain_values           A std::map containing domain data (e.g. nodal vectors for FE-Functions). If the control
//         *                                is distributed, it is contained in this map at the position "control". The state may always
//         *                                be found in this map at the position "state"
//         * @param need_neighbour          Describes whether we need all the GetNbr (= Get Neighbor) functions.
//         *
//         */
//        template<template<int, int> class FE, typename SPARSITYPATTERN, int dopedim, int dealdim>
//          FaceDataContainer(const Quadrature<dim - 1>& quad,
//              UpdateFlags update_flags,
//              SpaceTimeHandler<FE, dealii::MGDoFHandler, SPARSITYPATTERN,
//                  VECTOR, dopedim, dealdim>& sth,
//              const std::vector<
//                  typename dealii::MGDoFHandler<dim>::active_cell_iterator>& element,
//              const std::map<std::string, const Vector<double>*> &param_values,
//              const std::map<std::string, const VECTOR*> &domain_values,
//              bool need_neighbour)
//              : fdcinternal::FaceDataContainerInternal<VECTOR, dim>(
//                  param_values, domain_values, need_neighbour), element_(element), state_fe_values_(
//                  sth.GetMapping(), (sth.GetFESystem("state")), quad,
//                  update_flags), control_fe_values_(sth.GetMapping(),
//                  (sth.GetFESystem("control")), quad, update_flags)
//          {
//            state_index_ = sth.GetStateIndex();
//            if (state_index_ == 1)
//              control_index_ = 0;
//            else
//              control_index_ = 1;
//
//            if (need_neighbour) //so we need FEFAcevalues etc. for the neighbour too.
//            {
//              nbr_control_fe_values_ = new DOpEWrapper::FEFaceValues<dim>(
//                  sth.GetMapping(), (sth.GetFESystem("control")), quad,
//                  update_flags);
//              control_fe_subface_values_ =
//                  new DOpEWrapper::FESubfaceValues<dim>(sth.GetMapping(),
//                      (sth.GetFESystem("control")), quad, update_flags);
//            }
//            this->PrivateConstructor(quad, update_flags, sth, need_neighbour);
//          }
//        /**
//         * Constructor. Initializes the FaceFEValues objects. For PDE only
//         *
//         * @template SPARSITYPATTERN      The corresponding Sparsitypattern to the class-template VECTOR.
//         *
//         * @param quad                    Reference to the quadrature-rule which we use at the moment.
//         * @param update_flags            The update flags we need to initialize the FEValues obejcts
//         * @param sth                     A reference to the SpaceTimeHandler in use.
//         * @param element                    A vector of element iterators through which we gain most of the needed information (like
//         *                                material_ids, n_dfos, etc.)
//         * @param param_values            A std::map containing parameter data (e.g. non space dependent data). If the control
//         *                                is done by parameters, it is contained in this map at the position "control".
//         * @param domain_values           A std::map containing domain data (e.g. nodal vectors for FE-Functions). If the control
//         *                                is distributed, it is contained in this map at the position "control". The state may always
//         *                                be found in this map at the position "state".
//         * @param need_neighbour          Describes whether we need all the GetNbr (= Get Neighbor) functions.
//         *
//         */
//        template<template<int, int> class FE, typename SPARSITYPATTERN>
//          FaceDataContainer(const Quadrature<dim - 1>& quad,
//              UpdateFlags update_flags,
//              StateSpaceTimeHandler<FE, dealii::MGDoFHandler,
//                  SPARSITYPATTERN, VECTOR, dim>& sth,
//              const std::vector<
//                  typename dealii::MGDoFHandler<dim>::active_cell_iterator>& element,
//              const std::map<std::string, const Vector<double>*> &param_values,
//              const std::map<std::string, const VECTOR*> &domain_values,
//              bool need_neighbour)
//              : fdcinternal::FaceDataContainerInternal<VECTOR, dim>(
//                  param_values, domain_values, need_neighbour), element_(element), state_fe_values_(
//                  sth.GetMapping(), (sth.GetFESystem("state")), quad,
//                  update_flags), control_fe_values_(sth.GetMapping(),
//                  (sth.GetFESystem("state")), quad, update_flags)
//          {
//            state_index_ = sth.GetStateIndex();
//            control_index_ = element.size();
//            n_q_points_per_element_ = quad.size();
//            n_dofs_per_element_ = element[0]->get_fe().dofs_per_cell;
//
//            if (need_neighbour) //so we need FEFAcevalues for the neighbour too.
//            {
//              nbr_control_fe_values_ = new DOpEWrapper::FEFaceValues<dim>(
//                  sth.GetMapping(), (sth.GetFESystem("state")), quad,
//                  update_flags);
//              control_fe_subface_values_ =
//                  new DOpEWrapper::FESubfaceValues<dim>(sth.GetMapping(),
//                      (sth.GetFESystem("state")), quad, update_flags);
//            }
//            this->PrivateConstructor(quad, update_flags, sth, need_neighbour);
//          }
//
//        ~FaceDataContainer()
//        {
//          if (nbr_state_fe_values_ != NULL)
//          {
//            delete nbr_state_fe_values_;
//          }
//          if (nbr_control_fe_values_ != NULL)
//          {
//            delete nbr_control_fe_values_;
//          }
//          if (state_fe_subface_values_ != NULL)
//          {
//            delete state_fe_subface_values_;
//          }
//          if (control_fe_subface_values_ != NULL)
//          {
//            delete control_fe_subface_values_;
//          }
//        }
//        /*********************************************/
//        /*
//         * This function reinitializes the FEFaceValues on the actual face. Should
//         * be called prior to any of the get-functions.
//         *
//         * @param face_no     The 'local number' (i.e. from the perspective of the actual element) of the
//         *                    actual face.
//         */
//        inline void
//        ReInit(unsigned int face_no);
//
//        /*********************************************/
//        /*
//         * This function reinits the FESubfaceValues on the actual subface. Should
//         * be called prior to any of the get-functions.
//         *
//         * @param face_no     The 'local number' (i.e. from the perspective of the actual element) of the
//         *                    actual face.
//         * @param subface_no  The 'local number' (i.e. from the perspective of the actual element) of the
//         *                    actual subface.
//         */
//        inline void
//        ReInit(unsigned int face_no, unsigned int subface_no);
//
//        /*********************************************/
//        /*
//         * This function reinitializes the FE(Sub)FaceValues on the neighbor_element.
//         * This should be called prior to any of the get nbr_functions.
//         * Assumes that ReInit is called prior to this function.
//         */
//        inline void
//        ReInitNbr();
//
//        /*********************************************/
//        /**
//         * Get functions to extract data. They all assume that ReInit
//         * (resp. ReInitNbr for the GetNbr* functions) is executed
//         * before calling them.
//         */
//        inline unsigned int
//        GetNDoFsPerElement() const;
//        inline unsigned int
//        GetNbrNDoFsPerElement() const;
//        inline unsigned int
//        GetNQPoints() const;
//        inline unsigned int
//        GetNbrNQPoints() const;
//        inline unsigned int
//        GetMaterialId() const;
//        inline unsigned int
//        GetNbrMaterialId() const;
//        inline unsigned int
//        GetNbrMaterialId(unsigned int face) const;
//        inline bool
//        GetIsAtBoundary() const;
//        inline double
//        GetElementDiameter() const;
//        inline unsigned int
//        GetBoundaryIndicator() const;
//        inline const FEFaceValuesBase<dim>&
//        GetFEFaceValuesState() const;
//        inline const FEFaceValuesBase<dim>&
//        GetFEFaceValuesControl() const;
//
//        inline const FEFaceValuesBase<dim>&
//        GetNbrFEFaceValuesState() const;
//        inline const FEFaceValuesBase<dim>&
//        GetNbrFEFaceValuesControl() const;
//
//      private:
//        /*
//         * Helper Functions
//         */
//        unsigned int
//        GetStateIndex() const;
//        unsigned int
//        GetControlIndex() const;
//        /**
//         * This function contains common code of the constructors.
//         */
//        template<class STH>
//          void
//          PrivateConstructor(const Quadrature<dim - 1>& quad,
//              UpdateFlags update_flags, STH& sth, bool need_neighbour)
//          {
//            n_q_points_per_element_ = quad.size();
//            n_dofs_per_element_ = element_[0]->get_fe().dofs_per_cell;
//
//            if (need_neighbour) //so we need FEFAcevalues etc. for the neighbour too.
//            {
//              nbr_state_fe_values_ = new DOpEWrapper::FEFaceValues<dim>(
//                  (sth.GetFESystem("state")), quad, update_flags);
//              state_fe_subface_values_ = new DOpEWrapper::FESubfaceValues<dim>(
//                  (sth.GetFESystem("state")), quad, update_flags);
//            }
//            else
//            {
//              nbr_state_fe_values_ = NULL;
//              nbr_control_fe_values_ = NULL;
//              state_fe_subface_values_ = NULL;
//              control_fe_subface_values_ = NULL;
//            }
//            // These will point to the object (i.e. FaceValues or SubfaceValues) we actually use.
//            // With this, we have the same interface to the user independently of the type (i.e. face or subface)
//            state_fe_values_ptr_ = NULL;
//            control_fe_values_ptr_ = NULL;
//            nbr_state_fe_values_ptr_ = NULL;
//            nbr_control_fe_values_ptr_ = NULL;
//          }
//        /***********************************************************/
//        //"global" member data, part of every instantiation
//        unsigned int state_index_;
//        unsigned int control_index_;
//
//        const std::vector<
//            typename DOpEWrapper::DoFHandler<dim, dealii::MGDoFHandler >::active_cell_iterator> & element_;
//        DOpEWrapper::FEFaceValues<dim> state_fe_values_;
//        DOpEWrapper::FEFaceValues<dim> control_fe_values_;
//
//        DOpEWrapper::FEFaceValues<dim>* nbr_state_fe_values_;
//        DOpEWrapper::FEFaceValues<dim>* nbr_control_fe_values_;
//
//        DOpEWrapper::FESubfaceValues<dim>* state_fe_subface_values_;
//        DOpEWrapper::FESubfaceValues<dim>* control_fe_subface_values_;
//
//        dealii::FEFaceValuesBase<dim>* state_fe_values_ptr_;
//        dealii::FEFaceValuesBase<dim>* control_fe_values_ptr_;
//        dealii::FEFaceValuesBase<dim>* nbr_state_fe_values_ptr_;
//        dealii::FEFaceValuesBase<dim>* nbr_control_fe_values_ptr_;
//
//        unsigned int n_q_points_per_element_;
//        unsigned int n_dofs_per_element_;
//    };
//
  /* MGDofHandler */
  /****************************************************/


  template<typename VECTOR, int dim>
  class FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim> : public fdcinternal::FaceDataContainerInternal<
    VECTOR, dim>
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
    FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>(
      const hp::QCollection<dim - 1>& q_collection,
      UpdateFlags update_flags,
      SpaceTimeHandler<FE, dealii::hp::DoFHandler, SPARSITYPATTERN,
      VECTOR, dopedim, dealdim>& sth,
      const std::vector<
      typename DOpEWrapper::DoFHandler<dim, dealii::hp::DoFHandler>::active_cell_iterator>& element,
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
    FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>(
      const hp::QCollection<dim - 1>& q_collection,
      UpdateFlags update_flags,
      StateSpaceTimeHandler<FE, dealii::hp::DoFHandler, SPARSITYPATTERN,
      VECTOR, dealdim>& sth,
      const std::vector<
      typename DOpEWrapper::DoFHandler<dim, dealii::hp::DoFHandler>::active_cell_iterator>& element,
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
    ~FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>()
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
    typename DOpEWrapper::DoFHandler<dim, dealii::hp::DoFHandler>::active_cell_iterator>& element_;

    DOpEWrapper::HpFEFaceValues<dim> state_hp_fe_values_;
    DOpEWrapper::HpFEFaceValues<dim> control_hp_fe_values_;

    DOpEWrapper::HpFEFaceValues<dim> *nbr_state_hp_fe_values_;
    DOpEWrapper::HpFEFaceValues<dim> *nbr_control_hp_fe_values_;

    DOpEWrapper::HpFESubfaceValues<dim> *state_hp_fe_subface_values_;
    DOpEWrapper::HpFESubfaceValues<dim> *control_hp_fe_subface_values_;

    const dealii::FEFaceValuesBase<dim> *state_hp_fe_values_ptr_;
    const dealii::FEFaceValuesBase<dim> *control_hp_fe_values_ptr_;
    const dealii::FEFaceValuesBase<dim> *nbr_state_hp_fe_values_ptr_;
    const dealii::FEFaceValuesBase<dim> *nbr_control_hp_fe_values_ptr_;

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

    template<int dim, template<int, int> class DH>
    bool sanity_check(const
                      typename DOpEWrapper::DoFHandler<dim, DH>::active_cell_iterator &element_,
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
//Reenable warning or unused functions
#pragma GCC diagnostic pop
  }

  template<typename VECTOR, int dim>
  void
  FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::ReInit(
    unsigned int face_no)
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
  FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::ReInit(
    unsigned int face_no, unsigned int subface_no)
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
  FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::ReInitNbr()
  {
    Assert(this->NeedNeighbour(), ExcInternalError());
    Assert(
      element_[this->GetStateIndex()]->neighbor_index(this->GetFace()) != -1,
      TriaAccessorExceptions::ExcUnusedCellAsNeighbor())

    if (element_[this->GetStateIndex()]->neighbor(this->GetFace())->has_children())
      {
        //if neighbor is more refined
        const auto neighbor_child =
          element_[this->GetStateIndex()]->neighbor_child_on_subface(
            this->GetFace(), this->GetSubFace());

        // some sanity checks: Check, that the face and subface match and that the neighbour child
        // is not more refined.
        Assert((sanity_check<dim, dealii::DoFHandler>(element_[this->GetStateIndex()],
                                                      this->GetFace(),
                                                      this->GetSubFace()) == true), ExcInternalError());
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
  FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetNDoFsPerElement() const
  {
    return n_dofs_per_element_;
  }

  /***********************************************************************/

  template<typename VECTOR, int dim>
  unsigned int
  FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetNbrNDoFsPerElement() const
  {
    return n_dofs_per_element_;
  }

  /**********************************************/
  template<typename VECTOR, int dim>
  unsigned int
  FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetNQPoints() const
  {
    return n_q_points_per_element_;
  }

  /**********************************************/
  template<typename VECTOR, int dim>
  unsigned int
  FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetNbrNQPoints() const
  {
    return n_q_points_per_element_;
  }

  /**********************************************/
  template<typename VECTOR, int dim>
  unsigned int
  FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetMaterialId() const
  {
    return element_[0]->material_id();
  }

  /**********************************************/
  template<typename VECTOR, int dim>
  unsigned int
  FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetNbrMaterialId() const
  {
    return this->GetNbrMaterialId(this->GetFace());
  }

  /**********************************************/
  template<typename VECTOR, int dim>
  unsigned int
  FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetNbrMaterialId(
    unsigned int face) const
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
  FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetIsAtBoundary() const
  {
    return element_[0]->face(this->GetFace())->at_boundary();
  }

  /**********************************************/
  template<typename VECTOR, int dim>
  double
  FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetElementDiameter() const
  {
//      return element_[0]->face(this->GetFace())->diameter();
    return element_[0]->diameter();
  }

  /**********************************************/

  template<typename VECTOR, int dim>
  unsigned int
  FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetBoundaryIndicator() const
  {
#if DEAL_II_VERSION_GTE(8,3,0)
    return element_[0]->face(this->GetFace())->boundary_id();
#else
    return element_[0]->face(this->GetFace())->boundary_indicator();
#endif
  }

  /**********************************************/
  template<typename VECTOR, int dim>
  const FEFaceValuesBase<dim> &
  FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetFEFaceValuesState() const
  {
    return *state_fe_values_ptr_;
  }

  /**********************************************/
  template<typename VECTOR, int dim>
  const FEFaceValuesBase<dim> &
  FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetFEFaceValuesControl() const
  {
    return *control_fe_values_ptr_;
  }
  /**********************************************/
  template<typename VECTOR, int dim>
  const FEFaceValuesBase<dim> &
  FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetNbrFEFaceValuesState() const
  {
    return *nbr_state_fe_values_ptr_;
  }

  /**********************************************/
  template<typename VECTOR, int dim>
  const FEFaceValuesBase<dim> &
  FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetNbrFEFaceValuesControl() const
  {
    return *nbr_control_fe_values_ptr_;
  }

  template<typename VECTOR, int dim>
  unsigned int
  FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetStateIndex() const
  {
    return state_index_;
  }

  /***********************************************************************/

  template<typename VECTOR, int dim>
  unsigned int
  FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetControlIndex() const
  {
    return control_index_;
  }

  /***********************************************************************/
  /************************END*OF*IMPLEMENTATION**************************/
  /***********************************************************************/
  /***********************************************************************/
  /*****************IMPLEMENTATION for MGDoFHandler*********************/
  /***********************************************************************/
//
//
//
//  template<typename VECTOR, int dim>
//    void
//    FaceDataContainer<dealii::MGDoFHandler, VECTOR, dim>::ReInit(
//        unsigned int face_no)
//    {
//      this->SetFace(face_no);
//      state_fe_values_.reinit(element_[this->GetStateIndex()], face_no);
//      state_fe_values_ptr_ = &state_fe_values_;
//      //Make sure that the Control must be initialized.
//      if (this->GetControlIndex() < element_.size())
//      {
//        control_fe_values_.reinit(element_[this->GetControlIndex()], face_no);
//        control_fe_values_ptr_ = &control_fe_values_;
//      }
//    }
//
//  /***********************************************************************/
//
//  template<typename VECTOR, int dim>
//    void
//    FaceDataContainer<dealii::MGDoFHandler, VECTOR, dim>::ReInit(
//        unsigned int face_no, unsigned int subface_no)
//    {
//      this->SetFace(face_no);
//      this->SetSubFace(subface_no);
//      state_fe_subface_values_->reinit(element_[this->GetStateIndex()], face_no,
//          subface_no);
//      state_fe_values_ptr_ = state_fe_subface_values_;
//      //Make sure that the Control must be initialized.
//      if (this->GetControlIndex() < element_.size())
//      {
//        control_fe_subface_values_->reinit(element_[this->GetControlIndex()],
//            face_no, this->GetSubFace());
//        control_fe_values_ptr_ = control_fe_subface_values_;
//      }
//    }
//  /***********************************************************************/
//
//  template<typename VECTOR, int dim>
//    void
//    FaceDataContainer<dealii::MGDoFHandler, VECTOR, dim>::ReInitNbr()
//    {
//      Assert(this->NeedNeighbour(), ExcInternalError());
//      Assert(
//          element_[this->GetStateIndex()]->neighbor_index(this->GetFace()) != -1,
//          TriaAccessorExceptions::ExcUnusedCellAsNeighbor())
//
//      if (element_[this->GetStateIndex()]->neighbor(this->GetFace())->has_children())
//      {
//        //if neighbor is more refined
//        const auto neighbor_child =
//            element_[this->GetStateIndex()]->neighbor_child_on_subface(
//                this->GetFace(), this->GetSubFace());
//
//        // some sanity checks: Check, that the face and subface match and that the neighbour child
//        // is not more refined.
//        Assert((sanity_check<dim, dealii::hp::DoFHandler>(element_[this->GetStateIndex()],
//                           this->GetFace(),
//                           this->GetSubFace()) == true), ExcInternalError());
//        Assert(neighbor_child->has_children() == false, ExcInternalError());
//
//        nbr_state_fe_values_->reinit(neighbor_child,
//            element_[this->GetStateIndex()]->neighbor_of_neighbor(
//                this->GetFace()));
//        nbr_state_fe_values_ptr_ = nbr_state_fe_values_;
//
//        //Make sure that the Control must be initialized.
//        if (this->GetControlIndex() < element_.size())
//        {
//          const auto control_neighbor_child =
//              element_[this->GetControlIndex()]->neighbor_child_on_subface(
//                  this->GetFace(), this->GetSubFace());
//
//          nbr_control_fe_values_->reinit(control_neighbor_child,
//              element_[this->GetControlIndex()]->neighbor_of_neighbor(
//                  this->GetFace()));
//          nbr_control_fe_values_ptr_ = nbr_control_fe_values_;
//        }
//      }
//      else if (element_[this->GetStateIndex()]->neighbor_is_coarser(
//          this->GetFace()))
//      {
//        //if the neighbour is coarser
//        Assert(
//            element_[this->GetStateIndex()]->neighbor(this->GetFace())->level() == element_[this->GetStateIndex()]->level()-1,
//            ExcInternalError());
//        const auto neighbor = element_[this->GetStateIndex()]->neighbor(
//            this->GetFace());
//        const std::pair<unsigned int, unsigned int> faceno_subfaceno =
//            element_[this->GetStateIndex()]->neighbor_of_coarser_neighbor(
//                this->GetFace());
//        const unsigned int neighbor_face_no = faceno_subfaceno.first,
//            neighbor_subface_no = faceno_subfaceno.second;
//        state_fe_subface_values_->reinit(neighbor, neighbor_face_no,
//            neighbor_subface_no);
//        nbr_state_fe_values_ptr_ = state_fe_subface_values_;
//        if (this->GetControlIndex() < element_.size())
//        {
//          const auto control_neighbor =
//              element_[this->GetControlIndex()]->neighbor(this->GetFace());
//          const std::pair<unsigned int, unsigned int> control_faceno_subfaceno =
//              element_[this->GetControlIndex()]->neighbor_of_coarser_neighbor(
//                  this->GetFace());
//          const unsigned int control_neighbor_face_no =
//              control_faceno_subfaceno.first, control_neighbor_subface_no =
//              control_faceno_subfaceno.second;
//          control_fe_subface_values_->reinit(control_neighbor,
//              control_neighbor_face_no, control_neighbor_subface_no);
//          nbr_control_fe_values_ptr_ = control_fe_subface_values_;
//        }
//
//      }
//      else
//      {
//        const auto neighbor_state = element_[this->GetStateIndex()]->neighbor(
//            this->GetFace());
//        // neighbor element is as much refined as the
//        Assert(neighbor_state->level() == element_[this->GetStateIndex()]->level(),
//            ExcInternalError());
//        nbr_state_fe_values_->reinit(neighbor_state,
//            element_[this->GetStateIndex()]->neighbor_of_neighbor(
//                this->GetFace()));
//        nbr_state_fe_values_ptr_ = nbr_state_fe_values_;
//
//        //Make sure that the Control must be initialized.
//        if (this->GetControlIndex() < element_.size())
//        {
//          nbr_control_fe_values_->reinit(
//              element_[this->GetControlIndex()]->neighbor(this->GetFace()),
//              element_[this->GetControlIndex()]->neighbor_of_neighbor(
//                  this->GetFace()));
//          nbr_control_fe_values_ptr_ = nbr_control_fe_values_;
//        }
//      }
//    }
//  /***********************************************************************/
//  template<typename VECTOR, int dim>
//    unsigned int
//    FaceDataContainer<dealii::MGDoFHandler, VECTOR, dim>::GetNDoFsPerElement() const
//    {
//      return n_dofs_per_element_;
//    }
//
//  /***********************************************************************/
//
//  template<typename VECTOR, int dim>
//    unsigned int
//    FaceDataContainer<dealii::MGDoFHandler, VECTOR, dim>::GetNbrNDoFsPerElement() const
//    {
//      return n_dofs_per_element_;
//    }
//
//  /**********************************************/
//  template<typename VECTOR, int dim>
//    unsigned int
//    FaceDataContainer<dealii::MGDoFHandler, VECTOR, dim>::GetNQPoints() const
//    {
//      return n_q_points_per_element_;
//    }
//
//  /**********************************************/
//  template<typename VECTOR, int dim>
//    unsigned int
//    FaceDataContainer<dealii::MGDoFHandler, VECTOR, dim>::GetNbrNQPoints() const
//    {
//      return n_q_points_per_element_;
//    }
//
//  /**********************************************/
//  template<typename VECTOR, int dim>
//    unsigned int
//    FaceDataContainer<dealii::MGDoFHandler, VECTOR, dim>::GetMaterialId() const
//    {
//      return element_[0]->material_id();
//    }
//
//  /**********************************************/
//  template<typename VECTOR, int dim>
//    unsigned int
//    FaceDataContainer<dealii::MGDoFHandler, VECTOR, dim>::GetNbrMaterialId() const
//    {
//      return this->GetNbrMaterialId(this->GetFace());
//    }
//
//  /**********************************************/
//  template<typename VECTOR, int dim>
//    unsigned int
//    FaceDataContainer<dealii::MGDoFHandler, VECTOR, dim>::GetNbrMaterialId(
//        unsigned int face) const
//    {
//      if (element_[0]->neighbor_index(face) != -1)
//        return element_[0]->neighbor(face)->material_id();
//      else
//       {
//    std::stringstream out;
//    out << "There is no neighbor with number " << face;
//    throw DOpEException(out.str(),
//            "FaceDataContainer::GetNbrMaterialId");
//  }
//    }
//
//  /**********************************************/
//  template<typename VECTOR, int dim>
//    bool
//    FaceDataContainer<dealii::MGDoFHandler, VECTOR, dim>::GetIsAtBoundary() const
//    {
//      return element_[0]->face(this->GetFace())->at_boundary();
//    }
//
//  /**********************************************/
//  template<typename VECTOR, int dim>
//    double
//    FaceDataContainer<dealii::MGDoFHandler, VECTOR, dim>::GetElementDiameter() const
//    {
//      return element_[0]->face(this->GetFace())->diameter();
//    }
//
//  /**********************************************/
//
//  template<typename VECTOR, int dim>
//    unsigned int
//    FaceDataContainer<dealii::MGDoFHandler, VECTOR, dim>::GetBoundaryIndicator() const
//    {
//      return element_[0]->face(this->GetFace())->boundary_indicator();
//    }
//
//  /**********************************************/
//  template<typename VECTOR, int dim>
//    const FEFaceValuesBase<dim>&
//    FaceDataContainer<dealii::MGDoFHandler, VECTOR, dim>::GetFEFaceValuesState() const
//    {
//      return *state_fe_values_ptr_;
//    }
//
//  /**********************************************/
//  template<typename VECTOR, int dim>
//    const FEFaceValuesBase<dim>&
//    FaceDataContainer<dealii::MGDoFHandler, VECTOR, dim>::GetFEFaceValuesControl() const
//    {
//      return *control_fe_values_ptr_;
//    }
//  /**********************************************/
//  template<typename VECTOR, int dim>
//    const FEFaceValuesBase<dim>&
//    FaceDataContainer<dealii::MGDoFHandler, VECTOR, dim>::GetNbrFEFaceValuesState() const
//    {
//      return *nbr_state_fe_values_ptr_;
//    }
//
//  /**********************************************/
//  template<typename VECTOR, int dim>
//    const FEFaceValuesBase<dim>&
//    FaceDataContainer<dealii::MGDoFHandler, VECTOR, dim>::GetNbrFEFaceValuesControl() const
//    {
//      return *nbr_control_fe_values_ptr_;
//    }
//
//  template<typename VECTOR, int dim>
//    unsigned int
//    FaceDataContainer<dealii::MGDoFHandler, VECTOR, dim>::GetStateIndex() const
//    {
//      return state_index_;
//    }
//
//  /***********************************************************************/
//
//  template<typename VECTOR, int dim>
//    unsigned int
//    FaceDataContainer<dealii::MGDoFHandler, VECTOR, dim>::GetControlIndex() const
//    {
//      return control_index_;
//    }
//
  /***********************************************************************/
  /************************END*OF*IMPLEMENTATION**************************/
  /***********************************************************************/
  /***********************************************************************/
  /*****************IMPLEMENTATION for hp::DoFHandler*********************/
  /***********************************************************************/





  template<typename VECTOR, int dim>
  void
  FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::ReInit(
    unsigned int face_no)
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
  FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::ReInit(
    unsigned int face_no, unsigned int subface_no)
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
  FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::ReInitNbr()
  {
    Assert(this->NeedNeighbour(), ExcInternalError());
    Assert(
      element_[this->GetStateIndex()]->neighbor_index(this->GetFace()) != -1,
      TriaAccessorExceptions::ExcUnusedCellAsNeighbor())

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
  FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetNDoFsPerElement() const
  {
    return element_[0]->get_fe().dofs_per_cell;
  }

  /*********************************************/

  template<typename VECTOR, int dim>
  unsigned int
  FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetNbrNDoFsPerElement() const
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
  FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetNQPoints() const
  {
    return q_collection_[element_[0]->active_fe_index()].size();
  }

  /*********************************************/
  template<typename VECTOR, int dim>
  unsigned int
  FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetNbrNQPoints() const
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
  FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetMaterialId() const
  {
    return element_[0]->material_id();
  }
  /*********************************************/
  template<typename VECTOR, int dim>
  unsigned int
  FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetNbrMaterialId() const
  {
    return this->GetNbrMaterialId(this->GetFace());
  }

  /*********************************************/

  template<typename VECTOR, int dim>
  unsigned int
  FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetNbrMaterialId(
    unsigned int face) const
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
  FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetElementDiameter() const
  {
//      return element_[0]->face(this->GetFace())->diameter();
    return element_[0]->diameter();
  }

  /**********************************************/

  template<typename VECTOR, int dim>
  bool
  FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetIsAtBoundary() const
  {
    return element_[0]->face(this->GetFace())->at_boundary();
  }

  /**********************************************/
  template<typename VECTOR, int dim>
  unsigned int
  FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetBoundaryIndicator() const
  {
#if DEAL_II_VERSION_GTE(8,3,0)
    return element_[0]->face(this->GetFace())->boundary_id();
#else
    return element_[0]->face(this->GetFace())->boundary_indicator();
#endif
  }

  /*********************************************/
  template<typename VECTOR, int dim>
  const FEFaceValuesBase<dim> &
  FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetFEFaceValuesState() const
  {
    return *state_hp_fe_values_ptr_;
  }
  /*********************************************/
  template<typename VECTOR, int dim>
  const FEFaceValuesBase<dim> &
  FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetFEFaceValuesControl() const
  {
    return *control_hp_fe_values_ptr_;
  }
  /*********************************************/
  template<typename VECTOR, int dim>
  const FEFaceValuesBase<dim> &
  FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetNbrFEFaceValuesState() const
  {
    return *nbr_state_hp_fe_values_ptr_;
  }
  /*********************************************/
  template<typename VECTOR, int dim>
  const FEFaceValuesBase<dim> &
  FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetNbrFEFaceValuesControl() const
  {
    return *nbr_control_hp_fe_values_ptr_;
  }

  /**********************************************/
  template<typename VECTOR, int dim>
  unsigned int
  FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetStateIndex() const
  {
    return state_index_;
  }
  /*********************************************/
  template<typename VECTOR, int dim>
  unsigned int
  FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetControlIndex() const
  {
    return control_index_;
  }
} //end of namespace

#endif /* FACEDATACONTAINER_H_ */
