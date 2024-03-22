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

#ifndef NETWORK_FACEDATACONTAINER_H_
#define NETWORK_FACEDATACONTAINER_H_

#include <basic/spacetimehandler.h>
#include <basic/statespacetimehandler.h>
#include <wrapper/fevalues_wrapper.h>
#include <include/dopeexception.h>
#include <network/network_facedatacontainer_internal.h>

#include <sstream>

#include <deal.II/dofs/dof_handler.h>

using namespace dealii;

namespace DOpE
{
  namespace Networks
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
    class Network_FaceDataContainer : public fdcinternal::Network_FaceDataContainerInternal<
      dim>
    {
    public:
      Network_FaceDataContainer()
      {
        throw (DOpEException(
                 "Dummy class, this constructor should never get called.",
                 "ElementDataContainer<dealii::DoFHandler , VECTOR, dim>::ElementDataContainer"));
      }
    };

    /**
     * This two classes hold all the information we need in the integrator to
     * integrate something over a face of a element (could be a functional, a PDE, etc.).
     * Of particular importance: This class holds the (Sub)FaceFEValues objects.
     *
     * @template dim        1+ the dimension of the integral we are actually interested in.//TODO 1+??
     */

    template<typename VECTOR, int dim>
#if DEAL_II_VERSION_GTE(9,3,0)
    class Network_FaceDataContainer<false, VECTOR, dim> : public fdcinternal::Network_FaceDataContainerInternal<dim>
#else
    class Network_FaceDataContainer<dealii::DoFHandler, VECTOR, dim> : public fdcinternal::Network_FaceDataContainerInternal<dim>
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
      Network_FaceDataContainer(unsigned int pipe,
                                unsigned int n_pipes,
                                unsigned int n_comp,
                                const Quadrature<dim - 1>& quad,
                                UpdateFlags update_flags,
#if DEAL_II_VERSION_GTE(9,3,0)
                                SpaceTimeHandler<FE, false, SPARSITYPATTERN, dealii::Vector<double>,
                                dopedim, dealdim> &sth,
#else
                                SpaceTimeHandler<FE, dealii::DoFHandler, SPARSITYPATTERN, dealii::Vector<double>,
                                dopedim, dealdim> &sth,
#endif
#if DEAL_II_VERSION_GTE(9,3,0)
                                const std::vector<typename DOpEWrapper::DoFHandler<dim>::active_cell_iterator> &element,
#else
                                const std::vector<typename DOpEWrapper::DoFHandler<dim, dealii::DoFHandler>::active_cell_iterator> &element,
#endif
                                const std::map<std::string, const Vector<double>*> &param_values,
                                const std::map<std::string, const dealii::BlockVector<double> *> &domain_values,
                                bool need_neighbour) :
        fdcinternal::Network_FaceDataContainerInternal<dim>(pipe, n_pipes, n_comp, param_values,
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
      Network_FaceDataContainer(unsigned int pipe,
                                unsigned int n_pipes,
                                unsigned int n_comp,
                                const Quadrature<dim - 1>& quad,
                                UpdateFlags update_flags,
#if DEAL_II_VERSION_GTE(9,3,0)
                                StateSpaceTimeHandler<FE, false, SPARSITYPATTERN, dealii::Vector<double>,
                                dim> &sth,
#else
                                StateSpaceTimeHandler<FE, dealii::DoFHandler, SPARSITYPATTERN, dealii::Vector<double>,
                                dim> &sth,
#endif
#if DEAL_II_VERSION_GTE(9,3,0)
                                const std::vector<typename DOpEWrapper::DoFHandler<dim>::active_cell_iterator> &element,
#else
                                const std::vector<typename DOpEWrapper::DoFHandler<dim, dealii::DoFHandler>::active_cell_iterator> &element,
#endif
                                const std::map<std::string, const Vector<double>*> &param_values,
                                const std::map<std::string, const dealii::BlockVector<double> *> &domain_values,
                                bool need_neighbour) :
        fdcinternal::Network_FaceDataContainerInternal<dim>(pipe, n_pipes, n_comp, param_values,
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

      ~Network_FaceDataContainer()
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
      GetFEFaceValuesState() const override;
      inline const FEFaceValuesBase<dim> &
      GetFEFaceValuesControl() const override;

      inline const FEFaceValuesBase<dim> &
      GetNbrFEFaceValuesState() const override;
      inline const FEFaceValuesBase<dim> &
      GetNbrFEFaceValuesControl() const override;

      /**
       * Writes the values of the flux values at the pipe boundary at the quadrature points
       * into values. Fails if called on non boundary nodes.
       */
      void
      GetFluxValues(std::string name,
                    std::vector<double> &values) const;

      /*********************************************/
      /*
       * Same as above for the Vector valued case.
       */
      void
      GetFluxValues(std::string name,
                    std::vector<dealii::Vector<double> > &values) const;

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

      const std::vector<typename DOpEWrapper::DoFHandler<dim>::active_cell_iterator> &element_;
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
      bool sanity_check(const
                        typename DOpEWrapper::DoFHandler<dim>::active_cell_iterator &element_,
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
      bool sanity_check<1>(const
                           typename DOpEWrapper::DoFHandler<1>::active_cell_iterator &,
                           unsigned int,
                           unsigned int)
      {
        return  true;
      }
#else
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
    Network_FaceDataContainer<false, VECTOR, dim>::ReInit(
#else
    Network_FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::ReInit(
#endif
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
#if DEAL_II_VERSION_GTE(9,3,0)
    Network_FaceDataContainer<false, VECTOR, dim>::ReInit(
#else
    Network_FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::ReInit(
#endif
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
#if DEAL_II_VERSION_GTE(9,3,0)
    Network_FaceDataContainer<false, VECTOR, dim>::ReInitNbr()
#else
    Network_FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::ReInitNbr()
#endif
    {
      Assert(this->NeedNeighbour(), ExcInternalError());
      Assert(
        element_[this->GetStateIndex()]->neighbor_index(this->GetFace()) != -1,
        TriaAccessorExceptions::ExcCellNotUsed());

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
    Network_FaceDataContainer<false, VECTOR, dim>::GetNDoFsPerElement() const
#else
    Network_FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetNDoFsPerElement() const
#endif
    {
      return n_dofs_per_element_;
    }

    /***********************************************************************/

    template<typename VECTOR, int dim>
    unsigned int
#if DEAL_II_VERSION_GTE(9,3,0)
    Network_FaceDataContainer<false, VECTOR, dim>::GetNbrNDoFsPerElement() const
#else
    Network_FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetNbrNDoFsPerElement() const
#endif
    {
      return n_dofs_per_element_;
    }

    /**********************************************/
    template<typename VECTOR, int dim>
    unsigned int
#if DEAL_II_VERSION_GTE(9,3,0)
    Network_FaceDataContainer<false, VECTOR, dim>::GetNQPoints() const
#else
    Network_FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetNQPoints() const
#endif
    {
      return n_q_points_per_element_;
    }

    /**********************************************/
    template<typename VECTOR, int dim>
    unsigned int
#if DEAL_II_VERSION_GTE(9,3,0)
    Network_FaceDataContainer<false, VECTOR, dim>::GetNbrNQPoints() const
#else
    Network_FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetNbrNQPoints() const
#endif
    {
      return n_q_points_per_element_;
    }

    /**********************************************/
    template<typename VECTOR, int dim>
    unsigned int
#if DEAL_II_VERSION_GTE(9,3,0)
    Network_FaceDataContainer<false, VECTOR, dim>::GetMaterialId() const
#else
    Network_FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetMaterialId() const
#endif
    {
      return element_[0]->material_id();
    }

    /**********************************************/
    template<typename VECTOR, int dim>
    unsigned int
#if DEAL_II_VERSION_GTE(9,3,0)
    Network_FaceDataContainer<false, VECTOR, dim>::GetNbrMaterialId() const
#else
    Network_FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetNbrMaterialId() const
#endif
    {
      return this->GetNbrMaterialId(this->GetFace());
    }

    /**********************************************/
    template<typename VECTOR, int dim>
    unsigned int
#if DEAL_II_VERSION_GTE(9,3,0)
    Network_FaceDataContainer<false, VECTOR, dim>::GetNbrMaterialId(
#else
    Network_FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetNbrMaterialId(
#endif
      unsigned int face) const
    {
      if (element_[0]->neighbor_index(face) != -1)
        return element_[0]->neighbor(face)->material_id();
      else
        {
          std::stringstream out;
          out << "There is no neighbor with number " << face;
          throw DOpEException(out.str(),
                              "Network_FaceDataContainer::GetNbrMaterialId");
        }
    }

    /**********************************************/
    template<typename VECTOR, int dim>
    bool
#if DEAL_II_VERSION_GTE(9,3,0)
    Network_FaceDataContainer<false, VECTOR, dim>::GetIsAtBoundary() const
#else
    Network_FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetIsAtBoundary() const
#endif
    {
      return element_[0]->face(this->GetFace())->at_boundary();
    }

    /**********************************************/
    template<typename VECTOR, int dim>
    double
#if DEAL_II_VERSION_GTE(9,3,0)
    Network_FaceDataContainer<false, VECTOR, dim>::GetElementDiameter() const
#else
    Network_FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetElementDiameter() const
#endif
    {
//      return element_[0]->face(this->GetFace())->diameter();
      return element_[0]->diameter();
    }

    /**********************************************/

    template<typename VECTOR, int dim>
    unsigned int
#if DEAL_II_VERSION_GTE(9,3,0)
    Network_FaceDataContainer<false, VECTOR, dim>::GetBoundaryIndicator() const
#else
    Network_FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetBoundaryIndicator() const
#endif
    {
      return element_[0]->face(this->GetFace())->boundary_id();
    }

    /**********************************************/
    template<typename VECTOR, int dim>
    const FEFaceValuesBase<dim> &
#if DEAL_II_VERSION_GTE(9,3,0)
    Network_FaceDataContainer<false, VECTOR, dim>::GetFEFaceValuesState() const
#else
    Network_FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetFEFaceValuesState() const
#endif
    {
      return *state_fe_values_ptr_;
    }

    /**********************************************/
    template<typename VECTOR, int dim>
    const FEFaceValuesBase<dim> &
#if DEAL_II_VERSION_GTE(9,3,0)
    Network_FaceDataContainer<false, VECTOR, dim>::GetFEFaceValuesControl() const
#else
    Network_FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetFEFaceValuesControl() const
#endif
    {
      return *control_fe_values_ptr_;
    }
    /**********************************************/
    template<typename VECTOR, int dim>
    const FEFaceValuesBase<dim> &
#if DEAL_II_VERSION_GTE(9,3,0)
    Network_FaceDataContainer<false, VECTOR, dim>::GetNbrFEFaceValuesState() const
#else
    Network_FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetNbrFEFaceValuesState() const
#endif
    {
      return *nbr_state_fe_values_ptr_;
    }

    /**********************************************/
    template<typename VECTOR, int dim>
    const FEFaceValuesBase<dim> &
#if DEAL_II_VERSION_GTE(9,3,0)
    Network_FaceDataContainer<false, VECTOR, dim>::GetNbrFEFaceValuesControl() const
#else
    Network_FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetNbrFEFaceValuesControl() const
#endif
    {
      return *nbr_control_fe_values_ptr_;
    }

    template<typename VECTOR, int dim>
    unsigned int
#if DEAL_II_VERSION_GTE(9,3,0)
    Network_FaceDataContainer<false, VECTOR, dim>::GetStateIndex() const
#else
    Network_FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetStateIndex() const
#endif
    {
      return state_index_;
    }

    /***********************************************************************/

    template<typename VECTOR, int dim>
    unsigned int
#if DEAL_II_VERSION_GTE(9,3,0)
    Network_FaceDataContainer<false, VECTOR, dim>::GetControlIndex() const
#else
    Network_FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetControlIndex() const
#endif
    {
      return control_index_;
    }

    /***********************************************************************/

    template<typename VECTOR, int dim>
    void
#if DEAL_II_VERSION_GTE(9,3,0)
    Network_FaceDataContainer<false, VECTOR, dim>::GetFluxValues(
#else
    Network_FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetFluxValues(
#endif
      std::string name,
      std::vector<double> &values) const
    {
      typename std::map<std::string, const dealii::BlockVector<double> *>::const_iterator it =
        this->GetDomainValues().find(name);
      if (it == this->GetDomainValues().end())
        {
          throw DOpEException("Did not find " + name,
                              "Network_FaceDataContainerInternal::GetFluxValues");
        }
      assert(this->GetIsAtBoundary());
      assert(values.size() == 1);
      assert(this->GetNComp() == 1);
      assert(it->second->block(this->GetNPipes()).size() == 2*this->GetNPipes());
      if (this->GetBoundaryIndicator()==0)
        {
          //left boundary
          values[0] = it->second->block(this->GetNPipes())[this->GetPipe()*this->GetNComp()];
        }
      else
        {
          //right boundary
          values[0] = it->second->block(this->GetNPipes())[this->GetNPipes()*this->GetNComp()+this->GetPipe()*this->GetNComp()];
        }
    }

    /***********************************************************************/
    template<typename VECTOR, int dim>
    void
#if DEAL_II_VERSION_GTE(9,3,0)
    Network_FaceDataContainer<false, VECTOR, dim>::GetFluxValues(
#else
    Network_FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetFluxValues(
#endif
      std::string name,
      std::vector<dealii::Vector<double> > &values) const
    {
      typename std::map<std::string, const dealii::BlockVector<double> *>::const_iterator it =
        this->GetDomainValues().find(name);
      if (it == this->GetDomainValues().end())
        {
          throw DOpEException("Did not find " + name,
                              "ElementDataContainer::GetValues");
        }
      assert(this->GetIsAtBoundary());
      assert(values.size() == 1);
      assert(values[0].size() == this->GetNComp());
      assert(it->second->block(this->GetNPipes()).size() == 2*this->GetNPipes()*this->GetNComp());
      if (this->GetBoundaryIndicator()==0)
        {
          //left boundary
          for (unsigned int c = 0; c < this->GetNComp(); c++)
            {
              values[0][c] = it->second->block(this->GetNPipes())[this->GetPipe()*this->GetNComp()+c];
            }
        }
      else
        {
          //right boundary
          for (unsigned int c = 0; c < this->GetNComp(); c++)
            {
              values[0][c] = it->second->block(this->GetNPipes())[this->GetNPipes()*this->GetNComp()+this->GetPipe()*this->GetNComp()+c];
            }
        }
    }

    /***********************************************************************/
    /************************END*OF*IMPLEMENTATION**************************/
    /***********************************************************************/
  }
} //end of namespace

#endif /* NETWORK_FACEDATACONTAINER_H_ */
