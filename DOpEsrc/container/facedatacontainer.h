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

#ifndef FACEDATACONTAINER_H_
#define FACEDATACONTAINER_H_

#include "spacetimehandler.h"
#include "statespacetimehandler.h"
#include "fevalues_wrapper.h"
#include "dopeexception.h"
#include "facedatacontainer_internal.h"

#include <dofs/dof_handler.h>
#include <multigrid/mg_dof_handler.h>
#include <hp/dof_handler.h>

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
          throw(DOpEException(
              "Dummy class, this constructor should never get called.",
              "CellDataContainer<dealii::DoFHandler , VECTOR, dim>::CellDataContainer"));
        }
        ;
    };

  /**
   * This two classes hold all the information we need in the integrator to
   * integrate something over a face of a cell (could be a functional, a PDE, etc.).
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
         * @param cell                    A vector of cell iterators through which we gain most of the needed information (like
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
                  typename DOpEWrapper::DoFHandler<dim, dealii::DoFHandler>::active_cell_iterator>& cell,
              const std::map<std::string, const Vector<double>*> &param_values,
              const std::map<std::string, const VECTOR*> &domain_values,
              bool need_neighbour) :
              fdcinternal::FaceDataContainerInternal<VECTOR, dim>(param_values,
                  domain_values, need_neighbour), _cell(cell), _state_fe_values(
                  sth.GetMapping(), (sth.GetFESystem("state")), quad,
                  update_flags), _control_fe_values(sth.GetMapping(),
                  (sth.GetFESystem("control")), quad, update_flags)
          {
            _state_index = sth.GetStateIndex();
            if (_state_index == 1)
              _control_index = 0;
            else
              _control_index = 1;

            if (need_neighbour) //so we need FEFAcevalues etc. for the neighbour too.
            {
              _nbr_control_fe_values = new DOpEWrapper::FEFaceValues<dim>(
                  sth.GetMapping(), (sth.GetFESystem("control")), quad,
                  update_flags);
              _control_fe_subface_values =
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
         * @param cell                    A vector of cell iterators through which we gain most of the needed information (like
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
                  typename DOpEWrapper::DoFHandler<dim, dealii::DoFHandler>::active_cell_iterator>& cell,
              const std::map<std::string, const Vector<double>*> &param_values,
              const std::map<std::string, const VECTOR*> &domain_values,
              bool need_neighbour) :
              fdcinternal::FaceDataContainerInternal<VECTOR, dim>(param_values,
                  domain_values, need_neighbour), _cell(cell), _state_fe_values(
                  sth.GetMapping(), (sth.GetFESystem("state")), quad,
                  update_flags), _control_fe_values(sth.GetMapping(),
                  (sth.GetFESystem("state")), quad, update_flags)
          {
            _state_index = sth.GetStateIndex();
            _control_index = cell.size();
            _n_q_points_per_cell = quad.size();
            _n_dofs_per_cell = cell[0]->get_fe().dofs_per_cell;

            if (need_neighbour) //so we need FEFAcevalues for the neighbour too.
            {
              _nbr_control_fe_values = new DOpEWrapper::FEFaceValues<dim>(
                  sth.GetMapping(), (sth.GetFESystem("state")), quad,
                  update_flags);
              _control_fe_subface_values =
                  new DOpEWrapper::FESubfaceValues<dim>(sth.GetMapping(),
                      (sth.GetFESystem("state")), quad, update_flags);
            }
            this->PrivateConstructor(quad, update_flags, sth, need_neighbour);
          }

        ~FaceDataContainer()
        {
          if (_nbr_state_fe_values != NULL)
          {
            delete _nbr_state_fe_values;
          }
          if (_nbr_control_fe_values != NULL)
          {
            delete _nbr_control_fe_values;
          }
          if (_state_fe_subface_values != NULL)
          {
            delete _state_fe_subface_values;
          }
          if (_control_fe_subface_values != NULL)
          {
            delete _control_fe_subface_values;
          }
        }
        /*********************************************/
        /*
         * This function reinitializes the FEFaceValues on the actual face. Should
         * be called prior to any of the get-functions.
         *
         * @param face_no     The 'local number' (i.e. from the perspective of the actual cell) of the
         *                    actual face.
         */
        inline void
        ReInit(unsigned int face_no);

        /*********************************************/
        /*
         * This function reinits the FESubfaceValues on the actual subface. Should
         * be called prior to any of the get-functions.
         *
         * @param face_no     The 'local number' (i.e. from the perspective of the actual cell) of the
         *                    actual face.
         * @param subface_no  The 'local number' (i.e. from the perspective of the actual cell) of the
         *                    actual subface.
         */
        inline void
        ReInit(unsigned int face_no, unsigned int subface_no);

        /*********************************************/
        /*
         * This function reinitializes the FE(Sub)FaceValues on the neighbor_cell.
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
        GetNDoFsPerCell() const;
        inline unsigned int
        GetNbrNDoFsPerCell() const;
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
        GetCellDiameter() const;
        inline unsigned int
        GetBoundaryIndicator() const;
        inline const FEFaceValuesBase<dim>&
        GetFEFaceValuesState() const;
        inline const FEFaceValuesBase<dim>&
        GetFEFaceValuesControl() const;

        inline const FEFaceValuesBase<dim>&
        GetNbrFEFaceValuesState() const;
        inline const FEFaceValuesBase<dim>&
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
              UpdateFlags update_flags, STH& sth, bool need_neighbour)
          {
            _n_q_points_per_cell = quad.size();
            _n_dofs_per_cell = _cell[0]->get_fe().dofs_per_cell;

            if (need_neighbour) //so we need FEFAcevalues etc. for the neighbour too.
            {
              _nbr_state_fe_values = new DOpEWrapper::FEFaceValues<dim>(
                  (sth.GetFESystem("state")), quad, update_flags);
              _state_fe_subface_values = new DOpEWrapper::FESubfaceValues<dim>(
                  (sth.GetFESystem("state")), quad, update_flags);
            }
            else
            {
              _nbr_state_fe_values = NULL;
              _nbr_control_fe_values = NULL;
              _state_fe_subface_values = NULL;
              _control_fe_subface_values = NULL;
            }
            // These will point to the object (i.e. FaceValues or SubfaceValues) we actually use.
            // With this, we have the same interface to the user independently of the type (i.e. face or subface)
            _state_fe_values_ptr = NULL;
            _control_fe_values_ptr = NULL;
            _nbr_state_fe_values_ptr = NULL;
            _nbr_control_fe_values_ptr = NULL;
          }
        /***********************************************************/
        //"global" member data, part of every instantiation
        unsigned int _state_index;
        unsigned int _control_index;

        const std::vector<
            typename DOpEWrapper::DoFHandler<dim, dealii::DoFHandler>::active_cell_iterator> & _cell;
        DOpEWrapper::FEFaceValues<dim> _state_fe_values;
        DOpEWrapper::FEFaceValues<dim> _control_fe_values;

        DOpEWrapper::FEFaceValues<dim>* _nbr_state_fe_values;
        DOpEWrapper::FEFaceValues<dim>* _nbr_control_fe_values;

        DOpEWrapper::FESubfaceValues<dim>* _state_fe_subface_values;
        DOpEWrapper::FESubfaceValues<dim>* _control_fe_subface_values;

        dealii::FEFaceValuesBase<dim>* _state_fe_values_ptr;
        dealii::FEFaceValuesBase<dim>* _control_fe_values_ptr;
        dealii::FEFaceValuesBase<dim>* _nbr_state_fe_values_ptr;
        dealii::FEFaceValuesBase<dim>* _nbr_control_fe_values_ptr;

        unsigned int _n_q_points_per_cell;
        unsigned int _n_dofs_per_cell;
    };




  
  /****************************************************/
  /* MGDofHandler */


  /**
   * This two classes hold all the information we need in the integrator to
   * integrate something over a face of a cell (could be a functional, a PDE, etc.).
   * Of particular importance: This class holds the (Sub)FaceFEValues objects.
   *
   * @template VECTOR     Type of the vector we use in our computations (i.e. Vector<double> or BlockVector<double>)
   * @template dim        1+ the dimension of the integral we are actually interested in.//TODO 1+??
   */

  template<typename VECTOR, int dim>
    class FaceDataContainer<dealii::MGDoFHandler, VECTOR, dim> : public fdcinternal::FaceDataContainerInternal<
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
         * @param cell                    A vector of cell iterators through which we gain most of the needed information (like
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
              SpaceTimeHandler<FE, dealii::MGDoFHandler, SPARSITYPATTERN,
                  VECTOR, dopedim, dealdim>& sth,
              const std::vector<
                  typename dealii::MGDoFHandler<dim>::active_cell_iterator>& cell,
              const std::map<std::string, const Vector<double>*> &param_values,
              const std::map<std::string, const VECTOR*> &domain_values,
              bool need_neighbour)
              : fdcinternal::FaceDataContainerInternal<VECTOR, dim>(
                  param_values, domain_values, need_neighbour), _cell(cell), _state_fe_values(
                  sth.GetMapping(), (sth.GetFESystem("state")), quad,
                  update_flags), _control_fe_values(sth.GetMapping(),
                  (sth.GetFESystem("control")), quad, update_flags)
          {
            _state_index = sth.GetStateIndex();
            if (_state_index == 1)
              _control_index = 0;
            else
              _control_index = 1;

            if (need_neighbour) //so we need FEFAcevalues etc. for the neighbour too.
            {
              _nbr_control_fe_values = new DOpEWrapper::FEFaceValues<dim>(
                  sth.GetMapping(), (sth.GetFESystem("control")), quad,
                  update_flags);
              _control_fe_subface_values =
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
         * @param cell                    A vector of cell iterators through which we gain most of the needed information (like
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
              StateSpaceTimeHandler<FE, dealii::MGDoFHandler,
                  SPARSITYPATTERN, VECTOR, dim>& sth,
              const std::vector<
                  typename dealii::MGDoFHandler<dim>::active_cell_iterator>& cell,
              const std::map<std::string, const Vector<double>*> &param_values,
              const std::map<std::string, const VECTOR*> &domain_values,
              bool need_neighbour)
              : fdcinternal::FaceDataContainerInternal<VECTOR, dim>(
                  param_values, domain_values, need_neighbour), _cell(cell), _state_fe_values(
                  sth.GetMapping(), (sth.GetFESystem("state")), quad,
                  update_flags), _control_fe_values(sth.GetMapping(),
                  (sth.GetFESystem("state")), quad, update_flags)
          {
            _state_index = sth.GetStateIndex();
            _control_index = cell.size();
            _n_q_points_per_cell = quad.size();
            _n_dofs_per_cell = cell[0]->get_fe().dofs_per_cell;

            if (need_neighbour) //so we need FEFAcevalues for the neighbour too.
            {
              _nbr_control_fe_values = new DOpEWrapper::FEFaceValues<dim>(
                  sth.GetMapping(), (sth.GetFESystem("state")), quad,
                  update_flags);
              _control_fe_subface_values =
                  new DOpEWrapper::FESubfaceValues<dim>(sth.GetMapping(),
                      (sth.GetFESystem("state")), quad, update_flags);
            }
            this->PrivateConstructor(quad, update_flags, sth, need_neighbour);
          }

        ~FaceDataContainer()
        {
          if (_nbr_state_fe_values != NULL)
          {
            delete _nbr_state_fe_values;
          }
          if (_nbr_control_fe_values != NULL)
          {
            delete _nbr_control_fe_values;
          }
          if (_state_fe_subface_values != NULL)
          {
            delete _state_fe_subface_values;
          }
          if (_control_fe_subface_values != NULL)
          {
            delete _control_fe_subface_values;
          }
        }
        /*********************************************/
        /*
         * This function reinitializes the FEFaceValues on the actual face. Should
         * be called prior to any of the get-functions.
         *
         * @param face_no     The 'local number' (i.e. from the perspective of the actual cell) of the
         *                    actual face.
         */
        inline void
        ReInit(unsigned int face_no);

        /*********************************************/
        /*
         * This function reinits the FESubfaceValues on the actual subface. Should
         * be called prior to any of the get-functions.
         *
         * @param face_no     The 'local number' (i.e. from the perspective of the actual cell) of the
         *                    actual face.
         * @param subface_no  The 'local number' (i.e. from the perspective of the actual cell) of the
         *                    actual subface.
         */
        inline void
        ReInit(unsigned int face_no, unsigned int subface_no);

        /*********************************************/
        /*
         * This function reinitializes the FE(Sub)FaceValues on the neighbor_cell.
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
        GetNDoFsPerCell() const;
        inline unsigned int
        GetNbrNDoFsPerCell() const;
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
        GetCellDiameter() const;
        inline unsigned int
        GetBoundaryIndicator() const;
        inline const FEFaceValuesBase<dim>&
        GetFEFaceValuesState() const;
        inline const FEFaceValuesBase<dim>&
        GetFEFaceValuesControl() const;

        inline const FEFaceValuesBase<dim>&
        GetNbrFEFaceValuesState() const;
        inline const FEFaceValuesBase<dim>&
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
              UpdateFlags update_flags, STH& sth, bool need_neighbour)
          {
            _n_q_points_per_cell = quad.size();
            _n_dofs_per_cell = _cell[0]->get_fe().dofs_per_cell;

            if (need_neighbour) //so we need FEFAcevalues etc. for the neighbour too.
            {
              _nbr_state_fe_values = new DOpEWrapper::FEFaceValues<dim>(
                  (sth.GetFESystem("state")), quad, update_flags);
              _state_fe_subface_values = new DOpEWrapper::FESubfaceValues<dim>(
                  (sth.GetFESystem("state")), quad, update_flags);
            }
            else
            {
              _nbr_state_fe_values = NULL;
              _nbr_control_fe_values = NULL;
              _state_fe_subface_values = NULL;
              _control_fe_subface_values = NULL;
            }
            // These will point to the object (i.e. FaceValues or SubfaceValues) we actually use.
            // With this, we have the same interface to the user independently of the type (i.e. face or subface)
            _state_fe_values_ptr = NULL;
            _control_fe_values_ptr = NULL;
            _nbr_state_fe_values_ptr = NULL;
            _nbr_control_fe_values_ptr = NULL;
          }
        /***********************************************************/
        //"global" member data, part of every instantiation
        unsigned int _state_index;
        unsigned int _control_index;

        const std::vector<
            typename DOpEWrapper::DoFHandler<dim, dealii::MGDoFHandler >::active_cell_iterator> & _cell;
        DOpEWrapper::FEFaceValues<dim> _state_fe_values;
        DOpEWrapper::FEFaceValues<dim> _control_fe_values;

        DOpEWrapper::FEFaceValues<dim>* _nbr_state_fe_values;
        DOpEWrapper::FEFaceValues<dim>* _nbr_control_fe_values;

        DOpEWrapper::FESubfaceValues<dim>* _state_fe_subface_values;
        DOpEWrapper::FESubfaceValues<dim>* _control_fe_subface_values;

        dealii::FEFaceValuesBase<dim>* _state_fe_values_ptr;
        dealii::FEFaceValuesBase<dim>* _control_fe_values_ptr;
        dealii::FEFaceValuesBase<dim>* _nbr_state_fe_values_ptr;
        dealii::FEFaceValuesBase<dim>* _nbr_control_fe_values_ptr;

        unsigned int _n_q_points_per_cell;
        unsigned int _n_dofs_per_cell;
    };

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
         * @param cell                    A vector of cell iterators through which we gain most of the needed information (like
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
                  typename DOpEWrapper::DoFHandler<dim, dealii::hp::DoFHandler>::active_cell_iterator>& cell,
              const std::map<std::string, const Vector<double>*> &param_values,
              const std::map<std::string, const VECTOR*> &domain_values,
              bool need_neighbour) :
              fdcinternal::FaceDataContainerInternal<VECTOR, dim>(param_values,
                  domain_values, need_neighbour), _cell(cell), _state_hp_fe_values(
                  (sth.GetFESystem("state")), q_collection, update_flags), _control_hp_fe_values(
                  (sth.GetFESystem("control")), q_collection, update_flags), _q_collection(
                  q_collection)
          {
            _state_index = sth.GetStateIndex();
            if (_state_index == 1)
              _control_index = 0;
            else
              _control_index = 1;

            if (need_neighbour) //so we need FEFAcevalues for the neighbour too.
            {
              _nbr_control_hp_fe_values = new DOpEWrapper::HpFEFaceValues<dim>(
                  sth.GetMapping(), (sth.GetFESystem("control")), q_collection,
                  update_flags);
              _control_hp_fe_subface_values =
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
         * @param cell                    A vector of cell iterators through which we gain most of the needed information (like
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
                  typename DOpEWrapper::DoFHandler<dim, dealii::hp::DoFHandler>::active_cell_iterator>& cell,
              const std::map<std::string, const Vector<double>*> &param_values,
              const std::map<std::string, const VECTOR*> &domain_values,
              bool need_neighbour) :
              fdcinternal::FaceDataContainerInternal<VECTOR, dim>(param_values,
                  domain_values, need_neighbour), _cell(cell), _state_hp_fe_values(
                  (sth.GetFESystem("state")), q_collection, update_flags), _control_hp_fe_values(
                  (sth.GetFESystem("state")), q_collection, update_flags), _q_collection(
                  q_collection)
          {
            _state_index = sth.GetStateIndex();
            if (_state_index == 1)
              _control_index = 0;
            else
              _control_index = 1;

            if (need_neighbour)
            {
              _nbr_control_hp_fe_values = new DOpEWrapper::HpFEFaceValues<dim>(
                  sth.GetMapping(), (sth.GetFESystem("state")), q_collection,
                  update_flags);
              _control_hp_fe_subface_values =
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
          delete _nbr_state_hp_fe_values;
          _nbr_state_hp_fe_values = NULL;
          delete _nbr_control_hp_fe_values;
          _nbr_control_hp_fe_values = NULL;
          delete _state_hp_fe_subface_values;
          _state_hp_fe_subface_values = NULL;
          delete _control_hp_fe_subface_values;
          _control_hp_fe_subface_values = NULL;
        }

        /*********************************************/
        /*
         * This function reinits the FEFaceValues on the actual face. Should
         * be called prior to any of the get-functions.
         *
         * @param face_no     The 'local number' (i.e. from the perspective of the actual cell) of the
         *                    actual face.
         */
        inline void
        ReInit(unsigned int face_no);

        /*********************************************/
        /*
         * This function reinits the FESubfaceValues on the actual subface. Should
         * be called prior to any of the get-functions.
         *
         * @param face_no     The 'local number' (i.e. from the perspective of the actual cell) of the
         *                    actual face.
         * @param subface_no  The 'local number' (i.e. from the perspective of the actual cell) of the
         *                    actual subface.
         */
        inline void
        ReInit(unsigned int face_no, unsigned int subface_no);

        /*********************************************/
        /*
         * This function reinits the FEFaceValues on the neighbor_cell for the
         * case that the neighbor-cell coarser or as fine as the
         * (previously set!) actual cell. This should be called prior
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
        GetNDoFsPerCell() const;

        inline unsigned int
        GetNbrNDoFsPerCell() const;
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
        GetCellDiameter() const;

        inline const FEFaceValuesBase<dim>&
        GetFEFaceValuesState() const;
        inline const FEFaceValuesBase<dim>&
        GetFEFaceValuesControl() const;

        inline const FEFaceValuesBase<dim>&
        GetNbrFEFaceValuesState() const;
        inline const FEFaceValuesBase<dim>&
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
              UpdateFlags update_flags, STH& sth, bool need_neighbour)
          {
            if (need_neighbour) //so we need FEFAcevalues for the neighbour too.
            {
              _nbr_state_hp_fe_values = new DOpEWrapper::HpFEFaceValues<dim>(
                  sth.GetMapping(), (sth.GetFESystem("state")), q_collection,
                  update_flags);
              _state_hp_fe_subface_values = new DOpEWrapper::HpFESubfaceValues<
                  dim>(sth.GetMapping(), (sth.GetFESystem("state")),
                  q_collection, update_flags);

            }
            else
            {
              _nbr_state_hp_fe_values = NULL;
              _nbr_control_hp_fe_values = NULL;
              _state_hp_fe_subface_values = NULL;
              _control_hp_fe_subface_values = NULL;
            }

            _nbr_state_hp_fe_values_ptr = NULL;
            _nbr_control_hp_fe_values_ptr = NULL;
            _state_hp_fe_values_ptr = NULL;
            _control_hp_fe_values_ptr = NULL;
          }
        /***********************************************************/
        //"global" member data, part of every instantiation
        unsigned int _state_index;
        unsigned int _control_index;
        const std::vector<
            typename DOpEWrapper::DoFHandler<dim, dealii::hp::DoFHandler>::active_cell_iterator>& _cell;

        DOpEWrapper::HpFEFaceValues<dim> _state_hp_fe_values;
        DOpEWrapper::HpFEFaceValues<dim> _control_hp_fe_values;

        DOpEWrapper::HpFEFaceValues<dim>* _nbr_state_hp_fe_values;
        DOpEWrapper::HpFEFaceValues<dim>* _nbr_control_hp_fe_values;

        DOpEWrapper::HpFESubfaceValues<dim>* _state_hp_fe_subface_values;
        DOpEWrapper::HpFESubfaceValues<dim>* _control_hp_fe_subface_values;

        const dealii::FEFaceValuesBase<dim>* _state_hp_fe_values_ptr;
        const dealii::FEFaceValuesBase<dim>* _control_hp_fe_values_ptr;
        const dealii::FEFaceValuesBase<dim>* _nbr_state_hp_fe_values_ptr;
        const dealii::FEFaceValuesBase<dim>* _nbr_control_hp_fe_values_ptr;

        const hp::QCollection<dim - 1>& _q_collection;
    };

  /***********************************************************************/
  /************************IMPLEMENTATION*for*DoFHandler*********************************/
  /***********************************************************************/

  template<typename VECTOR, int dim>
    void
    FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::ReInit(
        unsigned int face_no)
    {
      this->SetFace(face_no);
      _state_fe_values.reinit(_cell[this->GetStateIndex()], face_no);
      _state_fe_values_ptr = &_state_fe_values;
      //Make sure that the Control must be initialized.
      if (this->GetControlIndex() < _cell.size())
      {
        _control_fe_values.reinit(_cell[this->GetControlIndex()], face_no);
        _control_fe_values_ptr = &_control_fe_values;
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
      _state_fe_subface_values->reinit(_cell[this->GetStateIndex()], face_no,
          subface_no);
      _state_fe_values_ptr = _state_fe_subface_values;
      //Make sure that the Control must be initialized.
      if (this->GetControlIndex() < _cell.size())
      {
        _control_fe_subface_values->reinit(_cell[this->GetControlIndex()],
            face_no, this->GetSubFace());
        _control_fe_values_ptr = _control_fe_subface_values;
      }
    }
  /***********************************************************************/

  template<typename VECTOR, int dim>
    void
    FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::ReInitNbr()
    {
      Assert(this->NeedNeighbour(), ExcInternalError());
      Assert(
          _cell[this->GetStateIndex()]->neighbor_index(this->GetFace()) != -1,
          TriaAccessorExceptions::ExcUnusedCellAsNeighbor())

      if (_cell[this->GetStateIndex()]->neighbor(this->GetFace())->has_children())
      {
        //if neighbor is more refined
        const auto neighbor_child =
            _cell[this->GetStateIndex()]->neighbor_child_on_subface(
                this->GetFace(), this->GetSubFace());

        // some sanity checks: Check, that the face and subface match and that the neighbour child
        // is not more refined.
        Assert(
            neighbor_child->face(_cell[this->GetStateIndex()]->neighbor_of_neighbor(this->GetFace())) == _cell[this->GetStateIndex()]->face(this->GetFace())->child(this->GetSubFace()),
            ExcInternalError());
        Assert(neighbor_child->has_children() == false, ExcInternalError());

        _nbr_state_fe_values->reinit(neighbor_child,
            _cell[this->GetStateIndex()]->neighbor_of_neighbor(
                this->GetFace()));
        _nbr_state_fe_values_ptr = _nbr_state_fe_values;

        //Make sure that the Control must be initialized.
        if (this->GetControlIndex() < _cell.size())
        {
          const auto control_neighbor_child =
              _cell[this->GetControlIndex()]->neighbor_child_on_subface(
                  this->GetFace(), this->GetSubFace());

          _nbr_control_fe_values->reinit(control_neighbor_child,
              _cell[this->GetControlIndex()]->neighbor_of_neighbor(
                  this->GetFace()));
          _nbr_control_fe_values_ptr = _nbr_control_fe_values;
        }
      }
      else if (_cell[this->GetStateIndex()]->neighbor_is_coarser(
          this->GetFace()))
      {
        //if the neighbour is coarser
        Assert(
            _cell[this->GetStateIndex()]->neighbor(this->GetFace())->level() == _cell[this->GetStateIndex()]->level()-1,
            ExcInternalError());
        const auto neighbor = _cell[this->GetStateIndex()]->neighbor(
            this->GetFace());
        const std::pair<unsigned int, unsigned int> faceno_subfaceno =
            _cell[this->GetStateIndex()]->neighbor_of_coarser_neighbor(
                this->GetFace());
        const unsigned int neighbor_face_no = faceno_subfaceno.first,
            neighbor_subface_no = faceno_subfaceno.second;
        _state_fe_subface_values->reinit(neighbor, neighbor_face_no,
            neighbor_subface_no);
        _nbr_state_fe_values_ptr = _state_fe_subface_values;
        if (this->GetControlIndex() < _cell.size())
        {
          const auto control_neighbor =
              _cell[this->GetControlIndex()]->neighbor(this->GetFace());
          const std::pair<unsigned int, unsigned int> control_faceno_subfaceno =
              _cell[this->GetControlIndex()]->neighbor_of_coarser_neighbor(
                  this->GetFace());
          const unsigned int control_neighbor_face_no =
              control_faceno_subfaceno.first, control_neighbor_subface_no =
              control_faceno_subfaceno.second;
          _control_fe_subface_values->reinit(control_neighbor,
              control_neighbor_face_no, control_neighbor_subface_no);
          _nbr_control_fe_values_ptr = _control_fe_subface_values;
        }

      }
      else
      {
        const auto neighbor_state = _cell[this->GetStateIndex()]->neighbor(
            this->GetFace());
        // neighbor cell is as much refined as the
        Assert(neighbor_state->level() == _cell[this->GetStateIndex()]->level(),
            ExcInternalError());
        _nbr_state_fe_values->reinit(neighbor_state,
            _cell[this->GetStateIndex()]->neighbor_of_neighbor(
                this->GetFace()));
        _nbr_state_fe_values_ptr = _nbr_state_fe_values;

        //Make sure that the Control must be initialized.
        if (this->GetControlIndex() < _cell.size())
        {
          _nbr_control_fe_values->reinit(
              _cell[this->GetControlIndex()]->neighbor(this->GetFace()),
              _cell[this->GetControlIndex()]->neighbor_of_neighbor(
                  this->GetFace()));
          _nbr_control_fe_values_ptr = _nbr_control_fe_values;
        }
      }
    }
  /***********************************************************************/
  template<typename VECTOR, int dim>
    unsigned int
    FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetNDoFsPerCell() const
    {
      return _n_dofs_per_cell;
    }

  /***********************************************************************/

  template<typename VECTOR, int dim>
    unsigned int
    FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetNbrNDoFsPerCell() const
    {
      return _n_dofs_per_cell;
    }

  /**********************************************/
  template<typename VECTOR, int dim>
    unsigned int
    FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetNQPoints() const
    {
      return _n_q_points_per_cell;
    }

  /**********************************************/
  template<typename VECTOR, int dim>
    unsigned int
    FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetNbrNQPoints() const
    {
      return _n_q_points_per_cell;
    }

  /**********************************************/
  template<typename VECTOR, int dim>
    unsigned int
    FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetMaterialId() const
    {
      return _cell[0]->material_id();
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
      if (_cell[0]->neighbor_index(face) != -1)
        return _cell[0]->neighbor(face)->material_id();
      else
        throw DOpEException("There is no neighbor with number " + face,
            "FaceDataContainer::GetNbrMaterialId");
    }

  /**********************************************/
  template<typename VECTOR, int dim>
    bool
    FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetIsAtBoundary() const
    {
      return _cell[0]->face(this->GetFace())->at_boundary();
    }

  /**********************************************/
  template<typename VECTOR, int dim>
    double
    FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetCellDiameter() const
    {
      return _cell[0]->face(this->GetFace())->diameter();
    }

  /**********************************************/

  template<typename VECTOR, int dim>
    unsigned int
    FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetBoundaryIndicator() const
    {
      return _cell[0]->face(this->GetFace())->boundary_indicator();
    }

  /**********************************************/
  template<typename VECTOR, int dim>
    const FEFaceValuesBase<dim>&
    FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetFEFaceValuesState() const
    {
      return *_state_fe_values_ptr;
    }

  /**********************************************/
  template<typename VECTOR, int dim>
    const FEFaceValuesBase<dim>&
    FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetFEFaceValuesControl() const
    {
      return *_control_fe_values_ptr;
    }
  /**********************************************/
  template<typename VECTOR, int dim>
    const FEFaceValuesBase<dim>&
    FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetNbrFEFaceValuesState() const
    {
      return *_nbr_state_fe_values_ptr;
    }

  /**********************************************/
  template<typename VECTOR, int dim>
    const FEFaceValuesBase<dim>&
    FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetNbrFEFaceValuesControl() const
    {
      return *_nbr_control_fe_values_ptr;
    }

  template<typename VECTOR, int dim>
    unsigned int
    FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetStateIndex() const
    {
      return _state_index;
    }

  /***********************************************************************/

  template<typename VECTOR, int dim>
    unsigned int
    FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetControlIndex() const
    {
      return _control_index;
    }

  /***********************************************************************/
  /************************END*OF*IMPLEMENTATION**************************/
  /***********************************************************************/
  /***********************************************************************/
  /*****************IMPLEMENTATION for MGDoFHandler*********************/
  /***********************************************************************/



  template<typename VECTOR, int dim>
    void
    FaceDataContainer<dealii::MGDoFHandler, VECTOR, dim>::ReInit(
        unsigned int face_no)
    {
      this->SetFace(face_no);
      _state_fe_values.reinit(_cell[this->GetStateIndex()], face_no);
      _state_fe_values_ptr = &_state_fe_values;
      //Make sure that the Control must be initialized.
      if (this->GetControlIndex() < _cell.size())
      {
        _control_fe_values.reinit(_cell[this->GetControlIndex()], face_no);
        _control_fe_values_ptr = &_control_fe_values;
      }
    }

  /***********************************************************************/

  template<typename VECTOR, int dim>
    void
    FaceDataContainer<dealii::MGDoFHandler, VECTOR, dim>::ReInit(
        unsigned int face_no, unsigned int subface_no)
    {
      this->SetFace(face_no);
      this->SetSubFace(subface_no);
      _state_fe_subface_values->reinit(_cell[this->GetStateIndex()], face_no,
          subface_no);
      _state_fe_values_ptr = _state_fe_subface_values;
      //Make sure that the Control must be initialized.
      if (this->GetControlIndex() < _cell.size())
      {
        _control_fe_subface_values->reinit(_cell[this->GetControlIndex()],
            face_no, this->GetSubFace());
        _control_fe_values_ptr = _control_fe_subface_values;
      }
    }
  /***********************************************************************/

  template<typename VECTOR, int dim>
    void
    FaceDataContainer<dealii::MGDoFHandler, VECTOR, dim>::ReInitNbr()
    {
      Assert(this->NeedNeighbour(), ExcInternalError());
      Assert(
          _cell[this->GetStateIndex()]->neighbor_index(this->GetFace()) != -1,
          TriaAccessorExceptions::ExcUnusedCellAsNeighbor())

      if (_cell[this->GetStateIndex()]->neighbor(this->GetFace())->has_children())
      {
        //if neighbor is more refined
        const auto neighbor_child =
            _cell[this->GetStateIndex()]->neighbor_child_on_subface(
                this->GetFace(), this->GetSubFace());

        // some sanity checks: Check, that the face and subface match and that the neighbour child
        // is not more refined.
        Assert(
            neighbor_child->face(_cell[this->GetStateIndex()]->neighbor_of_neighbor(this->GetFace())) == _cell[this->GetStateIndex()]->face(this->GetFace())->child(this->GetSubFace()),
            ExcInternalError());
        Assert(neighbor_child->has_children() == false, ExcInternalError());

        _nbr_state_fe_values->reinit(neighbor_child,
            _cell[this->GetStateIndex()]->neighbor_of_neighbor(
                this->GetFace()));
        _nbr_state_fe_values_ptr = _nbr_state_fe_values;

        //Make sure that the Control must be initialized.
        if (this->GetControlIndex() < _cell.size())
        {
          const auto control_neighbor_child =
              _cell[this->GetControlIndex()]->neighbor_child_on_subface(
                  this->GetFace(), this->GetSubFace());

          _nbr_control_fe_values->reinit(control_neighbor_child,
              _cell[this->GetControlIndex()]->neighbor_of_neighbor(
                  this->GetFace()));
          _nbr_control_fe_values_ptr = _nbr_control_fe_values;
        }
      }
      else if (_cell[this->GetStateIndex()]->neighbor_is_coarser(
          this->GetFace()))
      {
        //if the neighbour is coarser
        Assert(
            _cell[this->GetStateIndex()]->neighbor(this->GetFace())->level() == _cell[this->GetStateIndex()]->level()-1,
            ExcInternalError());
        const auto neighbor = _cell[this->GetStateIndex()]->neighbor(
            this->GetFace());
        const std::pair<unsigned int, unsigned int> faceno_subfaceno =
            _cell[this->GetStateIndex()]->neighbor_of_coarser_neighbor(
                this->GetFace());
        const unsigned int neighbor_face_no = faceno_subfaceno.first,
            neighbor_subface_no = faceno_subfaceno.second;
        _state_fe_subface_values->reinit(neighbor, neighbor_face_no,
            neighbor_subface_no);
        _nbr_state_fe_values_ptr = _state_fe_subface_values;
        if (this->GetControlIndex() < _cell.size())
        {
          const auto control_neighbor =
              _cell[this->GetControlIndex()]->neighbor(this->GetFace());
          const std::pair<unsigned int, unsigned int> control_faceno_subfaceno =
              _cell[this->GetControlIndex()]->neighbor_of_coarser_neighbor(
                  this->GetFace());
          const unsigned int control_neighbor_face_no =
              control_faceno_subfaceno.first, control_neighbor_subface_no =
              control_faceno_subfaceno.second;
          _control_fe_subface_values->reinit(control_neighbor,
              control_neighbor_face_no, control_neighbor_subface_no);
          _nbr_control_fe_values_ptr = _control_fe_subface_values;
        }

      }
      else
      {
        const auto neighbor_state = _cell[this->GetStateIndex()]->neighbor(
            this->GetFace());
        // neighbor cell is as much refined as the
        Assert(neighbor_state->level() == _cell[this->GetStateIndex()]->level(),
            ExcInternalError());
        _nbr_state_fe_values->reinit(neighbor_state,
            _cell[this->GetStateIndex()]->neighbor_of_neighbor(
                this->GetFace()));
        _nbr_state_fe_values_ptr = _nbr_state_fe_values;

        //Make sure that the Control must be initialized.
        if (this->GetControlIndex() < _cell.size())
        {
          _nbr_control_fe_values->reinit(
              _cell[this->GetControlIndex()]->neighbor(this->GetFace()),
              _cell[this->GetControlIndex()]->neighbor_of_neighbor(
                  this->GetFace()));
          _nbr_control_fe_values_ptr = _nbr_control_fe_values;
        }
      }
    }
  /***********************************************************************/
  template<typename VECTOR, int dim>
    unsigned int
    FaceDataContainer<dealii::MGDoFHandler, VECTOR, dim>::GetNDoFsPerCell() const
    {
      return _n_dofs_per_cell;
    }

  /***********************************************************************/

  template<typename VECTOR, int dim>
    unsigned int
    FaceDataContainer<dealii::MGDoFHandler, VECTOR, dim>::GetNbrNDoFsPerCell() const
    {
      return _n_dofs_per_cell;
    }

  /**********************************************/
  template<typename VECTOR, int dim>
    unsigned int
    FaceDataContainer<dealii::MGDoFHandler, VECTOR, dim>::GetNQPoints() const
    {
      return _n_q_points_per_cell;
    }

  /**********************************************/
  template<typename VECTOR, int dim>
    unsigned int
    FaceDataContainer<dealii::MGDoFHandler, VECTOR, dim>::GetNbrNQPoints() const
    {
      return _n_q_points_per_cell;
    }

  /**********************************************/
  template<typename VECTOR, int dim>
    unsigned int
    FaceDataContainer<dealii::MGDoFHandler, VECTOR, dim>::GetMaterialId() const
    {
      return _cell[0]->material_id();
    }

  /**********************************************/
  template<typename VECTOR, int dim>
    unsigned int
    FaceDataContainer<dealii::MGDoFHandler, VECTOR, dim>::GetNbrMaterialId() const
    {
      return this->GetNbrMaterialId(this->GetFace());
    }

  /**********************************************/
  template<typename VECTOR, int dim>
    unsigned int
    FaceDataContainer<dealii::MGDoFHandler, VECTOR, dim>::GetNbrMaterialId(
        unsigned int face) const
    {
      if (_cell[0]->neighbor_index(face) != -1)
        return _cell[0]->neighbor(face)->material_id();
      else
        throw DOpEException("There is no neighbor with number " + face,
            "FaceDataContainer::GetNbrMaterialId");
    }

  /**********************************************/
  template<typename VECTOR, int dim>
    bool
    FaceDataContainer<dealii::MGDoFHandler, VECTOR, dim>::GetIsAtBoundary() const
    {
      return _cell[0]->face(this->GetFace())->at_boundary();
    }

  /**********************************************/
  template<typename VECTOR, int dim>
    double
    FaceDataContainer<dealii::MGDoFHandler, VECTOR, dim>::GetCellDiameter() const
    {
      return _cell[0]->face(this->GetFace())->diameter();
    }

  /**********************************************/

  template<typename VECTOR, int dim>
    unsigned int
    FaceDataContainer<dealii::MGDoFHandler, VECTOR, dim>::GetBoundaryIndicator() const
    {
      return _cell[0]->face(this->GetFace())->boundary_indicator();
    }

  /**********************************************/
  template<typename VECTOR, int dim>
    const FEFaceValuesBase<dim>&
    FaceDataContainer<dealii::MGDoFHandler, VECTOR, dim>::GetFEFaceValuesState() const
    {
      return *_state_fe_values_ptr;
    }

  /**********************************************/
  template<typename VECTOR, int dim>
    const FEFaceValuesBase<dim>&
    FaceDataContainer<dealii::MGDoFHandler, VECTOR, dim>::GetFEFaceValuesControl() const
    {
      return *_control_fe_values_ptr;
    }
  /**********************************************/
  template<typename VECTOR, int dim>
    const FEFaceValuesBase<dim>&
    FaceDataContainer<dealii::MGDoFHandler, VECTOR, dim>::GetNbrFEFaceValuesState() const
    {
      return *_nbr_state_fe_values_ptr;
    }

  /**********************************************/
  template<typename VECTOR, int dim>
    const FEFaceValuesBase<dim>&
    FaceDataContainer<dealii::MGDoFHandler, VECTOR, dim>::GetNbrFEFaceValuesControl() const
    {
      return *_nbr_control_fe_values_ptr;
    }

  template<typename VECTOR, int dim>
    unsigned int
    FaceDataContainer<dealii::MGDoFHandler, VECTOR, dim>::GetStateIndex() const
    {
      return _state_index;
    }

  /***********************************************************************/

  template<typename VECTOR, int dim>
    unsigned int
    FaceDataContainer<dealii::MGDoFHandler, VECTOR, dim>::GetControlIndex() const
    {
      return _control_index;
    }

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
      _state_hp_fe_values.reinit(_cell[this->GetStateIndex()], face_no);
      _state_hp_fe_values_ptr = &_state_hp_fe_values.get_present_fe_values();
      //Make sure that the Control must be initialized.
      if (this->GetControlIndex() < _cell.size())
      {
        _control_hp_fe_values.reinit(_cell[this->GetControlIndex()], face_no);
        _control_hp_fe_values_ptr =
            &_control_hp_fe_values.get_present_fe_values();
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
      _state_hp_fe_subface_values->reinit(_cell[this->GetStateIndex()],
          this->GetFace(), this->GetSubFace());
      _state_hp_fe_values_ptr =
          &_state_hp_fe_subface_values->get_present_fe_values();
      //Make sure that the Control must be initialized.
      if (this->GetControlIndex() < _cell.size())
      {
        _control_hp_fe_subface_values->reinit(_cell[this->GetControlIndex()],
            this->GetFace(), this->GetSubFace());
        _control_hp_fe_values_ptr =
            &_control_hp_fe_subface_values->get_present_fe_values();
      }
    }

  /***********************************************************************/
  template<typename VECTOR, int dim>
    void
    FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::ReInitNbr()
    {
      Assert(this->NeedNeighbour(), ExcInternalError());
      Assert(
          _cell[this->GetStateIndex()]->neighbor_index(this->GetFace()) != -1,
          TriaAccessorExceptions::ExcUnusedCellAsNeighbor())

      if (_cell[this->GetStateIndex()]->neighbor(this->GetFace())->has_children())
      {
        //if neighbor is more refined
        const auto neighbor_child =
            _cell[this->GetStateIndex()]->neighbor_child_on_subface(
                this->GetFace(), this->GetSubFace());

        // some sanity checks: Check, that the face and subface match and that the neighbour child
        // is not more refined.
        Assert(
            neighbor_child->face(_cell[this->GetStateIndex()]->neighbor_of_neighbor(this->GetFace())) == _cell[this->GetStateIndex()]->face(this->GetFace())->child(this->GetSubFace()),
            ExcInternalError());
        Assert(neighbor_child->has_children() == false, ExcInternalError());

        _nbr_state_hp_fe_values->reinit(neighbor_child,
            _cell[this->GetStateIndex()]->neighbor_of_neighbor(
                this->GetFace()));
        _nbr_state_hp_fe_values_ptr =
            &_nbr_state_hp_fe_values->get_present_fe_values();

        //Make sure that the Control must be initialized.
        if (this->GetControlIndex() < _cell.size())
        {
          const auto control_neighbor_child =
              _cell[this->GetControlIndex()]->neighbor_child_on_subface(
                  this->GetFace(), this->GetSubFace());

          _nbr_control_hp_fe_values->reinit(control_neighbor_child,
              _cell[this->GetControlIndex()]->neighbor_of_neighbor(
                  this->GetFace()));
          _nbr_control_hp_fe_values_ptr =
              &_nbr_control_hp_fe_values->get_present_fe_values();
        }
      }
      else if (_cell[this->GetStateIndex()]->neighbor_is_coarser(
          this->GetFace()))
      {
        //if the neighbour is coarser
        Assert(
            _cell[this->GetStateIndex()]->neighbor(this->GetFace())->level() == _cell[this->GetStateIndex()]->level()-1,
            ExcInternalError());
        const auto neighbor = _cell[this->GetStateIndex()]->neighbor(
            this->GetFace());
        const std::pair<unsigned int, unsigned int> faceno_subfaceno =
            _cell[this->GetStateIndex()]->neighbor_of_coarser_neighbor(
                this->GetFace());
        const unsigned int neighbor_face_no = faceno_subfaceno.first,
            neighbor_subface_no = faceno_subfaceno.second;
        _state_hp_fe_subface_values->reinit(neighbor, neighbor_face_no,
            neighbor_subface_no);
        _nbr_state_hp_fe_values_ptr =
            &_state_hp_fe_subface_values->get_present_fe_values();
        if (this->GetControlIndex() < _cell.size())
        {
          const auto control_neighbor =
              _cell[this->GetControlIndex()]->neighbor(this->GetFace());
          const std::pair<unsigned int, unsigned int> control_faceno_subfaceno =
              _cell[this->GetControlIndex()]->neighbor_of_coarser_neighbor(
                  this->GetFace());
          const unsigned int control_neighbor_face_no =
              control_faceno_subfaceno.first, control_neighbor_subface_no =
              control_faceno_subfaceno.second;

          _control_hp_fe_subface_values->reinit(control_neighbor,
              control_neighbor_face_no, control_neighbor_subface_no);
          _nbr_control_hp_fe_values_ptr =
              &_control_hp_fe_subface_values->get_present_fe_values();
        }
      }
      else
      {
        const auto neighbor_state = _cell[this->GetStateIndex()]->neighbor(
            this->GetFace());
        // neighbor cell is as much refined as the
        Assert(neighbor_state->level() == _cell[this->GetStateIndex()]->level(),
            ExcInternalError());
        _nbr_state_hp_fe_values->reinit(neighbor_state,
            _cell[this->GetStateIndex()]->neighbor_of_neighbor(
                this->GetFace()));
        _nbr_state_hp_fe_values_ptr =
            &_nbr_state_hp_fe_values->get_present_fe_values();

        //Make sure that the Control must be initialized.
        if (this->GetControlIndex() < _cell.size())
        {
          _nbr_control_hp_fe_values->reinit(
              _cell[this->GetControlIndex()]->neighbor(this->GetFace()),
              _cell[this->GetControlIndex()]->neighbor_of_neighbor(
                  this->GetFace()));
          _nbr_control_hp_fe_values_ptr =
              &_nbr_control_hp_fe_values->get_present_fe_values();
        }
      }
    }

  /*********************************************/

  template<typename VECTOR, int dim>
    unsigned int
    FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetNDoFsPerCell() const
    {
      return _cell[0]->get_fe().dofs_per_cell;
    }

  /*********************************************/

  template<typename VECTOR, int dim>
    unsigned int
    FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetNbrNDoFsPerCell() const
    {
      if (_cell[0]->neighbor_index(this->GetFace()) != -1)
        return _cell[0]->neighbor(this->GetFace())->get_fe().dofs_per_cell;
      else
        throw DOpEException(
            "There is no neighbor with number" + this->GetFace(),
            "HpFaceDataContainer::GetNbrNDoFsPerCell");
    }
  /*********************************************/
  template<typename VECTOR, int dim>
    unsigned int
    FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetNQPoints() const
    {
      return _q_collection[_cell[0]->active_fe_index()].size();
    }

  /*********************************************/
  template<typename VECTOR, int dim>
    unsigned int
    FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetNbrNQPoints() const
    {
      if (_cell[0]->neighbor_index(this->GetFace()) != -1)
        return _q_collection[_cell[0]->neighbor(this->GetFace())->active_fe_index()].size();
      else
        throw DOpEException(
            "There is no neighbor with number" + this->GetFace(),
            "HpFaceDataContainer::GetNbrNQPoints");
    }
  /*********************************************/

  template<typename VECTOR, int dim>
    unsigned int
    FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetMaterialId() const
    {
      return _cell[0]->material_id();
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
      if (_cell[0]->neighbor_index(face) != -1)
        return _cell[0]->neighbor(face)->material_id();
      else
        throw DOpEException("There is no neighbor with number" + face,
            "HpFaceDataContainer::GetNbrMaterialId");
    }

  /*********************************************/

  template<typename VECTOR, int dim>
    double
    FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetCellDiameter() const
    {
      return _cell[0]->face(this->GetFace())->diameter();
    }

  /**********************************************/

  template<typename VECTOR, int dim>
    bool
    FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetIsAtBoundary() const
    {
      return _cell[0]->face(this->GetFace())->at_boundary();
    }

  /**********************************************/
  template<typename VECTOR, int dim>
    unsigned int
    FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetBoundaryIndicator() const
    {
      return _cell[0]->face(this->GetFace())->boundary_indicator();
    }

  /*********************************************/
  template<typename VECTOR, int dim>
    const FEFaceValuesBase<dim>&
    FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetFEFaceValuesState() const
    {
      return *_state_hp_fe_values_ptr;
    }
  /*********************************************/
  template<typename VECTOR, int dim>
    const FEFaceValuesBase<dim>&
    FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetFEFaceValuesControl() const
    {
      return *_control_hp_fe_values_ptr;
    }
  /*********************************************/
  template<typename VECTOR, int dim>
    const FEFaceValuesBase<dim>&
    FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetNbrFEFaceValuesState() const
    {
      return *_nbr_state_hp_fe_values_ptr;
    }
  /*********************************************/
  template<typename VECTOR, int dim>
    const FEFaceValuesBase<dim>&
    FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetNbrFEFaceValuesControl() const
    {
      return *_nbr_control_hp_fe_values_ptr;
    }

  /**********************************************/
  template<typename VECTOR, int dim>
    unsigned int
    FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetStateIndex() const
    {
      return _state_index;
    }
  /*********************************************/
  template<typename VECTOR, int dim>
    unsigned int
    FaceDataContainer<dealii::hp::DoFHandler, VECTOR, dim>::GetControlIndex() const
    {
      return _control_index;
    }
} //end of namespace

#endif /* FACEDATACONTAINER_H_ */
