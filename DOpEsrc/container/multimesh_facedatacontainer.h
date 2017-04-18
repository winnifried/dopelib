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

#ifndef MULTIMESH_FACEDATACONTAINER_H_
#define MULTIMESH_FACEDATACONTAINER_H_

#include <basic/spacetimehandler.h>
#include <basic/statespacetimehandler.h>
#include <wrapper/fevalues_wrapper.h>
#include <include/dopeexception.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/hp/dof_handler.h>

using namespace dealii;

namespace DOpE
{
  /**
   * Dummy Template Class, acts as kind of interface. Through template specialization, we
   * distinguish between the 'classic' and the 'hp' case.
   *
   * The Multimesh_ElementDataContainers, can deal with different meshes for the control
   * and state variable as long as both are given as refinements of a common coarse grid
   * by calculation of the respective values on a common refinement.
   *
   * @template DOFHANDLER The type of the dealii-dofhandler we use in our DoPEWrapper::DoFHandler, at the moment DoFHandler and hp::DoFHandler.
   * @template VECTOR     Type of the vector we use in our computations (i.e. Vector<double> or BlockVector<double>)
   * @template dim        The dimension of the integral we are actually interested in.
   */

  template<template<int, int> class DH, typename VECTOR, int dim>
  class Multimesh_FaceDataContainer
  {
  public:
    Multimesh_FaceDataContainer()
    {
      throw (DOpEException(
               "Dummy class, this constructor should never get called.",
               "FaceDataContainer<dealii::DoFHandler , VECTOR, dim>::FaceDataContainer"));
    }
    ;
  };

  /**
   * This two classes hold all the information we need in the integrator to
   * integrate something over a face of a element (could be a functional, a PDE, etc.).
   * Of particular importance: This class holds the FaceFEValues objects.
   *
   * @template VECTOR     Type of the vector we use in our computations (i.e. Vector<double> or BlockVector<double>)
   * @template dim        1+ the dimension of the integral we are actually interested in.
   */

  template<typename VECTOR, int dim>
  class Multimesh_FaceDataContainer<dealii::DoFHandler, VECTOR, dim>
  {

  public:
    /**
     * Constructor. Initializes the FaceFEValues objects.
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
     *
     */
    template<template<int, int> class FE, typename SPARSITYPATTERN, int dopedim,
             int dealdim>
    Multimesh_FaceDataContainer(
      const Quadrature<dim - 1>& quad,
      UpdateFlags update_flags,
      SpaceTimeHandler<FE, dealii::DoFHandler, SPARSITYPATTERN,
      VECTOR, dopedim, dealdim>& sth,
      const typename std::vector<typename dealii::DoFHandler<dim>::cell_iterator> &element,
      const typename std::vector<typename dealii::Triangulation<dim>::cell_iterator> &tria_element,
      const std::map<std::string, const Vector<double>*> &param_values,
      const std::map<std::string, const VECTOR *> &domain_values, bool/* just for compatibility*/) :
      param_values_(param_values),
      domain_values_(domain_values),
      element_(element),
      tria_element_(tria_element),
      state_fe_values_(sth.GetFESystem("state"), quad,
                       update_flags),
      control_fe_values_(sth.GetFESystem("control"), quad,
                         update_flags)
    {
      state_index_ = sth.GetStateIndex();
      if (state_index_ == 1)
        control_index_ = 0;
      else
        control_index_ = 1;
      n_q_points_per_element_ = quad.size();
      n_dofs_per_element_ = element[0]->get_fe().dofs_per_cell;
      control_prolongation_ = IdentityMatrix(element_[this->GetControlIndex()]->get_fe().dofs_per_cell);
      state_prolongation_ = IdentityMatrix(element_[this->GetStateIndex()]->get_fe().dofs_per_cell);
    }

    ~Multimesh_FaceDataContainer()
    {
    }
    /*********************************************/
    /*
     * This function reinits the FEValues on the actual face. Should
     * be called prior to any of the get-functions.
     *
     * @param face_no     The 'local number' (i.e. from the perspective of the actual element) of the
     *                    actual face.
     */
    inline void
    ReInit(unsigned int coarse_index,unsigned int fine_index, const FullMatrix<double> &prolongation_matrix, unsigned int face_no);

    /*********************************************/
    /**
     * Just for compatibility reasons.
     */
    inline void
    ReInitNbr() {}

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
    inline const DOpEWrapper::FEFaceValues<dim> &
    GetFEFaceValuesState() const;
    inline const DOpEWrapper::FEFaceValues<dim> &
    GetFEFaceValuesControl() const;

    /**********************************************/
    /*
     * Looks up the given name in parameter_data_ and returns the corresponding value
     * through 'value'.
     */
    void
    GetParamValues(std::string name, Vector<double> &value) const;

    /*********************************************/
    /**
     * Functions to extract values and gradients out of the FEFaceValues
     */

    /*
     * Writes the values of the state variable at the quadrature points into values.
     */
    inline void
    GetFaceValuesState(std::string name, std::vector<double> &values) const;
    /*********************************************/
    /*
     * Same as above for the Vector valued case.
     */
    inline void
    GetFaceValuesState(std::string name, std::vector<Vector<double> > &values) const;

    /*********************************************/
    /*
     * Writes the values of the control variable at the quadrature points into values
     */
    inline void
    GetFaceValuesControl(std::string name, std::vector<double> &values) const;
    /*********************************************/
    /*
     * Same as above for the Vector valued case.
     */
    inline void
    GetFaceValuesControl(std::string name, std::vector<Vector<double> > &values) const;
    /*********************************************/
    /*
     * Writes the values of the state gradient at the quadrature points into values.
     */

    template<int targetdim>
    inline void
    GetFaceGradsState(std::string name,
                      std::vector<Tensor<1, targetdim> > &values) const;

    /*********************************************/
    /*
     * Same as above for the Vector valued case.
     */
    template<int targetdim>
    inline void
    GetFaceGradsState(std::string name,
                      std::vector<std::vector<Tensor<1, targetdim> > > &values) const;

    /*********************************************/
    /*
     * Writes the values of the control gradient at the quadrature points into values.
     */
    template<int targetdim>
    inline void
    GetFaceGradsControl(std::string name,
                        std::vector<Tensor<1, targetdim> > &values) const;

    /*********************************************/
    /*
     * Same as above for the Vector valued case.
     */
    template<int targetdim>
    inline void
    GetFaceGradsControl(std::string name,
                        std::vector<std::vector<Tensor<1, targetdim> > > &values) const;

  private:
    /*
     * Helper Functions
     */
    unsigned int
    GetStateIndex() const;
    unsigned int
    GetControlIndex() const;
    unsigned int
    GetCoarseIndex() const
    {
      return coarse_index_;
    }
    unsigned int
    GetFineIndex() const
    {
      return fine_index_;
    }
    /***********************************************************/
    /**
     * Helper Function. Vector valued case.
     */
    inline void
    GetValues(typename dealii::DoFHandler<dim>::cell_iterator element,
              const FullMatrix<double> &prolongation,
              const DOpEWrapper::FEFaceValues<dim> &fe_values, std::string name,
              std::vector<double> &values) const;
    /***********************************************************/
    /**
     * Helper Function. Vector valued case.
     */
    inline void
    GetValues(typename dealii::DoFHandler<dim>::cell_iterator element,
              const FullMatrix<double> &prolongation,
              const DOpEWrapper::FEFaceValues<dim> &fe_values, std::string name,
              std::vector<Vector<double> > &values) const;
    /***********************************************************/
    /**
     * Helper Function.
     */
    template<int targetdim>
    inline void
    GetGrads(typename dealii::DoFHandler<dim>::cell_iterator element,
             const FullMatrix<double> &prolongation,
             const DOpEWrapper::FEFaceValues<dim> &fe_values,
             std::string name, std::vector<Tensor<1, targetdim> > &values) const;
    /***********************************************************/
    /**
     * Helper Function. Vector valued case.
     */
    template<int targetdim>
    inline void
    GetGrads(typename dealii::DoFHandler<dim>::cell_iterator element,
             const FullMatrix<double> &prolongation,
             const DOpEWrapper::FEFaceValues<dim> &fe_values,
             std::string name, std::vector<std::vector<Tensor<1, targetdim> > > &values) const;
    /***********************************************************/
    inline const std::map<std::string, const VECTOR *> &
    GetDomainValues() const
    {
      return domain_values_;
    }
    /***********************************************************/
    //"global" member data, part of every instantiation
    const std::map<std::string, const Vector<double>*> &param_values_;
    const std::map<std::string, const VECTOR *> &domain_values_;
    unsigned int state_index_;
    unsigned int control_index_;

    const typename std::vector<typename dealii::DoFHandler<dim>::cell_iterator> &element_;
    const typename std::vector<typename dealii::Triangulation<dim>::cell_iterator> &tria_element_;
    DOpEWrapper::FEFaceValues<dim> state_fe_values_;
    DOpEWrapper::FEFaceValues<dim> control_fe_values_;

    FullMatrix<double> control_prolongation_;
    FullMatrix<double> state_prolongation_;
    unsigned int coarse_index_, fine_index_;

    unsigned int n_q_points_per_element_;
    unsigned int n_dofs_per_element_;

    unsigned int face_;
  };


  /***********************************************************************/
  /************************IMPLEMENTATION*for*DoFHandler*********************************/
  /***********************************************************************/

  template<typename VECTOR, int dim>
  void
  Multimesh_FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::ReInit(
    unsigned int coarse_index,
    unsigned int fine_index,
    const FullMatrix<double> &prolongation_matrix,
    unsigned int face_no)
  {
    face_ = face_no;
    coarse_index_ = coarse_index;
    fine_index_ = fine_index;
    assert(this->GetControlIndex() < element_.size());

    if (coarse_index == this->GetStateIndex())
      {
        state_prolongation_ = prolongation_matrix;
        control_prolongation_ = IdentityMatrix(element_[this->GetControlIndex()]->get_fe().dofs_per_cell);
      }
    else
      {
        if (coarse_index == this->GetControlIndex())
          {
            control_prolongation_ = prolongation_matrix;
            state_prolongation_ = IdentityMatrix(element_[this->GetStateIndex()]->get_fe().dofs_per_cell);
          }
        else
          {
            control_prolongation_ = IdentityMatrix(element_[this->GetControlIndex()]->get_fe().dofs_per_cell);
            state_prolongation_ = IdentityMatrix(element_[this->GetStateIndex()]->get_fe().dofs_per_cell);
            fine_index_ = 0;
          }
      }

    state_fe_values_.reinit(tria_element_[GetFineIndex()], face_no);
    control_fe_values_.reinit(tria_element_[GetFineIndex()], face_no);
  }
  /***********************************************************************/
  template<typename VECTOR, int dim>
  unsigned int
  Multimesh_FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetNDoFsPerElement() const
  {
    return n_dofs_per_element_;
  }

  /***********************************************************************/

  template<typename VECTOR, int dim>
  unsigned int
  Multimesh_FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetNbrNDoFsPerElement() const
  {
    throw DOpEException("This function has not been written since we do not know what the right neigbour is!",
                        "Multimesh_FaceDataContainer::GetNbrNDoFsPerElement");
  }

  /**********************************************/
  template<typename VECTOR, int dim>
  unsigned int
  Multimesh_FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetNQPoints() const
  {
    return n_q_points_per_element_;
  }

  /**********************************************/
  template<typename VECTOR, int dim>
  unsigned int
  Multimesh_FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetMaterialId() const
  {
    return tria_element_[GetFineIndex()]->material_id();
  }

  /**********************************************/
  template<typename VECTOR, int dim>
  unsigned int
  Multimesh_FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetNbrMaterialId() const
  {
    return this->GetNbrMaterialId(face_);
  }

  template<typename VECTOR, int dim>
  unsigned int
  Multimesh_FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetNbrMaterialId(
    unsigned int /*face*/) const
  {
    throw DOpEException("This function has not been written since we do not know what the right neigbour is!",
                        "Multimesh_FaceDataContainer::GetNbrMaterialId");
  }

  /**********************************************/
  template<typename VECTOR, int dim>
  bool
  Multimesh_FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetIsAtBoundary() const
  {
    return tria_element_[GetFineIndex()]->face(face_)->at_boundary();
  }

  /**********************************************/
  template<typename VECTOR, int dim>
  double
  Multimesh_FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetElementDiameter() const
  {
    throw DOpEException("This function has not been written since we do not know what the right Diameter!",
                        "Multimesh_FaceDataContainer::GetElementDiameter");
  }

  /**********************************************/

  template<typename VECTOR, int dim>
  unsigned int
  Multimesh_FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetBoundaryIndicator() const
  {
    return tria_element_[GetFineIndex()]->face(face_)->boundary_indicator();
  }

  /**********************************************/
  template<typename VECTOR, int dim>
  const DOpEWrapper::FEFaceValues<dim> &
  Multimesh_FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetFEFaceValuesState() const
  {
    return state_fe_values_;
  }

  /**********************************************/
  template<typename VECTOR, int dim>
  const DOpEWrapper::FEFaceValues<dim> &
  Multimesh_FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetFEFaceValuesControl() const
  {
    return control_fe_values_;
  }

  /**********************************************/

  template<typename VECTOR, int dim>
  void
  Multimesh_FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetParamValues(
    std::string name, Vector<double> &value) const
  {
    typename std::map<std::string, const Vector<double>*>::const_iterator it =
      param_values_.find(name);
    if (it == param_values_.end())
      {
        throw DOpEException("Did not find " + name,
                            "Multimesh_FaceDataContainer::GetParamValues");
      }
    value = *(it->second);
  }

  /*********************************************/
  template<typename VECTOR, int dim>
  void
  Multimesh_FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetFaceValuesState(
    std::string name, std::vector<double> &values) const
  {
    this->GetValues(element_[this->GetStateIndex()],state_prolongation_,this->GetFEFaceValuesState(), name, values);
  }
  /*********************************************/
  template<typename VECTOR, int dim>
  void
  Multimesh_FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetFaceValuesState(
    std::string name, std::vector<Vector<double> > &values) const
  {
    this->GetValues(element_[this->GetStateIndex()],state_prolongation_,this->GetFEFaceValuesState(), name, values);

  }

  /*********************************************/
  template<typename VECTOR, int dim>
  void
  Multimesh_FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetFaceValuesControl(
    std::string name, std::vector<double> &values) const
  {
    this->GetValues(element_[this->GetControlIndex()],control_prolongation_,this->GetFEFaceValuesControl(), name, values);
  }

  /*********************************************/
  template<typename VECTOR, int dim>
  void
  Multimesh_FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetFaceValuesControl(
    std::string name, std::vector<Vector<double> > &values) const
  {
    this->GetValues(element_[this->GetControlIndex()],control_prolongation_,this->GetFEFaceValuesControl(), name, values);
  }

  /*********************************************/
  template<typename VECTOR, int dim>
  template<int targetdim>
  void
  Multimesh_FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetFaceGradsState(
    std::string name, std::vector<Tensor<1, targetdim> > &values) const
  {
    this->GetGrads<targetdim> (element_[this->GetStateIndex()],state_prolongation_,this->GetFEFaceValuesState(), name, values);
  }

  /*********************************************/
  template<typename VECTOR, int dim>
  template<int targetdim>
  void
  Multimesh_FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetFaceGradsState(
    std::string name, std::vector<std::vector<Tensor<1, targetdim> > > &values) const
  {
    this->GetGrads<targetdim> (element_[this->GetStateIndex()],state_prolongation_,this->GetFEFaceValuesState(), name, values);
  }

  /***********************************************************************/

  template<typename VECTOR, int dim>
  template<int targetdim>
  void
  Multimesh_FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetFaceGradsControl(
    std::string name, std::vector<Tensor<1, targetdim> > &values) const
  {
    this->GetGrads<targetdim> (element_[this->GetControlIndex()],control_prolongation_,this->GetFEFaceValuesControl(), name, values);
  }
  /***********************************************************************/

  template<typename VECTOR, int dim>
  template<int targetdim>
  void
  Multimesh_FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetFaceGradsControl(
    std::string name, std::vector<std::vector<Tensor<1, targetdim> > > &values) const
  {
    this->GetGrads<targetdim> (element_[this->GetControlIndex()],control_prolongation_,this->GetFEFaceValuesControl(), name, values);
  }

  /***********************************************************************/

  template<typename VECTOR, int dim>
  unsigned int
  Multimesh_FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetStateIndex() const
  {
    return state_index_;
  }

  /***********************************************************************/

  template<typename VECTOR, int dim>
  unsigned int
  Multimesh_FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetControlIndex() const
  {
    return control_index_;
  }

  /***********************************************************************/
  template<typename VECTOR, int dim>
  void
  Multimesh_FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetValues(
    typename dealii::DoFHandler<dim>::cell_iterator element,
    const FullMatrix<double> &prolongation,
    const DOpEWrapper::FEFaceValues<dim> &fe_values, std::string name,
    std::vector<double> &values) const
  {
    typename std::map<std::string, const VECTOR *>::const_iterator it =
      this->GetDomainValues().find(name);
    if (it == this->GetDomainValues().end())
      {
        throw DOpEException("Did not find " + name,
                            "Multimesh_FaceDataContainer::GetValues");
      }
    unsigned int dofs_per_element = element->get_fe().dofs_per_cell;
    //Now we get the values on the real element
    dealii::Vector<double> dof_values(dofs_per_element);
    dealii::Vector<double> dof_values_transformed(dofs_per_element);
    element->get_dof_values (*(it->second), dof_values);
    //Now compute the real values at the nodal points
    prolongation.vmult(dof_values_transformed,dof_values);

    //Copied from deal FEValuesBase<dim,spacedim>::get_function_values
    // see deal.II/source/fe/fe_values.cc
    unsigned int n_quadrature_points = GetNQPoints();
    std::fill_n (values.begin(), n_quadrature_points, 0);
    for (unsigned int shape_func=0; shape_func<dofs_per_element; ++shape_func)
      {
        const double value = dof_values_transformed(shape_func);
        if (value == 0.)
          continue;

        const double *shape_value_ptr = &(fe_values.shape_value(shape_func, 0));
        for (unsigned int point=0; point<n_quadrature_points; ++point)
          values[point] += value **shape_value_ptr++;
      }
  }

  /***********************************************************************/
  template<typename VECTOR, int dim>
  void
  Multimesh_FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetValues(
    typename dealii::DoFHandler<dim>::cell_iterator element,
    const FullMatrix<double> &prolongation,
    const DOpEWrapper::FEFaceValues<dim> &fe_values, std::string name,
    std::vector<Vector<double> > &values) const
  {
    typename std::map<std::string, const VECTOR *>::const_iterator it =
      this->GetDomainValues().find(name);
    if (it == this->GetDomainValues().end())
      {
        throw DOpEException("Did not find " + name,
                            "Multimesh_FaceDataContainer::GetValues");
      }

    unsigned int dofs_per_element = element->get_fe().dofs_per_cell;
    //Now we get the values on the real element
    dealii::Vector<double> dof_values(dofs_per_element);
    dealii::Vector<double> dof_values_transformed(dofs_per_element);
    element->get_dof_values (*(it->second), dof_values);
    //Now compute the real values at the nodal points
    prolongation.vmult(dof_values_transformed,dof_values);

    //Copied from deal FEValuesBase<dim,spacedim>::get_function_values
    // see deal.II/source/fe/fe_values.cc
    const unsigned int n_components = element->get_fe().n_components();
    unsigned int n_quadrature_points = GetNQPoints();
    for (unsigned i=0; i<values.size(); ++i)
      std::fill_n (values[i].begin(), values[i].size(), 0);

    for (unsigned int shape_func=0; shape_func<dofs_per_element; ++shape_func)
      {
        const double value = dof_values_transformed(shape_func);
        if (value == 0.)
          continue;

        if (element->get_fe().is_primitive(shape_func))
          {
            const unsigned int comp = element->get_fe().system_to_component_index(shape_func).first;
            for (unsigned int point=0; point<n_quadrature_points; ++point)
              values[point](comp) += value * fe_values.shape_value(shape_func,point);
          }
        else
          for (unsigned int c=0; c<n_components; ++c)
            {
              if (element->get_fe().get_nonzero_components(shape_func)[c] == false)
                continue;
              for (unsigned int point=0; point<n_quadrature_points; ++point)
                values[point](c) += value * fe_values.shape_value_component(shape_func,point,c);
            }
      }
  }

  /***********************************************************************/

  template<typename VECTOR, int dim>
  template<int targetdim>
  void
  Multimesh_FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetGrads(
    typename dealii::DoFHandler<dim>::cell_iterator element,
    const FullMatrix<double> &prolongation,
    const DOpEWrapper::FEFaceValues<dim> &fe_values, std::string name,
    std::vector<Tensor<1, targetdim> > &values) const
  {
    typename std::map<std::string, const VECTOR *>::const_iterator it =
      this->GetDomainValues().find(name);
    if (it == this->GetDomainValues().end())
      {
        throw DOpEException("Did not find " + name,
                            "Multimesh_FaceDataContainerBase::GetGrads");
      }
    unsigned int dofs_per_element = element->get_fe().dofs_per_cell;
    //Now we get the values on the real element
    dealii::Vector<double> dof_values(dofs_per_element);
    dealii::Vector<double> dof_values_transformed(dofs_per_element);
    element->get_dof_values (*(it->second), dof_values);
    //Now compute the real values at the nodal points
    prolongation.vmult(dof_values_transformed,dof_values);

    //Copied from deal FEValuesBase<dim,spacedim>::get_function_gradients
    unsigned int n_quadrature_points = GetNQPoints();
    std::fill_n (values.begin(), n_quadrature_points, Tensor<1,targetdim>());

    for (unsigned int shape_func=0; shape_func<dofs_per_element; ++shape_func)
      {
        const double value = dof_values_transformed(shape_func);
        if (value == 0.)
          continue;

        const Tensor<1,targetdim> *shape_gradient_ptr
          = &(fe_values.shape_grad(shape_func,0));
        for (unsigned int point=0; point<n_quadrature_points; ++point)
          values[point] += value **shape_gradient_ptr++;
      }
  }

  /***********************************************************************/

  template<typename VECTOR, int dim>
  template<int targetdim>
  void
  Multimesh_FaceDataContainer<dealii::DoFHandler, VECTOR, dim>::GetGrads(
    typename dealii::DoFHandler<dim>::cell_iterator element,
    const FullMatrix<double> &prolongation,
    const DOpEWrapper::FEFaceValues<dim> &fe_values, std::string name,
    std::vector<std::vector<Tensor<1, targetdim> > > &values) const
  {
    typename std::map<std::string, const VECTOR *>::const_iterator it =
      this->GetDomainValues().find(name);
    if (it == this->GetDomainValues().end())
      {
        throw DOpEException("Did not find " + name,
                            "Multimesh_FaceDataContainerBase::GetGrads");
      }

    unsigned int dofs_per_element = element->get_fe().dofs_per_cell;
    //Now we get the values on the real element
    dealii::Vector<double> dof_values(dofs_per_element);
    dealii::Vector<double> dof_values_transformed(dofs_per_element);
    element->get_dof_values (*(it->second), dof_values);
    //Now compute the real values at the nodal points
    prolongation.vmult(dof_values_transformed,dof_values);

    //Copied from deal FEValuesBase<dim,spacedim>::get_function_gradients
    const unsigned int n_components = element->get_fe().n_components();
    unsigned int n_quadrature_points = GetNQPoints();
    for (unsigned i=0; i<values.size(); ++i)
      std::fill_n (values[i].begin(), values[i].size(), Tensor<1,dim>());

    for (unsigned int shape_func=0; shape_func<dofs_per_element; ++shape_func)
      {
        const double value = dof_values_transformed(shape_func);
        if (value == 0.)
          continue;

        if (element->get_fe().is_primitive(shape_func))
          {
            const unsigned int comp = element->get_fe().system_to_component_index(shape_func).first;
            for (unsigned int point=0; point<n_quadrature_points; ++point)
              values[point][comp] += value * fe_values.shape_grad(shape_func,point);
          }
        else
          for (unsigned int c=0; c<n_components; ++c)
            {
              if (element->get_fe().get_nonzero_components(shape_func)[c] == false)
                continue;
              for (unsigned int point=0; point<n_quadrature_points; ++point)
                values[point][c] += value * fe_values.shape_grad_component(shape_func,point,c);
            }
      }
  }

  /***********************************************************************/
  /************************END*OF*IMPLEMENTATION**************************/
  /***********************************************************************/

}//end of namespace

#endif /* MULTIMESH_FACEDATACONTAINER_H_ */
