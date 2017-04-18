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

#ifndef RESIDUAL_ERROR_H_
#define RESIDUAL_ERROR_H_

#include <container/dwrdatacontainer.h>
#include <deal.II/fe/fe_tools.h>

namespace DOpE
{
  /**
   * This class is the base for all estimators of residualtype that
   * do not require a weight.
   * Although, technically this is not dual weighted!
   */
  template<typename VECTOR>
  class ResidualErrorContainer : public DWRDataContainerBase<VECTOR>
  {
  public:
    ResidualErrorContainer(DOpEtypes::EETerms ee_terms =
                             DOpEtypes::EETerms::mixed) :
      DWRDataContainerBase<VECTOR>(ee_terms)
    {
    }
    virtual void
    InitFace(double /*h*/) = 0;
    virtual void
    InitElement(double /*h*/) = 0;

  };

  /**
   * We need this overloaded function to have the same
   * interface for dwrdatacontainer and residualestimators.
   *
   * The function should actually never get called, but with this
   * construction, we save 4 unnecessary template parameters!
   */
  template<class EDC, typename VECTOR>
  EDC *
  ExtractEDC(const ResidualErrorContainer<VECTOR> & /*dwrc*/)
  {
    return NULL;
  }
  template<class FDC, typename VECTOR>
  FDC *
  ExtractFDC(const ResidualErrorContainer<VECTOR> & /*dwrc*/)
  {
    return NULL;
  }
  /**
   * This class implements the missing pieces of DWRDataContainer for
   * the case of the computation of a standard L2-residual error estimator.
   * Although, technicaly this is not dual weighted!
   */
  template<class STH, typename VECTOR, int dim>
  class L2ResidualErrorContainer : public ResidualErrorContainer<VECTOR>
  {
  public:
    L2ResidualErrorContainer(STH &sth, DOpEtypes::VectorStorageType state_behavior,
                             ParameterReader &param_reader, DOpEtypes::EETerms ee_terms =
                               DOpEtypes::EETerms::mixed) :
      ResidualErrorContainer<VECTOR>(ee_terms), sth_(sth), PI_h_u_(NULL), PI_h_z_(
        NULL)
    {
      if (this->GetEETerms() == DOpEtypes::primal_only
          || this->GetEETerms() == DOpEtypes::mixed)
        {
          PI_h_z_ = new StateVector<VECTOR>(&GetSTH(), state_behavior,
                                            param_reader);
        }
      if (this->GetEETerms() == DOpEtypes::dual_only
          || this->GetEETerms() == DOpEtypes::mixed)
        {
          PI_h_u_ = new StateVector<VECTOR>(&GetSTH(), state_behavior,
                                            param_reader);
        }
      weight_ = 0.;
    }

    virtual
    ~L2ResidualErrorContainer()
    {
      if (PI_h_z_ != NULL)
        delete PI_h_z_;
      if (PI_h_u_ != NULL)
        delete PI_h_u_;

    }

    std::string
    GetName() const
    {
      return "L2-Residual-Estimator";
    }

    void
    Initialize(unsigned int state_n_blocks,
               std::vector<unsigned int> &state_block_component)
    {
      state_n_blocks_ = state_n_blocks;
      state_block_component_ = &state_block_component;
    }

    /**
     * ReInits the DWRDataContainer, the higher order STH
     * as well as the weight-vectors.
     */
    void
    ReInit(unsigned int n_elements);

    StateVector<VECTOR> &
    GetPI_h_u()
    {
      return *PI_h_u_;
    }

    StateVector<VECTOR> &
    GetPI_h_z()
    {
      return *PI_h_z_;
    }
    ControlVector<VECTOR> &
    GetPI_h_q()
    {
      throw DOpEException("There is no Control in PDE Problems!",
                          "L2ResidualErrorContainer::PreparePI_h_q");
    }

    /**
     * Makes the patchwise higher order interpolant of the
     * primal soltion u. This is needed as a weight for the
     * dual residual.
     */
    void
    PreparePI_h_u(const StateVector<VECTOR> & /*u*/)
    {
      BuildConstantWeight(&(GetSTH().GetStateDoFHandler()),
                          GetPI_h_u().GetSpacialVector());
    }

    /**
     * Makes the patchwise higher order interpolant of the
     * dual solution z. This is needed as a weight for the
     * primal residual.
     */
    void
    PreparePI_h_z(const StateVector<VECTOR> & /*z*/)
    {
      BuildConstantWeight(&(GetSTH().GetStateDoFHandler()),
                          GetPI_h_z().GetSpacialVector());
    }
    /**
     * Makes the patchwise higher order interpolant of the
     * control q. This is needed as a weight for the
     * control residual.
     */
    void
    PreparePI_h_q(const ControlVector<VECTOR> & /*q*/)
    {
      throw DOpEException("There is no Control in PDE Problems!",
                          "HigherOrderDWRContainer::PreparePI_h_q");

    }

    /**
     * Implementation of virtual method from base class.
     */
    bool
    NeedDual() const
    {
      return false;
    }

    /**
     * Implementation of virtual method from base class.
     */
    virtual DOpEtypes::WeightComputation
    GetWeightComputation() const
    {
      return DOpEtypes::element_diameter;
    }

    /**
     * Implementation of virtual method from base class.
     */
    virtual DOpEtypes::ResidualEvaluation
    GetResidualEvaluation() const
    {
      return DOpEtypes::strong_residual;
    }

    /**
     * This should be applied to the residual in the integration
     * To assert that the squared norm is calculated
     */
    inline void
    ResidualModifier(double &res)
    {
      res = res * res * weight_;
    }

    inline void
    VectorResidualModifier(dealii::Vector<double> &res)
    {
      for (unsigned int i = 0; i < res.size(); i++)
        res(i) = res(i) * res(i) * weight_;
    }

    void
    InitFace(double h)
    {
      weight_ = h * h * h;
    }
    void
    InitElement(double h)
    {
      weight_ = h * h * h * h;
    }

  protected:
    STH &
    GetSTH()
    {
      return sth_;
    }

    template<template<int, int> class DH>
    void
    BuildConstantWeight(const DOpEWrapper::DoFHandler<dim, DH> *dofh,
                        VECTOR &vals)
    {
      VectorTools::interpolate(sth_.GetMapping(),
                               *(static_cast<const DH<dim, dim>*>(dofh)),
                               ConstantFunction<dim>(1., dofh->get_fe().n_components()), vals);
    }

  private:
    unsigned int state_n_blocks_;
    std::vector<unsigned int> *state_block_component_;
    double weight_;

    STH &sth_;

    StateVector<VECTOR> *PI_h_u_, *PI_h_z_;
  };

  template<class STH, typename VECTOR, int dim>
  void
  L2ResidualErrorContainer<STH, VECTOR, dim>::ReInit(unsigned int n_elements)
  {
    DWRDataContainerBase<VECTOR>::ReInit(n_elements);

    if (this->GetEETerms() == DOpEtypes::primal_only
        || this->GetEETerms() == DOpEtypes::mixed)
      {
        GetPI_h_z().ReInit();
      }
    if (this->GetEETerms() == DOpEtypes::dual_only
        || this->GetEETerms() == DOpEtypes::mixed)
      {
        GetPI_h_u().ReInit();
      }
  }

  /**************************************************************************/
  /**
   * This class implements the missing pieces of DWRDataContainer for
   * the case of the computation of a standard energynorm-residual error estimator.
   * Although, technically this is not dual weighted!
   */
  template<class STH, typename VECTOR, int dim>
  class H1ResidualErrorContainer : public ResidualErrorContainer<VECTOR>
  {
  public:
    H1ResidualErrorContainer(STH &sth, DOpEtypes::VectorStorageType state_behavior,
                             ParameterReader &param_reader, DOpEtypes::EETerms ee_terms =
                               DOpEtypes::EETerms::mixed) :
      ResidualErrorContainer<VECTOR>(ee_terms), sth_(sth), PI_h_u_(NULL), PI_h_z_(
        NULL)
    {
      if (this->GetEETerms() == DOpEtypes::primal_only
          || this->GetEETerms() == DOpEtypes::mixed)
        {
          PI_h_z_ = new StateVector<VECTOR>(&GetSTH(), state_behavior,
                                            param_reader);
        }
      if (this->GetEETerms() == DOpEtypes::dual_only
          || this->GetEETerms() == DOpEtypes::mixed)
        {
          PI_h_u_ = new StateVector<VECTOR>(&GetSTH(), state_behavior,
                                            param_reader);
        }
    }

    virtual
    ~H1ResidualErrorContainer()
    {
      if (PI_h_z_ != NULL)
        delete PI_h_z_;
      if (PI_h_u_ != NULL)
        delete PI_h_u_;

    }

    std::string
    GetName() const
    {
      return "H1-Residual-Estimator";
    }

    void
    Initialize(unsigned int state_n_blocks,
               std::vector<unsigned int> &state_block_component)
    {
      state_n_blocks_ = state_n_blocks;
      state_block_component_ = &state_block_component;
    }

    /**
     * ReInits the DWRDataContainer, the higher order STH
     * as well as the weight-vectors.
     */
    void
    ReInit(unsigned int n_elements);

    StateVector<VECTOR> &
    GetPI_h_u()
    {
      return *PI_h_u_;
    }

    StateVector<VECTOR> &
    GetPI_h_z()
    {
      return *PI_h_z_;
    }
    ControlVector<VECTOR> &
    GetPI_h_q()
    {
      throw DOpEException("There is no Control in PDE Problems!",
                          "H1ResidualErrorContainer::PreparePI_h_q");
    }

    /**
     * Makes the patchwise higher order interpolant of the
     * primal soltion u. This is needed as a weight for the
     * dual residual.
     */
    void
    PreparePI_h_u(const StateVector<VECTOR> & /*u*/)
    {
      BuildConstantWeight(&(GetSTH().GetStateDoFHandler()),
                          GetPI_h_u().GetSpacialVector());
    }

    /**
     * Makes the patchwise higher order interpolant of the
     * dual solution z. This is needed as a weight for the
     * primal residual.
     */
    void
    PreparePI_h_z(const StateVector<VECTOR> & /*z*/)
    {
      BuildConstantWeight(&(GetSTH().GetStateDoFHandler()),
                          GetPI_h_z().GetSpacialVector());
    }
    /**
     * Makes the patchwise higher order interpolant of the
     * control q. This is needed as a weight for the
     * control residual.
     */
    void
    PreparePI_h_q(const ControlVector<VECTOR> & /*q*/)
    {
      throw DOpEException("There is no Control in PDE Problems!",
                          "HigherOrderDWRContainer::PreparePI_h_q");

    }

    /**
     * Implementation of virtual method from base class.
     */
    bool
    NeedDual() const
    {
      return false;
    }

    /**
     * Implementation of virtual method from base class.
     */
    virtual DOpEtypes::WeightComputation
    GetWeightComputation() const
    {
      return DOpEtypes::element_diameter;
    }

    /**
     * Implementation of virtual method from base class.
     */
    virtual DOpEtypes::ResidualEvaluation
    GetResidualEvaluation() const
    {
      return DOpEtypes::strong_residual;
    }

    /**
     * This should be applied to the residual in the integration
     * To assert that the squared norm is calculated
     */
    inline void
    ResidualModifier(double &res)
    {
      res = res * res * weight_;
    }
    inline void
    VectorResidualModifier(dealii::Vector<double> &res)
    {
      for (unsigned int i = 0; i < res.size(); i++)
        res(i) = res(i) * res(i) * weight_;
    }

    void
    InitFace(double h)
    {
      weight_ = h;
    }
    void
    InitElement(double h)
    {
      weight_ = h * h;
    }

  protected:
    STH &
    GetSTH()
    {
      return sth_;
    }

    template<template<int, int> class DH>
    void
    BuildConstantWeight(const DOpEWrapper::DoFHandler<dim, DH> *dofh,
                        VECTOR &vals)
    {
      VectorTools::interpolate(sth_.GetMapping(),
                               dofh->GetDEALDoFHandler(),
                               ConstantFunction<dim>(1., dofh->get_fe().n_components()), vals);
    }

  private:
    unsigned int state_n_blocks_;
    std::vector<unsigned int> *state_block_component_;
    double weight_;

    STH &sth_;

    StateVector<VECTOR> *PI_h_u_, *PI_h_z_;
  };

  template<class STH, typename VECTOR, int dim>
  void
  H1ResidualErrorContainer<STH, VECTOR, dim>::ReInit(unsigned int n_elements)
  {
    DWRDataContainerBase<VECTOR>::ReInit(n_elements);

    if (this->GetEETerms() == DOpEtypes::primal_only
        || this->GetEETerms() == DOpEtypes::mixed)
      {
        GetPI_h_z().ReInit();
      }
    if (this->GetEETerms() == DOpEtypes::dual_only
        || this->GetEETerms() == DOpEtypes::mixed)
      {
        GetPI_h_u().ReInit();
      }
  }

}

#endif /* RESIDUAL_ERROR */
