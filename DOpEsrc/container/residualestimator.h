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

#ifndef RESIDUAL_ERROR_H_
#define RESIDUAL_ERROR_H_

#include <container/dwrdatacontainer.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/base/function.h>

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

    virtual
    ~ResidualErrorContainer() {}

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
    GetName() const override
    {
      return "L2-Residual-Estimator";
    }

    void
    Initialize(unsigned int /*state_n_blocks*/,
               std::vector<unsigned int> &/*state_block_component*/)
    {
    }

    /**
     * ReInits the DWRDataContainer, the higher order STH
     * as well as the weight-vectors.
     */
    void
    ReInit();

    StateVector<VECTOR> &
    GetPI_h_u() override
    {
      return *PI_h_u_;
    }

    StateVector<VECTOR> &
    GetPI_h_z() override
    {
      return *PI_h_z_;
    }
    ControlVector<VECTOR> &
    GetPI_h_q() override
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
    PreparePI_h_u(const StateVector<VECTOR> & /*u*/) override
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
    PreparePI_h_z(const StateVector<VECTOR> & /*z*/) override
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
    PreparePI_h_q(const ControlVector<VECTOR> & /*q*/) override
    {
      throw DOpEException("There is no Control in PDE Problems!",
                          "HigherOrderDWRContainer::PreparePI_h_q");

    }

    /**
     * Implementation of virtual method from base class.
     */
    bool
    NeedDual() const override
    {
      return false;
    }

    /**
     * Implementation of virtual method from base class.
     */
    virtual DOpEtypes::WeightComputation
    GetWeightComputation() const override
    {
      return DOpEtypes::element_diameter;
    }

    /**
     * Implementation of virtual method from base class.
     */
    virtual DOpEtypes::ResidualEvaluation
    GetResidualEvaluation() const override
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
    InitFace(double h) override
    {
      weight_ = h * h * h;
    }
    void
    InitElement(double h) override
    {
      weight_ = h * h * h * h;
    }

  protected:
    STH &
    GetSTH()
    {
      return sth_;
    }

#if DEAL_II_VERSION_GTE(9,3,0)
#else
    template<template<int, int> class DH>
#endif
    void
#if DEAL_II_VERSION_GTE(9,3,0)
    BuildConstantWeight(const DOpEWrapper::DoFHandler<dim> *dofh,
#else
    BuildConstantWeight(const DOpEWrapper::DoFHandler<dim, DH> *dofh,
#endif
                        VECTOR &vals)
    {
      VectorTools::interpolate(sth_.GetMapping(),
#if DEAL_II_VERSION_GTE(9,3,0)
                               *(static_cast<const dealii::DoFHandler<dim, dim>*>(dofh)),
#else
                               *(static_cast<const DH<dim, dim>*>(dofh)),
#endif
#if DEAL_II_VERSION_GTE(9,3,0)
                               Functions::ConstantFunction<dim>(1., dofh->get_fe().n_components()), vals);
#else
                               ConstantFunction<dim>(1., dofh->get_fe().n_components()), vals);
#endif
    }

  private:
    double weight_;

    STH &sth_;

    StateVector<VECTOR> *PI_h_u_, *PI_h_z_;
  };

  template<class STH, typename VECTOR, int dim>
  void
  L2ResidualErrorContainer<STH, VECTOR, dim>::ReInit()
  {
    DWRDataContainerBase<VECTOR>::ReInit(GetSTH());

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
    GetName() const override
    {
      return "H1-Residual-Estimator";
    }

    void
    Initialize(unsigned int /*state_n_blocks*/,
               std::vector<unsigned int> &/*state_block_component*/)
    {
    }

    /**
     * ReInits the DWRDataContainer, the higher order STH
     * as well as the weight-vectors.
     */
    void
    ReInit();

    StateVector<VECTOR> &
    GetPI_h_u() override
    {
      return *PI_h_u_;
    }

    StateVector<VECTOR> &
    GetPI_h_z() override
    {
      return *PI_h_z_;
    }
    ControlVector<VECTOR> &
    GetPI_h_q() override
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
    PreparePI_h_u(const StateVector<VECTOR> & /*u*/) override
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
    PreparePI_h_z(const StateVector<VECTOR> & /*z*/) override
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
    PreparePI_h_q(const ControlVector<VECTOR> & /*q*/) override
    {
      throw DOpEException("There is no Control in PDE Problems!",
                          "HigherOrderDWRContainer::PreparePI_h_q");

    }

    /**
     * Implementation of virtual method from base class.
     */
    bool
    NeedDual() const override
    {
      return false;
    }

    /**
     * Implementation of virtual method from base class.
     */
    virtual DOpEtypes::WeightComputation
    GetWeightComputation() const override
    {
      return DOpEtypes::element_diameter;
    }

    /**
     * Implementation of virtual method from base class.
     */
    virtual DOpEtypes::ResidualEvaluation
    GetResidualEvaluation() const override
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
    InitFace(double h) override
    {
      weight_ = h;
    }
    void
    InitElement(double h) override
    {
      weight_ = h * h;
    }

  protected:
    STH &
    GetSTH()
    {
      return sth_;
    }

#if DEAL_II_VERSION_GTE(9,3,0)

#else
    template<template<int, int> class DH>
#endif
    void
#if DEAL_II_VERSION_GTE(9,3,0)
    BuildConstantWeight(const DOpEWrapper::DoFHandler<dim> *dofh,
#else
    BuildConstantWeight(const DOpEWrapper::DoFHandler<dim, DH> *dofh,
#endif
                        VECTOR &vals)
    {
      VectorTools::interpolate(sth_.GetMapping(),
                               dofh->GetDEALDoFHandler(),
#if DEAL_II_VERSION_GTE(9,3,0)
                               Functions::ConstantFunction<dim>(1., dofh->get_fe().n_components()), vals);
#else
                               ConstantFunction<dim>(1., dofh->get_fe().n_components()), vals);
#endif
    }

  private:
    double weight_;

    STH &sth_;

    StateVector<VECTOR> *PI_h_u_, *PI_h_z_;
  };

  template<class STH, typename VECTOR, int dim>
  void
  H1ResidualErrorContainer<STH, VECTOR, dim>::ReInit()
  {
    DWRDataContainerBase<VECTOR>::ReInit(GetSTH());

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
