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

#ifndef HIGHER_ORDER_DWRC_CONTROL_H_
#define HIGHER_ORDER_DWRC_CONTROL_H_

#include <container/dwrdatacontainer.h>
#include <deal.II/fe/fe_tools.h>

namespace DOpE
{
  /**
   * This class implements the missing pieces of DWRDataContainer for
   * the case of the DWRMethod with higher order interpolation of the weights
   * and evaluation of strong element residuals.
   * This version also includes weights for the control
   */
  template<class STH, class IDC, class EDC, class FDC, typename VECTOR>
  class HigherOrderDWRContainerControl : public DWRDataContainer<STH, IDC, EDC, FDC,
    VECTOR>
  {
  public:
    /**
       * Constructor.
       *
       * @param higher_order_sth      The STH we use for the higher order interpolation.
       * @param higher_order_idc      The IDC we use for the higher order interpolation.
       *                              Contains also the quadrature rules
       * @param control_behavior      Behaviour of the ControlVectors.
       * @param state_behavior        Behaviour of the StateVectors.
       * @param param_reader          The parameter reader we use here.
       * @param ee_terms              Which part of the error estimators do we want
       *                              to compute? (primal, dual, both).
       */
    HigherOrderDWRContainerControl(STH &higher_order_sth, IDC &higher_order_idc,
                                   DOpEtypes::VectorStorageType control_behavior, DOpEtypes::VectorStorageType state_behavior, ParameterReader &param_reader,
                                   DOpEtypes::EETerms ee_terms = DOpEtypes::EETerms::mixed,
                                   DOpEtypes::ResidualEvaluation res_eval = DOpEtypes::strong_residual)
      : DWRDataContainer<STH, IDC, EDC, FDC, VECTOR>(ee_terms), sth_higher_order_(
        higher_order_sth), idc_higher_order_(higher_order_idc), res_eval_(res_eval),
      PI_h_u_(NULL), PI_h_z_(NULL), PI_h_q_(NULL)
    {
      if (this->GetEETerms() == DOpEtypes::primal_only
          || this->GetEETerms() == DOpEtypes::mixed
          || this->GetEETerms() == DOpEtypes::mixed_control)
        {
          PI_h_z_ = new StateVector<VECTOR>(&GetHigherOrderSTH(),
                                            state_behavior, param_reader);
        }
      if (this->GetEETerms() == DOpEtypes::dual_only
          || this->GetEETerms() == DOpEtypes::mixed
          || this->GetEETerms() == DOpEtypes::mixed_control)
        {
          PI_h_u_ = new StateVector<VECTOR>(&GetHigherOrderSTH(),
                                            state_behavior, param_reader);
        }
      if (this->GetEETerms() == DOpEtypes::mixed_control)
        {
          PI_h_q_ = new ControlVector<VECTOR>(&GetHigherOrderSTH(),
                                              control_behavior);
        }
    }

    virtual
    ~HigherOrderDWRContainerControl()
    {
      if (PI_h_z_ != NULL)
        delete PI_h_z_;
      if (PI_h_u_ != NULL)
        delete PI_h_u_;
      if (PI_h_q_ != NULL)
        delete PI_h_q_;
    }

    std::string
    GetName() const
    {
      return "DWR-Estimator-with-Control";
    }

    template<class STH2>
    void
    Initialize(STH2 *sth, unsigned int control_n_blocks,
               std::vector<unsigned int> &control_block_component,
               unsigned int state_n_blocks,
               std::vector<unsigned int> &state_block_component,
               const std::vector<unsigned int> *cdcol,
               const std::vector<std::vector<bool> > *cdcomp,
               const std::vector<unsigned int> *sdcol,
               const std::vector<std::vector<bool> > *sdcomp)
    {
      sth_ = dynamic_cast<STH *>(sth);
      control_n_blocks_        =  control_n_blocks;
      control_block_component_ = &control_block_component;
      state_n_blocks_        =  state_n_blocks;
      state_block_component_ = &state_block_component;
      control_dirichlet_colors_ = cdcol;
      control_dirichlet_comps_ = cdcomp;
      state_dirichlet_colors_ = sdcol;
      state_dirichlet_comps_ = sdcomp;
    }

    /**
     * ReInits the DWRDataContainer, the higher order STH
     * as well as the weight-vectors.
     */
    void
    ReInit(unsigned int n_elements);

    STH &
    GetWeightSTH()
    {
      return GetHigherOrderSTH();
    }

    const STH &
    GetWeightSTH() const
    {
      return GetHigherOrderSTH();
    }

    IDC &
    GetWeightIDC()
    {
      return GetHigherOrderIDC();
    }

    const IDC &
    GetWeightIDC() const
    {
      return GetHigherOrderIDC();
    }

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
      return *PI_h_q_;
    }

    /**
     * Makes the patchwise higher order interpolant of the
     * primal soltion u. This is needed as a weight for the
     * dual residual.
     */
    void
    PreparePI_h_u(const StateVector<VECTOR> &u)
    {
      VECTOR u_high;
      u_high.reinit(GetPI_h_u().GetSpacialVector());

      dealii::FETools::extrapolate(
        GetSTH().GetStateDoFHandler().GetDEALDoFHandler(),
        u.GetSpacialVector(),
        GetHigherOrderSTH().GetStateDoFHandler().GetDEALDoFHandler(),
        GetHigherOrderSTH().GetStateDoFConstraints(),
        GetPI_h_u().GetSpacialVector());
      dealii::FETools::interpolate(
        GetSTH().GetStateDoFHandler().GetDEALDoFHandler(),
        u.GetSpacialVector(),
        GetHigherOrderSTH().GetStateDoFHandler().GetDEALDoFHandler(),
        GetHigherOrderSTH().GetStateDoFConstraints(), u_high);
      GetPI_h_u().GetSpacialVector().add(-1., u_high);
    }

    void
    PreparePI_h_u(const VECTOR &u)
    {
      VECTOR u_high;
      u_high.reinit(GetPI_h_u().GetSpacialVector());

      dealii::FETools::extrapolate(
        GetSTH().GetStateDoFHandler().GetDEALDoFHandler(), u,
        GetHigherOrderSTH().GetStateDoFHandler().GetDEALDoFHandler(),
        GetHigherOrderSTH().GetStateDoFConstraints(),
        GetPI_h_u().GetSpacialVector());
      dealii::FETools::interpolate(
        GetSTH().GetStateDoFHandler().GetDEALDoFHandler(), u,
        GetHigherOrderSTH().GetStateDoFHandler().GetDEALDoFHandler(),
        GetHigherOrderSTH().GetStateDoFConstraints(), u_high);
      GetPI_h_u().GetSpacialVector().add(-1., u_high);
    }

    /**
     * Makes the patchwise higher order interpolant of the
     * dual solution z. This is needed as a weight for the
     * primal residual.
     */
    void
    PreparePI_h_z(const StateVector<VECTOR> &z)
    {
      VECTOR z_high;
      z_high.reinit(GetPI_h_z().GetSpacialVector());

      dealii::FETools::extrapolate(
        GetSTH().GetStateDoFHandler().GetDEALDoFHandler(),
        z.GetSpacialVector(),
        GetHigherOrderSTH().GetStateDoFHandler().GetDEALDoFHandler(),
        GetHigherOrderSTH().GetStateDoFConstraints(),
        GetPI_h_z().GetSpacialVector());

      dealii::FETools::interpolate(
        GetSTH().GetStateDoFHandler().GetDEALDoFHandler(),
        z.GetSpacialVector(),
        GetHigherOrderSTH().GetStateDoFHandler().GetDEALDoFHandler(),
        GetHigherOrderSTH().GetStateDoFConstraints(), z_high);

      GetPI_h_z().GetSpacialVector().add(-1., z_high);

    }
    /**
     * Makes the patchwise higher order interpolant of the
     * control q. This is needed as a weight for the
     * control residual.
     */
    void
#if dope_dimension > 0
    PreparePI_h_q(const ControlVector<VECTOR> &q)
#else
    PreparePI_h_q(const ControlVector<VECTOR> &/*q*/)
#endif
    {
#if dope_dimension > 0
      VECTOR q_high;
      q_high.reinit(GetPI_h_q().GetSpacialVector());

      dealii::FETools::extrapolate(
        GetSTH().GetControlDoFHandler().GetDEALDoFHandler(),
        q.GetSpacialVector(),
        GetHigherOrderSTH().GetControlDoFHandler().GetDEALDoFHandler(),
        GetHigherOrderSTH().GetControlDoFConstraints(),
        GetPI_h_q().GetSpacialVector());

      dealii::FETools::interpolate(
        GetSTH().GetControlDoFHandler().GetDEALDoFHandler(),
        q.GetSpacialVector(),
        GetHigherOrderSTH().GetControlDoFHandler().GetDEALDoFHandler(),
        GetHigherOrderSTH().GetControlDoFConstraints(), q_high);

      GetPI_h_q().GetSpacialVector().add(-1., q_high);
      //FIXME With this construction we can not deal with control contraints,
      //There, the real weight needs to be build on the elements...
#else
      //The reasonable choice if control dimension is 0 is that the
      //The error in the control is 0.
      GetPI_h_q() = 0.;
#endif
    }

    /**
     * Implementation of virtual method from base class.
     */
    virtual EDC &
    GetElementWeight() const
    {
      return GetHigherOrderIDC().GetElementDataContainer();
    }

    /**
     * Implementation of virtual method from base class.
     */
    virtual FDC &
    GetFaceWeight() const
    {
      return GetHigherOrderIDC().GetFaceDataContainer();
    }

    /**
     * Implementation of virtual method from base class.
     */
    bool
    NeedDual() const
    {
      return true;
    }

    /**
     * Implementation of virtual method from base class.
     */
    virtual DOpEtypes::WeightComputation
    GetWeightComputation() const
    {
      return DOpEtypes::higher_order_interpolation;
    }

    /**
     * Implementation of virtual method from base class.
     */
    virtual DOpEtypes::ResidualEvaluation
    GetResidualEvaluation() const
    {
//          return DOpEtypes::strong_residual;
      return res_eval_;
    }

    /**
     * This should be applied to the residual in the integration
     * Here we don't do anything because it should be the identity for DWR
     */
    inline void
    ResidualModifier(double & /*res*/)
    {

    }
    inline void
    VectorResidualModifier(dealii::Vector<double> & /*res*/)
    {

    }

  protected:
    STH &
    GetSTH()
    {
      return *sth_;
    }

    STH &
    GetHigherOrderSTH()
    {
      return sth_higher_order_;
    }

    const STH &
    GetHigherOrderSTH() const
    {
      return sth_higher_order_;
    }

    IDC &
    GetHigherOrderIDC()
    {
      return idc_higher_order_;
    }

    const IDC &
    GetHigherOrderIDC() const
    {
      return idc_higher_order_;
    }

  private:
    unsigned int control_n_blocks_, state_n_blocks_;
    std::vector<unsigned int> *control_block_component_;
    std::vector<unsigned int> *state_block_component_;
    const std::vector<unsigned int> *control_dirichlet_colors_;
    const std::vector<std::vector<bool> > *control_dirichlet_comps_;
    const std::vector<unsigned int> *state_dirichlet_colors_;
    const std::vector<std::vector<bool> > *state_dirichlet_comps_;


    STH &sth_higher_order_;
    STH *sth_;
    IDC &idc_higher_order_;

    const DOpEtypes::ResidualEvaluation res_eval_;

    StateVector<VECTOR> *PI_h_u_, *PI_h_z_;
    ControlVector<VECTOR> *PI_h_q_;
  };

  template<class STH, class IDC, class EDC, class FDC, typename VECTOR>
  void
  HigherOrderDWRContainerControl<STH, IDC, EDC, FDC, VECTOR>::ReInit(
    unsigned int n_elements)
  {
    DWRDataContainer<STH, IDC, EDC, FDC, VECTOR>::ReInit(n_elements);

    GetHigherOrderSTH().ReInit(control_n_blocks_, *control_block_component_,DirichletDescriptor(*control_dirichlet_colors_,*control_dirichlet_comps_), state_n_blocks_, *state_block_component_,DirichletDescriptor(*state_dirichlet_colors_,*state_dirichlet_comps_));
    if (this->GetEETerms() == DOpEtypes::primal_only
        || this->GetEETerms() == DOpEtypes::mixed
        || this->GetEETerms() == DOpEtypes::mixed_control)
      {
        GetPI_h_z().ReInit();
      }
    if (this->GetEETerms() == DOpEtypes::dual_only
        || this->GetEETerms() == DOpEtypes::mixed
        || this->GetEETerms() == DOpEtypes::mixed_control)
      {
        GetPI_h_u().ReInit();
      }
    if ( this->GetEETerms() == DOpEtypes::mixed_control)
      {
        GetPI_h_q().ReInit();
      }
  }
}

#endif /* HIGHER_ORDER_DWRC_H_ */
