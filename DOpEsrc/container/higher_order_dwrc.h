/*
 * higher_order_dwrc.h
 *
 *  Created on: Mar 23, 2012
 *      Author: cgoll
 */

#ifndef _HIGHER_ORDER_DWRC_H_
#define _HIGHER_ORDER_DWRC_H_

#include "dwrdatacontainer.h"
#include <deal.II/fe/fe_tools.h>

namespace DOpE
{
  /**
   * This class implements the missing pieces of DWRDataContainer for
   * the case of the DWRMethod with higher order interpolation of the weights
   * and evaluation of strong cell residuals.
   */
  template<class STH, class IDC, class CDC, class FDC, typename VECTOR>
    class HigherOrderDWRContainer : public DWRDataContainer<STH, IDC, CDC, FDC,
        VECTOR>
    {
      public:
        /**
         * Constructor.
         *
         * @param higher_order_sth      The STH we use for the higher order interpolation.
         * @param higher_order_idc      The IDC we use for the higher order interpolation.
         *                              Contains also the quadrature rules
         * @param state_behavior        Behaviour of the StateVectors.
         * @param param_reader          The parameter reader we use here.
         * @param ee_terms              Which part of the error estimators do we want
         *                              to compute? (primal, dual, both).
         */
        HigherOrderDWRContainer(STH& higher_order_sth, IDC& higher_order_idc,
            std::string state_behavior, ParameterReader &param_reader,
            DOpEtypes::EETerms ee_terms = DOpEtypes::EETerms::mixed,
            DOpEtypes::ResidualEvaluation res_eval = DOpEtypes::strong_residual)
            : DWRDataContainer<STH, IDC, CDC, FDC, VECTOR>(ee_terms), _sth_higher_order(
                higher_order_sth), _idc_higher_order(higher_order_idc), _PI_h_u(
                NULL), _PI_h_z(NULL), _res_eval(res_eval)
        {
          if (this->GetEETerms() == DOpEtypes::primal_only
              || this->GetEETerms() == DOpEtypes::mixed)
          {
            _PI_h_z = new StateVector<VECTOR>(&GetHigherOrderSTH(),
                state_behavior, param_reader);
          }
          if (this->GetEETerms() == DOpEtypes::dual_only
              || this->GetEETerms() == DOpEtypes::mixed)
          {
            _PI_h_u = new StateVector<VECTOR>(&GetHigherOrderSTH(),
                state_behavior, param_reader);
          }
        }

        virtual
        ~HigherOrderDWRContainer()
        {
          if (_PI_h_z != NULL)
            delete _PI_h_z;
          if (_PI_h_u != NULL)
            delete _PI_h_u;
        }

        std::string
        GetName() const
        {
          return "DWR-Estimator";
        }

        template<class STH2>
          void
          Initialize(STH2* sth, unsigned int state_n_blocks,
              std::vector<unsigned int>& state_block_component)
          {
            _sth = dynamic_cast<STH*>(sth);
            _state_n_blocks = state_n_blocks;
            _state_block_component = &state_block_component;
          }

        /**
         * ReInits the DWRDataContainer, the higher order STH
         * as well as the weight-vectors.
         */
        void
        ReInit(unsigned int n_cells);

        STH&
        GetWeightSTH()
        {
          return GetHigherOrderSTH();
        }

        const STH&
        GetWeightSTH() const
        {
          return GetHigherOrderSTH();
        }

        IDC&
        GetWeightIDC()
        {
          return GetHigherOrderIDC();
        }

        const IDC&
        GetWeightIDC() const
        {
          return GetHigherOrderIDC();
        }

        StateVector<VECTOR>&
        GetPI_h_u()
        {
          return *_PI_h_u;
        }

        StateVector<VECTOR>&
        GetPI_h_z()
        {
          return *_PI_h_z;
        }

        /**
         * Makes the patchwise higher order interpolant of the
         * primal soltion u. This is needed as a weight for the
         * dual residual.
         */
        void
        PreparePI_h_u(const StateVector<VECTOR>& u)
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
        PreparePI_h_u(const VECTOR& u)
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
        PreparePI_h_z(const StateVector<VECTOR>& z)
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
         * Implementation of virtual method from base class.
         */
        virtual CDC&
        GetCellWeight() const
        {
          return GetHigherOrderIDC().GetCellDataContainer();
        }

        /**
         * Implementation of virtual method from base class.
         */
        virtual FDC&
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
          return _res_eval;
        }

        /**
         * This should be applied to the residual in the integration
         * To assert that the squared norm is calculated
         */
        inline double
        ResidualModifier(double res)
        {
          return res;
        }

      protected:
        STH&
        GetSTH()
        {
          return *_sth;
        }

        STH&
        GetHigherOrderSTH()
        {
          return _sth_higher_order;
        }

        const STH&
        GetHigherOrderSTH() const
        {
          return _sth_higher_order;
        }

        IDC&
        GetHigherOrderIDC()
        {
          return _idc_higher_order;
        }

        const IDC&
        GetHigherOrderIDC() const
        {
          return _idc_higher_order;
        }

      private:
        unsigned int _state_n_blocks;
        std::vector<unsigned int>* _state_block_component;

        STH& _sth_higher_order;
        STH* _sth;
        IDC& _idc_higher_order;

        const DOpEtypes::ResidualEvaluation _res_eval;

        StateVector<VECTOR> * _PI_h_u, *_PI_h_z;
    };

  template<class STH, class IDC, class CDC, class FDC, typename VECTOR>
    void
    HigherOrderDWRContainer<STH, IDC, CDC, FDC, VECTOR>::ReInit(
        unsigned int n_cells)
    {
      DWRDataContainer<STH, IDC, CDC, FDC, VECTOR>::ReInit(n_cells);

      GetHigherOrderSTH().ReInit(_state_n_blocks, *_state_block_component);
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

#endif /* HIGHER_ORDER_DWRC_H_ */
