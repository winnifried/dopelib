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
  template<class STH, class IDC, class CDC, class FDC, typename VECTOR>
    class HigherOrderDWRContainer : public DWRDataContainer<CDC, FDC, VECTOR>
    {
      public:
        HigherOrderDWRContainer(STH& higher_order_sth, IDC& higher_order_idc,
            std::string state_behavior, ParameterReader &param_reader,
            DOpEtypes::EETerms ee_terms = DOpEtypes::EETerms::mixed)
            : DWRDataContainer<CDC, FDC, VECTOR>(ee_terms), _sth_higher_order(
                higher_order_sth), _idc_higher_order(higher_order_idc), _PI_h_u(
                NULL), _PI_h_z(NULL)
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
        }
        ;

        template<class STH2>
          void
          Initialize(STH2* sth, unsigned int state_n_blocks,
              std::vector<unsigned int>& state_block_component)
          {
            _sth = dynamic_cast<STH*>(sth);
            _state_n_blocks = state_n_blocks;
            _state_block_component = &state_block_component;
          }

        void
        ReInit(unsigned int n_cells);

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

        void
        PreparePI_h_u(const StateVector<VECTOR>& u)
        {
          VECTOR u_high;
          u_high.reinit(GetPI_h_u().GetSpacialVector());

          dealii::FETools::extrapolate(
              GetSTH().GetStateDoFHandler().GetDEALDoFHandler(),
              u.GetSpacialVector(),
              GetHigherOrderSTH().GetStateDoFHandler().GetDEALDoFHandler(),
              GetHigherOrderSTH().GetStateHangingNodeConstraints(),
              GetPI_h_u().GetSpacialVector());
          dealii::FETools::interpolate(
              GetSTH().GetStateDoFHandler().GetDEALDoFHandler(),
              u.GetSpacialVector(),
              GetHigherOrderSTH().GetStateDoFHandler().GetDEALDoFHandler(),
              GetHigherOrderSTH().GetStateHangingNodeConstraints(), u_high);
          GetPI_h_u().GetSpacialVector().add(-1., u_high);
        }
        void
        PreparePI_h_z(const StateVector<VECTOR>& z)
        {
          VECTOR z_high;
          z_high.reinit(GetPI_h_z().GetSpacialVector());

          dealii::FETools::extrapolate(
              GetSTH().GetStateDoFHandler().GetDEALDoFHandler(),
              z.GetSpacialVector(),
              GetHigherOrderSTH().GetStateDoFHandler().GetDEALDoFHandler(),
              GetHigherOrderSTH().GetStateHangingNodeConstraints(),
              GetPI_h_z().GetSpacialVector());
          dealii::FETools::interpolate(
              GetSTH().GetStateDoFHandler().GetDEALDoFHandler(),
              z.GetSpacialVector(),
              GetHigherOrderSTH().GetStateDoFHandler().GetDEALDoFHandler(),
              GetHigherOrderSTH().GetStateHangingNodeConstraints(), z_high);
          GetPI_h_z().GetSpacialVector().add(-1., z_high);

        }


        virtual CDC&
        GetCellWeight() const
        {
          return GetHigherOrderIDC().GetCellDataContainer();
        }

        FDC&
        GetFaceWeight() const
        {
          return GetHigherOrderIDC().GetFaceDataContainer();
        }

        bool
        NeedDual() const
        {
          return true;
        }

        virtual DOpEtypes::WeightComputation
        GetWeightComputation() const
        {
          return DOpEtypes::higher_order_interpolation;
        }
        virtual DOpEtypes::ResidualEvaluation
        GetResidualEvaluation() const
        {
          return DOpEtypes::strong_residual;
        }
      protected:
        STH&
        GetSTH()
        {
          return *_sth;
        }

      private:
        unsigned int _state_n_blocks;
        std::vector<unsigned int>* _state_block_component;

        STH& _sth_higher_order;
        STH* _sth;
        IDC& _idc_higher_order;

        StateVector<VECTOR> * _PI_h_u, *_PI_h_z;
    };

  template<class STH, class IDC, class CDC, class FDC, typename VECTOR>
    void
    HigherOrderDWRContainer<STH, IDC, CDC, FDC, VECTOR>::ReInit(
        unsigned int n_cells)
    {
      DWRDataContainer<CDC, FDC, VECTOR>::ReInit(n_cells);

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
