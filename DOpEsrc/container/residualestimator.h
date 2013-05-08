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

#ifndef _RESIDUAL_ERROR_H_
#define _RESIDUAL_ERROR_H_

#include "dwrdatacontainer.h"
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
        InitCell(double /*h*/) = 0;

    };

  /**
   * We need this overloaded function to have the same
   * interface for dwrdatacontainer and residualestimators.
   *
   * The function should actually never get called, but with this
   * construction, we save 4 unnecessary template parameters!
   */
  template<class CDC, typename VECTOR>
    CDC*
    ExtractCDC(const ResidualErrorContainer<VECTOR>& /*dwrc*/)
    {
      return NULL;
    }
  template<class FDC, typename VECTOR>
    FDC*
    ExtractFDC(const ResidualErrorContainer<VECTOR>& /*dwrc*/)
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
        L2ResidualErrorContainer(STH& sth, std::string state_behavior,
            ParameterReader &param_reader, DOpEtypes::EETerms ee_terms =
                DOpEtypes::EETerms::mixed) :
            ResidualErrorContainer<VECTOR>(ee_terms), _sth(sth), _PI_h_u(NULL), _PI_h_z(
                NULL)
        {
          if (this->GetEETerms() == DOpEtypes::primal_only
              || this->GetEETerms() == DOpEtypes::mixed)
          {
            _PI_h_z = new StateVector<VECTOR>(&GetSTH(), state_behavior,
                param_reader);
          }
          if (this->GetEETerms() == DOpEtypes::dual_only
              || this->GetEETerms() == DOpEtypes::mixed)
          {
            _PI_h_u = new StateVector<VECTOR>(&GetSTH(), state_behavior,
                param_reader);
          }
          _weight = 0.;
        }

        virtual
        ~L2ResidualErrorContainer()
        {
          if (_PI_h_z != NULL)
            delete _PI_h_z;
          if (_PI_h_u != NULL)
            delete _PI_h_u;

        }

        std::string
        GetName() const
        {
          return "L2-Residual-Estimator";
        }

        void
        Initialize(unsigned int state_n_blocks,
            std::vector<unsigned int>& state_block_component)
        {
          _state_n_blocks = state_n_blocks;
          _state_block_component = &state_block_component;
        }

        /**
         * ReInits the DWRDataContainer, the higher order STH
         * as well as the weight-vectors.
         */
        void
        ReInit(unsigned int n_cells);

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
        ControlVector<VECTOR>&
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
        PreparePI_h_u(const StateVector<VECTOR>& /*u*/)
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
        PreparePI_h_z(const StateVector<VECTOR>& /*z*/)
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
        PreparePI_h_q(const ControlVector<VECTOR>& /*q*/)
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
          return DOpEtypes::cell_diameter;
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
        ResidualModifier(double& res)
        {
          res = res * res * _weight;
        }

        inline void
        VectorResidualModifier(dealii::Vector<double>& res)
        {
          for (unsigned int i = 0; i < res.size(); i++)
            res(i) = res(i) * res(i) * _weight;
        }

        void
        InitFace(double h)
        {
          _weight = h * h * h;
        }
        void
        InitCell(double h)
        {
          _weight = h * h * h * h;
        }

      protected:
        STH&
        GetSTH()
        {
          return _sth;
        }

        template<template<int, int> class DH>
          void
          BuildConstantWeight(const DOpEWrapper::DoFHandler<dim, DH>* dofh,
              VECTOR& vals)
          {
            VectorTools::interpolate(_sth.GetMapping(),
                *(static_cast<const DH<dim, dim>*>(dofh)),
                ConstantFunction<dim>(1., dofh->get_fe().n_components()), vals);
          }

      private:
        unsigned int _state_n_blocks;
        std::vector<unsigned int>* _state_block_component;
        double _weight;

        STH& _sth;

        StateVector<VECTOR> * _PI_h_u, *_PI_h_z;
    };

  template<class STH, typename VECTOR, int dim>
    void
    L2ResidualErrorContainer<STH, VECTOR, dim>::ReInit(unsigned int n_cells)
    {
      DWRDataContainerBase<VECTOR>::ReInit(n_cells);

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
        H1ResidualErrorContainer(STH& sth, std::string state_behavior,
            ParameterReader &param_reader, DOpEtypes::EETerms ee_terms =
                DOpEtypes::EETerms::mixed) :
            ResidualErrorContainer<VECTOR>(ee_terms), _sth(sth), _PI_h_u(NULL), _PI_h_z(
                NULL)
        {
          if (this->GetEETerms() == DOpEtypes::primal_only
              || this->GetEETerms() == DOpEtypes::mixed)
          {
            _PI_h_z = new StateVector<VECTOR>(&GetSTH(), state_behavior,
                param_reader);
          }
          if (this->GetEETerms() == DOpEtypes::dual_only
              || this->GetEETerms() == DOpEtypes::mixed)
          {
            _PI_h_u = new StateVector<VECTOR>(&GetSTH(), state_behavior,
                param_reader);
          }
        }

        virtual
        ~H1ResidualErrorContainer()
        {
          if (_PI_h_z != NULL)
            delete _PI_h_z;
          if (_PI_h_u != NULL)
            delete _PI_h_u;

        }

        std::string
        GetName() const
        {
          return "H1-Residual-Estimator";
        }

        void
        Initialize(unsigned int state_n_blocks,
            std::vector<unsigned int>& state_block_component)
        {
          _state_n_blocks = state_n_blocks;
          _state_block_component = &state_block_component;
        }

        /**
         * ReInits the DWRDataContainer, the higher order STH
         * as well as the weight-vectors.
         */
        void
        ReInit(unsigned int n_cells);

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
        ControlVector<VECTOR>&
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
        PreparePI_h_u(const StateVector<VECTOR>& /*u*/)
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
        PreparePI_h_z(const StateVector<VECTOR>& /*z*/)
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
        PreparePI_h_q(const ControlVector<VECTOR>& /*q*/)
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
          return DOpEtypes::cell_diameter;
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
        ResidualModifier(double& res)
        {
          res = res * res * _weight;
        }
        inline void
        VectorResidualModifier(dealii::Vector<double>& res)
        {
          for (unsigned int i = 0; i < res.size(); i++)
            res(i) = res(i) * res(i) * _weight;
        }

        void
        InitFace(double h)
        {
          _weight = h;
        }
        void
        InitCell(double h)
        {
          _weight = h * h;
        }

      protected:
        STH&
        GetSTH()
        {
          return _sth;
        }

        template<template<int, int> class DH>
          void
          BuildConstantWeight(const DOpEWrapper::DoFHandler<dim, DH>* dofh,
              VECTOR& vals)
          {
            VectorTools::interpolate(_sth.GetMapping(),
                dofh->GetDEALDoFHandler(),
                ConstantFunction<dim>(1., dofh->get_fe().n_components()), vals);
          }

      private:
        unsigned int _state_n_blocks;
        std::vector<unsigned int>* _state_block_component;
        double _weight;

        STH& _sth;

        StateVector<VECTOR> * _PI_h_u, *_PI_h_z;
    };

  template<class STH, typename VECTOR, int dim>
    void
    H1ResidualErrorContainer<STH, VECTOR, dim>::ReInit(unsigned int n_cells)
    {
      DWRDataContainerBase<VECTOR>::ReInit(n_cells);

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
