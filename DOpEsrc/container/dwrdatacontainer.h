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

#ifndef DWRDATACONTAINER_H_
#define DWRDATACONTAINER_H_

#include <include/constraintvector.h>
#include <include/controlvector.h>
#include <include/statevector.h>
#include <basic/dopetypes.h>
#include <include/parameterreader.h>
#include <basic/dopetypes.h>

namespace DOpE
{

  /**
   * This class hosts all the information we need for the
   * error evaluation (weights, additional needed DoF handler, error indicators.)
   *
   * DWRDataContainerBase is a base/interface class. As there are different possibilities
   * to implement the DWR method, for example the computation of the weights
   * is outsourced to the derived classes.
   *
   * @template VECTOR   Vector type used in the computation used in
   *                    the computation of the PDE solution.
   *
   */
  template<typename VECTOR>
  class DWRDataContainerBase
  {
  public:
    DWRDataContainerBase(DOpEtypes::EETerms ee_terms =
                           DOpEtypes::EETerms::mixed)
      : ee_terms_(ee_terms)
    {
      lock_ = true;
      switch (this->GetEETerms())
        {
        case DOpEtypes::primal_only:
          n_error_comps_ = 2;
          break;
        case DOpEtypes::dual_only:
          n_error_comps_ = 2;
          break;
        case DOpEtypes::mixed:
          n_error_comps_ = 2;
          break;
        case DOpEtypes::mixed_control:
          n_error_comps_ = 3;
          break;
        default:
          throw DOpEException("Unknown DOpEtypes::EEterms.",
                              "DWRDataContainer::ReleaseLock");
          break;
        }
    }

    virtual
    ~DWRDataContainerBase()
    {
    }

    virtual std::string
    GetName() const = 0;

    /**
     * This initializes the vector of the error indicators and locks them.
     * The vector of the error indicators can only get returned if the lock
     * is released (see ReleaseLock()).
     */
    virtual void
    ReInit(unsigned int n_elements);

    /**
     * Releases the lock and fills the vector of error indicators
     * accordingly to the previously given EETerms-enum (see dopetypes.h).
     * It basically sums up the primal and dual error indicators using
     * the rule given by EETerms.
     */
    void
    ReleaseLock()
    {
      lock_ = false;

      switch (this->GetEETerms())
        {
        case DOpEtypes::primal_only:
          error_ind_ = GetPrimalErrorIndicators();
          break;
        case DOpEtypes::dual_only:
          error_ind_ = GetDualErrorIndicators();
          break;
        case DOpEtypes::mixed:
          error_ind_.equ(0.5, GetPrimalErrorIndicators());
          error_ind_.add(0.5, GetDualErrorIndicators());
          break;
        case DOpEtypes::mixed_control:
          error_ind_.equ(0.5, GetPrimalErrorIndicators());
          error_ind_.add(0.5, GetDualErrorIndicators());
          error_ind_.add(0.5, GetControlErrorIndicators());
          break;
        default:
          throw DOpEException("Unknown DOpEtypes::EEterms.",
                              "DWRDataContainer::ReleaseLock");
          break;
        }
    }

    /**
     * This function sums up the entries of the vector of the error
     * indicators. So make sure this vector is correctly filled (reminder:
     * after computing the error indicators, make sure that you call
     * ReleaseLock()1)
     *
     * @ return   Error in the previously specified functional.
     */
    double
    GetError() const
    {
      double error = 0;
      for (unsigned int i = 0; i < GetErrorIndicators().size(); ++i)
        {
          error += GetErrorIndicators()(i);
        }
      return error;
    }

    /**
     * This function sums up the entries of the vector of the error
     * indicators. So make sure this vector is correctly filled (reminder:
     * after computing the error indicators, make sure that you call
     * ReleaseLock()1)
     *
     * @ return   Error in the previously specified functional.
     */
    double
    GetPrimalError() const
    {
      double error = 0;
      for (unsigned int i = 0; i < GetAllErrorIndicators()[1]->size(); ++i)
        {
          error += GetAllErrorIndicators()[1]->operator()(i);
        }
      return error;
    }

    /**
     * This function sums up the entries of the vector of the error
     * indicators. So make sure this vector is correctly filled (reminder:
     * after computing the error indicators, make sure that you call
     * ReleaseLock()1)
     *
     * @ return   Error in the previously specified functional.
     */
    double
    GetDualError() const
    {
      double error = 0;
      for (unsigned int i = 0; i < GetAllErrorIndicators()[2]->size(); ++i)
        {
          error += GetAllErrorIndicators()[2]->operator()(i);
        }
      return error;
    }

    /**
     * This function sums up the entries of the vector of the error
     * indicators. So make sure this vector is correctly filled (reminder:
     * after computing the error indicators, make sure that you call
     * ReleaseLock()1)
     *
     * @ return   Error in the previously specified functional.
     */
    double
    GetControlError() const
    {
      double error = 0;
      for (unsigned int i = 0; i < GetAllErrorIndicators()[3]->size(); ++i)
        {
          error += GetAllErrorIndicators()[3]->operator()(i);
        }
      return error;
    }

    /**
     * Returns the vector of the error indicators. You have to
     * call ReleaseLock() prior to this function.
     *
     * @return  Vector of raw error indicators (i.e. with sign)
     */
    const Vector<double> &
    GetErrorIndicators() const
    {
      if (lock_)
        {
          throw DOpEException("Error indicators are still locked.",
                              "DWRDataContainer::GetErrorIndicators");
        }
      else
        {
          return error_ind_;
        }
    }

    /**
     * Returns the vector of the error indicators. You have to
     * call ReleaseLock() prior to this function.
     *
     * @return  Vector of raw primal error indicators (i.e. with sign)
     */
    Vector<double> &
    GetPrimalErrorIndicators()
    {
      return error_ind_primal_;
    }

    const Vector<double> &
    GetPrimalErrorIndicators() const
    {
      return error_ind_primal_;
    }

    /**
     * Returns the vector of the dual error indicators. You dont have
     * to call ReleaseLock() prior to this function.
     *
     * @return  Vector of raw dual error indicators (i.e. with sign)
     */
    Vector<double> &
    GetDualErrorIndicators()
    {
      return error_ind_dual_;
    }

    const Vector<double> &
    GetDualErrorIndicators() const
    {
      return error_ind_dual_;
    }

    /**
     * Returns the vector of the error indicators. You have to
     * call ReleaseLock() prior to this function.
     *
     * @return  Vector of raw control error indicators (i.e. with sign)
     */
    Vector<double> &
    GetControlErrorIndicators()
    {
      return error_ind_control_;
    }

    const Vector<double> &
    GetControlErrorIndicators() const
    {
      return error_ind_control_;
    }

    /**
     * Returns the vector of the error indicators. You have to
     * call ReleaseLock() prior to this function.
     *
     * @return  Vector of raw error indicators given by the index
    *           0 = primal, 1 = dual, 2 = control
     */
    Vector<double> &
    GetErrorIndicators(unsigned int i)
    {
      if ( i == 0)
        return error_ind_primal_;
      else if (i == 1)
        return error_ind_dual_;
      else if (i == 2)
        return error_ind_control_;
      else
        throw DOpEException("Unknown Indicator","DWRDataContainer::GetErrorIndicators");
    }

    const Vector<double> &
    GetErrorIndicators(unsigned int i) const
    {
      if ( i == 0)
        return error_ind_primal_;
      else if (i == 1)
        return error_ind_dual_;
      else if (i == 2)
        return error_ind_control_;
      else
        throw DOpEException("Unknown Indicator","DWRDataContainer::GetErrorIndicators");
    }

    unsigned int GetNErrorComps() const
    {
      //number of error components...
      return n_error_comps_;
    }

    /**
     * Returns the a vector of pointers to the primal, dual and 'summed up'
     * (according to the enum EEterms) error indicators.
     *
     * @return  Vector of pointers to the 'summed up' error indicators, the primal
     *          error indicators and the dual indicators (in this order).
     */
    std::vector<const Vector<double>*>
    GetAllErrorIndicators() const
    {
      std::vector<const Vector<double>*> res;
      if (lock_)
        {
          throw DOpEException("Error indicators are still locked.",
                              "DWRDataContainer::GetErrorIndicators");
        }
      else
        {
          res.push_back(&error_ind_);
          res.push_back(&this->GetPrimalErrorIndicators());
          res.push_back(&this->GetDualErrorIndicators());
          res.push_back(&this->GetControlErrorIndicators());
        }
      return res;
    }

    /**
     * @return    How do we compute the weights? See dopetypes.h for the possibilities.
     */
    virtual DOpEtypes::WeightComputation
    GetWeightComputation() const = 0;

    /**
     * @return    In which form do we evaluate the residuals? See dopetypes.h for the possibilities.
     */
    virtual DOpEtypes::ResidualEvaluation
    GetResidualEvaluation() const =0;

    /**
     * @return   Which terms do we compute for the error evaluation? See dopetypes.h for the possibilities.
     */
    DOpEtypes::EETerms
    GetEETerms() const
    {
      return ee_terms_;
    }

    /**
     * TODO We would like DWRDataContainerBase in the solution algorithms,
     * but we need the specialication  in the integrator.ComputeRefinementIndicators.
     * How to achieve this?
     */
    template<class PROBLEM, class INTEGRATOR>
    void
    ComputeRefinementIndicators(PROBLEM &problem, INTEGRATOR &integrator)
    {
      integrator.ComputeRefinementIndicators(problem, *this);
    }

    /**
     * Specifies, if we need the solution of the adjoint equation.
     * Pure virtual.
     *
     * @return    Do we need the computation of a adjoint equation?
     */
    virtual bool
    NeedDual() const = 0;

    /**
     * Returns the FE-VECTORS of the weights used in the error evaluation.
     *
     * @return  maps between string and VECTOR-pointer, where the latter hold
     *          the information of the weight-functions.
     */
    const std::map<std::string, const VECTOR *> &
    GetWeightData() const
    {
      return weight_data_;
    }

    /**
     * Deletes the weights.
     */
    void
    ClearWeightData()
    {
      weight_data_.clear();
    }

    /**
     * Computes the functions that compute the weights and puts them
     * into weight_data_.
     *
     * @param u   The FE-vector of the primal solution.
     * @param z   The FE-vector of the dual solution.
     */
    void
    PrepareWeights(const StateVector<VECTOR> &u, const StateVector<VECTOR> &z)
    {
      //Dependend on the GetEETerms we let the dwrc compute the primal and/or dual weights
      switch (GetEETerms())
        {
        case DOpEtypes::primal_only:
          PreparePI_h_z(z);
          AddWeightData("weight_for_primal_residual",
                        &(GetPI_h_z().GetSpacialVector()));
          break;
        case DOpEtypes::dual_only:
          PreparePI_h_u(u);
          AddWeightData("weight_for_dual_residual",
                        &(GetPI_h_u().GetSpacialVector()));
          break;
        case DOpEtypes::mixed:
          PreparePI_h_u(u);
          AddWeightData("weight_for_dual_residual",
                        &(GetPI_h_u().GetSpacialVector()));
          PreparePI_h_z(z);
          AddWeightData("weight_for_primal_residual",
                        &(GetPI_h_z().GetSpacialVector()));
          break;
        case DOpEtypes::mixed_control:
          PreparePI_h_u(u);
          AddWeightData("weight_for_dual_residual",
                        &(GetPI_h_u().GetSpacialVector()));
          PreparePI_h_z(z);
          AddWeightData("weight_for_primal_residual",
                        &(GetPI_h_z().GetSpacialVector()));
          break;
        default:
          throw DOpEException("Unknown DOpEtypes::EETerms!",
                              "DWRDataContainerBase::PrepareWeights");
          break;
        }
    }
    /**
     * Computes the functions that compute the weights and puts them
     * into weight_data_.
     *
     * @param q   The FE-vector of the control.
     */
    void
    PrepareWeights(const ControlVector<VECTOR> &q)
    {
      //Dependend on the GetEETerms we let the dwrc compute the primal and/or dual weights
      switch (GetEETerms())
        {
#if dope_dimension > 0
        case DOpEtypes::mixed_control:
          PreparePI_h_q(q);
          AddWeightData("weight_for_control_residual",
                        &(GetPI_h_q().GetSpacialVector()));
          break;
#endif
        case DOpEtypes::primal_only:
          //Do nothing to be compatible with residual estimators not using the control!
          break;
        default:
          throw DOpEException("Unknown DOpEtypes::EETerms!",
                              "DWRDataContainerBase::PrepareWeights");
          break;
        }
    }

  protected:
    //TODO PI_h_u and PI_h_z are probably not the best names.
    //Basically, one could replace them with 'primalweight' and 'dualweight',
    //but i found them a little bit misleading. ~cg
    virtual StateVector<VECTOR> &
    GetPI_h_u() = 0;

    virtual StateVector<VECTOR> &
    GetPI_h_z() = 0;

    virtual ControlVector<VECTOR> &
    GetPI_h_q() = 0;

    virtual void
    PreparePI_h_u(const StateVector<VECTOR> &u) = 0;

    virtual void
    PreparePI_h_z(const StateVector<VECTOR> &z) = 0;

    virtual void
    PreparePI_h_q(const ControlVector<VECTOR> &q) = 0;

    void
    AddWeightData(std::string name, const VECTOR *new_data)
    {
      if (weight_data_.find(name) != weight_data_.end())
        {
          throw DOpEException(
            "Adding multiple Data with name " + name + " is prohibited!",
            "Integrator::AddDomainData");
        }
      weight_data_.insert(
        std::pair<std::string, const VECTOR *>(name, new_data));
    }

  private:
    DOpEtypes::EETerms ee_terms_;
    bool lock_;
    unsigned int n_error_comps_;
    std::map<std::string, const VECTOR *> weight_data_;
    Vector<double> error_ind_, error_ind_primal_, error_ind_dual_, error_ind_control_;
  }
  ;

  template<typename VECTOR>
  void
  DWRDataContainerBase<VECTOR>::ReInit(unsigned int n_elements)
  {
    error_ind_.reinit(n_elements);
    GetPrimalErrorIndicators().reinit(n_elements);
    GetDualErrorIndicators().reinit(n_elements);
    GetControlErrorIndicators().reinit(n_elements);
    lock_ = true;
  }

  /**
   * Adds just the pure virtual functions GetElementWeight() and GetFaceWeight().
   * They have to get implemented in derived classes. These two methods are
   * excluded from DWRDataContainerBase() to save two template parameters.
   */
  template<class STH, class IDC, class EDC, class FDC, typename VECTOR>
  class DWRDataContainer : public DWRDataContainerBase<VECTOR>
  {
  public:
    DWRDataContainer(
      DOpEtypes::EETerms ee_terms = DOpEtypes::EETerms::mixed)
      : DWRDataContainerBase<VECTOR>(ee_terms)
    {
    }
    ;
    virtual
    ~DWRDataContainer()
    {
    }
    ;

    /**
     * Returns a ElementDataContainer for the weights on the
     * elements. Pure virtual.
     */
    virtual EDC &
    GetElementWeight() const = 0;

    /**
     * Returns a FaceDataContainer for the weight on the
     * faces (and boundaries.). Pure virtual.
     */
    virtual FDC &
    GetFaceWeight() const = 0;

    virtual STH &
    GetWeightSTH() = 0;
    virtual const STH &
    GetWeightSTH() const = 0;

    virtual IDC &
    GetWeightIDC() = 0;
    virtual const IDC &
    GetWeightIDC() const = 0;
  };

  /**
   * We need this overloaded function to have the same
   * interface for dwrdatacontainer and residualestimators,
   * see there for further information.
   */
  template<class EDC, class STH, class IDC, class FDC, typename VECTOR>
  EDC *
  ExtractEDC(const DWRDataContainer<STH, IDC, EDC, FDC, VECTOR> &dwrc)
  {
    return &dwrc.GetElementWeight();
  }
  template<class FDC, class STH, class IDC, class EDC, typename VECTOR>
  FDC *
  ExtractFDC(const DWRDataContainer<STH, IDC, EDC, FDC, VECTOR> &dwrc)
  {
    return &dwrc.GetFaceWeight();
  }

} //end of namespace
#endif /* DWRDATACONTAINER_H_ */
