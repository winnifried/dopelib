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
      : ee_terms_(ee_terms), current_time_dof_(0)
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
     * This sets the vector to a given time-dof. 
     */
    void SetTime(unsigned int time_dof);
    
    /**
     * This initializes the vector of the error indicators and locks them.
     * The vector of the error indicators can only get returned if the lock
     * is released (see ReleaseLock()).
     */
    template<typename STH>
    void
    ReInit(STH& sth);

    /**
     * Releases the lock and fills the vector of error indicators
     * accordingly to the previously given EETerms-enum (see dopetypes.h).
     * It basically sums up the primal and dual error indicators using
     * the rule given by EETerms.
     * Only the lock for the given time is released, so it must be called 
     * for all time_dofs to be accessed.
     */
    void
    ReleaseLock()
    {
      locks_[current_time_dof_] = false;

      switch (this->GetEETerms())
        {
        case DOpEtypes::primal_only:
          error_ind_[current_time_dof_] = GetPrimalErrorIndicators();
	  time_step_error_primal_[current_time_dof_]=0.;
	  for ( unsigned int i = 0; i < GetPrimalErrorIndicators().size(); i++)
	  {
	    time_step_error_primal_[current_time_dof_]+=GetPrimalErrorIndicators()(i);
	  }
          break;
        case DOpEtypes::dual_only:
          error_ind_[current_time_dof_] = GetDualErrorIndicators();
	  time_step_error_dual_[current_time_dof_]=0.;
	  for ( unsigned int i = 0; i < GetDualErrorIndicators().size(); i++)
	  {
	    time_step_error_dual_[current_time_dof_]+=GetDualErrorIndicators()(i);
	  }
          break;
        case DOpEtypes::mixed:
          error_ind_[current_time_dof_].equ(0.5, GetPrimalErrorIndicators());
          error_ind_[current_time_dof_].add(0.5, GetDualErrorIndicators());
	  time_step_error_primal_[current_time_dof_]=0.;
	  time_step_error_dual_[current_time_dof_]=0.;
	  assert(GetPrimalErrorIndicators().size() == GetDualErrorIndicators().size());
	  for ( unsigned int i = 0; i < GetPrimalErrorIndicators().size(); i++)
	  {
	    time_step_error_primal_[current_time_dof_]+=GetPrimalErrorIndicators()(i);
	    time_step_error_dual_[current_time_dof_]+=GetDualErrorIndicators()(i);
	  }
          break;
        case DOpEtypes::mixed_control:
          error_ind_[current_time_dof_].equ(0.5, GetPrimalErrorIndicators());
          error_ind_[current_time_dof_].add(0.5, GetDualErrorIndicators());
          error_ind_[current_time_dof_].add(0.5, GetControlErrorIndicators());
	  time_step_error_primal_[current_time_dof_]=0.;
	  time_step_error_dual_[current_time_dof_]=0.;
	  time_step_error_control_[current_time_dof_]=0.;
	  assert(GetPrimalErrorIndicators().size() == GetDualErrorIndicators().size());
	  assert(GetPrimalErrorIndicators().size() == GetControlErrorIndicators().size());
	  for ( unsigned int i = 0; i < GetPrimalErrorIndicators().size(); i++)
	  {
	    time_step_error_primal_[current_time_dof_]+=GetPrimalErrorIndicators()(i);
	    time_step_error_dual_[current_time_dof_]+=GetDualErrorIndicators()(i);
	    time_step_error_control_[current_time_dof_]+=GetControlErrorIndicators()(i);
	  }
          break;
        default:
          throw DOpEException("Unknown DOpEtypes::EEterms.",
                              "DWRDataContainer::ReleaseLock");
          break;
        }
      //Summing selected errors in timestep, and apply the absolute value to indicators
      time_step_error_[current_time_dof_] = 0.;
      for ( unsigned int i = 0; i < error_ind_[current_time_dof_].size(); i++)
      {
	time_step_error_[current_time_dof_]+=error_ind_[current_time_dof_](i);
	error_ind_[current_time_dof_](i) = std::fabs(error_ind_[current_time_dof_](i));
      }
      //Adding to global space-time summs
      switch (this->GetEETerms())
      {
      case DOpEtypes::primal_only:
	primal_error_ += time_step_error_primal_[current_time_dof_];
	break;
      case DOpEtypes::dual_only:
	dual_error_ += time_step_error_dual_[current_time_dof_];
	break;
      case DOpEtypes::mixed:
	primal_error_ += time_step_error_primal_[current_time_dof_];
	dual_error_ += time_step_error_dual_[current_time_dof_];
	break;
      case DOpEtypes::mixed_control:
	primal_error_ += time_step_error_primal_[current_time_dof_];
	dual_error_ += time_step_error_dual_[current_time_dof_];
	control_error_ += time_step_error_control_[current_time_dof_];
	break;
      default:
	throw DOpEException("Unknown DOpEtypes::EEterms.",
			    "DWRDataContainer::ReleaseLock");
	break;
      }
      error_ += time_step_error_[current_time_dof_];
      //Check if global lock can be released
      lock_ = false;
      for ( unsigned int i = 1; i<= n_time_points_; i++)//0 or 1?
      {
	lock_ = lock_||locks_[i];
	 //if(i==n_time_points_) // not nice! but it unlocks the error indicators in the last timestep
	  //{
	  	//lock_=0;
	  //}
      }
    }
    /**
     * Returns the global space-time error (according to the chosen EETerms).
     * ReleaseLock() must have been called.
     *
     * @ return   Error in the previously specified functional.
     */
    double
    GetError() const
    {
      if(lock_)
      {
	throw DOpEException("Error indicators are still locked.",
			    "DWRDataContainer::GetError");
      }
      else
      {
	return error_;
      }
    }

    /**
     * Returns the global space-time primal error indicator.
     * ReleaseLock() must have been called.
     *
     * @ return   Error in the previously specified functional.
     */
    double
    GetPrimalError() const
    {
      assert(this->GetEETerms() != DOpEtypes::dual_only);
      if(lock_)
      {
	throw DOpEException("Error indicators are still locked.",
			    "DWRDataContainer::GetPrimalError");
      }
      else
      {
	return primal_error_;
      }
    }

    /**
     * Returns the global space-time dual error indicator.
     * ReleaseLock() must have been called.
     *
     * @ return   Error in the previously specified functional.
     */
    double
    GetDualError() const
    {
     assert(this->GetEETerms() != DOpEtypes::primal_only);
      if(lock_)
      {
	throw DOpEException("Error indicators are still locked.",
			    "DWRDataContainer::GetDualError");
      }
      else
      {
	return dual_error_;
      }
    }

    /**
     * Returns the global space-time control error indicator.
     * ReleaseLock() must have been called.
     *
     * @ return   Error in the previously specified functional.
     */
    double
    GetControlError() const
    {
      assert(this->GetEETerms() != DOpEtypes::mixed_control);
      if(lock_)
      {
	throw DOpEException("Error indicators are still locked.",
			    "DWRDataContainer::GetControlError");
      }
      else
      {
	return control_error_;
      }
    }
    
    /**
     * Returns the global error (according to the chosen EETerms) at the current time step.
     * ReleaseLock() must have been called.
     *
     * @ return   Error in the previously specified functional.
     */
    double
    GetStepError() const
    {
      if(locks_[current_time_dof_])
      {
	throw DOpEException("Error indicators are still locked.",
			    "DWRDataContainer::GetStepError");
      }
      else
      {
	return time_step_error_[current_time_dof_];
      }
    }

    /**
     * Returns the global primal error indicator at the current time step.
     * ReleaseLock() must have been called.
     *
     * @ return   Error in the previously specified functional.
     */
    double
    GetPrimalStepError() const
    {
      assert(this->GetEETerms() != DOpEtypes::dual_only);
      if(locks_[current_time_dof_])
      {
	throw DOpEException("Error indicators are still locked.",
			    "DWRDataContainer::GetPrimalStepError");
      }
      else
      {
	return time_step_error_primal_[current_time_dof_];
      }
    }

    /**
     * Returns the global dual error indicator at the current time step.
     * ReleaseLock() must have been called.
     *
     * @ return   Error in the previously specified functional.
     */
    double
    GetDualStepError() const
    {
     assert(this->GetEETerms() != DOpEtypes::primal_only);
      if(locks_[current_time_dof_])
      {
	throw DOpEException("Error indicators are still locked.",
			    "DWRDataContainer::GetDualStepError");
      }
      else
      {
	return time_step_error_dual_[current_time_dof_];
      }
    }

    /**
     * Returns the global control error indicator at the current time step.
     * ReleaseLock() must have been called.
     *
     * @ return   Error in the previously specified functional.
     */
    double
    GetControlStepError() const
    {
      assert(this->GetEETerms() != DOpEtypes::mixed_control);
      if(locks_[current_time_dof_])
      {
	throw DOpEException("Error indicators are still locked.",
			    "DWRDataContainer::GetControlStepError");
      }
      else
      {
	return time_step_error_control_[current_time_dof_];
      }
    }

    /**
     * Returns the vector of the error indicators. You have to
     * call ReleaseLock() prior to this function.
     *
     * @return  Vector of raw error indicators (i.e. with sign)
     */
    const std::vector<Vector<double> >&
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
     * Returns the vector of the error indicators. 
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
     * Returns the vector of the dual error indicators. 
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
     * Returns the vector of the control error indicators.
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
     * Returns the vector of the error indicators. 
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

    /*
     * In some cases, the evaluation of an error indicator requires precomputed nodal values,
     * e.g., for the obstacle problem full contact nodes need to be identified.
     * This function returns the number of such vectors needed by the error estimator
     */
    virtual unsigned int NPrecomputedNodalValues()
    {
      return 0;
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
    unsigned int n_time_points_;
    unsigned int current_time_dof_;
    std::map<std::string, const VECTOR *> weight_data_;
    Vector<double> error_ind_primal_, error_ind_dual_, error_ind_control_;
    std::vector<double> time_step_error_, time_step_error_primal_,time_step_error_dual_, time_step_error_control_;
    std::vector<Vector<double> > error_ind_;
    double error_, primal_error_, dual_error_, control_error_;
    std::vector<bool> locks_;
  };

  template<typename VECTOR>
    template<typename STH>
    void
    DWRDataContainerBase<VECTOR>::ReInit(STH& sth)
  {
    n_time_points_ = sth.GetMaxTimePoint();
    error_ind_.resize(n_time_points_+1);
    lock_ = true; 
    locks_.clear();
    locks_.resize(n_time_points_+1,true);
    time_step_error_.resize(n_time_points_+1);
    //Initialize global in space time-step-vectors if needed
    switch (GetEETerms())
    {
    case DOpEtypes::primal_only:
      time_step_error_primal_.resize(n_time_points_+1);
      break;
    case DOpEtypes::dual_only:
      time_step_error_dual_.resize(n_time_points_+1);
      break;
    case DOpEtypes::mixed:
      time_step_error_primal_.resize(n_time_points_+1);
      time_step_error_dual_.resize(n_time_points_+1);
      break;
    case DOpEtypes::mixed_control:
      time_step_error_primal_.resize(n_time_points_+1);
      time_step_error_dual_.resize(n_time_points_+1);
      time_step_error_control_.resize(n_time_points_+1);
      break;
    default:
      throw DOpEException("Unknown DOpEtypes::EETerms!",
			  "DWRDataContainerBase::ReInit");
      break;
    }
    //Resize the time-point vectors (if needed)
    for(unsigned int i = 0; i <= n_time_points_; i++)
    {
#if DEAL_II_VERSION_GTE(8,4,0)
      const unsigned int n_elements =
	sth.GetStateDoFHandler(i).get_triangulation().n_active_cells();
#else
      const unsigned int n_elements =
	sth.GetStateDoFHandler(i).get_tria().n_active_cells();
#endif
      error_ind_[i].reinit(n_elements);
      if ( i == 0 )
      {
	switch (GetEETerms())
	{
	case DOpEtypes::primal_only:
	  GetPrimalErrorIndicators().reinit(n_elements);
	  GetDualErrorIndicators().reinit(n_elements);
	  break;
	case DOpEtypes::dual_only:
	  GetPrimalErrorIndicators().reinit(n_elements);
	  GetDualErrorIndicators().reinit(n_elements);
	  break;
	case DOpEtypes::mixed:
	  GetPrimalErrorIndicators().reinit(n_elements);
	  GetDualErrorIndicators().reinit(n_elements);
	  break;
	case DOpEtypes::mixed_control:
	  GetPrimalErrorIndicators().reinit(n_elements);
	  GetDualErrorIndicators().reinit(n_elements);
	  GetControlErrorIndicators().reinit(n_elements);
	  break;
	default:
	  throw DOpEException("Unknown DOpEtypes::EETerms!",
			      "DWRDataContainerBase::ReInit");
	  break;
	}
      }
    }
    error_=primal_error_=dual_error_=control_error_=0.;
    current_time_dof_=std::numeric_limits<unsigned int>::max();
    SetTime(0);
  }

  template<typename VECTOR>
    void
    DWRDataContainerBase<VECTOR>::SetTime(unsigned int time_dof)
  {
    assert(time_dof <= n_time_points_);
    if( current_time_dof_ != time_dof)
    {
      //FIXME - the time should be used to steer access to vectors!
      current_time_dof_=time_dof;
      if (this->GetEETerms() == DOpEtypes::primal_only
	  || this->GetEETerms() == DOpEtypes::mixed
	  || this->GetEETerms() == DOpEtypes::mixed_control)
      {
	GetPrimalErrorIndicators().reinit(error_ind_[time_dof].size());
	GetPI_h_z().SetTimeDoFNumber(time_dof);
      }
      if (this->GetEETerms() == DOpEtypes::dual_only
	  || this->GetEETerms() == DOpEtypes::mixed
	  || this->GetEETerms() == DOpEtypes::mixed_control)
      {
	GetDualErrorIndicators().reinit(error_ind_[time_dof].size());
	GetPI_h_u().SetTimeDoFNumber(time_dof);
      }
      if ( this->GetEETerms() == DOpEtypes::mixed_control)
      {
	GetControlErrorIndicators().reinit(error_ind_[time_dof].size());
	GetPI_h_q().SetTimeDoFNumber(time_dof);
      }
    }
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
    virtual
    ~DWRDataContainer()
    {
    }

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
