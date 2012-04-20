/*
 * dwrdatacontainer.h
 *
 *  Created on: Mar 22, 2012
 *      Author: cgoll
 */
#ifndef _DWRDATACONTAINER_H_
#define _DWRDATACONTAINER_H_

#include "statevector.h"
#include "dopetypes.h"
#include "parameterreader.h"
#include "dopetypes.h"

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
            : _ee_terms(ee_terms)
        {
          _lock = true;
        }

        virtual
        ~DWRDataContainerBase()
        {
        }
        ;

        /**
         * This initializes the vector of the error indicators and locks them.
         * The vector of the error indicators can only get returned if the lock
         * is released (see ReleaseLock()).
         */
        virtual void
        ReInit(unsigned int n_cells);

        /**
         * Releases the lock and fills the vector of error indicators
         * accordingly to the previously given EETerms-enum (see dopetypes.h).
         * It basically sums up the primal and dual error indicators using
         * the rule given by EETerms.
         */
        void
        ReleaseLock()
        {
          _lock = false;
          switch (this->GetEETerms())
          {
            case DOpEtypes::primal_only:
              _error_ind = GetPrimalErrorIndicators();
              break;
            case DOpEtypes::dual_only:
              _error_ind = GetDualErrorIndicators();
              break;
            case DOpEtypes::mixed:
              _error_ind.equ(0.5, GetPrimalErrorIndicators(), 0.5,
                  GetDualErrorIndicators());
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
            error += GetErrorIndicators()[i];
          }
          return error;
        }

        /**
         * Returns the vector of the error indicators. You have to
         * call ReleaseLock() prior to this function.
         *
         * @return  Vector of raw error indicators (i.e. with sign)
         */
        const Vector<double>&
        GetErrorIndicators() const
        {
          if (_lock)
          {
            throw DOpEException("Error indicators are still locked.",
                "DWRDataContainer::GetErrorIndicators");
          }
          else
          {
            return _error_ind;
          }
        }

        /**
         * Returns the vector of the error indicators. You have to
         * call ReleaseLock() prior to this function.
         *
         * @return  Vector of raw primal error indicators (i.e. with sign)
         */
        Vector<double>&
        GetPrimalErrorIndicators()
        {
          return _error_ind_primal;
        }

        /**
         * Returns the vector of the dual error indicators. You dont have
         * to call ReleaseLock() prior to this function.
         *
         * @return  Vector of raw dual error indicators (i.e. with sign)
         */
        Vector<double>&
        GetDualErrorIndicators()
        {
          return _error_ind_dual;
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
          std::vector<Vector<double>*> res;
          if (_lock)
          {
            throw DOpEException("Error indicators are still locked.",
                "DWRDataContainer::GetErrorIndicators");
          }
          else
          {
            res.push_back(&_error_ind);
            res.push_back(&this->GetPrimalErrorIndicators());
            res.push_back(&this->GetDualErrorIndicators());
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
          return _ee_terms;
        }

        /**
         * TODO We would like DWRDataContainerBase in the solution algorithms,
         * but we need the specialication  in the integrator.ComputeRefinementIndicators.
         * How to achieve this?
         */
        template<class PROBLEM, class INTEGRATOR>
          void
          ComputeRefinementIndicators(PROBLEM& problem, INTEGRATOR& integrator)
          {
            integrator.ComputeRefinementIndicators(problem, *this);
          }

//        std::string
//        GetProblemType() const
//        {
//          std::string ret = DOpEtypes::GetProblemType(GetResidualEvaluation())
//              + DOpEtypes::GetProblemType(GetWeightComputation())
//              + DOpEtypes::GetProblemType(GetEETerms());
//          return ret;
//        }

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
        const std::map<std::string, const VECTOR*>&
        GetWeightData() const
        {
          return _weight_data;
        }


        /**
         * Deletes the weights.
         */
        void
        ClearWeightData()
        {
          _weight_data.clear();
        }


        /**
         * Computes the functions that compute the weights and puts them
         * into _weight_data.
         *
         * @param u   The FE-vector of the primal solution.
         * @param z   The FE-vector of the dual solution.
         */
        void
        PrepareWeights(StateVector<VECTOR>& u, StateVector<VECTOR>& z)
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
            default:
              throw DOpEException("Unknown DOpEtypes::EETerms!",
                  "StatPDEProblem::PrepareWeights");
              break;
          }
        }
      protected:
        //TODO PI_h_u and PI_h_z are probably not the best names.
        //Basically, one could replace them with 'primalweight' and 'dualweight',
        //but i found them a little bit misleading. ~cg
        virtual StateVector<VECTOR>&
        GetPI_h_u() = 0;

        virtual StateVector<VECTOR>&
        GetPI_h_z() = 0;

        virtual void
        PreparePI_h_u(const StateVector<VECTOR>& u) = 0;

        virtual void
        PreparePI_h_z(const StateVector<VECTOR>& z) = 0;

        void
        AddWeightData(std::string name, const VECTOR* new_data)
        {
          if (_weight_data.find(name) != _weight_data.end())
          {
            throw DOpEException(
                "Adding multiple Data with name " + name + " is prohibited!",
                "Integrator::AddDomainData");
          }
          _weight_data.insert(
              std::pair<std::string, const VECTOR*>(name, new_data));
        }

      private:
        DOpEtypes::EETerms _ee_terms;
        bool _lock;
        std::map<std::string, const VECTOR*> _weight_data;
        Vector<double> _error_ind, _error_ind_primal, _error_ind_dual;
    };

  template<typename VECTOR>
    void
    DWRDataContainerBase<VECTOR>::ReInit(unsigned int n_cells)
    {
      _error_ind.reinit(n_cells);
      GetPrimalErrorIndicators().reinit(n_cells);
      GetDualErrorIndicators().reinit(n_cells);
      _lock = true;
    }

  /**
   * Adds just the pure virtual functions GetCellWeight() and GetFaceWeight().
   * They have to get implemented in derived classes. These two methods are
   * excluded from DWRDataContainerBase() to save two template parameters.
   */
  template<class CDC, class FDC, typename VECTOR>
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
         * Returns a CellDataContainer for the weights on the
         * cells. Pure virtual.
         */
        virtual CDC&
        GetCellWeight() const = 0;

        /**
         * Returns a FaceDataContainer for the weight on the
         * faces (and boundaries.). Pure virtual.
         */
        virtual FDC&
        GetFaceWeight() const = 0;
    };

} //end of namespace
#endif /* DWRDATACONTAINER_H_ */
