/*
 * ts_base_primal.h
 *
 *  Created on: 04.05.2012
 *      Author: cgoll
 */

#ifndef _PRIMAL_TS_BASE_H_
#define _PRIMAL_TS_BASE_H_

#include "ts_base.h"
namespace DOpE
{
  template<typename OPTPROBLEM, typename SPARSITYPATTERN, typename VECTOR,
      int dopedim, int dealdim,
      typename FE = DOpEWrapper::FiniteElement<dealdim>,
      typename DOFHANDLER = dealii::DoFHandler<dealdim> >
    class PrimalTSBase : public TSBase<OPTPROBLEM, SPARSITYPATTERN, VECTOR,
        dopedim, dealdim, FE, DOFHANDLER>
    {
      public:
        PrimalTSBase(OPTPROBLEM& OP) :
            TSBase<OPTPROBLEM, SPARSITYPATTERN, VECTOR, dopedim, dealdim, FE,
                DOFHANDLER>(OP)
        {
        }

        ~PrimalTSBase()
        {
        }

        /******************************************************/
        /****For the initial values ***************/
        template<typename DATACONTAINER>
          void
          Init_CellEquation(const DATACONTAINER& cdc,
              dealii::Vector<double> &local_cell_vector, double scale,
              double scale_ico)
          {
            this->GetProblem().Init_CellEquation(cdc, local_cell_vector, scale,
                scale_ico);
          }

        template<typename DATACONTAINER>
          void
          Init_CellRhs(const DATACONTAINER& cdc,
              dealii::Vector<double> &local_cell_vector, double scale)
          {
            this->GetProblem().Init_CellRhs(cdc, local_cell_vector, scale);
          }

        template<typename DATACONTAINER>
          void
          Init_CellMatrix(const DATACONTAINER& cdc,
              dealii::FullMatrix<double> &local_entry_matrix, double scale,
              double scale_ico)
          {
            this->GetProblem().Init_CellMatrix(cdc, local_entry_matrix, scale,
                scale_ico);
          }

        void
        Init_PointRhs(
            const std::map<std::string, const dealii::Vector<double>*> &/*param_values*/,
            const std::map<std::string, const VECTOR*> &/*domain_values*/,
            VECTOR& /*rhs_vector*/, double /*scale=1.*/)
        {
        }

        template<typename FACEDATACONTAINER>
          void
          Init_FaceEquation(const FACEDATACONTAINER& /*fdc*/,
              dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/)
          {
          }

        template<typename FACEDATACONTAINER>
          void
          Init_InterfaceEquation(const FACEDATACONTAINER& /*fdc*/,
              dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/)
          {
          }

        template<typename FACEDATACONTAINER>
          void
          Init_BoundaryEquation(const FACEDATACONTAINER& /*fdc*/,
              dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/)
          {
          }

        template<typename FACEDATACONTAINER>
          void
          Init_FaceMatrix(const FACEDATACONTAINER& /*fdc*/,
              FullMatrix<double> &/*local_entry_matrix*/)
          {
          }

        template<typename FACEDATACONTAINER>
          void
          Init_InterfaceMatrix(const FACEDATACONTAINER& /*fdc*/,
              FullMatrix<double> &/*local_entry_matrix*/)
          {
          }

        template<typename FACEDATACONTAINER>
          void
          Init_BoundaryMatrix(const FACEDATACONTAINER& /*fdc*/,
              FullMatrix<double> &/*local_cell_matrix*/)
          {
          }

        /****End the initial values ***************/
    };
}

#endif /* TS_BASE_PRIMAL_H_ */
