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

  /**
   * This class contains the methods which all primal time stepping schemes share.
   */
  template<typename OPTPROBLEM, typename SPARSITYPATTERN, typename VECTOR,
      int dopedim, int dealdim,
      typename FE = dealii::FESystem<dealdim>,
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
              dealii::FullMatrix<double> &local_entry_matrix, double scale, double scale_ico)
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
              dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/, double /*scale_ico*/)
          {
          }

        template<typename FACEDATACONTAINER>
          void
          Init_InterfaceEquation(const FACEDATACONTAINER& /*fdc*/,
              dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/, double /*scale_ico*/)
          {
          }

        template<typename FACEDATACONTAINER>
          void
          Init_BoundaryEquation(const FACEDATACONTAINER& /*fdc*/,
              dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/, double /*scale_ico*/)
          {
          }

        template<typename FACEDATACONTAINER>
          void
          Init_FaceMatrix(const FACEDATACONTAINER& /*fdc*/,
              FullMatrix<double> &/*local_entry_matrix*/, double /*scale*/, double /*scale_ico*/)
          {
          }

        template<typename FACEDATACONTAINER>
          void
          Init_InterfaceMatrix(const FACEDATACONTAINER& /*fdc*/,
              FullMatrix<double> &/*local_entry_matrix*/, double /*scale*/, double /*scale_ico*/)
          {
          }

        template<typename FACEDATACONTAINER>
          void
          Init_BoundaryMatrix(const FACEDATACONTAINER& /*fdc*/,
              FullMatrix<double> &/*local_cell_matrix*/, double /*scale*/, double /*scale_ico*/)
          {
          }

        /****End the initial values ***************/
    };
}

#endif /* TS_BASE_PRIMAL_H_ */
