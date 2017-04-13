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

#ifndef INSTAT_PDE_PROBLEM_CONTAINER_
#define INSTAT_PDE_PROBLEM_CONTAINER_

#include <container/pdeproblemcontainer.h>

namespace DOpE
{
  /**
   * Container class for all nonstationary pde problems.
   * This class collects all problem depended data needed to
   * calculate the solution to the optimization problem.
   *
   *
   * @tparam PRIMALTSPROBLEM        The description of the time discretization scheme for the PDE
   *                                (and tangent PDE).
   * @tparam ADJOINTTSPROBLEM       The description of the time discretization scheme for the adjoint
   *                                PDE (and all auxilliary adjoint problems).
   * @tparam PDE                    The description of the PDE, see PDEInterface for details.
   * @tparam DD                     The description of the Dirichlet data, see
   *                                DirichletDataInterface for details.
   * @tparam SPARSITYPATTERN        The sparsity pattern to be used in the stiffness matrix.
   * @tparam VECTOR                 The vector type in which the coordinate vector of the
   *                                solution is to be stored.
   * @tparam dealdim                The dimension of the domain in which the PDE is considered.
   * @tparam FE                     The finite element under consideration.
   * @tparam DH                     The spatial DoFHandler to be used when evaluating the
   *                                weak form.
   */
  template<template<typename BASE_PROB, typename SPARSITYPATTERN, typename VECTOR, int dealdim, template<int, int> class FE, template<int, int> class DH> class PRIMALTSPROBLEM,
           template<typename BASE_PROB, typename SPARSITYPATTERN, typename VECTOR, int dealdim, template<int, int> class FE, template<int, int> class DH> class  ADJOINTTSPROBLEM,
           typename PDE,
           typename DD, typename SPARSITYPATTERN,
           typename VECTOR, int dealdim,
           template<int, int> class FE = FESystem,
           template<int, int> class DH = dealii::DoFHandler>
  class InstatPDEProblemContainer : public PDEProblemContainer<PDE,DD,
    SPARSITYPATTERN, VECTOR,dealdim,FE, DH>
  {
  public:
    InstatPDEProblemContainer(PDE &pde,
                              StateSpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR,dealdim> &STH)
      : PDEProblemContainer<PDE,DD,SPARSITYPATTERN,VECTOR,dealdim,FE, DH>(
        pde,STH), ts_state_problem_(NULL)
    {
    }

    ~InstatPDEProblemContainer()
    {
      if (ts_state_problem_ != NULL)
        {
          delete ts_state_problem_;
        }
    }

    void ReInit(std::string algo_type)
    {
      if (ts_state_problem_ != NULL)
        {
          delete ts_state_problem_;
          ts_state_problem_ = NULL;
        }

      PDEProblemContainer<PDE,DD,SPARSITYPATTERN,VECTOR,dealdim,FE, DH>::ReInit(algo_type);

    }

    std::string GetName() const
    {
      return "InstatPDEProblemContainer";
    }

    /*******************************************************************************************/
    /**
     * Returns a description of the PDE
     */
    PRIMALTSPROBLEM<StateProblem<
    PDEProblemContainer<PDE,DD,SPARSITYPATTERN,VECTOR,dealdim,FE, DH>,
                        PDE, DD, SPARSITYPATTERN, VECTOR, dealdim>,
                        SPARSITYPATTERN, VECTOR, dealdim, FE, DH>& GetStateProblem()
    {
      if (ts_state_problem_ == NULL)
        {
          ts_state_problem_ = new PRIMALTSPROBLEM<StateProblem<
          PDEProblemContainer<PDE,DD,SPARSITYPATTERN,VECTOR,dealdim,FE, DH>,
          PDE, DD, SPARSITYPATTERN, VECTOR, dealdim>,
          SPARSITYPATTERN, VECTOR, dealdim, FE, DH>(PDEProblemContainer<PDE,DD,
                                                    SPARSITYPATTERN,VECTOR,dealdim,FE, DH>::GetStateProblem());
        }
      return *ts_state_problem_;
    }

  private:
    PRIMALTSPROBLEM<StateProblem<
    PDEProblemContainer<PDE,DD,SPARSITYPATTERN,VECTOR,dealdim,FE, DH>,
    PDE, DD, SPARSITYPATTERN, VECTOR, dealdim>,
    SPARSITYPATTERN, VECTOR, dealdim, FE, DH>* ts_state_problem_;

  };
}
#endif
