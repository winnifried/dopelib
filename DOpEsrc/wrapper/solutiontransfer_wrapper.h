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

#ifndef DOPE_SOLUTIONTRANSFER_H_
#define DOPE_SOLUTIONTRANSFER_H_

#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/dofs/dof_handler.h>
#if ! DEAL_II_VERSION_GTE(9,3,0)
#include <deal.II/hp/dof_handler.h>
#endif

namespace DOpEWrapper
{
  /**
   * @class SolutionTransfer
   *
   * This class provides a wrapper for the dealii::SolutionTransfer
   * objects. It is used to cope with the non existing instantiation
   * of the dealii::SolutionTransfer<dim,VECTOR,MGDoFHandler> object.
   *
   * For all values of DH it simply is the corresponding dealii::SolutionTransfer
   * object, except for the MGDoFHandler, when instead the DoFHandler
   * is used.
   *
   * @tparam <dim>              The dimension in which the problem is posed.
   * @tparma <VECTOR>           The vector type used to store the unknowns.
   * @tparam <DH>               The dealii DofHandler type used.
   */

#if DEAL_II_VERSION_GTE(9,3,0)
#if DEAL_II_VERSION_GTE(9,4,0)
  template <int dim, typename VECTOR>
    class SolutionTransfer : public dealii::SolutionTransfer<dim,VECTOR,dim>
#else	//Deal Version in [9.3.0,9.4.0)
  template <int dim, typename VECTOR>
    class SolutionTransfer : public dealii::SolutionTransfer<dim,VECTOR, dealii::DoFHandler<dim,dim> >
#endif
#else  //Deal Version < 9.3.0
  template <int dim, typename VECTOR, template<int, int> class DH = dealii::DoFHandler>
    class SolutionTransfer : public dealii::SolutionTransfer<dim,VECTOR, DH<dim,dim> >
#endif
  {
  public:
#if DEAL_II_VERSION_GTE(9,3,0)
#if DEAL_II_VERSION_GTE(9,4,0)
    SolutionTransfer(const dealii::DoFHandler<dim,dim> &dof) : dealii::SolutionTransfer<dim,VECTOR,dim>(dof)
#else	//Deal Version in [9.3.0,9.4.0)
    SolutionTransfer(const dealii::DoFHandler<dim,dim> &dof) : dealii::SolutionTransfer<dim,VECTOR, dealii::DoFHandler<dim,dim> >(dof)
#endif
#else  //Deal Version < 9.3.0
  SolutionTransfer(const DH<dim,dim> &dof) : dealii::SolutionTransfer<dim,VECTOR, DH<dim,dim> >(dof)
#endif
  {
    }
  };

}//Endof Namespace DOpEWrapper
#endif
