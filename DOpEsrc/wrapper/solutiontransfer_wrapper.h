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

#ifndef DOPE_SOLUTIONTRANSFER_H_
#define DOPE_SOLUTIONTRANSFER_H_

#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/hp/dof_handler.h>
//#include <deal.II/multigrid/mg_dof_handler.h>

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

  template <int dim, typename VECTOR, template<int, int> class DH = dealii::DoFHandler>
  class SolutionTransfer : public dealii::SolutionTransfer<dim,VECTOR, DH<dim,dim> >
  {
  public:
    SolutionTransfer(const DH<dim,dim> &dof) : dealii::SolutionTransfer<dim,VECTOR, DH<dim,dim> >(dof)
    {
    }
  };

// //Special treatment of MGDoFHandler...
// template <int dim, typename VECTOR>
//   class SolutionTransfer<dim,VECTOR,MGDoFHandler> : public dealii::SolutionTransfer<dim,VECTOR, dealii::DoFHandler<dim,dim> >
//   {
//    public:
//     SolutionTransfer(const MGDoFHandler<dim,dim> &dof) :
//          dealii::SolutionTransfer<dim,VECTOR, dealii::DoFHandler<dim,dim> >(dof)
//    {
//    }
//   };

}//Endof Namespace DOpEWrapper
#endif
