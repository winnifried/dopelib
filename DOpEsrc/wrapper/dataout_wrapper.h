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

#ifndef DOPE_DATAOUT_H_
#define DOPE_DATAOUT_H_

#include <deal.II/numerics/data_out.h>
#include <deal.II/dofs/dof_handler.h>
#if ! DEAL_II_VERSION_GTE(9,3,0)
#include <deal.II/hp/dof_handler.h>
#endif

namespace DOpEWrapper
{
#if DEAL_II_VERSION_GTE(9,3,0)
  /**
   * @class DataOut
   *
   * This class provides a wrapper for the dealii::DataOut
   * objects. It is used to cope with the non existing instantiation
   * of the dealii::DataOut<dim,MGDoFHandler> object.
   *
   * For all values of DH it simply is the corresponding dealii::DataOut
   * object, except for the MGDoFHandler, when instead the DoFHandler
   * is used.
   *
   * @tparam <dim>              The dimension in which the problem is posed.
   */
#else
  /**
   * @class DataOut
   *
   * This class provides a wrapper for the dealii::DataOut
   * objects. It is used to cope with the non existing instantiation
   * of the dealii::DataOut<dim,MGDoFHandler> object.
   *
   * For all values of DH it simply is the corresponding dealii::DataOut
   * object, except for the MGDoFHandler, when instead the DoFHandler
   * is used.
   *
   * @tparam <dim>              The dimension in which the problem is posed.
   * @tparam <DH>               The dealii DofHandler type used.
   */
#endif
#if DEAL_II_VERSION_GTE(9,3,0)
#if DEAL_II_VERSION_GTE(9,3,1)
  template <int dim>
  class DataOut : public dealii::DataOut<dim>
#else
  template <int dim>
  class DataOut : public dealii::DataOut<dim, dealii::DoFHandler<dim, dim> >
#endif
#else
  template <int dim, template <int, int> class DH = dealii::DoFHandler>
  class DataOut : public dealii::DataOut<dim, DH<dim, dim> >
#endif
  {
  public:
    DataOut ()
    {
    }
  };

}//Endof Namespace DOpEWrapper
#endif
