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

#ifndef _DOPE_DATAOUT_H_
#define _DOPE_DATAOUT_H_

#include <numerics/data_out.h>
#include <dofs/dof_handler.h>
#include <hp/dof_handler.h>
#include <multigrid/mg_dof_handler.h>

namespace DOpEWrapper
{
  template <int dim, template<int, int> class DH = dealii::DoFHandler>
    class DataOut : public dealii::DataOut<dim, DH<dim,dim> >
    {
     public:
      DataOut() 
      {
      } 
    };

  //Special treatment of MGDoFHandler...
  template <int dim>
    class DataOut<dim, dealii::MGDoFHandler> : public dealii::DataOut<dim, dealii::DoFHandler<dim,dim> >
    {
     public:
      DataOut() 
      {
      }
    };

}//Endof Namespace DOpEWrapper
#endif
