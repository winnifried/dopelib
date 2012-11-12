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
 * mapping_wrapper.cc
 *
 *  Created on: Oct 8, 2012
 *      Author: cgoll
 */

#include"mapping_wrapper.h"
#include <dofs/dof_handler.h>
#include <hp/dof_handler.h>

namespace DOpEWrapper
{

  template<int dim>
    Mapping<dim, dealii::DoFHandler<dim> >
    StaticMappingQ1<dim, dealii::DoFHandler<dim> >::mapping_q1(1);


  template<int dim>
    Mapping<dim, dealii::hp::DoFHandler<dim> >
    StaticMappingQ1<dim, dealii::hp::DoFHandler<dim> >::mapping_q1(
        StaticMappingQ1<dim, dealii::DoFHandler<dim> >::mapping_q1);

}

template class DOpEWrapper::Mapping<1, dealii::DoFHandler<1> >;
template class DOpEWrapper::Mapping<2, dealii::DoFHandler<2> >;
template class DOpEWrapper::Mapping<3, dealii::DoFHandler<3> >;

template class DOpEWrapper::Mapping<1, dealii::hp::DoFHandler<1> >;
template class DOpEWrapper::Mapping<2, dealii::hp::DoFHandler<2> >;
template class DOpEWrapper::Mapping<3, dealii::hp::DoFHandler<3> >;

template class DOpEWrapper::StaticMappingQ1<1, dealii::DoFHandler<1> >;
template class DOpEWrapper::StaticMappingQ1<2, dealii::DoFHandler<2> >;
template class DOpEWrapper::StaticMappingQ1<3, dealii::DoFHandler<3> >;

template class DOpEWrapper::StaticMappingQ1<1, dealii::hp::DoFHandler<1> >;
template class DOpEWrapper::StaticMappingQ1<2, dealii::hp::DoFHandler<2> >;
template class DOpEWrapper::StaticMappingQ1<3, dealii::hp::DoFHandler<3> >;
