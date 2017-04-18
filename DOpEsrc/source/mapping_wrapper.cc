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


#include<wrapper/mapping_wrapper.h>
#include <deal.II/dofs/dof_handler.h>
//#include <deal.II/multigrid/mg_dof_handler.h>
#include <deal.II/hp/dof_handler.h>

namespace DOpEWrapper
{

  template<int dim>
  Mapping<dim, dealii::DoFHandler >
  StaticMappingQ1<dim, dealii::DoFHandler>::mapping_q1(1);


  template<int dim>
  Mapping<dim, dealii::hp::DoFHandler>
  StaticMappingQ1<dim, dealii::hp::DoFHandler>::mapping_q1(
    StaticMappingQ1<dim, dealii::DoFHandler >::mapping_q1);

//  template<int dim>
//    Mapping<dim, dealii::MGDoFHandler>
//    StaticMappingQ1<dim, dealii::MGDoFHandler >::mapping_q1(
//        StaticMappingQ1<dim, dealii::DoFHandler >::mapping_q1);

}

template class DOpEWrapper::Mapping<1, dealii::DoFHandler>;
template class DOpEWrapper::Mapping<2, dealii::DoFHandler>;
template class DOpEWrapper::Mapping<3, dealii::DoFHandler>;

//template class DOpEWrapper::Mapping<1, dealii::MGDoFHandler>;
//template class DOpEWrapper::Mapping<2, dealii::MGDoFHandler>;
//template class DOpEWrapper::Mapping<3, dealii::MGDoFHandler>;

template class DOpEWrapper::Mapping<1, dealii::hp::DoFHandler>;
template class DOpEWrapper::Mapping<2, dealii::hp::DoFHandler>;
template class DOpEWrapper::Mapping<3, dealii::hp::DoFHandler>;

template struct DOpEWrapper::StaticMappingQ1<1, dealii::DoFHandler>;
template struct DOpEWrapper::StaticMappingQ1<2, dealii::DoFHandler>;
template struct DOpEWrapper::StaticMappingQ1<3, dealii::DoFHandler>;

//template struct DOpEWrapper::StaticMappingQ1<1, dealii::MGDoFHandler>;
//template struct DOpEWrapper::StaticMappingQ1<2, dealii::MGDoFHandler>;
//template struct DOpEWrapper::StaticMappingQ1<3, dealii::MGDoFHandler>;

template struct DOpEWrapper::StaticMappingQ1<1, dealii::hp::DoFHandler>;
template struct DOpEWrapper::StaticMappingQ1<2, dealii::hp::DoFHandler>;
template struct DOpEWrapper::StaticMappingQ1<3, dealii::hp::DoFHandler>;
