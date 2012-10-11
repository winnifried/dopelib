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
