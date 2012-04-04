/*
 * celldatacontainer_internal.cc
 *
 *  Created on: Apr 3, 2012
 *      Author: cgoll
 */

#include "celldatacontainer_internal.h"

namespace DOpE
{
  namespace cdcinternal
  {
    //explicit instantiations
    template class CellDataContainerInternal<dealii::Vector<double>,
        deal_II_dimension> ;
    template class CellDataContainerInternal<dealii::BlockVector<double>,
        deal_II_dimension> ;
    template class CellDataContainerInternal<dealii::Vector<float>,
        deal_II_dimension> ;
    template class CellDataContainerInternal<dealii::BlockVector<float>,
        deal_II_dimension> ;

  }
}

