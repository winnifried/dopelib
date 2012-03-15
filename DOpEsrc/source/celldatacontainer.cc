#include "celldatacontainer.h"

//Explicit instantiation
template class DOpE::CellDataContainer<dealii::DoFHandler<deal_II_dimension>,
    dealii::Vector<double>, deal_II_dimension>;
template class DOpE::CellDataContainer<dealii::DoFHandler<deal_II_dimension>,
    dealii::BlockVector<double>, deal_II_dimension>;
template class DOpE::CellDataContainer<
    dealii::hp::DoFHandler<deal_II_dimension>, dealii::Vector<double>,
    deal_II_dimension>;
template class DOpE::CellDataContainer<
    dealii::hp::DoFHandler<deal_II_dimension>, dealii::BlockVector<double>,
    deal_II_dimension>;
