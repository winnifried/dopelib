#include "multimesh_celldatacontainer.h"

//Explicit instantiation
template class DOpE::Multimesh_CellDataContainer<dealii::DoFHandler<deal_II_dimension>,
    dealii::Vector<double>, deal_II_dimension>;
template class DOpE::Multimesh_CellDataContainer<dealii::DoFHandler<deal_II_dimension>,
    dealii::BlockVector<double>, deal_II_dimension>;

