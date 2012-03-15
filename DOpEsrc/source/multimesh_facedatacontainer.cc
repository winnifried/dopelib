#include "multimesh_facedatacontainer.h"

#if deal_II_dimension > 1 //Faces dont make sense in one spacedimension
template class DOpE::Multimesh_FaceDataContainer<dealii::DoFHandler<deal_II_dimension>, dealii::Vector<double>, deal_II_dimension>;
template class DOpE::Multimesh_FaceDataContainer<dealii::DoFHandler<deal_II_dimension>, dealii::BlockVector<double>, deal_II_dimension>;

#endif
