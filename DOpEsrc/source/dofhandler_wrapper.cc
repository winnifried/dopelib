#include "dofhandler_wrapper.h"


#if dope_dimension == deal_II_dimension
template class DOpEWrapper::DoFHandler<deal_II_dimension>;
template class DOpEWrapper::DoFHandler<deal_II_dimension, dealii::hp::DoFHandler<deal_II_dimension> >;
#elif deal_II_dimension != dope_dimension
template class DOpEWrapper::DoFHandler<deal_II_dimension>;
template class DOpEWrapper::DoFHandler<dope_dimension, dealii::DoFHandler<deal_II_dimension> >;
template class DOpEWrapper::DoFHandler<deal_II_dimension, dealii::hp::DoFHandler<deal_II_dimension> >;
template class DOpEWrapper::DoFHandler<dope_dimension, dealii::hp::DoFHandler<deal_II_dimension> >;
#endif

