#include "pdeproblemcontainer.h"
#include "pdeinterface.h"
#include "functionalinterface.h"
#include "dirichletdatainterface.h"
#include "dopeexception.h"
#include "primaldirichletdata.h"
#include "tangentdirichletdata.h"
#include "constraintinterface.h"

#if dope_dimension == deal_II_dimension

template class DOpE::PDEProblemContainer<DOpE::PDEInterface<DOpE::CellDataContainer,DOpE::FaceDataContainer,
							    dealii::DoFHandler<deal_II_dimension>, 
							    dealii::BlockVector<double>, deal_II_dimension>, 
					 DOpE::DirichletDataInterface<
					   dealii::BlockVector<double>, deal_II_dimension>,
					 dealii::BlockSparsityPattern, dealii::BlockVector<double>, 
					 deal_II_dimension>;

template class DOpE::PDEProblemContainer<
  DOpE::PDEInterface<DOpE::CellDataContainer,DOpE::FaceDataContainer,
		     dealii::DoFHandler<deal_II_dimension>, dealii::Vector<double>, deal_II_dimension>, 
  DOpE::DirichletDataInterface<dealii::Vector<double>, deal_II_dimension>,
  dealii::SparsityPattern, dealii::Vector<double>, deal_II_dimension>;

template class DOpE::PDEProblemContainer<DOpE::PDEInterface<DOpE::CellDataContainer,DOpE::FaceDataContainer,
							    dealii::hp::DoFHandler<deal_II_dimension>,
							    dealii::BlockVector<double>, deal_II_dimension>, 
					 DOpE::DirichletDataInterface<
					   dealii::BlockVector<double>, deal_II_dimension>,
					 dealii::BlockSparsityPattern, dealii::BlockVector<double>, 
					 deal_II_dimension,
					 DOpEWrapper::FECollection<deal_II_dimension>, 
					 dealii::hp::DoFHandler<deal_II_dimension> >;

template class DOpE::PDEProblemContainer<
  DOpE::PDEInterface<DOpE::CellDataContainer,DOpE::FaceDataContainer,
		     dealii::hp::DoFHandler<deal_II_dimension>, dealii::Vector<double>, deal_II_dimension>,
  DOpE::DirichletDataInterface<dealii::Vector<double>, deal_II_dimension>,
  dealii::SparsityPattern, dealii::Vector<double>, deal_II_dimension,
  DOpEWrapper::FECollection<deal_II_dimension>, dealii::hp::DoFHandler<deal_II_dimension> >;

#endif
