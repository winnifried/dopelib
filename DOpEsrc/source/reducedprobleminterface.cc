#include "reducedprobleminterface.h"
#include "optproblem.h"
#include "pdeinterface.h"
#include "functionalinterface.h"
#include "dirichletdatainterface.h"
#include "constraintinterface.h"

/******************************************************/
/******************************************************/
template class DOpE::ReducedProblemInterface<
  DOpE::OptProblem<
    DOpE::FunctionalInterface<DOpE::CellDataContainer,DOpE::FaceDataContainer,
			      dealii::DoFHandler<deal_II_dimension>,dealii::BlockVector<double>,
			      dope_dimension,deal_II_dimension>,
    DOpE::FunctionalInterface<DOpE::CellDataContainer,DOpE::FaceDataContainer,
			      dealii::DoFHandler<deal_II_dimension>,dealii::BlockVector<double>,
			      dope_dimension,deal_II_dimension>,
    DOpE::PDEInterface<DOpE::CellDataContainer,DOpE::FaceDataContainer,
		       dealii::DoFHandler<deal_II_dimension>,dealii::BlockVector<double>,
		       dope_dimension,deal_II_dimension>,
    DOpE::DirichletDataInterface<dealii::BlockVector<double>,dope_dimension,deal_II_dimension>,
    DOpE::ConstraintInterface<DOpE::CellDataContainer,DOpE::FaceDataContainer,
			      dealii::DoFHandler<deal_II_dimension>,dealii::BlockVector<double>,dope_dimension,deal_II_dimension>,
    dealii::BlockSparsityPattern,
    dealii::BlockVector<double>,
    dope_dimension,deal_II_dimension>,
  dealii::BlockVector<double>,
  dope_dimension,deal_II_dimension>;

////////////////////////////////////////////hp//////////////////////////////////////////////
template class DOpE::ReducedProblemInterface<
  DOpE::OptProblem<
    DOpE::FunctionalInterface<DOpE::CellDataContainer,DOpE::FaceDataContainer,
			      dealii::hp::DoFHandler<deal_II_dimension>,dealii::BlockVector<double>,
			      dope_dimension,deal_II_dimension>,
    DOpE::FunctionalInterface<DOpE::CellDataContainer,DOpE::FaceDataContainer,
			      dealii::hp::DoFHandler<deal_II_dimension>,dealii::BlockVector<double>,
			      dope_dimension,deal_II_dimension>,
    DOpE::PDEInterface<DOpE::CellDataContainer,DOpE::FaceDataContainer,
		       dealii::hp::DoFHandler<deal_II_dimension>,dealii::BlockVector<double>,
		       dope_dimension,deal_II_dimension>,
    DOpE::DirichletDataInterface<dealii::BlockVector<double>,dope_dimension,deal_II_dimension>,
    DOpE::ConstraintInterface<DOpE::CellDataContainer,DOpE::FaceDataContainer,
			      dealii::hp::DoFHandler<deal_II_dimension>,dealii::BlockVector<double>,dope_dimension,deal_II_dimension>,
    dealii::BlockSparsityPattern,
    dealii::BlockVector<double>,
    dope_dimension,deal_II_dimension, DOpEWrapper::FECollection<deal_II_dimension>, dealii::hp::DoFHandler<deal_II_dimension> >,
  dealii::BlockVector<double>,
  dope_dimension,deal_II_dimension>;
