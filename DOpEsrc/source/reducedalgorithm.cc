#include "reducedalgorithm.h"
#include "pdeinterface.h"
#include "functionalinterface.h"
#include "dirichletdatainterface.h"
#include "constraintinterface.h"


/******************************************************/
/******************************************************/
/***************************First classic DoFHandler****************/
template class DOpE::ReducedAlgorithm<
  DOpE::OptProblem<
    DOpE::FunctionalInterface<DOpE::CellDataContainer,DOpE::FaceDataContainer,
			      dealii::DoFHandler<deal_II_dimension>,dealii::BlockVector<double>,
			      dope_dimension,deal_II_dimension>,
    DOpE::FunctionalInterface<DOpE::CellDataContainer,DOpE::FaceDataContainer,
			      dealii::DoFHandler<deal_II_dimension>,dealii::BlockVector<double>,
			      dope_dimension,deal_II_dimension>,
    DOpE::PDEInterface<DOpE::CellDataContainer,DOpE::FaceDataContainer,
		       dealii::DoFHandler<deal_II_dimension>, dealii::BlockVector<double>, 
		       dope_dimension,deal_II_dimension>,
    DOpE::DirichletDataInterface<dealii::Vector<double>, dope_dimension,deal_II_dimension>,
    DOpE::ConstraintInterface<DOpE::CellDataContainer,DOpE::FaceDataContainer,
			      dealii::DoFHandler<deal_II_dimension>, dealii::BlockVector<double>, dope_dimension,deal_II_dimension>,
    dealii::BlockSparsityPattern,
    dealii::BlockVector<double>,
    dope_dimension,
    deal_II_dimension>,
  dealii::BlockVector<double>,
  dope_dimension,
  deal_II_dimension>;

  template class DOpE::ReducedAlgorithm<
    DOpE::OptProblem<
      DOpE::FunctionalInterface<DOpE::CellDataContainer,DOpE::FaceDataContainer,
				dealii::DoFHandler<deal_II_dimension>,dealii::Vector<double>,
				dope_dimension,deal_II_dimension>,
      DOpE::FunctionalInterface<DOpE::CellDataContainer,DOpE::FaceDataContainer,
				dealii::DoFHandler<deal_II_dimension>,dealii::Vector<double>,
				dope_dimension,deal_II_dimension>,
      DOpE::PDEInterface<DOpE::CellDataContainer,DOpE::FaceDataContainer,
			 dealii::DoFHandler<deal_II_dimension>,dealii::Vector<double>, 
			 dope_dimension,deal_II_dimension>,
      DOpE::DirichletDataInterface<dealii::Vector<double>, dope_dimension,deal_II_dimension>,
      DOpE::ConstraintInterface<DOpE::CellDataContainer,DOpE::FaceDataContainer,
				dealii::DoFHandler<deal_II_dimension>,dealii::Vector<double>, dope_dimension,deal_II_dimension>,
      dealii::SparsityPattern,
      dealii::Vector<double>,
      dope_dimension,
      deal_II_dimension>,
    dealii::Vector<double>,
    dope_dimension,
    deal_II_dimension>;

  /***************************Then hp DoFHandler****************/
//
//  template class DOpE::ReducedAlgorithm<
//    DOpE::OptProblem<
//      DOpE::FunctionalInterface<dealii::BlockVector<double>,dope_dimension,deal_II_dimension>,
//      DOpE::PDEInterface<dealii::BlockVector<double>, dope_dimension,deal_II_dimension>,
//      DOpE::DirichletDataInterface<dealii::Vector<double>, dope_dimension,deal_II_dimension>,
//      DOpE::ConstraintInterface<dealii::BlockVector<double>, dope_dimension,deal_II_dimension>,
//      dealii::BlockSparsityPattern,
//      dealii::BlockVector<double>,
//      dope_dimension,
//      deal_II_dimension>,
//    dealii::BlockVector<double>,
//    dope_dimension,
//    deal_II_dimension>;
//
//    template class DOpE::ReducedAlgorithm<
//      DOpE::OptProblem<
//        DOpE::FunctionalInterface<dealii::Vector<double>,dope_dimension,deal_II_dimension>,
//        DOpE::PDEInterface<dealii::Vector<double>, dope_dimension,deal_II_dimension>,
//        DOpE::DirichletDataInterface<dealii::Vector<double>, dope_dimension,deal_II_dimension>,
//        DOpE::ConstraintInterface<dealii::Vector<double>, dope_dimension,deal_II_dimension>,
//        dealii::SparsityPattern,
//        dealii::Vector<double>,
//        dope_dimension,
//        deal_II_dimension>,
//      dealii::Vector<double>,
//      dope_dimension,
//      deal_II_dimension>;



