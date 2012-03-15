#include "statpdeproblem.h"
#include "preconditioner_wrapper.h"
#include "integratordatacontainer.h"


#if dope_dimension == deal_II_dimension

/******************************************************/
#define _IDC     DOpE::IntegratorDataContainer<dealii::DoFHandler<deal_II_dimension>, dealii::Quadrature<deal_II_dimension>, dealii::Quadrature<deal_II_dimension-1>,dealii::BlockVector<double>, deal_II_dimension  >
#define _DOP     DOpE::PDEProblemContainer<DOpE::PDEInterface<DOpE::CellDataContainer,DOpE::FaceDataContainer,dealii::DoFHandler<deal_II_dimension>,dealii::BlockVector<double>,deal_II_dimension>,DOpE::DirichletDataInterface<dealii::BlockVector<double>,deal_II_dimension>,dealii::BlockSparsityPattern,dealii::BlockVector<double>,deal_II_dimension>
#define _DIN     DOpE::Integrator<_IDC,dealii::BlockVector<double>,double,deal_II_dimension>
#define _DCGS    DOpE::CGLinearSolverWithMatrix<DOpEWrapper::PreconditionIdentity_Wrapper<dealii::BlockSparseMatrix<double> >, dealii::BlockSparsityPattern,dealii::BlockSparseMatrix<double>,dealii::BlockVector<double>,deal_II_dimension>
#define _DGMRESS DOpE::GMRESLinearSolverWithMatrix<DOpEWrapper::PreconditionIdentity_Wrapper<dealii::BlockSparseMatrix<double> >,dealii::BlockSparsityPattern,dealii::BlockSparseMatrix<double>,dealii::BlockVector<double>,deal_II_dimension>
#define _DDS     DOpE::DirectLinearSolverWithMatrix<dealii::BlockSparsityPattern,dealii::BlockSparseMatrix<double>,dealii::BlockVector<double>,deal_II_dimension>

/******************************************************/
template class DOpE::StatPDEProblem<
  DOpE::NewtonSolver<_DIN,_DCGS,dealii::BlockVector<double>,deal_II_dimension>,
  _DIN,
  _DOP,dealii::BlockVector<double>,deal_II_dimension>;

/******************************************************/
template class DOpE::StatPDEProblem<
  DOpE::NewtonSolver<_DIN,_DGMRESS,dealii::BlockVector<double>,deal_II_dimension>,
  _DIN,
  _DOP,dealii::BlockVector<double>,deal_II_dimension>;

/******************************************************/
template class DOpE::StatPDEProblem<
 DOpE::NewtonSolver<_DIN,_DDS,dealii::BlockVector<double>,deal_II_dimension>,
  _DIN,
  _DOP,dealii::BlockVector<double>,deal_II_dimension>;

/******************************************************/
#undef _IDC
#undef _DOP
#undef _DIN
#undef _DCGS
#undef _DGMRESS
#undef _DDS
//////////////////////////////////////hp/////////////////////////////////////

/******************************************************/
#define _IDC     DOpE::IntegratorDataContainer<dealii::hp::DoFHandler<deal_II_dimension>, dealii::hp::QCollection<deal_II_dimension>, dealii::hp::QCollection<deal_II_dimension-1>,dealii::BlockVector<double>, deal_II_dimension  >
#define _DOP     DOpE::PDEProblemContainer<DOpE::PDEInterface<DOpE::CellDataContainer,DOpE::FaceDataContainer,dealii::hp::DoFHandler<deal_II_dimension>,dealii::BlockVector<double>,deal_II_dimension>,DOpE::DirichletDataInterface<dealii::BlockVector<double>,deal_II_dimension>,dealii::BlockSparsityPattern,dealii::BlockVector<double>,deal_II_dimension,DOpEWrapper::FECollection<deal_II_dimension>, dealii::hp::DoFHandler<deal_II_dimension>>
#define _DIN     DOpE::Integrator<_IDC,dealii::BlockVector<double>,double,deal_II_dimension>
#define _DCGS    DOpE::CGLinearSolverWithMatrix<DOpEWrapper::PreconditionIdentity_Wrapper<dealii::BlockSparseMatrix<double> >, dealii::BlockSparsityPattern,dealii::BlockSparseMatrix<double>,dealii::BlockVector<double>,deal_II_dimension>
#define _DGMRESS DOpE::GMRESLinearSolverWithMatrix<DOpEWrapper::PreconditionIdentity_Wrapper<dealii::BlockSparseMatrix<double> >,dealii::BlockSparsityPattern,dealii::BlockSparseMatrix<double>,dealii::BlockVector<double>,deal_II_dimension>
#define _DDS     DOpE::DirectLinearSolverWithMatrix<dealii::BlockSparsityPattern,dealii::BlockSparseMatrix<double>,dealii::BlockVector<double>,deal_II_dimension>

/******************************************************/
template class DOpE::StatPDEProblem<
  DOpE::NewtonSolver<_DIN,_DCGS,dealii::BlockVector<double>,deal_II_dimension>,
  _DIN,
  _DOP,dealii::BlockVector<double>,deal_II_dimension>;

/******************************************************/
template class DOpE::StatPDEProblem<
  DOpE::NewtonSolver<_DIN,_DGMRESS,dealii::BlockVector<double>,deal_II_dimension>,
  _DIN,
  _DOP,dealii::BlockVector<double>,deal_II_dimension>;

/******************************************************/
template class DOpE::StatPDEProblem<
 DOpE::NewtonSolver<_DIN,_DDS,dealii::BlockVector<double>,deal_II_dimension>,
  _DIN,
  _DOP,dealii::BlockVector<double>,deal_II_dimension>;

/******************************************************/
#undef _IDC
#undef _DOP
#undef _DIN
#undef _DCGS
#undef _DGMRESS
#undef _DDS

#endif
