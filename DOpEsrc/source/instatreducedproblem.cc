#include "instatreducedproblem.h"
#include "optproblem.h"
#include "instatoptproblemcontainer.h"
#include "forward_euler_problem.h"
#include "backward_euler_problem.h"
#include "crank_nicolson_problem.h"
#include "shifted_crank_nicolson_problem.h"
#include "fractional_step_theta_problem.h"
#include "preconditioner_wrapper.h"
#include "instat_step_newtonsolver.h"
#include "fractional_step_theta_step_newtonsolver.h"
#include "directlinearsolver.h"
#include "cglinearsolver.h"
#include "gmreslinearsolver.h"
#include "integrator.h"
#include "integratordatacontainer.h"

/******************************************************/
#if dope_dimension == deal_II_dimension
//*****************************************************/
//Everything with 'normal' DoFHandler
//*****************************************************/

/******************************************************/
/******************************************************/
//First the ones using BlockVector
/******************************************************/
#define VECTOR dealii::BlockVector<double>
#define SPARSITYPATTERN dealii::BlockSparsityPattern
#define MATRIX dealii::BlockSparseMatrix<double>
#define DOFHANDLER dealii::DoFHandler<dope_dimension>
#define IDC     DOpE::IntegratorDataContainer<DOFHANDLER, dealii::Quadrature<deal_II_dimension>, dealii::Quadrature<deal_II_dimension-1>, VECTOR, deal_II_dimension  >
#define FE      DOpEWrapper::FiniteElement<deal_II_dimension>
#define FUNC    DOpE::FunctionalInterface<DOpE::CellDataContainer,DOpE::FaceDataContainer,DOFHANDLER,VECTOR,dope_dimension,deal_II_dimension>
#define PDE     DOpE::PDEInterface<DOpE::CellDataContainer,DOpE::FaceDataContainer,DOFHANDLER,VECTOR,dope_dimension,deal_II_dimension>
#define DD      DOpE::DirichletDataInterface<VECTOR,dope_dimension,deal_II_dimension>
#define CONS    DOpE::ConstraintInterface<DOpE::CellDataContainer,DOpE::FaceDataContainer,DOFHANDLER,VECTOR,dope_dimension,deal_II_dimension>

#define DOP     DOpE::OptProblem<FUNC,FUNC,PDE,DD,CONS,SPARSITYPATTERN,VECTOR,dope_dimension,deal_II_dimension, FE, DOFHANDLER>
#define PROB    DOpE::StateProblem<DOP,PDE,DD,SPARSITYPATTERN,VECTOR,dope_dimension,deal_II_dimension>
#define DCGS    DOpE::CGLinearSolverWithMatrix<DOpEWrapper::PreconditionIdentity_Wrapper<MATRIX>, SPARSITYPATTERN,MATRIX,VECTOR,deal_II_dimension>
#define DGMRESS DOpE::GMRESLinearSolverWithMatrix<DOpEWrapper::PreconditionIdentity_Wrapper<MATRIX>, SPARSITYPATTERN,MATRIX,VECTOR,deal_II_dimension>
#define DDS     DOpE::DirectLinearSolverWithMatrix<SPARSITYPATTERN,MATRIX,VECTOR,deal_II_dimension>

#define DIN     DOpE::Integrator<IDC, VECTOR,double,deal_II_dimension>
#define NLS1    DOpE::InstatStepNewtonSolver<DIN, DCGS, VECTOR , deal_II_dimension>
#define NLS2    DOpE::InstatStepNewtonSolver<DIN, DGMRESS, VECTOR , deal_II_dimension>
#define NLS3    DOpE::InstatStepNewtonSolver<DIN, DDS, VECTOR , deal_II_dimension>
#define FNLS1   DOpE::FractionalStepThetaStepNewtonSolver<DIN, DCGS, VECTOR , deal_II_dimension>
#define FNLS2   DOpE::FractionalStepThetaStepNewtonSolver<DIN, DGMRESS, VECTOR , deal_II_dimension>
#define FNLS3   DOpE::FractionalStepThetaStepNewtonSolver<DIN, DDS, VECTOR , deal_II_dimension>

#define TSP1    DOpE::ForwardEulerProblem<PROB, SPARSITYPATTERN, VECTOR, dope_dimension,deal_II_dimension>
#define TSP2    DOpE::BackwardEulerProblem<PROB, SPARSITYPATTERN, VECTOR, dope_dimension,deal_II_dimension>
#define TSP3    DOpE::CrankNicolsonProblem<PROB, SPARSITYPATTERN, VECTOR, dope_dimension,deal_II_dimension>
#define TSP4    DOpE::ShiftedCrankNicolsonProblem<PROB, SPARSITYPATTERN, VECTOR, dope_dimension,deal_II_dimension>
#define TSP5    DOpE::FractionalStepThetaProblem<PROB, SPARSITYPATTERN, VECTOR, dope_dimension,deal_II_dimension>

#define DIOP1   DOpE::InstatOptProblemContainer<TSP1,FUNC,FUNC,PDE,DD,CONS,SPARSITYPATTERN, VECTOR, dope_dimension,deal_II_dimension>
#define DIOP2   DOpE::InstatOptProblemContainer<TSP2,FUNC,FUNC,PDE,DD,CONS,SPARSITYPATTERN, VECTOR, dope_dimension,deal_II_dimension>
#define DIOP3   DOpE::InstatOptProblemContainer<TSP3,FUNC,FUNC,PDE,DD,CONS,SPARSITYPATTERN, VECTOR, dope_dimension,deal_II_dimension>
#define DIOP4   DOpE::InstatOptProblemContainer<TSP4,FUNC,FUNC,PDE,DD,CONS,SPARSITYPATTERN, VECTOR, dope_dimension,deal_II_dimension>
#define DIOP5   DOpE::InstatOptProblemContainer<TSP5,FUNC,FUNC,PDE,DD,CONS,SPARSITYPATTERN, VECTOR, dope_dimension,deal_II_dimension>

///******************************************************/
template class DOpE::InstatReducedProblem<NLS1, NLS1,DIN,DIN,DIOP1,VECTOR,dope_dimension, deal_II_dimension>;
template class DOpE::InstatReducedProblem<NLS1, NLS1,DIN,DIN,DIOP2,VECTOR,dope_dimension, deal_II_dimension>;
template class DOpE::InstatReducedProblem<NLS1, NLS1,DIN,DIN,DIOP3,VECTOR,dope_dimension, deal_II_dimension>;
template class DOpE::InstatReducedProblem<NLS1, NLS1,DIN,DIN,DIOP4,VECTOR,dope_dimension, deal_II_dimension>;
template class DOpE::InstatReducedProblem<NLS1,FNLS1,DIN,DIN,DIOP5,VECTOR,dope_dimension, deal_II_dimension>;
/******************************************************/
template class DOpE::InstatReducedProblem<NLS2, NLS2,DIN,DIN,DIOP1,VECTOR,dope_dimension, deal_II_dimension>;
template class DOpE::InstatReducedProblem<NLS2, NLS2,DIN,DIN,DIOP2,VECTOR,dope_dimension, deal_II_dimension>;
template class DOpE::InstatReducedProblem<NLS2, NLS2,DIN,DIN,DIOP3,VECTOR,dope_dimension, deal_II_dimension>;
template class DOpE::InstatReducedProblem<NLS2, NLS2,DIN,DIN,DIOP4,VECTOR,dope_dimension, deal_II_dimension>;
template class DOpE::InstatReducedProblem<NLS2,FNLS2,DIN,DIN,DIOP5,VECTOR,dope_dimension, deal_II_dimension>;
/******************************************************/
template class DOpE::InstatReducedProblem<NLS3, NLS3,DIN,DIN,DIOP1,VECTOR,dope_dimension, deal_II_dimension>;
template class DOpE::InstatReducedProblem<NLS3, NLS3,DIN,DIN,DIOP2,VECTOR,dope_dimension, deal_II_dimension>;
template class DOpE::InstatReducedProblem<NLS3, NLS3,DIN,DIN,DIOP3,VECTOR,dope_dimension, deal_II_dimension>;
template class DOpE::InstatReducedProblem<NLS3, NLS3,DIN,DIN,DIOP4,VECTOR,dope_dimension, deal_II_dimension>;
template class DOpE::InstatReducedProblem<NLS3,FNLS3,DIN,DIN,DIOP5,VECTOR,dope_dimension, deal_II_dimension>;
#undef VECTOR
#undef SPARSITYPATTERN
#undef MATRIX
#undef DOFHANDLER
#undef IDC
#undef FE
#undef FUNC
#undef PDE
#undef DD
#undef CONS
#undef DOP
#undef PROB
#undef DCGS
#undef DGMRESS
#undef DDS
#undef DIN
#undef NLS1
#undef NLS2
#undef NLS3
#undef FNLS1
#undef FNLS2
#undef FNLS3
#undef TSP1
#undef TSP2
#undef TSP3
#undef TSP4
#undef TSP5
#undef DIOP1
#undef DIOP2
#undef DIOP3
#undef DIOP4
#undef DIOP5

/******************************************************/
//Now we are using Vector<double>
/******************************************************/
#define VECTOR dealii::Vector<double>
#define SPARSITYPATTERN dealii::SparsityPattern
#define MATRIX dealii::SparseMatrix<double>
#define DOFHANDLER dealii::DoFHandler<dope_dimension>
#define IDC     DOpE::IntegratorDataContainer<DOFHANDLER, dealii::Quadrature<deal_II_dimension>, dealii::Quadrature<deal_II_dimension-1>, VECTOR, deal_II_dimension  >
#define FE      DOpEWrapper::FiniteElement<deal_II_dimension>
#define FUNC    DOpE::FunctionalInterface<DOpE::CellDataContainer,DOpE::FaceDataContainer,DOFHANDLER,VECTOR,dope_dimension,deal_II_dimension>
#define PDE     DOpE::PDEInterface<DOpE::CellDataContainer,DOpE::FaceDataContainer,DOFHANDLER,VECTOR,dope_dimension,deal_II_dimension>
#define DD      DOpE::DirichletDataInterface<VECTOR,dope_dimension,deal_II_dimension>
#define CONS    DOpE::ConstraintInterface<DOpE::CellDataContainer,DOpE::FaceDataContainer,DOFHANDLER,VECTOR,dope_dimension,deal_II_dimension>

#define DOP     DOpE::OptProblem<FUNC,FUNC,PDE,DD,CONS,SPARSITYPATTERN,VECTOR,dope_dimension,deal_II_dimension, FE, DOFHANDLER>
#define PROB    DOpE::StateProblem<DOP,PDE,DD,SPARSITYPATTERN,VECTOR,dope_dimension,deal_II_dimension>
#define DCGS    DOpE::CGLinearSolverWithMatrix<DOpEWrapper::PreconditionIdentity_Wrapper<MATRIX>, SPARSITYPATTERN,MATRIX,VECTOR,deal_II_dimension>
#define DGMRESS DOpE::GMRESLinearSolverWithMatrix<DOpEWrapper::PreconditionIdentity_Wrapper<MATRIX>, SPARSITYPATTERN,MATRIX,VECTOR,deal_II_dimension>
#define DDS     DOpE::DirectLinearSolverWithMatrix<SPARSITYPATTERN,MATRIX,VECTOR,deal_II_dimension>

#define DIN     DOpE::Integrator<IDC,VECTOR,double,deal_II_dimension>
#define NLS1    DOpE::InstatStepNewtonSolver<DIN, DCGS, VECTOR , deal_II_dimension>
#define NLS2    DOpE::InstatStepNewtonSolver<DIN, DGMRESS, VECTOR , deal_II_dimension>
#define NLS3    DOpE::InstatStepNewtonSolver<DIN, DDS, VECTOR , deal_II_dimension>
#define FNLS1   DOpE::FractionalStepThetaStepNewtonSolver<DIN, DCGS, VECTOR , deal_II_dimension>
#define FNLS2   DOpE::FractionalStepThetaStepNewtonSolver<DIN, DGMRESS, VECTOR , deal_II_dimension>
#define FNLS3   DOpE::FractionalStepThetaStepNewtonSolver<DIN, DDS, VECTOR , deal_II_dimension>

#define TSP1    DOpE::ForwardEulerProblem<PROB, SPARSITYPATTERN, VECTOR, dope_dimension,deal_II_dimension>
#define TSP2    DOpE::BackwardEulerProblem<PROB, SPARSITYPATTERN, VECTOR, dope_dimension,deal_II_dimension>
#define TSP3    DOpE::CrankNicolsonProblem<PROB, SPARSITYPATTERN, VECTOR, dope_dimension,deal_II_dimension>
#define TSP4    DOpE::ShiftedCrankNicolsonProblem<PROB, SPARSITYPATTERN, VECTOR, dope_dimension,deal_II_dimension>
#define TSP5    DOpE::FractionalStepThetaProblem<PROB, SPARSITYPATTERN, VECTOR, dope_dimension,deal_II_dimension>

#define DIOP1   DOpE::InstatOptProblemContainer<TSP1,FUNC,FUNC,PDE,DD,CONS,SPARSITYPATTERN, VECTOR, dope_dimension,deal_II_dimension>
#define DIOP2   DOpE::InstatOptProblemContainer<TSP2,FUNC,FUNC,PDE,DD,CONS,SPARSITYPATTERN, VECTOR, dope_dimension,deal_II_dimension>
#define DIOP3   DOpE::InstatOptProblemContainer<TSP3,FUNC,FUNC,PDE,DD,CONS,SPARSITYPATTERN, VECTOR, dope_dimension,deal_II_dimension>
#define DIOP4   DOpE::InstatOptProblemContainer<TSP4,FUNC,FUNC,PDE,DD,CONS,SPARSITYPATTERN, VECTOR, dope_dimension,deal_II_dimension>
#define DIOP5   DOpE::InstatOptProblemContainer<TSP5,FUNC,FUNC,PDE,DD,CONS,SPARSITYPATTERN, VECTOR, dope_dimension,deal_II_dimension>

///******************************************************/
template class DOpE::InstatReducedProblem<NLS1, NLS1,DIN,DIN,DIOP1,VECTOR,dope_dimension, deal_II_dimension>;
template class DOpE::InstatReducedProblem<NLS1, NLS1,DIN,DIN,DIOP2,VECTOR,dope_dimension, deal_II_dimension>;
template class DOpE::InstatReducedProblem<NLS1, NLS1,DIN,DIN,DIOP3,VECTOR,dope_dimension, deal_II_dimension>;
template class DOpE::InstatReducedProblem<NLS1, NLS1,DIN,DIN,DIOP4,VECTOR,dope_dimension, deal_II_dimension>;
template class DOpE::InstatReducedProblem<NLS1,FNLS1,DIN,DIN,DIOP5,VECTOR,dope_dimension, deal_II_dimension>;
/******************************************************/
template class DOpE::InstatReducedProblem<NLS2, NLS2,DIN,DIN,DIOP1,VECTOR,dope_dimension, deal_II_dimension>;
template class DOpE::InstatReducedProblem<NLS2, NLS2,DIN,DIN,DIOP2,VECTOR,dope_dimension, deal_II_dimension>;
template class DOpE::InstatReducedProblem<NLS2, NLS2,DIN,DIN,DIOP3,VECTOR,dope_dimension, deal_II_dimension>;
template class DOpE::InstatReducedProblem<NLS2, NLS2,DIN,DIN,DIOP4,VECTOR,dope_dimension, deal_II_dimension>;
template class DOpE::InstatReducedProblem<NLS2,FNLS2,DIN,DIN,DIOP5,VECTOR,dope_dimension, deal_II_dimension>;
/******************************************************/
template class DOpE::InstatReducedProblem<NLS3, NLS3,DIN,DIN,DIOP1,VECTOR,dope_dimension, deal_II_dimension>;
template class DOpE::InstatReducedProblem<NLS3, NLS3,DIN,DIN,DIOP2,VECTOR,dope_dimension, deal_II_dimension>;
template class DOpE::InstatReducedProblem<NLS3, NLS3,DIN,DIN,DIOP3,VECTOR,dope_dimension, deal_II_dimension>;
template class DOpE::InstatReducedProblem<NLS3, NLS3,DIN,DIN,DIOP4,VECTOR,dope_dimension, deal_II_dimension>;
template class DOpE::InstatReducedProblem<NLS3,FNLS3,DIN,DIN,DIOP5,VECTOR,dope_dimension, deal_II_dimension>;
#undef VECTOR
#undef SPARSITYPATTERN
#undef MATRIX
#undef DOFHANDLER
#undef IDC
#undef FE
#undef FUNC
#undef PDE
#undef DD
#undef CONS
#undef DOP
#undef PROB
#undef DCGS
#undef DGMRESS
#undef DDS
#undef DIN
#undef NLS1
#undef NLS2
#undef NLS3
#undef FNLS1
#undef FNLS2
#undef FNLS3
#undef TSP1
#undef TSP2
#undef TSP3
#undef TSP4
#undef TSP5
#undef DIOP1
#undef DIOP2
#undef DIOP3
#undef DIOP4
#undef DIOP5

//*****************************************************/
//Everything with 'hp' DoFHandler
//*****************************************************/

/******************************************************/
///******************************************************/
////First the ones using BlockVector
///******************************************************/
#define VECTOR dealii::BlockVector<double>
#define SPARSITYPATTERN dealii::BlockSparsityPattern
#define MATRIX dealii::BlockSparseMatrix<double>
#define DOFHANDLER dealii::hp::DoFHandler<dope_dimension>
#define IDC     DOpE::IntegratorDataContainer<DOFHANDLER, dealii::hp::QCollection<deal_II_dimension>, dealii::hp::QCollection<deal_II_dimension-1>, VECTOR, deal_II_dimension  >
#define FE      DOpEWrapper::FECollection<deal_II_dimension>
#define FUNC    DOpE::FunctionalInterface<DOpE::CellDataContainer,DOpE::FaceDataContainer,DOFHANDLER,VECTOR,dope_dimension,deal_II_dimension>
#define PDE     DOpE::PDEInterface<DOpE::CellDataContainer,DOpE::FaceDataContainer,DOFHANDLER,VECTOR,dope_dimension,deal_II_dimension>
#define DD      DOpE::DirichletDataInterface<VECTOR,dope_dimension,deal_II_dimension>
#define CONS    DOpE::ConstraintInterface<DOpE::CellDataContainer,DOpE::FaceDataContainer,DOFHANDLER,VECTOR,dope_dimension,deal_II_dimension>

#define DOP     DOpE::OptProblem<FUNC,FUNC,PDE,DD,CONS,SPARSITYPATTERN,VECTOR,dope_dimension,deal_II_dimension, FE, DOFHANDLER>
#define PROB    DOpE::StateProblem<DOP,PDE,DD,SPARSITYPATTERN,VECTOR,dope_dimension,deal_II_dimension>
#define DCGS    DOpE::CGLinearSolverWithMatrix<DOpEWrapper::PreconditionIdentity_Wrapper<MATRIX>, SPARSITYPATTERN,MATRIX,VECTOR,deal_II_dimension>
#define DGMRESS DOpE::GMRESLinearSolverWithMatrix<DOpEWrapper::PreconditionIdentity_Wrapper<MATRIX>, SPARSITYPATTERN,MATRIX,VECTOR,deal_II_dimension>
#define DDS     DOpE::DirectLinearSolverWithMatrix<SPARSITYPATTERN,MATRIX,VECTOR,deal_II_dimension>

#define DIN     DOpE::Integrator<IDC, VECTOR,double,deal_II_dimension>
#define NLS1    DOpE::InstatStepNewtonSolver<DIN, DCGS, VECTOR , deal_II_dimension>
#define NLS2    DOpE::InstatStepNewtonSolver<DIN, DGMRESS, VECTOR , deal_II_dimension>
#define NLS3    DOpE::InstatStepNewtonSolver<DIN, DDS, VECTOR , deal_II_dimension>
#define FNLS1   DOpE::FractionalStepThetaStepNewtonSolver<DIN, DCGS, VECTOR , deal_II_dimension>
#define FNLS2   DOpE::FractionalStepThetaStepNewtonSolver<DIN, DGMRESS, VECTOR , deal_II_dimension>
#define FNLS3   DOpE::FractionalStepThetaStepNewtonSolver<DIN, DDS, VECTOR , deal_II_dimension>

#define TSP1    DOpE::ForwardEulerProblem<PROB, SPARSITYPATTERN, VECTOR, dope_dimension,deal_II_dimension, FE, DOFHANDLER>
#define TSP2    DOpE::BackwardEulerProblem<PROB, SPARSITYPATTERN, VECTOR, dope_dimension,deal_II_dimension, FE, DOFHANDLER>
#define TSP3    DOpE::CrankNicolsonProblem<PROB, SPARSITYPATTERN, VECTOR, dope_dimension,deal_II_dimension, FE, DOFHANDLER>
#define TSP4    DOpE::ShiftedCrankNicolsonProblem<PROB, SPARSITYPATTERN, VECTOR, dope_dimension,deal_II_dimension, FE, DOFHANDLER>
#define TSP5    DOpE::FractionalStepThetaProblem<PROB, SPARSITYPATTERN, VECTOR, dope_dimension,deal_II_dimension, FE, DOFHANDLER>

#define DIOP1   DOpE::InstatOptProblemContainer<TSP1,FUNC,FUNC,PDE,DD,CONS,SPARSITYPATTERN, VECTOR, dope_dimension,deal_II_dimension, FE, DOFHANDLER>
#define DIOP2   DOpE::InstatOptProblemContainer<TSP2,FUNC,FUNC,PDE,DD,CONS,SPARSITYPATTERN, VECTOR, dope_dimension,deal_II_dimension, FE, DOFHANDLER>
#define DIOP3   DOpE::InstatOptProblemContainer<TSP3,FUNC,FUNC,PDE,DD,CONS,SPARSITYPATTERN, VECTOR, dope_dimension,deal_II_dimension, FE, DOFHANDLER>
#define DIOP4   DOpE::InstatOptProblemContainer<TSP4,FUNC,FUNC,PDE,DD,CONS,SPARSITYPATTERN, VECTOR, dope_dimension,deal_II_dimension, FE, DOFHANDLER>
#define DIOP5   DOpE::InstatOptProblemContainer<TSP5,FUNC,FUNC,PDE,DD,CONS,SPARSITYPATTERN, VECTOR, dope_dimension,deal_II_dimension, FE, DOFHANDLER>

///******************************************************/
template class DOpE::InstatReducedProblem<NLS1, NLS1,DIN,DIN,DIOP1,VECTOR,dope_dimension, deal_II_dimension>;
template class DOpE::InstatReducedProblem<NLS1, NLS1,DIN,DIN,DIOP2,VECTOR,dope_dimension, deal_II_dimension>;
template class DOpE::InstatReducedProblem<NLS1, NLS1,DIN,DIN,DIOP3,VECTOR,dope_dimension, deal_II_dimension>;
template class DOpE::InstatReducedProblem<NLS1, NLS1,DIN,DIN,DIOP4,VECTOR,dope_dimension, deal_II_dimension>;
template class DOpE::InstatReducedProblem<NLS1,FNLS1,DIN,DIN,DIOP5,VECTOR,dope_dimension, deal_II_dimension>;
/******************************************************/
template class DOpE::InstatReducedProblem<NLS2, NLS2,DIN,DIN,DIOP1,VECTOR,dope_dimension, deal_II_dimension>;
template class DOpE::InstatReducedProblem<NLS2, NLS2,DIN,DIN,DIOP2,VECTOR,dope_dimension, deal_II_dimension>;
template class DOpE::InstatReducedProblem<NLS2, NLS2,DIN,DIN,DIOP3,VECTOR,dope_dimension, deal_II_dimension>;
template class DOpE::InstatReducedProblem<NLS2, NLS2,DIN,DIN,DIOP4,VECTOR,dope_dimension, deal_II_dimension>;
template class DOpE::InstatReducedProblem<NLS2,FNLS2,DIN,DIN,DIOP5,VECTOR,dope_dimension, deal_II_dimension>;
/******************************************************/
template class DOpE::InstatReducedProblem<NLS3, NLS3,DIN,DIN,DIOP1,VECTOR,dope_dimension, deal_II_dimension>;
template class DOpE::InstatReducedProblem<NLS3, NLS3,DIN,DIN,DIOP2,VECTOR,dope_dimension, deal_II_dimension>;
template class DOpE::InstatReducedProblem<NLS3, NLS3,DIN,DIN,DIOP3,VECTOR,dope_dimension, deal_II_dimension>;
template class DOpE::InstatReducedProblem<NLS3, NLS3,DIN,DIN,DIOP4,VECTOR,dope_dimension, deal_II_dimension>;
template class DOpE::InstatReducedProblem<NLS3,FNLS3,DIN,DIN,DIOP5,VECTOR,dope_dimension, deal_II_dimension>;
#undef VECTOR
#undef SPARSITYPATTERN
#undef MATRIX
#undef DOFHANDLER
#undef IDC
#undef FE
#undef FUNC
#undef PDE
#undef DD
#undef CONS
#undef DOP
#undef PROB
#undef DCGS
#undef DGMRESS
#undef DDS
#undef DIN
#undef NLS1
#undef NLS2
#undef NLS3
#undef FNLS1
#undef FNLS2
#undef FNLS3
#undef TSP1
#undef TSP2
#undef TSP3
#undef TSP4
#undef TSP5
#undef DIOP1
#undef DIOP2
#undef DIOP3
#undef DIOP4
#undef DIOP5   

/******************************************************/
//Now we are using Vector<double>
/******************************************************/
#define VECTOR dealii::Vector<double>
#define SPARSITYPATTERN dealii::SparsityPattern
#define MATRIX dealii::SparseMatrix<double>
#define DOFHANDLER dealii::hp::DoFHandler<dope_dimension>
#define IDC     DOpE::IntegratorDataContainer<DOFHANDLER, dealii::hp::QCollection<deal_II_dimension>, dealii::hp::QCollection<deal_II_dimension-1>, VECTOR, deal_II_dimension  >
#define FE      DOpEWrapper::FECollection<deal_II_dimension>
#define FUNC    DOpE::FunctionalInterface<DOpE::CellDataContainer,DOpE::FaceDataContainer,DOFHANDLER,VECTOR,dope_dimension,deal_II_dimension>
#define PDE     DOpE::PDEInterface<DOpE::CellDataContainer,DOpE::FaceDataContainer,DOFHANDLER,VECTOR,dope_dimension,deal_II_dimension>
#define DD      DOpE::DirichletDataInterface<VECTOR,dope_dimension,deal_II_dimension>
#define CONS    DOpE::ConstraintInterface<DOpE::CellDataContainer,DOpE::FaceDataContainer,DOFHANDLER,VECTOR,dope_dimension,deal_II_dimension>

#define DOP     DOpE::OptProblem<FUNC,FUNC,PDE,DD,CONS,SPARSITYPATTERN,VECTOR,dope_dimension,deal_II_dimension, FE, DOFHANDLER>
#define PROB    DOpE::StateProblem<DOP,PDE,DD,SPARSITYPATTERN,VECTOR,dope_dimension,deal_II_dimension>
#define DCGS    DOpE::CGLinearSolverWithMatrix<DOpEWrapper::PreconditionIdentity_Wrapper<MATRIX>, SPARSITYPATTERN,MATRIX,VECTOR,deal_II_dimension>
#define DGMRESS DOpE::GMRESLinearSolverWithMatrix<DOpEWrapper::PreconditionIdentity_Wrapper<MATRIX>, SPARSITYPATTERN,MATRIX,VECTOR,deal_II_dimension>
#define DDS     DOpE::DirectLinearSolverWithMatrix<SPARSITYPATTERN,MATRIX,VECTOR,deal_II_dimension>

#define DIN     DOpE::Integrator<IDC,VECTOR,double,deal_II_dimension>
#define NLS1    DOpE::InstatStepNewtonSolver<DIN, DCGS, VECTOR , deal_II_dimension>
#define NLS2    DOpE::InstatStepNewtonSolver<DIN, DGMRESS, VECTOR , deal_II_dimension>
#define NLS3    DOpE::InstatStepNewtonSolver<DIN, DDS, VECTOR , deal_II_dimension>
#define FNLS1   DOpE::FractionalStepThetaStepNewtonSolver<DIN, DCGS, VECTOR , deal_II_dimension>
#define FNLS2   DOpE::FractionalStepThetaStepNewtonSolver<DIN, DGMRESS, VECTOR , deal_II_dimension>
#define FNLS3   DOpE::FractionalStepThetaStepNewtonSolver<DIN, DDS, VECTOR , deal_II_dimension>

#define TSP1    DOpE::ForwardEulerProblem<PROB, SPARSITYPATTERN, VECTOR, dope_dimension,deal_II_dimension, FE, DOFHANDLER>
#define TSP2    DOpE::BackwardEulerProblem<PROB, SPARSITYPATTERN, VECTOR, dope_dimension,deal_II_dimension, FE, DOFHANDLER>
#define TSP3    DOpE::CrankNicolsonProblem<PROB, SPARSITYPATTERN, VECTOR, dope_dimension,deal_II_dimension, FE, DOFHANDLER>
#define TSP4    DOpE::ShiftedCrankNicolsonProblem<PROB, SPARSITYPATTERN, VECTOR, dope_dimension,deal_II_dimension, FE, DOFHANDLER>
#define TSP5    DOpE::FractionalStepThetaProblem<PROB, SPARSITYPATTERN, VECTOR, dope_dimension,deal_II_dimension, FE, DOFHANDLER>

#define DIOP1   DOpE::InstatOptProblemContainer<TSP1,FUNC,FUNC,PDE,DD,CONS,SPARSITYPATTERN, VECTOR, dope_dimension,deal_II_dimension, FE, DOFHANDLER>
#define DIOP2   DOpE::InstatOptProblemContainer<TSP2,FUNC,FUNC,PDE,DD,CONS,SPARSITYPATTERN, VECTOR, dope_dimension,deal_II_dimension, FE, DOFHANDLER>
#define DIOP3   DOpE::InstatOptProblemContainer<TSP3,FUNC,FUNC,PDE,DD,CONS,SPARSITYPATTERN, VECTOR, dope_dimension,deal_II_dimension, FE, DOFHANDLER>
#define DIOP4   DOpE::InstatOptProblemContainer<TSP4,FUNC,FUNC,PDE,DD,CONS,SPARSITYPATTERN, VECTOR, dope_dimension,deal_II_dimension, FE, DOFHANDLER>
#define DIOP5   DOpE::InstatOptProblemContainer<TSP5,FUNC,FUNC,PDE,DD,CONS,SPARSITYPATTERN, VECTOR, dope_dimension,deal_II_dimension, FE, DOFHANDLER>

///******************************************************/
template class DOpE::InstatReducedProblem<NLS1, NLS1,DIN,DIN,DIOP1,VECTOR,dope_dimension, deal_II_dimension>;
template class DOpE::InstatReducedProblem<NLS1, NLS1,DIN,DIN,DIOP2,VECTOR,dope_dimension, deal_II_dimension>;
template class DOpE::InstatReducedProblem<NLS1, NLS1,DIN,DIN,DIOP3,VECTOR,dope_dimension, deal_II_dimension>;
template class DOpE::InstatReducedProblem<NLS1, NLS1,DIN,DIN,DIOP4,VECTOR,dope_dimension, deal_II_dimension>;
template class DOpE::InstatReducedProblem<NLS1,FNLS1,DIN,DIN,DIOP5,VECTOR,dope_dimension, deal_II_dimension>;
/******************************************************/
template class DOpE::InstatReducedProblem<NLS2, NLS2,DIN,DIN,DIOP1,VECTOR,dope_dimension, deal_II_dimension>;
template class DOpE::InstatReducedProblem<NLS2, NLS2,DIN,DIN,DIOP2,VECTOR,dope_dimension, deal_II_dimension>;
template class DOpE::InstatReducedProblem<NLS2, NLS2,DIN,DIN,DIOP3,VECTOR,dope_dimension, deal_II_dimension>;
template class DOpE::InstatReducedProblem<NLS2, NLS2,DIN,DIN,DIOP4,VECTOR,dope_dimension, deal_II_dimension>;
template class DOpE::InstatReducedProblem<NLS2,FNLS2,DIN,DIN,DIOP5,VECTOR,dope_dimension, deal_II_dimension>;
/******************************************************/
template class DOpE::InstatReducedProblem<NLS3, NLS3,DIN,DIN,DIOP1,VECTOR,dope_dimension, deal_II_dimension>;
template class DOpE::InstatReducedProblem<NLS3, NLS3,DIN,DIN,DIOP2,VECTOR,dope_dimension, deal_II_dimension>;
template class DOpE::InstatReducedProblem<NLS3, NLS3,DIN,DIN,DIOP3,VECTOR,dope_dimension, deal_II_dimension>;
template class DOpE::InstatReducedProblem<NLS3, NLS3,DIN,DIN,DIOP4,VECTOR,dope_dimension, deal_II_dimension>;
template class DOpE::InstatReducedProblem<NLS3,FNLS3,DIN,DIN,DIOP5,VECTOR,dope_dimension, deal_II_dimension>;
#undef VECTOR
#undef SPARSITYPATTERN
#undef MATRIX
#undef DOFHANDLER
#undef IDC
#undef FE
#undef FUNC
#undef PDE
#undef DD
#undef CONS
#undef DOP
#undef PROB
#undef DCGS
#undef DGMRESS
#undef DDS
#undef DIN
#undef NLS1
#undef NLS2
#undef NLS3
#undef FNLS1
#undef FNLS2
#undef FNLS3
#undef TSP1
#undef TSP2
#undef TSP3
#undef TSP4
#undef TSP5
#undef DIOP1
#undef DIOP2
#undef DIOP3
#undef DIOP4
#undef DIOP5

//endif dope_dimension == deal_II_dimension
/******************************************************/
/******************************************************/
/******************************************************/

// Initizaliations for mixed dimension type problems
#elif dope_dimension == 0

#endif
