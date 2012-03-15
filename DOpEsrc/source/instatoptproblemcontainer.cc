#include "instatoptproblemcontainer.h"
#include "pdeinterface.h"
#include "functionalinterface.h"
#include "dirichletdatainterface.h"
#include "dopeexception.h"
#include "primaldirichletdata.h"
#include "tangentdirichletdata.h"
#include "constraintinterface.h"
#include "forward_euler_problem.h"
#include "backward_euler_problem.h"
#include "crank_nicolson_problem.h"
#include "shifted_crank_nicolson_problem.h"
#include "fractional_step_theta_problem.h"

//////////////First normal DoFHandler/////////////////////////////////////////////////////
//////////////BLOCKVERSION////////////////////////
#define _VEC dealii::BlockVector<double>
#define _DOFH dealii::DoFHandler<deal_II_dimension>
#define _SP dealii::BlockSparsityPattern

#define _FUNC DOpE::FunctionalInterface<DOpE::CellDataContainer,DOpE::FaceDataContainer,_DOFH,_VEC, dope_dimension,deal_II_dimension>
#define _PDE DOpE::PDEInterface<DOpE::CellDataContainer,DOpE::FaceDataContainer,_DOFH, _VEC,dope_dimension, deal_II_dimension>
#define _DD DOpE::DirichletDataInterface<_VEC, dope_dimension, deal_II_dimension>
#define _CONS DOpE::ConstraintInterface<DOpE::CellDataContainer,DOpE::FaceDataContainer,_DOFH,_VEC, dope_dimension, deal_II_dimension>
#define _OP DOpE::OptProblem<_FUNC,_FUNC,_PDE,_DD,_CONS,_SP,_VEC,dope_dimension,deal_II_dimension>
#define _PROB DOpE::StateProblem<_OP,_PDE,_DD,_SP,_VEC,dope_dimension,deal_II_dimension>

#define _FETS DOpE::ForwardEulerProblem<_PROB,_SP,_VEC,dope_dimension,deal_II_dimension>
#define _BETS DOpE::BackwardEulerProblem<_PROB,_SP,_VEC,dope_dimension,deal_II_dimension>
#define _CNTS DOpE::CrankNicolsonProblem<_PROB,_SP,_VEC,dope_dimension,deal_II_dimension>
#define _SCNTS DOpE::ShiftedCrankNicolsonProblem<_PROB,_SP,_VEC,dope_dimension,deal_II_dimension>
#define _FSTS DOpE::FractionalStepThetaProblem<_PROB,_SP,_VEC,dope_dimension,deal_II_dimension>

template class DOpE::InstatOptProblemContainer<_FETS,_FUNC,_FUNC,_PDE,_DD,_CONS,_SP,_VEC,dope_dimension,deal_II_dimension>;
template class DOpE::InstatOptProblemContainer<_BETS,_FUNC,_FUNC,_PDE,_DD,_CONS,_SP,_VEC,dope_dimension,deal_II_dimension>;
template class DOpE::InstatOptProblemContainer<_CNTS,_FUNC,_FUNC,_PDE,_DD,_CONS,_SP,_VEC,dope_dimension,deal_II_dimension>;
template class DOpE::InstatOptProblemContainer<_SCNTS,_FUNC,_FUNC,_PDE,_DD,_CONS,_SP,_VEC,dope_dimension,deal_II_dimension>;
template class DOpE::InstatOptProblemContainer<_FSTS,_FUNC,_FUNC,_PDE,_DD,_CONS,_SP,_VEC,dope_dimension,deal_II_dimension>;

#undef _VEC  
#undef _DOFH 
#undef _SP   
#undef _FUNC 
#undef _PDE  
#undef _DD   
#undef _CONS 
#undef _OP   
#undef _PROB 
#undef _FETS 
#undef _BETS 
#undef _CNTS 
#undef _SCNTS 
#undef _FSTS 
/////////////////////NORMALVECTORS////////////////////////////
#define _VEC dealii::Vector<double>
#define _DOFH dealii::DoFHandler<deal_II_dimension>
#define _SP dealii::SparsityPattern

#define _FUNC DOpE::FunctionalInterface<DOpE::CellDataContainer,DOpE::FaceDataContainer,_DOFH,_VEC, dope_dimension,deal_II_dimension>
#define _PDE DOpE::PDEInterface<DOpE::CellDataContainer,DOpE::FaceDataContainer,_DOFH, _VEC,dope_dimension, deal_II_dimension>
#define _DD DOpE::DirichletDataInterface<_VEC, dope_dimension, deal_II_dimension>
#define _CONS DOpE::ConstraintInterface<DOpE::CellDataContainer,DOpE::FaceDataContainer,_DOFH,_VEC, dope_dimension, deal_II_dimension>
#define _OP DOpE::OptProblem<_FUNC,_FUNC,_PDE,_DD,_CONS,_SP,_VEC,dope_dimension,deal_II_dimension>
#define _PROB DOpE::StateProblem<_OP,_PDE,_DD,_SP,_VEC,dope_dimension,deal_II_dimension>

#define _FETS DOpE::ForwardEulerProblem<_PROB,_SP,_VEC,dope_dimension,deal_II_dimension>
#define _BETS DOpE::BackwardEulerProblem<_PROB,_SP,_VEC,dope_dimension,deal_II_dimension>
#define _CNTS DOpE::CrankNicolsonProblem<_PROB,_SP,_VEC,dope_dimension,deal_II_dimension>
#define _SCNTS DOpE::ShiftedCrankNicolsonProblem<_PROB,_SP,_VEC,dope_dimension,deal_II_dimension>
#define _FSTS DOpE::FractionalStepThetaProblem<_PROB,_SP,_VEC,dope_dimension,deal_II_dimension>

template class DOpE::InstatOptProblemContainer<_FETS,_FUNC,_FUNC,_PDE,_DD,_CONS,_SP,_VEC,dope_dimension,deal_II_dimension>;
template class DOpE::InstatOptProblemContainer<_BETS,_FUNC,_FUNC,_PDE,_DD,_CONS,_SP,_VEC,dope_dimension,deal_II_dimension>;
template class DOpE::InstatOptProblemContainer<_CNTS,_FUNC,_FUNC,_PDE,_DD,_CONS,_SP,_VEC,dope_dimension,deal_II_dimension>;
template class DOpE::InstatOptProblemContainer<_SCNTS,_FUNC,_FUNC,_PDE,_DD,_CONS,_SP,_VEC,dope_dimension,deal_II_dimension>;
template class DOpE::InstatOptProblemContainer<_FSTS,_FUNC,_FUNC,_PDE,_DD,_CONS,_SP,_VEC,dope_dimension,deal_II_dimension>;

#undef _VEC  
#undef _DOFH 
#undef _SP   
#undef _FUNC 
#undef _PDE  
#undef _DD   
#undef _CONS 
#undef _OP   
#undef _PROB 
#undef _FETS 
#undef _BETS 
#undef _CNTS 
#undef _SCNTS 
#undef _FSTS 

//////////////Then hp DoFHandler/////////////////////////////////////////////////////
//////////////BLOCKVERSION////////////////////////
#define _VEC dealii::BlockVector<double>
#define _DOFH dealii::hp::DoFHandler<deal_II_dimension>
#define _FE DOpEWrapper::FECollection<deal_II_dimension>
#define _SP dealii::BlockSparsityPattern

#define _FUNC DOpE::FunctionalInterface<DOpE::CellDataContainer,DOpE::FaceDataContainer,_DOFH,_VEC, dope_dimension,deal_II_dimension>
#define _PDE DOpE::PDEInterface<DOpE::CellDataContainer,DOpE::FaceDataContainer,_DOFH, _VEC,dope_dimension, deal_II_dimension>
#define _DD DOpE::DirichletDataInterface<_VEC, dope_dimension, deal_II_dimension>
#define _CONS DOpE::ConstraintInterface<DOpE::CellDataContainer,DOpE::FaceDataContainer,_DOFH,_VEC, dope_dimension, deal_II_dimension>
#define _OP DOpE::OptProblem<_FUNC,_FUNC,_PDE,_DD,_CONS,_SP,_VEC,dope_dimension,deal_II_dimension, _FE, _DOFH>
#define _PROB DOpE::StateProblem<_OP,_PDE,_DD,_SP,_VEC,dope_dimension,deal_II_dimension>

#define _FETS DOpE::ForwardEulerProblem<_PROB,_SP,_VEC,dope_dimension,deal_II_dimension, _FE, _DOFH>
#define _BETS DOpE::BackwardEulerProblem<_PROB,_SP,_VEC,dope_dimension,deal_II_dimension, _FE, _DOFH>
#define _CNTS DOpE::CrankNicolsonProblem<_PROB,_SP,_VEC,dope_dimension,deal_II_dimension, _FE, _DOFH>
#define _SCNTS DOpE::ShiftedCrankNicolsonProblem<_PROB,_SP,_VEC,dope_dimension,deal_II_dimension, _FE, _DOFH>
#define _FSTS DOpE::FractionalStepThetaProblem<_PROB,_SP,_VEC,dope_dimension,deal_II_dimension, _FE, _DOFH>

template class DOpE::InstatOptProblemContainer<_FETS,_FUNC,_FUNC,_PDE,_DD,_CONS,_SP,_VEC,dope_dimension,deal_II_dimension, _FE, _DOFH>;
template class DOpE::InstatOptProblemContainer<_BETS,_FUNC,_FUNC,_PDE,_DD,_CONS,_SP,_VEC,dope_dimension,deal_II_dimension, _FE, _DOFH>;
template class DOpE::InstatOptProblemContainer<_CNTS,_FUNC,_FUNC,_PDE,_DD,_CONS,_SP,_VEC,dope_dimension,deal_II_dimension, _FE, _DOFH>;
template class DOpE::InstatOptProblemContainer<_SCNTS,_FUNC,_FUNC,_PDE,_DD,_CONS,_SP,_VEC,dope_dimension,deal_II_dimension, _FE, _DOFH>;
template class DOpE::InstatOptProblemContainer<_FSTS,_FUNC,_FUNC,_PDE,_DD,_CONS,_SP,_VEC,dope_dimension,deal_II_dimension, _FE, _DOFH>;

#undef _VEC
#undef _DOFH
#undef _FE
#undef _SP
#undef _FUNC
#undef _PDE
#undef _DD
#undef _CONS
#undef _OP
#undef _PROB
#undef _FETS
#undef _BETS
#undef _CNTS
#undef _SCNTS
#undef _FSTS
/////////////////////NORMALVECTORS////////////////////////////
#define _VEC dealii::Vector<double>
#define _DOFH dealii::hp::DoFHandler<deal_II_dimension>
#define _FE DOpEWrapper::FECollection<deal_II_dimension>
#define _SP dealii::SparsityPattern
#define _FUNC DOpE::FunctionalInterface<DOpE::CellDataContainer,DOpE::FaceDataContainer,_DOFH,_VEC, dope_dimension,deal_II_dimension>
#define _PDE DOpE::PDEInterface<DOpE::CellDataContainer,DOpE::FaceDataContainer,_DOFH, _VEC,dope_dimension, deal_II_dimension>
#define _DD DOpE::DirichletDataInterface<_VEC, dope_dimension, deal_II_dimension>
#define _CONS DOpE::ConstraintInterface<DOpE::CellDataContainer,DOpE::FaceDataContainer,_DOFH,_VEC, dope_dimension, deal_II_dimension>
#define _OP DOpE::OptProblem<_FUNC,_FUNC,_PDE,_DD,_CONS,_SP,_VEC,dope_dimension,deal_II_dimension, _FE, _DOFH>
#define _PROB DOpE::StateProblem<_OP,_PDE,_DD,_SP,_VEC,dope_dimension,deal_II_dimension>

#define _FETS DOpE::ForwardEulerProblem<_PROB,_SP,_VEC,dope_dimension,deal_II_dimension, _FE, _DOFH>
#define _BETS DOpE::BackwardEulerProblem<_PROB,_SP,_VEC,dope_dimension,deal_II_dimension, _FE, _DOFH>
#define _CNTS DOpE::CrankNicolsonProblem<_PROB,_SP,_VEC,dope_dimension,deal_II_dimension, _FE, _DOFH>
#define _SCNTS DOpE::ShiftedCrankNicolsonProblem<_PROB,_SP,_VEC,dope_dimension,deal_II_dimension, _FE, _DOFH>
#define _FSTS DOpE::FractionalStepThetaProblem<_PROB,_SP,_VEC,dope_dimension,deal_II_dimension, _FE, _DOFH>

template class DOpE::InstatOptProblemContainer<_FETS,_FUNC,_FUNC,_PDE,_DD,_CONS,_SP,_VEC,dope_dimension,deal_II_dimension, _FE, _DOFH>;
template class DOpE::InstatOptProblemContainer<_BETS,_FUNC,_FUNC,_PDE,_DD,_CONS,_SP,_VEC,dope_dimension,deal_II_dimension, _FE, _DOFH>;
template class DOpE::InstatOptProblemContainer<_CNTS,_FUNC,_FUNC,_PDE,_DD,_CONS,_SP,_VEC,dope_dimension,deal_II_dimension, _FE, _DOFH>;
template class DOpE::InstatOptProblemContainer<_SCNTS,_FUNC,_FUNC,_PDE,_DD,_CONS,_SP,_VEC,dope_dimension,deal_II_dimension, _FE, _DOFH>;
template class DOpE::InstatOptProblemContainer<_FSTS,_FUNC,_FUNC,_PDE,_DD,_CONS,_SP,_VEC,dope_dimension,deal_II_dimension, _FE, _DOFH>;

#undef _VEC
#undef _DOFH
#undef _FE
#undef _SP
#undef _FUNC
#undef _PDE
#undef _DD
#undef _CONS
#undef _OP
#undef _PROB
#undef _FETS
#undef _BETS
#undef _CNTS
#undef _SCNTS
#undef _FSTS



