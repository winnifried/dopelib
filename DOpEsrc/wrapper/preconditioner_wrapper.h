#ifndef _DOPE_PRECONDITIONER_H_
#define _DOPE_PRECONDITIONER_H_

#include <lac/precondition.h>
#include <lac/sparse_ilu.h>

namespace DOpEWrapper
{
  template <typename MATRIX>
    class PreconditionSSOR_Wrapper : public dealii::PreconditionSSOR<MATRIX>
  {
  public:
    void initialize(const MATRIX& A)
    {
      dealii::PreconditionSSOR<MATRIX>::initialize(A,1);
    }
  };

  template <typename MATRIX>
    class PreconditionIdentity_Wrapper : public dealii::PreconditionIdentity
  {
  public:
    void initialize(const MATRIX& /*A*/)
    {
    }
  };

  template <typename number>
    class PreconditionSparseILU_Wrapper : public dealii::SparseILU<number>
  {
  public:
    void initialize(const SparseMatrix<number>& A)
    {
      dealii::SparseILU<number>::initialize(A);
    }
  };
}

#endif
