/**
*
* Copyright (C) 2012-2014 by the DOpElib authors
*
* This file is part of DOpElib
*
* DOpElib is free software: you can redistribute it
* and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either
* version 3 of the License, or (at your option) any later
* version.
*
* DOpElib is distributed in the hope that it will be
* useful, but WITHOUT ANY WARRANTY; without even the implied
* warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
* PURPOSE.  See the GNU General Public License for more
* details.
*
* Please refer to the file LICENSE.TXT included in this distribution
* for further information on this license.
*
**/

#ifndef DOPE_PRECONDITIONER_H_
#define DOPE_PRECONDITIONER_H_

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/sparse_ilu.h>

/**
 * @file preconditioner_wrapper.h
 *
 * This file contains a collection of Wrappers to the
 * different preconditioners provided by dealii so
 * that they all have the same interface allowing
 * their use as template arguments in our linear solvers.
 *
 * Note that they all can be used in the linear solvers by dealii
 * but they do not have the same initialization methods!
 */
namespace DOpEWrapper
{
  /**
    * @class PreconditionSSOR_Wrapper
    *
    * Wrapper for the dealii::PreconditionSSOR preconditioner.
    *
    * This is provided to provide a unified initialization interface
    * to the preconditioners making them useable as template arguments
    * in our linear solvers.
    *
    * @tparam <MATRIX>   The used matrix type
    */
  template <typename MATRIX>
  class PreconditionSSOR_Wrapper : public dealii::PreconditionSSOR<MATRIX>
  {
  public:
    void initialize(const MATRIX &A)
    {
      dealii::PreconditionSSOR<MATRIX>::initialize(A,1);
    }
  };

  /**
   * @class PreconditionBlockSSOR_Wrapper
   *
   * Wrapper for the dealii::PreconditionBlockSSOR preconditioner.
   *
   * This is provided to provide a unified initialization interface
   * to the preconditioners making them useable as template arguments
   * in our linear solvers.
   *
   * @tparam <MATRIX>      The used matrix type
   * @tparam <blocksize>   The Blocksize to be considered
   */
  template <typename MATRIX,int blocksize>
  class PreconditionBlockSSOR_Wrapper : public dealii::PreconditionBlockSSOR<MATRIX>
  {
  public:
    void initialize(const MATRIX &A)
    {
      dealii::PreconditionBlockSSOR<MATRIX>::initialize(A,blocksize);
    }
  };


  /**
    * @class PreconditionIdentity_Wrapper
    *
    * Wrapper for the dealii::PreconditionIdentity preconditioner.
    *
    * This is provided to provide a unified initialization interface
    * to the preconditioners making them useable as template arguments
    * in our linear solvers.
    *
    * @tparam <MATRIX>   The used matrix type
    */
  template <typename MATRIX>
  class PreconditionIdentity_Wrapper : public dealii::PreconditionIdentity
  {
  public:
    void initialize(const MATRIX & /*A*/)
    {
    }
  };

  /**
    * @class PreconditionSparseILU_Wrapper
    *
    * Wrapper for the dealii::PreconditionSparseILU preconditioner.
    *
    * This is provided to provide a unified initialization interface
    * to the preconditioners making them useable as template arguments
    * in our linear solvers.
    *
    * @tparam <MATRIX>   The used matrix type
    */
  template <typename number>
  class PreconditionSparseILU_Wrapper : public dealii::SparseILU<number>
  {
  public:
    void initialize(const SparseMatrix<number> &A)
    {
      dealii::SparseILU<number>::initialize(A);
    }
  };
}

#endif
