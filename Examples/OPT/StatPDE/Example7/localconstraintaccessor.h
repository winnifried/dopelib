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

#ifndef LOCAL_CONSTRAINT_ACCESSOR_H_
#define LOCAL_CONSTRAINT_ACCESSOR_H_

namespace DOpE
{

  class LocalConstraintAccessor
  {
  public:
    LocalConstraintAccessor()
    {
      rho_min_ = 1.e-4;
      rho_max_ = 1.;
    }

    void
    SetRhoMin(double val)
    {
      rho_min_ = val;
    }

    /* Write Control Values into Constraint vector shifted by the lower or upper bounds respectively*/
    inline void
    ControlToLowerConstraint(const dealii::BlockVector<double> &control,
                             dealii::BlockVector<double> &constraints)
    {
      assert(constraints.block(0).size() == 2 * control.block(0).size());
      for (unsigned int i = 0; i < control.block(0).size(); i++)
        {
          //Add Control Constraints, such that if control is feasible all  entries are not positive!
          constraints.block(0)(i) = rho_min_ - control.block(0)(i);
        }
    }

    inline void
    ControlToUpperConstraint(const dealii::BlockVector<double> &control,
                             dealii::BlockVector<double> &constraints)
    {
      assert(constraints.block(0).size() == 2 * control.block(0).size());
      for (unsigned int i = 0; i < control.block(0).size(); i++)
        {
          //Add Control Constraints, such that if control is feasible all  entries are not positive!
          constraints.block(0)(control.block(0).size() + i) = control.block(0)(
                                                                i) - rho_max_;
        }
    }

    inline void
    CopyLocalConstraintToTensor(const dealii::BlockVector<double> &constraint,
                                Tensor<2, 1> &tensor, unsigned int index) const
    {
      tensor[0][0] = constraint.block(0)(index);
    }
    inline void
    CopyTensorToLocalConstraint(const Tensor<2, 1> &tensor,
                                dealii::BlockVector<double> &constraint, unsigned int index) const
    {
      constraint.block(0)(index) = tensor[0][0];
    }

    inline void
    CopyLocalControlToVector(const dealii::BlockVector<double> &control,
                             Vector<double> &vec, unsigned int index) const
    {
      if (index >= control.block(0).size())
        vec(0) = control.block(0)(index - control.block(0).size());
      else
        vec(0) = control.block(0)(index);
    }
    inline void
    CopyLocalControlToTensor(const dealii::BlockVector<double> &control,
                             Tensor<2, 1> &tensor, unsigned int index) const
    {
      if (index >= control.block(0).size())
        tensor[0][0] = control.block(0)(index - control.block(0).size());
      else
        tensor[0][0] = control.block(0)(index);
    }
    inline void
    CopyTensorToLocalControl(const Tensor<2, 1> &tensor,
                             dealii::BlockVector<double> &control, unsigned int index) const
    {
      if (index >= control.block(0).size())
        control.block(0)(index - control.block(0).size()) = tensor[0][0];
      else
        control.block(0)(index) = tensor[0][0];
    }
    inline void
    AddTensorToLocalControl(const Tensor<2, 1> &tensor,
                            dealii::BlockVector<double> &control, unsigned int index) const
    {
      if (index >= control.block(0).size())
        control.block(0)(index - control.block(0).size()) += tensor[0][0];
      else
        control.block(0)(index) += tensor[0][0];
    }
    inline void
    AddVectorToLocalControl(const Vector<double> &vec,
                            dealii::BlockVector<double> &control, unsigned int index) const
    {
      assert(vec.size() == 1);
      if (index < control.block(0).size())
        {
          control.block(0)(index) += vec(0);
        }
      else
        {
          assert(index - control.block(0).size() < control.block(0).size());
          control.block(0)(index - control.block(0).size()) += vec(0);
        }
    }

    unsigned int
    GetNLocalControlDoFs(const dealii::BlockVector<double> *q) const
    {
      assert(q->n_blocks() == 1);
      return q->block(0).size();
    }
//Badname, because ret. ndofs
    unsigned int
    GetNLocalControlConstraintDoFs(
      const dealii::BlockVector<double> *q) const
    {
      return q->block(0).size();
    }


    inline void
    GetControlBoxConstraints(dealii::BlockVector<double> &lb,
                             dealii::BlockVector<double> &ub) const
    {
      lb = rho_min_;
      ub = rho_max_;
    }


    inline void
    ProjectToPositiveAndNegativePart(const Tensor<2, 1> &A,
                                     Tensor<2, 1> &A_plus, Tensor<2, 1> &A_minus, double &max_EV)
    {
      A_plus[0][0] = std::max(0., A[0][0]);
      A_minus[0][0] = std::min(0., A[0][0]);
      max_EV = A[0][0];
    }

    // How many controls are Blocked locally, for a scalar its one, a symmetric 2x2 Matrix is 6 ...
    inline unsigned int
    NLocalDirections() const
    {
      return 1;
    }

    inline void
    LocalControlToConstraintBlocks(const dealii::BlockVector<double> *q,
                                   std::vector<std::vector<unsigned int> > &constraint)
    {
      constraint.resize(q->size(), std::vector<unsigned int>(2, 0));
      for (unsigned int i = 0; i < q->size(); i++)
        {
          constraint[i][0] = i;
          constraint[i][1] = i + q->size();
        }
    }

    inline void
    GetLocalConstraintDerivative(Tensor<2, 1> &derivative,
                                 const dealii::BlockVector<double> &constraints,
                                 unsigned int local_block, unsigned int /*local_index*/) const
    {
//        assert(local_index == 0);
      if (local_block < constraints.block(0).size() / 2)
        {
          // It is a lower bound
          derivative[0][0] = -1.;
        }
      else
        {
          // It is an upper bound
          derivative[0][0] = 1.;
        }
    }

    inline void
    GetLocalConstraintSecondDerivative(Tensor<2, 1> &derivative,
                                       const dealii::BlockVector<double> & /*constraints*/,
                                       unsigned int /*local_block*/,
                                       unsigned int /*local_index_1*/,
                                       unsigned int /*local_index_2*/) const
    {
//        assert(local_index_1 == 0);
//        assert(local_index_2 == 0);

      derivative[0][0] = 0.;
    }

    inline void
    GetLocalConstraintDerivative(Tensor<2, 1> &derivative,
                                 const dealii::BlockVector<double> &constraints,
                                 unsigned int local_block) const
    {
      if (local_block < constraints.block(0).size() / 2)
        {
          // It is a lower bound
          derivative[0][0] = -1.;
        }
      else
        {
          // It is an upper bound
          derivative[0][0] = 1.;
        }
    }

    inline void
    GetLocalConstraintSecondDerivative(Tensor<2, 1> &derivative,
                                       const dealii::BlockVector<double> & /*constraints*/,
                                       unsigned int /*local_block*/) const
    {
      derivative[0][0] = 0.;
    }
    /************************************************************************************************/

  private:
    double rho_min_, rho_max_;
  };
}
#endif
