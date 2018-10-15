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

#ifndef CONSTRAINTS_H_
#define CONSTRAINTS_H_

namespace DOpE
{
  /**
   * This class is designed to describe the constraints imposed on the problem.
   *
   */
  class Constraints
  {
  public:
    /**
     * Constructor for Constraint Descriptions
     *
     * @param local_control_constraints    Each entry in the vector corresponds to one
     *                                     block of the control variable.
     *                                     (Here blocks refer to the blocks in a
     *                                      deal.II BlockVector)
     *                                     Each entry consist of a vector of length two.
     *                                     In these, the first unsigned int describes
     *                                     how many local entries in this Block are
     *                                     locally constrained, and the second entry
     *                                     defines how many constraints are given on
     *                                     this quantity.
     * @param global_constraints           The number of global constraints on the
     *                                     control and state variable.
     */
    Constraints(
      const std::vector<std::vector<unsigned int> > &local_control_constraints,
      unsigned int global_constraints)
    {
      local_control_constraints_.resize(local_control_constraints.size());
      local_control_constraints_per_block_.resize(
        local_control_constraints.size());
      for (unsigned int i = 0; i < local_control_constraints.size(); i++)
        {
          local_control_constraints_[i].resize(2);
          assert(local_control_constraints[i].size() == 2);
          local_control_constraints_[i][0] = local_control_constraints[i][0];
          local_control_constraints_[i][1] = local_control_constraints[i][1];
          local_control_constraints_per_block_[i] = 0;
        }
      global_constraints_ = global_constraints;
      n_dofs_ = 0;
      n_local_control_dofs_ = 0;
    }
    /**
     * Copy Constructor
     */
    Constraints(const Constraints &c)
    {
      local_control_constraints_.resize(c.local_control_constraints_.size());
      local_control_constraints_per_block_.resize(
        c.local_control_constraints_per_block_.size());
      for (unsigned int i = 0; i < c.local_control_constraints_.size(); i++)
        {
          local_control_constraints_[i].resize(2);
          assert(c.local_control_constraints_[i].size() == 2);
          local_control_constraints_[i][0] = c.local_control_constraints_[i][0];
          local_control_constraints_[i][1] = c.local_control_constraints_[i][1];
          local_control_constraints_per_block_[i] = 0;
        }
      global_constraints_ = c.global_constraints_;
      n_dofs_ = 0;
      n_local_control_dofs_ = 0;
    }
    /**
     * Constructor to be used when  no constraints are present.
     */
    Constraints()
    {
      local_control_constraints_.clear();
      global_constraints_ = 0;
      local_control_constraints_per_block_.clear();
      n_dofs_ = 0;
      n_local_control_dofs_ = 0;
    }

    /**
     * Reinitialize the required constraints.
     *
     */
    void
    ReInit(std::vector<unsigned int> &control_dofs_per_block)
    {
      if (local_control_constraints_per_block_.size()
          == control_dofs_per_block.size())
        {
          n_dofs_ = 0;
          for (unsigned int i = 0; i < local_control_constraints_.size(); i++)
            {
              assert(
                control_dofs_per_block[i] % local_control_constraints_[i][0]
                == 0);
              local_control_constraints_per_block_[i] = control_dofs_per_block[i]
                                                        / local_control_constraints_[i][0]
                                                        * local_control_constraints_[i][1];
              n_dofs_ += local_control_constraints_per_block_[i];
              n_local_control_dofs_ += local_control_constraints_per_block_[i];
            }
          n_dofs_ += global_constraints_;
        }
      else
        {
          local_control_constraints_.clear();
          global_constraints_ = 0;
          local_control_constraints_per_block_.clear();
          n_dofs_ = 0;
          n_local_control_dofs_ = 0;
        }
    }

    /**
     * Returns the total number of local in time constraints.
     */
    unsigned int
    n_dofs(std::string name) const
    {
      if (name == "local")
        {
          return n_local_control_dofs_;
        }
      if (name == "global")
        {
          return global_constraints_;
        }

      throw DOpEException("Unknown name " + name, "constraints::n_dofs_");
    }

    /**
     * Returns the DoFs Per Block Vector
     */
    const std::vector<unsigned int> &
    GetDoFsPerBlock(std::string name) const
    {
      if (name == "local")
        {
          return local_control_constraints_per_block_;
        }
      throw DOpEException("Unknown name " + name, "constraints::n_dofs_");
    }

  private:
    std::vector<std::vector<unsigned int> > local_control_constraints_;
    unsigned int global_constraints_;

    std::vector<unsigned int> local_control_constraints_per_block_;
    unsigned int n_dofs_, n_local_control_dofs_;
  };

}

#endif
