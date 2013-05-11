/**
 *
 * Copyright (C) 2012 by the DOpElib authors
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

#ifndef _CONSTRAINTS_H_
#define _CONSTRAINTS_H_

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
       *                                     Where the first unsigned int describes
       *                                     how many local entries in this Block are
       *                                     localy constrained, and the second entry
       *                                     defines how many constraints are given on
       *                                     this quantity.
       * @param global_constraints           The number of global constraints on the
       *                                     control and state variable.
       */
      Constraints(
          const std::vector<std::vector<unsigned int> >& local_control_constraints,
          unsigned int global_constraints)
      {
        _local_control_constraints.resize(local_control_constraints.size());
        _local_control_constraints_per_block.resize(
            local_control_constraints.size());
        for (unsigned int i = 0; i < local_control_constraints.size(); i++)
        {
          _local_control_constraints[i].resize(2);
          assert(local_control_constraints[i].size() == 2);
          _local_control_constraints[i][0] = local_control_constraints[i][0];
          _local_control_constraints[i][1] = local_control_constraints[i][1];
          _local_control_constraints_per_block[i] = 0;
        }
        _global_constraints = global_constraints;
        _n_dofs = 0;
        _n_local_control_dofs = 0;
      }
      /**
       * Copy Constructor
       */
      Constraints(const Constraints& c)
      {
        _local_control_constraints.resize(c._local_control_constraints.size());
        _local_control_constraints_per_block.resize(
            c._local_control_constraints_per_block.size());
        for (unsigned int i = 0; i < c._local_control_constraints.size(); i++)
        {
          _local_control_constraints[i].resize(2);
          assert(c._local_control_constraints[i].size() == 2);
          _local_control_constraints[i][0] = c._local_control_constraints[i][0];
          _local_control_constraints[i][1] = c._local_control_constraints[i][1];
          _local_control_constraints_per_block[i] = 0;
        }
        _global_constraints = c._global_constraints;
        _n_dofs = 0;
        _n_local_control_dofs = 0;
      }
      /**
       * Constructor to be used when  no constraints are present.
       */
      Constraints()
      {
        _local_control_constraints.clear();
        _global_constraints = 0;
        _local_control_constraints_per_block.clear();
        _n_dofs = 0;
        _n_local_control_dofs = 0;
      }

      /**
       * Reinitialize the required constraints.
       *
       */
      void
      ReInit(std::vector<unsigned int>& control_dofs_per_block)
      {
        if (_local_control_constraints_per_block.size()
            == control_dofs_per_block.size())
        {
          _n_dofs = 0;
          for (unsigned int i = 0; i < _local_control_constraints.size(); i++)
          {
            assert(
                control_dofs_per_block[i] % _local_control_constraints[i][0]
                    == 0);
            _local_control_constraints_per_block[i] = control_dofs_per_block[i]
                / _local_control_constraints[i][0]
                * _local_control_constraints[i][1];
            _n_dofs += _local_control_constraints_per_block[i];
            _n_local_control_dofs += _local_control_constraints_per_block[i];
          }
          _n_dofs += _global_constraints;
        }
        else
        {
          _local_control_constraints.clear();
          _global_constraints = 0;
          _local_control_constraints_per_block.clear();
          _n_dofs = 0;
          _n_local_control_dofs = 0;
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
          return _n_local_control_dofs;
        }
        if (name == "global")
        {
          return _global_constraints;
        }

        throw DOpEException("Unknown name " + name, "constraints::_n_dofs");
      }

      /**
       * Returns the DoFs Per Block Vector
       */
      const std::vector<unsigned int>&
      GetDoFsPerBlock(std::string name) const
      {
        if (name == "local")
        {
          return _local_control_constraints_per_block;
        }
        throw DOpEException("Unknown name " + name, "constraints::_n_dofs");
      }

    private:
      std::vector<std::vector<unsigned int> > _local_control_constraints;
      unsigned int _global_constraints;

      std::vector<unsigned int> _local_control_constraints_per_block;
      unsigned int _n_dofs, _n_local_control_dofs;
  };

}

#endif
