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

#ifndef POINTCONSTRAINTSMAKER_H_
#define POINTCONSTRAINTSMAKER_H_

#include "userdefineddofconstraints.h"

namespace DOpE
{
  /**
   * This class is used to implement zero dirichlet boundary values given
   * on a set of points given in the constructor.
   * The dirichlet components are given by the second argument.
   * 
   * @params c_points    A vector containing the points to be constrained
   * @params c_comps     A vector containing vectors of booleans. These booleans
   *                     indicate which components should be constraint at the point.
   *                     Both vetors are assumed to be in identical order!
   */
	template<typename DOFHANDLER, int dopedim, int dealdim = dopedim>
	class PointConstraints : public UserDefinedDoFConstraints<DOFHANDLER,dopedim,dealdim>
	{
	public:
	  PointConstraints(const std::vector<Point<dealdim> > &c_points, 
			   const std::vector<std::vector<bool> > &c_comps)
	: UserDefinedDoFConstraints<DOFHANDLER,dopedim,dealdim>(), _c_points(c_points), _c_comps(c_comps)
	{
	  if(_c_points.size() != _c_comps.size())
	    throw DOpEException("Number of Entries not matching!","PointConstraints::PointConstraints");
	}
	  
	  virtual void MakeStateDoFConstraints(
	    const DOpEWrapper::DoFHandler<dealdim, DOFHANDLER> & dof_handler,
	    dealii::ConstraintMatrix& constraint_matrix) const;
	
	  virtual void
	    MakeControlDoFConstraints(
	      const DOpEWrapper::DoFHandler<dopedim, DOFHANDLER> & /*dof_handler*/,
	      dealii::ConstraintMatrix& /*dof_constraints*/) const {}

	private:
	  
	  const std::vector<Point<dealdim> > & _c_points;
	  const std::vector<std::vector<bool> > & _c_comps;

	};

	template<typename DOFHANDLER, int dopedim, int dealdim>
	  void PointConstraints<DOFHANDLER, dopedim, dealdim>::MakeStateDoFConstraints(
	    const DOpEWrapper::DoFHandler<dealdim, DOFHANDLER > & dof_handler,
	    dealii::ConstraintMatrix& constraint_matrix) const
	{
	  MappingQ1<dealdim> mapping;
	  std::vector<Point<dealdim> > support_points(dof_handler.n_dofs());
	  STHInternals::MapDoFsToSupportPoints(this->GetMapping(),dof_handler, support_points);
	  for(unsigned int i = 0; i < _c_points.size(); i++)
	  {	  
	    std::vector<bool> selected_dofs(dof_handler.n_dofs());
	    DoFTools::extract_dofs(dof_handler,_c_comps[i],selected_dofs);
	    bool found = false;
	    for(unsigned int p = 0; p < support_points.size(); p++)
	    {
	      if(_c_points[i].distance(support_points[p]) < 1.e-12)
	      {
		found = true;
		if(selected_dofs[p] == true)
		{
		  constraint_matrix.add_line(p);
		}
		//Note that you cannot stop the loop here, since more than 
		//one dof may be located at a given point!
	      }
	    }
	    if(!found)
	    {
	      throw DOpEException("Points not found!","PointConstraints::MakeStateDoFConstraints");
	    }
	  }
	}

}

#endif /* POINTCONSTRAINTSMAKER_H_ */
