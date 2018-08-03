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

#ifndef STH_INTERNALS_H_
#define STH_INTERNALS_H_

#include <vector> 
#include <wrapper/mapping_wrapper.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_tools.h>

#include <wrapper/dofhandler_wrapper.h>

using namespace dealii;

namespace DOpE
{
  namespace STHInternals
  {
    /**
     * Calls the deal.II map_dofs_to_support_points routine.
     * For DoFHandler
     */
    template<typename VECTOR, int dealdim>
    void
    MapDoFsToSupportPoints(
      const DOpEWrapper::Mapping<dealdim, dealii::DoFHandler > &mapping,
      const DOpEWrapper::DoFHandler<dealdim, dealii::DoFHandler > &dh,
      VECTOR &support_points)
    {

//        MappingQ1 < dealdim > mapping;

      DoFTools::map_dofs_to_support_points(mapping, dh, support_points);
    }

//    /**
//     * Calls the deal.II map_dofs_to_support_points routine.
//     * For MGDoFHandler (Experimental)
//     */
//    template<typename VECTOR, int dealdim>
//      void
//      MapDoFsToSupportPoints(
//          const DOpEWrapper::Mapping<dealdim, dealii::MGDoFHandler >& mapping,
//          const DOpEWrapper::DoFHandler<dealdim, dealii::MGDoFHandler >& dh,
//          VECTOR& support_points)
//      {
//
////        MappingQ1 < dealdim > mapping;
//
//        DoFTools::map_dofs_to_support_points(mapping, dh, support_points);
//      }

    /**
     * Calls the deal.II map_dofs_to_support_points routine.
     * For hp::DoFHandler
     */
    template<typename VECTOR, int dealdim>
    void
    MapDoFsToSupportPoints(
      const DOpEWrapper::Mapping<dealdim, dealii::hp::DoFHandler > &mapping,
      const DOpEWrapper::DoFHandler<dealdim, dealii::hp::DoFHandler > &dh,
      VECTOR &support_points)
    {

#if DEAL_II_VERSION_GTE(7,2,0)
//        MappingQ1<dealdim> mapping;
//        hp::MappingCollection<dealdim> map_col(mapping);

      DoFTools::map_dofs_to_support_points(mapping, dh, support_points);
//        DoFTools::map_dofs_to_support_points(map_col, dh, support_points);
#else
      throw DOpEException(
        "Your deal.ii version is too old. We need DoFTools::map_dofs_to_support_points for hp::DoFhandler"
        " (Implemented since 7.2, revision 24975)!",
        "MapDoFsToSupportPoints");
#endif
    }

    /**
     * Create a vector associating to each global-vertex-id the number 
     * of neighboring elements.
     *
     * Returns 0 if a vertex is hanging
     */
    template<int dim>
    void CalculateNeigbourElementsToVertices(dealii::Triangulation<dim>& triangulation, std::vector<unsigned int>& n_neighbour_to_vertex)
    {
      //Build the list of neighbours
      n_neighbour_to_vertex.resize(triangulation.n_vertices(),0);
      for(unsigned int i = 0; i < n_neighbour_to_vertex.size(); i++)
      {
	auto cells = GridTools::find_cells_adjacent_to_vertex(triangulation,i);
	unsigned int count = 0;
	int level = -1;
	bool hanging = false;
	for(unsigned int c=0; c < cells.size(); c++)
	{
	  if( c == 0 )
	  {
	    level = cells[c]->level();
	  }
	  else
	  {
	    if(level != cells[c]->level())
	    {
	      //The vertex maybe hanging, we must check
	      hanging = true;
	    }
	  }
	  count++;
	}
	if(hanging == true)
	{
	  //Check if vertex is really hanging
	  //A hanging Vertex is not a vertex of one of is neighbouring
	  //elements
	  for(unsigned int c=0; c < cells.size(); c++)
	  {
	    bool local_present=false;
	    for(unsigned int j = 0; j < GeometryInfo<dim>::vertices_per_cell; j++)
	    {
	      local_present=local_present||(cells[c]->vertex_index(j) == i);
	    }
	    hanging = hanging && local_present;
	    if(local_present==false)
	    {
	      break;
	    }
	  }
	  //if still hanging the vertex was in all elements (its not hanging)
	  if( !hanging )
	  {
	    count=0;
	  }
	}
	n_neighbour_to_vertex[i]=count;
      }
    }
    
  }//End of namespace STHInternals
}

#endif /* STH_INTERNALS_H_ */
