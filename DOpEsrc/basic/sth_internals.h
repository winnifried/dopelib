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

#include <wrapper/mapping_wrapper.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/hp/mapping_collection.h>

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
  }
}

#endif /* STH_INTERNALS_H_ */
