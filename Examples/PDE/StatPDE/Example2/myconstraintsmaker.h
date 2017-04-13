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
#ifndef MYCONSTRAINTSMAKER_H_
#define MYCONSTRAINTSMAKER_H_

#include <include/userdefineddofconstraints.h>

namespace DOpE
{
  /**
   * This class implements the periodicity-constraints.
   */
  template<template<int, int> class DH, int dim>
  class PeriodicityConstraints : public UserDefinedDoFConstraints<DH, dim>
  {
  public:
    PeriodicityConstraints() :
      UserDefinedDoFConstraints<DH, dim>()
    {
    }
    static void
    declare_params(ParameterReader &param_reader);

    virtual void
    MakeStateDoFConstraints(
      const DOpEWrapper::DoFHandler<dim, DH> &dof_handler,
      dealii::ConstraintMatrix &constraint_matrix) const;

    struct DoFInfo
    {
      DoFInfo()
      {
      }
      Point<dim> location;
    };
  private:
    /**
     * Determins whether the unsigned int dof is part of the vector<unsigned int> vector.
     */
    bool
    IsElement(unsigned int dof, std::vector<unsigned int> vector) const
    {
      std::vector<unsigned int>::iterator it;
      for (it = vector.begin(); it < vector.end(); it++)
        {
          if (dof == *it)
            return true;
        }
      return false;
    }
    ;

  };

  template<template<int, int> class DH, int dim>
  void
  PeriodicityConstraints<DH, dim>::declare_params(
    ParameterReader &param_reader)
  {
  }

  /**
   * This Function incorporates the constraints for
   *  periodic boundary conditions into the ConstraintMatrix
   *  constraint_matrix and closes it.
   *
   */

  template<template<int, int> class DH, int dim>
  void
  PeriodicityConstraints<DH, dim>::MakeStateDoFConstraints(
    const DOpEWrapper::DoFHandler<dim, DH> &dof_handler,
    dealii::ConstraintMatrix &constraint_matrix) const
  {
    /* Does not work on locally refined grids. We can only couple
     * dofs on a rectangular boundary.
     * We couple boundary_color 0 with 1 (in x direction) and boundary color 2
     *with 3(in y direction)
     */
    /****************************************************************************/
    unsigned int n_components = dof_handler.get_fe().n_components();
    unsigned int n_dofs = dof_handler.n_dofs();
    //get support points on the faces...make sure they exist
    assert(dof_handler.get_fe().has_face_support_points());
    const std::vector<Point<dim - 1> > &face_unit_support_points =
      dof_handler.get_fe().get_unit_face_support_points();

    //then make a quadrature-rule with them
    dealii::Quadrature<dim - 1> quadrature_formula(face_unit_support_points);
    typename DOpEWrapper::FEFaceValues<dim> fe_face_values(
      dof_handler.get_fe(), quadrature_formula,
      UpdateFlags(dealii::update_q_points));

    const unsigned int n_q_points = quadrature_formula.size();
    std::vector<unsigned int> global_dof_indices(
      dof_handler.get_fe().dofs_per_face);

    /************************************************************************************/
    //sides - components - map of dof-indices to location of  dof
    std::vector<std::vector<std::map<unsigned int, DoFInfo> > > dof_locations;
    dof_locations.resize(dim); //we need only half the sides
    for (int d = 0; d < dim; d++)
      dof_locations[d].resize(n_components);

    //first loop over all elements...
    for (typename DOpEWrapper::DoFHandler<dim, DH>::active_cell_iterator element =
           dof_handler.begin_active(); element != dof_handler.end(); ++element)
      {
        //...then loop over all faces.
        for (unsigned int face = 0;
             face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
          {
#if DEAL_II_VERSION_GTE(8,3,0)
            int boundary_indicator = element->face(face)->boundary_id();
#else
            int boundary_indicator = element->face(face)->boundary_indicator();
#endif
            // Proceed only if the boundary indicator is lower than 4 and
            //if face is on a boundary and the corresponding boundary indicator is even
            if (element->face(face)->at_boundary() && boundary_indicator < 4
                && boundary_indicator % 2 == 0)
              {
                element->face(face)->get_dof_indices(global_dof_indices);
                fe_face_values.reinit(element, face);
                //Now loop over all dofs on this face
                for (unsigned int i = 0; i < n_q_points; i++)
                  {
                    //dof_handler.get_fe().system_to_component_index(i).first gives the only nonzero component
                    dof_locations.at(boundary_indicator / 2).at(
                      dof_handler.get_fe().system_to_component_index(i).first)[global_dof_indices.at(
                          i)].location = fe_face_values.quadrature_point(i);
                  } //endfor nqpoints
              } //endif boundary etc.
          } //endfor
      } //endfor active_cell_iterator

    /*
     * now set the constraints
     * we need the following construct to save all the
     * couplings. The components of 'couplings' stand for:
     *  component - dof - couples with whom?
     *
     *  We need this complicated construct because we do not
     *  know how else to deal with the dofs in the corner.
     */
    std::vector<std::vector<std::vector<unsigned int> > > couplings(
      n_components);
    for (unsigned int i = 0; i < n_components; i++)
      {
        couplings.at(i).resize(n_dofs);
      }
    dealii::Point<dim> actual_dof_location;
    for (typename DOpEWrapper::DoFHandler<dim, DH>::active_cell_iterator element =
           dof_handler.begin_active(); element != dof_handler.end(); ++element)
      {
        for (unsigned int face = 0;
             face < dealii::GeometryInfo<dim>::faces_per_cell; ++face)
          {
#if DEAL_II_VERSION_GTE(8,3,0)
            int boundary_indicator = element->face(face)->boundary_id();
#else
            int boundary_indicator = element->face(face)->boundary_indicator();
#endif
            //Now loop over the remaining dofs, i.e. the ones with an odd boundary_indicator
            if (element->face(face)->at_boundary() && boundary_indicator < 4
                && boundary_indicator % 2 == 1)
              {
                element->face(face)->get_dof_indices(global_dof_indices);
                fe_face_values.reinit(element, face);
                //loop over all dofs on this face
                for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
                  {
                    int border_dim = boundary_indicator / 2; //in which direction (x or y) do we actually set the constraints?
                    typename std::map<unsigned int, DoFInfo>::const_iterator p =
                      dof_locations.at(border_dim).at(
                        dof_handler.get_fe().system_to_component_index(q_point).first).begin();
                    int actual_component =
                      dof_handler.get_fe().system_to_component_index(q_point).first;
                    /*
                     * Now move  through the map and look out for the point wich
                     * corresponds to the location of the actual dof. Add a constraints
                     * for these two.
                     */
                    for (;
                         p != dof_locations.at(border_dim).at(actual_component).end();
                         ++p)
                      {
                        double sum = 0.;
                        actual_dof_location = fe_face_values.quadrature_point(q_point);
                        actual_dof_location -= p->second.location;
                        /*
                         * Compute the distance, ignore the actual component.
                         */
                        for (int d = 0; d < dim; d++)
                          {
                            if (d != border_dim)
                              {
                                sum += std::fabs(actual_dof_location(d));
                              }
                          }
                        if (sum < 1e-12)
                          {
                            /*
                             * If we got here, we want to couple p->first with global_dof_indices.at(q_point),
                             * so add them into couplings (but only if they are not already there.)
                             */
                            if (!IsElement(p->first,
                                           couplings.at(actual_component).at(
                                             global_dof_indices.at(q_point))))
                              {
                                couplings.at(actual_component).at(
                                  global_dof_indices.at(q_point)).push_back(p->first);
                              }
                            if (!IsElement(global_dof_indices.at(q_point),
                                           couplings.at(actual_component).at(p->first)))
                              {
                                couplings.at(actual_component).at(p->first).push_back(
                                  global_dof_indices.at(q_point));
                              }
                            break;
                          }
                        Assert(
                          p != dof_locations.at(border_dim).at( dof_handler.get_fe().system_to_component_index( q_point).first).end(),
                          ExcMessage("No corresponding degree of freedom was found!"));
                      } //endfor p
                  } //endfor nqpoints
              } //endif
          } //endfor
      } //endfor active_cell_iterator

    //now set the 'normal' constraints, we will do the corner
    //case later. Normal couplings are indicated by the fact
    //that they couple only with one other degree of freedom
    std::vector<std::vector<unsigned int> > corners(n_components);
    for (unsigned int comp = 0; comp < n_components; comp++)
      {
        for (unsigned int i = 0; i < n_dofs; i++)
          {
            if (couplings.at(comp).at(i).size() > 0) //ansonsten schon in constraintmatrix geschrieben
              //oder der dof gehoert zu einer anderen componente
              {
                if (couplings.at(comp).at(i).size() == 1) //also normale constraints
                  {
                    assert(
                      couplings.at(comp).at(couplings.at(comp).at(i)[0])[0] == i);
                    //falls also dof i mit j verknuepft sein soll, dof j aber nicht mit i!

                    constraint_matrix.add_line(i);
                    constraint_matrix.add_entry(i, couplings.at(comp).at(i)[0], 1.0);
                    couplings.at(comp).at(couplings.at(comp).at(i)[0]).clear();
                  }
                else if (couplings.at(comp).at(i).size() == 2) //i.e. a corner
                  {
                    if (!IsElement(i, corners.at(comp)))
                      corners.at(comp).push_back(i);
                  }
                else
                  {
                    throw DOpEException("What shall I do? Wrong number of couplings",
                                        "PeriodicityConstraints<dim>::MakeConstraints");
                  }
              }
          }
      }

    //now do the corners:
    for (unsigned int comp = 0; comp < n_components; comp++)
      {
        for (unsigned int i = 0; i < corners.at(comp).size() - 1; i++)
          {
            constraint_matrix.add_line(corners.at(comp).at(i));
            constraint_matrix.add_entry(corners.at(comp).at(i),
                                        corners.at(comp).at(i + 1), 1.0);
          }
      }
  }

}

#endif /* MYCONSTRAINTSMAKER_H_ */
