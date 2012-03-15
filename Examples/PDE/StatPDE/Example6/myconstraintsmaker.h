/*
 * myconstraintsmaker.h
 *
 *  Created on: Aug 10, 2011
 *      Author: cgoll
 */

#ifndef MYCONSTRAINTSMAKER_H_
#define MYCONSTRAINTSMAKER_H_

#include "constraintsmaker.h"

namespace DOpE
{
	template<int dim>
	class PeriodicityConstraints: public ConstraintsMaker<dealii::DoFHandler<dim>, dim>
	{
		public:
			PeriodicityConstraints()
					: ConstraintsMaker<dealii::DoFHandler<dim>, dim>()
			{
			}
			static void declare_params(ParameterReader &param_reader);

			virtual void MakeConstraints(
			    const DOpEWrapper::DoFHandler<dim, dealii::DoFHandler<dim> > & dof_handler,
			    dealii::ConstraintMatrix& constraint_matrix) const;
//			virtual void MakeConstraints(
//			    const DOpEWrapper::HpDoFHandler<dim> & hp_dof_handler,
//			    dealii::ConstraintMatrix& constraint_matrix) const;
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
			bool IsElement(unsigned int dof, std::vector<unsigned int> vector) const
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

	template<int dim>
	void PeriodicityConstraints<dim>::declare_params(
	    ParameterReader & param_reader)
	{
	}

	/**
	 * This Function incorporates writes the constraints for
	 *  periodic boundarz conditions into the ConstraintMatrix
	 *  constraint_matrix and closes it.
	 *
	 */
	template<int dim>
	void PeriodicityConstraints<dim>::MakeConstraints(
	    const DOpEWrapper::DoFHandler<dim, dealii::DoFHandler<dim> > & dof_handler,
	    dealii::ConstraintMatrix& constraint_matrix) const
	{
		/* Does not work on locally refined grids. We can only couple
		 * dofs on a rectangular boundary.
		 * We couple boundary_color 0 with 1 (in x direction) and boundary color 2
		*with 3(in y direction)
		*/
		constraint_matrix.clear();
		DoFTools::make_hanging_node_constraints(
		    static_cast<const dealii::DoFHandler<dim>&>(dof_handler),
		    constraint_matrix);
		/****************************************************************************/
		unsigned int n_components = dof_handler.get_fe().n_components();
		unsigned int n_dofs = dof_handler.n_dofs();
		//get support points on the faces...make sure they exist
		assert(dof_handler.get_fe().has_face_support_points());
		const std::vector<Point<dim - 1> > &face_unit_support_points =
		    dof_handler.get_fe().get_unit_face_support_points();

		//then make a quadrature-rule with them
		dealii::Quadrature < dim - 1 > quadrature_formula(face_unit_support_points);
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

		//first loop over all cells...
		for (typename DOpEWrapper::DoFHandler<dim, dealii::DoFHandler<dim> >::active_cell_iterator cell =
		    dof_handler.begin_active(); cell != dof_handler.end(); ++cell)
		{
			//...then loop over all faces.
			for (unsigned int face = 0;
			    face < dealii::GeometryInfo < dim > ::faces_per_cell; ++face)
			    {
				int boundary_indicator = cell->face(face)->boundary_indicator();
				// Proceed only if the boundary indicator is lower than 4 and
        //if face is on a boundary and the corresponding boundary indicator is even
				if (cell->face(face)->at_boundary() && boundary_indicator < 4
				    && boundary_indicator % 2 == 0)
				{
					cell->face(face)->get_dof_indices(global_dof_indices);
					fe_face_values.reinit(cell, face);
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
		for (typename DOpEWrapper::DoFHandler<dim, dealii::DoFHandler<dim> >::active_cell_iterator cell =
		    dof_handler.begin_active(); cell != dof_handler.end(); ++cell)
		{
			for (unsigned int face = 0;
			    face < dealii::GeometryInfo < dim > ::faces_per_cell; ++face)
			    {
				int boundary_indicator = cell->face(face)->boundary_indicator();
        //Now loop over the remaining dofs, i.e. the ones with an odd boundary_indicator
				if (cell->face(face)->at_boundary() && boundary_indicator < 4
				    && boundary_indicator % 2 == 1)
				{
					cell->face(face)->get_dof_indices(global_dof_indices);
					fe_face_values.reinit(cell, face);
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
						for (; p != dof_locations.at(border_dim).at(actual_component).end();
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
								if (!IsElement(
								    p->first,
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
							    p
							        != dof_locations.at(border_dim).at(
							            dof_handler.get_fe().system_to_component_index(
							                q_point).first).end(),
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
				if (couplings.at(comp).at(i).size() > 0)//ansonsten schon in constraintmatrix geschrieben
					//oder der dof gehoert zu einer anderen componente
				{
					if (couplings.at(comp).at(i).size() == 1) //also normale constraints
					{
						assert (couplings.at(comp).at(couplings.at(comp).at(i)[0])[0] == i);
						//falls also dof i mit j verknuepft sein soll, dof j aber nicht mit i!

						constraint_matrix.add_line(i);
						constraint_matrix.add_entry(i, couplings.at(comp).at(i)[0],
						                                   1.0);
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
		constraint_matrix.close();
	}

//	template<int dim>
//	void PeriodicityConstraints<dim>::MakeConstraints(
//	    const DOpEWrapper::HpDoFHandler<dim> & dof_handler,
//	    dealii::ConstraintMatrix& constraint_matrix) const
//	{
//		throw DOpEException("Not Implemented", "PeriodicityConstraints::MakeConstraints(hp)");
//	}
}

#endif /* MYCONSTRAINTSMAKER_H_ */
