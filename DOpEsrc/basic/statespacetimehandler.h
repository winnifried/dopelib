/**
 *
 * Copyright (C) 2012-2018 by the DOpElib authors
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

#ifndef STATE_SPACE_TIME_HANDLER_H_
#define STATE_SPACE_TIME_HANDLER_H_

#include <basic/spacetimehandler_base.h>
#include <interfaces/active_fe_index_setter_interface.h>
#include <wrapper/dataout_wrapper.h>
#include <wrapper/mapping_wrapper.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_vector_base.h>
#include <deal.II/lac/block_vector.h>
#if DEAL_II_VERSION_GTE(9,1,1)
#include <deal.II/lac/affine_constraints.h>
#else
#include <deal.II/lac/constraint_matrix.h>
#endif
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
//#include <deal.II/multigrid/mg_dof_handler.h>
//#include <deal.II/multigrid/mg_constrained_dofs.h>

#include <vector>
#include <iostream>
#include <sstream>

namespace DOpE
{
#if DEAL_II_VERSION_GTE(9,3,0)
  /**
   * Interface to the dimension depended functionality of a
   * StateSpaceTimeDoFHandler.
   *
   * @tparam <FE>               The finite element type we use (i.e. 'normal' finite elements vs. hp::FECollections)
   * @tparam <DH>       false for normal, true for HP-DoFhandler
   * @tparam <SPARSITYPATTERN>  The sparsity pattern for control & state. This is needed as a class template, because
   *                            member function templates are not allowed for virtual member functions.
   * @tparam <VECTOR>           The vector type for control & state (i.e. dealii::Vector<double> or dealii::BlockVector<double>)
   * @tparam<dealdim>           The dimension for the state variable. This is the dimension the
   *                            mesh is in.
   */
  template<template<int, int> class FE, bool DH, typename SPARSITYPATTERN,
           typename VECTOR, int dealdim>
#else
  /**
   * Interface to the dimension depended functionality of a
   * StateSpaceTimeDoFHandler.
   *
   * @tparam <FE>               The finite element type we use (i.e. 'normal' finite elements vs. hp::FECollections)
   * @tparam <DH>       The dofhandler type we use (i.e. 'normal' dofhandler vs. hp::dofhandler)
   * @tparam <SPARSITYPATTERN>  The sparsity pattern for control & state. This is needed as a class template, because
   *                            member function templates are not allowed for virtual member functions.
   * @tparam <VECTOR>           The vector type for control & state (i.e. dealii::Vector<double> or dealii::BlockVector<double>)
   * @tparam<dealdim>           The dimension for the state variable. This is the dimension the
   *                            mesh is in.
   */
  template<template<int, int> class FE, template<int, int> class DH, typename SPARSITYPATTERN,
           typename VECTOR, int dealdim>
#endif
  class StateSpaceTimeHandler : public SpaceTimeHandlerBase<VECTOR>
  {
  public:
    StateSpaceTimeHandler() :
      SpaceTimeHandlerBase<VECTOR>()
    {
      domain_dofhandler_vector_.resize (1);
    }
    StateSpaceTimeHandler(dealii::Triangulation<1> &times) :
      SpaceTimeHandlerBase<VECTOR>(times)
    {
      domain_dofhandler_vector_.resize (1);
    }
    StateSpaceTimeHandler(
      const ActiveFEIndexSetterInterface<dealdim> &index_setter) :
      SpaceTimeHandlerBase<VECTOR>(), fe_index_setter_(&index_setter)
    {
      domain_dofhandler_vector_.resize (1);
    }
    StateSpaceTimeHandler (dealii::Triangulation<1> &times,
                           const ActiveFEIndexSetterInterface<dealdim> &index_setter) :
      SpaceTimeHandlerBase<VECTOR>(times), fe_index_setter_(&index_setter)
    {
      domain_dofhandler_vector_.resize (1);
    }
    virtual
    ~StateSpaceTimeHandler ()
    {
    }

    /**
     * Initializes the dof handlers corresponding to the finite elements.
     *
     * @param state_n_blocks            Number of Blocks for the state variable
     * @param state_block_components    Component to Block mapping for the state
     * @param DD                        Description of the DirichletBoundaries
     */
    virtual void
    ReInit (unsigned int state_n_blocks,
            const std::vector<unsigned int> &state_block_component,
            const DirichletDescriptor &DD) = 0;

    /******************************************************/

    /**
     * Returns a reference to the DoF Handler for the State at the current time point.
     */
#if DEAL_II_VERSION_GTE(9,3,0)
    virtual const DOpEWrapper::DoFHandler<dealdim> &
#else
    virtual const DOpEWrapper::DoFHandler<dealdim, DH> &
#endif
      GetStateDoFHandler (unsigned int /*time_point*/= std::numeric_limits<unsigned int>::max()) const =0;

    /******************************************************/

    /**
     * Returns a reference to the Mapping in use.
     */
    virtual const DOpEWrapper::Mapping<dealdim, DH> &
    GetMapping () const = 0;

    /******************************************************/

    /**
     * Returns a reference to a vector of DoFHandlers, the order of the DoFHandlers must
     * be set prior by SetDoFHandlerOrdering
     */
#if DEAL_II_VERSION_GTE(9,3,0)
    const std::vector<const DOpEWrapper::DoFHandler<dealdim>*> &
#else
    const std::vector<const DOpEWrapper::DoFHandler<dealdim, DH>*> &
#endif
    GetDoFHandler (unsigned int time_point = std::numeric_limits<unsigned int>::max()) const
    {
      domain_dofhandler_vector_[0] = &GetStateDoFHandler (time_point);
      return domain_dofhandler_vector_;
    }

    /******************************************************/

    /**
     * Returns a vector of the begin_active-celliterators of the
     * DoFHandlers in use.
     */
    std::vector<
#if DEAL_II_VERSION_GTE(9,3,0)
      typename DOpEWrapper::DoFHandler<dealdim>::active_cell_iterator>
#else
      typename DOpEWrapper::DoFHandler<dealdim, DH>::active_cell_iterator>
#endif
    GetDoFHandlerBeginActive () const
    {
      std::vector<
#if DEAL_II_VERSION_GTE(9,3,0)
	typename DOpEWrapper::DoFHandler<dealdim>::active_cell_iterator> ret (
#else
	typename DOpEWrapper::DoFHandler<dealdim, DH>::active_cell_iterator> ret (
#endif
        this->GetDoFHandler ().size ());
      for (unsigned int dh = 0; dh < this->GetDoFHandler ().size (); dh++)
        {
          ret[dh] = this->GetDoFHandler ()[dh]->begin_active ();
        }
      return ret;
    }

    /******************************************************/

    /**
     * Returns a vector of the end-celliterators of the
     * DoFHandlers in use.
     */

    std::vector<
#if DEAL_II_VERSION_GTE(9,3,0)
	typename DOpEWrapper::DoFHandler<dealdim>::active_cell_iterator>
#else
	typename DOpEWrapper::DoFHandler<dealdim, DH>::active_cell_iterator>
#endif
    GetDoFHandlerEnd () const
    {
      std::vector<
#if DEAL_II_VERSION_GTE(9,3,0)
	typename DOpEWrapper::DoFHandler<dealdim>::active_cell_iterator> ret (
#else
	typename DOpEWrapper::DoFHandler<dealdim, DH>::active_cell_iterator> ret (
#endif
        this->GetDoFHandler ().size ());
      for (unsigned int dh = 0; dh < this->GetDoFHandler ().size (); dh++)
        {
          ret[dh] = this->GetDoFHandler ()[dh]->end ();
        }
      return ret;
    }


    /******************************************************/

    /**
     * Experimental status:
     * Returns a vector of the begin-celliterators of the
     * DoFHandlers in use.
     * Iterator for multigrid's matrix assembling running
     * over all cells on all levels.
     */
    std::vector<
#if DEAL_II_VERSION_GTE(9,3,0)
	typename DOpEWrapper::DoFHandler<dealdim>::cell_iterator>
#else
	typename DOpEWrapper::DoFHandler<dealdim, DH>::cell_iterator>
#endif
    GetDoFHandlerBeginActiveAllLevels () const
    {
      std::vector<
#if DEAL_II_VERSION_GTE(9,3,0)
	typename DOpEWrapper::DoFHandler<dealdim>::cell_iterator> ret (
#else
	typename DOpEWrapper::DoFHandler<dealdim, DH>::cell_iterator> ret (
#endif
        this->GetDoFHandler ().size ());
      for (unsigned int dh = 0; dh < this->GetDoFHandler ().size (); dh++)
        {
          ret[dh] = this->GetDoFHandler ()[dh]->begin_active ();
        }
      return ret;
    }

    /******************************************************/

    /**
     * Experimental status:
     * Returns a vector of the end-celliterators of the
     * DoFHandlers in use.
     * Iterator for multigrid's matrix assembling running
     * over all cells on all levels.
     */

    std::vector<
#if DEAL_II_VERSION_GTE(9,3,0)
	typename DOpEWrapper::DoFHandler<dealdim>::cell_iterator>
#else
	typename DOpEWrapper::DoFHandler<dealdim, DH>::cell_iterator>
#endif
    GetDoFHandlerEndAllLevels () const
    {
      std::vector<
#if DEAL_II_VERSION_GTE(9,3,0)
	typename DOpEWrapper::DoFHandler<dealdim>::cell_iterator> ret (
#else
	typename DOpEWrapper::DoFHandler<dealdim, DH>::cell_iterator> ret (
#endif
        this->GetDoFHandler ().size ());
      for (unsigned int dh = 0; dh < this->GetDoFHandler ().size (); dh++)
        {
          ret[dh] = this->GetDoFHandler ()[dh]->end ();
        }
      return ret;
    }



    /******************************************************/

    /**
     * Returns the order of the StateDofHandler set by SetDoFHandlerOrdering.
     *
     */
    unsigned int
    GetStateIndex ()
    {
      return 0;
    }

    /******************************************************/
    /**
     * Returns a const reference to the ActiveFEIndexSetterInterface object stored in
     * this class. This function is only useful in the hp case.
     */
    const ActiveFEIndexSetterInterface<dealdim> &
    GetFEIndexSetter () const
    {
      //makes only sense in the hp case.
      return *fe_index_setter_;
    }

    /******************************************************/
    /*
     * This function sets for every cell the right fe index for the state variable.
     * This is only useful in the hp case!
     *
     * @param dof_handler   The dof_handler for which the fe indices have to be set.
     */
    void
    SetActiveFEIndicesState(
#if DEAL_II_VERSION_GTE(9,3,0)
      DOpEWrapper::DoFHandler<dealdim> &dof_handler)
#else
      DOpEWrapper::DoFHandler<dealdim, DH> &dof_handler)
#endif
    {
      if (std::is_same<FE<dealdim,dealdim>,dealii::hp::FECollection<dealdim,dealdim> >::value)
        {
          for (auto element =
                 dof_handler.begin_active(); element != dof_handler.end(); ++element)
            {
              this->GetFEIndexSetter ().SetActiveFEIndexState (element);
            }
        }
    }


    /**
     * Returns the locally owned DoFs for the given type of vector at given time point.
     *
     * @ param type Indicates for which quantity (state, constrol, constraint, local constraint)
     * we want to know the number of DoFs per block.
     * @ param time_point Indicating the time at which we want to know the DoFs. -1 means now.
     */
    virtual dealii::IndexSet
    GetLocallyOwnedDoFs (const DOpEtypes::VectorType type,
                         unsigned int time_point = std::numeric_limits<unsigned int>::max()) const
    {
       switch (type)
        {
        case DOpEtypes::VectorType::state:
          return GetStateDoFHandler (time_point).GetDEALDoFHandler().locally_owned_dofs();

        case DOpEtypes::VectorType::constraint:
        case DOpEtypes::VectorType::local_constraint:
        case DOpEtypes::VectorType::control:
          assert(false);
          return dealii::IndexSet ();

        default:
          abort ();
          return dealii::IndexSet ();
        } 
    }

    /**
     * Returns the locally relevant DoFs for the given type of vector at given time point.
     *
     * @ param type Indicates for which quantity (state, constrol, constraint, local constraint)
     * we want to know the number of DoFs per block.
     * @ param time_point Indicating the time at which we want to know the DoFs. -1 means now.
     */
    virtual dealii::IndexSet
    GetLocallyRelevantDoFs (const DOpEtypes::VectorType type,
                            unsigned int time_point = std::numeric_limits<unsigned int>::max()) const
    {
      switch (type)
      {
      case DOpEtypes::VectorType::state:
      {
	dealii::IndexSet result;
	DoFTools::extract_locally_relevant_dofs (GetStateDoFHandler (time_point).GetDEALDoFHandler(),
						 result);
	return result;
      }

      case DOpEtypes::VectorType::constraint:
      case DOpEtypes::VectorType::local_constraint:
      case DOpEtypes::VectorType::control:
	assert(false);
	return IndexSet ();
	
        default:
          abort ();
          return dealii::IndexSet ();
      } 
    }
    
    /******************************************************/
    /*
     //  * Experimental status for MG prec
     //  */
//  virtual const dealii::MGConstrainedDoFs &
//    GetMGConstrainedDoFs() const
//    {
//      throw DOpEException(
//                "Not used for normal DofHandler",
//                "StateSpaceTimeHandler.h");
//
//    }

    /******************************************************/
    /**
     * Get the triangulation.
     */
    virtual const dealii::Triangulation<dealdim> &
    GetTriangulation () const
    {
      throw DOpEException(
        "Not used for normal DofHandler",
        "StateSpaceTimeHandler.h");

    }

    /******************************************************/

    /**
     * Returns the state DoF-Constraints at the current time
     */
#if DEAL_II_VERSION_GTE(9,1,1)
    virtual const dealii::AffineConstraints<double> &
    GetStateDoFConstraints (unsigned int /*time_point*/= std::numeric_limits<unsigned int>::max()) const=0;
#else
    virtual const dealii::ConstraintMatrix &
    GetStateDoFConstraints (unsigned int /*time_point*/= std::numeric_limits<unsigned int>::max()) const=0;
#endif
    /******************************************************/

    /**
     * Returns the state HN-Constraints at the current time
     */
#if DEAL_II_VERSION_GTE(9,1,1)
    virtual const dealii::AffineConstraints<double>
    &
    GetStateHNConstraints (unsigned int /*time_point*/= std::numeric_limits<unsigned int>::max()) const=0;
#else
    virtual const dealii::ConstraintMatrix
    &
    GetStateHNConstraints (unsigned int /*time_point*/= std::numeric_limits<unsigned int>::max()) const=0;
#endif
    /*******************************************************/

    /**
     * Returns a Reference to a vector of points where the FEs have their support points.
     * on the current spatial mesh (if they do have that compare dealii::DoFTools>>map_dofs_to_support_points!).
     */
    virtual const std::vector<dealii::Point<dealdim> > &
    GetMapDoFToSupportPoints (unsigned int /*time_point*/= std::numeric_limits<unsigned int>::max())=0;

    /**
     * Returns the list of the number of neighbouring elements to the vertices
     */

    virtual const std::vector<unsigned int>* GetNNeighbourElements(unsigned int /*time_point*/= std::numeric_limits<unsigned int>::max()) = 0;
        
    /******************************************************/

    /**
     * Computes the current sparsity pattern for the state variable
     */
    virtual void
      ComputeStateSparsityPattern (SPARSITYPATTERN &sparsity, unsigned int /*time_point*/= std::numeric_limits<unsigned int>::max()) const=0;

    /******************************************************/

//        /**
//   * Experimental status:
//         * Needed for MG prec.
//         */
//        virtual void
//    ComputeMGStateSparsityPattern(dealii::MGLevelObject<dealii::BlockSparsityPattern> & /*mg_sparsity_patterns*/,
//          unsigned int /*n_levels*/) const
//  {
//     throw DOpEException(
//                "Not used for normal DofHandler",
//                "StateSpaceTimeHandler.h");
//  }
//
//  /******************************************************/
//
//        /**
//         * Experimental status:
//         * Needed for MG prec.
//         */
//        virtual void
//    ComputeMGStateSparsityPattern(dealii::MGLevelObject<dealii::SparsityPattern> & /*mg_sparsity_patterns*/,
//          unsigned int /*n_levels*/) const
//  {
//     throw DOpEException(
//                "Not used for normal DofHandler",
//                "StateSpaceTimeHandler.h");
//  }


    /******************************************************/
    /**
     * Returns a const Reference to the FESystem indicated by the string 'name', i.e. state oder control.
     */

    virtual const FE<dealdim, dealdim> &
    GetFESystem (std::string name) const=0;

    /******************************************************/

#if DEAL_II_VERSION_GTE(9,3,0)
    DOpEWrapper::DataOut<dealdim> &
#else
    DOpEWrapper::DataOut<dealdim, DH> &
#endif
    GetDataOut ()
    {
      data_out_.clear ();
      return data_out_;
    }

    /**
     * Implementation of virtual function in SpaceTimeHandlerBase
     */
    virtual void
    WriteToFile (const VECTOR &v,
                 std::string name,
                 std::string outfile,
                 std::string dof_type,
                 std::string filetype);

    virtual void
    WriteToFileElementwise(const Vector<float> &v,
			   std::string name,
                           std::string outfile,
			   std::string dof_type,
			   std::string filetype,
			   int n_patches);

  protected:
    //we need this here, because we know the type of the DoFHandler in use.
    //This saves us a template argument for statpdeproblem etc.
#if DEAL_II_VERSION_GTE(9,3,0)
    DOpEWrapper::DataOut<dealdim> data_out_;
#else
    DOpEWrapper::DataOut<dealdim, DH> data_out_;
#endif
    const ActiveFEIndexSetterInterface<dealdim> *fe_index_setter_ = NULL;
#if DEAL_II_VERSION_GTE(9,3,0)
    mutable std::vector<const DOpEWrapper::DoFHandler<dealdim>*> domain_dofhandler_vector_;
#else
    mutable std::vector<const DOpEWrapper::DoFHandler<dealdim, DH>*> domain_dofhandler_vector_;
#endif

  };

#if DEAL_II_VERSION_GTE(9,3,0)
  template <template <int, int> class FE, bool DH,
            typename SPARSITYPATTERN, typename VECTOR, int dealdim>
#else
  template <template <int, int> class FE, template <int, int> class DH,
            typename SPARSITYPATTERN, typename VECTOR, int dealdim>
#endif
    void
    StateSpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR, dealdim>::WriteToFile (
      const VECTOR &v,
      std::string name,
      std::string outfile,
      std::string dof_type,
      std::string filetype)
  {
    // TODO remove MPI_COMM_WORLD
    const bool parallel = dealii::Utilities::MPI::n_mpi_processes (
                            MPI_COMM_WORLD)
                          > 1;

    if (dof_type == "state")
      {
        auto &data_out = GetDataOut ();
        data_out.attach_dof_handler (GetStateDoFHandler ());

#if DEAL_II_VERSION_GTE(9,3,0)
#if DEAL_II_VERSION_GTE(9,3,1)
	data_out.add_data_vector (v, name,DataOut_DoFData<dealdim,dealdim>::DataVectorType::type_dof_data);
#else
	data_out.add_data_vector (v, name,DataOut_DoFData<dealii::DoFHandler<dealdim,dealdim>,dealdim,dealdim>::DataVectorType::type_dof_data);
#endif
#else
	data_out.add_data_vector (v, name,DataOut_DoFData<DH<dealdim,dealdim>,dealdim,dealdim>::DataVectorType::type_dof_data);
#endif
        data_out.build_patches ();
        // From statpdeproblem.h:
        // TODO: mapping[0] is a workaround, as deal does not support interpolate
        // boundary_values with a mapping collection at this point.
        // data_out.build_patches(GetMapping()[0]);

        std::string _outfile = outfile;

        if (parallel)
          {
            Assert(filetype == ".vtu", ExcNotImplemented());
            const unsigned int rank = Utilities::MPI::this_mpi_process (
                                        MPI_COMM_WORLD);
            _outfile = dealii::Utilities::replace_in_string (outfile, ".vtu",
                                                             "_" + dealii::Utilities::int_to_string (rank) + ".vtu");
          }

        std::ofstream output (_outfile.c_str ());

        if (filetype == ".vtk")
          {
            data_out.write_vtk (output);
          }
        else if (filetype == ".vtu")
          {
            data_out.write_vtu (output);
          }
        else if (filetype == ".gpl")
          {
            data_out.write_gnuplot (output);
          }
        else
          {
            throw DOpEException (
              "Don't know how to write filetype `" + filetype + "'!",
              "StateSpaceTimeHandler::WriteToFile");
          }

        // In parallel computation, one cpu has to write a master file,
        // containing information about all files from other cpus.
        if (parallel)
          {
            if (Utilities::MPI::this_mpi_process (MPI_COMM_WORLD) == 0)
              {
                std::vector<std::string> filenames;
                for (unsigned int i = 0;
                     i < Utilities::MPI::n_mpi_processes (MPI_COMM_WORLD); ++i)
                  {
                    // TODO this is a hacky solution ... should be improved
                    std::string filename =
                      dealii::Utilities::replace_in_string (outfile, ".vtu",
                                                            "_" + dealii::Utilities::int_to_string (i)
                                                            + ".vtu");
                    filenames.push_back (filename);
                  }
                std::string master_name =
                  dealii::Utilities::replace_in_string (outfile, ".vtu",
                                                        ".pvtu");
                std::ofstream master_output (master_name.c_str ());
                data_out.write_pvtu_record (master_output, filenames);
              }
          }

        data_out.clear ();
      }
    else
      {
        throw DOpEException ("No such DoFHandler `" + dof_type + "'!",
                             "StateSpaceTimeHandler::WriteToFile");
      }
  }

#if DEAL_II_VERSION_GTE(9,3,0)
  template <template <int, int> class FE, bool DH, 
            typename SPARSITYPATTERN, typename VECTOR, int dealdim>
#else
  template <template <int, int> class FE, template <int, int> class DH,
            typename SPARSITYPATTERN, typename VECTOR, int dealdim>
#endif
  void
  StateSpaceTimeHandler<FE, DH, SPARSITYPATTERN, VECTOR, dealdim>::WriteToFileElementwise (
    const Vector<float> &v,
    std::string name,
    std::string outfile,
    std::string dof_type,
    std::string filetype,
    int n_patches)
  {
       if (dof_type == "state")
      {
        auto &data_out = GetDataOut ();
        data_out.attach_dof_handler(GetStateDoFHandler());

#if DEAL_II_VERSION_GTE(9,3,0)
#if DEAL_II_VERSION_GTE(9,3,1)
	data_out.add_data_vector(v, name,DataOut_DoFData<dealdim,dealdim>::DataVectorType::type_cell_data);
#else
	data_out.add_data_vector(v, name,DataOut_DoFData<dealii::DoFHandler<dealdim,dealdim>,dealdim,dealdim>::DataVectorType::type_cell_data);
#endif
#else
	data_out.add_data_vector(v, name,DataOut_DoFData<DH<dealdim,dealdim>,dealdim,dealdim>::DataVectorType::type_cell_data);
#endif
        data_out.build_patches(n_patches);

        std::ofstream output(outfile.c_str());

        if (filetype == ".vtk")
          {
            data_out.write_vtk(output);
          }
        else
          {
            throw DOpEException(
              "Don't know how to write filetype `" + filetype + "'!",
              "StateSpaceTimeHandler::WriteToFileElementwise");
          }
        data_out.clear();
      }
    else
      {
        throw DOpEException("No such DoFHandler `" + dof_type + "'!",
                            "StateSpaceTimeHandler::WriteToFileElementwise");
      }
  }


}

#endif
