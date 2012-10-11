/*
 * celldatacontainer.h
 *
 *  Created on: May 18, 2011
 *      Author: cgoll
 */

#ifndef CELLDATACONTAINER_H_
#define CELLDATACONTAINER_H_

#include "spacetimehandler.h"
#include "statespacetimehandler.h"
#include "fevalues_wrapper.h"
#include "dopeexception.h"
#include "celldatacontainer_internal.h"

#include <dofs/dof_handler.h>
#include <hp/dof_handler.h>

using namespace dealii;

namespace DOpE
{
  /**
   * Dummy Template Class, acts as kind of interface.
   * Through template specialization for DOFHANDLER, we
   * distinguish between the 'classic' and the 'hp' case.
   *
   * @template DOFHANDLER The type of the dealii-dofhandler we use in
   *                      our DOpEWrapper::DoFHandler, at the moment
   *                      DoFHandler and hp::DoFHandler.
   * @template VECTOR     Type of the vector we use in our computations
   *                      (i.e. Vector<double> or BlockVector<double>)
   * @template dim        The dimension of the integral we are actually
   *                      interested in.
   */

  template<typename DOFHANDLER, typename VECTOR, int dim>
    class CellDataContainer : public cdcinternal::CellDataContainerInternal<
        VECTOR, dim>
    {
      public:
        CellDataContainer()
        {
          throw(DOpE::DOpEException(
              "Dummy class, this constructor should never get called.",
              "CellDataContainer<dealii::DoFHandler<dim> , VECTOR, dim>::CellDataContainer"));
        }
        ;
    };

  /**
   * This two classes hold all the information we need in the integrator to
   * integrate something over a cell (could be a functional, a PDE, etc.).
   * Of particular importance: This class holds the FEValues objects.
   *
   * @template VECTOR     Type of the vector we use in our computations (i.e. Vector<double> or BlockVector<double>)
   * @template dim        The dimension of the integral we are actually interested in.
   */

  template<typename VECTOR, int dim>
    class CellDataContainer<dealii::DoFHandler<dim>, VECTOR, dim> : public cdcinternal::CellDataContainerInternal<
        VECTOR, dim>
    {

      public:
        /**
         * Constructor. Initializes the FEValues objects.
         *
         * @template FE                   Type of the finite element in use. Must be compatible with dealii::DofHandler. //TODO Should we fix this?
         * @template SPARSITYPATTERN      The corresponding Sparsitypattern to the class-template VECTOR.
         * @template dopedim              The dimension of the control variable.
         * @template dealdim              The dimension of the state variable.
         *
         * @param quad                    Reference to the quadrature-rule which we use at the moment.
         * @param update_flags            The update flags we need to initialize the FEValues obejcts
         * @param sth                     A reference to the SpaceTimeHandler in use.
         * @param cell                    A vector of cell iterators through which we gain most of the needed information (like
         *                                material_ids, n_dfos, etc.)
         * @param param_values            A std::map containing parameter data (e.g. non space dependent data). If the control
         *                                is done by parameters, it is contained in this map at the position "control".
         * @param domain_values           A std::map containing domain data (e.g. nodal vectors for FE-Functions). If the control
         *                                is distributed, it is contained in this map at the position "control". The state may always
         *                                be found in this map at the position "state"
         *
         */
        template<typename FE, typename SPARSITYPATTERN, int dopedim, int dealdim>
          CellDataContainer(const Quadrature<dim>& quad,
              UpdateFlags update_flags,
              SpaceTimeHandler<FE, dealii::DoFHandler<dim>, SPARSITYPATTERN,
                  VECTOR, dopedim, dealdim>& sth,
              const std::vector<
                  typename dealii::DoFHandler<dim>::active_cell_iterator>& cell,
              const std::map<std::string, const Vector<double>*> &param_values,
              const std::map<std::string, const VECTOR*> &domain_values)
              : cdcinternal::CellDataContainerInternal<VECTOR, dim>(
                  param_values, domain_values), _cell(cell), _state_fe_values(
                  sth.GetMapping(), (sth.GetFESystem("state")), quad,
                  update_flags), _control_fe_values(sth.GetMapping(),
                  (sth.GetFESystem("control")), quad, update_flags)
          {
            _state_index = sth.GetStateIndex();
            if (_state_index == 1)
              _control_index = 0;
            else
              _control_index = 1;
            _n_q_points_per_cell = quad.size();
            _n_dofs_per_cell = cell[0]->get_fe().dofs_per_cell;
          }

        /**
         * Constructor. Initializes the FEValues objects. When only a PDE is used.
         *
         * @template FE                   Type of the finite element in use.
         * @template SPARSITYPATTERN      The corresponding Sparsitypattern to the class-template VECTOR.
         *
         * @param quad                    Reference to the quadrature-rule which we use at the moment.
         * @param update_flags            The update flags we need to initialize the FEValues obejcts
         * @param sth                     A reference to the SpaceTimeHandler in use.
         * @param cell                    A vector of cell iterators through which we gain most of the needed information (like
         *                                material_ids, n_dfos, etc.)
         * @param param_values            A std::map containing parameter data (e.g. non space dependent data). If the control
         *                                is done by parameters, it is contained in this map at the position "control".
         * @param domain_values           A std::map containing domain data (e.g. nodal vectors for FE-Functions). If the control
         *                                is distributed, it is contained in this map at the position "control". The state may always
         *                                be found in this map at the position "state"
         *
         */
        template<typename FE, typename SPARSITYPATTERN>
          CellDataContainer(const Quadrature<dim>& quad,
              UpdateFlags update_flags,
              StateSpaceTimeHandler<FE, dealii::DoFHandler<dim>,
                  SPARSITYPATTERN, VECTOR, dim>& sth,
              const std::vector<
                  typename dealii::DoFHandler<dim>::active_cell_iterator>& cell,
              const std::map<std::string, const Vector<double>*> &param_values,
              const std::map<std::string, const VECTOR*> &domain_values)
              : cdcinternal::CellDataContainerInternal<VECTOR, dim>(
                  param_values, domain_values), _cell(cell), _state_fe_values(
                  sth.GetMapping(), (sth.GetFESystem("state")), quad,
                  update_flags), _control_fe_values(sth.GetMapping(),
                  (sth.GetFESystem("state")), quad, update_flags)
          {
            _state_index = sth.GetStateIndex();
            _control_index = cell.size(); //Make sure they are never used ...
            _n_q_points_per_cell = quad.size();
            _n_dofs_per_cell = cell[0]->get_fe().dofs_per_cell;
          }
        ~CellDataContainer()
        {
        }
        /*********************************************/
        /*
         * This function reinits the FEValues on the actual cell. Should
         * be called prior to any of the get-functions.
         */
        inline void
        ReInit();

        /*********************************************/
        /**
         * Get functions to extract data. They all assume that ReInit
         * is executed before calling them. Self explanatory.
         */
        inline unsigned int
        GetNDoFsPerCell() const;
        inline unsigned int
        GetNQPoints() const;
        inline unsigned int
        GetMaterialId() const;
        inline unsigned int
        GetNbrMaterialId(unsigned int face) const;
        inline bool
        GetIsAtBoundary() const;
        inline double
        GetCellDiameter() const;
        inline const DOpEWrapper::FEValues<dim>&
        GetFEValuesState() const;
        inline const DOpEWrapper::FEValues<dim>&
        GetFEValuesControl() const;
      private:
        /*
         * Helper Functions
         */
        unsigned int
        GetStateIndex() const;
        unsigned int
        GetControlIndex() const;

        /***********************************************************/
        //"global" member data, part of every instantiation
        unsigned int _state_index;
        unsigned int _control_index;

        const std::vector<typename dealii::DoFHandler<dim>::active_cell_iterator> & _cell;
        DOpEWrapper::FEValues<dim> _state_fe_values;
        DOpEWrapper::FEValues<dim> _control_fe_values;

        unsigned int _n_q_points_per_cell;
        unsigned int _n_dofs_per_cell;
    };

  template<typename VECTOR, int dim>
    class CellDataContainer<dealii::hp::DoFHandler<dim>, VECTOR, dim> : public cdcinternal::CellDataContainerInternal<
        VECTOR, dim>
    {

      public:
        /**
         * Constructor. Initializes the hp_fe_falues objects.
         *
         * @template FE                   Type of the finite element in use. Must be compatible with hp::DoFHandler. //TODO Should we fix this?
         * @template SPARSITYPATTERN      The corresponding Sparsitypattern to the class-template VECTOR.
         * @template dopedim              The dimension of the control variable.
         * @template dealdim              The dimension of the state variable.
         *
         * @param quad                    Reference to the qcollection-rule which we use at the moment.
         * @param update_flags            The update flags we need to initialize the FEValues obejcts
         * @param sth                     A reference to the SpaceTimeHandler in use.
         * @param cell                    A vector of cell iterators through which we gain most of the needed information (like
         *                                material_ids, n_dfos, etc.)
         * @param param_values            A std::map containing parameter data (e.g. non space dependent data). If the control
         *                                is done by parameters, it is contained in this map at the position "control".
         * @param domain_values           A std::map containing domain data (e.g. nodal vectors for FE-Functions). If the control
         *                                is distributed, it is contained in this map at the position "control". The state may always
         *                                be found in this map at the position "state"
         *
         */
        template<typename FE, typename SPARSITYPATTERN, int dopedim, int dealdim>
          CellDataContainer(const hp::QCollection<dim>& q_collection,
              UpdateFlags update_flags,
              SpaceTimeHandler<FE, dealii::hp::DoFHandler<dim>, SPARSITYPATTERN,
                  VECTOR, dopedim, dealdim>& sth,
              const std::vector<
                  typename DOpEWrapper::DoFHandler<dim,
                      dealii::hp::DoFHandler<dim> >::active_cell_iterator>& cell,
              const std::map<std::string, const Vector<double>*> &param_values,
              const std::map<std::string, const VECTOR*> &domain_values)
              : cdcinternal::CellDataContainerInternal<VECTOR, dim>(
                  param_values, domain_values), _cell(cell), _state_hp_fe_values(
                  sth.GetMapping(), (sth.GetFESystem("state")), q_collection,
                  update_flags), _control_hp_fe_values(sth.GetMapping(),
                  (sth.GetFESystem("control")), q_collection, update_flags), _q_collection(
                  q_collection)
          {
            _state_index = sth.GetStateIndex();
            if (_state_index == 1)
              _control_index = 0;
            else
              _control_index = 1;
          }

        /**
         * Constructor. Initializes the hp_fe_falues objects. For PDE only.
         *
         * @template SPARSITYPATTERN      The corresponding Sparsitypattern to the class-template VECTOR.
         *
         * @param quad                    Reference to the qcollection-rule which we use at the moment.
         * @param update_flags            The update flags we need to initialize the FEValues obejcts
         * @param sth                     A reference to the SpaceTimeHandler in use.
         * @param cell                    A vector of cell iterators through which we gain most of the needed information (like
         *                                material_ids, n_dfos, etc.)
         * @param param_values            A std::map containing parameter data (e.g. non space dependent data). If the control
         *                                is done by parameters, it is contained in this map at the position "control".
         * @param domain_values           A std::map containing domain data (e.g. nodal vectors for FE-Functions). If the control
         *                                is distributed, it is contained in this map at the position "control". The state may always
         *                                be found in this map at the position "state"
         *
         */
        template<typename FE, typename SPARSITYPATTERN>
          CellDataContainer(const hp::QCollection<dim>& q_collection,
              UpdateFlags update_flags,
              StateSpaceTimeHandler<FE, dealii::hp::DoFHandler<dim>,
                  SPARSITYPATTERN, VECTOR, dim>& sth,
              const std::vector<
                  typename DOpEWrapper::DoFHandler<dim,
                      dealii::hp::DoFHandler<dim> >::active_cell_iterator>& cell,
              const std::map<std::string, const Vector<double>*> &param_values,
              const std::map<std::string, const VECTOR*> &domain_values)
              : cdcinternal::CellDataContainerInternal<VECTOR, dim>(
                  param_values, domain_values), _cell(cell), _state_hp_fe_values(
                  sth.GetMapping(), (sth.GetFESystem("state")), q_collection,
                  update_flags), _control_hp_fe_values(sth.GetMapping(),
                  (sth.GetFESystem("state")), q_collection, update_flags), _q_collection(
                  q_collection)
          {
            _state_index = sth.GetStateIndex();
            _control_index = cell.size(); //Make sure they are never used ...
          }
        ~CellDataContainer()
        {
        }

        /*****************************************************************/
        /*
         * This function reinits the hp::FEValues on the actual cell. Should
         * be called prior to any of the get-functions.
         */
        inline void
        ReInit();
        /*********************************************/
        /**
         * Get functions to extract data. They all assume that ReInit
         * is executed before calling them. Self explanatory.
         */

        inline unsigned int
        GetNDoFsPerCell() const;
        inline unsigned int
        GetNQPoints() const;
        inline unsigned int
        GetMaterialId() const;
        inline unsigned int
        GetNbrMaterialId(unsigned int face) const;
        inline bool
        GetIsAtBoundary() const;
        inline double
        GetCellDiameter() const;

        inline const DOpEWrapper::FEValues<dim>&
        GetFEValuesState() const;
        inline const DOpEWrapper::FEValues<dim>&
        GetFEValuesControl() const;

      private:
        unsigned int
        GetStateIndex() const;
        unsigned int
        GetControlIndex() const;
        const std::map<std::string, const VECTOR*> &
        GetDomainValues() const;
        /***********************************************************/
        /**
         * Helper Function.
         * Hier koennte man ueber ein Template nachdenken.
         */
        inline void
        GetValues(const DOpEWrapper::FEValues<dim>& fe_values, std::string name,
            std::vector<double>& values) const;
        /***********************************************************/
        /**
         * Helper Function. Vector valued case.
         */
        inline void
        GetValues(const DOpEWrapper::FEValues<dim>& fe_values, std::string name,
            std::vector<Vector<double> >& values) const;

        /***********************************************************/
        /**
         * Helper Function.
         */
        template<int targetdim>
          inline void
          GetGrads(const DOpEWrapper::FEValues<dim>& fe_values,
              std::string name,
              std::vector<Tensor<1, targetdim> >& values) const;
        /***********************************************************/
        /**
         * Helper Function. Vector valued case.
         */
        template<int targetdim>
          inline void
          GetGrads(const DOpEWrapper::FEValues<dim>& fe_values,
              std::string name,
              std::vector<std::vector<Tensor<1, targetdim> > >& values) const;

        /***********************************************************/
        //"global" member data, part of every instantiation
        unsigned int _state_index;
        unsigned int _control_index;

        const std::vector<
            typename dealii::hp::DoFHandler<dim>::active_cell_iterator> & _cell;
        DOpEWrapper::HpFEValues<dim> _state_hp_fe_values;
        DOpEWrapper::HpFEValues<dim> _control_hp_fe_values;

        const hp::QCollection<dim>& _q_collection;
    };

  /***********************************************************************/
  /************************IMPLEMENTATION for DoFHandler*********************************/
  /***********************************************************************/

  template<typename VECTOR, int dim>
    void
    DOpE::CellDataContainer<dealii::DoFHandler<dim>, VECTOR, dim>::ReInit()
    {
      _state_fe_values.reinit(_cell[this->GetStateIndex()]);
      //Make sure that the Control must be initialized.
      if (this->GetControlIndex() < _cell.size())
        _control_fe_values.reinit(_cell[this->GetControlIndex()]);
    }

  /***********************************************************************/
  template<typename VECTOR, int dim>
    unsigned int
    CellDataContainer<dealii::DoFHandler<dim>, VECTOR, dim>::GetNDoFsPerCell() const
    {
      return _n_dofs_per_cell;
    }

  /**********************************************/
  template<typename VECTOR, int dim>
    unsigned int
    CellDataContainer<dealii::DoFHandler<dim>, VECTOR, dim>::GetNQPoints() const
    {
      return _n_q_points_per_cell;
    }

  /**********************************************/
  template<typename VECTOR, int dim>
    unsigned int
    CellDataContainer<dealii::DoFHandler<dim>, VECTOR, dim>::GetMaterialId() const
    {
      return _cell[0]->material_id();
    }

  /**********************************************/
  template<typename VECTOR, int dim>
    unsigned int
    CellDataContainer<dealii::DoFHandler<dim>, VECTOR, dim>::GetNbrMaterialId(
        unsigned int face) const
    {
      if (_cell[0]->neighbor_index(face) != -1)
        return _cell[0]->neighbor(face)->material_id();
      else
        throw DOpEException("There is no neighbor with number " + face,
            "CellDataContainer::GetNbrMaterialId");
    }

  /**********************************************/
  template<typename VECTOR, int dim>
    bool
    CellDataContainer<dealii::DoFHandler<dim>, VECTOR, dim>::GetIsAtBoundary() const
    {
      return _cell[0]->at_boundary();
    }

  template<typename VECTOR, int dim>
    double
    CellDataContainer<dealii::DoFHandler<dim>, VECTOR, dim>::GetCellDiameter() const
    {
      return _cell[0]->diameter();
    }

  /**********************************************/
  template<typename VECTOR, int dim>
    const DOpEWrapper::FEValues<dim>&
    CellDataContainer<dealii::DoFHandler<dim>, VECTOR, dim>::GetFEValuesState() const
    {
      return _state_fe_values;
    }

  /**********************************************/
  template<typename VECTOR, int dim>
    const DOpEWrapper::FEValues<dim>&
    CellDataContainer<dealii::DoFHandler<dim>, VECTOR, dim>::GetFEValuesControl() const
    {
      return _control_fe_values;
    }

  /***********************************************************************/

  template<typename VECTOR, int dim>
    unsigned int
    CellDataContainer<dealii::DoFHandler<dim>, VECTOR, dim>::GetStateIndex() const
    {
      return _state_index;
    }

  /***********************************************************************/

  template<typename VECTOR, int dim>
    unsigned int
    CellDataContainer<dealii::DoFHandler<dim>, VECTOR, dim>::GetControlIndex() const
    {
      return _control_index;
    }

  /***********************************************************************/
  /************************END*OF*IMPLEMENTATION**************************/
  /***********************************************************************/
  /***********************************************************************/
  /************************IMPLEMENTATION*********************************/
  /***********************************************************************/

  template<typename VECTOR, int dim>
    void
    CellDataContainer<dealii::hp::DoFHandler<dim>, VECTOR, dim>::ReInit()
    {
      _state_hp_fe_values.reinit(_cell[this->GetStateIndex()]);
      //Make sure that the Control must be initialized.
      if (this->GetControlIndex() < _cell.size())
        _control_hp_fe_values.reinit(_cell[this->GetControlIndex()]);
    }
  /*********************************************/

  template<typename VECTOR, int dim>
    unsigned int
    CellDataContainer<dealii::hp::DoFHandler<dim>, VECTOR, dim>::GetNDoFsPerCell() const
    {
      return _cell[0]->get_fe().dofs_per_cell;
    }
  /*********************************************/
  template<typename VECTOR, int dim>
    unsigned int
    CellDataContainer<dealii::hp::DoFHandler<dim>, VECTOR, dim>::GetNQPoints() const
    {
      return (_q_collection[_cell[0]->active_fe_index()]).size();
    }
  /*********************************************/

  template<typename VECTOR, int dim>
    unsigned int
    CellDataContainer<dealii::hp::DoFHandler<dim>, VECTOR, dim>::GetMaterialId() const
    {
      return _cell[0]->material_id();
    }
  /*********************************************/

  template<typename VECTOR, int dim>
    unsigned int
    CellDataContainer<dealii::hp::DoFHandler<dim>, VECTOR, dim>::GetNbrMaterialId(
        unsigned int face) const
    {
      if (_cell[0]->neighbor_index(face) != -1)
        return _cell[0]->neighbor(face)->material_id();
      else
        throw DOpEException("There is no neighbor with number" + face,
            "HpCellDataContainer::GetNbrMaterialId");
    }
  /*********************************************/

  template<typename VECTOR, int dim>
    bool
    CellDataContainer<dealii::hp::DoFHandler<dim>, VECTOR, dim>::GetIsAtBoundary() const
    {
      return _cell[0]->at_boundary();
    }

  /*********************************************/

  template<typename VECTOR, int dim>
    double
    CellDataContainer<dealii::hp::DoFHandler<dim>, VECTOR, dim>::GetCellDiameter() const
    {
      return _cell[0]->diameter();
    }

  /*********************************************/
  template<typename VECTOR, int dim>
    const DOpEWrapper::FEValues<dim>&
    CellDataContainer<dealii::hp::DoFHandler<dim>, VECTOR, dim>::GetFEValuesState() const
    {
      return static_cast<const DOpEWrapper::FEValues<dim>&>(_state_hp_fe_values.get_present_fe_values());
    }
  /*********************************************/
  template<typename VECTOR, int dim>
    const DOpEWrapper::FEValues<dim>&
    CellDataContainer<dealii::hp::DoFHandler<dim>, VECTOR, dim>::GetFEValuesControl() const
    {
      return static_cast<const DOpEWrapper::FEValues<dim>&>(_control_hp_fe_values.get_present_fe_values());
    }
  /*********************************************/

  template<typename VECTOR, int dim>
    unsigned int
    CellDataContainer<dealii::hp::DoFHandler<dim>, VECTOR, dim>::GetStateIndex() const
    {
      return _state_index;
    }
  /*********************************************/
  template<typename VECTOR, int dim>
    unsigned int
    CellDataContainer<dealii::hp::DoFHandler<dim>, VECTOR, dim>::GetControlIndex() const
    {
      return _control_index;
    }
} //end of namespace

#endif /* WORKINGTITLE_H_ */
