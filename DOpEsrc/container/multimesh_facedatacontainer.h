#ifndef MULTIMESH_FACEDATACONTAINER_H_
#define MULTIMESH_FACEDATACONTAINER_H_

#include "spacetimehandler.h"
#include "statespacetimehandler.h"
#include "fevalues_wrapper.h"
#include "dopeexception.h"

#include <dofs/dof_handler.h>
#include <hp/dof_handler.h>

using namespace dealii;

namespace DOpE
{
  /**
   * Dummy Template Class, acts as kind of interface. Through template specialization, we
   * distinguish between the 'classic' and the 'hp' case.
   *
   */

  template<typename DOFHANDLER, typename VECTOR, int dim>
    class Multimesh_FaceDataContainer
    {
      public:
        Multimesh_FaceDataContainer()
        {
          throw(DOpEException(
              "Dummy class, this constructor should never get called.",
              "FaceDataContainer<dealii::DoFHandler<dim> , VECTOR, dim>::FaceDataContainer"));
        }
        ;
    };

  /**
   * This two classes hold all the information we need in the integrator to
   * integrate something over a face of a cell (could be a functional, a PDE, etc.).
   * Of particular importance: This class holds the FaceFEValues objects.
   *
   * @template VECTOR     Type of the vector we use in our computations (i.e. Vector<double> or BlockVector<double>)
   * @template dim        1+ the dimension of the integral we are actually interested in.
   */

  template<typename VECTOR, int dim>
    class Multimesh_FaceDataContainer<dealii::DoFHandler<dim>, VECTOR, dim>
    {

      public:
        /**
         * Constructor. Initializes the FaceFEValues objects.
         *
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
        template<typename FE, typename SPARSITYPATTERN, int dopedim,
            int dealdim>
          Multimesh_FaceDataContainer(
              const Quadrature<dim - 1>& quad,
              UpdateFlags update_flags,
              SpaceTimeHandler<FE, dealii::DoFHandler<dim>, SPARSITYPATTERN,
                  VECTOR, dopedim, dealdim>& sth,
              const typename std::vector<typename dealii::DoFHandler<dim>::cell_iterator>& cell,
              const typename std::vector<typename dealii::Triangulation<dim>::cell_iterator>& tria_cell,
              const std::map<std::string, const Vector<double>*> &param_values,
              const std::map<std::string, const VECTOR*> &domain_values, bool/* just for compatibility*/) :
                _param_values(param_values),
                _domain_values(domain_values),
                _cell(cell),
                _tria_cell(tria_cell),
                _state_fe_values(*(sth.GetFESystem("state")), quad,
                    update_flags),
                _control_fe_values(*(sth.GetFESystem("control")), quad,
                    update_flags)
          {
            _state_index = sth.GetStateIndex();
            if (_state_index == 1)
              _control_index = 0;
            else
              _control_index = 1;
            _n_q_points_per_cell = quad.size();
            _n_dofs_per_cell = cell[0]->get_fe().dofs_per_cell;
	    _control_prolongation = IdentityMatrix(_cell[this->GetControlIndex()]->get_fe().dofs_per_cell);
	    _state_prolongation = IdentityMatrix(_cell[this->GetStateIndex()]->get_fe().dofs_per_cell);
          }

        ~Multimesh_FaceDataContainer()
        {
        }
        /*********************************************/
        /*
         * This function reinits the FEValues on the actual face. Should
         * be called prior to any of the get-functions.
         *
         * @param face_no     The 'local number' (i.e. from the perspective of the actual cell) of the
         *                    actual face.
         */
        inline void
	  ReInit(unsigned int coarse_index,unsigned int fine_index, const FullMatrix<double>& prolongation_matrix, unsigned int face_no);

        /*********************************************/
        /**
         * Just for compatibility reasons.
         */
        inline void
        ReInitNbr(){}

        /*********************************************/
        /**
         * Get functions to extract data. They all assume that ReInit
         * is executed before calling them.
         */
        inline unsigned int
        GetNDoFsPerCell() const;
        inline unsigned int
        GetNbrNDoFsPerCell() const;
        inline unsigned int
        GetNQPoints() const;
        inline unsigned int
        GetMaterialId() const;
        inline unsigned int
        GetNbrMaterialId() const;
        inline unsigned int
        GetNbrMaterialId(unsigned int face) const;
        inline bool
        GetIsAtBoundary() const;
        inline double
        GetCellDiameter() const;
        inline unsigned int
        GetBoundaryIndicator() const;
        inline const DOpEWrapper::FEFaceValues<dim>&
        GetFEFaceValuesState() const;
        inline const DOpEWrapper::FEFaceValues<dim>&
        GetFEFaceValuesControl() const;

        /**********************************************/
        /*
         * Looks up the given name in _parameter_data and returns the corresponding value
         * through 'value'.
         */
        void
        GetParamValues(std::string name, Vector<double>& value) const;

        /*********************************************/
        /**
         * Functions to extract values and gradients out of the FEFaceValues
         */

        /*
         * Writes the values of the state variable at the quadrature points into values.
         */
        inline void
        GetFaceValuesState(std::string name, std::vector<double>& values) const;
        /*********************************************/
        /*
         * Same as above for the Vector valued case.
         */
        inline void
        GetFaceValuesState(std::string name, std::vector<Vector<double> >& values) const;

        /*********************************************/
        /*
         * Writes the values of the control variable at the quadrature points into values
         */
        inline void
        GetFaceValuesControl(std::string name, std::vector<double>& values) const;
        /*********************************************/
        /*
         * Same as above for the Vector valued case.
         */
        inline void
        GetFaceValuesControl(std::string name, std::vector<Vector<double> >& values) const;
        /*********************************************/
        /*
         * Writes the values of the state gradient at the quadrature points into values.
         */

        template<int targetdim>
          inline void
              GetFaceGradsState(std::string name,
                  std::vector<Tensor<1, targetdim> >& values) const;

        /*********************************************/
        /*
         * Same as above for the Vector valued case.
         */
        template<int targetdim>
          inline void
          GetFaceGradsState(std::string name,
              std::vector<std::vector<Tensor<1, targetdim> > >& values) const;

        /*********************************************/
        /*
         * Writes the values of the control gradient at the quadrature points into values.
         */
        template<int targetdim>
          inline void
          GetFaceGradsControl(std::string name,
              std::vector<Tensor<1, targetdim> >& values) const;

        /*********************************************/
        /*
         * Same as above for the Vector valued case.
         */
        template<int targetdim>
          inline void
          GetFaceGradsControl(std::string name,
              std::vector<std::vector<Tensor<1, targetdim> > >& values) const;

      private:
        /*
         * Helper Functions
         */
        unsigned int
        GetStateIndex() const;
        unsigned int
        GetControlIndex() const;
	unsigned int
	  GetCoarseIndex() const { return _coarse_index; } 
        unsigned int
	  GetFineIndex() const { return _fine_index; } 
        /***********************************************************/
        /**
         * Helper Function. Vector valued case.
         */
        inline void
        GetValues(typename dealii::DoFHandler<dim>::cell_iterator cell,
		  const FullMatrix<double>& prolongation,
		  const DOpEWrapper::FEFaceValues<dim>& fe_values, std::string name,
		  std::vector<double>& values) const;
        /***********************************************************/
        /**
         * Helper Function. Vector valued case.
         */
        inline void
        GetValues(typename dealii::DoFHandler<dim>::cell_iterator cell,
		  const FullMatrix<double>& prolongation,
		  const DOpEWrapper::FEFaceValues<dim>& fe_values, std::string name,
		  std::vector<Vector<double> >& values) const;
        /***********************************************************/
        /**
         * Helper Function.
         */
        template<int targetdim>
          inline void
          GetGrads(typename dealii::DoFHandler<dim>::cell_iterator cell,
		   const FullMatrix<double>& prolongation,
		   const DOpEWrapper::FEFaceValues<dim>& fe_values,
		   std::string name, std::vector<Tensor<1, targetdim> >& values) const;
        /***********************************************************/
        /**
         * Helper Function. Vector valued case.
         */
        template<int targetdim>
          inline void
          GetGrads(typename dealii::DoFHandler<dim>::cell_iterator cell,
		   const FullMatrix<double>& prolongation,
		   const DOpEWrapper::FEFaceValues<dim>& fe_values,
		   std::string name, std::vector<std::vector<Tensor<1, targetdim> > >& values) const;
        /***********************************************************/
        inline const std::map<std::string, const VECTOR*> &
        GetDomainValues() const
        {
          return _domain_values;
        }
        /***********************************************************/
        //"global" member data, part of every instantiation
        const std::map<std::string, const Vector<double>*> &_param_values;
        const std::map<std::string, const VECTOR*> &_domain_values;
        unsigned int _state_index;
        unsigned int _control_index;

 	const typename std::vector<typename dealii::DoFHandler<dim>::cell_iterator>& _cell;
	const typename std::vector<typename dealii::Triangulation<dim>::cell_iterator>& _tria_cell;
        DOpEWrapper::FEFaceValues<dim> _state_fe_values;
        DOpEWrapper::FEFaceValues<dim> _control_fe_values;

        FullMatrix<double> _control_prolongation;
	FullMatrix<double> _state_prolongation;
	unsigned int _coarse_index, _fine_index;

	unsigned int _n_q_points_per_cell;
        unsigned int _n_dofs_per_cell;

        unsigned int _face;
    };
 

  /***********************************************************************/
  /************************IMPLEMENTATION*for*DoFHandler*********************************/
  /***********************************************************************/

  template<typename VECTOR, int dim>
    void
    Multimesh_FaceDataContainer<dealii::DoFHandler<dim>, VECTOR, dim>::ReInit(
      unsigned int coarse_index,
      unsigned int fine_index, 
      const FullMatrix<double>& prolongation_matrix,
      unsigned int face_no)
    {
      _face = face_no;
      _coarse_index = coarse_index;
      _fine_index = fine_index;
      assert(this->GetControlIndex() < _cell.size());
      
      if(coarse_index == this->GetStateIndex())
      {
	_state_prolongation = prolongation_matrix;
	_control_prolongation = IdentityMatrix(_cell[this->GetControlIndex()]->get_fe().dofs_per_cell);
      }
      else
      {
	if(coarse_index == this->GetControlIndex())
	{
	   _control_prolongation = prolongation_matrix;
	   _state_prolongation = IdentityMatrix(_cell[this->GetStateIndex()]->get_fe().dofs_per_cell);
	}
	else
	{
	  _control_prolongation = IdentityMatrix(_cell[this->GetControlIndex()]->get_fe().dofs_per_cell);
	  _state_prolongation = IdentityMatrix(_cell[this->GetStateIndex()]->get_fe().dofs_per_cell);
	  _fine_index = 0;
	}
      }

      _state_fe_values.reinit(_tria_cell[GetFineIndex()], face_no);
      _control_fe_values.reinit(_tria_cell[GetFineIndex()], face_no);
    }
  /***********************************************************************/
  template<typename VECTOR, int dim>
    unsigned int
    Multimesh_FaceDataContainer<dealii::DoFHandler<dim>, VECTOR, dim>::GetNDoFsPerCell() const
    {
      return _n_dofs_per_cell;
    }

  /***********************************************************************/

  template<typename VECTOR, int dim>
    unsigned int
    Multimesh_FaceDataContainer<dealii::DoFHandler<dim>, VECTOR, dim>::GetNbrNDoFsPerCell() const
    {
      throw DOpEException("This function has not been written since we do not know what the right neigbour is!",
            "Multimesh_FaceDataContainer::GetNbrNDoFsPerCell");
    }

  /**********************************************/
  template<typename VECTOR, int dim>
    unsigned int
    Multimesh_FaceDataContainer<dealii::DoFHandler<dim>, VECTOR, dim>::GetNQPoints() const
    {
      return _n_q_points_per_cell;
    }

  /**********************************************/
  template<typename VECTOR, int dim>
    unsigned int
    Multimesh_FaceDataContainer<dealii::DoFHandler<dim>, VECTOR, dim>::GetMaterialId() const
    {
      return _tria_cell[GetFineIndex()]->material_id();
    }

  /**********************************************/
  template<typename VECTOR, int dim>
    unsigned int
    Multimesh_FaceDataContainer<dealii::DoFHandler<dim>, VECTOR, dim>::GetNbrMaterialId() const
    {
      return this->GetNbrMaterialId(_face);
    }

  template<typename VECTOR, int dim>
    unsigned int
    Multimesh_FaceDataContainer<dealii::DoFHandler<dim>, VECTOR, dim>::GetNbrMaterialId(
      unsigned int /*face*/) const
    {
      throw DOpEException("This function has not been written since we do not know what the right neigbour is!",
			  "Multimesh_FaceDataContainer::GetNbrMaterialId");
    }

  /**********************************************/
  template<typename VECTOR, int dim>
    bool
    Multimesh_FaceDataContainer<dealii::DoFHandler<dim>, VECTOR, dim>::GetIsAtBoundary() const
    {
      return _tria_cell[GetFineIndex()]->face(_face)->at_boundary();
    }

  /**********************************************/
  template<typename VECTOR, int dim>
    double
    Multimesh_FaceDataContainer<dealii::DoFHandler<dim>, VECTOR, dim>::GetCellDiameter() const
    {
      throw DOpEException("This function has not been written since we do not know what the right Diameter!",
			  "Multimesh_FaceDataContainer::GetCellDiameter");
    }

  /**********************************************/

  template<typename VECTOR, int dim>
    unsigned int
    Multimesh_FaceDataContainer<dealii::DoFHandler<dim>, VECTOR, dim>::GetBoundaryIndicator() const
    {
      return _tria_cell[GetFineIndex()]->face(_face)->boundary_indicator();
    }

  /**********************************************/
  template<typename VECTOR, int dim>
    const DOpEWrapper::FEFaceValues<dim>&
    Multimesh_FaceDataContainer<dealii::DoFHandler<dim>, VECTOR, dim>::GetFEFaceValuesState() const
    {
      return _state_fe_values;
    }

  /**********************************************/
  template<typename VECTOR, int dim>
    const DOpEWrapper::FEFaceValues<dim>&
    Multimesh_FaceDataContainer<dealii::DoFHandler<dim>, VECTOR, dim>::GetFEFaceValuesControl() const
    {
      return _control_fe_values;
    }

  /**********************************************/

  template<typename VECTOR, int dim>
    void
    Multimesh_FaceDataContainer<dealii::DoFHandler<dim>, VECTOR, dim>::GetParamValues(
        std::string name, Vector<double>& value) const
    {
      typename std::map<std::string, const Vector<double>*>::const_iterator it =
          _param_values.find(name);
      if (it == _param_values.end())
        {
          throw DOpEException("Did not find " + name,
              "Multimesh_FaceDataContainer::GetParamValues");
        }
      value = *(it->second);
    }

  /*********************************************/
  template<typename VECTOR, int dim>
    void
    Multimesh_FaceDataContainer<dealii::DoFHandler<dim>, VECTOR, dim>::GetFaceValuesState(
        std::string name, std::vector<double>& values) const
    {
      this->GetValues(_cell[this->GetStateIndex()],_state_prolongation,this->GetFEFaceValuesState(), name, values);
    }
  /*********************************************/
  template<typename VECTOR, int dim>
    void
    Multimesh_FaceDataContainer<dealii::DoFHandler<dim>, VECTOR, dim>::GetFaceValuesState(
        std::string name, std::vector<Vector<double> >& values) const
    {
      this->GetValues(_cell[this->GetStateIndex()],_state_prolongation,this->GetFEFaceValuesState(), name, values);

    }

  /*********************************************/
  template<typename VECTOR, int dim>
    void
    Multimesh_FaceDataContainer<dealii::DoFHandler<dim>, VECTOR, dim>::GetFaceValuesControl(
        std::string name, std::vector<double>& values) const
    {
      this->GetValues(_cell[this->GetControlIndex()],_control_prolongation,this->GetFEFaceValuesControl(), name, values);
    }

  /*********************************************/
  template<typename VECTOR, int dim>
    void
    Multimesh_FaceDataContainer<dealii::DoFHandler<dim>, VECTOR, dim>::GetFaceValuesControl(
        std::string name, std::vector<Vector<double> >& values) const
    {
      this->GetValues(_cell[this->GetControlIndex()],_control_prolongation,this->GetFEFaceValuesControl(), name, values);
    }

  /*********************************************/
  template<typename VECTOR, int dim>
    template<int targetdim>
      void
      Multimesh_FaceDataContainer<dealii::DoFHandler<dim>, VECTOR, dim>::GetFaceGradsState(
          std::string name, std::vector<Tensor<1, targetdim> >& values) const
      {
        this->GetGrads<targetdim> (_cell[this->GetStateIndex()],_state_prolongation,this->GetFEFaceValuesState(), name, values);
      }

  /*********************************************/
  template<typename VECTOR, int dim>
    template<int targetdim>
      void
      Multimesh_FaceDataContainer<dealii::DoFHandler<dim>, VECTOR, dim>::GetFaceGradsState(
          std::string name, std::vector<std::vector<Tensor<1, targetdim> > >& values) const
      {
        this->GetGrads<targetdim> (_cell[this->GetStateIndex()],_state_prolongation,this->GetFEFaceValuesState(), name, values);
      }

  /***********************************************************************/

  template<typename VECTOR, int dim>
    template<int targetdim>
      void
      Multimesh_FaceDataContainer<dealii::DoFHandler<dim>, VECTOR, dim>::GetFaceGradsControl(
          std::string name, std::vector<Tensor<1, targetdim> >& values) const
      {
        this->GetGrads<targetdim> (_cell[this->GetControlIndex()],_control_prolongation,this->GetFEFaceValuesControl(), name, values);
      }
  /***********************************************************************/

  template<typename VECTOR, int dim>
    template<int targetdim>
      void
      Multimesh_FaceDataContainer<dealii::DoFHandler<dim>, VECTOR, dim>::GetFaceGradsControl(
          std::string name, std::vector<std::vector<Tensor<1, targetdim> > >& values) const
      {
        this->GetGrads<targetdim> (_cell[this->GetControlIndex()],_control_prolongation,this->GetFEFaceValuesControl(), name, values);
      }

  /***********************************************************************/

  template<typename VECTOR, int dim>
    unsigned int
    Multimesh_FaceDataContainer<dealii::DoFHandler<dim>, VECTOR, dim>::GetStateIndex() const
    {
      return _state_index;
    }

  /***********************************************************************/

  template<typename VECTOR, int dim>
    unsigned int
    Multimesh_FaceDataContainer<dealii::DoFHandler<dim>, VECTOR, dim>::GetControlIndex() const
    {
      return _control_index;
    }

  /***********************************************************************/
  template<typename VECTOR, int dim>
    void
    Multimesh_FaceDataContainer<dealii::DoFHandler<dim>, VECTOR, dim>::GetValues(
        typename dealii::DoFHandler<dim>::cell_iterator cell,
	const FullMatrix<double>& prolongation,
	const DOpEWrapper::FEFaceValues<dim>& fe_values, std::string name,
        std::vector<double>& values) const
    {
      typename std::map<std::string, const VECTOR*>::const_iterator it =
          this->GetDomainValues().find(name);
      if (it == this->GetDomainValues().end())
        {
          throw DOpEException("Did not find " + name,
              "Multimesh_FaceDataContainer::GetValues");
        }
      unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
      //Now we get the values on the real cell
      dealii::Vector<double> dof_values(dofs_per_cell);
      dealii::Vector<double> dof_values_transformed(dofs_per_cell);
      cell->get_dof_values (*(it->second), dof_values);
      //Now compute the real values at the nodal points 
      prolongation.vmult(dof_values_transformed,dof_values);
      
      //Copied from deal FEValuesBase<dim,spacedim>::get_function_values
      // see deal.II/source/fe/fe_values.cc
      unsigned int n_quadrature_points = GetNQPoints();
      std::fill_n (values.begin(), n_quadrature_points, 0);
      for (unsigned int shape_func=0; shape_func<dofs_per_cell; ++shape_func)
      {
	const double value = dof_values_transformed(shape_func);
	if (value == 0.)
	continue;
	
	const double *shape_value_ptr = &(fe_values.shape_value(shape_func, 0));
	for (unsigned int point=0; point<n_quadrature_points; ++point)
	  values[point] += value * *shape_value_ptr++;
      }
    }

  /***********************************************************************/
  template<typename VECTOR, int dim>
    void
    Multimesh_FaceDataContainer<dealii::DoFHandler<dim>, VECTOR, dim>::GetValues(
      typename dealii::DoFHandler<dim>::cell_iterator cell,
      const FullMatrix<double>& prolongation,
      const DOpEWrapper::FEFaceValues<dim>& fe_values, std::string name,
      std::vector<Vector<double> >& values) const
    {
      typename std::map<std::string, const VECTOR*>::const_iterator it =
          this->GetDomainValues().find(name);
      if (it == this->GetDomainValues().end())
        {
          throw DOpEException("Did not find " + name,
              "Multimesh_FaceDataContainer::GetValues");
        }
      
       unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
      //Now we get the values on the real cell
      dealii::Vector<double> dof_values(dofs_per_cell);
      dealii::Vector<double> dof_values_transformed(dofs_per_cell);
      cell->get_dof_values (*(it->second), dof_values);
      //Now compute the real values at the nodal points 
      prolongation.vmult(dof_values_transformed,dof_values);
      
      //Copied from deal FEValuesBase<dim,spacedim>::get_function_values
      // see deal.II/source/fe/fe_values.cc
       const unsigned int n_components = cell->get_fe().n_components();
      unsigned int n_quadrature_points = GetNQPoints();
      for (unsigned i=0;i<values.size();++i)
	std::fill_n (values[i].begin(), values[i].size(), 0);
      
      for (unsigned int shape_func=0; shape_func<dofs_per_cell; ++shape_func)
      {
	const double value = dof_values_transformed(shape_func);
	if (value == 0.)
	  continue;
	
	if (cell->get_fe().is_primitive(shape_func))
	{
	  const unsigned int comp = cell->get_fe().system_to_component_index(shape_func).first;
	  for (unsigned int point=0; point<n_quadrature_points; ++point)
	    values[point](comp) += value * fe_values.shape_value(shape_func,point);
	}
	else
	  for (unsigned int c=0; c<n_components; ++c)
	  {
	    if (cell->get_fe().get_nonzero_components(shape_func)[c] == false)
	      continue;
	    for (unsigned int point=0; point<n_quadrature_points; ++point)
	      values[point](c) += value * fe_values.shape_value_component(shape_func,point,c);
	  }
      }
    }

  /***********************************************************************/

  template<typename VECTOR, int dim>
    template<int targetdim>
      void
      Multimesh_FaceDataContainer<dealii::DoFHandler<dim>, VECTOR, dim>::GetGrads(
          typename dealii::DoFHandler<dim>::cell_iterator cell,
	  const FullMatrix<double>& prolongation,
	  const DOpEWrapper::FEFaceValues<dim>& fe_values, std::string name,
          std::vector<Tensor<1, targetdim> >& values) const
      {
        typename std::map<std::string, const VECTOR*>::const_iterator it =
            this->GetDomainValues().find(name);
        if (it == this->GetDomainValues().end())
          {
            throw DOpEException("Did not find " + name,
                "Multimesh_FaceDataContainerBase::GetGrads");
          }
        unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
	//Now we get the values on the real cell
	dealii::Vector<double> dof_values(dofs_per_cell);
	dealii::Vector<double> dof_values_transformed(dofs_per_cell);
	cell->get_dof_values (*(it->second), dof_values);
	//Now compute the real values at the nodal points 
	prolongation.vmult(dof_values_transformed,dof_values);
      
        //Copied from deal FEValuesBase<dim,spacedim>::get_function_gradients
	unsigned int n_quadrature_points = GetNQPoints();
	std::fill_n (values.begin(), n_quadrature_points, Tensor<1,targetdim>());
	
	for (unsigned int shape_func=0; shape_func<dofs_per_cell; ++shape_func)
	{
	  const double value = dof_values_transformed(shape_func);
	  if (value == 0.)
	    continue;
	  
	  const Tensor<1,targetdim> *shape_gradient_ptr
	    = &(fe_values.shape_grad(shape_func,0));
	  for (unsigned int point=0; point<n_quadrature_points; ++point)
	    values[point] += value * *shape_gradient_ptr++;
	}
      }

  /***********************************************************************/

  template<typename VECTOR, int dim>
    template<int targetdim>
      void
      Multimesh_FaceDataContainer<dealii::DoFHandler<dim>, VECTOR, dim>::GetGrads(
          typename dealii::DoFHandler<dim>::cell_iterator cell,
	  const FullMatrix<double>& prolongation,
	  const DOpEWrapper::FEFaceValues<dim>& fe_values, std::string name,
          std::vector<std::vector<Tensor<1, targetdim> > >& values) const
      {
        typename std::map<std::string, const VECTOR*>::const_iterator it =
            this->GetDomainValues().find(name);
        if (it == this->GetDomainValues().end())
          {
            throw DOpEException("Did not find " + name,
                "Multimesh_FaceDataContainerBase::GetGrads");
          }
       
	unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
	//Now we get the values on the real cell
	dealii::Vector<double> dof_values(dofs_per_cell);
	dealii::Vector<double> dof_values_transformed(dofs_per_cell);
	cell->get_dof_values (*(it->second), dof_values);
	//Now compute the real values at the nodal points 
	prolongation.vmult(dof_values_transformed,dof_values);
      
        //Copied from deal FEValuesBase<dim,spacedim>::get_function_gradients
	const unsigned int n_components = cell->get_fe().n_components();
	unsigned int n_quadrature_points = GetNQPoints();
	for (unsigned i=0;i<values.size();++i)
	  std::fill_n (values[i].begin(), values[i].size(), Tensor<1,dim>());
	
	for (unsigned int shape_func=0; shape_func<dofs_per_cell; ++shape_func)
	{
	  const double value = dof_values_transformed(shape_func);
	  if (value == 0.)
	    continue;
	  
	  if (cell->get_fe().is_primitive(shape_func))
	  {
	    const unsigned int comp = cell->get_fe().system_to_component_index(shape_func).first;
	    for (unsigned int point=0; point<n_quadrature_points; ++point)
	      values[point][comp] += value * fe_values.shape_grad(shape_func,point);
	  }
	  else
	    for (unsigned int c=0; c<n_components; ++c)
	    {
	      if (cell->get_fe().get_nonzero_components(shape_func)[c] == false)
		continue;
	      for (unsigned int point=0; point<n_quadrature_points; ++point)
		values[point][c] += value * fe_values.shape_grad_component(shape_func,point,c);
	    }
	}
      }

  /***********************************************************************/
  /************************END*OF*IMPLEMENTATION**************************/
  /***********************************************************************/

}//end of namespace

#endif /* MULTIMESH_FACEDATACONTAINER_H_ */
