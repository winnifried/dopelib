#ifndef _LOCALPDE_
#define _LOCALPDE_

#include "pdeinterface.h"
#include "celldatacontainer.h"
#include "facedatacontainer.h"

#include "my_functions.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

template<typename VECTOR, int dopedim, int dealdim>
class LocalPDE: public PDEInterface<CellDataContainer,FaceDataContainer,dealii::DoFHandler<dealdim>, VECTOR,dopedim, dealdim>
{
	public:

 LocalPDE():_state_block_components(1, 0)
    {

    }

  //Initial Values from Control
  void Init_CellRhs(const dealii::Function<dealdim>* /*init_values*/,
		    const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
		    dealii::Vector<double> &local_cell_vector, double scale)
  {
    const DOpEWrapper::FEValues<dealdim> & state_fe_values =
      cdc.GetFEValuesState();
    unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
    unsigned int n_q_points = cdc.GetNQPoints();
    _qvalues.resize(n_q_points);
    cdc.GetValuesControl("control",_qvalues);
    
    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
    {
      for (unsigned int i = 0; i < n_dofs_per_cell; i++)
      {
	local_cell_vector(i) += scale
	  * _qvalues[q_point] * state_fe_values.shape_value(i,q_point)
	  * state_fe_values.JxW(q_point);
      }
    }
  }
  //Initial Values from Control
  void Init_CellRhs_Q(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
		      dealii::Vector<double> &local_cell_vector, double scale)
  {
    const DOpEWrapper::FEValues<dealdim> & control_fe_values =
      cdc.GetFEValuesControl();
    unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
    unsigned int n_q_points = cdc.GetNQPoints();
    _zvalues.resize(n_q_points);
    cdc.GetValuesState("adjoint",_zvalues);
    
    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
    {
      for (unsigned int i = 0; i < n_dofs_per_cell; i++)
      {
	local_cell_vector(i) += scale
	  * control_fe_values.shape_value(i,q_point)* _zvalues[q_point] 
	  * control_fe_values.JxW(q_point);
      }
    }
  }    
  //Initial Values from Control
  void Init_CellRhs_QT(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
		       dealii::Vector<double> &local_cell_vector, double scale)
  {
    const DOpEWrapper::FEValues<dealdim> & state_fe_values =
      cdc.GetFEValuesState();
    unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
    unsigned int n_q_points = cdc.GetNQPoints();
    _dqvalues.resize(n_q_points);
    cdc.GetValuesControl("dq",_dqvalues);
    
    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
    {
      for (unsigned int i = 0; i < n_dofs_per_cell; i++)
      {
	local_cell_vector(i) += scale
	  * _dqvalues[q_point] * state_fe_values.shape_value(i,q_point)
	  * state_fe_values.JxW(q_point);
      }
    }
  }    
 //Initial Values from Control
  void Init_CellRhs_QTT(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
		       dealii::Vector<double> &local_cell_vector, double scale)
  {
    const DOpEWrapper::FEValues<dealdim> & control_fe_values =
      cdc.GetFEValuesControl();
    unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
    unsigned int n_q_points = cdc.GetNQPoints();
    _dzvalues.resize(n_q_points);
    cdc.GetValuesState("adjoint_hessian",_dzvalues);
    
    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
    {
      for (unsigned int i = 0; i < n_dofs_per_cell; i++)
      {
	local_cell_vector(i) += scale
	  * control_fe_values.shape_value(i,q_point)* _dzvalues[q_point] 
	  * control_fe_values.JxW(q_point);
      }
    }
  }  

  // Domain values for cells
  void CellEquation(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc, dealii::Vector<double> &local_cell_vector,double scale, double /*scale_ico*/)
  {
    assert(this->_problem_type == "state");
    
    const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
    unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
    unsigned int n_q_points = cdc.GetNQPoints();
    
    _uvalues.resize(n_q_points);
    _ugrads.resize(n_q_points);
    
    cdc.GetValuesState("last_newton_solution", _uvalues);
    cdc.GetGradsState("last_newton_solution", _ugrads);
        
    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
    {
      for (unsigned int i = 0; i < n_dofs_per_cell; i++)
      {
	const double phi_i = state_fe_values.shape_value(i, q_point);
	const Tensor<1, dealdim> phi_i_grads = state_fe_values.shape_grad(i, q_point);
	
	local_cell_vector(i) += scale * ((_ugrads[q_point]*phi_i_grads) + _uvalues[q_point]*_uvalues[q_point]*phi_i) * state_fe_values.JxW(q_point);
      }
    }
  }
    // Domain values for cells
  void CellEquation_U(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc, dealii::Vector<double> &local_cell_vector,double scale, double /*scale_ico*/)
  {
    assert(this->_problem_type == "adjoint");
    
    const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
    unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
    unsigned int n_q_points = cdc.GetNQPoints();
    
    _uvalues.resize(n_q_points);
    _zvalues.resize(n_q_points);
    _zgrads.resize(n_q_points);
    
    cdc.GetValuesState("state", _uvalues);
    cdc.GetValuesState("last_newton_solution", _zvalues);
    cdc.GetGradsState("last_newton_solution", _zgrads);
        
    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
    {
      for (unsigned int i = 0; i < n_dofs_per_cell; i++)
      {
	const double phi_i = state_fe_values.shape_value(i, q_point);
	const Tensor<1, dealdim> phi_i_grads = state_fe_values.shape_grad(i, q_point);
	
	local_cell_vector(i) += scale * ((_zgrads[q_point]*phi_i_grads) + 2.*_uvalues[q_point]*_zvalues[q_point]*phi_i) * state_fe_values.JxW(q_point);
      }
    }
  }
  // Domain values for cells
  void CellEquation_UT(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc, dealii::Vector<double> &local_cell_vector,double scale, double /*scale_ico*/)
  {
    assert(this->_problem_type == "tangent");
    
    const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
    unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
    unsigned int n_q_points = cdc.GetNQPoints();
    
    _uvalues.resize(n_q_points);
    _duvalues.resize(n_q_points);
    _dugrads.resize(n_q_points);
    
    cdc.GetValuesState("state", _uvalues);
    cdc.GetValuesState("last_newton_solution", _duvalues);
    cdc.GetGradsState("last_newton_solution", _dugrads);
        
    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
    {
      for (unsigned int i = 0; i < n_dofs_per_cell; i++)
      {
	const double phi_i = state_fe_values.shape_value(i, q_point);
	const Tensor<1, dealdim> phi_i_grads = state_fe_values.shape_grad(i, q_point);
	
	local_cell_vector(i) += scale * ((_dugrads[q_point]*phi_i_grads) + 2.*_duvalues[q_point]*_uvalues[q_point]*phi_i) * state_fe_values.JxW(q_point);
      }
    }
  }
  // Domain values for cells
  void CellEquation_UTT(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc, dealii::Vector<double> &local_cell_vector,double scale, double /*scale_ico*/)
  {
    assert(this->_problem_type == "adjoint_hessian");
    
    const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
    unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
    unsigned int n_q_points = cdc.GetNQPoints();
    
    _uvalues.resize(n_q_points);
    _dzvalues.resize(n_q_points);
    _dzgrads.resize(n_q_points);
    
    cdc.GetValuesState("state", _uvalues);
    cdc.GetValuesState("last_newton_solution", _dzvalues);
    cdc.GetGradsState("last_newton_solution", _dzgrads);
        
    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
    {
      for (unsigned int i = 0; i < n_dofs_per_cell; i++)
      {
	const double phi_i = state_fe_values.shape_value(i, q_point);
	const Tensor<1, dealdim> phi_i_grads = state_fe_values.shape_grad(i, q_point);
	
	local_cell_vector(i) += scale * ((_dzgrads[q_point]*phi_i_grads) + 2.*_uvalues[q_point]*_dzvalues[q_point]*phi_i) * state_fe_values.JxW(q_point);
      }
    }
  }
  // Domain values for cells
  void CellEquation_UU(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc, dealii::Vector<double> &local_cell_vector,double scale, double /*scale_ico*/)
  {
    assert(this->_problem_type == "adjoint_hessian");
    
    const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
    unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
    unsigned int n_q_points = cdc.GetNQPoints();
    
    _uvalues.resize(n_q_points);
    _ugrads.resize(n_q_points);
    
    cdc.GetValuesState("tangent", _duvalues);
    cdc.GetValuesState("adjoint", _zvalues);
    cdc.GetGradsState("adjoint", _zgrads);
    
        
    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
    {
      for (unsigned int i = 0; i < n_dofs_per_cell; i++)
      {
	const double phi_i = state_fe_values.shape_value(i, q_point);
	const Tensor<1, dealdim> phi_i_grads = state_fe_values.shape_grad(i, q_point);
	
	local_cell_vector(i) += scale * ((_zgrads[q_point]*phi_i_grads) + 2.*_zvalues[q_point]*_duvalues[q_point]*phi_i) * state_fe_values.JxW(q_point);
      }
    }
  }
  
  void CellEquation_Q(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& /*cdc*/,
		      dealii::Vector<double> &/*local_cell_vector*/,
		      double /*scale*/, double /*scale_ico*/) { }
  void CellEquation_QT(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& /*cdc*/,
		       dealii::Vector<double> &/*local_cell_vector*/,
		       double /*scale*/, double /*scale_ico*/) { }
  void CellEquation_QTT(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& /*cdc*/,
			dealii::Vector<double> &/*local_cell_vector*/,
			double /*scale*/, double /*scale_ico*/) { }
  void CellEquation_QU(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& /*cdc*/,
		       dealii::Vector<double> &/*local_cell_vector*/,
		       double /*scale*/, double /*scale_ico*/) { }
  void CellEquation_UQ(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& /*cdc*/,
		       dealii::Vector<double> &/*local_cell_vector*/,
		       double /*scale*/, double /*scale_ico*/) { }
  void CellEquation_QQ(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& /*cdc*/,
		       dealii::Vector<double> &/*local_cell_vector*/,
		       double /*scale*/, double /*scale_ico*/) { }

  void CellMatrix(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc, FullMatrix<double> &local_entry_matrix, double scale, double)
  {
    const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
    unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
    unsigned int n_q_points = cdc.GetNQPoints();
    
    //if(this->_problem_type == "state")
    if(this->_problem_type == "state")
      cdc.GetValuesState("last_newton_solution", _uvalues);
    else
      cdc.GetValuesState("state", _uvalues);  
    
    std::vector<double> phi_values(n_dofs_per_cell);	  
    std::vector<Tensor<1, dealdim> > phi_grads(n_dofs_per_cell);
    
    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
    {
      for (unsigned int k = 0; k < n_dofs_per_cell; k++)
      {
	phi_values[k] = state_fe_values.shape_value(k, q_point);
	phi_grads[k] = state_fe_values.shape_grad(k, q_point);
      }
      
      for (unsigned int i = 0; i < n_dofs_per_cell; i++)
      {
	for (unsigned int j = 0; j < n_dofs_per_cell; j++)
	{
	  local_entry_matrix(i, j) += scale * ((phi_grads[j] * phi_grads[i]) + 2*_uvalues[q_point]*phi_values[j]*phi_values[i]) * state_fe_values.JxW(q_point);
	}
      }
    }
  }


  void CellRightHandSide(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
			 dealii::Vector<double> &local_cell_vector,
			 double scale)
  {
    assert(this->_problem_type == "state");
    
    const DOpEWrapper::FEValues<dealdim> & fe_values = cdc.GetFEValuesState();
    unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
    unsigned int n_q_points = cdc.GetNQPoints();
    
    RightHandSideFunction fvalues;
    fvalues.SetTime(_my_time);
    
    for (unsigned int q_point=0;q_point<n_q_points; ++q_point)
    {
      const Point<2> quadrature_point=fe_values.quadrature_point(q_point);
      for(unsigned int i = 0; i < n_dofs_per_cell; i++)
      {

	local_cell_vector(i) += scale * fvalues.value(quadrature_point) * fe_values.shape_value(i,q_point)
	  * fe_values.JxW(q_point);
      }
    }
  }
  
  void CellTimeEquation(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
			dealii::Vector<double> &local_cell_vector,
			double scale)
  {
    assert(this->_problem_type == "state");
    
    const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
    unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
    unsigned int n_q_points = cdc.GetNQPoints();
    
    _uvalues.resize(n_q_points);
    
    cdc.GetValuesState("last_newton_solution", _uvalues);
    
    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
    {
      for (unsigned int i = 0; i < n_dofs_per_cell; i++)
      {
	const double phi_i = state_fe_values.shape_value(i, q_point);
	local_cell_vector(i) += scale * (_uvalues[q_point] * phi_i)
	  * state_fe_values.JxW(q_point);
      }
    }
  }
  
  void CellTimeEquation_U(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
			  dealii::Vector<double> &local_cell_vector,
			  double scale)
  {
    assert(this->_problem_type == "adjoint");
    
    const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
    unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
    unsigned int n_q_points = cdc.GetNQPoints();
    
    _zvalues.resize(n_q_points);
    
    cdc.GetValuesState("last_newton_solution", _zvalues);
    
    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
    {      
      for (unsigned int i = 0; i < n_dofs_per_cell; i++)
      {
	const double phi_i = state_fe_values.shape_value(i, q_point);
	local_cell_vector(i) += scale * (_zvalues[q_point] * phi_i)
	  * state_fe_values.JxW(q_point);
      }
    }
  }  
  
  void CellTimeEquation_UT(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
			   dealii::Vector<double> &local_cell_vector,
			   double scale)
  {
    assert(this->_problem_type == "tangent");
    
    const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
    unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
    unsigned int n_q_points = cdc.GetNQPoints();
    
    _duvalues.resize(n_q_points);
    
    cdc.GetValuesState("last_newton_solution", _duvalues);

    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
    {
      for (unsigned int i = 0; i < n_dofs_per_cell; i++)
      {
	const double phi_i = state_fe_values.shape_value(i, q_point);
	local_cell_vector(i) += scale * (_duvalues[q_point] * phi_i)
	  * state_fe_values.JxW(q_point);
      }
    }
  }  

  void CellTimeEquation_UTT(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
			    dealii::Vector<double> &local_cell_vector,
			    double scale)
  {
    assert(this->_problem_type == "adjoint_hessian");
    
    const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
    unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
    unsigned int n_q_points = cdc.GetNQPoints();
    
    _dzvalues.resize(n_q_points);
    
    cdc.GetValuesState("last_newton_solution", _dzvalues);
    
    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
    {
      
      for (unsigned int i = 0; i < n_dofs_per_cell; i++)
      {
	const double phi_i = state_fe_values.shape_value(i, q_point);
	local_cell_vector(i) += scale * (_dzvalues[q_point] * phi_i)
	  * state_fe_values.JxW(q_point);
      }
    }
  }

  void CellTimeMatrix(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
		      FullMatrix<double> &local_entry_matrix)
  {
    const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
    unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
    unsigned int n_q_points = cdc.GetNQPoints();
    
    std::vector<double> phi(n_dofs_per_cell);
    
    for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
    {
      for (unsigned int k = 0; k < n_dofs_per_cell; k++)
      {
	phi[k] = state_fe_values.shape_value(k, q_point);
      }
      for (unsigned int i = 0; i < n_dofs_per_cell; i++)
      {
	for (unsigned int j = 0; j < n_dofs_per_cell; j++)
	{
	  local_entry_matrix(j, i) += (phi[i] * phi[j]) * state_fe_values.JxW(q_point);
	}
      }
    }
  }

  void CellTimeEquationExplicit(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& /*cdc*/,
				dealii::Vector<double> &,
				double ) {}
  void CellTimeEquationExplicit_U(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& /*cdc*/,
				  dealii::Vector<double> &,
				  double ) {}
  void CellTimeEquationExplicit_UT(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& /*cdc*/,
				   dealii::Vector<double> &,
				   double ) {}
  void CellTimeEquationExplicit_UTT(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& /*cdc*/,
				    dealii::Vector<double> &,
				    double ) {}
  void CellTimeEquationExplicit_UU(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& /*cdc*/,
				   dealii::Vector<double> &,
				   double ) {}
  void CellTimeMatrixExplicit(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& /*cdc*/,
			      FullMatrix<double> &/*local_entry_matrix*/)  {}
  

  void ControlCellEquation(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
			   dealii::Vector<double> &local_cell_vector,
			   double scale)
  {
    const DOpEWrapper::FEValues<dealdim> & control_fe_values = cdc.GetFEValuesControl();
    unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
    unsigned int n_q_points = cdc.GetNQPoints();
    {
      assert((this->_problem_type == "gradient")||(this->_problem_type == "hessian"));
      _funcgradvalues.resize(n_q_points);
      cdc.GetValuesControl("last_newton_solution",_funcgradvalues);
    }
    
    for(unsigned int q_point = 0; q_point < n_q_points; q_point++)
    {
      for(unsigned int i = 0; i < n_dofs_per_cell; i++)
      {
	local_cell_vector(i) += scale *(_funcgradvalues[q_point] * control_fe_values.shape_value (i, q_point))
	  * control_fe_values.JxW(q_point);
      }
    }
  }

  void ControlCellMatrix(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
			 FullMatrix<double> &local_entry_matrix)
  {
    const DOpEWrapper::FEValues<dealdim> & control_fe_values = cdc.GetFEValuesControl();
    unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
    unsigned int n_q_points = cdc.GetNQPoints();
    
    for(unsigned int q_point = 0; q_point < n_q_points; q_point++)
    {
      for(unsigned int i = 0; i < n_dofs_per_cell; i++)
      {
	for(unsigned int j = 0; j < n_dofs_per_cell; j++)
	  {
	    local_entry_matrix(i,j) += control_fe_values.shape_value (i, q_point)*control_fe_values.shape_value (j, q_point)
	      * control_fe_values.JxW(q_point);
	  }
      }
    }
  }


  UpdateFlags GetUpdateFlags() const
  {
    if (this->_problem_type == "state"|| this->_problem_type == "adjoint"|| this->_problem_type == "adjoint_hessian"
	|| this->_problem_type == "tangent")
      				return update_values | update_gradients | update_quadrature_points;
    else if (this->_problem_type == "gradient" || this->_problem_type == "hessian")
      				return update_values | update_quadrature_points;
    else
      throw DOpEException("Unknown Problem Type " + this->_problem_type,
			  "LocalPDE::GetUpdateFlags");
  }
  
  UpdateFlags GetFaceUpdateFlags() const
  {
    if (this->_problem_type == "state"|| this->_problem_type == "adjoint"|| this->_problem_type == "adjoint_hessian"
	|| this->_problem_type == "tangent" || this->_problem_type == "gradient" || this->_problem_type == "hessian")
      return update_default;
    else
      throw DOpEException("Unknown Problem Type " + this->_problem_type,
			  "LocalPDE::GetFaceUpdateFlags");
  }
  
  unsigned int GetControlNBlocks() const
  {
    return 1;
  }
  
  unsigned int GetStateNBlocks() const
  {
    return 1;
  }
  
  std::vector<unsigned int>& GetControlBlockComponent()
  {
    return _block_components;
  }
  const std::vector<unsigned int>& GetControlBlockComponent() const
  {
    return _block_components;
  }
  std::vector<unsigned int>& GetStateBlockComponent()
  {
    return _state_block_components;
  }
  const std::vector<unsigned int>& GetStateBlockComponent() const
  {
    return _state_block_components;
  }
  
  
  void SetTime(double t) const
  {
    _my_time=t;}
  
private:
  vector<double> _fvalues;
  vector<double> _uvalues;
  vector<double> _qvalues;
  vector<double> _dqvalues;
  vector<double> _zvalues;
  vector<double> _dzvalues;
  vector<double> _duvalues;
  vector<double> _funcgradvalues;
  mutable double _my_time;

  vector<Tensor<1, dealdim> > _ugrads;
  vector<Tensor<1, dealdim> > _zgrads;
  vector<Tensor<1, dealdim> > _dugrads;
  vector<Tensor<1, dealdim> > _dzgrads;
  
  vector<unsigned int> _state_block_components;
  vector<unsigned int> _block_components;
};
#endif
