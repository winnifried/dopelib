#ifndef _LOCALPDE_
#define _LOCALPDE_

#include "pdeinterface.h"

using namespace std;
using namespace dealii;
using namespace DOpE;

template<typename VECTOR, int dopedim, int dealdim>
  class LocalPDE : public PDEInterface<CellDataContainer,FaceDataContainer,dealii::DoFHandler<dealdim>, VECTOR, dopedim,dealdim>
  {
  public:
  LocalPDE() : _block_components(2,0), _c_block_components(1,0)
      {  
	_alpha = 5.e-3;
      }

    void CellEquation(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
		      dealii::Vector<double> &local_cell_vector,
		      double scale, double /*scale_ico*/)
    {
      const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
      unsigned int n_q_points = cdc.GetNQPoints();
      {
	//Reading data
	assert(this->_problem_type == "state");
	_uvalues.resize(n_q_points,Vector<double> (2));
	_fvalues.resize(n_q_points);
	_ugrads.resize(n_q_points,vector<Tensor<1, 2> > (2));
	//Geting u
	cdc.GetGradsState("last_newton_solution",_ugrads);
	cdc.GetValuesState("last_newton_solution",_uvalues);
      }

      for(unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
	  _fvalues[q_point] = ud(state_fe_values.quadrature_point(q_point)(0),state_fe_values.quadrature_point(q_point)(1));
	for(unsigned int i = 0; i < n_dofs_per_cell; i++)
	{
	  const FEValuesExtractors::Scalar u(0);
	  const FEValuesExtractors::Scalar z(1);
	  local_cell_vector(i) += scale *(_ugrads[q_point][0] * state_fe_values[u].gradient (i, q_point)
					  + 1./_alpha*_uvalues[q_point](1)*state_fe_values[u].value (i, q_point)
					  + _ugrads[q_point][1] * state_fe_values[z].gradient (i, q_point) 
					  + (_fvalues[q_point]-_uvalues[q_point](0))*state_fe_values[z].value (i, q_point)
	    )
	    * state_fe_values.JxW(q_point);
	}
      }
    }

    void CellRightHandSide(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
			   dealii::Vector<double> &local_cell_vector,
			   double scale)
    {
      //const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
      //unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
      //unsigned int n_q_points = cdc.GetNQPoints();
      //double x = 0.;
      //double y = 0.;
      //for (unsigned int q_point=0;q_point<n_q_points; ++q_point)
      //{
      //	x = state_fe_values.quadrature_point(q_point)(0);
      //	y = state_fe_values.quadrature_point(q_point)(1);
      //	
      //	for(unsigned int i = 0; i < n_dofs_per_cell; i++)
      //	{
      //	  local_cell_vector(i) += scale *(rhs(x,y)*state_fe_values.value (i, q_point))
      //	    * state_fe_values.JxW(q_point);
      //	}
      //}
    }

    void CellMatrix(const CellDataContainer<dealii::DoFHandler<dealdim>, VECTOR, dealdim>& cdc,
		    FullMatrix<double> &local_entry_matrix, double scale, double /*scale_ico*/)
    {
      const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
      unsigned int n_q_points = cdc.GetNQPoints();
      
      _uvalues.resize(n_q_points,Vector<double> (2));
      cdc.GetValuesState("last_newton_solution",_uvalues);

      for(unsigned int q_point = 0; q_point < n_q_points; q_point++)
      {
	for(unsigned int i = 0; i < n_dofs_per_cell; i++)
	{
	  for(unsigned int j = 0; j < n_dofs_per_cell; j++)
	  {
	    const FEValuesExtractors::Scalar u(0);
            const FEValuesExtractors::Scalar z(1);
	    local_entry_matrix(i,j) += scale * (state_fe_values[u].gradient (i, q_point) *
				 state_fe_values[u].gradient (j, q_point) *
				 state_fe_values.JxW (q_point));
	    local_entry_matrix(i,j) += 1./_alpha*scale * (state_fe_values[u].value (i, q_point) *
				 state_fe_values[z].value (j, q_point) *
				 state_fe_values.JxW (q_point));
	    local_entry_matrix(i,j) += scale * (state_fe_values[z].gradient (i, q_point) *
				 state_fe_values[z].gradient (j, q_point) *
				 state_fe_values.JxW (q_point));
	    local_entry_matrix(i,j) -= scale * (state_fe_values[z].value (i, q_point) *
				 state_fe_values[u].value (j, q_point) *
				 state_fe_values.JxW (q_point));
	   
	  }
	}
      }
    }


    UpdateFlags GetUpdateFlags() const
    {
      if((this->_problem_type == "adjoint") || (this->_problem_type == "state")
	 || (this->_problem_type == "tangent")|| (this->_problem_type == "adjoint_hessian")||(this->_problem_type == "hessian"))
	return update_values | update_gradients | update_quadrature_points;
      else if((this->_problem_type == "gradient"))
	return update_values | update_quadrature_points;
      else
	throw DOpEException("Unknown Problem Type "+this->_problem_type ,"LocalPDE::GetUpdateFlags");
    }

    unsigned int GetControlNBlocks() const{ return 1;}
    unsigned int GetStateNBlocks() const{ return 1;}
    std::vector<unsigned int>& GetControlBlockComponent(){ return _c_block_components; }
    const std::vector<unsigned int>& GetControlBlockComponent() const{ return _c_block_components; }
    std::vector<unsigned int>& GetStateBlockComponent(){ return _block_components; }
    const std::vector<unsigned int>& GetStateBlockComponent() const{ return _block_components; }

  protected:

  private:
    double ud(double x, double y)
    {
      double ret=0.;
      
      //local grid raster
      double delta = 1.;
      
      //D
      if( (x > 2)&& (x < 6))
      {
	if( ( y > 4) && (y < 10))
	{
	  if ( x < 3)
	  {
	    ret = 1.;
	  }
	  else
	  {
	    if(x < 5)
	    {
	      if(y < 5 || y > 9 )
	      {
		ret = 1.;
	      }
	    }
	    else
	    {
	      if(y > 5 && y < 9 )
	      {
		ret = 1.;
	      }
	    }
	  }
	}
      }
      //O
      if( (x > 7)&& (x < 11))
      {
	if( ( y > 4) && (y < 10))
	{
	  if( x < 8 || x > 10)
	  {
	    if(y > 5 && y < 9)
	    {
	      ret = 1.;
	    }
	  }
	  else
	  {
	    if(y < 5 || y > 9)
	    {
	      ret = 1.;
	    }
	  }
	}
      }
      //p
      if( (x > 12)&& (x < 15))
      {
	if( ( y > 2) && (y < 7))
	{
	  if(x < 13)
	  {
	    ret = 1.;
	  }
	  else if( x < 14 )
	  {
	    if( y>4)
	    {
	      if( (y < 5) || ( y > 6))
	      {
		ret = 1.;
	      }
	    }
	  }
	  else if (y > 4)
	  {
	    ret = 1.;
	  }
	}
      }
      //E
      if( (x > 16)&& (x < 19))
      {
	if( ( y > 4) && (y < 10))
	{
	  if( x < 17)
	  {
	    ret = 1.;
	  }
	  else if(x< 18)
	  {
	    if( (y<5)||(y>9)||( (y > 6.5) && (y < 7.5)) )
	    {
	      ret = 1.;
	    }
	  }
	  else if(x< 19)
	  {
	    if((y<5)||(y>9))
	    {
	      ret = 1.;
	    }
	  }
	}
      }
      //l
      if( (x > 21)&& (x < 22))
      {
	if( ( y > 4) && (y < 10))
	{
	  ret =1.;
	}
      }
      //i
      if( (x > 23)&& (x < 24))
      {
	if( (( y > 4) && (y < 7))||(( y > 8) && (y < 9)))
	{
	  ret =1.;
	}
      }
      //b
      if( (x > 25)&& (x < 28))
      {
	if( ( y > 4) && (y < 10))
	{
	  if(x<26)
	  {
	    ret = 1.;
	  }
	  else if(x <27)
	  {
	    if( y < 7)
	    {
	      if( (y < 5) || (y > 6))
	      {
		ret = 1.;
	      }
	    }
	  }
	  else if(y < 7)
	  {
	    ret = 1.;
	  }
	}
      }

      return ret;
    }

    double _alpha;
    vector<Vector<double> > _uvalues;
    vector<double> _fvalues;
    vector<vector<Tensor<1,dealdim> > > _ugrads;
    
    vector<unsigned int> _block_components;
    vector<unsigned int> _c_block_components;   
  };
#endif
