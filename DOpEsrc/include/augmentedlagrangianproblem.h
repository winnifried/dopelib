#ifndef _AugmentedLagrangianProblem_H_
#define _AugmentedLagrangianProblem_H_

namespace DOpE
{
  /**
   * Class to compute time dependent problems with an augmented Lagrangian 
   * for the inequality constraints
   * This class already implements a hyperbolic block separable approximation
   * of the reduced cost functional following K. Svanberg.
   *
   * @tparam <CONSTRAINTACCESSOR>  An object that gives information on how to use the constraint vector
   * @tparam <OPTPROBLEM>  The problem to deal with.
   * @tparam <dopedim>     The dimension for the control variable.
   * @tparam <dealdim>     The dimension of the state variable.
   * @tparam <localdim>    The dimension of  the control-constraint Matrix-Variable  (i.e. the root of the 
   *                       block size for the block seperable approximation)
   *
   */
  template<typename CONSTRAINTACCESSOR,typename STH, typename OPTPROBLEM, int dopedim, int dealdim, int localdim>
    class AugmentedLagrangianProblem
  {
  public:
    AugmentedLagrangianProblem<CONSTRAINTACCESSOR,STH,OPTPROBLEM, dopedim, dealdim,localdim>(OPTPROBLEM& OP,CONSTRAINTACCESSOR& CA) : _OP(OP), _CA(CA) { _p = 1.; _rho = 0.;}
    ~AugmentedLagrangianProblem<CONSTRAINTACCESSOR,STH,OPTPROBLEM, dopedim, dealdim,localdim> () {}

    /******************************************************/

    std::string GetName() const { return "AugmentedLagrangian"; }

    /******************************************************/
    
    //TODO This is Pfush needed to split into different subproblems and allow optproblem to
      //be substituted as any of these problems. Can be removed once the splitting is complete.
    AugmentedLagrangianProblem<CONSTRAINTACCESSOR,STH,OPTPROBLEM, dopedim, dealdim,localdim>&
      GetBaseProblem() { return *this; }
      
    /******************************************************/
    
    void ReInit(std::string algo_type) { _OP.ReInit(algo_type); }
   
    /******************************************************/
   
    void RegisterOutputHandler(DOpEOutputHandler<dealii::BlockVector<double> >* OH) { _OP.RegisterOutputHandler(OH); }
   
    /******************************************************/
    
    void RegisterExceptionHandler(DOpEExceptionHandler<dealii::BlockVector<double> >* OH) { _OP.RegisterExceptionHandler(OH); }
   
    /******************************************************/
   
     void SetType(std::string type,unsigned int num=0) { _OP.SetType(type,num); }

     /**
      * Sets the value of the Augmented Lagrangian Value
      * It is assumed that p is positive.
      */
     void SetValue(double p, std::string name) { 
       if("p"==name)
       { 
	 assert(p>0.); 
 	 _p = p;
       }
       else if ("mma_functional" == name)
       {
	 _J = p;
       }
       else if ("rho" == name)
       {
	 _rho = p;
       }
       else
       {
	 throw DOpEException("Unknown value "+name,"AumentedLagrangianProblem::SetType");
       }
     }

 //   /**
 //    * Computes the FE values on a cell.
 //    * @param cell      Reference of the actual cell.
 //    */
 //   void ComputeCellFEValues(const std::vector<typename dealii::DoFHandler<dealdim>::active_cell_iterator>& cell)
 //   {
 //     _OP.ComputeCellFEValues(cell);
 //   }

    //TODO Pfush
    void ComputeReducedGlobalConstraintHessian(const ConstraintVector<dealii::BlockVector<double> >& constraints, 
						dealii::Vector<double>& hessian)
    {
      const dealii::Vector<double>& constr = constraints.GetGlobalConstraints();

      for(unsigned int i =0; i < constr.size(); i++)
      {
	double Z = (constr(i)+_p)/(-1.*_p*_p);
	hessian(i) = -2.*_p*_p*Z*Z*Z;
      }
    }

    //TODO Pfush
    void ComputeReducedConstraintGradient(const ConstraintVector<dealii::BlockVector<double> >& direction, 
					  const ConstraintVector<dealii::BlockVector<double> >& constraints, 
					  ConstraintVector<dealii::BlockVector<double> >& gradient)
    {

      {
      const dealii::BlockVector<double>& dir = direction.GetSpacialVector("local");
      const dealii::BlockVector<double>& constr = constraints.GetSpacialVector("local");
      dealii::BlockVector<double>& grad = gradient.GetSpacialVector("local");

      Tensor<2,localdim> local_constraints, local_dir;
      Tensor<2,localdim> Z, identity; 
      identity = 0;
	  for(unsigned int i = 0; i<localdim; i++)
	    identity[i][i] = _p;

      for(unsigned int i = 0 ; i < _CA.GetNLocalControlConstraintBlocks(&dir); i++)
      {
	_CA.CopyLocalConstraintToTensor(constr,local_constraints,i);
	_CA.CopyLocalConstraintToTensor(dir,local_dir,i);
	Z = local_constraints+identity;
	Z *= -1./(_p*_p);
	local_constraints = _p*_p*Z*local_dir*Z;
	assert(local_constraints[0][0] >= 0.);
	_CA.CopyTensorToLocalConstraint(local_constraints,grad,i);
      }
      }
      {
      //Loop over global constraints.     
      const dealii::Vector<double>& dir    = direction.  GetGlobalConstraints();
      const dealii::Vector<double>& constr = constraints.GetGlobalConstraints();
      dealii::Vector<double>& grad         = gradient.   GetGlobalConstraints();

      for(unsigned int i =0; i < dir.size(); i++)
      {
	double dd_phi = (constr(i)+_p)/(-1.*_p*_p);
	dd_phi *= dd_phi;
	dd_phi *= dir(i)*_p*_p;
	assert(dd_phi >= 0.);
	grad(i) = dd_phi;
      }
      }
    }
    /******************************************************/ 
    
    void InitMultiplier(ConstraintVector<dealii::BlockVector<double> >& m, const ControlVector<dealii::BlockVector<double> >& gradient) const
    {
      double scale = 1.;
	
      {
	dealii::BlockVector<double>& bv_m = m.GetSpacialVector("local");
	const dealii::BlockVector<double>& bv_grad = gradient.GetSpacialVector();

	Tensor<2,localdim> identity;
	Vector<double> grad(localdim);

	identity = 0;
	
	//Loop over local control and state constraints
	for(unsigned int i = 0 ; i < _CA.GetNLocalControlConstraintBlocks(&bv_m); i++)
	{
	  _CA.CopyLocalControlToVector(bv_grad,grad,i);
	  for(unsigned int j = 0; j<localdim; j++)
	    identity[j][j] = scale*(1.+fabs(grad(j)));
	  _CA.CopyTensorToLocalConstraint(identity,bv_m,i);
	}
      }
      {
	dealii::Vector<double>& bv_m = m.GetGlobalConstraints();
	scale *= (1+gradient.Norm("infty"));
	for(unsigned int i =0; i < bv_m.size(); i++)
	{
	  bv_m(i) = scale;
	}
      }
      
    }

    /******************************************************/ 

    double AlgebraicFunctional(const std::map<std::string, const dealii::Vector<double>* > &values,
			       const std::map<std::string, const dealii::BlockVector<double>* > &block_values)
    {
      double ret = _J;
      if(dopedim == 0)
	throw DOpEException("Not implemented for this dopedim ","AumentedLagrangianProblem::AlgebraicFunctional");

      //Hohle die benoetigten Werte
      const dealii::BlockVector<double>* linearization_point;
      const dealii::BlockVector<double>* functional_gradient;
      const dealii::BlockVector<double>* mma_multiplier;
      const dealii::BlockVector<double>* eval_point;
      const dealii::Vector<double>* mma_multiplier_global;
      const dealii::Vector<double>* constraint_values_global;
      const dealii::BlockVector<double>* constraint_values;
      const dealii::BlockVector<double>* mma_lower_asymptote;
      const dealii::BlockVector<double>* mma_upper_asymptote;
      {
	eval_point               = GetBlockVector(block_values,"control");
	linearization_point = GetBlockVector(block_values,"mma_control");
	functional_gradient = GetBlockVector(block_values,"mma_functional_gradient");
	mma_multiplier      = GetBlockVector(block_values,"mma_multiplier_local");
   	mma_multiplier_global      = GetVector(values,"mma_multiplier_global");
	constraint_values = GetBlockVector(block_values,"constraints_local");
     	constraint_values_global = GetVector(values,"constraints_global");
	mma_lower_asymptote = GetBlockVector(block_values,"mma_lower_asymptote");
	mma_upper_asymptote = GetBlockVector(block_values,"mma_upper_asymptote");
      }
      //Compute Value of augmented Lagrangian
      {
	//The hyperbolic approximation to the Cost Functional
	Tensor<2,localdim> grad_J, lower_asymptote, upper_asymptote, control, point, grad_J_plus, grad_J_minus;
	Tensor<2,localdim> p_val, q_val, uf, lf, uf_cor, lf_cor,tmp;

	//assume bounds are constant, but choose outside the feasible region
	//_CA.FillLowerUpperControlBound(lower_asymptote,upper_asymptote,true);
	double tau = 0.;
	for(unsigned int i = 0 ; i < _CA.GetNLocalControlBlocks(linearization_point); i++)
	{
	  _CA.CopyLocalControlToTensor(*functional_gradient,grad_J,i);
	  _CA.CopyLocalControlToTensor(*linearization_point,control,i);
	  _CA.CopyLocalControlToTensor(*eval_point,point,i);
	  _CA.CopyLocalControlToTensor(*mma_lower_asymptote,lower_asymptote,i);
	  _CA.CopyLocalControlToTensor(*mma_upper_asymptote,upper_asymptote,i);

	  //Project to positive and negative part! And compute tau = largest eigenvalue of grad_J
	  _CA.ProjectToPositiveAndNegativePart(grad_J,grad_J_plus,grad_J_minus,tau);
	  //anpassen von tau, s.d. -  gradJ + tau Id pos def.
	  tmp = invert(upper_asymptote-lower_asymptote); //Eigentlich upper_bound-lower_bound
	  p_val = (upper_asymptote-control)*(grad_J_plus +_rho/2.*tmp)*(upper_asymptote-control);
	  q_val = (control-lower_asymptote)*(-1.*grad_J_minus+_rho/2.*tmp)*(control-lower_asymptote);

	  uf = invert(upper_asymptote-point);
	  uf_cor = invert(upper_asymptote-control);
	  lf = invert(point-lower_asymptote);
	  lf_cor = invert(control-lower_asymptote);
	  
	  ret += scalar_product(p_val,uf);
	  ret -= scalar_product(p_val,uf_cor);
	  ret += scalar_product(q_val,lf);
	  ret -= scalar_product(q_val,lf_cor);
	}
      }
      {
	//Now multiplier times constraints
	assert(mma_multiplier->n_blocks()==constraint_values->n_blocks());
	for(unsigned int i = 0; i < mma_multiplier->n_blocks(); i++)
	{
	  assert(mma_multiplier->block(i).size()==constraint_values->block(i).size());
	  if(mma_multiplier->block(i).size() > 0)
	  {
	    ret+= mma_multiplier->block(i) * constraint_values->block(i);
	  }
	}
	ret += (*mma_multiplier_global)*(*constraint_values_global);
      }
      
      return ret; 
    }
    /******************************************************/ 

    void AlgebraicResidual(dealii::BlockVector<double>& residual,
			   const std::map<std::string, const dealii::Vector<double>* > &values,
			   const std::map<std::string, const dealii::BlockVector<double>* > &block_values)
    {
      if(this->GetType() == "gradient")
      {
	//Compute the gradient with respect to the controlvariable, e.g. eval_point
	
	if(dopedim == 0)
	  throw DOpEException("Not implemented for this dopedim ","AumentedLagrangianProblem::AlgebraicFunctional");
	
	//Hohle die benoetigten Werte
	const dealii::BlockVector<double>* linearization_point;
	const dealii::BlockVector<double>* functional_gradient;
	const dealii::BlockVector<double>* mma_multiplier;
	const dealii::BlockVector<double>* eval_point;
	const dealii::Vector<double>* mma_multiplier_global;
	const dealii::Vector<double>* constraint_values_global;
	const dealii::BlockVector<double>* constraint_values;
	const dealii::BlockVector<double>* mma_lower_asymptote;
	const dealii::BlockVector<double>* mma_upper_asymptote;
	
	{
	  eval_point               = GetBlockVector(block_values,"control");
	  linearization_point = GetBlockVector(block_values,"mma_control");
	  functional_gradient = GetBlockVector(block_values,"mma_functional_gradient");
	  mma_multiplier      = GetBlockVector(block_values,"mma_multiplier_local");
	  mma_multiplier_global      = GetVector(values,"mma_multiplier_global");
	  constraint_values = GetBlockVector(block_values,"constraints_local");
	  constraint_values_global = GetVector(values,"constraints_global");
	  mma_lower_asymptote = GetBlockVector(block_values,"mma_lower_asymptote");
	  mma_upper_asymptote = GetBlockVector(block_values,"mma_upper_asymptote");
	}
	{
	  //Ableitung der hyperbolischen Approximation an das Cost functional
	  Tensor<2,localdim> grad_J, lower_asymptote, upper_asymptote, control, point, grad_J_plus, grad_J_minus, identity;
	  Tensor<2,localdim> p_val, q_val, uf, lf, tmp, result;
	  identity = 0;
	  for(unsigned int i = 0; i<localdim; i++)
	    identity[i][i] = 1.;
	  double tau = 0.;
	  
	  //assume bounds are constant, but choose outside the feasible region
	  //_CA.FillLowerUpperControlBound(lower_asymptote,upper_asymptote,true);
	  
	  for(unsigned int i = 0 ; i < _CA.GetNLocalControlBlocks(linearization_point); i++)
	  {
	    _CA.CopyLocalControlToTensor(*functional_gradient,grad_J,i);
	    _CA.CopyLocalControlToTensor(*linearization_point,control,i);
	    _CA.CopyLocalControlToTensor(*eval_point,point,i);
	    _CA.CopyLocalControlToTensor(*mma_lower_asymptote,lower_asymptote,i);
	    _CA.CopyLocalControlToTensor(*mma_upper_asymptote,upper_asymptote,i);
	  
	    //Project to positive and negative part! And compute tau = largest eigenvalue of grad_J
	    _CA.ProjectToPositiveAndNegativePart(grad_J,grad_J_plus,grad_J_minus,tau);
	    //anpassen von tau, s.d. -  gradJ + tau Id pos def.
	    tmp = invert(upper_asymptote-lower_asymptote); //Eigentlich upper_bound-lower_bound
	    p_val = (upper_asymptote-control)*(grad_J_plus +_rho/2.*tmp)*(upper_asymptote-control);
	    q_val = (control-lower_asymptote)*(-1.*grad_J_minus+_rho/2.*tmp)*(control-lower_asymptote);

	    uf = invert(upper_asymptote-point);
	    lf = invert(point-lower_asymptote);

	    result = uf*p_val*uf;
	    result -= lf*q_val*lf;
	    //TODO Check derivative!
	    _CA.CopyTensorToLocalControl(result,residual,i);
	    //Das war J'
	  }
	  //Ableitung von Multiplier times constraints
	  Tensor<2,localdim> local_constraints,local_multiplier,local_constraint_derivative;
	  
	  {
	    identity *= _p;
	    Vector<double> local_control_directions(_CA.NLocalDirections());

	    Tensor<2,localdim> Z;
	    Tensor<2,localdim> lhs ;
	    //Local Blocked constraints
	    std::vector<std::vector<unsigned int> > control_to_constraint_index;
	    _CA.LocalControlToConstraintBlocks(linearization_point,control_to_constraint_index);
	    for(unsigned int i = 0 ; i < _CA.GetNLocalControlBlocks(linearization_point); i++)
	    {
	      for( unsigned int j =  0; j < control_to_constraint_index[i].size(); j++)
	      {
		unsigned int index = control_to_constraint_index[i][j];
	      
		_CA.CopyLocalConstraintToTensor(*constraint_values,local_constraints,index);
		_CA.CopyLocalConstraintToTensor(*mma_multiplier,local_multiplier,index);

		//Compute Z
		Z = local_constraints+identity;
		Z *= -1./(_p*_p);

		_CA.GetLocalConstraintDerivative(local_constraint_derivative,*constraint_values,index);
	    
		lhs = Z*local_multiplier*Z*local_constraint_derivative;
		lhs *= _p*_p;
		_CA.AddTensorToLocalControl(lhs,residual,i);
	      }
	    }
	  }
	}
      }
      else if(this->GetType() == "hessian")
      {
	residual = 0.;
	//Compute the gradient with respect to the controlvariable, e.g. eval_point
	
	if(dopedim == 0)
	  throw DOpEException("Not implemented for this dopedim ","AumentedLagrangianProblem::AlgebraicFunctional");
	
	//Hohle die benoetigten Werte
	const dealii::BlockVector<double>* linearization_point;
	const dealii::BlockVector<double>* functional_gradient;
	const dealii::BlockVector<double>* mma_multiplier;
	const dealii::BlockVector<double>* eval_point;
	const dealii::BlockVector<double>* direction;
	const dealii::Vector<double>* mma_multiplier_global;
	const dealii::BlockVector<double>* constraint_values;
	const dealii::Vector<double>* constraint_values_global;
	std::vector<const dealii::BlockVector<double>*> constraint_gradients;
	const dealii::BlockVector<double>* mma_lower_asymptote;
	const dealii::BlockVector<double>* mma_upper_asymptote;

	{
	  eval_point               = GetBlockVector(block_values,"control");
	  linearization_point = GetBlockVector(block_values,"mma_control");
	  functional_gradient = GetBlockVector(block_values,"mma_functional_gradient");
	  mma_multiplier      = GetBlockVector(block_values,"mma_multiplier_local");
	  mma_multiplier_global      = GetVector(values,"mma_multiplier_global");
	  
	  mma_lower_asymptote = GetBlockVector(block_values,"mma_lower_asymptote");
	  mma_upper_asymptote = GetBlockVector(block_values,"mma_upper_asymptote");

	  constraint_values = GetBlockVector(block_values,"constraints_local");
	  constraint_values_global = GetVector(values,"constraints_global");
     
	  constraint_gradients.resize(mma_multiplier_global->size(),NULL);
	  for(unsigned int i = 0; i < constraint_gradients.size(); i++)
	  {
	    std::stringstream name;
	    name << "constraint_gradient_"<<i;
	    constraint_gradients[i] = GetBlockVector(block_values,name.str());
	  }
	  direction = GetBlockVector(block_values,"dq");
	}
	//local in time global in space constraint times multiplier derivative (only phi'' \nabla g \nabla g^T) 
	// the other term is computed elsewhere
	{
	  unsigned int global_block = mma_multiplier->n_blocks()-1;
	  for(unsigned int i = 0; i < constraint_gradients.size(); i++)
	  {
	    //phi''
	    double Z = (constraint_values_global->operator()(i)+_p)/(-1.*_p*_p);
	    double dd_phi = *(constraint_gradients[i])*(*direction);
	    dd_phi *= -_p*_p*2.*Z*Z*Z;
	    dd_phi *= mma_multiplier->block(global_block)(i);
	    residual.add(dd_phi,*(constraint_gradients[i]));
	  }
	}
	//local in time and space constraints times multiplier derivative
	{
	  //Erstmal phi'' \nabla g \nabla g^T
	  Tensor<2,localdim> local_constraints,local_multiplier,local_constraint_derivative, identity, tmp;
	  identity = 0;
	  for(unsigned int i = 0; i<localdim; i++)
	  {
	    identity[i][i] = _p;
	  }
	  
	  Vector<double> local_control_directions(_CA.NLocalDirections());
	  Tensor<2,localdim> Z;
	  Tensor<2,localdim> lhs ;
	  Vector<double> dq(_CA.NLocalDirections());
	  Vector<double> H_dq(_CA.NLocalDirections());
	  
	  //Local Blocked constraints
	  for(unsigned int i = 0 ; i < _CA.GetNLocalControlConstraintBlocks(mma_multiplier); i++)
	  {
	    _CA.CopyLocalConstraintToTensor(*constraint_values,local_constraints,i);
	    _CA.CopyLocalConstraintToTensor(*mma_multiplier,local_multiplier,i);
	    _CA.CopyLocalControlToVector(*direction,dq,i);
	    //Compute Z
	    _CA.CopyLocalConstraintToTensor(*constraint_values,local_constraints,i);
	    
	    Z = local_constraints+identity;
	    Z *= -1./(_p*_p);
	    lhs = Z*local_multiplier*Z;
	    lhs *= _p*_p;
	    local_control_directions = 0.;
	    
	    for(unsigned int j = 0; j < _CA.NLocalDirections(); j++)
	    {
	      _CA.GetLocalConstraintDerivative(local_constraint_derivative,*constraint_values,i,j);
	      tmp = lhs*local_constraint_derivative*Z;
	
	      for(unsigned int k = 0; k < _CA.NLocalDirections(); k++)
	      {
		//phi''\nabla g \nabla g^T
		_CA.GetLocalConstraintDerivative(local_constraint_derivative,*constraint_values,i,k);
		local_control_directions(k) -=  2*scalar_product(tmp,local_constraint_derivative)*dq(k);
	
		// phi' \nabla^2g
		_CA.GetLocalConstraintSecondDerivative(local_constraint_derivative,*constraint_values,i,j,k);
		local_control_directions(k) += scalar_product(lhs,local_constraint_derivative)*dq(k);
	      }
	    }
	    _CA.AddVectorToLocalControl(local_control_directions,residual,i);
	  }
	}
	//derivative of hyperbolic functional approximation
	{
	  Tensor<2,localdim> grad_J, lower_asymptote, upper_asymptote, control, point, grad_J_plus, grad_J_minus;
	  Tensor<2,localdim> p_val, q_val, uf, lf,tmp,dq,result;
	  double tau = 0.;
	  
	  //assume bounds are constant, but choose outside the feasible region
	  //_CA.FillLowerUpperControlBound(lower_asymptote,upper_asymptote,true);
	  
	  for(unsigned int i = 0 ; i < _CA.GetNLocalControlBlocks(linearization_point); i++)
	  {
	    _CA.CopyLocalControlToTensor(*functional_gradient,grad_J,i);
	    _CA.CopyLocalControlToTensor(*linearization_point,control,i);
	    _CA.CopyLocalControlToTensor(*eval_point,point,i);
	    _CA.CopyLocalControlToTensor(*direction,dq,i);
	    _CA.CopyLocalControlToTensor(*mma_lower_asymptote,lower_asymptote,i);
	    _CA.CopyLocalControlToTensor(*mma_upper_asymptote,upper_asymptote,i);

	    //Project to positive and negative part! And compute tau = largest eigenvalue of grad_J
	    _CA.ProjectToPositiveAndNegativePart(grad_J,grad_J_plus,grad_J_minus,tau);

	    //anpassen von tau, s.d. -  gradJ + tau Id pos def.
	    tmp = invert(upper_asymptote-lower_asymptote); //Eigentlich upper_bound-lower_bound
	    p_val = (upper_asymptote-control)*(grad_J_plus +_rho/2.*tmp)*(upper_asymptote-control);
	    q_val = (control-lower_asymptote)*(-1.*grad_J_minus+_rho/2.*tmp)*(control-lower_asymptote);
	  
	    uf = invert(upper_asymptote-point);
	    lf = invert(point-lower_asymptote);

	    result = uf*p_val*uf*dq*uf + uf*dq*uf*p_val*uf;
	    result += lf*q_val*lf*dq*lf + lf*dq*lf*q_val*lf;

	    if(scalar_product(dq,result) <  0.)
	    {
	      std::cout<<"negative block! "<<i<<std::endl;
	      abort();
	    }
	    _CA.AddTensorToLocalControl(result,residual,i);
	    //Das war J''
	  }
	}
      }
      else if(this->GetType() == "hessian_inverse")
      {
	//Compute Residual = H^{-1}dq
	//Where H^{-1} is the Hessian of the mma approximation and the block-local constraints

	//Hohle die benoetigten Werte
	const dealii::BlockVector<double>* linearization_point; 
	const dealii::BlockVector<double>* functional_gradient;
	const dealii::BlockVector<double>* mma_multiplier;
	const dealii::BlockVector<double>* eval_point;
	const dealii::BlockVector<double>* direction;
	const dealii::Vector<double>* mma_multiplier_global;
	const dealii::Vector<double>* constraint_values_global;
	const dealii::BlockVector<double>* constraint_values;
	const dealii::BlockVector<double>* mma_lower_asymptote;
	const dealii::BlockVector<double>* mma_upper_asymptote;
	
	{
	  eval_point               = GetBlockVector(block_values,"control");
	  linearization_point = GetBlockVector(block_values,"mma_control");
	  functional_gradient = GetBlockVector(block_values,"mma_functional_gradient");
	  mma_multiplier      = GetBlockVector(block_values,"mma_multiplier_local");
	  mma_multiplier_global      = GetVector(values,"mma_multiplier_global");
	  constraint_values = GetBlockVector(block_values,"constraints_local");
	  constraint_values_global = GetVector(values,"constraints_global");
	  mma_lower_asymptote = GetBlockVector(block_values,"mma_lower_asymptote");
	  mma_upper_asymptote = GetBlockVector(block_values,"mma_upper_asymptote");

	  direction = GetBlockVector(block_values,"dq");
	}
	
	std::vector<std::vector<unsigned int> > control_to_constraint_index;
	_CA.LocalControlToConstraintBlocks(linearization_point,control_to_constraint_index);
	
	Tensor<2,localdim> H,identity,result;
	Tensor<2,localdim> grad_J, lower_asymptote, upper_asymptote, control, point, grad_J_plus, grad_J_minus;
	Tensor<2,localdim> p_val, q_val, uf, lf,tmp,dq;
	Tensor<2,localdim> Z,local_constraints,local_multiplier,local_constraint_derivative,local_constraint_second_derivative;

	double tau = 0.;
	  
	identity = 0;
	for(unsigned int i = 0; i<localdim; i++)
	{
	  identity[i][i] = _p;
	}
	//assume bounds are constant, but choose outside the feasible region
	//_CA.FillLowerUpperControlBound(lower_asymptote,upper_asymptote,true);

	//Build  local Blocks of the hessian
	for(unsigned int i = 0 ; i < _CA.GetNLocalControlBlocks(linearization_point); i++)
	{
	  H = 0.;
	  
	  //MMA Approx
	  _CA.CopyLocalControlToTensor(*functional_gradient,grad_J,i);
	  _CA.CopyLocalControlToTensor(*linearization_point,control,i);
	  _CA.CopyLocalControlToTensor(*eval_point,point,i);
	  _CA.CopyLocalControlToTensor(*direction,dq,i);
	  _CA.CopyLocalControlToTensor(*mma_lower_asymptote,lower_asymptote,i);
	  _CA.CopyLocalControlToTensor(*mma_upper_asymptote,upper_asymptote,i);

	  //Project to positive and negative part! And compute tau = largest eigenvalue of grad_J
	  _CA.ProjectToPositiveAndNegativePart(grad_J,grad_J_plus,grad_J_minus,tau);
	  //anpassen von tau, s.d. -  gradJ + tau Id pos def.
	  tmp = invert(upper_asymptote-lower_asymptote); //Eigentlich upper_bound-lower_bound
	  p_val = (upper_asymptote-control)*(grad_J_plus +_rho/2.*tmp)*(upper_asymptote-control);
	  q_val = (control-lower_asymptote)*(-1.*grad_J_minus+_rho/2.*tmp)*(control-lower_asymptote);
	  
	  uf = invert(upper_asymptote-point);
	  lf = invert(point-lower_asymptote);
	  H += 2.*uf*p_val*uf*uf;
	  H += 2.*lf*q_val*lf*lf;

	  //Now Constraint derivatives
	  for( unsigned int j =  0; j < control_to_constraint_index[i].size(); j++)
	  {
	    unsigned int index = control_to_constraint_index[i][j];
	    _CA.CopyLocalConstraintToTensor(*constraint_values,local_constraints,index);
	    _CA.CopyLocalConstraintToTensor(*mma_multiplier,local_multiplier,index);
	  	    
	    Z = local_constraints+identity;
	    Z *= -1./(_p*_p);
	    
	    _CA.GetLocalConstraintDerivative(local_constraint_derivative,*constraint_values,index);
	    _CA.GetLocalConstraintSecondDerivative(local_constraint_second_derivative,*constraint_values,index);
	  	
	    H += _p*_p*local_multiplier*Z*Z*(local_constraint_second_derivative-2.*Z*local_constraint_derivative*local_constraint_derivative);
	  }
	  //Computation of H done
	  result = invert(H)*dq;
	  _CA.AddTensorToLocalControl(result,residual,i);
	}
      }
      else
      {
	throw DOpEException("Unknown Type: "+this->GetType(),"AugmentedLagrangianProblem::AlgebraicResidual");
      }
    }
    /******************************************************/ 
    template<typename DATACONTAINER>
      double CellFunctional(const DATACONTAINER& cdc)
    {
      return _OP.CellFunctional(cdc);
    }
    
    /******************************************************/ 

    /**
     * The Augmented Lagrangian Problem is purely algebraic, hence this returns zero
     */       
    double PointFunctional(const std::map<std::string, const dealii::Vector<double>* > &param_values,
			   const std::map<std::string, const dealii::BlockVector<double>* > &domain_values) 
    { return _OP.PointFunctional(param_values,domain_values); }

    /******************************************************/ 

    /**
     * The Augmented Lagrangian Problem is purely algebraic, hence this returns zero
     */ 	
template<typename FACEDATACONTAINER>
    double BoundaryFunctional(const FACEDATACONTAINER& fdc)
    {
      return _OP.BoundaryFunctional(fdc);
    }
    
    /******************************************************/ 

    /**
     * The Augmented Lagrangian Problem is purely algebraic, hence this returns zero
     */ 
template<typename FACEDATACONTAINER>
    double FaceFunctional(const FACEDATACONTAINER& fdc)
    {
      return _OP.FaceFunctional(fdc);
    }

    /******************************************************/ 

template<typename DATACONTAINER>
    void CellEquation(const DATACONTAINER& cdc,
		      dealii::Vector<double> &local_cell_vector,
		      double scale=1., double scale_ico=1.)
    {
      _OP.CellEquation(cdc, local_cell_vector, scale,scale_ico);
    }

    /******************************************************/ 

template<typename DATACONTAINER>
    void CellRhs(const DATACONTAINER& cdc,
		 dealii::Vector<double> &local_cell_vector, 
		 double scale=1.)
    {
      if(this->GetType() == "gradient")
      { 
	dealii::Vector<double> mma_multiplier_global;
	dealii::Vector<double> constraint_values_global;
      
	std::string tmp = this->GetType();
	unsigned int tmp_num = this->GetTypeNum();

	{  
	  cdc.GetParamValues("mma_multiplier_global",mma_multiplier_global);
	  cdc.GetParamValues("constraints_global",constraint_values_global);
	}
	for(unsigned int i = 0; i < mma_multiplier_global.size(); i++)
	{
	  _OP.SetType("global_constraint_gradient",i);
	  double local_scaling = (constraint_values_global(i)+_p)/(-1.*_p*_p);
	  local_scaling *= local_scaling;
	  local_scaling *= mma_multiplier_global(i);
	  local_scaling *= (_p*_p);
	  _OP.CellRhs(cdc,
		      local_cell_vector, 
		      scale*local_scaling);
	}
	_OP.SetType(tmp,tmp_num);
      }
      else if(this->GetType() == "hessian")
      {
	std::string tmp = this->GetType();
	unsigned int tmp_num = this->GetTypeNum();
	dealii::Vector<double> mma_multiplier_global;
	dealii::Vector<double> constraint_values_global;
	
	{  
   	  cdc.GetParamValues("mma_multiplier_global",mma_multiplier_global);
	  cdc.GetParamValues("constraints_global",constraint_values_global);  
	}
	
	for(unsigned int i = 0; i < mma_multiplier_global.size(); i++)
	{
	  _OP.SetType("global_constraint_hessian",i);
	  double local_scaling = (constraint_values_global(i)+_p);
	  local_scaling *= local_scaling;
	  local_scaling *= mma_multiplier_global(i);
	  local_scaling *= 1./(_p*_p);
	  _OP.CellRhs(cdc,
		      local_cell_vector, 
		      scale*local_scaling);
	}
	_OP.SetType(tmp,tmp_num);
      }
      else if(this->GetType() == "global_constraint_gradient")
      {
	_OP.CellRhs(cdc, local_cell_vector, scale);
      }
      else
      {
	throw DOpEException("Not Implemented","AugmentedLagrangianProblem::CellRhs");
      }
      //_OP.CellRhs(param_values, domain_values, n_dofs_per_cell, n_q_points, material_id, cell_diameter, local_cell_vector, scale);
    }
    
    /******************************************************/ 

template<typename DATACONTAINER>
    void CellMatrix(const DATACONTAINER& cdc,
		    dealii::FullMatrix<double> &local_entry_matrix, double scale = 1.,
		    double scale_ico = 1.)
    { 
      _OP.CellMatrix(cdc, local_entry_matrix,scale, scale_ico);              
    }
   
     /******************************************************/ 

    /**
     * Not implemented so far. Returns just _OP.FaceEquation(...). For more information we refer to 
     * the file optproblem.h
     */
template<typename FACEDATACONTAINER>
    void FaceEquation(const FACEDATACONTAINER& fdc,
		      dealii::Vector<double> &local_cell_vector, double scale=1.)
    { 	
      throw DOpEException("Not Implemented","AugmentedLagrangianProblem::FaceEquation");
      _OP.FaceEquation(fdc, local_cell_vector, scale);
    }

    /******************************************************/ 

    /**
     * Not implemented so far. Returns just _OP.FaceRhs(...). For more information we refer to 
     * the file optproblem.h
     */
template<typename FACEDATACONTAINER>
    void FaceRhs(const FACEDATACONTAINER& fdc,
		 dealii::Vector<double> &local_cell_vector, double scale=1.)
    { 
      throw DOpEException("Not Implemented","AugmentedLagrangianProblem::FaceRhs");
      _OP.FaceRhs(fdc,local_cell_vector, scale);
    }

    /******************************************************/ 

    /**
     * Not implemented so far. Returns just _OP.FaceMatrix(...). For more information we refer to 
     * the file optproblem.h
     */
template<typename FACEDATACONTAINER>
    void FaceMatrix(const FACEDATACONTAINER& fdc,
		    dealii::FullMatrix<double> &local_entry_matrix)
    { 
      throw DOpEException("Not Implemented","AugmentedLagrangianProblem::FaceMatrix");
      _OP.FaceMatrix(fdc, local_entry_matrix); 

    }
   
    /******************************************************/ 

template<typename FACEDATACONTAINER>
    void BoundaryEquation(const FACEDATACONTAINER& fdc,
			  dealii::Vector<double> &local_cell_vector, 
			  double scale=1.)
    { 
      throw DOpEException("Not Implemented","AugmentedLagrangianProblem::BoundaryEquation");
      _OP.BoundaryEquation(fdc,local_cell_vector, scale);
    }

    /******************************************************/ 

    /**
     * Not implemented so far. Returns just _OP.FaceMatrix(...). For more information we refer to 
     * the file optproblem.h
     */
template<typename FACEDATACONTAINER>
  void BoundaryRhs(const FACEDATACONTAINER& fdc,
		   dealii::Vector<double> &local_cell_vector, double scale=1.)
    {      
      throw DOpEException("Not Implemented","AugmentedLagrangianProblem::BoundaryRhs");
      _OP.BoundaryRhs(fdc,local_cell_vector, scale);
    }
    
    /******************************************************/ 

template<typename FACEDATACONTAINER>
    void BoundaryMatrix(const FACEDATACONTAINER& fdc,
			dealii::FullMatrix<double> &local_cell_matrix)
    { 
      _OP.BoundaryMatrix(fdc,local_cell_matrix); 
    }
    /******************************************************/ 
     template<typename FACEDATACONTAINER>
       void  InterfaceEquation(const FACEDATACONTAINER& dc,
			       dealii::Vector<double> &local_cell_vector, double scale = 1.)
     {
       _OP.InterfaceEquation(dc,local_cell_vector,scale);
     }
    /******************************************************/ 
     template<typename FACEDATACONTAINER>
        void
        InterfaceMatrix(const FACEDATACONTAINER& dc,
            dealii::FullMatrix<double> &local_entry_matrix)
     {
       _OP.InterfaceMatrix(dc,local_entry_matrix);
     }

    /******************************************************/ 

    void ComputeLocalControlConstraints(dealii::BlockVector<double>& constraints,
					const std::map<std::string, const dealii::Vector<double>* > &/*values*/,
					const std::map<std::string, const dealii::BlockVector<double>* > &block_values)
    {
//      _OP.ComputeLocalConstraints(control,state,constraints);
//TODO here the order should be consistent with the order in the given constraints...
//TODO also we should do this for all blocks...
      
      const dealii::BlockVector<double>& lower_bound =  *GetBlockVector(block_values,"mma_lower_bound");
      const dealii::BlockVector<double>& upper_bound =  *GetBlockVector(block_values,"mma_upper_bound");
      const dealii::BlockVector<double>& control     =  *GetBlockVector(block_values,"control");

      assert(constraints.block(0).size() == 2*control.block(0).size()); 
      for(unsigned int i=0; i < control.block(0).size(); i++)
      {
	//Add Control Constraints, such that if control is feasible all  entries are not positive!
	constraints.block(0)(i) = lower_bound.block(0)(i) - control.block(0)(i);
	constraints.block(0)(control.block(0).size()+i) = control.block(0)(i) - upper_bound.block(0)(i);
      }
    }
    
   /******************************************************/ 
        
    void GetControlBoxConstraints(dealii::BlockVector<double>& lb, dealii::BlockVector<double>& ub) const
    {
      //abort();
      lb = GetAuxiliaryControl("mma_lower_bound")->GetSpacialVector();
      ub = GetAuxiliaryControl("mma_upper_bound")->GetSpacialVector();
      //There should be the local constraints...
      //_OP.GetControlBoxConstraints(lb, ub);
    }

   /******************************************************/ 

    /**
     * A pointer to the quadrature rule for domain values
     *
     * @return A const pointer to the QuadratureFormula()
     */ 
    const dealii::SmartPointer<const dealii::Quadrature<dealdim> > GetQuadratureFormula() const{ return _OP.GetQuadratureFormula(); }
 
    /******************************************************/ 

    /**
     * A pointer to the quadrature rule for face- and boundary-face values
     *
     * @return A const pointer to the FaceQuadratureFormula()
     */ 
    const dealii::SmartPointer<const dealii::Quadrature<dealdim-1> > GetFaceQuadratureFormula() const{ return _OP.GetFaceQuadratureFormula(); }
    
    /******************************************************/ 

    /**
     * A pointer to the whole FESystem 
     *
     * @return A const pointer to the FESystem()
     */ 
    const dealii::SmartPointer<const DOpEWrapper::FiniteElement<dealdim> > GetFESystem() const{ return _OP.GetFESystem(); }

    /******************************************************/ 

    /**
     * This function determines whether a loop over all faces is required or not. 
     *
     * @return Returns whether or not this functional has components on faces between elements.
     *         The default value is false.
     */
    bool HasFaces() const{ return _OP.HasFaces(); }
    bool HasInterfaces() const { return _OP.HasInterfaces(); }
    
    /******************************************************/ 

    /**
     * This function returns the update flags for domain values 
     * for the computation of shape values, gradients, etc.
     * For detailed explication, please visit `Finite element access/FEValues classes' in 
     * the deal.ii manual.
     *
     * @return Returns the update flags to use in a computation.
     */
    dealii::UpdateFlags GetUpdateFlags() const{ return _OP.GetUpdateFlags(); }

    /******************************************************/ 
    
    /**
     * This function returns the update flags for face values 
     * for the computation of shape values, gradients, etc.
     * For detailed explication, please visit 
     * `FEFaceValues< dim, spacedim > Class Template Reference' in 
     * the deal.ii manual.
     *
     * @return Returns the update flags for faces to use in a computation.
     */
    dealii::UpdateFlags GetFaceUpdateFlags() const{ return _OP.GetFaceUpdateFlags(); }

    /******************************************************/ 
    
    /**
     * A std::vector of integer values which contains the colors of Dirichlet boundaries.
     *
     * @return Returns the Dirichlet Colors.
     */
    const std::vector<unsigned int>& GetDirichletColors() const{ return _OP.GetDirichletColors(); }

    /******************************************************/ 
    
    /**
     * A std::vector of boolean values to decide at which parts of the boundary and solutions variables 
     * Dirichlet values should be applied.
     *
     * @return Returns a component mask for each boundary color.
     */
    const std::vector<bool>& GetDirichletCompMask(unsigned int color) const{ return _OP.GetDirichletCompMask(color); }

    /******************************************************/ 
    
    /**
     * This dealii::Function of dimension `dealdim' knows what Dirichlet values to apply 
     * on each boundary part with color 'color'.
     *
     * @return Returns a dealii::Function of Dirichlet values of the boundary part with color 'color'.
     */
    const dealii::Function<dealdim>& GetDirichletValues(unsigned int color,
//							const DOpEWrapper::DoFHandler<dopedim> & control_dof_handler,
//							const DOpEWrapper::DoFHandler<dealdim> &state_dof_handler,
							const std::map<std::string, const dealii::Vector<double>* > &param_values,
							const std::map<std::string, const dealii::BlockVector<double>* > &domain_values) const
    { return _OP.GetDirichletValues(color,param_values,domain_values); }
     
    /******************************************************/ 
    
    /**
     * This dealii::Function of dimension `dealdim' applys the initial values to the PDE- or Optimization
     * problem, respectively.
     *
     * @return Returns a dealii::Function of initial values.
     */
    const dealii::Function<dealdim>& GetInitialValues() const{ return _OP.GetInitialValues(); }
   
    /******************************************************/ 
    
    /**
     * A std::vector of integer values which contains the colors of the boundary equation.
     *
     * @return Returns colors for the boundary equation.
     */
    const std::vector<unsigned int>& GetBoundaryEquationColors() const{ return _OP.GetBoundaryEquationColors(); }
    
    /******************************************************/ 
    
    /**
     * A std::vector of integer values which contains the colors of the boundary functionals.
     *
     * @return Returns colors for the boundary functionals.
     */
    const std::vector<unsigned int>& GetBoundaryFunctionalColors() const{ return _OP.GetBoundaryFunctionalColors(); }
    
    /******************************************************/ 
    
    /**
     * This function returns the number of functionals to be considered in the problem.
     *
     * @return Returns the number of functionals.
     */
    unsigned int GetNFunctionals() const{ return 0; }

    /******************************************************/ 
    
    /**
     * This function gets the number of blocks considered in the PDE problem. 
     * Example 1: in fluid problems we have to find velocities and pressure 
     * --> number of blocks is 2.
     * Example 2: in FSI problems we have to find velocities, displacements, and pressure.
     *  --> number of blocks is 3.
     *
     * @return Returns the number of blocks.
     */
    unsigned int GetNBlocks() const{ return _OP.GetNBlocks(); }

    /******************************************************/ 
    
    /**
     * A function which has the number of degrees of freedom for the block `b'.
     *
     * @return Returns the number of DoFs for block `b'.
     */
    unsigned int GetDoFsPerBlock(unsigned int b) const{ return _OP.GetDoFsPerBlock(b); }

    /******************************************************/ 
    
    /**
     * A std::vector which contains the number of degrees of freedom per block.
     *
     * @return Returns a vector with DoFs.
     */
    const std::vector<unsigned int>&  GetDoFsPerBlock() const{ return _OP.GetDoFsPerBlock(); }

    /******************************************************/ 
    
    /**
     * A dealii function. Please visit: ConstraintMatrix in the deal.ii manual.
     *
     * @return Returns a matrix with hanging node constraints.
     */
    const dealii::ConstraintMatrix& GetHangingNodeConstraints() const{ return _OP.GetHangingNodeConstraints(); }
   

    std::string GetType() const{ return _OP.GetType(); }
    unsigned int GetTypeNum() const{ return _OP.GetTypeNum(); }
    std::string GetDoFType() const{ return _OP.GetDoFType(); }
  
    /******************************************************/ 
    
    /**
     * This function describes what type of Functional is considered
     * Here it is computed by algebraic operations on the vectors.
     */
    std::string GetFunctionalType() const
    { 
      if(this->GetType() == "cost_functional" || this->GetType() == "gradient"|| this->GetType() == "hessian"|| this->GetType() == "hessian_inverse") 
	return "algebraic"; 
      return _OP.GetFunctionalType();
    }

    /******************************************************/ 
    
    /**
     * This function is used to name the Functional, this is helpful to distinguish different Functionals in the output.
     *
     * @return A string. This is the name beeing displayed next to the computed values.
     */
    std::string GetFunctionalName() const{ return "Seperable Augmented Lagrangian"; }
 
    /******************************************************/ 
    
    std::string GetConstraintType() const { return _OP.GetConstraintType(); }

    /******************************************************/ 
    
    bool HasControlInDirichletData() const { return _OP.HasControlInDirichletData(); }

    /******************************************************/ 
    
    /**
     * A pointer to the OutputHandler() object.
     *
     * @return The OutputHandler() object.
     */
    DOpEOutputHandler<dealii::BlockVector<double> >* GetOutputHandler() { return _OP.GetOutputHandler(); }
  
    /******************************************************/ 
    
    /**
     * A pointer to the SpaceTimeHandler<dopedim,dealdim>  object.
     *
     * @return The SpaceTimeHandler() object.
     */
    const STH* GetSpaceTimeHandler() const  { return _OP.GetSpaceTimeHandler(); }
  
    /******************************************************/ 
    
    /**
     * A pointer to the SpaceTimeHandler<dopedim,dealdim>  object.
     *
     * @return The SpaceTimeHandler() object.
     */
    STH* GetSpaceTimeHandler() { return _OP.GetSpaceTimeHandler(); }
  
    /******************************************************/ 
        
    void ComputeSparsityPattern (BlockSparsityPattern & sparsity) const { _OP.ComputeSparsityPattern(sparsity); }
  
    /******************************************************/ 
        
    bool IsFeasible(const ConstraintVector<dealii::BlockVector<double> >&  g) const 
    { 
      //We do require that x is choosen in the domain of phi(g), this is equivalent to phi(g(x)) > -p
      return _OP.IsLargerThan(g,-_p);
    }
      
    /******************************************************/ 
    
    void PostProcessConstraints(ConstraintVector<dealii::BlockVector<double> >&  g) const
    {
       //_OP.PostProcessConstraints(g,process_global_in_time_constraints);
      {
	dealii::BlockVector<double>& bv_g = g.GetSpacialVector("local");
	
	Tensor<2,localdim> tmp,tmp2,identity;
	identity = 0;
	for(unsigned int i = 0; i<localdim; i++)
	  identity[i][i] = 1.;
	//Loop over local control and state constraints
	for(unsigned int i = 0 ; i < _CA.GetNLocalControlConstraintBlocks(&bv_g); i++)
	{
	  _CA.CopyLocalConstraintToTensor(bv_g,tmp,i);
	  tmp -= _p*identity;
	  tmp2 = invert(tmp);
	  tmp2 *= -_p*_p;
	  tmp2 -= _p*identity;
	  _CA.CopyTensorToLocalConstraint(tmp2,bv_g,i);
	}
      }
      {
	dealii::Vector<double>& bv_g = g.GetGlobalConstraints();
	//Loop over global constraints.
	for(unsigned int i =0; i < bv_g.size(); i++)
	{
	  bv_g(i) = -_p*_p/(bv_g(i)-_p)-_p;
	}
      }
    }
    void AddAuxiliaryControl(const ControlVector<dealii::BlockVector<double> >* c, std::string name) { _OP.AddAuxiliaryControl(c,name);  }
    const ControlVector<dealii::BlockVector<double> >* GetAuxiliaryControl(std::string name) const { return _OP.GetAuxiliaryControl(name);  }
    void AddAuxiliaryConstraint(const ConstraintVector<dealii::BlockVector<double> >* c, std::string name) { _OP.AddAuxiliaryConstraint(c,name);  }
    void DeleteAuxiliaryControl(std::string name){ _OP.DeleteAuxiliaryControl(name);  }
    void DeleteAuxiliaryConstraint(std::string name){ _OP.DeleteAuxiliaryConstraint(name);  }
    const ConstraintVector<dealii::BlockVector<double> >* GetAuxiliaryConstraint(std::string name) { return _OP.GetAuxiliaryConstraint(name); }

    template<typename INTEGRATOR>
      void AddAuxiliaryToIntegrator(INTEGRATOR& integrator) { _OP.AddAuxiliaryToIntegrator<INTEGRATOR>(integrator);  }
    template<typename INTEGRATOR>
       void DeleteAuxiliaryFromIntegrator(INTEGRATOR& integrator) { _OP.DeleteAuxiliaryFromIntegrator<INTEGRATOR>(integrator);  }
   
    /*************************************************************************************/
    bool GetFEValuesNeededToBeInitialized() const
    {
      return _OP.GetFEValuesNeededToBeInitialized();
    }

    /******************************************************/
    void SetFEValuesAreInitialized()
    {
      _OP.SetFEValuesAreInitialized();
    }
    
    /******************************************************/
    const std::map<std::string, unsigned int>&
    GetFunctionalPosition() const
    {
      return _OP.GetFunctionalPosition();
    }

  private:
    OPTPROBLEM& _OP;
    CONSTRAINTACCESSOR& _CA;
    double _p, _J, _rho;

    const dealii::BlockVector<double>* GetBlockVector( const std::map<std::string, const BlockVector<double>* >& values, std::string name)
    {	
      typename std::map<std::string, const BlockVector<double>* >::const_iterator it = values.find(name);
      if(it == values.end())
	{
	  throw DOpEException("Did not find " + name,"AugmentedLagrangian::GetBlockVector");
	}
	return it->second;
    }
    const dealii::Vector<double>* GetVector( const std::map<std::string, const Vector<double>* >& values, std::string name)
    {
      typename std::map<std::string, const Vector<double>* >::const_iterator it = values.find(name);
      if(it == values.end())
	{
	  throw DOpEException("Did not find " + name,"AugmentedLagrangian::GetVector");
	}
	return it->second;
    }
  };
}

#endif
