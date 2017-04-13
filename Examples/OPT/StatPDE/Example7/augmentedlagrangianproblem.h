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

#ifndef AugmentedLagrangianProblem_H_
#define AugmentedLagrangianProblem_H_

namespace DOpE
{
  /**
   * Class to compute time dependent problems with an augmented Lagrangian
   * for the inequality constraints
   * This class already implements a hyperbolic block separable approximation
   * of the reduced cost functional following K. Svanberg.
   *
   * @tparam <CONSTRAINTACCESSOR>  An object that gives information on how to use the constraint vector
   * @tparam <STH>                 The space time handler.
   * @tparam <OPTPROBLEM>          The problem to deal with, i.e., an OptProblemContainer.
   * @tparam <dopedim>             The dimension for the control variable.
   * @tparam <dealdim>             The dimension of the state variable.
   * @tparam <localdim>            The dimension of  the control-constraint Matrix-Variable
   *                               (i.e. the root of the
   *                               block size for the block seperable approximation)
   *
   */
  template<typename CONSTRAINTACCESSOR, typename STH, typename OPTPROBLEM,
           int dopedim, int dealdim, int localdim>
  class AugmentedLagrangianProblem
  {
  public:
    AugmentedLagrangianProblem<CONSTRAINTACCESSOR, STH, OPTPROBLEM, dopedim,
                               dealdim, localdim>(OPTPROBLEM &OP, CONSTRAINTACCESSOR &CA) :
                                 OP_(OP), CA_(CA)
    {
      p_ = 1.;
      rho_ = 0.;
    }
    ~AugmentedLagrangianProblem<CONSTRAINTACCESSOR, STH, OPTPROBLEM,
    dopedim, dealdim, localdim>()
    {
    }

    /******************************************************/

    std::string
    GetName() const
    {
      return "AugmentedLagrangian";
    }

    /******************************************************/

    //TODO This is Pfush needed to split into different subproblems and allow optproblem to
    //be substituted as any of these problems. Can be removed once the splitting is complete.
    AugmentedLagrangianProblem<CONSTRAINTACCESSOR, STH, OPTPROBLEM, dopedim,
    dealdim, localdim>&
    GetBaseProblem()
    {
      return *this;
    }

    /******************************************************/

    void
    ReInit(std::string algo_type)
    {
      OP_.ReInit(algo_type);
    }

    /******************************************************/

    void
    RegisterOutputHandler(
      DOpEOutputHandler<dealii::BlockVector<double> > *OH)
    {
      OP_.RegisterOutputHandler(OH);
    }

    /******************************************************/

    void
    RegisterExceptionHandler(
      DOpEExceptionHandler<dealii::BlockVector<double> > *OH)
    {
      OP_.RegisterExceptionHandler(OH);
    }

    /******************************************************/

    void
    SetType(std::string type, unsigned int num = 0)
    {
      OP_.SetType(type, num);
    }

    /**
     * Sets the value of the Augmented Lagrangian Value
     * It is assumed that p is positive.
     */
    void
    SetValue(double p, std::string name)
    {
      if ("p" == name)
        {
          assert(p > 0.);
          p_ = p;
        }
      else if ("mma_functional" == name)
        {
          J_ = p;
        }
      else if ("rho" == name)
        {
          rho_ = p;
        }
      else
        {
          throw DOpEException("Unknown value " + name,
                              "AumentedLagrangianProblem::SetType");
        }
    }

    /**
           * Calculates the hessian of the global constraints in the augmented Lagrangian
     */
    void
    ComputeReducedGlobalConstraintHessian(
      const ConstraintVector<dealii::BlockVector<double> > &constraints,
      dealii::Vector<double> &hessian)
    {
      const dealii::Vector<double> &constr =
        constraints.GetGlobalConstraints();

      for (unsigned int i = 0; i < constr.size(); i++)
        {
          double Z = (constr(i) + p_) / (-1. * p_ * p_);
          hessian(i) = -2. * p_ * p_ * Z * Z * Z;
        }
    }

    /**
           * Calculates the gradient of the constraints in the augmented Lagrangian
     */
    void
    ComputeReducedConstraintGradient(
      const ConstraintVector<dealii::BlockVector<double> > &direction,
      const ConstraintVector<dealii::BlockVector<double> > &constraints,
      ConstraintVector<dealii::BlockVector<double> > &gradient)
    {

      {
        const dealii::BlockVector<double> &dir = direction.GetSpacialVector(
                                                   "local");
        const dealii::BlockVector<double> &constr =
          constraints.GetSpacialVector("local");
        dealii::BlockVector<double> &grad = gradient.GetSpacialVector(
                                              "local");

        Tensor<2, localdim> local_constraints, local_dir;
        Tensor<2, localdim> Z, identity;
        identity = 0;
        for (unsigned int i = 0; i < localdim; i++)
          identity[i][i] = p_;

        for (unsigned int i = 0;
             i < CA_.GetNLocalControlConstraintDoFs(&dir); i++)
          {
            CA_.CopyLocalConstraintToTensor(constr, local_constraints, i);
            CA_.CopyLocalConstraintToTensor(dir, local_dir, i);
            Z = local_constraints + identity;
            Z *= -1. / (p_ * p_);
            local_constraints = p_ * p_ * Z * local_dir * Z;
            assert(local_constraints[0][0] >= 0.);
            CA_.CopyTensorToLocalConstraint(local_constraints, grad, i);
          }
      }
      {
        //Loop over global constraints.
        const dealii::Vector<double> &dir =
          direction.GetGlobalConstraints();
        const dealii::Vector<double> &constr =
          constraints.GetGlobalConstraints();
        dealii::Vector<double> &grad = gradient.GetGlobalConstraints();

        for (unsigned int i = 0; i < dir.size(); i++)
          {
            double dd_phi = (constr(i) + p_) / (-1. * p_ * p_);
            dd_phi *= dd_phi;
            dd_phi *= dir(i) * p_ * p_;
            assert(dd_phi >= 0.);
            grad(i) = dd_phi;
          }
      }
    }
    /******************************************************/
    /**
     * Initializes the multiplier used in the augmented Lagrangian
     */
    void
    InitMultiplier(ConstraintVector<dealii::BlockVector<double> > &m,
                   const ControlVector<dealii::BlockVector<double> > &gradient) const
    {
      double scale = 1.;

      {
        dealii::BlockVector<double> &bv_m = m.GetSpacialVector("local");
        const dealii::BlockVector<double> &bv_grad =
          gradient.GetSpacialVector();

        Tensor < 2, localdim > identity;
        Vector<double> grad(localdim);

        identity = 0;

        //Loop over local control and state constraints
        for (unsigned int i = 0;
             i < CA_.GetNLocalControlConstraintDoFs(&bv_m); i++)
          {
            CA_.CopyLocalControlToVector(bv_grad, grad, i);
            for (unsigned int j = 0; j < localdim; j++)
              identity[j][j] = scale * (1. + fabs(grad(j)));
            CA_.CopyTensorToLocalConstraint(identity, bv_m, i);
          }
      }
      {
        dealii::Vector<double> &bv_m = m.GetGlobalConstraints();
        scale *= (1 + gradient.Norm("infty"));
        for (unsigned int i = 0; i < bv_m.size(); i++)
          {
            bv_m(i) = scale;
          }
      }

    }

    /******************************************************/
    /**
     * Evaluates the augmented Lagrangian. It is an algebraic function
     * since we directly operate on the dofs, and the PDE is eliminated by the
     * outer MMA-Algorithm
     */
    double
    AlgebraicFunctional(
      const std::map<std::string, const dealii::Vector<double>*> &values,
      const std::map<std::string, const dealii::BlockVector<double>*> &block_values)
    {
      double ret = J_;
      if (dopedim == 0)
        throw DOpEException("Not implemented for this dopedim ",
                            "AumentedLagrangianProblem::AlgebraicFunctional");

      //Get the required values
      const dealii::BlockVector<double> *linearization_point;
      const dealii::BlockVector<double> *functional_gradient;
      const dealii::BlockVector<double> *mma_multiplier;
      const dealii::BlockVector<double> *eval_point;
      const dealii::Vector<double> *mma_multiplier_global;
      const dealii::Vector<double> *constraint_values_global;
      const dealii::BlockVector<double> *constraint_values;
      const dealii::BlockVector<double> *mma_lower_asymptote;
      const dealii::BlockVector<double> *mma_upper_asymptote;
      {
        eval_point = GetBlockVector(block_values, "control");
        linearization_point = GetBlockVector(block_values, "mma_control");
        functional_gradient = GetBlockVector(block_values,
                                             "mma_functional_gradient");
        mma_multiplier = GetBlockVector(block_values,
                                        "mma_multiplier_local");
        mma_multiplier_global = GetVector(values, "mma_multiplier_global");
        constraint_values = GetBlockVector(block_values,
                                           "constraints_local");
        constraint_values_global = GetVector(values, "constraints_global");
        mma_lower_asymptote = GetBlockVector(block_values,
                                             "mma_lower_asymptote");
        mma_upper_asymptote = GetBlockVector(block_values,
                                             "mma_upper_asymptote");
      }
      //Compute Value of augmented Lagrangian
      {
        //The hyperbolic approximation to the Cost Functional
        Tensor<2, localdim> grad_J, lower_asymptote, upper_asymptote,
               control, point, grad_J_plus, grad_J_minus;
        Tensor<2, localdim> p_val, q_val, uf, lf, uf_cor, lf_cor, tmp;

        //assume bounds are constant, but choose outside the feasible region
        //CA_.FillLowerUpperControlBound(lower_asymptote,upper_asymptote,true);
        double tau = 0.;
        for (unsigned int i = 0;
             i < CA_.GetNLocalControlDoFs(linearization_point); i++)
          {
            CA_.CopyLocalControlToTensor(*functional_gradient, grad_J, i);
            CA_.CopyLocalControlToTensor(*linearization_point, control, i);
            CA_.CopyLocalControlToTensor(*eval_point, point, i);
            CA_.CopyLocalControlToTensor(*mma_lower_asymptote,
                                         lower_asymptote, i);
            CA_.CopyLocalControlToTensor(*mma_upper_asymptote,
                                         upper_asymptote, i);

            //Project to positive and negative part! And compute tau = largest eigenvalue of grad_J
            CA_.ProjectToPositiveAndNegativePart(grad_J, grad_J_plus,
                                                 grad_J_minus, tau);
            //adjust tau, such that. -  gradJ + tau Id is pos def.
            tmp = invert(upper_asymptote - lower_asymptote);
            p_val = (upper_asymptote - control)
                    * (grad_J_plus + rho_ / 2. * tmp)
                    * (upper_asymptote - control);
            q_val = (control - lower_asymptote)
                    * (-1. * grad_J_minus + rho_ / 2. * tmp)
                    * (control - lower_asymptote);

            uf = invert(upper_asymptote - point);
            uf_cor = invert(upper_asymptote - control);
            lf = invert(point - lower_asymptote);
            lf_cor = invert(control - lower_asymptote);

            ret += scalar_product(p_val, uf);
            ret -= scalar_product(p_val, uf_cor);
            ret += scalar_product(q_val, lf);
            ret -= scalar_product(q_val, lf_cor);
          }
      }
      {
        //Now multiplier times constraints
        assert(mma_multiplier->n_blocks() == constraint_values->n_blocks());
        for (unsigned int i = 0; i < mma_multiplier->n_blocks(); i++)
          {
            assert(
              mma_multiplier->block(i).size()
              == constraint_values->block(i).size());
            if (mma_multiplier->block(i).size() > 0)
              {
                ret += mma_multiplier->block(i) * constraint_values->block(i);
              }
          }
        ret += (*mma_multiplier_global) * (*constraint_values_global);
      }

      return ret;
    }
    /******************************************************/

    /**
     * Evaluates the residual of the Algebraic problem
     */
    void
    AlgebraicResidual(dealii::BlockVector<double> &residual,
                      const std::map<std::string, const dealii::Vector<double>*> &values,
                      const std::map<std::string, const dealii::BlockVector<double>*> &block_values)
    {
      if (this->GetType() == "gradient")
        {
          //Compute the gradient with respect to the controlvariable, e.g. eval_point

          if (dopedim == 0)
            throw DOpEException("Not implemented for this dopedim ",
                                "AumentedLagrangianProblem::AlgebraicFunctional");

          //get the required values
          const dealii::BlockVector<double> *linearization_point;
          const dealii::BlockVector<double> *functional_gradient;
          const dealii::BlockVector<double> *mma_multiplier;
          const dealii::BlockVector<double> *eval_point;
          const dealii::BlockVector<double> *constraint_values;
          const dealii::BlockVector<double> *mma_lower_asymptote;
          const dealii::BlockVector<double> *mma_upper_asymptote;

          {
            eval_point = GetBlockVector(block_values, "control");
            linearization_point = GetBlockVector(block_values, "mma_control");
            functional_gradient = GetBlockVector(block_values,
                                                 "mma_functional_gradient");
            mma_multiplier = GetBlockVector(block_values,
                                            "mma_multiplier_local");
            constraint_values = GetBlockVector(block_values,
                                               "constraints_local");
            mma_lower_asymptote = GetBlockVector(block_values,
                                                 "mma_lower_asymptote");
            mma_upper_asymptote = GetBlockVector(block_values,
                                                 "mma_upper_asymptote");
          }
          {
            //Derivative of the hyperbolic approximation to the cost functional
            Tensor<2, localdim> grad_J, lower_asymptote, upper_asymptote,
                   control, point, grad_J_plus, grad_J_minus, identity;
            Tensor<2, localdim> p_val, q_val, uf, lf, tmp, result;
            identity = 0;
            for (unsigned int i = 0; i < localdim; i++)
              identity[i][i] = 1.;
            double tau = 0.;

            for (unsigned int i = 0;
                 i < CA_.GetNLocalControlDoFs(linearization_point); i++)
              {
                CA_.CopyLocalControlToTensor(*functional_gradient, grad_J, i);
                CA_.CopyLocalControlToTensor(*linearization_point, control, i);
                CA_.CopyLocalControlToTensor(*eval_point, point, i);
                CA_.CopyLocalControlToTensor(*mma_lower_asymptote,
                                             lower_asymptote, i);
                CA_.CopyLocalControlToTensor(*mma_upper_asymptote,
                                             upper_asymptote, i);

                //Project to positive and negative part! And compute tau = largest eigenvalue of grad_J
                CA_.ProjectToPositiveAndNegativePart(grad_J, grad_J_plus,
                                                     grad_J_minus, tau);
                //adjust tau, such that -  gradJ + tau Id  is pos def.
                tmp = invert(upper_asymptote - lower_asymptote);
                p_val = (upper_asymptote - control)
                        * (grad_J_plus + rho_ / 2. * tmp)
                        * (upper_asymptote - control);
                q_val = (control - lower_asymptote)
                        * (-1. * grad_J_minus + rho_ / 2. * tmp)
                        * (control - lower_asymptote);

                uf = invert(upper_asymptote - point);
                lf = invert(point - lower_asymptote);

                result = uf * p_val * uf;
                result -= lf * q_val * lf;
                CA_.CopyTensorToLocalControl(result, residual, i);
                //Now done with J'
              }
            //Derivative of  Multiplier times constraints
            Tensor<2, localdim> local_constraints, local_multiplier,
                   local_constraint_derivative;

            {
              identity *= p_;
              Vector<double> local_control_directions(CA_.NLocalDirections());

              Tensor < 2, localdim > Z;
              Tensor < 2, localdim > lhs;
              //Local Blocked constraints
              std::vector<std::vector<unsigned int> > control_to_constraint_index;
              CA_.LocalControlToConstraintBlocks(linearization_point,
                                                 control_to_constraint_index);
              for (unsigned int i = 0;
                   i < CA_.GetNLocalControlDoFs(linearization_point); i++)
                {
                  for (unsigned int j = 0;
                       j < control_to_constraint_index[i].size(); j++)
                    {
                      unsigned int index = control_to_constraint_index[i][j];

                      CA_.CopyLocalConstraintToTensor(*constraint_values,
                                                      local_constraints, index);
                      CA_.CopyLocalConstraintToTensor(*mma_multiplier,
                                                      local_multiplier, index);

                      //Compute Z
                      Z = local_constraints + identity;
                      Z *= -1. / (p_ * p_);

                      CA_.GetLocalConstraintDerivative(
                        local_constraint_derivative, *constraint_values, index);

                      lhs = Z * local_multiplier * Z
                            * local_constraint_derivative;
                      lhs *= p_ * p_;
                      CA_.AddTensorToLocalControl(lhs, residual, i);
                    }
                }
            }
          }
        }
      else if (this->GetType() == "hessian")
        {
          residual = 0.;
          //Compute the gradient with respect to the controlvariable, e.g. eval_point

          if (dopedim == 0)
            throw DOpEException("Not implemented for this dopedim ",
                                "AumentedLagrangianProblem::AlgebraicFunctional");

          //Get the required values
          const dealii::BlockVector<double> *linearization_point;
          const dealii::BlockVector<double> *functional_gradient;
          const dealii::BlockVector<double> *mma_multiplier;
          const dealii::BlockVector<double> *eval_point;
          const dealii::BlockVector<double> *direction;
          const dealii::Vector<double> *mma_multiplier_global;
          const dealii::BlockVector<double> *constraint_values;
          const dealii::Vector<double> *constraint_values_global;
          std::vector<const dealii::BlockVector<double>*> constraint_gradients;
          const dealii::BlockVector<double> *mma_lower_asymptote;
          const dealii::BlockVector<double> *mma_upper_asymptote;

          {
            eval_point = GetBlockVector(block_values, "control");
            linearization_point = GetBlockVector(block_values, "mma_control");
            functional_gradient = GetBlockVector(block_values,
                                                 "mma_functional_gradient");
            mma_multiplier = GetBlockVector(block_values,
                                            "mma_multiplier_local");
            mma_multiplier_global = GetVector(values,
                                              "mma_multiplier_global");

            mma_lower_asymptote = GetBlockVector(block_values,
                                                 "mma_lower_asymptote");
            mma_upper_asymptote = GetBlockVector(block_values,
                                                 "mma_upper_asymptote");

            constraint_values = GetBlockVector(block_values,
                                               "constraints_local");
            constraint_values_global = GetVector(values,
                                                 "constraints_global");

            constraint_gradients.resize(mma_multiplier_global->size(), NULL);
            for (unsigned int i = 0; i < constraint_gradients.size(); i++)
              {
                std::stringstream name;
                name << "constraint_gradient_" << i;
                constraint_gradients[i] = GetBlockVector(block_values,
                                                         name.str());
              }
            direction = GetBlockVector(block_values, "dq");
          }
          //local in time global in space constraint times multiplier derivative
          //(only phi'' \nabla g \nabla g^T)
          // the other term is computed elsewhere
          {
            unsigned int global_block = mma_multiplier->n_blocks() - 1;
            for (unsigned int i = 0; i < constraint_gradients.size(); i++)
              {
                //phi''
                double Z = (constraint_values_global->operator()(i) + p_)
                           / (-1. * p_ * p_);
                double dd_phi = *(constraint_gradients[i]) * (*direction);
                dd_phi *= -p_ * p_ * 2. * Z * Z * Z;
                dd_phi *= mma_multiplier->block(global_block)(i);
                residual.add(dd_phi, *(constraint_gradients[i]));
              }
          }
          //local in time and space constraints times multiplier derivative
          {
            //Now phi'' \nabla g \nabla g^T
            Tensor<2, localdim> local_constraints, local_multiplier,
                   local_constraint_derivative, identity, tmp;
            identity = 0;
            for (unsigned int i = 0; i < localdim; i++)
              {
                identity[i][i] = p_;
              }

            Vector<double> local_control_directions(CA_.NLocalDirections());
            Tensor < 2, localdim > Z;
            Tensor < 2, localdim > lhs;
            Vector<double> dq(CA_.NLocalDirections());
            Vector<double> H_dq(CA_.NLocalDirections());

            //Local Blocked constraints
            for (unsigned int i = 0;
                 i < CA_.GetNLocalControlConstraintDoFs(mma_multiplier); i++)
              {
                CA_.CopyLocalConstraintToTensor(*constraint_values,
                                                local_constraints, i);
                CA_.CopyLocalConstraintToTensor(*mma_multiplier,
                                                local_multiplier, i);
                CA_.CopyLocalControlToVector(*direction, dq, i);
                //Compute Z
                CA_.CopyLocalConstraintToTensor(*constraint_values,
                                                local_constraints, i);

                Z = local_constraints + identity;
                Z *= -1. / (p_ * p_);
                lhs = Z * local_multiplier * Z;
                lhs *= p_ * p_;
                local_control_directions = 0.;

                for (unsigned int j = 0; j < CA_.NLocalDirections(); j++)
                  {
                    CA_.GetLocalConstraintDerivative(local_constraint_derivative,
                                                     *constraint_values, i, j);
                    tmp = lhs * local_constraint_derivative * Z;

                    for (unsigned int k = 0; k < CA_.NLocalDirections(); k++)
                      {
                        //phi''\nabla g \nabla g^T
                        CA_.GetLocalConstraintDerivative(
                          local_constraint_derivative, *constraint_values, i, k);
                        local_control_directions(k) -= 2
                                                       * scalar_product(tmp, local_constraint_derivative)
                                                       * dq(k);

                        // phi' \nabla^2g
                        CA_.GetLocalConstraintSecondDerivative(
                          local_constraint_derivative, *constraint_values, i, j,
                          k);
                        local_control_directions(k) += scalar_product(lhs,
                                                                      local_constraint_derivative) * dq(k);
                      }
                  }
                CA_.AddVectorToLocalControl(local_control_directions, residual,
                                            i);
              }
          }
          //derivative of hyperbolic functional approximation
          {
            Tensor<2, localdim> grad_J, lower_asymptote, upper_asymptote,
                   control, point, grad_J_plus, grad_J_minus;
            Tensor<2, localdim> p_val, q_val, uf, lf, tmp, dq, result;
            double tau = 0.;

            for (unsigned int i = 0;
                 i < CA_.GetNLocalControlDoFs(linearization_point); i++)
              {
                CA_.CopyLocalControlToTensor(*functional_gradient, grad_J, i);
                CA_.CopyLocalControlToTensor(*linearization_point, control, i);
                CA_.CopyLocalControlToTensor(*eval_point, point, i);
                CA_.CopyLocalControlToTensor(*direction, dq, i);
                CA_.CopyLocalControlToTensor(*mma_lower_asymptote,
                                             lower_asymptote, i);
                CA_.CopyLocalControlToTensor(*mma_upper_asymptote,
                                             upper_asymptote, i);

                //Project to positive and negative part! And compute tau = largest eigenvalue of grad_J
                CA_.ProjectToPositiveAndNegativePart(grad_J, grad_J_plus,
                                                     grad_J_minus, tau);

                //adjust tau, s.t. -  gradJ + tau Id is pos def.
                tmp = invert(upper_asymptote - lower_asymptote);
                p_val = (upper_asymptote - control)
                        * (grad_J_plus + rho_ / 2. * tmp)
                        * (upper_asymptote - control);
                q_val = (control - lower_asymptote)
                        * (-1. * grad_J_minus + rho_ / 2. * tmp)
                        * (control - lower_asymptote);

                uf = invert(upper_asymptote - point);
                lf = invert(point - lower_asymptote);

                result = uf * p_val * uf * dq * uf + uf * dq * uf * p_val * uf;
                result += lf * q_val * lf * dq * lf + lf * dq * lf * q_val * lf;

                if (scalar_product(dq, result) < 0.)
                  {
                    std::cout << "negative block! " << i << std::endl;
                    abort();
                  }
                CA_.AddTensorToLocalControl(result, residual, i);
                //Done with J''
              }
          }
        }
      else if (this->GetType() == "hessian_inverse")
        {
          //Compute Residual = H^{-1}dq
          //Where H^{-1} is the Hessian of the mma approximation and the block-local constraints

          //Get the required values
          const dealii::BlockVector<double> *linearization_point;
          const dealii::BlockVector<double> *functional_gradient;
          const dealii::BlockVector<double> *mma_multiplier;
          const dealii::BlockVector<double> *eval_point;
          const dealii::BlockVector<double> *direction;
          const dealii::BlockVector<double> *constraint_values;
          const dealii::BlockVector<double> *mma_lower_asymptote;
          const dealii::BlockVector<double> *mma_upper_asymptote;

          {
            eval_point = GetBlockVector(block_values, "control");
            linearization_point = GetBlockVector(block_values, "mma_control");
            functional_gradient = GetBlockVector(block_values,
                                                 "mma_functional_gradient");
            mma_multiplier = GetBlockVector(block_values,
                                            "mma_multiplier_local");
            constraint_values = GetBlockVector(block_values,
                                               "constraints_local");
            mma_lower_asymptote = GetBlockVector(block_values,
                                                 "mma_lower_asymptote");
            mma_upper_asymptote = GetBlockVector(block_values,
                                                 "mma_upper_asymptote");

            direction = GetBlockVector(block_values, "dq");
          }

          std::vector<std::vector<unsigned int> > control_to_constraint_index;
          CA_.LocalControlToConstraintBlocks(linearization_point,
                                             control_to_constraint_index);

          Tensor<2, localdim> H, identity, result;
          Tensor<2, localdim> grad_J, lower_asymptote, upper_asymptote,
                 control, point, grad_J_plus, grad_J_minus;
          Tensor<2, localdim> p_val, q_val, uf, lf, tmp, dq;
          Tensor<2, localdim> Z, local_constraints, local_multiplier,
                 local_constraint_derivative, local_constraint_second_derivative;

          double tau = 0.;

          identity = 0;
          for (unsigned int i = 0; i < localdim; i++)
            {
              identity[i][i] = p_;
            }

          //Build  local Blocks of the hessian
          for (unsigned int i = 0;
               i < CA_.GetNLocalControlDoFs(linearization_point); i++)
            {
              H = 0.;

              //MMA Approx
              CA_.CopyLocalControlToTensor(*functional_gradient, grad_J, i);
              CA_.CopyLocalControlToTensor(*linearization_point, control, i);
              CA_.CopyLocalControlToTensor(*eval_point, point, i);
              CA_.CopyLocalControlToTensor(*direction, dq, i);
              CA_.CopyLocalControlToTensor(*mma_lower_asymptote,
                                           lower_asymptote, i);
              CA_.CopyLocalControlToTensor(*mma_upper_asymptote,
                                           upper_asymptote, i);

              //Project to positive and negative part! And compute tau = largest eigenvalue of grad_J
              CA_.ProjectToPositiveAndNegativePart(grad_J, grad_J_plus,
                                                   grad_J_minus, tau);
              //adjust tau, such that -  gradJ + tau Id is pos def.
              tmp = invert(upper_asymptote - lower_asymptote);
              p_val = (upper_asymptote - control)
                      * (grad_J_plus + rho_ / 2. * tmp)
                      * (upper_asymptote - control);
              q_val = (control - lower_asymptote)
                      * (-1. * grad_J_minus + rho_ / 2. * tmp)
                      * (control - lower_asymptote);

              uf = invert(upper_asymptote - point);
              lf = invert(point - lower_asymptote);
              H += 2. * uf * p_val * uf * uf;
              H += 2. * lf * q_val * lf * lf;

              //Now Constraint derivatives
              for (unsigned int j = 0;
                   j < control_to_constraint_index[i].size(); j++)
                {
                  unsigned int index = control_to_constraint_index[i][j];
                  CA_.CopyLocalConstraintToTensor(*constraint_values,
                                                  local_constraints, index);
                  CA_.CopyLocalConstraintToTensor(*mma_multiplier,
                                                  local_multiplier, index);

                  Z = local_constraints + identity;
                  Z *= -1. / (p_ * p_);

                  CA_.GetLocalConstraintDerivative(local_constraint_derivative,
                                                   *constraint_values, index);
                  CA_.GetLocalConstraintSecondDerivative(
                    local_constraint_second_derivative, *constraint_values,
                    index);

                  H += p_ * p_ * local_multiplier * Z * Z
                       * (local_constraint_second_derivative
                          - 2. * Z * local_constraint_derivative
                          * local_constraint_derivative);
                }
              //Computation of H done
              result = invert(H) * dq;
              CA_.AddTensorToLocalControl(result, residual, i);
            }
        }
      else
        {
          throw DOpEException("Unknown Type: " + this->GetType(),
                              "AugmentedLagrangianProblem::AlgebraicResidual");
        }
    }
    /******************************************************/
    template<typename DATACONTAINER>
    double
    ElementFunctional(const DATACONTAINER &edc)
    {
      return OP_.ElementFunctional(edc);
    }

    /******************************************************/

    double
    PointFunctional(
      const std::map<std::string, const dealii::Vector<double>*> &param_values,
      const std::map<std::string, const dealii::BlockVector<double>*> &domain_values)
    {
      return OP_.PointFunctional(param_values, domain_values);
    }

    /******************************************************/

    template<typename FACEDATACONTAINER>
    double
    BoundaryFunctional(const FACEDATACONTAINER &fdc)
    {
      return OP_.BoundaryFunctional(fdc);
    }

    /******************************************************/

    template<typename FACEDATACONTAINER>
    double
    FaceFunctional(const FACEDATACONTAINER &fdc)
    {
      return OP_.FaceFunctional(fdc);
    }

    /******************************************************/

    template<typename DATACONTAINER>
    void
    ElementEquation(const DATACONTAINER &edc,
                    dealii::Vector<double> &local_vector, double scale = 1.,
                    double scale_ico = 1.)
    {
      OP_.ElementEquation(edc, local_vector, scale, scale_ico);
    }

    /******************************************************/

    template<typename DATACONTAINER>
    void
    ElementRhs(const DATACONTAINER &edc,
               dealii::Vector<double> &local_vector, double scale = 1.)
    {
      if (this->GetType() == "gradient")
        {
          dealii::Vector<double> mma_multiplier_global;
          dealii::Vector<double> constraint_values_global;

          std::string tmp = this->GetType();
          unsigned int tmp_num = this->GetTypeNum();

          {
            edc.GetParamValues("mma_multiplier_global",
                               mma_multiplier_global);
            edc.GetParamValues("constraints_global",
                               constraint_values_global);
          }
          for (unsigned int i = 0; i < mma_multiplier_global.size(); i++)
            {
              OP_.SetType("global_constraint_gradient", i);
              double local_scaling = (constraint_values_global(i) + p_)
                                     / (-1. * p_ * p_);
              local_scaling *= local_scaling;
              local_scaling *= mma_multiplier_global(i);
              local_scaling *= (p_ * p_);
              OP_.ElementRhs(edc, local_vector, scale * local_scaling);
            }
          OP_.SetType(tmp, tmp_num);
        }
      else if (this->GetType() == "hessian")
        {
          std::string tmp = this->GetType();
          unsigned int tmp_num = this->GetTypeNum();
          dealii::Vector<double> mma_multiplier_global;
          dealii::Vector<double> constraint_values_global;

          {
            edc.GetParamValues("mma_multiplier_global",
                               mma_multiplier_global);
            edc.GetParamValues("constraints_global",
                               constraint_values_global);
          }

          for (unsigned int i = 0; i < mma_multiplier_global.size(); i++)
            {
              OP_.SetType("global_constraint_hessian", i);
              double local_scaling = (constraint_values_global(i) + p_);
              local_scaling *= local_scaling;
              local_scaling *= mma_multiplier_global(i);
              local_scaling *= 1. / (p_ * p_);
              OP_.ElementRhs(edc, local_vector, scale * local_scaling);
            }
          OP_.SetType(tmp, tmp_num);
        }
      else if (this->GetType() == "global_constraint_gradient")
        {
          OP_.ElementRhs(edc, local_vector, scale);
        }
      else
        {
          throw DOpEException("Not Implemented",
                              "AugmentedLagrangianProblem::ElementRhs");
        }
    }
    /******************************************************/

    void
    PointRhs(const std::map<std::string, const dealii::Vector<double>*> &,
             const std::map<std::string, const BlockVector<double>*> &,
             BlockVector<double> &, double)
    {
      throw DOpEException("Not Implemented",
                          "AugmentedLagrangianProblem::PointRhs");
    }

    /******************************************************/

    template<typename DATACONTAINER>
    void
    ElementMatrix(const DATACONTAINER &edc,
                  dealii::FullMatrix<double> &local_matrix, double scale = 1.,
                  double scale_ico = 1.)
    {
      OP_.ElementMatrix(edc, local_matrix, scale, scale_ico);
    }

    /******************************************************/

    template<typename FACEDATACONTAINER>
    void
    FaceEquation(const FACEDATACONTAINER & /*fcd*/,
                 dealii::Vector<double> &/*local_vector*/, double /*scale*/ = 1.,
                 double /*scale_ico*/ = 1.)
    {
      throw DOpEException("Not Implemented",
                          "AugmentedLagrangianProblem::FaceEquation");
    }

    /******************************************************/

    template<typename FACEDATACONTAINER>
    void
    FaceRhs(const FACEDATACONTAINER & /*fcd*/,
            dealii::Vector<double> &/*local_vector*/, double /*scale*/ = 1.)
    {
      throw DOpEException("Not Implemented",
                          "AugmentedLagrangianProblem::FaceRhs");
    }

    /******************************************************/

    template<typename FACEDATACONTAINER>
    void
    FaceMatrix(const FACEDATACONTAINER & /*fcd*/,
               dealii::FullMatrix<double> &/*local_matrix*/, double /*scale*/ = 1.,
               double /*scale_ico*/ = 1.)
    {
      throw DOpEException("Not Implemented",
                          "AugmentedLagrangianProblem::FaceMatrix");
    }

    /******************************************************/

    template<typename FACEDATACONTAINER>
    void
    BoundaryEquation(const FACEDATACONTAINER & /*fcd*/,
                     dealii::Vector<double> &/*local_vector*/, double /*scale*/ = 1.,
                     double /*scale_ico*/ = 1.)
    {
      throw DOpEException("Not Implemented",
                          "AugmentedLagrangianProblem::BoundaryEquation");
    }

    /******************************************************/

    template<typename FACEDATACONTAINER>
    void
    BoundaryRhs(const FACEDATACONTAINER & /*fcd*/,
                dealii::Vector<double> &/*local_vector*/, double /*scale*/ = 1.)
    {
      throw DOpEException("Not Implemented",
                          "AugmentedLagrangianProblem::BoundaryRhs");
    }

    /******************************************************/

    template<typename FACEDATACONTAINER>
    void
    BoundaryMatrix(const FACEDATACONTAINER & /*fcd*/,
                   dealii::FullMatrix<double> &/*local_matrix*/, double /*scale*/ = 1.,
                   double /*scale_ico*/ = 1.)
    {
      throw DOpEException("Not Implemented",
                          "AugmentedLagrangianProblem::BoundaryRhs");
    }
    /******************************************************/
    template<typename FACEDATACONTAINER>
    void
    InterfaceEquation(const FACEDATACONTAINER & /*dc*/,
                      dealii::Vector<double> &/*local_vector*/, double /*scale*/ = 1.,
                      double /*scale_ico*/ = 1.)
    {
      throw DOpEException("Not Implemented",
                          "AugmentedLagrangianProblem::BoundaryRhs");
    }
    /******************************************************/
    template<typename FACEDATACONTAINER>
    void
    InterfaceMatrix(const FACEDATACONTAINER & /*dc*/,
                    dealii::FullMatrix<double> &/*local_matrix*/, double /*scale*/ = 1.,
                    double /*scale_ico*/ = 1.)
    {
      throw DOpEException("Not Implemented",
                          "AugmentedLagrangianProblem::BoundaryRhs");
    }

    /******************************************************/

    void
    ComputeLocalControlConstraints(dealii::BlockVector<double> &constraints,
                                   const std::map<std::string, const dealii::Vector<double>*> &/*values*/,
                                   const std::map<std::string, const dealii::BlockVector<double>*> &block_values)
    {
      const dealii::BlockVector<double> &lower_bound = *GetBlockVector(
                                                         block_values, "mma_lower_bound");
      const dealii::BlockVector<double> &upper_bound = *GetBlockVector(
                                                         block_values, "mma_upper_bound");
      const dealii::BlockVector<double> &control = *GetBlockVector(
                                                     block_values, "control");

      assert(constraints.block(0).size() == 2 * control.block(0).size());
      for (unsigned int i = 0; i < control.block(0).size(); i++)
        {
          //Add Control Constraints, such that if control is feasible all  entries are not positive!
          constraints.block(0)(i) = lower_bound.block(0)(i)
                                    - control.block(0)(i);
          constraints.block(0)(control.block(0).size() + i) =
            control.block(0)(i) - upper_bound.block(0)(i);
        }
    }

    /******************************************************/

    void
    GetControlBoxConstraints(dealii::BlockVector<double> &lb,
                             dealii::BlockVector<double> &ub) const
    {
      lb = GetAuxiliaryControl("mma_lower_bound")->GetSpacialVector();
      ub = GetAuxiliaryControl("mma_upper_bound")->GetSpacialVector();
    }

    /******************************************************/

    /**
     * A pointer to the quadrature rule for domain values
     *
     * @return A const pointer to the QuadratureFormula()
     */
    const dealii::SmartPointer<const dealii::Quadrature<dealdim> >
    GetQuadratureFormula() const
    {
      return OP_.GetQuadratureFormula();
    }

    /******************************************************/

    /**
     * A pointer to the quadrature rule for face- and boundary-face values
     *
     * @return A const pointer to the FaceQuadratureFormula()
     */
    const dealii::SmartPointer<const dealii::Quadrature<dealdim - 1> >
    GetFaceQuadratureFormula() const
    {
      return OP_.GetFaceQuadratureFormula();
    }

    /******************************************************/

    /**
     * A pointer to the whole FESystem
     *
     * @return A const pointer to the FESystem()
     */
    const dealii::SmartPointer<const dealii::FESystem<dealdim> >
    GetFESystem() const
    {
      return OP_.GetFESystem();
    }

    /******************************************************/

    /**
     * This function determines whether a loop over all faces is required or not.
     *
     * @return Returns whether or not this functional has components on faces between elements.
     *         The default value is false.
     */
    bool
    HasFaces() const
    {
      return OP_.HasFaces();
    }
    bool
    HasPoints() const
    {
      return OP_.HasPoints();
    }
    ;
    bool
    HasInterfaces() const
    {
      return OP_.HasInterfaces();
    }
    template<typename ELEMENTITERATOR>
    bool AtInterface(ELEMENTITERATOR &element, unsigned int face) const
    {
      return OP_.AtInterface(element,face);
    }
    /******************************************************/

    /**
     * This function returns the update flags for domain values
     * for the computation of shape values, gradients, etc.
     * For detailed explication, please visit `Finite element access/FEValues classes' in
     * the deal.ii manual.
     *
     * @return Returns the update flags to use in a computation.
     */
    dealii::UpdateFlags
    GetUpdateFlags() const
    {
      return OP_.GetUpdateFlags();
    }

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
    dealii::UpdateFlags
    GetFaceUpdateFlags() const
    {
      return OP_.GetFaceUpdateFlags();
    }

    /******************************************************/

    /**
     * A std::vector of integer values which contains the colors of Dirichlet boundaries.
     *
     * @return Returns the Dirichlet Colors.
     */
    const std::vector<unsigned int> &
    GetDirichletColors() const
    {
      return OP_.GetDirichletColors();
    }

    /******************************************************/

    /**
     * A std::vector of boolean values to decide at which parts of the boundary and solutions variables
     * Dirichlet values should be applied.
     *
     * @return Returns a component mask for each boundary color.
     */
    const std::vector<bool> &
    GetDirichletCompMask(unsigned int color) const
    {
      return OP_.GetDirichletCompMask(color);
    }

    /******************************************************/

    /**
     * This dealii::Function of dimension `dealdim' knows what Dirichlet values to apply
     * on each boundary part with color 'color'.
     *
     * @return Returns a dealii::Function of Dirichlet values of the boundary part with color 'color'.
     */
    const dealii::Function<dealdim> &
    GetDirichletValues(unsigned int color,
                       const std::map<std::string, const dealii::Vector<double>*> &param_values,
                       const std::map<std::string, const dealii::BlockVector<double>*> &domain_values) const
    {
      return OP_.GetDirichletValues(color, param_values, domain_values);
    }

    /******************************************************/

    /**
     * This dealii::Function of dimension `dealdim' applys the initial values to the PDE- or Optimization
     * problem, respectively.
     *
     * @return Returns a dealii::Function of initial values.
     */
    const dealii::Function<dealdim> &
    GetInitialValues() const
    {
      return OP_.GetInitialValues();
    }

    /******************************************************/

    /**
     * A std::vector of integer values which contains the colors of the boundary equation.
     *
     * @return Returns colors for the boundary equation.
     */
    const std::vector<unsigned int> &
    GetBoundaryEquationColors() const
    {
      return OP_.GetBoundaryEquationColors();
    }

    /******************************************************/

    /**
     * A std::vector of integer values which contains the colors of the boundary functionals.
     *
     * @return Returns colors for the boundary functionals.
     */
    const std::vector<unsigned int> &
    GetBoundaryFunctionalColors() const
    {
      return OP_.GetBoundaryFunctionalColors();
    }

    /******************************************************/

    /**
     * This function returns the number of functionals to be considered in the problem.
     *
     * @return Returns the number of functionals.
     */
    unsigned int
    GetNFunctionals() const
    {
      return 0;
    }

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
    unsigned int
    GetNBlocks() const
    {
      return OP_.GetNBlocks();
    }

    /******************************************************/

    /**
     * A function which has the number of degrees of freedom for the block `b'.
     *
     * @return Returns the number of DoFs for block `b'.
     */
    unsigned int
    GetDoFsPerBlock(unsigned int b) const
    {
      return OP_.GetDoFsPerBlock(b);
    }

    /******************************************************/

    /**
     * A std::vector which contains the number of degrees of freedom per block.
     *
     * @return Returns a vector with DoFs.
     */
    const std::vector<unsigned int> &
    GetDoFsPerBlock() const
    {
      return OP_.GetDoFsPerBlock();
    }

    /******************************************************/

    /**
     * A dealii function. Please visit: ConstraintMatrix in the deal.ii manual.
     *
     * @return Returns a matrix with hanging node constraints.
     */
    const dealii::ConstraintMatrix &
    GetDoFConstraints() const
    {
      return OP_.GetDoFConstraints();
    }

    std::string
    GetType() const
    {
      return OP_.GetType();
    }
    unsigned int
    GetTypeNum() const
    {
      return OP_.GetTypeNum();
    }
    std::string
    GetDoFType() const
    {
      return OP_.GetDoFType();
    }

    /******************************************************/

    /**
     * This function describes what type of Functional is considered
     * Here it is computed by algebraic operations on the vectors.
     */
    std::string
    GetFunctionalType() const
    {
      if (this->GetType() == "cost_functional"
          || this->GetType() == "gradient" || this->GetType() == "hessian"
          || this->GetType() == "hessian_inverse")
        return "algebraic";
      return OP_.GetFunctionalType();
    }

    /******************************************************/

    /**
     * This function is used to name the Functional, this is helpful to distinguish different Functionals in the output.
     *
     * @return A string. This is the name beeing displayed next to the computed values.
     */
    std::string
    GetFunctionalName() const
    {
      return "Seperable Augmented Lagrangian";
    }

    /******************************************************/

    std::string
    GetConstraintType() const
    {
      return OP_.GetConstraintType();
    }

    /******************************************************/

    bool
    HasControlInDirichletData() const
    {
      return OP_.HasControlInDirichletData();
    }

    /******************************************************/

    /**
     * A pointer to the OutputHandler() object.
     *
     * @return The OutputHandler() object.
     */
    DOpEOutputHandler<dealii::BlockVector<double> > *
    GetOutputHandler()
    {
      return OP_.GetOutputHandler();
    }

    /******************************************************/

    /**
     * A pointer to the SpaceTimeHandler<dopedim,dealdim>  object.
     *
     * @return The SpaceTimeHandler() object.
     */
    const STH *
    GetSpaceTimeHandler() const
    {
      return OP_.GetSpaceTimeHandler();
    }

    /******************************************************/

    /**
     * A pointer to the SpaceTimeHandler<dopedim,dealdim>  object.
     *
     * @return The SpaceTimeHandler() object.
     */
    STH *
    GetSpaceTimeHandler()
    {
      return OP_.GetSpaceTimeHandler();
    }

    /******************************************************/

    void
    ComputeSparsityPattern(BlockSparsityPattern &sparsity) const
    {
      OP_.ComputeSparsityPattern(sparsity);
    }

    /******************************************************/

    void
    PostProcessConstraints(
      ConstraintVector<dealii::BlockVector<double> > &g) const
    {
      //OP_.PostProcessConstraints(g,process_global_in_time_constraints);
      {
        dealii::BlockVector<double> &bv_g = g.GetSpacialVector("local");

        Tensor<2, localdim> tmp, tmp2, identity;
        identity = 0;
        for (unsigned int i = 0; i < localdim; i++)
          identity[i][i] = 1.;
        //Loop over local control and state constraints
        for (unsigned int i = 0;
             i < CA_.GetNLocalControlConstraintDoFs(&bv_g); i++)
          {
            CA_.CopyLocalConstraintToTensor(bv_g, tmp, i);
            tmp -= p_ * identity;
            tmp2 = invert(tmp);
            tmp2 *= -p_ * p_;
            tmp2 -= p_ * identity;
            CA_.CopyTensorToLocalConstraint(tmp2, bv_g, i);
          }
      }
      {
        dealii::Vector<double> &bv_g = g.GetGlobalConstraints();
        //Loop over global constraints.
        for (unsigned int i = 0; i < bv_g.size(); i++)
          {
            bv_g(i) = -p_ * p_ / (bv_g(i) - p_) - p_;
          }
      }
    }
    void
    AddAuxiliaryControl(
      const ControlVector<dealii::BlockVector<double> > *c,
      std::string name)
    {
      OP_.AddAuxiliaryControl(c, name);
    }
    const ControlVector<dealii::BlockVector<double> > *
    GetAuxiliaryControl(std::string name) const
    {
      return OP_.GetAuxiliaryControl(name);
    }
    void
    AddAuxiliaryConstraint(
      const ConstraintVector<dealii::BlockVector<double> > *c,
      std::string name)
    {
      OP_.AddAuxiliaryConstraint(c, name);
    }
    void
    DeleteAuxiliaryControl(std::string name)
    {
      OP_.DeleteAuxiliaryControl(name);
    }
    void
    DeleteAuxiliaryConstraint(std::string name)
    {
      OP_.DeleteAuxiliaryConstraint(name);
    }
    const ConstraintVector<dealii::BlockVector<double> > *
    GetAuxiliaryConstraint(std::string name)
    {
      return OP_.GetAuxiliaryConstraint(name);
    }

    template<typename INTEGRATOR>
    void
    AddAuxiliaryToIntegrator(INTEGRATOR &integrator)
    {
      OP_.AddAuxiliaryToIntegrator(integrator);
    }
    template<typename INTEGRATOR>
    void
    DeleteAuxiliaryFromIntegrator(INTEGRATOR &integrator)
    {
      OP_.DeleteAuxiliaryFromIntegrator(integrator);
    }

    /*************************************************************************************/
    bool
    GetFEValuesNeededToBeInitialized() const
    {
      return OP_.GetFEValuesNeededToBeInitialized();
    }

    /******************************************************/
    void
    SetFEValuesAreInitialized()
    {
      OP_.SetFEValuesAreInitialized();
    }

    /******************************************************/
    const std::map<std::string, unsigned int> &
    GetFunctionalPosition() const
    {
      return OP_.GetFunctionalPosition();
    }

  private:
    OPTPROBLEM &OP_;
    CONSTRAINTACCESSOR &CA_;
    double p_, J_, rho_;

    const dealii::BlockVector<double> *
    GetBlockVector(
      const std::map<std::string, const BlockVector<double>*> &values,
      std::string name)
    {
      typename std::map<std::string, const BlockVector<double>*>::const_iterator it =
        values.find(name);
      if (it == values.end())
        {
          throw DOpEException("Did not find " + name,
                              "AugmentedLagrangian::GetBlockVector");
        }
      return it->second;
    }
    const dealii::Vector<double> *
    GetVector(const std::map<std::string, const Vector<double>*> &values,
              std::string name)
    {
      typename std::map<std::string, const Vector<double>*>::const_iterator it =
        values.find(name);
      if (it == values.end())
        {
          throw DOpEException("Did not find " + name,
                              "AugmentedLagrangian::GetVector");
        }
      return it->second;
    }
  };
}

#endif
