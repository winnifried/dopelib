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

#ifndef DOPE_FUNCTION_H_
#define DOPE_FUNCTION_H_

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>

namespace DOpEWrapper
{
  /**
   * @class Function
   *
   * A dope function which is derived from the dealii function interface.
   * This wrapper is needed to assert that the class have a
   * method SetTime so that we have a unified way to comunicate
   * the time in nonstationary problems.
   *
   * @tparam<dim>    The considered function has dimension `dim'
   */
  template<int dim>
  class Function : public dealii::Function<dim>
  {
  public:
    Function(const unsigned int n_components = 1,
             const double initial_time = 0.0)
      : dealii::Function<dim>(n_components, initial_time)
    {
      init_time_ = initial_time;
    }

    /******************************************************/
    void
    vector_value(const dealii::Point<dim> &p,
                 dealii::Vector<double> &return_value) const
    {
      Assert(return_value.size() == this->n_components,
             dealii::ExcDimensionMismatch(return_value.size(), this->n_components));
      for (unsigned int comp = 0; comp < this->n_components; comp++)
        {
          return_value(comp) = this->value(p, comp);
        }
    }

    /**
     * Gives a dealii::Function the actual time within a time stepping
     * scheme. Necessary when dealing with time dependent boundary conditions.
     *
     * @param Gives actual time to the function.
     */
    virtual void
    SetTime(double /*time*/) const
    {
    }
    ;

    /**
     * Returns the initial time given in the constructor.
     */
    double
    InitialTime() const
    {
      return init_time_;
    }
    ;
  private:
    double init_time_;
  };

  /******************************************************/
  /******************************************************/

  /**
   * @class ZeroFunctions
   *
   * A dope zero function which is derived from the dealii zero function interface.
   */
  template<int dim>
  class ZeroFunction : public Function<dim>
  {
  public:

    ZeroFunction(const unsigned int n_components = 1)
      : Function<dim>(n_components)
    {
    }

    virtual
    ~ZeroFunction()
    {
    }

    virtual double
    value(const dealii::Point<dim> &/*p*/,
          const unsigned int /*component*/) const
    {
      return 0.0;
    }

    virtual void
    vector_value(const dealii::Point<dim> &/*p*/,
                 dealii::Vector<double> &return_value) const
    {
      Assert(return_value.size() == this->n_components,
             ExcDimensionMismatch (return_value.size(), this->n_components));

      std::fill(return_value.begin(), return_value.end(), 0.0);
    }

    virtual void
    value_list(const std::vector<dealii::Point<dim> > &/*points*/,
               std::vector<double> &values,
               const unsigned int /*component*/ = 0) const
    {
      // Assert(values.size() == points.size(),
      //   ExcDimensionMismatch(values.size(), points.size()));

      std::fill(values.begin(), values.end(), 0.);
    }

    virtual void
    vector_value_list(const std::vector<dealii::Point<dim> > &points,
                      std::vector<dealii::Vector<double> > &values) const
    {
      Assert(values.size() == points.size(),
             ExcDimensionMismatch(values.size(), points.size()));

      for (unsigned int i = 0; i < points.size(); ++i)
        {
          Assert(values[i].size() == this->n_components,
                 ExcDimensionMismatch(values[i].size(), this->n_components));
          std::fill(values[i].begin(), values[i].end(), 0.);
        };
    }

    virtual dealii::Tensor<1, dim>
    gradient(const dealii::Point<dim> &/*p*/,
             const unsigned int /*component*/ = 0) const
    {
      return dealii::Tensor<1, dim>();
    }

    virtual void
    vector_gradient(const dealii::Point<dim> &/*p*/,
                    std::vector<dealii::Tensor<1, dim> > &gradients) const
    {
      Assert(gradients.size() == this->n_components,
             ExcDimensionMismatch(gradients.size(), this->n_components));

      for (unsigned int c = 0; c < this->n_components; ++c)
        gradients[c].clear();
    }

    virtual void
    gradient_list(const std::vector<dealii::Point<dim> > &points,
                  std::vector<dealii::Tensor<1, dim> > &gradients,
                  const unsigned int /*component*/ = 0) const
    {
      Assert(gradients.size() == points.size(),
             ExcDimensionMismatch(gradients.size(), points.size()));

      for (unsigned int i = 0; i < points.size(); ++i)
        gradients[i].clear();
    }

    virtual void
    vector_gradient_list(const std::vector<dealii::Point<dim> > &points,
                         std::vector<std::vector<dealii::Tensor<1, dim> > > &gradients) const
    {
      Assert(gradients.size() == points.size(),
             ExcDimensionMismatch(gradients.size(), points.size()));
      for (unsigned int i = 0; i < points.size(); ++i)
        {
          Assert(gradients[i].size() == this->n_components,
                 ExcDimensionMismatch(gradients[i].size(), this->n_components));
          for (unsigned int c = 0; c < this->n_components; ++c)
            gradients[i][c].clear();
        };
    }
  };

  /******************************************************/

  /**
   * @class ZeroFunctions
   *
   * A dope constant function which is derived from the dealii constant function interface.
   *
   */
  template<int dim>
  class ConstantFunction : public ZeroFunction<dim>
  {
  public:
    ConstantFunction(const double value,
                     const unsigned int n_components = 1)
      : ZeroFunction<dim>(n_components), function_value_(value)
    {
    }
    ;

    virtual
    ~ConstantFunction()
    {
    }
    ;

    virtual double
    value(const dealii::Point<dim> &/*p*/,
          const unsigned int /*component*/) const
    {
      return function_value_;
    }

    virtual void
    vector_value(const dealii::Point<dim> &/*p*/,
                 dealii::Vector<double> &return_value) const
    {
      Assert(return_value.size() == this->n_components,
             ExcDimensionMismatch (return_value.size(), this->n_components));

      std::fill(return_value.begin(), return_value.end(), function_value_);
    }

    virtual void
    value_list(const std::vector<dealii::Point<dim> > &/*points*/,
               std::vector<double> &values,
               const unsigned int /*component*/ = 0) const
    {
//      Assert(values.size() == points.size(),
//       ExcDimensionMismatch(values.size(), points.size()));

      std::fill(values.begin(), values.end(), function_value_);
    }

    virtual void
    vector_value_list(const std::vector<dealii::Point<dim> > &points,
                      std::vector<dealii::Vector<double> > &values) const
    {
      Assert(values.size() == points.size(),
             ExcDimensionMismatch(values.size(), points.size()));

      for (unsigned int i = 0; i < points.size(); ++i)
        {
          Assert(values[i].size() == this->n_components,
                 ExcDimensionMismatch(values[i].size(), this->n_components));
          std::fill(values[i].begin(), values[i].end(), function_value_);
        };
    }

  private:
    const double function_value_;
  };

} //end of namespace

#endif
