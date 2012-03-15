#ifndef _DOPE_FUNCTION_H_
#define _DOPE_FUNCTION_H_

#include <base/function.h>

namespace DOpEWrapper
{
/**
 * A dope function which is derived from the dealii function interface.
 *
 * @tparam<dim>    The considered function has dimension `dim'
 */
template<int dim>
class Function: public dealii::Function<dim>
{
  public:
    Function(const unsigned int n_components = 1, const double initial_time = 0.0) :
      dealii::Function<dim>(n_components, initial_time)
    {
      _init_time = initial_time;
    }

    /******************************************************/
    void vector_value(const dealii::Point<dim> &p, dealii::Vector<double> &return_value) const
    {
      Assert (return_value.size() == this->n_components,
          ExcDimensionMismatch (return_value.size(), this->n_components));
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
    virtual void SetTime(double time __attribute__((unused))) const
    {
    }
    ;

    /**
     * Returns the initial time given in the constructor.
     */
    double InitialTime() const
    {
      return _init_time;
    }
    ;
  private:
    double _init_time;
};

/******************************************************/
/******************************************************/

/**
 * A dope zero function which is derived from the dealii zero function interface.
 */
template<int dim>
class ZeroFunction: public Function<dim>
{
  public:

    ZeroFunction(const unsigned int n_components = 1) :
      Function<dim> (n_components)
    {
    }

    virtual ~ZeroFunction()
    {
    }

    virtual double value(const dealii::Point<dim> &p __attribute__((unused)), const unsigned int component __attribute__((unused))) const
    {
      return 0.0;
    }

    virtual void vector_value(const dealii::Point<dim> &p __attribute__((unused)), dealii::Vector<double> &return_value) const
    {
      Assert (return_value.size() == this->n_components,
          ExcDimensionMismatch (return_value.size(), this->n_components));

      std::fill(return_value.begin(), return_value.end(), 0.0);
    }

    virtual void value_list(const std::vector<dealii::Point<dim> > &points,
                            std::vector<double> &values,
                            const unsigned int component __attribute__((unused)) = 0) const
    {
      Assert (values.size() == points.size(),
          ExcDimensionMismatch(values.size(), points.size()));

      std::fill(values.begin(), values.end(), 0.);
    }

    virtual void vector_value_list(const std::vector<dealii::Point<dim> > &points, std::vector<
        dealii::Vector<double> > &values) const
    {
      Assert (values.size() == points.size(),
          ExcDimensionMismatch(values.size(), points.size()));

      for (unsigned int i = 0; i < points.size(); ++i)
      {
        Assert (values[i].size() == this->n_components,
            ExcDimensionMismatch(values[i].size(), this->n_components));
        std::fill(values[i].begin(), values[i].end(), 0.);
      };
    }

    virtual dealii::Tensor<1, dim> gradient(const dealii::Point<dim> &p __attribute__((unused)),
                                            const unsigned int component __attribute__((unused)) =
                                                0) const
    {
      return dealii::Tensor<1, dim>();
    }

    virtual void vector_gradient(const dealii::Point<dim> &p __attribute__((unused)),
                                 std::vector<dealii::Tensor<1, dim> > &gradients) const
    {
      Assert (gradients.size() == this->n_components,
          ExcDimensionMismatch(gradients.size(), this->n_components));

      for (unsigned int c = 0; c < this->n_components; ++c)
        gradients[c].clear();
    }

    virtual void gradient_list(const std::vector<dealii::Point<dim> > &points, std::vector<
        dealii::Tensor<1, dim> > &gradients, const unsigned int component __attribute__((unused)) =
        0) const
    {
      Assert (gradients.size() == points.size(),
          ExcDimensionMismatch(gradients.size(), points.size()));

      for (unsigned int i = 0; i < points.size(); ++i)
        gradients[i].clear();
    }

    virtual void vector_gradient_list(const std::vector<dealii::Point<dim> > &points, std::vector<
        std::vector<dealii::Tensor<1, dim> > > &gradients) const
    {
      Assert (gradients.size() == points.size(),
          ExcDimensionMismatch(gradients.size(), points.size()));
      for (unsigned int i = 0; i < points.size(); ++i)
      {
        Assert (gradients[i].size() == this->n_components,
            ExcDimensionMismatch(gradients[i].size(), this->n_components));
        for (unsigned int c = 0; c < this->n_components; ++c)
          gradients[i][c].clear();
      };
    }
};

/******************************************************/

/**
 * A dope constant function which is derived from the dealii constant function interface.
 */
template<int dim>
class ConstantFunction: public ZeroFunction<dim>
{
  public:
    ConstantFunction(const double value, const unsigned int n_components = 1) :
      ZeroFunction<dim> (n_components), _function_value(value)
    {
    }
    ;

    virtual ~ConstantFunction()
    {
    }
    ;

    virtual double value(const dealii::Point<dim> &p __attribute__((unused)), const unsigned int component __attribute__((unused))) const
    {
      return _function_value;
    }

    virtual void vector_value(const dealii::Point<dim> &p __attribute__((unused)), dealii::Vector<double> &return_value) const
    {
      Assert (return_value.size() == this->n_components,
          ExcDimensionMismatch (return_value.size(), this->n_components));

      std::fill(return_value.begin(), return_value.end(), _function_value);
    }

    virtual void value_list(const std::vector<dealii::Point<dim> > &points,
                            std::vector<double> &values,
                            const unsigned int component __attribute__((unused)) = 0) const
    {
      Assert (values.size() == points.size(),
          ExcDimensionMismatch(values.size(), points.size()));

      std::fill(values.begin(), values.end(), _function_value);
    }

    virtual void vector_value_list(const std::vector<dealii::Point<dim> > &points, std::vector<
        dealii::Vector<double> > &values) const
    {
      Assert (values.size() == points.size(),
          ExcDimensionMismatch(values.size(), points.size()));

      for (unsigned int i = 0; i < points.size(); ++i)
      {
        Assert (values[i].size() == this->n_components,
            ExcDimensionMismatch(values[i].size(), this->n_components));
        std::fill(values[i].begin(), values[i].end(), _function_value);
      };
    }

  private:
    const double _function_value;
};

}//end of namespace

#endif
