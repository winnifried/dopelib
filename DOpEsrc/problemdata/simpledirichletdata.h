#ifndef _SIMPLE_DIRICHLET_H_
#define _SIMPLE_DIRICHLET_H_

#include "dirichletdatainterface.h"
#include "function_wrapper.h"

namespace DOpE
{

  /**
   * A Simple Interface Class, that sets DirichletData given by a DOpEWrapper::Function. This means they don't depend on control or state values
   */
  template<typename VECTOR, int dopedim, int dealdim=dopedim>
    class SimpleDirichletData : public DirichletDataInterface<VECTOR, dopedim,dealdim>
  {
  public:
  SimpleDirichletData(const DOpEWrapper::Function<dealdim>& data) : DirichletDataInterface<VECTOR, dopedim,dealdim>(), _data(data)
    {}

  double Data(
//              const DOpEWrapper::DoFHandler<dopedim> * control_dof_handler __attribute__((unused)),
//	      const DOpEWrapper::DoFHandler<dealdim> *state_dof_handler __attribute__((unused)),
	      const std::map<std::string, const dealii::Vector<double>* > *param_values __attribute__((unused)),
	      const std::map<std::string, const VECTOR* > *domain_values __attribute__((unused)),
	      unsigned int color __attribute__((unused)),
	      const dealii::Point<dealdim>& point,
	      const unsigned int component) const
  {
    return _data.value(point,component);
  }

  double Data_Q(
//                const DOpEWrapper::DoFHandler<dopedim> * control_dof_handler __attribute__((unused)),
//		const DOpEWrapper::DoFHandler<dealdim> *state_dof_handler __attribute__((unused)),
		const std::map<std::string, const dealii::Vector<double>* > *param_values __attribute__((unused)),
		const std::map<std::string, const VECTOR* > *domain_values __attribute__((unused)),
		unsigned int color __attribute__((unused)),
		const dealii::Point<dealdim>& point __attribute__((unused)),
		const unsigned int component __attribute__((unused))) const
  {
    return 0.;
  }

  void SetTime(double time) const
  {
    _data.SetTime(time);
  }

  unsigned int n_components() const
  {
    return _data.n_components;
  }

  double InitialTime() const { return _data.InitialTime();}

  private:
    const DOpEWrapper::Function<dealdim>& _data;
  };

}


#endif
