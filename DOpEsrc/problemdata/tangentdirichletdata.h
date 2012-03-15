#ifndef _TANGENT_DIRICHLET_DATA_H_
#define _TANGENT_DIRICHLET_DATA_H_

#include "function_wrapper.h"
#include "dofhandler_wrapper.h"
#include "fevalues_wrapper.h"

namespace DOpE
{

  /**
   * This class is used to extract the Dirichlet Data for the Tangent Problem
   */
  template<typename DD, typename VECTOR, int dopedim, int dealdim>
    class TangentDirichletData : public DOpEWrapper::Function<dealdim>
  {
  public:
  TangentDirichletData(const DD& data) : DOpEWrapper::Function<dealdim>(data.n_components(), data.InitialTime()), _dirichlet_data(data)
    {
//      _control_dof_handler = NULL;
//      _state_dof_handler = NULL;
      _param_values = NULL;
      _domain_values = NULL;
      _color = 0;
    }

    /**
     * Initializes the private data, should be called prior to any value call!
     */
    void ReInit(
//                const DOpEWrapper::DoFHandler<dopedim> & control_dof_handler,
//		const DOpEWrapper::DoFHandler<dealdim> &state_dof_handler,
		const std::map<std::string, const dealii::Vector<double>* > &param_values,
		const std::map<std::string, const VECTOR* > &domain_values,
		unsigned int color)
    {
//      _control_dof_handler = &control_dof_handler;
//      _state_dof_handler = &state_dof_handler;
      _param_values = &param_values;
      _domain_values = &domain_values;
      _color = color;
    }


    /**
     * Accesses the values of the dirichlet data
     */
    double value (const dealii::Point<dealdim>   &p,
		  const unsigned int  component) const
    {
      return _dirichlet_data.Data_Q(
//                                    _control_dof_handler,
//			     _state_dof_handler,
			     _param_values,
			     _domain_values,
			     _color,
			     p,
			     component);
    }

    /**
     * This Function is used to transfer the current time to the dirichlet data if needed this should be stored.
     *
     * @param time      The current time
     */
    void SetTime(double time) const
    {
      _dirichlet_data.SetTime(time);
    }
  private:
    const DD& _dirichlet_data;
//    const DOpEWrapper::DoFHandler<dopedim>*  _control_dof_handler;
//    const DOpEWrapper::DoFHandler<dealdim>* _state_dof_handler;
    const std::map<std::string, const dealii::Vector<double>* >* _param_values;
    const std::map<std::string, const VECTOR* >* _domain_values;
    unsigned int _color;
  };
}
#endif
