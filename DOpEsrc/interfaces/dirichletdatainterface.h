#ifndef _DIRICHLET_INTERFAC_H_
#define _DIRICHLET_INTERFAC_H_

namespace DOpE
{
  /**
   * Interface for DirichletData
   */
  template<typename VECTOR, int dopedim, int dealdim=dopedim>
    class DirichletDataInterface
  {
  public:
    /**
     * This Function should return the dirichlet value in the component component at the given point
     *
     * @param control_dof_handler   The DOpEWrapper::DoFHandler for the control variable
     * @param state_dof_handler     The DOpEWrapper::DoFHandler for the state variable
     * @param param_values          A std::map containing parameter data (e.g. non space dependent data). If the control
     *                              is done by parameters, it is contained in this map at the position "control".
     * @param domain_values         A std::map containing domain data (e.g. nodal vectors for FE-Functions). If the control
     *                              is distributed, it is contained in this map at the position "control".
     *                              WARNING! never use the value "last_newton_solution" in this map!
     * @param color                 A color indicating the boundary at which we are
     * @param point                 The point at which we would like to evaluate the dirichlet data.
     * @param component             An unsigned integer indicating the component we would like to use.
     *
     * @return                      The dirichletdata for the componten at point.
     */
     virtual double Data(
//                         const DOpEWrapper::DoFHandler<dopedim> * control_dof_handler __attribute__((unused)),
//			 const DOpEWrapper::DoFHandler<dealdim> *state_dof_handler __attribute__((unused)),
			 const std::map<std::string, const dealii::Vector<double>* > *param_values __attribute__((unused)),
			 const std::map<std::string, const VECTOR* > *domain_values __attribute__((unused)),
			 unsigned int color __attribute__((unused)),
			 const dealii::Point<dealdim>& point __attribute__((unused)),
			 unsigned int component __attribute__((unused))) const =0;

     /**
     * This Function should return the derivative wrt. the control of the dirichlet value at the in the component component at the given point.
     * The point at which the derivative should be evaluated is called "control" and the direction "dq" in the corresponding map.
     *
     * @param control_dof_handler   The DOpEWrapper::DoFHandler for the control variable
     * @param state_dof_handler     The DOpEWrapper::DoFHandler for the state variable
     * @param param_values          A std::map containing parameter data (e.g. non space dependent data). If the control
     *                              is done by parameters, it is contained in this map at the position "control".
     * @param domain_values         A std::map containing domain data (e.g. nodal vectors for FE-Functions). If the control
     *                              is distributed, it is contained in this map at the position "control".
     *                              WARNING! never use the value "last_newton_solution" in this map!
     * @param color                 A color indicating the boundary at which we are
     * @param point                 The point at which we would like to evaluate the dirichlet data.
     * @param component             An unsigned integer indicating the component we would like to use.
     *
     * @return                      The dirichletdata for the componten at point.
     */
   virtual double Data_Q(
//                         const DOpEWrapper::DoFHandler<dopedim> * control_dof_handler,
//			 const DOpEWrapper::DoFHandler<dealdim> *state_dof_handler,
			 const std::map<std::string, const dealii::Vector<double>* > *param_values,
			 const std::map<std::string, const VECTOR* > *domain_values,
			 unsigned int color,
			 const dealii::Point<dealdim>& point,
			 unsigned int component)  const=0;
   /**
     * This Function should return the transposed derivative wrt. the control of the dirichlet value.
     * The output  should be ordered by components in the control. The Testfunction is given in
     * the domain_values map with the name "adjoint_residual"
     *
     * @param control_dof_handler   The DOpEWrapper::DoFHandler for the control variable
     * @param state_dof_handler     The DOpEWrapper::DoFHandler for the state variable
     * @param param_values          A std::map containing parameter data (e.g. non space dependent data). If the control
     *                              is done by parameters, it is contained in this map at the position "control".
     * @param domain_values         A std::map containing domain data (e.g. nodal vectors for FE-Functions). If the control
     *                              is distributed, it is contained in this map at the position "control".
     *                              WARNING! never use the value "last_newton_solution" in this map!
     * @param color                 A color indicating the boundary at which we are
     * @param point                 The point at which we would like to evaluate the dirichlet data.
     * @param component             An unsigned integer indicating the component we would like to use.
     * @param dof_number            The number of the dof with support on the given point.
     *
     * @param local_vector          The transposed dirichlet data at the point
     */
   virtual void Data_QT (
//                         const DOpEWrapper::DoFHandler<dopedim> * control_dof_handler __attribute__((unused)),
//			 const DOpEWrapper::DoFHandler<dealdim> *state_dof_handler __attribute__((unused)),
			 const std::map<std::string, const dealii::Vector<double>* > *param_values __attribute__((unused)),
			 const std::map<std::string, const VECTOR* > *domain_values __attribute__((unused)),
			 unsigned int color __attribute__((unused)),
			 const dealii::Point<dealdim>& point __attribute__((unused)),
			 unsigned int component __attribute__((unused)),
			 unsigned int  dof_number __attribute__((unused)),
			 dealii::Vector<double>& local_vector __attribute__((unused))) const {}
   /**
     * This Function should return the transposed second derivative wrt. the control of the dirichlet value.
     * The output  should be ordered by components in the control. The Testfunction is given in
     * the domain_values map with the name "hessian_residual". The second testfunction is in the control space
     * and denoted by "dq"
     *
     * @param control_dof_handler   The DOpEWrapper::DoFHandler for the control variable
     * @param state_dof_handler     The DOpEWrapper::DoFHandler for the state variable
     * @param param_values          A std::map containing parameter data (e.g. non space dependent data). If the control
     *                              is done by parameters, it is contained in this map at the position "control".
     * @param domain_values         A std::map containing domain data (e.g. nodal vectors for FE-Functions). If the control
     *                              is distributed, it is contained in this map at the position "control".
     *                              WARNING! never use the value "last_newton_solution" in this map!
     * @param color                 A color indicating the boundary at which we are
     * @param point                 The point at which we would like to evaluate the dirichlet data.
     * @param component             An unsigned integer indicating the component we would like to use.
     * @param dof_number            The number of the dof with support on the given point.
     *
     * @param local_vector          The transposed dirichlet data at the point
     */
   virtual void Data_QQT (
//                          const DOpEWrapper::DoFHandler<dopedim> * control_dof_handler __attribute__((unused)),
//			  const DOpEWrapper::DoFHandler<dealdim> *state_dof_handler __attribute__((unused)),
			  const std::map<std::string, const dealii::Vector<double>* > *param_values __attribute__((unused)),
			  const std::map<std::string, const VECTOR* > *domain_values __attribute__((unused)),
			  unsigned int color __attribute__((unused)),
			  const dealii::Point<dealdim>& point __attribute__((unused)),
			  unsigned int component __attribute__((unused)),
			  unsigned int  dof_number __attribute__((unused)),
			  dealii::Vector<double>& local_vector __attribute__((unused))) const {}
   /**
    * This Function is used to transfer the current time to the dirichlet data if needed this should be stored.
    *
    * @param time      The current time
    */
   virtual void SetTime(double time __attribute__((unused))) const {}

   /**
    * Returns the number of components used for all derived DOpEWrapper::Function objects
    */
   virtual unsigned int n_components() const { return 0;}
   /**
    * Returns the Initial Time used for all derived DOpEWrapper::Function objects
    */
   virtual double InitialTime() const { return 0.0;}
   /**
    * This Function must return true if the DirichletData depend on the Control.
    * Otherwise Gradient evaluation will fail.
    */
   virtual bool NeedsControl() const { return  false; }
  };
}
#endif
