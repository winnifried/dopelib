#ifndef NETWORK_INTERFACE_
#define NETWORK_INTERFACE_

#include <include/parameterreader.h>

namespace DOpE
{
  namespace Networks
  {

    class NetworkInterface
    {
    public:
      virtual unsigned int GetNPipes() const = 0;
      virtual unsigned int GetNNodes() const = 0;

      /**
       * Evaluates the residual of the pipe coupling conditions of the vector u of coupling
       * variables, i.e., if the coupling condition is Au - b = 0 the residual is res = Au-b.
       *
       * @param res  The residual of the coupling condition
       * @param u    The vector in which the residual is to be calculated
       * @param present_in_outflow A vector indicating which flux variables are outflow.
       **/
      virtual void PipeCouplingResidual(dealii::Vector<double> &res,
                                        const dealii::Vector<double> &u,
                                        const std::vector<bool> &present_in_outflow) const = 0;

      /**
       * Evaluates the Matrix of the pipe coupling conditions and the coupling to the outflow
       * flux.
       *
       * @param matrix  The matrix to be calculated
       * @param present_in_outflow A vector indicating which flux variables are outflow.
       **/
      virtual void CouplingMatrix(dealii::SparseMatrix<double> &matrix,
                                  const std::vector<bool> &present_in_outflow) const = 0;

      /**
       * For initial conditions, it is sometime needed that an other coupling is used,
       * In particular, if initial values are given by algebraic equations on the
       * pipe, then all fluxes must be found as outflow fluxes.
       *
       * @param res  The residual of the coupling condition
       * @param u    The vector in which the residual is to be calculated
       * @param present_in_outflow A vector indicating which flux variables are outflow.
       **/

      virtual void Init_PipeCouplingResidual(dealii::Vector<double> &res,
                                             const dealii::Vector<double> &u,
                                             const std::vector<bool> &/*present_in_outflow*/) const
      {
        //Sizes must all be equal (and 2*NPipes*NComp)
        assert(res.size()==u.size());
        //assert(res.size()==present_in_outflow.size());
        assert(2*(res.size()/2) == res.size());//an even number

        for (unsigned int i = 0; i< res.size(); i++)
          {
            //assert(present_in_outflow[i]);
            res[i] = -u[i];
          }
      }

      /**
       * The matrix corresponding to the Init_PipeCouplingResidual
       *
       * @param matrix  The matrix to be calculated
       * @param present_in_outflow A vector indicating which flux variables are outflow.
       **/
      virtual void Init_CouplingMatrix(dealii::SparseMatrix<double> &matrix,
                                       const std::vector<bool> &/*present_in_outflow*/) const
      {
        assert(matrix.m()==matrix.n());
        //assert(matrix.m()==present_in_outflow.size());
        for (unsigned int i = 0; i< matrix.m(); i++)
          {
            //assert(present_in_outflow[i]);
            matrix.set(i,i,-1);
          }
      }


      /**
       * Adds the flux-flux coupling indices to the given SparsityPattern
       * Needs to reinitialize the given Pattern to the correct size
       *
       * @param sparsity   The pattern into which the entries need to be stored.
       **/
      virtual void GetFluxSparsityPattern(dealii::SparsityPattern &sparsity) const = 0;

    };
  }
}

#endif
