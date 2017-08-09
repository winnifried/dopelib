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
       **/
      virtual void PipeCouplingResidual(dealii::Vector<double>& res, const dealii::Vector<double>& u) const = 0;

      /**
       * Evaluates the Matrix of the pipe coupling conditions and the coupling to the outflow 
       * flux.
       *
       * @param matrix  The matrix to be calculated
       * @param present_in_outflow A vector indicating which flux variables are outflow.
       **/
      virtual void CouplingMatrix(dealii::SparseMatrix<double>& matrix, const std::vector<bool>& present_in_outflow) const = 0;


      /** 
       * Adds the flux-flux coupling indices to the given SparsityPattern
       * Needs to reinitialize the given Pattern to the correct size
       * 
       * @param sparsity   The pattern into which the entries need to be stored.
       **/
      virtual void GetFluxSparsityPattern(dealii::SparsityPattern& sparsity) const = 0;
    };
  }
}

#endif
