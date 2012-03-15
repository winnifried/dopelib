#ifndef _PDEPROBLEM_INTERFACE_H_
#define _PDEPROBLEM_INTERFACE_H_

#include "dopeexceptionhandler.h"
#include "outputhandler.h"
#include "controlvector.h"
#include "constraintvector.h"
#include "reducedprobleminterface.h"
#include "DOpEtypes.h"

#include <assert.h>

#include <lac/vector.h>

namespace DOpE
{
  //Predeclaration necessary
  template<typename VECTOR>
    class DOpEOutputHandler;
  template<typename VECTOR>
    class DOpEExceptionHandler;
  /////////////////////////////


  /**
   * A template for different solver types to be used for solving
   * PDE- as well as optimization problems.
   */
  template<typename PROBLEM, typename VECTOR, int dealdim>
    class PDEProblemInterface : public ReducedProblemInterface_Base<VECTOR>
    {
      public:
        PDEProblemInterface(PROBLEM *OP, int base_priority = 0) :
          ReducedProblemInterface_Base<VECTOR> ()
        {
          _OP = OP;
          _base_priority = base_priority;
          _post_index = "_" + this->GetProblem()->GetName();
        }
        ~PDEProblemInterface()
        {
        }

        /******************************************************/

        /**
         * Basic function which is given to instatsolver.h and statsolver.h, respectively,
         * and reinitializes vectors, matrices, etc.
         *
         */
        virtual void
        ReInit()
        {
          this->GetProblem()->ReInit("reduced");
        }

        /******************************************************/

        /**
         * Basic function to compute reduced functionals.
         *
         */
        virtual void
        ComputeReducedFunctionals()=0;

        /******************************************************/

        /**
         * Basic function to compute the error indicators with
         * the DWR method and higher order interpolation to gain the weights.
         * We assume that the state is already computed,
         * whereas the dual solution will be computed inside this function.
         *
         * @param ref_ind           The Vector in which the function writes
         *                          the error indicators on the different cells.
         *                          This vector is resized to the number of cells
         *                          of the actual grid.
         * @param ee_state          Which terms of the error Identity should get computed
         *                          (i.e. primal-term, dual-term, both)?
         * @param weight_comp       How to compute the weights?
         *
         * @return                  The error in the previously specified functional.
         *
         */
        virtual float
        ComputeRefinementIndicators(Vector<float>& ref_ind,
            DOpEtypes::EE_state ee_state = DOpEtypes::EE_state::mixed,
            DOpEtypes::WeightComputation weight_comp =
                DOpEtypes::WeightComputation::higher_order_interpolation) = 0;

        /******************************************************/
        /**
         * Sets the type of the Problem _OP. This function secures the proper initialization of the
         * FEValues after the type has changed. See also the documentation of SetType in optproblem.h
         */
        void
        SetProblemType(std::string type, unsigned int num = 0)
        {
          this->GetProblem()->SetType(type, num);
        }

      protected:

        PROBLEM*
        GetProblem()
        {
          return _OP;
        }
        std::string
        GetPostIndex()
        {
          return _post_index;
        }
        int
        GetBasePriority()
        {
          return _base_priority;
        }
      private:
        PROBLEM* _OP;
        int _base_priority;
        std::string _post_index;
    };

}
#endif
