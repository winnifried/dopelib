#ifndef _PDE_INTERFACE_H_
#define _PDE_INTERFACE_H_

#include <map>
#include <string>

#include <fe/fe_system.h>
#include <fe/fe_values.h>
#include <fe/mapping.h>
#include <lac/full_matrix.h>
#include <base/function.h>

#include "fevalues_wrapper.h"
#include "celldatacontainer.h"
#include "facedatacontainer.h"
#include "multimesh_celldatacontainer.h"
#include "multimesh_facedatacontainer.h"

namespace DOpE
{

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim = dopedim>
    class PDEInterface
    {
      public:
        PDEInterface();
        virtual
        ~PDEInterface();

        /******************************************************/

        /**
         * Documentation in optproblemcontainer.h.
         */
        virtual void
        CellEquation(const CDC<DOFHANDLER, VECTOR, dealdim>& cdc,
            dealii::Vector<double> &/*local_cell_vector*/, double scale,
            double /*scale_ico*/);

        /******************************************************/

        /**
         * Documentation in optproblemcontainer.h.
         */

        virtual void
        StrongCellResidual(const CDC<DOFHANDLER, VECTOR, dealdim>& cdc,
            const CDC<DOFHANDLER, VECTOR, dealdim>& cdc_weight, double&,
            double scale, double /*scale_ico*/);

        /******************************************************/

        /**
         * Documentation in optproblemcontainer.h.
         */
        virtual void
        CellTimeEquation(const CDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/);

        /******************************************************/

        /**
         * Documentation in optproblemcontainer.h.
         */
        virtual void
        CellTimeEquationExplicit(const CDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/);

        /******************************************************/

        virtual void
        CellEquation_U(const CDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::Vector<double> &/*local_cell_vector*/, double scale);

        /******************************************************/

        virtual void
        StrongCellResidual_U(const CDC<DOFHANDLER, VECTOR, dealdim>&,
            const CDC<DOFHANDLER, VECTOR, dealdim>& cdc_weight, double&,
            double scale);

        /******************************************************/

        virtual void
        CellEquation_UT(const CDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/);

        /******************************************************/

        virtual void
        CellEquation_UTT(const CDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/);

        /******************************************************/

        virtual void
        CellEquation_Q(const CDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/);

        /******************************************************/

        virtual void
        CellEquation_QT(const CDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/);

        /******************************************************/

        virtual void
        CellEquation_QTT(const CDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/);

        /******************************************************/

        virtual void
        CellEquation_UU(const CDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/);

        /******************************************************/

        virtual void
        CellEquation_QU(const CDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/);

        /******************************************************/

        virtual void
        CellEquation_UQ(const CDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/);

        /******************************************************/

        virtual void
        CellEquation_QQ(const CDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/);

        /******************************************************/

        /**
         * Documentation in optproblemcontainer.h.
         */
        virtual void
        CellRightHandSide(const CDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/);

        /******************************************************/

        /**
         * Documentation in optproblemcontainer.h.
         */
        virtual void
        CellMatrix(const CDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::FullMatrix<double> &/*local_entry_matrix*/,
            double /*scale*/, double /*scale_ico*/);

        /******************************************************/

        /**
         * Documentation in optproblemcontainer.h.
         */
        virtual void
        CellTimeMatrix(const CDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::FullMatrix<double> &/*local_entry_matrix*/);

        /******************************************************/

        /**
         * Documentation in optproblemcontainer.h.
         */
        virtual void
        CellTimeMatrixExplicit(const CDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::FullMatrix<double> &/*local_entry_matrix*/);

        /******************************************************/

        virtual void
        CellMatrix_T(const CDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::FullMatrix<double> &/*local_entry_matrix*/, double, double);

        /******************************************************/

        virtual void
        ControlCellEquation(const CDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/);

        /******************************************************/

        virtual void
        ControlCellMatrix(const CDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::FullMatrix<double> &/*local_entry_matrix*/);
        /******************************************************/

        /******************************************************/
        // Functions for Face Integrals
        /**
         * Documentation in optproblemcontainer.h.
         */
        virtual void
        FaceEquation(const FDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/);
        /******************************************************/

        /**
         * Documentation in optproblemcontainer.h.
         */

        virtual void
        StrongFaceResidual(const FDC<DOFHANDLER, VECTOR, dealdim>&,
            const FDC<DOFHANDLER, VECTOR, dealdim>& fdc_weight, double&,
            double /*scale*/);

        /******************************************************/

        virtual void
        FaceEquation_U(const FDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/);

        /******************************************************/

        virtual void
        StrongFaceResidual_U(const FDC<DOFHANDLER, VECTOR, dealdim>&,
            const FDC<DOFHANDLER, VECTOR, dealdim>& cdc_weight, double&,
            double /*scale*/);

        /******************************************************/

        virtual void
        FaceEquation_UT(const FDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/);

        /******************************************************/

        virtual void
        FaceEquation_UTT(const FDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/);

        /******************************************************/

        virtual void
        FaceEquation_Q(const FDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/);

        /******************************************************/

        virtual void
        FaceEquation_QT(const FDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/);

        /******************************************************/

        virtual void
        FaceEquation_QTT(const FDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/);

        /******************************************************/

        virtual void
        FaceEquation_UU(const FDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/);

        /******************************************************/

        virtual void
        FaceEquation_QU(const FDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/);

        /******************************************************/

        virtual void
        FaceEquation_UQ(const FDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/);

        /******************************************************/

        virtual void
        FaceEquation_QQ(const FDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/);

        /******************************************************/

        virtual void
        FaceRightHandSide(const FDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/);

        /******************************************************/

        /**
         * Documentation in optproblemcontainer.h.
         */
        virtual void
        FaceMatrix(const FDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::FullMatrix<double> &/*local_entry_matrix*/);

        /******************************************************/

        virtual void
        FaceMatrix_T(const FDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::FullMatrix<double> &/*local_entry_matrix*/);

        /******************************************************/

        virtual void
        ControlFaceEquation(const FDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/);

        /******************************************************/

        virtual void
        ControlFaceMatrix(const FDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::FullMatrix<double> &/*local_entry_matrix*/);

        /******************************************************/
        //Functions for Interface Integrals
        virtual void
        InterfaceMatrix(const FDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::FullMatrix<double> &/*local_entry_matrix*/);

        /******************************************************/
        //Functions for Interface Integrals
        virtual void
        InterfaceMatrix_T(const FDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::FullMatrix<double> &/*local_entry_matrix*/);

        /******************************************************/
        /**
         * Documentation in optproblemcontainer.h
         */

        virtual void
        InterfaceEquation(const FDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/);

        /******************************************************/
        /**
         * Documentation in optproblemcontainer.h
         */

        virtual void
        InterfaceEquation_U(const FDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/);

        /******************************************************/
        // Functions for Boundary Integrals
        /**
         * Documentation in optproblemcontainer.h.
         */
        virtual void
        BoundaryEquation(const FDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/);

        /******************************************************/

        /**
         * Documentation in optproblemcontainer.h.
         */

        virtual void
        StrongBoundaryResidual(const FDC<DOFHANDLER, VECTOR, dealdim>&,
            const FDC<DOFHANDLER, VECTOR, dealdim>& fdc_weight, double&,
            double /*scale*/);

        /******************************************************/

        virtual void
        BoundaryEquation_U(const FDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/);

        /******************************************************/

        virtual void
        StrongBoundaryResidual_U(const FDC<DOFHANDLER, VECTOR, dealdim>&,
            const FDC<DOFHANDLER, VECTOR, dealdim>& fdc_weight, double&,
            double /*scale*/);

        /******************************************************/

        virtual void
        BoundaryEquation_UT(const FDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/);

        /******************************************************/

        virtual void
        BoundaryEquation_UTT(const FDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/);

        /******************************************************/

        virtual void
        BoundaryEquation_Q(const FDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/);

        /******************************************************/

        virtual void
        BoundaryEquation_QT(const FDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/);

        /******************************************************/

        virtual void
        BoundaryEquation_QTT(const FDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/);

        /******************************************************/

        virtual void
        BoundaryEquation_UU(const FDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/);

        /******************************************************/

        virtual void
        BoundaryEquation_QU(const FDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/);

        /******************************************************/

        virtual void
        BoundaryEquation_UQ(const FDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/);

        /******************************************************/

        virtual void
        BoundaryEquation_QQ(const FDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/);

        /******************************************************/

        virtual void
        BoundaryRightHandSide(const FDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/);

        /******************************************************/

        /**
         * Documentation in optproblemcontainer.h.
         */
        virtual void
        BoundaryMatrix(const FDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::FullMatrix<double> &/*local_entry_matrix*/);

        /******************************************************/

        virtual void
        BoundaryMatrix_T(const FDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::FullMatrix<double> &/*local_entry_matrix*/);

        /******************************************************/

        virtual void
        ControlBoundaryEquation(const FDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/);

        /******************************************************/

        virtual void
        ControlBoundaryMatrix(const FDC<DOFHANDLER, VECTOR, dealdim>&,
            dealii::FullMatrix<double> &/*local_entry_matrix*/);

        /******************************************************/
      /******************************************************/
      /****For the initial values ***************/
      /* Default is componentwise L2 projection */ 
      virtual void
        Init_CellEquation(const CDC<DOFHANDLER, VECTOR, dealdim>& cdc,
            dealii::Vector<double> &local_cell_vector, double scale,
            double /*scale_ico*/)
        {
          const DOpEWrapper::FEValues<dealdim> & state_fe_values =
              cdc.GetFEValuesState();
          unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
          unsigned int n_q_points = cdc.GetNQPoints();
          std::vector<dealii::Vector<double> > uvalues;
          uvalues.resize(n_q_points,
              dealii::Vector<double>(this->GetStateNComponents()));
          cdc.GetValuesState("last_newton_solution", uvalues);

          dealii::Vector<double> f_values(
              dealii::Vector<double>(this->GetStateNComponents()));

          for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
            {
              for (unsigned int i = 0; i < n_dofs_per_cell; i++)
                {
                  for (unsigned int comp = 0;
                      comp < this->GetStateNComponents(); comp++)
                    {
                      local_cell_vector(i) += scale
                          * (state_fe_values.shape_value_component(i, q_point,
                              comp) * uvalues[q_point](comp))
                          * state_fe_values.JxW(q_point);
                    }
                }
            } //endfor q_point
        }

        virtual void
        Init_CellRhs(const dealii::Function<dealdim>* init_values,
            const CDC<DOFHANDLER, VECTOR, dealdim>& cdc,
            dealii::Vector<double> &local_cell_vector, double scale)
        {
          const DOpEWrapper::FEValues<dealdim> & state_fe_values =
              cdc.GetFEValuesState();
          unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
          unsigned int n_q_points = cdc.GetNQPoints();

          dealii::Vector<double> f_values(
              dealii::Vector<double>(this->GetStateNComponents()));

          for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
            {
              init_values->vector_value(
                  state_fe_values.quadrature_point(q_point), f_values);

              for (unsigned int i = 0; i < n_dofs_per_cell; i++)
                {
                  for (unsigned int comp = 0;
                      comp < this->GetStateNComponents(); comp++)
                    {
                      local_cell_vector(i) += scale
                          * (f_values(comp)
                              * state_fe_values.shape_value_component(i,
                                  q_point, comp))
                          * state_fe_values.JxW(q_point);
                    }
                }
            }
        }

      virtual void Init_CellMatrix(const CDC<DOFHANDLER, VECTOR, dealdim>& cdc,
			   dealii::FullMatrix<double> &local_entry_matrix, double scale,
				   double /*scale_ico*/)
      {
	const DOpEWrapper::FEValues<dealdim> & state_fe_values = cdc.GetFEValuesState();
	unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();
	unsigned int n_q_points = cdc.GetNQPoints();
	
	
	for (unsigned int q_point=0;q_point<n_q_points;q_point++)
	{
	  for (unsigned int i=0;i<n_dofs_per_cell;i++)
	  {
	    for (unsigned int j=0;j<n_dofs_per_cell;j++)
	    {
	      for( unsigned int comp =0; comp < this->GetStateNComponents(); comp++)
	      {
		local_entry_matrix(i,j) +=  scale * state_fe_values.shape_value_component(i,q_point,comp) *state_fe_values.shape_value_component(j,q_point,comp)
		  * state_fe_values.JxW(q_point);
	      }
	    }
	  }
	}
      }
      /*************************************************************/
        virtual dealii::UpdateFlags
        GetUpdateFlags() const;
        virtual dealii::UpdateFlags
        GetFaceUpdateFlags() const;
        virtual bool
        HasFaces() const;
        virtual bool
        HasInterfaces() const;

        /******************************************************/

        void
        SetProblemType(std::string type);

        /******************************************************/

        virtual unsigned int
        GetControlNBlocks() const;
        virtual unsigned int
        GetStateNBlocks() const;
        virtual std::vector<unsigned int>&
        GetControlBlockComponent();
        virtual const std::vector<unsigned int>&
        GetControlBlockComponent() const;
        virtual std::vector<unsigned int>&
        GetStateBlockComponent();
        virtual const std::vector<unsigned int>&
        GetStateBlockComponent() const;

        /******************************************************/

        virtual void
        SetTime(double t __attribute__((unused))) const
        {
        }

        /******************************************************/

        unsigned int
        GetStateNComponents() const;

        /******************************************************/
      protected:
        std::string _problem_type;

      private:
    };
}

#endif
