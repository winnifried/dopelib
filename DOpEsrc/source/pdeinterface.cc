#include "pdeinterface.h"
#include "dopeexception.h"

#include <iostream>

using namespace dealii;

namespace DOpE
{

  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::PDEInterface()
    {

    }

  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::~PDEInterface()
    {

    }

  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::CellEquation(
        const CDC<DOFHANDLER, VECTOR, dealdim>& /*cdc*/,
        dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/,
        double /*scale_ico*/)
    {
      throw DOpEException("Not Implemented", "PDEInterface::CellEquation");
    }

  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::StrongCellResidual(
        const CDC<DOFHANDLER, VECTOR, dealdim>& /*cdc*/,
        const CDC<DOFHANDLER, VECTOR, dealdim>& /*cdc_weight*/, double&,
        double /*scale*/, double /*scale_ico*/)
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::StrongCellResidual");
    }

  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::CellTimeEquation(
        const CDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/ ,
        double /*scale*/ )
    {
      throw DOpEException("Not Implemented", "PDEInterface::CellTimeEquation");
    }

  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::CellTimeEquation_U(
        const CDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/ ,
        double /*scale*/ )
    {
      throw DOpEException("Not Implemented", "PDEInterface::CellTimeEquation_U");
    }
  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::CellTimeEquation_UT(
        const CDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/ ,
        double /*scale*/ )
    {
      throw DOpEException("Not Implemented", "PDEInterface::CellTimeEquation_UT");
    }
  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::CellTimeEquation_UTT(
        const CDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/ ,
        double /*scale*/ )
    {
      throw DOpEException("Not Implemented", "PDEInterface::CellTimeEquation_UTT");
    }

  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::CellTimeEquationExplicit(
        const CDC<DOFHANDLER, VECTOR, dealdim>& cdc,
        dealii::Vector<double> &local_cell_vector, double scale)
    {
      this->CellTimeEquation(cdc, local_cell_vector, scale);
    }

  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::CellTimeEquationExplicit_U(
        const CDC<DOFHANDLER, VECTOR, dealdim>& cdc,
        dealii::Vector<double> &local_cell_vector, double scale)
    {
      this->CellTimeEquation_U(cdc, local_cell_vector, scale);
    }

  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::CellTimeEquationExplicit_UT(
        const CDC<DOFHANDLER, VECTOR, dealdim>& cdc,
        dealii::Vector<double> &local_cell_vector, double scale)
    {
      this->CellTimeEquation_UT(cdc, local_cell_vector, scale);
    }

  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::CellTimeEquationExplicit_UTT(
        const CDC<DOFHANDLER, VECTOR, dealdim>& cdc,
        dealii::Vector<double> &local_cell_vector, double scale)
    {
      this->CellTimeEquation_UTT(cdc, local_cell_vector, scale);
    }

  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::CellTimeEquationExplicit_UU(
      const CDC<DOFHANDLER, VECTOR, dealdim>& /*cdc*/,
      dealii::Vector<double> &/*local_cell_vector*/, double /*scale*/)
    {
    }

  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::CellEquation_U(
        const CDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/ ,
        double /*scale*/, double /*scale_ico*/ )
    {
      throw DOpEException("Not Implemented", "PDEInterface::CellEquation_U");
    }

  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::StrongCellResidual_U(
        const CDC<DOFHANDLER, VECTOR, dealdim>&,
        const CDC<DOFHANDLER, VECTOR, dealdim>& /*cdc_weight*/, double&,
        double /*scale*/ )
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::StrongCellResidual_U");
    }

  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::CellEquation_UT(
        const CDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/ ,
         double /*scale*/, double /*scale_ico*/ )
    {
      throw DOpEException("Not Implemented", "PDEInterface::CellEquation_UT");
    }

  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::CellEquation_UTT(
        const CDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/ ,
        double /*scale*/, double /*scale_ico*/ )
    {
      throw DOpEException("Not Implemented", "PDEInterface::CellEquation_UTT");
    }

  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::CellEquation_Q(
        const CDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/ ,
        double /*scale*/, double /*scale_ico*/ )
    {
      throw DOpEException("Not Implemented", "PDEInterface::CellEquation_Q");
    }

  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::CellEquation_QT(
        const CDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/ ,
        double /*scale*/, double /*scale_ico*/ )
    {
      throw DOpEException("Not Implemented", "PDEInterface::CellEquation_QT");
    }

  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::CellEquation_QTT(
        const CDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/ ,
        double /*scale*/, double /*scale_ico*/ )
    {
      throw DOpEException("Not Implemented", "PDEInterface::CellEquation_QTT");
    }

  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::CellEquation_UU(
        const CDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/ ,
        double /*scale*/, double /*scale_ico*/ )
    {
      throw DOpEException("Not Implemented", "PDEInterface::CellEquation_UU");
    }

  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::CellEquation_QU(
        const CDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/ ,
        double /*scale*/, double /*scale_ico*/ )
    {
      throw DOpEException("Not Implemented", "PDEInterface::CellEquation_QU");
    }

  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::CellEquation_UQ(
        const CDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/ ,
        double /*scale*/, double /*scale_ico*/ )
    {
      throw DOpEException("Not Implemented", "PDEInterface::CellEquation_UQ");
    }

  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::CellEquation_QQ(
        const CDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/ ,
        double /*scale*/, double /*scale_ico*/ )
    {
      throw DOpEException("Not Implemented", "PDEInterface::CellEquation_QQ");
    }

  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::ControlCellEquation(
        const CDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/ ,
        double /*scale*/ )
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::ControlCellEquation");
    }

  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::CellRightHandSide(
        const CDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/ ,
        double /*scale*/ )
    {
      throw DOpEException("Not Implemented", "PDEInterface::CellRightHandSide");
    }

  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::ControlCellMatrix(
        const CDC<DOFHANDLER, VECTOR, dealdim>&,
        FullMatrix<double> &/*local_entry_matrix*/ )
    {
      throw DOpEException("Not Implemented", "PDEInterface::ControlCellMatrix");
    }

  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::CellMatrix(
        const CDC<DOFHANDLER, VECTOR, dealdim>&,
        FullMatrix<double> &/*local_entry_matrix*/, double/*scale*/,
        double /*scale_ico*/)
    {
      throw DOpEException("Not Implemented", "PDEInterface::CellMatrix");
    }

  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::CellTimeMatrix(
        const CDC<DOFHANDLER, VECTOR, dealdim>&,
        FullMatrix<double> &/*local_entry_matrix*/ )
    {
      throw DOpEException("Not Implemented", "PDEInterface::CellTimeMatrix");
    }
  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::CellTimeMatrix_T(
        const CDC<DOFHANDLER, VECTOR, dealdim>& cdc,
        FullMatrix<double> &local_entry_matrix )
    {
      FullMatrix<double> tmp_mat = local_entry_matrix;
      tmp_mat = 0.;

      //FIXME is this the right behaviour in the instationary case? or what
      //are the correct values for scale and scale_ico?
      CellTimeMatrix(cdc, tmp_mat);
      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();

      for (unsigned int i = 0; i < n_dofs_per_cell; i++)
      {
        for (unsigned int j = 0; j < n_dofs_per_cell; j++)
        {
          local_entry_matrix(j, i) += tmp_mat(i, j);
        }
      }
    }

  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::CellTimeMatrixExplicit(
        const CDC<DOFHANDLER, VECTOR, dealdim>& cdc,
        FullMatrix<double> &local_entry_matrix)
    {
      this->CellTimeMatrix(cdc, local_entry_matrix);
    }

  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::CellTimeMatrixExplicit_T(
        const CDC<DOFHANDLER, VECTOR, dealdim>& cdc,
        FullMatrix<double> &local_entry_matrix)
    {
      FullMatrix<double> tmp_mat = local_entry_matrix;
      tmp_mat = 0.;

      //FIXME is this the right behaviour in the instationary case? or what
      //are the correct values for scale and scale_ico?
      CellTimeMatrixExplicit(cdc, tmp_mat);
      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();

      for (unsigned int i = 0; i < n_dofs_per_cell; i++)
      {
        for (unsigned int j = 0; j < n_dofs_per_cell; j++)
        {
          local_entry_matrix(j, i) += tmp_mat(i, j);
        }
      }
    }

  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::CellMatrix_T(
        const CDC<DOFHANDLER, VECTOR, dealdim>& cdc,
        FullMatrix<double> &local_entry_matrix, double scale, double scale_ico)
    {
      FullMatrix<double> tmp_mat = local_entry_matrix;
      tmp_mat = 0.;

      //FIXME is this the right behaviour in the instationary case? or what
      //are the correct values for scale and scale_ico?
      CellMatrix(cdc, tmp_mat, scale, scale_ico);
      unsigned int n_dofs_per_cell = cdc.GetNDoFsPerCell();

      for (unsigned int i = 0; i < n_dofs_per_cell; i++)
      {
        for (unsigned int j = 0; j < n_dofs_per_cell; j++)
        {
          local_entry_matrix(j, i) += tmp_mat(i, j);
        }
      }
    }

  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::FaceEquation(
        const FDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/ ,
        double /*scale*/, double /*scale_ico*/ )
    {
      throw DOpEException("Not Implemented", "PDEInterface::FaceEquation");
    }

  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::StrongFaceResidual(
        const FDC<DOFHANDLER, VECTOR, dealdim>&,
        const FDC<DOFHANDLER, VECTOR, dealdim>& /*fdc_weight*/, double&,
        double /*scale*/ )
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::StrongFaceResidual");
    }

  /********************************************/
  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::FaceEquation_U(
        const FDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/ ,
        double /*scale*/, double /*scale_ico*/ )
    {
      throw DOpEException("Not Implemented", "PDEInterface::FaceEquation_U");
    }

  /********************************************/
  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::StrongFaceResidual_U(
        const FDC<DOFHANDLER, VECTOR, dealdim>&,
        const FDC<DOFHANDLER, VECTOR, dealdim>& /*fdc_weight*/, double&,
        double /*scale*/ )
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::StrongFaceResidual_U");
    }

  /********************************************/
  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::FaceEquation_UT(
        const FDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/ ,
        double /*scale*/, double /*scale_ico*/ )
    {
      throw DOpEException("Not Implemented", "PDEInterface::FaceEquation_UT");
    }

  /********************************************/
  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::FaceEquation_UTT(
        const FDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/,
        double /*scale*/, double /*scale_ico*/ )
    {
      throw DOpEException("Not Implemented", "PDEInterface::FaceEquation_UTT");
    }

  /********************************************/
  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::FaceEquation_Q(
        const FDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/ ,
        double /*scale*/, double /*scale_ico*/ )
    {
      throw DOpEException("Not Implemented", "PDEInterface::FaceEquation_Q");
    }

  /********************************************/
  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::FaceEquation_QT(
        const FDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/ ,
        double /*scale*/, double /*scale_ico*/ )
    {
      throw DOpEException("Not Implemented", "PDEInterface::FaceEquation_QT");
    }

  /********************************************/
  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::FaceEquation_QTT(
        const FDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/ ,
        double /*scale*/, double /*scale_ico*/ )
    {
      throw DOpEException("Not Implemented", "PDEInterface::FaceEquation_QTT");
    }

  /********************************************/
  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::FaceEquation_UU(
        const FDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/ ,
        double /*scale*/, double /*scale_ico*/ )
    {
      throw DOpEException("Not Implemented", "PDEInterface::FaceEquation_UU");
    }

  /********************************************/
  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::FaceEquation_QU(
        const FDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/ ,
        double /*scale*/, double /*scale_ico*/ )
    {
      throw DOpEException("Not Implemented", "PDEInterface::FaceEquation_QU");
    }

  /********************************************/
  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::FaceEquation_UQ(
        const FDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/ ,
        double /*scale*/, double /*scale_ico*/ )
    {
      throw DOpEException("Not Implemented", "PDEInterface::FaceEquation_UQ");
    }

  /********************************************/
  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::FaceEquation_QQ(
        const FDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/ ,
        double /*scale*/, double /*scale_ico*/ )
    {
      throw DOpEException("Not Implemented", "PDEInterface::FaceEquation_QQ");
    }

  /********************************************/
  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::FaceRightHandSide(
        const FDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/ ,
        double /*scale*/ )
    {
      throw DOpEException("Not Implemented", "PDEInterface::FaceRightHandSide");
    }

  /********************************************/
  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::FaceMatrix(
        const FDC<DOFHANDLER, VECTOR, dealdim>&,
        FullMatrix<double> &/*local_entry_matrix*/, double /*scale*/, double /*scale_ico*/ )
    {
      throw DOpEException("Not Implemented", "PDEInterface::FaceMatrix");
    }

  /********************************************/
  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::FaceMatrix_T(
        const FDC<DOFHANDLER, VECTOR, dealdim>& fdc,
        FullMatrix<double> &local_entry_matrix, double scale, double scale_ico)
    {
      FullMatrix<double> tmp_mat = local_entry_matrix;
      tmp_mat = 0.;

      FaceMatrix(fdc, tmp_mat,scale,scale_ico);
      unsigned int n_dofs_per_cell = fdc.GetNDoFsPerCell();

      for (unsigned int i = 0; i < n_dofs_per_cell; i++)
      {
        for (unsigned int j = 0; j < n_dofs_per_cell; j++)
        {
          local_entry_matrix(j, i) += tmp_mat(i, j);
        }
      }
    }

  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::InterfaceEquation(
        const FDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/ ,
        double /*scale*/, double /*scale_ico*/ )
    {
      throw DOpEException("Not Implemented", "PDEInterface::InterfaceEquation");
    }
  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::InterfaceEquation_U(
        const FDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/ ,
        double /*scale*/, double /*scale_ico*/ )
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::InterfaceEquation_U");
    }

  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::InterfaceMatrix(
        const FDC<DOFHANDLER, VECTOR, dealdim>&,
        FullMatrix<double> &/*local_entry_matrix*/ , double /*scale*/, double /*scale_ico*/)
    {
      throw DOpEException("Not Implemented", "PDEInterface::InterfaceMatrix");
    }

  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::InterfaceMatrix_T(
        const FDC<DOFHANDLER, VECTOR, dealdim>&,
        FullMatrix<double> &/*local_entry_matrix*/ , double /*scale*/, double /*scale_ico*/)
    {
      throw DOpEException("Not Implemented", "PDEInterface::InterfaceMatrix_T");
    }

  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::BoundaryEquation(
        const FDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/ ,
        double /*scale*/, double /*scale_ico*/ )
    {
      throw DOpEException("Not Implemented", "PDEInterface::BoundaryEquation");
    }

  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::StrongBoundaryResidual(
        const FDC<DOFHANDLER, VECTOR, dealdim>&,
        const FDC<DOFHANDLER, VECTOR, dealdim>& /*Fdc_weight*/, double&,
        double /*scale*/ )
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::StrongBoundaryResidual");
    }

  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::BoundaryEquation_U(
        const FDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/ ,
        double /*scale*/, double /*scale_ico*/ )
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::BoundaryEquation_U");
    }

  /********************************************/
  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::StrongBoundaryResidual_U(
        const FDC<DOFHANDLER, VECTOR, dealdim>&,
        const FDC<DOFHANDLER, VECTOR, dealdim>& /*fdc_weight*/, double&,
        double /*scale*/ )
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::StrongBoundaryResidual_U");
    }

  /********************************************/
  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::BoundaryEquation_UT(
        const FDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/ ,
        double /*scale*/, double /*scale_ico*/ )
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::BoundaryEquation_UT");
    }

  /********************************************/
  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::BoundaryEquation_UTT(
        const FDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/ ,
        double /*scale*/, double /*scale_ico*/ )
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::BoundaryEquation_UTT");
    }

  /********************************************/
  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::BoundaryEquation_Q(
        const FDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/ ,
        double /*scale*/, double /*scale_ico*/ )
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::BoundaryEquation_Q");
    }

  /********************************************/
  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::BoundaryEquation_QT(
        const FDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/ ,
        double /*scale*/, double /*scale_ico*/ )
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::BoundaryEquation_QT");
    }

  /********************************************/
  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::BoundaryEquation_QTT(
        const FDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/ ,
        double /*scale*/, double /*scale_ico*/ )
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::BoundaryEquation_QTT");
    }

  /********************************************/
  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::BoundaryEquation_UU(
        const FDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/ ,
        double /*scale*/, double /*scale_ico*/ )
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::BoundaryEquation_UU");
    }

  /********************************************/
  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::BoundaryEquation_QU(
        const FDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/ ,
        double /*scale*/, double /*scale_ico*/ )
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::BoundaryEquation_QU");
    }

  /********************************************/
  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::BoundaryEquation_UQ(
        const FDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/ ,
        double /*scale*/, double /*scale_ico*/ )
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::BoundaryEquation_UQ");
    }

  /********************************************/
  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::BoundaryEquation_QQ(
        const FDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/ ,
        double /*scale*/, double /*scale_ico*/ )
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::BoundaryEquation_QQ");
    }

  /********************************************/
  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::BoundaryRightHandSide(
        const FDC<DOFHANDLER, VECTOR, dealdim>&,
        dealii::Vector<double> &/*local_cell_vector*/ ,
        double /*scale*/ )
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::BoundaryRightHandSide");
    }

  /********************************************/
  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::BoundaryMatrix(
        const FDC<DOFHANDLER, VECTOR, dealdim>&,
        FullMatrix<double> &/*local_entry_matrix*/, double /*scale*/, double /*scale_ico*/)
    {
      throw DOpEException("Not Implemented", "PDEInterface::BoundaryMatrix");
    }

  /********************************************/
  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::BoundaryMatrix_T(
        const FDC<DOFHANDLER, VECTOR, dealdim>& fdc,
        FullMatrix<double> &local_entry_matrix, double scale, double scale_ico)
    {
      FullMatrix<double> tmp_mat = local_entry_matrix;
      tmp_mat = 0.;

      BoundaryMatrix(fdc, tmp_mat,scale, scale_ico);
      unsigned int n_dofs_per_cell = fdc.GetNDoFsPerCell();

      for (unsigned int i = 0; i < n_dofs_per_cell; i++)
      {
        for (unsigned int j = 0; j < n_dofs_per_cell; j++)
        {
          local_entry_matrix(j, i) += tmp_mat(i, j);
        }
      }

    }

  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    UpdateFlags
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::GetUpdateFlags() const
    {
      return update_default; //no update
    }

  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    UpdateFlags
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::GetFaceUpdateFlags() const
    {
      return update_default; //no update
    }

  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    bool
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::HasFaces() const
    {
      return false;
    }
  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    bool
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::HasInterfaces() const
    {
      return false;
    }

  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    unsigned int
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::GetControlNBlocks() const
    {
      throw DOpEException("Not Implemented", "PDEInterface::GetControlNBlocks");
    }

  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    unsigned int
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::GetStateNBlocks() const
    {
      throw DOpEException("Not Implemented", "PDEInterface::GetStateNBlocks");
    }

  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    std::vector<unsigned int>&
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::GetControlBlockComponent()
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::GetControlBlockComponent");
    }

  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    const std::vector<unsigned int>&
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::GetControlBlockComponent() const
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::GetControlBlockComponent");
    }

  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    std::vector<unsigned int>&
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::GetStateBlockComponent()
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::GetStateBlockComponent");
    }

  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    const std::vector<unsigned int>&
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::GetStateBlockComponent() const
    {
      throw DOpEException("Not Implemented",
          "PDEInterface::GetStateBlockComponent");
    }

  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    unsigned int
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::GetStateNComponents() const
    {
      return this->GetStateBlockComponent().size();
    }

  /********************************************/

  template<
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class CDC,
      template<typename DOFHANDLER, typename VECTOR, int dealdim> class FDC,
      typename DOFHANDLER, typename VECTOR, int dopedim, int dealdim>
    void
    PDEInterface<CDC, FDC, DOFHANDLER, VECTOR, dopedim, dealdim>::SetProblemType(
        std::string type)
    {
      _problem_type = type;
    }

/********************************************/

} //Endof namespace
/********************************************/
/********************************************/
template class DOpE::PDEInterface<DOpE::CellDataContainer,
    DOpE::FaceDataContainer, dealii::DoFHandler<deal_II_dimension>,
    dealii::BlockVector<double>, dope_dimension, deal_II_dimension>;
template class DOpE::PDEInterface<DOpE::CellDataContainer,
    DOpE::FaceDataContainer, dealii::DoFHandler<deal_II_dimension>,
    dealii::Vector<double>, dope_dimension, deal_II_dimension>;
template class DOpE::PDEInterface<DOpE::Multimesh_CellDataContainer,
    DOpE::Multimesh_FaceDataContainer, dealii::DoFHandler<deal_II_dimension>,
    dealii::BlockVector<double>, dope_dimension, deal_II_dimension>;
template class DOpE::PDEInterface<DOpE::Multimesh_CellDataContainer,
    DOpE::Multimesh_FaceDataContainer, dealii::DoFHandler<deal_II_dimension>,
    dealii::Vector<double>, dope_dimension, deal_II_dimension>;
template class DOpE::PDEInterface<DOpE::CellDataContainer,
    DOpE::FaceDataContainer, dealii::hp::DoFHandler<deal_II_dimension>,
    dealii::BlockVector<double>, dope_dimension, deal_II_dimension>;
template class DOpE::PDEInterface<DOpE::CellDataContainer,
    DOpE::FaceDataContainer, dealii::hp::DoFHandler<deal_II_dimension>,
    dealii::Vector<double>, dope_dimension, deal_II_dimension>;

