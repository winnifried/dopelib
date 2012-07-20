/*
 * problemcontainer_internal.h
 *
 *  Created on: Jul 19, 2012
 *      Author: cgoll
 */

#ifndef PROBLEMCONTAINER_INTERNAL_H_
#define PROBLEMCONTAINER_INTERNAL_H_

namespace DOpE
{
  /**
   * This class houses some functions and variables common in
   * pdeproblemcontainer.h and optproblemcontainer.h. Both classes
   * are derived from this one. With this, we prevent code duplicity.
   *
   */
  //FIXME There is the possibility to put more into this class.

  template<class PDE>
    class ProblemContainerInternal
    {
      public:
        ProblemContainerInternal(PDE& pde)
            : _pde(pde)
        {

        }
        /**
         * Computes the contribution of the cell to overall error
         * in a previously specified functional. For example, this
         * could be a residual with appropriate weights.
         *
         * @template CDC                Class of the celldatacontainer in use,
         *                              distinguishes between hp- and classical case.
         * @template FDC                Class of the facedatacontainer in use,
         *                              distinguishes between hp- and classical case.
         *
         * @param cdc                   A DataContainer holding all the needed information
         *                              for the computation of the residuum on the cell.
         * @param dwrc                  A DWRDataContainer containing all the information
         *                              needed to evaluate the error on the cell (form of the residual,
         *                              the weights, etc.).
         * @param cell_contrib          Vector in which we write the contribution of the cell to the overall
         *                              error. 1st component: primal_part, 2nd component: dual_part
         * @param scale                 A scaling factor which is -1 or 1 depending on the subroutine to compute.
         * @param scale_ico             A scaling factor for terms which will be treated fully implicit
         *                              in an instationary equation.
         */
        template<class CDC, class DWRC>
          void
          CellErrorContribution(const CDC& cdc, const DWRC& dwrc,
              std::vector<double>& cell_contrib, double scale,
              double /*scale_ico*/);

        /******************************************************/

        /**
         * Computes the contribution of the face to overall error
         * in a previously specified functional. This is the place
         * where for instance jump terms come into play.
         *
         * It has the same functionality
         * as CellErrorContribution, so we refer to its documentation.
         *
         */
        template<class FDC, class DWRC>
          void
          FaceErrorContribution(const FDC& fdc, const DWRC& dwrc,
              std::vector<double>& error_contrib, double scale = 1.);

        /******************************************************/

        /**
         * Computes the contribution of the boundary to overall error
         * in a previously specified functional.
         *
         * It has the same functionality
         * as CellErrorContribution, so we refer to its documentation.
         *
         */
        template<class FDC, class DWRC>
          void
          BoundaryErrorContribution(const FDC& dc, const DWRC& dwrc,
              std::vector<double>&, double scale = 1.);


        const PDE&
        GetPDE() const
        {
          return _pde;
        }

        std::string
        GetType() const
        {
          return _problem_type;
        }

        unsigned int
        GetTypeNum() const
        {
          return _problem_type_num;
        }

      protected:
        PDE&
        GetPDE()
        {
          return _pde;
        }
        void
        SetTypeInternal(std::string a)
        {
          _problem_type = a;
        }

        void
        SetTypeNumInternal(unsigned int i)
        {
          _problem_type_num = i;
        }

      private:
        std::string _problem_type, _algo_type;

        unsigned int _problem_type_num;
        PDE& _pde;

    };

  /******************************************************/

  template<typename PDE>
    template<class CDC, class DWRC>
      void
      ProblemContainerInternal<PDE>::CellErrorContribution(const CDC& cdc,
          const DWRC& dwrc, std::vector<double>& error, double scale,
          double scale_ico)
      {
        Assert(GetType() == "error_evaluation", ExcInternalError());

        if (dwrc.GetResidualEvaluation() == DOpEtypes::strong_residual)
        {

          if (dwrc.GetWeightComputation()
              == DOpEtypes::higher_order_interpolation)
          {
            CDC* cdc_w = ExtractCDC<CDC>(dwrc);
            switch (dwrc.GetEETerms())
            {
              case DOpEtypes::primal_only:
                GetPDE().StrongCellResidual(cdc, *cdc_w, error[0], scale,
                    scale_ico);
                break;
              case DOpEtypes::dual_only:
                GetPDE().StrongCellResidual_U(cdc, *cdc_w, error[1], scale);
                break;
              case DOpEtypes::mixed:
                GetPDE().StrongCellResidual(cdc, *cdc_w, error[0], scale,
                    scale_ico);
                GetPDE().StrongCellResidual_U(cdc, *cdc_w, error[1], scale);
                break;
              default:
                throw DOpEException("Not implemented for this EETerm.",
                    "PDEProblemContainer::CellErrorContribution");
                break;
            }

          }
          else if (dwrc.GetWeightComputation() == DOpEtypes::cell_diameter)
          {
            switch (dwrc.GetEETerms())
            {
              case DOpEtypes::primal_only:
                GetPDE().StrongCellResidual(cdc, cdc, error[0], scale,
                    scale_ico);
                break;
              case DOpEtypes::dual_only:
                GetPDE().StrongCellResidual_U(cdc, cdc, error[1], scale);
                break;
              case DOpEtypes::mixed:
                GetPDE().StrongCellResidual(cdc, cdc, error[0], scale,
                    scale_ico);
                GetPDE().StrongCellResidual_U(cdc, cdc, error[1], scale);
                break;
              default:
                throw DOpEException("Not implemented for this EETerm.",
                    "PDEProblemContainer::CellErrorContribution");
                break;
            }
          }
          else
          {
            throw DOpEException("Not implemented for this WeightComputation.",
                "PDEProblemContainer::CellErrorContribution");
          }
        }
        else
        {
          throw DOpEException("Not implemented for this ResidualEvaluation.",
              "PDEProblemContainer::CellErrorContribution");
        }
      }

  /******************************************************/

  template<typename PDE>
    template<class FDC, class DWRC>
      void
      ProblemContainerInternal<PDE>::FaceErrorContribution(const FDC& fdc,
          const DWRC& dwrc, std::vector<double>& error, double scale)
      {
        Assert(GetType() == "error_evaluation", ExcInternalError());

        if (dwrc.GetResidualEvaluation() == DOpEtypes::strong_residual)
        {

          if (dwrc.GetWeightComputation()
              == DOpEtypes::higher_order_interpolation)
          {
            FDC* fdc_w = ExtractFDC<FDC>(dwrc);
            switch (dwrc.GetEETerms())
            {
              case DOpEtypes::primal_only:
                GetPDE().StrongFaceResidual(fdc, *fdc_w, error[0], scale);
                break;
              case DOpEtypes::dual_only:
                GetPDE().StrongFaceResidual_U(fdc, *fdc_w, error[1], scale);
                break;
              case DOpEtypes::mixed:
                GetPDE().StrongFaceResidual(fdc, *fdc_w, error[0], scale);
                GetPDE().StrongFaceResidual_U(fdc, *fdc_w, error[1], scale);
                break;
              default:
                throw DOpEException("Not implemented for this EETerm.",
                    "PDEProblemContainer::FaceErrorContribution");
                break;
            }
          }
          else if (dwrc.GetWeightComputation() == DOpEtypes::cell_diameter)
          {
            switch (dwrc.GetEETerms())
            {
              case DOpEtypes::primal_only:
                GetPDE().StrongFaceResidual(fdc, fdc, error[0], scale);
                break;
              case DOpEtypes::dual_only:
                GetPDE().StrongFaceResidual_U(fdc, fdc, error[1], scale);
                break;
              case DOpEtypes::mixed:
                GetPDE().StrongFaceResidual(fdc, fdc, error[0], scale);
                GetPDE().StrongFaceResidual_U(fdc, fdc, error[1], scale);
                break;
              default:
                throw DOpEException("Not implemented for this EETerm.",
                    "PDEProblemContainer::FaceErrorContribution");
                break;
            }

          }
          else
          {
            throw DOpEException("Not implemented for this WeightComputation.",
                "PDEProblemContainer::FaceErrorContribution");
          }
        }
        else
        {
          throw DOpEException("Not implemented for this ResidualEvaluation.",
              "PDEProblemContainer::FaceErrorContribution");
        }
      }

  /******************************************************/

  template<typename PDE>
    template<class FDC, class DWRC>
      void
      ProblemContainerInternal<PDE>::BoundaryErrorContribution(const FDC& fdc,
          const DWRC& dwrc, std::vector<double>& error, double scale)
      {
        Assert(GetType() == "error_evaluation", ExcInternalError());

        if (dwrc.GetResidualEvaluation() == DOpEtypes::strong_residual)
        {
          if (dwrc.GetWeightComputation()
              == DOpEtypes::higher_order_interpolation)
          {
            FDC* fdc_w = ExtractFDC<FDC>(dwrc);
            switch (dwrc.GetEETerms())
            {
              case DOpEtypes::primal_only:
                GetPDE().StrongBoundaryResidual(fdc, *fdc_w, error[0], scale);
                break;
              case DOpEtypes::dual_only:
                GetPDE().StrongBoundaryResidual_U(fdc, *fdc_w, error[1], scale);
                break;
              case DOpEtypes::mixed:
                GetPDE().StrongBoundaryResidual(fdc, *fdc_w, error[0], scale);
                GetPDE().StrongBoundaryResidual_U(fdc, *fdc_w, error[1], scale);
                break;
              default:
                throw DOpEException("Not implemented for this EETerm.",
                    "PDEProblemContainer::BoundaryErrorContribution");
                break;
            }
          }
          else if (dwrc.GetWeightComputation() == DOpEtypes::cell_diameter)
          {
            GetPDE().StrongBoundaryResidual(fdc, fdc, error[0], scale);
          }
          else
          {
            throw DOpEException("Not implemented for this WeightComputation.",
                "PDEProblemContainer::BoundaryErrorContribution");

          }
        }
        else
        {
          throw DOpEException("Not implemented for this ResidualEvaluation.",
              "PDEProblemContainer::BoundaryErrorContribution");
        }
      }
}

#endif /* PROBLEMCONTAINER_INTERNAL_H_ */
