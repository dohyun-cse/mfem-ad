#pragma once

#include "ad-native-def.hpp"

namespace mfem
{
inline void ADFunction::Gradient(const Vector &x, const Vector &param,
                                 Vector &J) const
{
   MFEM_ASSERT(x.Size() == n_inputs,
               "ADFunction::Gradient: x.Size() must match n_inputs");
   MFEM_ASSERT(param.Size() == n_param,
               "ADFunction::Gradient: param.Size() must match n_param");
   J.SetSize(x.Size());
   ADVector x_ad(x);
   for (int i=0; i < x.Size(); i++)
   {
      x_ad[i].gradient = 1.0;
      ADReal_t result = (*this)(x_ad, param);
      J[i] = result.gradient;
      x_ad[i].gradient = 0.0;
   }
}

inline void ADFunction::Hessian(const Vector &x, const Vector &param,
                                DenseMatrix &H) const
{
   MFEM_ASSERT(x.Size() == n_inputs,
               "ADFunction::Hessian: x.Size() must match n_inputs");
   MFEM_ASSERT(param.Size() == n_param,
               "ADFunction::Hessian: param.Size() must match n_param");
   H.SetSize(x.Size(), x.Size());
   AD2Vector x_ad(x);
   for (int i=0; i<x.Size(); i++) // Loop for the first derivative
   {
      x_ad[i].value.gradient = 1.0;
      for (int j=0; j<=i; j++)
      {
         x_ad[j].gradient = ADReal_t{1.0, 0.0};
         AD2Real_t result = (*this)(x_ad, param);
         H(j, i) = result.gradient.gradient;
         H(i, j) = result.gradient.gradient;
         x_ad[j].gradient = ADReal_t{0.0, 0.0}; // Reset gradient for next iteration
      }
      x_ad[i].value.gradient = 0.0;
   }
}

/// Perform the local action of the NonlinearFormIntegrator
template<ADEvalInput mode>
real_t ADNonlinearFormIntegrator<mode>::GetElementEnergy(
   const FiniteElement &el,
   ElementTransformation &Tr,
   const Vector &elfun)
{
   const int dof = el.GetDof();
   const int order = el.GetOrder();

   const int sdim = Tr.GetSpaceDim();
   const int dim = el.GetDim();

   real_t energy = 0.0;

   int shapedim = int(mode & ADEvalInput::VALUE) + int(mode & ADEvalInput::GRAD) *
                  sdim;
   allshapes.SetSize(dof, shapedim);

   if constexpr (hasFlag(mode, ADEvalInput::VALUE))
   {
      allshapes.GetColumnReference(0, shape);
   }
   if constexpr (hasFlag(mode, ADEvalInput::GRAD))
   {
      dshape.UseExternalData(allshapes.GetData() + dof*int(mode & ADEvalInput::VALUE),
                             dof, sdim);
   }
   x.SetSize(f.n_input);
   if constexpr (hasFlag(mode, ADEvalInput::VECTOR))
   {
      elfun_matview.UseExternalData(const_cast<real_t*>(elfun.GetData()),
                                    dof, vdim);
      xmat.UseExternalData(x.GetData(), f.n_input / vdim, vdim);
   }

   const IntegrationRule * ir = GetIntegrationRule(el, el, Tr);
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Tr.SetIntPoint(&ip);

      if constexpr (hasFlag(mode, ADEvalInput::PARAM))
      {
         MFEM_ASSERT(param_cf == nullptr,
                     "ADNonlinearFormIntegrator: "
                     "param_cf should be set before AssembleElementVector");
         param_cf->Eval(param, Tr, ip);
      }

      if constexpr (hasFlag(mode, ADEvalInput::VALUE))
      {
         el.CalcPhysShape(Tr, shape);
      }

      if constexpr (hasFlag(mode, ADEvalInput::GRAD))
      {
         el.CalcPhysDShape(Tr, dshape);
         VectorDiffusionIntegrator intg;
      }

      if constexpr (hasFlag(mode, ADEvalInput::VECTOR))
      {
         MultAtB(allshapes, elfun_matview, xmat);
      }
      else
      {
         allshapes.MultTranspose(elfun, x);
      }
      energy += f(x, param)*Tr.Weight()*ip.weight;
   }
   return energy;
}

/// Compute the local <grad f, v>
template<ADEvalInput mode>
void ADNonlinearFormIntegrator<mode>::AssembleElementVector(
   const FiniteElement &el,
   ElementTransformation &Tr,
   const Vector &elfun, Vector &elvect)
{
   // out << "ADNonlinearFormIntegrator::AssembleElementVector: "
   //     << "mode = " << static_cast<int>(mode) << std::endl;
   const int dof = el.GetDof();
   const int order = el.GetOrder();

   const int sdim = Tr.GetSpaceDim();
   const int dim = el.GetDim();

   real_t w;
   elvect.SetSize(dof*vdim);
   elvect = 0.0;

   x.SetSize(f.n_input);
   j.SetSize(f.n_input);

   int shapedim = hasFlag(mode, ADEvalInput::VALUE) + hasFlag(mode,
                  ADEvalInput::GRAD) * sdim;
   allshapes.SetSize(dof, shapedim);

   if constexpr (hasFlag(mode, ADEvalInput::VALUE))
   {
      allshapes.GetColumnReference(0, shape);
   }
   if constexpr (hasFlag(mode, ADEvalInput::GRAD))
   {
      dshape.UseExternalData(allshapes.GetData() + dof*hasFlag(mode,
                             ADEvalInput::VALUE),
                             dof, sdim);
   }

   if constexpr (hasFlag(mode, ADEvalInput::VECTOR))
   {
      elfun_matview.UseExternalData(const_cast<real_t*>(elfun.GetData()),
                                    dof, vdim);
      elvectmat.UseExternalData(elvect.GetData(), dof, vdim);
      xmat.UseExternalData(x.GetData(), f.n_input / vdim, vdim);
      jmat.UseExternalData(j.GetData(), f.n_input / vdim, vdim);
   }

   const IntegrationRule * ir = GetIntegrationRule(el, el, Tr);
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Tr.SetIntPoint(&ip);
      w = ip.weight * Tr.Weight();

      if constexpr (hasFlag(mode, ADEvalInput::PARAM))
      {
         MFEM_ASSERT(param_cf == nullptr,
                     "ADNonlinearFormIntegrator: "
                     "param_cf should be set before AssembleElementVector");
         param_cf->Eval(param, Tr, ip);
      }

      // Get value shape
      if constexpr (hasFlag(mode, ADEvalInput::VALUE)) { el.CalcPhysShape(Tr, shape); }

      // Get gradient shape
      if constexpr (hasFlag(mode, ADEvalInput::GRAD)) { el.CalcPhysDShape(Tr, dshape); }

      // Convert dof to x = [[value, grad], [value, grad], ...]
      if constexpr (hasFlag(mode, ADEvalInput::VECTOR)) { MultAtB(allshapes, elfun_matview, xmat); }
      else { allshapes.MultTranspose(elfun, x); }

      f.Gradient(x, param, j);
      j *= w;

      if constexpr (hasFlag(mode, ADEvalInput::VECTOR))
      {
         AddMult(allshapes, jmat, elvectmat);
      }
      else
      {
         allshapes.AddMult(j, elvect);
      }
   }
   // out << "ADNonlinearFormIntegrator::AssembleElementVector: "
   //     << "mode = " << static_cast<int>(mode) << " done." << std::endl;
}

/// Assemble the local <H_f(x)(u), v>
template<ADEvalInput mode>
void ADNonlinearFormIntegrator<mode>::AssembleElementGrad(
   const FiniteElement &el,
   ElementTransformation &Tr,
   const Vector &elfun, DenseMatrix &elmat)
{
   // out << "ADNonlinearFormIntegrator::AssembleElementGrad: "
   //     << "mode = " << static_cast<int>(mode) << std::endl;
   const int dof = el.GetDof();
   const int order = el.GetOrder();

   const int sdim = Tr.GetSpaceDim();
   const int dim = el.GetDim();

   real_t w;
   elmat.SetSize(dof*vdim);
   elmat = 0.0;

   int shapedim = (hasFlag(mode, ADEvalInput::VALUE)) + (hasFlag(mode,
                  ADEvalInput::GRAD)) * sdim;
   allshapes.SetSize(dof, shapedim);

   x.SetSize(f.n_input);
   H.SetSize(f.n_input);
   Hx.SetSize(dof*vdim, shapedim);

   if constexpr (hasFlag(mode, ADEvalInput::VALUE))
   {
      allshapes.GetColumnReference(0, shape);
   }
   if constexpr (hasFlag(mode, ADEvalInput::GRAD))
   {
      dshape.UseExternalData(allshapes.GetData()
                             + dof*(hasFlag(mode, ADEvalInput::VALUE)),
                             dof, sdim);
   }

   if constexpr (hasFlag(mode, ADEvalInput::VECTOR))
   {
      elfun_matview.UseExternalData(const_cast<real_t*>(elfun.GetData()),
                                    dof, vdim);
      xmat.UseExternalData(x.GetData(), f.n_input / vdim, vdim);
      jmat.UseExternalData(j.GetData(), f.n_input / vdim, vdim);
   }

   const IntegrationRule * ir = GetIntegrationRule(el, el, Tr);
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Tr.SetIntPoint(&ip);
      w = ip.weight * Tr.Weight();

      if constexpr (hasFlag(mode, ADEvalInput::PARAM))
      {
         MFEM_ASSERT(param_cf == nullptr,
                     "ADNonlinearFormIntegrator: "
                     "param_cf should be set before AssembleElementVector");
         param_cf->Eval(param, Tr, ip);
      }

      // Get value shape
      if constexpr (hasFlag(mode, ADEvalInput::VALUE)) { el.CalcPhysShape(Tr, shape); }

      // Get gradient shape
      if constexpr (hasFlag(mode, ADEvalInput::GRAD)) { el.CalcPhysDShape(Tr, dshape); }

      // Convert dof to x = [[value, grad], [value, grad], ...]
      if constexpr (hasFlag(mode, ADEvalInput::VECTOR)) { MultAtB(allshapes, elfun_matview, xmat); }
      else { allshapes.MultTranspose(elfun, x); }

      f.Hessian(x, param, H);
      H *= w;

      if constexpr (hasFlag(mode, ADEvalInput::VECTOR))
      {
         AddMult(allshapes, H, elmat);
         MFEM_ABORT("NOT YET IMPLEMENTED");
      }
      else
      {
         Mult(allshapes, H, Hx);
         AddMultABt(allshapes, Hx, elmat);
      }
   }
   // out << "ADNonlinearFormIntegrator::AssembleElementGrad: "
   //     << "mode = " << static_cast<int>(mode) << " done." << std::endl;
}

/// @brief Perform the local action of the NonlinearFormIntegrator resulting
/// from a face integral term.
template<ADEvalInput mode>
void ADNonlinearFormIntegrator<mode>::AssembleFaceVector(
   const FiniteElement &el1,
   const FiniteElement &el2,
   FaceElementTransformations &Tr,
   const Vector &elfun, Vector &elvect)
{
}


/// @brief Assemble the local action of the gradient of the
/// NonlinearFormIntegrator resulting from a face integral term.
template<ADEvalInput mode>
void ADNonlinearFormIntegrator<mode>::AssembleFaceGrad(const FiniteElement &el1,
      const FiniteElement &el2,
      FaceElementTransformations &Tr,
      const Vector &elfun, DenseMatrix &elmat)
{}

} // namespace mfem
