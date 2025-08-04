#pragma once

#include "_ad-native.hpp"

namespace mfem
{
inline void ADFunction::Gradient(const Vector &x, const Vector &param,
                                 Vector &J) const
{
   MFEM_ASSERT(x.Size() == n_input,
               "ADFunction::Gradient: x.Size() must match n_input");
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
   MFEM_ASSERT(x.Size() == n_input,
               "ADFunction::Hessian: x.Size() must match n_input");
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

template <ADEvalInput mode>
inline int ADNonlinearFormIntegrator<mode>::InitInputShapes(
   const FiniteElement &el,
   ElementTransformation &Tr,
   DenseMatrix &shapes,
   Vector &value_shapes,
   DenseMatrix &grad_shapes)
{
   const int sdim = Tr.GetSpaceDim();
   const int shapedim = (hasFlag(mode, ADEvalInput::VALUE) ? 1 : 0)
                        + (hasFlag(mode, ADEvalInput::GRAD) ? sdim : 0);
   const int dof = el.GetDof();
   shapes.SetSize(dof, shapedim);

   if constexpr (hasFlag(mode, ADEvalInput::VALUE))
   {
      shapes.GetColumnReference(0, value_shapes);
   }
   if constexpr (hasFlag(mode, ADEvalInput::GRAD))
   {
      grad_shapes.UseExternalData(shapes.GetData()
                                  + dof*(hasFlag(mode, ADEvalInput::VALUE)),
                                  dof, sdim);
   }
   return shapedim;
}

template <ADEvalInput mode>
inline void ADNonlinearFormIntegrator<mode>::CalcInputShapes(
   const FiniteElement &el,
   ElementTransformation &Tr,
   const IntegrationPoint &ip,
   VectorCoefficient *parameter_cf,
   Vector &parameter,
   Vector &value_shapes,
   DenseMatrix &grad_shapes)
{
   if constexpr (hasFlag(mode, ADEvalInput::PARAM))
   {
      MFEM_ASSERT(parameter_cf == nullptr,
                  "ADNonlinearFormIntegrator: "
                  "param_cf should be set before AssembleElement...");
      parameter_cf->Eval(parameter, Tr, ip);
   }

   // Get value shape
   if constexpr (hasFlag(mode, ADEvalInput::VALUE)) { el.CalcPhysShape(Tr, value_shapes); }

   // Get gradient shape
   if constexpr (hasFlag(mode, ADEvalInput::GRAD)) { el.CalcPhysDShape(Tr, grad_shapes); }
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

   int shapedim = InitInputShapes(el, Tr, allshapes, shape, dshape);
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

      CalcInputShapes(el, Tr, ip, param_cf, param, shape, dshape);

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
   const int dof = el.GetDof();
   const int order = el.GetOrder();

   const int sdim = Tr.GetSpaceDim();
   const int dim = el.GetDim();

   real_t w;
   elvect.SetSize(dof*vdim);
   elvect = 0.0;

   x.SetSize(f.n_input);
   j.SetSize(f.n_input);

   int shapedim = InitInputShapes(el, Tr, allshapes, shape, dshape);

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

      CalcInputShapes(el, Tr, ip, param_cf, param, shape, dshape);

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
}

/// Assemble the local <H_f(x)(u), v>
template<ADEvalInput mode>
void ADNonlinearFormIntegrator<mode>::AssembleElementGrad(
   const FiniteElement &el,
   ElementTransformation &Tr,
   const Vector &elfun, DenseMatrix &elmat)
{
   const int dof = el.GetDof();
   const int order = el.GetOrder();

   const int sdim = Tr.GetSpaceDim();
   const int dim = el.GetDim();

   real_t w;
   elmat.SetSize(dof*vdim);
   elmat = 0.0;

   int shapedim = InitInputShapes(el, Tr, allshapes, shape, dshape);

   x.SetSize(f.n_input);
   H.SetSize(f.n_input);
   Hx.SetSize(dof, shapedim*vdim*vdim);


   if constexpr (hasFlag(mode, ADEvalInput::VECTOR))
   {
      elfun_matview.UseExternalData(const_cast<real_t*>(elfun.GetData()),
                                    dof, vdim);
      xmat.UseExternalData(x.GetData(), f.n_input / vdim, vdim);
      partelmat.SetSize(dof, dof);
      Hs.UseExternalData(H.GetData(), f.n_input / vdim, vdim*f.n_input);
   }

   const IntegrationRule * ir = GetIntegrationRule(el, el, Tr);
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Tr.SetIntPoint(&ip);
      w = ip.weight * Tr.Weight();
      CalcInputShapes(el, Tr, ip, param_cf, param, shape, dshape);

      // Convert dof to x = [[value, grad], [value, grad], ...]
      if constexpr (hasFlag(mode, ADEvalInput::VECTOR)) { MultAtB(allshapes, elfun_matview, xmat); }
      else { allshapes.MultTranspose(elfun, x); }

      f.Hessian(x, param, H);
      H *= w;

      if constexpr (hasFlag(mode, ADEvalInput::VECTOR))
      {
         Mult(allshapes, Hs, Hx);
         const int nel = shapedim*dof;
         for (int c=0; c<vdim; c++)
         {
            for (int r=0; r<=c; r++)
            {
               Hxsub.UseExternalData(Hx.GetData() + (c*vdim + r)*nel, dof, shapedim);
               MultABt(allshapes, Hxsub, partelmat);
               elmat.AddSubMatrix(c*dof, r*dof, partelmat);
               if (c != r)
               {
                  elmat.AddSubMatrix(r*dof, c*dof, partelmat);
               }
            }
         }
      }
      else
      {
         Mult(allshapes, H, Hx);
         AddMultABt(allshapes, Hx, elmat);
      }
   }
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
   MFEM_ABORT("ADNonlinearFormIntegrator::AssembleFaceVector: "
              "This method is not implemented.");
}


/// @brief Assemble the local action of the gradient of the
/// NonlinearFormIntegrator resulting from a face integral term.
template<ADEvalInput mode>
void ADNonlinearFormIntegrator<mode>::AssembleFaceGrad(const FiniteElement &el1,
      const FiniteElement &el2,
      FaceElementTransformations &Tr,
      const Vector &elfun, DenseMatrix &elmat)
{
   MFEM_ABORT("ADNonlinearFormIntegrator::AssembleFaceVector: "
              "This method is not implemented.");
}

} // namespace mfem
