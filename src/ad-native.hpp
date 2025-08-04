#pragma once

#include "_ad-native.hpp"

namespace mfem
{
template<std::size_t N>
struct __loop_index
{
   static const constexpr size_t value = N;
   constexpr operator std::size_t() const { return N; }
};
template <class F, std::size_t... Is>
void _constexpr_for(F func, std::index_sequence<Is...>)
{
   (func(__loop_index<Is> {}), ...);
}

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

template <bool is_param_cf, ADEval mode>
inline int ADNonlinearFormIntegrator<is_param_cf, mode>::InitInputShapes(
   const FiniteElement &el,
   ElementTransformation &Tr,
   DenseMatrix &shapes,
   Vector &value_shapes,
   DenseMatrix &grad_shapes)
{
   const int sdim = Tr.GetSpaceDim();
   const int shapedim = (hasFlag(mode, ADEval::VALUE) ? 1 : 0)
                        + (hasFlag(mode, ADEval::GRAD) ? sdim : 0);
   const int dof = el.GetDof();
   shapes.SetSize(dof, shapedim);

   if constexpr (hasFlag(mode, ADEval::VALUE))
   {
      shapes.GetColumnReference(0, value_shapes);
   }
   if constexpr (hasFlag(mode, ADEval::GRAD))
   {
      grad_shapes.UseExternalData(shapes.GetData()
                                  + dof*(hasFlag(mode, ADEval::VALUE)),
                                  dof, sdim);
   }
   return shapedim;
}

template <bool is_param_cf, ADEval mode>
inline void ADNonlinearFormIntegrator<is_param_cf, mode>::CalcInputShapes(
   const FiniteElement &el,
   ElementTransformation &Tr,
   const IntegrationPoint &ip,
   std::shared_ptr<VectorCoefficient> &parameter_cf,
   Vector &parameter,
   Vector &value_shapes,
   DenseMatrix &grad_shapes)
{
   if constexpr (is_param_cf)
   {
      MFEM_ASSERT(parameter_cf.get() == nullptr,
                  "ADNonlinearFormIntegrator: "
                  "Parameter coefficient should be set before AssembleElement...");
      parameter_cf->Eval(parameter, Tr, ip);
   }

   // Get value shape
   if constexpr (hasFlag(mode, ADEval::VALUE)) { el.CalcPhysShape(Tr, value_shapes); }

   // Get gradient shape
   if constexpr (hasFlag(mode, ADEval::GRAD)) { el.CalcPhysDShape(Tr, grad_shapes); }
}

/// Perform the local action of the NonlinearFormIntegrator
template <bool is_param_cf, ADEval mode>
real_t ADNonlinearFormIntegrator<is_param_cf, mode>::GetElementEnergy(
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
   if constexpr (hasFlag(mode, ADEval::VECTOR))
   {
      elfun_matview.UseExternalData(const_cast<real_t*>(elfun.GetData()),
                                    dof, vdim);
      xmat.UseExternalData(x.GetData(), shapedim, vdim);
   }

   const IntegrationRule * ir = GetIntegrationRule(el, Tr);
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Tr.SetIntPoint(&ip);

      CalcInputShapes(el, Tr, ip, param_cf, param, shape, dshape);

      if constexpr (hasFlag(mode, ADEval::VECTOR))
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
template <bool is_param_cf, ADEval mode>
void ADNonlinearFormIntegrator<is_param_cf, mode>::AssembleElementVector(
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
   jac.SetSize(f.n_input);

   int shapedim = InitInputShapes(el, Tr, allshapes, shape, dshape);

   if constexpr (hasFlag(mode, ADEval::VECTOR))
   {
      elfun_matview.UseExternalData(const_cast<real_t*>(elfun.GetData()),
                                    dof, vdim);
      elvectmat.UseExternalData(elvect.GetData(), dof, vdim);
      xmat.UseExternalData(x.GetData(), shapedim, vdim);
      jacMat.UseExternalData(jac.GetData(), shapedim, vdim);
   }

   const IntegrationRule * ir = GetIntegrationRule(el, Tr);
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Tr.SetIntPoint(&ip);
      w = ip.weight * Tr.Weight();

      CalcInputShapes(el, Tr, ip, param_cf, param, shape, dshape);

      // Convert dof to x = [[value, grad], [value, grad], ...]
      if constexpr (hasFlag(mode, ADEval::VECTOR)) { MultAtB(allshapes, elfun_matview, xmat); }
      else { allshapes.MultTranspose(elfun, x); }

      f.Gradient(x, param, jac);
      jac *= w;

      if constexpr (hasFlag(mode, ADEval::VECTOR))
      {
         AddMult(allshapes, jacMat, elvectmat);
      }
      else
      {
         allshapes.AddMult(jac, elvect);
      }
   }
}

/// Assemble the local <H_f(x)(u), v>
template <bool is_param_cf, ADEval mode>
void ADNonlinearFormIntegrator<is_param_cf, mode>::AssembleElementGrad(
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
   MFEM_ASSERT(shapedim*vdim == f.n_input,
               "ADNonlinearFormIntegrator: "
               "shapedim*vdim must match n_input");

   x.SetSize(f.n_input);
   H.SetSize(f.n_input);
   Hx.SetSize(dof, shapedim*vdim*vdim);


   if constexpr (hasFlag(mode, ADEval::VECTOR))
   {
      elfun_matview.UseExternalData(const_cast<real_t*>(elfun.GetData()),
                                    dof, vdim);
      xmat.UseExternalData(x.GetData(), shapedim, vdim);
      partelmat.SetSize(dof, dof);
      Hs.UseExternalData(H.GetData(), shapedim, vdim*shapedim*vdim);
   }

   const IntegrationRule * ir = GetIntegrationRule(el, Tr);
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Tr.SetIntPoint(&ip);
      w = ip.weight * Tr.Weight();
      CalcInputShapes(el, Tr, ip, param_cf, param, shape, dshape);

      // Convert dof to x = [[value, grad], [value, grad], ...]
      if constexpr (hasFlag(mode, ADEval::VECTOR)) { MultAtB(allshapes, elfun_matview, xmat); }
      else { allshapes.MultTranspose(elfun, x); }

      f.Hessian(x, param, H);
      H *= w;

      if constexpr (hasFlag(mode, ADEval::VECTOR))
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
template <bool is_param_cf, ADEval mode>
void ADNonlinearFormIntegrator<is_param_cf, mode>::AssembleFaceVector(
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
template <bool is_param_cf, ADEval mode>
void ADNonlinearFormIntegrator<is_param_cf, mode>::AssembleFaceGrad(
   const FiniteElement &el1,
   const FiniteElement &el2,
   FaceElementTransformations &Tr,
   const Vector &elfun, DenseMatrix &elmat)
{
   MFEM_ABORT("ADNonlinearFormIntegrator::AssembleFaceVector: "
              "This method is not implemented.");
}

template <bool is_param_cf, ADEval... modes>
inline std::array<int, sizeof...(modes)>
                              ADBlockNonlinearFormIntegrator<is_param_cf, modes...>::InitInputShapes(
                                 const Array<const FiniteElement *>& el,
                                 ElementTransformation &Tr,
                                 std::vector<DenseMatrix> &shapes,
                                 std::vector<Vector> &value_shapes,
                                 std::vector<DenseMatrix> &grad_shapes)
{
   MFEM_ASSERT(el.Size() == numSpaces,
               "ADBlockNonlinearFormIntegrator: "
               "el.Size() must match numSpaces");
   const int sdim = Tr.GetSpaceDim();
   std::array<int, sizeof...(modes)> shapedims{};

   _constexpr_for([&](auto i)
   {
      shapedims[i] = (hasFlag(modes_arr[i], ADEval::VALUE) ? 1 : 0)
                     + (hasFlag(modes_arr[i], ADEval::GRAD) ? sdim : 0);
      const int dof = el[i]->GetDof();
      shapes[i].SetSize(dof, shapedims[i]);

      if constexpr (hasFlag(modes_arr[i], ADEval::VALUE))
      {
         shapes[i].GetColumnReference(0, value_shapes[i]);
      }
      if constexpr (hasFlag(modes_arr[i], ADEval::GRAD))
      {
         grad_shapes[i].UseExternalData(shapes[i].GetData()
                                        + dof*(hasFlag(modes_arr[i], ADEval::VALUE)),
                                        dof, sdim);
      }
   }, std::make_index_sequence<sizeof...(modes)> {});
   return shapedims;
}
template <bool is_param_cf, ADEval... modes>
inline void
ADBlockNonlinearFormIntegrator<is_param_cf, modes...>::CalcInputShapes(
   const Array<const FiniteElement *>& el,
   ElementTransformation &Tr,
   const IntegrationPoint &ip,
   std::shared_ptr<VectorCoefficient> &parameter_cf,
   Vector &parameter,
   std::vector<Vector> &value_shapes,
   std::vector<DenseMatrix> &grad_shapes)
{
   if constexpr (is_param_cf)
   {
      MFEM_ASSERT(parameter_cf.get() == nullptr,
                  "ADNonlinearFormIntegrator: "
                  "Parameter coefficient should be set before AssembleElement...");
      parameter_cf->Eval(parameter, Tr, ip);
   }
   _constexpr_for([&](auto i)
   {
      // Get value shape
      if constexpr (hasFlag(modes_arr[i], ADEval::VALUE))
      { el[i]->CalcPhysShape(Tr, value_shapes[i]); }

      // Get gradient shape
      if constexpr (hasFlag(modes_arr[i], ADEval::GRAD))
      { el[i]->CalcPhysDShape(Tr, grad_shapes[i]); }
   }, std::make_index_sequence<sizeof...(modes)> {});
}

/// Compute the local energy
template <bool is_param_cf, ADEval... modes>
real_t ADBlockNonlinearFormIntegrator<is_param_cf, modes...>::GetElementEnergy(
   const Array<const FiniteElement *> &el,
   ElementTransformation &Tr,
   const Array<const Vector*> &elfun)
{
   MFEM_ASSERT(el.Size() == numSpaces,
               "ADBlockNonlinearFormIntegrator: "
               "el.Size() must match numSpaces");
   std::array<int, numSpaces> dof{};
   std::array<int, numSpaces> order{};
   for (int i=0; i<numSpaces; i++)
   {
      dof[i] = el[i]->GetDof();
      order[i] = el[i]->GetOrder();
   }

   const int sdim = Tr.GetSpaceDim();
   const int dim = el[0]->GetDim();

   real_t energy = 0.0;

   std::array<int, numSpaces> shapedim(InitInputShapes(el, Tr, allshapes, shape,
                                       dshape));
   x.SetSize(f.n_input);
   int x_idx = 0;
   _constexpr_for([&](auto vi)
   {
      xvar[vi].MakeRef(x, x_idx, shapedim[vi]*vdim[vi]);
      x_idx += shapedim[vi]*vdim[vi];
      if constexpr (hasFlag(modes_arr[vi], ADEval::VECTOR))
      {
         elfun_matview[vi].UseExternalData(const_cast<real_t*>(elfun[vi]->GetData()),
                                           dof[vi], vdim[vi]);
         xmat[vi].UseExternalData(xvar[vi].GetData(), shapedim[vi], vdim[vi]);
      }
   }, std::make_index_sequence<sizeof...(modes)> {});

   const IntegrationRule * ir = GetIntegrationRule(el, Tr);
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Tr.SetIntPoint(&ip);

      CalcInputShapes(el, Tr, ip, param_cf, param, shape, dshape);

      _constexpr_for([&](auto vi)
      {
         if constexpr (hasFlag(modes_arr[vi], ADEval::VECTOR))
         {
            MultAtB(allshapes[vi], elfun_matview[vi], xmat[vi]);
         }
         else
         {
            allshapes[vi].MultTranspose(*elfun[vi], xvar[vi]);
         }
      }, std::make_index_sequence<sizeof...(modes)> {});
      energy += f(x, param)*Tr.Weight()*ip.weight;
   }
   return energy;
}

/// Perform the local action of the NonlinearFormIntegrator
template <bool is_param_cf, ADEval... modes>
void ADBlockNonlinearFormIntegrator<is_param_cf, modes...>::AssembleElementVector(
   const Array<const FiniteElement *>&el,
   ElementTransformation &Tr,
   const Array<const Vector *>&elfun,
   const Array<Vector *>&elvect)
{
   MFEM_ASSERT(el.Size() == numSpaces,
               "ADBlockNonlinearFormIntegrator: "
               "el.Size() must match numSpaces");
   std::array<int, numSpaces> dof{};
   std::array<int, numSpaces> order{};
   for (int i=0; i<numSpaces; i++)
   {
      dof[i] = el[i]->GetDof();
      order[i] = el[i]->GetOrder();
   }

   const int sdim = Tr.GetSpaceDim();
   const int dim = el[0]->GetDim();
   for (int i=0; i<numSpaces; i++)
   {
      elvect[i]->SetSize(dof[i]*vdim[i]);
      *elvect[i] = 0.0;
   }

   std::array<int, numSpaces> shapedim(InitInputShapes(el, Tr, allshapes, shape,
                                       dshape));
   Array<int> x_idx(numSpaces+1);
   x_idx[0] = 0;
   for (int i=0; i<numSpaces; i++)
   {
      x_idx[i+1] = shapedim[i]*vdim[i];
   }
   x_idx.PartialSum();
   x.SetSize(f.n_input);
   jac.SetSize(f.n_input);
   _constexpr_for([&](auto vi)
   {
      xvar[vi].MakeRef(x, x_idx[vi], shapedim[vi]*vdim[vi]);
      jacVar[vi].MakeRef(jac, x_idx[vi], shapedim[vi]*vdim[vi]);
      if constexpr (hasFlag(modes_arr[vi], ADEval::VECTOR))
      {
         elfun_matview[vi].UseExternalData(const_cast<real_t*>(elfun[vi]->GetData()),
                                           dof[vi], vdim[vi]);
         xmat[vi].UseExternalData(xvar[vi].GetData(), shapedim[vi], vdim[vi]);
         jacVarMat[vi].UseExternalData(jacVar[vi].GetData(), shapedim[vi], vdim[vi]);
      }
   }, std::make_index_sequence<sizeof...(modes)> {});

   const IntegrationRule * ir = GetIntegrationRule(el, Tr);
   real_t w;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Tr.SetIntPoint(&ip);
      w = Tr.Weight()*ip.weight;

      CalcInputShapes(el, Tr, ip, param_cf, param, shape, dshape);

      _constexpr_for([&](auto vi)
      {
         if constexpr (hasFlag(modes_arr[vi], ADEval::VECTOR))
         {
            MultAtB(allshapes[vi], elfun_matview[vi], xmat[vi]);
         }
         else
         {
            allshapes[vi].MultTranspose(*elfun[vi], xvar[vi]);
         }
      }, std::make_index_sequence<sizeof...(modes)> {});
      f.Gradient(x, param, jac);
      jac *= w;

      _constexpr_for([&](auto vi)
      {
         if constexpr (hasFlag(modes_arr[vi], ADEval::VECTOR))
         {
            AddMult(allshapes[vi], jacVarMat[vi], elvectmat[vi]);
         }
         else
         {
            allshapes[vi].AddMult(jacVar[vi], *elvect[vi]);
         }
      }, std::make_index_sequence<sizeof...(modes)> {});
   }
}

/// Perform the local action of the NonlinearFormIntegrator
template <bool is_param_cf, ADEval... modes>
void ADBlockNonlinearFormIntegrator<is_param_cf, modes...>::AssembleElementGrad(
   const Array<const FiniteElement *>&el,
   ElementTransformation &Tr,
   const Array<const Vector *>&elfun,
   const Array2D<DenseMatrix *>&elmat)
{
   MFEM_ABORT("ADBlockNonlinearFormIntegrator::AssembleElementGrad: "
              "Not yet implemented");
   // MFEM_ASSERT(el.Size() == numSpaces,
   //             "ADBlockNonlinearFormIntegrator: "
   //             "el.Size() must match numSpaces");
   // Array<int> dof(numSpaces);
   // Array<int> order(numSpaces);
   // for (int i=0; i<numSpaces; i++)
   // {
   //    dof[i] = el[i]->GetDof();
   //    order[i] = el[i]->GetOrder();
   // }
   //
   // const int sdim = Tr.GetSpaceDim();
   // const int dim = el[0]->GetDim();
   // for (int j=0; j<numSpaces; j++)
   // {
   //    for (int i=0; i<numSpaces; i++)
   //    {
   //       elmat(i,j)->SetSize(dof[i]*vdim[i], dof[j]*vdim[j]);
   //       *elmat(i,j) = 0.0;
   //    }
   // }
   //
   // std::array<int, numSpaces> shapedim(InitInputShapes(el, Tr, allshapes, shape,
   //                                     dshape));
   // Array<int> x_idx(numSpaces+1);
   // x_idx[0] = 0;
   // for (int i=0; i<numSpaces; i++)
   // {
   //    x_idx[i+1] = shapedim[i]*vdim[i];
   // }
   // x_idx.PartialSum();
   // x.SetSize(f.n_input);
   // H.SetSize(f.n_input);
   //
   // _constexpr_for([&](auto vi)
   //                {
   // Hx.SetSize(dof, shapedim*vdim*vdim);
   // if constexpr (hasFlag(mode, ADEval::VECTOR))
   // {
   //    elfun_matview.UseExternalData(const_cast<real_t*>(elfun.GetData()),
   //                                  dof, vdim);
   //    xmat.UseExternalData(x.GetData(), shapedim, vdim);
   //    partelmat.SetSize(dof, dof);
   //    Hs.UseExternalData(H.GetData(), shapedim, vdim*shapedim*vdim);
   // }
   //                }, std::make_index_sequence<sizeof...(modes)> {});
   //
   // const IntegrationRule * ir = GetIntegrationRule(el, Tr);
   // for (int i = 0; i < ir->GetNPoints(); i++)
   // {
   //    const IntegrationPoint &ip = ir->IntPoint(i);
   //    Tr.SetIntPoint(&ip);
   //    w = ip.weight * Tr.Weight();
   //    CalcInputShapes(el, Tr, ip, param_cf, param, shape, dshape);
   //
   //    // Convert dof to x = [[value, grad], [value, grad], ...]
   //    if constexpr (hasFlag(mode, ADEval::VECTOR)) { MultAtB(allshapes, elfun_matview, xmat); }
   //    else { allshapes.MultTranspose(elfun, x); }
   //
   //    f.Hessian(x, param, H);
   //    H *= w;
   //
   //    if constexpr (hasFlag(mode, ADEval::VECTOR))
   //    {
   //       Mult(allshapes, Hs, Hx);
   //       const int nel = shapedim*dof;
   //       for (int c=0; c<vdim; c++)
   //       {
   //          for (int r=0; r<=c; r++)
   //          {
   //             Hxsub.UseExternalData(Hx.GetData() + (c*vdim + r)*nel, dof, shapedim);
   //             MultABt(allshapes, Hxsub, partelmat);
   //             elmat.AddSubMatrix(c*dof, r*dof, partelmat);
   //             if (c != r)
   //             {
   //                elmat.AddSubMatrix(r*dof, c*dof, partelmat);
   //             }
   //          }
   //       }
   //    }
   //    else
   //    {
   //       Mult(allshapes, H, Hx);
   //       AddMultABt(allshapes, Hx, elmat);
   //    }
   // }
}

/// @brief Perform the local action of the NonlinearFormIntegrator resulting
/// from a face integral term.
template <bool is_param_cf, ADEval... modes>
void ADBlockNonlinearFormIntegrator<is_param_cf, modes...>::AssembleFaceVector(
   const Array<const FiniteElement *>&el1,
   const Array<const FiniteElement *>&el2,
   FaceElementTransformations &Tr,
   const Array<const Vector *>&elfun,
   const Array<Vector *>&elvect)
{
   MFEM_ABORT("ADBlockNonlinearFormIntegrator::AssembleFaceVector: "
              "This method is not implemented.");
}


/// @brief Assemble the local action of the gradient of the
/// NonlinearFormIntegrator resulting from a face integral term.
template <bool is_param_cf, ADEval... modes>
void ADBlockNonlinearFormIntegrator<is_param_cf, modes...>::AssembleFaceGrad(
   const Array<const FiniteElement *>&el1,
   const Array<const FiniteElement *>&el2,
   FaceElementTransformations &Tr,
   const Array<const Vector *>&elfun,
   const Array2D<DenseMatrix *>&elmat)
{
   MFEM_ABORT("ADBlockNonlinearFormIntegrator::AssembleFaceGrad: "
              "This method is not implemented.");
}


} // namespace mfem
