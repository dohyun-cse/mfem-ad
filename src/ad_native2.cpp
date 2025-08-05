#include "ad_native2.hpp"

namespace mfem
{

void ADFunction2::Gradient(const Vector &x, ElementTransformation &Tr,
                           const IntegrationPoint &ip,
                           Vector &J) const
{
   MFEM_ASSERT(x.Size() == n_input,
               "ADFunction2::Gradient: x.Size() must match n_input");
   MFEM_ASSERT(param.Size() == n_param,
               "ADFunction2::Gradient: param.Size() must match n_param");
   J.SetSize(x.Size());
   ADVector x_ad(x);
   ProcessParameters(Tr, ip);
   for (int i=0; i < x.Size(); i++)
   {
      x_ad[i].gradient = 1.0;
      ADReal_t result = (*this)(x_ad);
      J[i] = result.gradient;
      x_ad[i].gradient = 0.0;
   }
}

void ADFunction2::Hessian(const Vector &x, ElementTransformation &Tr,
                          const IntegrationPoint &ip,
                          DenseMatrix &H) const
{
   MFEM_ASSERT(x.Size() == n_input,
               "ADFunction2::Hessian: x.Size() must match n_input");
   MFEM_ASSERT(param.Size() == n_param,
               "ADFunction2::Hessian: param.Size() must match n_param");
   H.SetSize(x.Size(), x.Size());
   AD2Vector x_ad(x);
   ProcessParameters(Tr, ip);
   for (int i=0; i<x.Size(); i++) // Loop for the first derivative
   {
      x_ad[i].value.gradient = 1.0;
      for (int j=0; j<=i; j++)
      {
         x_ad[j].gradient = ADReal_t{1.0, 0.0};
         AD2Real_t result = (*this)(x_ad);
         H(j, i) = result.gradient.gradient;
         H(i, j) = result.gradient.gradient;
         x_ad[j].gradient = ADReal_t{0.0, 0.0}; // Reset gradient for next iteration
      }
      x_ad[i].value.gradient = 0.0;
   }
}
SumADFunction2 ADFunction2::operator+(const ADFunction2& g) const
{ return SumADFunction2(*this, g); }
ShiftedADFunction2 ADFunction2::operator+(real_t a) const
{ return ShiftedADFunction2(*this, a); }
ShiftedADFunction2 ADFunction2::operator-(real_t a) const
{ return ShiftedADFunction2(*this, -a); }

SumADFunction2 ADFunction2::Add(const ADFunction2&g, real_t a) const
{
   if (a == 1.0) { return (*this) + g; }
   else { return SumADFunction2(*this, g, 1.0, a); }
}
ProductADFunction2 ADFunction2::operator*(const ADFunction2& g) const
{ return ProductADFunction2(*this, g); }
ScaledADFunction2 ADFunction2::operator*(real_t a) const
{ return ScaledADFunction2(*this, a); }

real_t ADPGEnergy::Eval(const Vector &x) const
{
   // variables
   const Vector x1(x.GetData(), f.n_input);
   const Vector latent(x.GetData() + f.n_input, primal_size);

   // evaluate mixed value
   real_t cross_entropy = 0.0;
   for (int i=0; i<dual_entropy.n_input; i++)
   {
      cross_entropy += x1[primal_begin + i]*(latent[i] - latent_k[i]);
   }
   f.Eval(x1);
   return f.Eval(x1) + (1.0 / alpha)*(cross_entropy - dual_entropy.Eval(latent));
}

// default Jacobian evaluator
ADReal_t ADPGEnergy::operator()(const ADVector &x) const
{
   MFEM_ASSERT(x.Size() == n_input,
               "ADFunction2::operator(): x.Size() must match n_input");
   MFEM_ASSERT(param.Size() == n_param,
               "ADFunction2::operator(): param.Size() must match n_param");

   // variables
   ADVector x1, latent;
   x1.SetDataAndSize(x.GetData(), f.n_input);
   latent.SetDataAndSize(x.GetData() + f.n_input,
                         primal_size);

   // evaluate mixed value
   ADReal_t cross_entropy{};
   for (int i=0; i<primal_size; i++)
   {
      cross_entropy += x1[primal_begin+i]*(latent[i] - latent_k[i]);
   }
   return f(x1) + (1.0 / alpha)*(cross_entropy - dual_entropy(latent));
}

// default Hessian evaluator
AD2Real_t ADPGEnergy::operator()(const AD2Vector &x) const
{
   MFEM_ASSERT(x.Size() == n_input,
               "ADFunction2::operator(): x.Size() must match n_input");
   MFEM_ASSERT(param.Size() == n_param,
               "ADFunction2::operator(): param.Size() must match n_param");

   // variables
   AD2Vector x1, latent;
   x1.SetDataAndSize(x.GetData(), f.n_input);
   latent.SetDataAndSize(x.GetData() + f.n_input,
                         primal_size);

   // evaluate mixed value
   AD2Real_t cross_entropy{};
   for (int i=0; i<primal_size; i++)
   {
      cross_entropy += x1[primal_begin+i]*(latent[i] - latent_k[i]);
   }
   return f(x1) + (1.0 / alpha)*(cross_entropy - dual_entropy(latent));
}
} // namespace mfem
