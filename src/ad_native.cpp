#include "ad_native.hpp"

namespace mfem
{

void ADFunction::Gradient(const Vector &x, ElementTransformation &Tr,
                           const IntegrationPoint &ip,
                           Vector &J) const
{
   MFEM_ASSERT(x.Size() == n_input,
               "ADFunction::Gradient: x.Size() must match n_input");
   MFEM_ASSERT(param.Size() == n_param,
               "ADFunction::Gradient: param.Size() must match n_param");
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

void ADFunction::Hessian(const Vector &x, ElementTransformation &Tr,
                          const IntegrationPoint &ip,
                          DenseMatrix &H) const
{
   MFEM_ASSERT(x.Size() == n_input,
               "ADFunction::Hessian: x.Size() must match n_input");
   MFEM_ASSERT(param.Size() == n_param,
               "ADFunction::Hessian: param.Size() must match n_param");
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
real_t ADPGEnergy::operator()(const Vector &x) const
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
   f(x1);
   return f(x1) + (1.0 / alpha)*(cross_entropy - dual_entropy(latent));
}

// default Jacobian evaluator
ADReal_t ADPGEnergy::operator()(const ADVector &x) const
{
   MFEM_ASSERT(x.Size() == n_input,
               "ADFunction::operator(): x.Size() must match n_input");
   MFEM_ASSERT(param.Size() == n_param,
               "ADFunction::operator(): param.Size() must match n_param");

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
               "ADFunction::operator(): x.Size() must match n_input");
   MFEM_ASSERT(param.Size() == n_param,
               "ADFunction::operator(): param.Size() must match n_param");

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
