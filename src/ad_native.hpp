#pragma once

#include "mfem.hpp"
#include "miniapps/autodiff/tadvector.hpp"
#include "miniapps/autodiff/taddensemat.hpp"

namespace mfem
{

// Use mfem-native autodiff types
// If other autodiff libraries are used,
// define ADReal_t, ADVector, ADMatrix, ... types accordingly.

// First order dual
typedef future::dual<real_t, real_t> ADReal_t;
typedef TAutoDiffVector<ADReal_t> ADVector;
typedef TAutoDiffDenseMatrix<ADReal_t> ADMatrix;

// second order dual (nested dual)
typedef future::dual<ADReal_t, ADReal_t> AD2Real_t;
typedef TAutoDiffVector<AD2Real_t> AD2Vector;
typedef TAutoDiffDenseMatrix<AD2Real_t> AD2Matrix;

// Interface for AutoDiff functions
// Use MAKE_AD_FUNCTION macro to create a derived structure.
// The gradient is evaluated with forward mode autodiff,
// and the Hessian is evaluated with forward over forward autodiff.
// The Hessian is assumed to be symmetric, to save computational cost.
// See, ADFunction::Hessian() for details.
struct ADFunction
{
   int n_input;
   int n_param;
   ADFunction(int n_input, int n_param=0)
      : n_input(n_input), n_param(n_param) { }
   // default evaluator
   virtual real_t operator()(const Vector &x, const Vector &param) const
   { MFEM_ABORT("Not implemented. Use MAKE_AD_FUNCTOR macro to create derived structure"); }

   // default Jacobian evaluator
   virtual ADReal_t operator()(const ADVector &x, const Vector &param) const
   { MFEM_ABORT("Not implemented. Use MAKE_AD_FUNCTOR macro to create derived structure"); }

   // default Hessian evaluator
   virtual AD2Real_t operator()(const AD2Vector &x, const Vector &param) const
   { MFEM_ABORT("Not implemented. Use MAKE_AD_FUNCTOR macro to create derived structure"); }

   // Evaluate the gradient, using forward mode autodiff
   virtual void Gradient(const Vector &x, const Vector &param, Vector &J) const;
   // Evaluate the Hessian, using forward over forward autodiff
   // The Hessian assumed to be symmetric.
   virtual void Hessian(const Vector &x, const Vector &param,
                        DenseMatrix &H) const;
};

// Construct augmented energy for proximal Galerkin
// psi =
// L(u, psi) = f(u) + (1/alpha)(u*(psi-psi_k) - E^*(psi))
// so that
// dL/du = df/du + (1/alpha)(psi-psi_k)
// dL/dpsi = (1/alpha)(u - dE^*(psi))
// When primal is not full vector, set primal_begin
// The parameter should be [org_param, entropy_param, alpha, psi_k]
struct ADPGEnergy : public ADFunction
{
   int primal_begin;
   int primal_size;
   ADFunction &f;
   ADFunction &dual_entropy;
   ADPGEnergy(ADFunction &f, ADFunction &dual_entropy,
              int primal_begin=0);

   real_t operator()(const Vector &x, const Vector &param) const override;

   // default Jacobian evaluator
   ADReal_t operator()(const ADVector &x, const Vector &param) const override;

   // default Hessian evaluator
   AD2Real_t operator()(const AD2Vector &x, const Vector &param) const override;
};

// Make Autodiff Function
// See, DiffusionEnergy, ..., for example of usage.
// @param name will be the name of the structure
// @param SCALAR is the name of templated scalar type
// @param VEC is the name of templated vector type
// @param MAT is the name of templated matrix type
// @param param is additional parameter name (will not be differentiated)
// @param body is the main function body. Use T() to create 0 T-typed value.
#define MAKE_AD_FUNCTION(name, SCALAR, VEC, MAT, var, param, body)                    \
struct name : public ADFunction                                                       \
{                                                                                     \
   name(int n, int n_param=0)                                                         \
      : ADFunction(n, n_param) { }                                                    \
                                                                                      \
   real_t operator()(const Vector &var, const Vector &param) const override           \
   {                                                                                  \
      MFEM_ASSERT(var.Size() == n_input,                                              \
                 "ADFunction::operator(): var.Size() must match n_input")             \
      MFEM_ASSERT(param.Size() == n_param,                                            \
                 "ADFunction::operator(): var.Size() must match n_input")             \
      using SCALAR = real_t;                                                          \
      using VEC = Vector;                                                             \
      using MAT = DenseMatrix;                                                        \
      body                                                                            \
   }                                                                                  \
                                                                                      \
   ADReal_t operator()(const ADVector &var, const Vector &param) const override       \
   {                                                                                  \
      MFEM_ASSERT(var.Size() == n_input,                                              \
                 "ADFunction::operator(): var.Size() must match n_input")             \
      MFEM_ASSERT(param.Size() == n_param,                                            \
                 "ADFunction::operator(): var.Size() must match n_input")             \
      using SCALAR = ADReal_t;                                                        \
      using VEC = ADVector;                                                           \
      using MAT = ADMatrix;                                                           \
      body                                                                            \
   }                                                                                  \
                                                                                      \
   AD2Real_t operator()(const AD2Vector &var, const Vector &param) const override     \
   {                                                                                  \
      MFEM_ASSERT(var.Size() == n_input,                                              \
                 "ADFunction::operator(): var.Size() must match n_input")             \
      MFEM_ASSERT(param.Size() == n_param,                                            \
                 "ADFunction::operator(): var.Size() must match n_input")             \
      using SCALAR = AD2Real_t;                                                       \
      using VEC = AD2Vector;                                                          \
      using MAT = AD2Matrix;                                                          \
      body                                                                            \
   }                                                                                  \
   using ADFunction::Gradient;                                                        \
   using ADFunction::Hessian;                                                         \
};

MAKE_AD_FUNCTION(MassEnergy, T, V, M, x, dummy,
{
   return 0.5*(x*x);
});

MAKE_AD_FUNCTION(DiffusionEnergy, T, V, M, gradu, dummy,
{
   return 0.5*(gradu*gradu);
});

MAKE_AD_FUNCTION(HeteroDiffusionEnergy, T, V, M, gradu, kappa,
{
   return (kappa[0]*0.5)*(gradu*gradu);
});

MAKE_AD_FUNCTION(AnisoDiffuionEnergy, T, V, M, gradu, kappa,
{
   T result = T();
   const int dim = gradu.Size();
   for (int i=0; i<dim; i++)
   {
      for (int j=0; j<dim; j++)
      {
         result += kappa[i*dim + j]*gradu[i]*gradu[j];
      }
   }
   return result;
});

MAKE_AD_FUNCTION(LinearElasticityEnergy, T, V, M, gradu, lame,
{
   const int dim = round(sqrt(gradu.Size()));
   const real_t lambda = lame(0);
   const real_t mu = lame(1);

   T divnorm = T();
   for (int i=0; i<dim; i++)
   {
      divnorm += gradu[i*dim + i];
   }
   divnorm = divnorm*divnorm;

   T h1_norm = T();
   for (int i=0; i<dim; i++)
   {
      for (int j=0; j<dim; j++)
      {
         T symm = 0.5*(gradu[i*dim + j] + gradu[j*dim + i]);
         h1_norm += symm*symm;
      }
   }

   return 0.5*lambda*divnorm + mu*h1_norm;
});

template <typename value_type, typename gradient_type, typename other_type>
MFEM_HOST_DEVICE
inline future::dual<value_type, gradient_type> max(
   future::dual<value_type, gradient_type> a,
   other_type b)
{
   if (a > b)
   {
      return a;
   }
   else if (a < b)
   {
      if constexpr (std::is_same<other_type, real_t>::value)
      {
         return future::dual<value_type, gradient_type> {b};
      }
      else
      {
         return b;
      }
   }
   else
   {
      // If values are equal, return the average (subgradient)
      return 0.5*(a + b);
   }
}
inline real_t max(const real_t a, const real_t b) { return std::max(a,b); }

template <typename value_type, typename gradient_type, typename other_type>
MFEM_HOST_DEVICE
inline future::dual<value_type, gradient_type> min(
   future::dual<value_type, gradient_type> a,
   other_type b)
{
   if (a < b)
   {
      return a;
   }
   else if (a > b)
   {
      if constexpr (std::is_same<other_type, real_t>::value)
      {
         return future::dual<value_type, gradient_type> {b};
      }
      else
      {
         return b;
      }
   }
   else
   {
      // If values are equal, return the average (subgradient)
      return 0.5*(a + b);
   }
}
MFEM_HOST_DEVICE
inline real_t min(const real_t a, const real_t b) { return std::min(a,b); }

// Dual entropy for (negative) Shannon entropy (xlogx - x) with half bound
// when bound[1] = 1, [lower, inf[
// when bound[1] = -1, ]-inf, upper]
//
// The resulting dual is (f(pm1*(x - shift)))^*
// = f^*(pm1*x^*) + shift*pm1*x^*
MAKE_AD_FUNCTION(ShannonEntropy, T, V, M, x, bound,
{
   MFEM_ASSERT(x.Size() == 1, "ShannonEntropy: input must have size 1");
   MFEM_ASSERT(bound.Size() == 2, "ShannonEntropy: bound must have size 2");
   MFEM_ASSERT(std::abs(bound[1]) == 1.0, "ShannonEntropy: bound[1] must be 1 or -1");

   real_t pm1 = bound[1];
   real_t shift = bound[0];
   return pm1*(exp(x[0]*pm1)) + shift*x[0];
});

// Dual entropy for (negative) Fermi-Dirac with [lower, upper] bounds
MAKE_AD_FUNCTION(FermiDiracEntropy, T, V, M, x, bound,
{
   MFEM_ASSERT(x.Size() == 1, "FermiDiracEntropy: input must have size 1");
   MFEM_ASSERT(bound.Size() == 2, "FermiDiracEntropy: bound must have size 2");
   MFEM_ASSERT(bound[1] - bound[0] > 0, "FermiDiracEntropy: upper must be greater than lower");

   const real_t scale = bound[1] - bound[0];
   const real_t shift = bound[0];
   T z = x[0]*scale;

   // Use a numerically stable implementation of log(1+exp(z))
   if (z > 0)
   {
      return z + log(1.0 + exp(-z)) + shift*x[0];
   }
   else
   {
      return log(1.0 + exp(z)) + shift*x[0];
   }
});

MAKE_AD_FUNCTION(HellingerEntropy, T, V, M, x, bound,
{
   MFEM_ASSERT(bound.Size() == 1, "HellingerEntropy: bound must have size 2");
   MFEM_ASSERT(bound[0] > 0, "HellingerEntropy: bound must be positive");

   const real_t scale = bound[0];
   T val_sqrd = (x*x)*(scale*scale);
   return sqrt(1+val_sqrd);
});

// Dual entropy for (negative) Simplex entropy with
// x_i >= 0 sum_i x_i = bound
// Also known as cateborical entropy or multinomial Shannon entropy
MAKE_AD_FUNCTION(SimplexEntropy, T, V, M, x, bound,
{
   MFEM_ASSERT(bound.Size() == 1, "SimplexEntropy: bound must have size 1");
   MFEM_ASSERT(bound[0] > 0, "SimplexEntropy: bound must be positive");

   const real_t scale = bound[0];
   T maxval = x[0];
   for (int i=1; i<x.Size(); i++)
   {
      maxval = max(maxval, x[i]);
   }

   T sum_exp = T();
   for (int i=0; i<x.Size(); i++)
   {
      sum_exp += exp(x[i]);
   }
   return scale*log(sum_exp);
});

}
