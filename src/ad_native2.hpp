#pragma once

#include "mfem.hpp"
#include "miniapps/autodiff/tadvector.hpp"
#include "miniapps/autodiff/taddensemat.hpp"

namespace mfem
{
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

struct SumADFunction2;
struct ProductADFunction2;
struct ScaledADFunction2;
struct ShiftedADFunction2;
// Interface for AutoDiff functions
// Use MAKE_AD_FUNCTION2 macro to create a derived structure.
// The gradient is evaluated with forward mode autodiff,
// and the Hessian is evaluated with forward over forward autodiff.
// The Hessian is assumed to be symmetric, to save computational cost.
// See, ADFunction2::Hessian() for details.
struct ADFunction2
{

public:
   virtual void ProcessParameters(ElementTransformation &Tr,
                                  const IntegrationPoint &ip) const
   {
      // DO nothing by default
   }
   virtual real_t Eval(const Vector &x) const
   { MFEM_ABORT("Not implemented. Use AD_IMPL macro to implement all path"); }

   int n_input;
   ADFunction2(int n_input)
      : n_input(n_input) { }
   // default evaluator
   virtual real_t operator()(const Vector &x, ElementTransformation &Tr,
                             const IntegrationPoint &ip) const
   {
      ProcessParameters(Tr, ip);
      return Eval(x);
   }

   // default Jacobian evaluator
   virtual ADReal_t operator()(const ADVector &x) const
   { MFEM_ABORT("Not implemented. Use MAKE_AD_FUNCTOR macro to create derived structure"); }

   // default Hessian evaluator
   virtual AD2Real_t operator()(const AD2Vector &x) const
   { MFEM_ABORT("Not implemented. Use MAKE_AD_FUNCTOR macro to create derived structure"); }

   // Evaluate the gradient, using forward mode autodiff
   virtual void Gradient(const Vector &x, ElementTransformation &Tr,
                         const IntegrationPoint &ip, Vector &J) const;
   // Evaluate the Hessian, using forward over forward autodiff
   // The Hessian assumed to be symmetric.
   virtual void Hessian(const Vector &x, ElementTransformation &Tr,
                        const IntegrationPoint &ip,
                        DenseMatrix &H) const;
   SumADFunction2 operator+(const ADFunction2& g) const;
   ShiftedADFunction2 operator+(real_t a) const;
   ShiftedADFunction2 operator-(real_t a) const;
   SumADFunction2 Add(const ADFunction2&g, const real_t a) const;
   ProductADFunction2 operator*(const ADFunction2& g) const;
   ScaledADFunction2 operator*(real_t a) const;
};
// Macro to generate type-varying implementation for ADFunction2.
// See, DiffusionEnergy, ..., for example of usage.
// @param SCALAR is the name of templated scalar type
// @param VEC is the name of templated vector type
// @param MAT is the name of templated matrix type
// @param body is the main function body. Use T() to create T-typed 0.
#define AD_IMPL(SCALAR, VEC, MAT, var, body)                                           \
   using ADFunction2::operator();                                                      \
   real_t Eval(const Vector &var) const override                                       \
   {                                                                                   \
      MFEM_ASSERT(var.Size() == n_input,                                               \
                 "ADFunction2::operator(): var.Size() must match n_input")             \
      MFEM_ASSERT(param.Size() == n_param,                                             \
                 "ADFunction2::operator(): var.Size() must match n_input")             \
      using SCALAR = real_t;                                                           \
      using VEC = Vector;                                                              \
      using MAT = DenseMatrix;                                                         \
      body                                                                             \
   }                                                                                   \
                                                                                       \
   ADReal_t operator()(const ADVector &var) const override                             \
   {                                                                                   \
      MFEM_ASSERT(var.Size() == n_input,                                               \
                 "ADFunction2::operator(): var.Size() must match n_input")             \
      MFEM_ASSERT(param.Size() == n_param,                                             \
                 "ADFunction2::operator(): var.Size() must match n_input")             \
      using SCALAR = ADReal_t;                                                         \
      using VEC = ADVector;                                                            \
      using MAT = ADMatrix;                                                            \
      body                                                                             \
   }                                                                                   \
                                                                                       \
   AD2Real_t operator()(const AD2Vector &var) const override                           \
   {                                                                                   \
      MFEM_ASSERT(var.Size() == n_input,                                               \
                 "ADFunction2::operator(): var.Size() must match n_input")             \
      MFEM_ASSERT(param.Size() == n_param,                                             \
                 "ADFunction2::operator(): var.Size() must match n_input")             \
      using SCALAR = AD2Real_t;                                                        \
      using VEC = AD2Vector;                                                           \
      using MAT = AD2Matrix;                                                           \
      body                                                                             \
   }


struct ProductADFunction2 : public ADFunction2
{
   const ADFunction2 &f1;
   const ADFunction2 &f2;

   ProductADFunction2(const ADFunction2 &f1, const ADFunction2 &f2)
      : ADFunction2(f1.n_input)
      , f1(f1), f2(f2)
   {
      MFEM_VERIFY(f1.n_input == f2.n_input,
                  "ProductADFunction2: f1 and f2 must have the same n_input and n_param");
   }
   void ProcessParameters(ElementTransformation &Tr,
                          const IntegrationPoint &ip) const override
   {
      f1.ProcessParameters(Tr, ip);
      f2.ProcessParameters(Tr, ip);
   }

   real_t Eval(const Vector &x) const override
   { return f1.Eval(x)*f2.Eval(x); }
   real_t operator()(const Vector &x, ElementTransformation &Tr,
                     const IntegrationPoint &ip) const override
   { return f1(x, Tr, ip) * f2(x, Tr, ip); }
   ADReal_t operator()(const ADVector &x) const override
   { return f1(x) * f2(x); }
   AD2Real_t operator()(const AD2Vector &x) const override
   { return f1(x) * f2(x); }
};

struct ScaledADFunction2 : public ADFunction2
{
   const ADFunction2 &f1;
   real_t a;

   ScaledADFunction2(const ADFunction2 &f1, real_t a)
      : ADFunction2(f1.n_input)
      , f1(f1), a(a)
   { }
   void ProcessParameters(ElementTransformation &Tr,
                          const IntegrationPoint &ip) const override
   {
      f1.ProcessParameters(Tr, ip);
   }
   void SetScale(real_t a) { this->a = a; }
   // default evaluator
   real_t Eval(const Vector &x) const override
   { return f1.Eval(x)*a; }
   real_t operator()(const Vector &x, ElementTransformation &Tr,
                     const IntegrationPoint &ip) const override
   { return f1(x, Tr, ip) * a; }

   // default Jacobian evaluator
   ADReal_t operator()(const ADVector &x) const override
   { return f1(x) * a; }

   // default Hessian evaluator
   AD2Real_t operator()(const AD2Vector &x) const override
   { return f1(x) * a; }
};

struct ReferenceConstantADFunction2 : public ADFunction2
{
   real_t &a;

   ReferenceConstantADFunction2(real_t &a, int n_input)
      : ADFunction2(n_input)
      , a(a)
   { }
   real_t operator()(const Vector &x, ElementTransformation &Tr,
                     const IntegrationPoint &ip) const override
   { return a; }

   // default Jacobian evaluator
   real_t Eval(const Vector &x) const override
   { return a; }
   ADReal_t operator()(const ADVector &x) const override
   { return ADReal_t{a, 0.0}; }

   // default Hessian evaluator
   AD2Real_t operator()(const AD2Vector &x) const override
   { return AD2Real_t{a, 0.0}; }

   void Gradient(const Vector &x, ElementTransformation &Tr,
                 const IntegrationPoint &ip, Vector &J) const override
   {
      J.SetSize(x.Size());
      J = 0.0; // Gradient is zero for constant function
   }
   void Hessian(const Vector &x, ElementTransformation &Tr,
                const IntegrationPoint &ip,
                DenseMatrix &H) const override
   {
      H.SetSize(x.Size(), x.Size());
      H = 0.0; // Hessian is zero for constant function
   }
};

struct ShiftedADFunction2 : public ADFunction2
{
   const ADFunction2 &f1;
   real_t a;


   ShiftedADFunction2(const ADFunction2 &f1, real_t a)
      : ADFunction2(f1.n_input)
      , f1(f1), a(a)
   { }
   void ProcessParameters(ElementTransformation &Tr,
                          const IntegrationPoint &ip) const override
   {
      f1.ProcessParameters(Tr, ip);
   }
   void SetShift(real_t a) { this->a = a; }
   // default evaluator
   real_t Eval(const Vector &x) const override
   { return f1.Eval(x) + a; }
   real_t operator()(const Vector &x, ElementTransformation &Tr,
                     const IntegrationPoint &ip) const override
   { return f1(x, Tr, ip) + a; }

   // default Jacobian evaluator
   ADReal_t operator()(const ADVector &x) const override
   { return f1(x) + a; }

   // default Hessian evaluator
   AD2Real_t operator()(const AD2Vector &x) const override
   { return f1(x) + a; }
};

struct SumADFunction2 : public ADFunction2
{
   const ADFunction2 &f1;
   const ADFunction2 &f2;
   const real_t a, b;
   mutable Vector Jac;
   mutable DenseMatrix Hess;

   SumADFunction2(const ADFunction2 &f1, const ADFunction2 &f2, real_t a=1.0,
                  real_t b=1.0)
      : ADFunction2(f1.n_input)
      , f1(f1), f2(f2), a(a), b(b)
   {
      MFEM_VERIFY(f1.n_input == f2.n_input,
                  "SumADFunction2: f1 and f2 must have the same n_input and n_param");
   }
   void ProcessParameters(ElementTransformation &Tr,
                          const IntegrationPoint &ip) const override
   {
      f1.ProcessParameters(Tr, ip);
      f2.ProcessParameters(Tr, ip);
   }
   // default evaluator
   real_t Eval(const Vector &x) const override
   { return a*f1.Eval(x) + b*f2.Eval(x); }
   real_t operator()(const Vector &x, ElementTransformation &Tr,
                     const IntegrationPoint &ip) const override
   { return a*f1(x, Tr, ip) + b*f2(x, Tr, ip); }

   // default Jacobian evaluator
   ADReal_t operator()(const ADVector &x) const override
   { return a*f1(x) + b*f2(x); }

   // default Hessian evaluator
   AD2Real_t operator()(const AD2Vector &x) const override
   { return a*f1(x) + b*f2(x); }

   void Gradient(const Vector &x, ElementTransformation &Tr,
                 const IntegrationPoint &ip, Vector &J) const override
   {
      f1.Gradient(x, Tr, ip, Jac);
      f2.Gradient(x, Tr, ip, J);
      J *= b;
      J.Add(a, Jac);
   }
   void Hessian(const Vector &x, ElementTransformation &Tr,
                const IntegrationPoint &ip,
                DenseMatrix &H) const override
   {
      f1.Hessian(x, Tr, ip, Hess);
      f2.Hessian(x, Tr, ip, H);
      H *= b;
      H.Add(a, Hess);
   }
};

// Construct augmented energy for proximal Galerkin
// psi =
// L(u, psi) = f(u) + (1/alpha)(u*(psi-psi_k) - E^*(psi))
// so that
// dL/du = df/du + (1/alpha)(psi-psi_k)
// dL/dpsi = (1/alpha)(u - dE^*(psi))
// When primal is not full vector, set primal_begin
// The parameter should be [org_param, entropy_param, alpha, psi_k]
struct ADPGEnergy : public ADFunction2
{
   int primal_begin;
   int primal_size;
   ADFunction2 &f;
   ADFunction2 &dual_entropy;
   VectorCoefficient *latent_k_cf;
   mutable Vector latent_k;
   real_t alpha = 1.0;
   std::unique_ptr<VectorCoefficient> owned_cf;

   ADPGEnergy(ADFunction2 &f, ADFunction2 &dual_entropy, int primal_begin=0)
      : ADFunction2(f.n_input + dual_entropy.n_input)
      , f(f), dual_entropy(dual_entropy)
      , primal_begin(primal_begin)
      , primal_size(dual_entropy.n_input)
      , latent_k(primal_size)
   {
      MFEM_VERIFY(f.n_input >= primal_begin + primal_size,
                  "ADPGEnergy: f.n_input must be larger than "
                  "primal_begin + primal_size");
   }
   ADPGEnergy(ADFunction2 &f, ADFunction2 &dual_entropy,
              Coefficient &latent_k, int primal_begin=0)
      : ADPGEnergy(f, dual_entropy, primal_begin)
   {
      SetPrevLatent(latent_k);
   }
   ADPGEnergy(ADFunction2 &f, ADFunction2 &dual_entropy,
              GridFunction &latent_k, int primal_begin=0)
      : ADPGEnergy(f, dual_entropy, primal_begin)
   {
      SetPrevLatent(latent_k);
   }
   ADPGEnergy(ADFunction2 &f, ADFunction2 &dual_entropy,
              QuadratureFunction &latent_k, int primal_begin=0)
      : ADPGEnergy(f, dual_entropy, primal_begin)
   {
      SetPrevLatent(latent_k);
   }

   void SetAlpha(real_t alpha) { this->alpha = alpha; }

   void SetPrevLatent(Coefficient &psi_k)
   {
      auto cf = new VectorArrayCoefficient(1);
      cf->Set(0, &psi_k, false);
      owned_cf.reset(cf);
      latent_k_cf = cf;
   }
   void SetPrevLatent(GridFunction &psi_k)
   {
      if (psi_k.FESpace()->GetVDim() == 1)
      {
         auto cf = new VectorArrayCoefficient(1);
         cf->Set(0, new GridFunctionCoefficient(&psi_k), true);
         owned_cf.reset(cf);
         latent_k_cf = cf;
      }
      else
      {
         owned_cf = std::make_unique<VectorGridFunctionCoefficient>(&psi_k);
         latent_k_cf = owned_cf.get();
      }
   }
   void SetPrevLatent(QuadratureFunction &psi_k)
   {
      if (psi_k.GetVDim() == 1)
      {
         auto cf = new VectorArrayCoefficient(1);
         cf->Set(0, new QuadratureFunctionCoefficient(psi_k), true);
         owned_cf.reset(cf);
         latent_k_cf = cf;
      }
      else
      {
         owned_cf = std::make_unique<VectorQuadratureFunctionCoefficient>(psi_k);
         latent_k_cf = owned_cf.get();
      }
   }
   void SetPrevLatent(VectorCoefficient &psi_k_cf)
   {
      if (owned_cf) { owned_cf.reset(); }
      this->latent_k_cf = &psi_k_cf;
      latent_k.SetSize(psi_k_cf.GetVDim());
   }

   void ProcessParameters(ElementTransformation &Tr,
                          const IntegrationPoint &ip) const override
   {
      MFEM_ASSERT(latent_k_cf != nullptr,
                  "ADPGEnergy: latent_k_cf is not set. Use SetPrevLatent() to set it.");
      latent_k_cf->Eval(latent_k, Tr, ip);
      dual_entropy.ProcessParameters(Tr, ip);
      f.ProcessParameters(Tr, ip);
   }

   real_t Eval(const Vector &x) const override;
   using ADFunction2::operator();
   // default Jacobian evaluator
   ADReal_t operator()(const ADVector &x) const override;

   // default Hessian evaluator
   AD2Real_t operator()(const AD2Vector &x) const override;
};

struct MassEnergy : public ADFunction2
{
   MassEnergy(int n_var)
      : ADFunction2(n_var)
   {}
   AD_IMPL(T, V, M, x, return 0.5*(x*x););
};
struct DiffusionEnergy : public ADFunction2
{
   DiffusionEnergy(int dim)
      : ADFunction2(dim)
   {}
   AD_IMPL(T, V, M, gradu, return 0.5*(gradu*gradu););
};
struct HeteroDiffusionEnergy : public ADFunction2
{
   Coefficient &K;
   mutable real_t kappa;
   void ProcessParameters(ElementTransformation &Tr,
                          const IntegrationPoint &ip) const override
   {
      kappa = K.Eval(Tr, ip);
   }
   HeteroDiffusionEnergy(int dim, Coefficient &K)
      : ADFunction2(dim), K(K)
   {}

   AD_IMPL(T, V, M, gradu, return (kappa*0.5)*(gradu*gradu);)
};
struct AnisoDiffuionEnergy : public ADFunction2
{
   MatrixCoefficient &K;
   mutable DenseMatrix kappa;
   void ProcessParameters(ElementTransformation &Tr,
                          const IntegrationPoint &ip) const override
   {
      K.Eval(kappa, Tr, ip);
   }
   AnisoDiffuionEnergy(int dim, MatrixCoefficient &K)
      : ADFunction2(dim), K(K), kappa(K.GetHeight(), K.GetWidth())
   {
      MFEM_VERIFY(dim == K.GetHeight() && dim == K.GetWidth(),
                  "AnisoDiffuionEnergy: K must be a square matrix of size dim");
   }

   AD_IMPL(T, V, M, gradu,
   {
      T result = T();
      const int dim = gradu.Size();
      for (int i=0; i<dim; i++)
      {
         for (int j=0; j<dim; j++)
         {
            result += kappa(i,j)*gradu[i]*gradu[j];
         }
      }
      return result;
   });
};

struct LinearElasticityEnergy : public ADFunction2
{
   Coefficient *lambda_cf;
   Coefficient *mu_cf;
   const int dim;
   mutable real_t lambda;
   mutable real_t mu;
   std::vector<std::unique_ptr<Coefficient>> owned_cf;
   void ProcessParameters(ElementTransformation &Tr,
                          const IntegrationPoint &ip) const override
   {
      lambda = lambda_cf->Eval(Tr, ip);
      mu = mu_cf->Eval(Tr, ip);
   }
   LinearElasticityEnergy(int dim, Coefficient &lambda, Coefficient &mu)
      : ADFunction2(dim*dim), lambda_cf(&lambda), mu_cf(&mu), dim(dim)
   {}
   LinearElasticityEnergy(int dim, real_t lambda, real_t mu)
      : ADFunction2(dim*dim), dim(dim)
   {
      owned_cf.resize(2);
      owned_cf[0] = std::make_unique<ConstantCoefficient>(lambda);
      lambda_cf = owned_cf[0].get();
      owned_cf[1] = std::make_unique<ConstantCoefficient>(mu);
      mu_cf = owned_cf[1].get();
   }
   AD_IMPL(T, V, M, gradu,
   {
      T divnorm = T();
      for (int i=0; i<dim; i++) { divnorm += gradu[i*dim + i]; }
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
};
// Dual entropy for (negative) Shannon entropy (xlogx - x) with half bound
// when bound[1] = 1, [lower, inf[
// when bound[1] = -1, ]-inf, upper]
//
// The resulting dual is (f(pm1*(x - shift)))^*
// = f^*(pm1*x^*) + shift*pm1*x^*
struct ShannonEntropy : public ADFunction2
{
   Coefficient &bound;
   mutable real_t shift;
   int sign;
   ShannonEntropy(Coefficient &bound, int sign)
      : ADFunction2(1)
      , bound(bound)
      , sign(sign)
   {
      MFEM_VERIFY(sign == 1 || sign == -1,
                  "ShannonEntropy: sign must be 1 or -1");
   }
   void ProcessParameters(ElementTransformation &Tr,
                          const IntegrationPoint &ip) const override
   {
      shift = bound.Eval(Tr, ip);
   }
   AD_IMPL(T, V, M, x, return sign*(exp(x[0]*sign)) + shift*x[0]; );
};

// Dual entropy for (negative) Fermi-Dirac with [lower, upper] bounds
struct FermiDiracEntropy : public ADFunction2
{
   Coefficient *upper_bound;
   Coefficient *lower_bound;
   std::vector<std::unique_ptr<Coefficient>> owned_cf;
   mutable real_t shift;
   mutable real_t scale;

   int sign;
   FermiDiracEntropy(Coefficient &lower_bound, Coefficient &upper_bound)
      : ADFunction2(1)
      , lower_bound(&lower_bound)
      , upper_bound(&upper_bound)
   { }
   FermiDiracEntropy(GridFunction &lower_bound, GridFunction &upper_bound)
      : ADFunction2(1)
   {
      MFEM_VERIFY(lower_bound.FESpace()->GetVDim() == 1 &&
                  upper_bound.FESpace()->GetVDim() == 1,
                  "FermiDiracEntropy: lower_bound and upper_bound must be scalar GridFunctions");
      owned_cf.resize(2);
      owned_cf[0] = std::make_unique<GridFunctionCoefficient>(&lower_bound);
      this->lower_bound = owned_cf[0].get();
      owned_cf[1] = std::make_unique<GridFunctionCoefficient>(&upper_bound);
      this->upper_bound = owned_cf[1].get();
   }
   FermiDiracEntropy(real_t lower_bound, real_t upper_bound)
      : ADFunction2(1)
   {
      owned_cf.resize(2);
      owned_cf[0] = std::make_unique<ConstantCoefficient>(lower_bound);
      this->lower_bound = owned_cf[0].get();
      owned_cf[1] = std::make_unique<ConstantCoefficient>(upper_bound);
      this->upper_bound = owned_cf[1].get();
   }

   void ProcessParameters(ElementTransformation &Tr,
                          const IntegrationPoint &ip) const override
   {
      shift = lower_bound->Eval(Tr, ip);
      scale = upper_bound->Eval(Tr, ip) - shift;
   }
   AD_IMPL(T, V, M, x,
   {
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
};
// Dual entropy for (negative) Hellinger entropy with bound > 0
struct HellingerEntropy : public ADFunction2
{
   Coefficient &bound;
   mutable real_t scale;

   HellingerEntropy(Coefficient &bound)
      : ADFunction2(1), bound(bound)
   { }
   void ProcessParameters(ElementTransformation &Tr,
                          const IntegrationPoint &ip) const override
   {
      scale = bound.Eval(Tr, ip);
      MFEM_VERIFY(scale > 0, "HellingerEntropy: bound must be positive");
   }
   AD_IMPL(T, V, M, x, return sqrt(1 + (x*x)*(scale*scale)););
};

// Dual entropy for (negative) Simplex entropy with
// x_i >= 0 sum_i x_i = bound
// Also known as cateborical entropy or multinomial Shannon entropy
struct SimplexEntropy : public ADFunction2
{
   Coefficient &bound;
   mutable real_t scale;

   SimplexEntropy(Coefficient &bound)
      : ADFunction2(1), bound(bound)
   { }
   void ProcessParameters(ElementTransformation &Tr,
                          const IntegrationPoint &ip) const override
   {
      scale = bound.Eval(Tr, ip);
      MFEM_ASSERT(scale >= 0, "SimplexEntropy: bound must be non-negative");
      if (scale == 0)
      {
         MFEM_WARNING("SimplexEntropy: bound is zero, entropy is undefined");
      }
   }
   AD_IMPL(T, V, M, x,
   {
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
};
}
