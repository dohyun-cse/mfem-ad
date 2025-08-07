#pragma once
#include "mfem.hpp"
#include "ad_native.hpp"

namespace mfem
{
// PGStepSizeRule defines the step size rule for the Proximal Galerkin (PG) method.
// See RuleType for the available rules
struct PGStepSizeRule
{
   enum RuleType
   {
      CONSTANT, // alpha0
      POLY, // alpha0 * (iter+1)^ratio
      EXP, // alpha0 * ratio^iter
      DOUBLE_EXP, // alpha0 * ratio^(ratio2^iter)
      // ... add more rules as needed
      INVALID // used to check for valid rule types
   };
   RuleType rule_type;

   real_t max_alpha;
   real_t alpha0; // initial step size
   real_t ratio; // poly degree (POLY), exponential base (EXP, DOUBLE_EXP)
   real_t ratio2; // nested exponential base (DOUBLE_EXP)

   PGStepSizeRule(int rule_type,
                  real_t alpha0 = 1.0, real_t max_alpha = 1e06,
                  real_t ratio = -1.0, real_t ratio2 = -1.0);

   /// Get the step size for the given iteration
   real_t Get(int iter) const;
};

// Base struct for dual entropy functions
struct ADEntropy : public ADFunction
{
   using ADFunction::ADFunction;
};

// Construct augmented energy for proximal Galerkin
// psi =
// L(u, psi) = f(u) + (1/alpha)(u*(psi-psi_k) - E^*(psi))
// so that
// dL/du = df/du + (1/alpha)(psi-psi_k)
// dL/dpsi = (1/alpha)(u - dE^*(psi))
// When primal is not full vector, set primal_begin
// The parameter should be [org_param, entropy_param, alpha, psi_k]
struct ADPGFunctional : public ADFunction
{
   int primal_begin;
   int primal_size;
   ADFunction &f;
   ADEntropy &dual_entropy;
   GridFunction *latent_k_gf;
   mutable Vector latent_k;
   real_t alpha = 1.0;
   std::unique_ptr<VectorCoefficient> owned_cf;

   ADPGFunctional(ADFunction &f, ADEntropy &dual_entropy, int primal_begin=0)
      : ADFunction(f.n_input + dual_entropy.n_input)
      , f(f), dual_entropy(dual_entropy)
      , primal_begin(primal_begin)
      , primal_size(dual_entropy.n_input)
      , latent_k(primal_size)
   {
      MFEM_VERIFY(f.n_input >= primal_begin + primal_size,
                  "ADPGFunctional: f.n_input must be larger than "
                  "primal_begin + primal_size");
   }
   ADPGFunctional(ADFunction &f, ADEntropy &dual_entropy,
                  GridFunction &latent_k, int primal_begin=0)
      : ADPGFunctional(f, dual_entropy, primal_begin)
   {
      SetPrevLatent(latent_k);
   }

   ADEntropy &GetEntropy() const { return dual_entropy; }
   std::vector<ADEntropy*> GetEntropies() const
   {
      std::vector<ADEntropy*> entropies{ &dual_entropy };
      if (dynamic_cast<ADPGFunctional*>(&f))
      {
         // f is also a PGFunctional, we append its entropy
         auto pgf = dynamic_cast<const ADPGFunctional*>(&f);
         auto f_entropies = pgf->GetEntropies();
         entropies.insert(entropies.end(), f_entropies.begin(), f_entropies.end());
      }
      return entropies;
   }

   // Set the penalty parameter alpha
   // if propagate is true, and f is also ADPGFunctional,
   // then propagate the alpha to f
   void SetAlpha(real_t alpha, bool propagate=false)
   {
      this->alpha = alpha;
      if (propagate)
      {
         // propagate alpha to f if f is also ADPGFunctional
         if (auto pgf = dynamic_cast<ADPGFunctional*>(&f))
         {
            pgf->SetAlpha(alpha, propagate);
         }
      }
   }
   real_t GetAlpha() const { return alpha; }

   void SetPrevLatent(GridFunction &psi_k)
   {
      MFEM_VERIFY(psi_k.FESpace()->GetVDim() == primal_size,
                  "ADPGFunctional: psi_k must have the same dimension as "
                  "dual_entropy.n_input");
      latent_k_gf = &psi_k;
   }

   void ProcessParameters(ElementTransformation &Tr,
                          const IntegrationPoint &ip) const override
   {
      MFEM_ASSERT(latent_k_cf != nullptr,
                  "ADPGFunctional: latent_k_cf is not set. Use SetPrevLatent() to set it.");
      latent_k_gf->GetVectorValue(Tr, ip, latent_k);
      dual_entropy.ProcessParameters(Tr, ip);
      f.ProcessParameters(Tr, ip);
   }

   real_t operator()(const Vector &x) const override;
   using ADFunction::operator();
   // default Jacobian evaluator
   ADReal_t operator()(const ADVector &x) const override;

   // default Hessian evaluator
   AD2Real_t operator()(const AD2Vector &x) const override;
};

// Proximal Galerkin functional with QF latent variable
struct ADPGFunctionalQ : public ADPGFunctional
{
   QuadratureFunction *latent_k_qf;

   ADPGFunctionalQ(ADFunction &f, ADEntropy &dual_entropy, int primal_begin=0)
      : ADPGFunctional(f, dual_entropy, primal_begin)
   {
      MFEM_VERIFY(f.n_input >= primal_begin + primal_size,
                  "ADPGFunctional: f.n_input must be larger than "
                  "primal_begin + primal_size");
   }
   ADPGFunctionalQ(ADFunction &f, ADEntropy &dual_entropy,
                   QuadratureFunction &latent_k, int primal_begin=0)
      : ADPGFunctional(f, dual_entropy, primal_begin)
   {
      SetPrevLatent(latent_k);
   }

   void SetPrevLatent(QuadratureFunction &psi_k)
   {
      MFEM_VERIFY(psi_k.GetVDim() == primal_size,
                  "ADPGFunctional: psi_k must have the same dimension as "
                  "dual_entropy.n_input");
      latent_k_qf = &psi_k;
   }

   void ProcessParameters(ElementTransformation &Tr,
                          const IntegrationPoint &ip) const override
   {
      MFEM_ASSERT(latent_k_cf != nullptr,
                  "ADPGFunctional: latent_k_cf is not set. Use SetPrevLatent() to set it.");
      latent_k_qf->GetValues(Tr.ElementNo, ip.index, latent_k);
      dual_entropy.ProcessParameters(Tr, ip);
      f.ProcessParameters(Tr, ip);
   }

   real_t operator()(const Vector &x) const override;
   using ADFunction::operator();
   // default Jacobian evaluator
   ADReal_t operator()(const ADVector &x) const override;

   // default Hessian evaluator
   AD2Real_t operator()(const AD2Vector &x) const override;
};

enum LatentType
{
   COEFFICIENT,
   GF,
   QF
};


// Dual entropy for (negative) Shannon entropy (xlogx - x) with half bound
// when bound[1] = 1, [lower, inf[
// when bound[1] = -1, ]-inf, upper]
//
// The resulting dual is (f(pm1*(x - shift)))^*
// = f^*(pm1*x^*) + shift*pm1*x^*
struct ShannonEntropy : public ADEntropy
{
   Coefficient &bound;
   mutable real_t shift;
   int sign;
   ShannonEntropy(Coefficient &bound, int sign)
      : ADEntropy(1)
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
struct FermiDiracEntropy : public ADEntropy
{
   Coefficient *upper_bound;
   Coefficient *lower_bound;
   std::vector<std::unique_ptr<Coefficient>> owned_cf;
   mutable real_t shift;
   mutable real_t scale;

   int sign;
   FermiDiracEntropy(Coefficient &lower_bound, Coefficient &upper_bound)
      : ADEntropy(1)
      , lower_bound(&lower_bound)
      , upper_bound(&upper_bound)
   { }
   FermiDiracEntropy(GridFunction &lower_bound, GridFunction &upper_bound)
      : ADEntropy(1)
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
   FermiDiracEntropy(QuadratureFunction &lower_bound,
                     QuadratureFunction &upper_bound)
      : ADEntropy(1)
   {
      MFEM_VERIFY(lower_bound.GetVDim() == 1 &&
                  upper_bound.GetVDim() == 1,
                  "FermiDiracEntropy: lower_bound and upper_bound must be scalar GridFunctions");
      owned_cf.resize(2);
      owned_cf[0] = std::make_unique<QuadratureFunctionCoefficient>(lower_bound);
      this->lower_bound = owned_cf[0].get();
      owned_cf[1] = std::make_unique<QuadratureFunctionCoefficient>(upper_bound);
      this->upper_bound = owned_cf[1].get();
   }
   FermiDiracEntropy(real_t lower_bound, real_t upper_bound)
      : ADEntropy(1)
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
struct HellingerEntropy : public ADEntropy
{
   Coefficient &bound;
   mutable real_t scale;

   HellingerEntropy(Coefficient &bound)
      : ADEntropy(1), bound(bound)
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
struct SimplexEntropy : public ADEntropy
{
   Coefficient &bound;
   mutable real_t scale;

   SimplexEntropy(Coefficient &bound)
      : ADEntropy(1), bound(bound)
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
} // namespace mfem
