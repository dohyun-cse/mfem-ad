#pragma once
#include "mfem.hpp"
#include "pg.hpp"
#include "tools.hpp"
#include "ad_intg.hpp"

namespace mfem
{
struct SIMPFunction : public ADFunction
{
   const Vector *E;
   real_t p;
   SIMPFunction(Evaluator::param_t E, real_t simp_exp)
      : ADFunction(Evaluator::GetSize(E)), p(simp_exp)
   {
      evaluator.Add(E);
      this->E = &evaluator.val.GetBlock(0);
   }
   AD_IMPL(T, V, M, x,
   {
      T result = T();
      for (int i=0; i<x.Size(); i++)
      {
         result += (*E)[i]*pow(x[i], p);
      }
      return result;
   });
};

// A functional that depends on parameters
// e.g., f(x; f_0(param), ..., f_n(param))
// where f_i:R^m->R are differentiable functions of the parameters
// and f itself is differentiable with respect to x and f_i.
// We assume that f is linear with respect to f_i,
// so that we can write
// df/dc_i = df/df_j * df_j/dc_i
//
// The evaluator will hold [states, f_i(params)]
// In the regular mode, the state will not be evaluated,
// but the parameters will be evaluated.
// At the parameter gradient evaluation,
// the states will be evaluated.
class ParametrizedFunctional : public ADFunction
{
private:
   std::vector<std::unique_ptr<DifferentiableCoefficient>> param_coeffs;
   const int param_begin;
   const int n_state;
   const int n_f;
   const int param_dim;
   class ParamGradient : public VectorCoefficient
   {
   private:
      ParametrizedFunctional &parent;
      mutable Vector dfdc;
   public:
      ParamGradient(ParametrizedFunctional &parent)
         : VectorCoefficient(parent.param_dim), parent(parent)
      {}
      void Eval(Vector &J, ElementTransformation &Tr,
                const IntegrationPoint &ip) override;
   };
   friend class ParamGradient;
   ParamGradient param_grad;

public:
   ParametrizedFunctional(int n_input,
                          int param_dim,
                          std::initializer_list<Evaluator::param_t> state_srcs,
                          std::initializer_list<ADFunction*> param_funcs,
                          std::initializer_list<Evaluator::param_t> param_srcs)
      : ADFunction(n_input, param_funcs.size())
      , param_begin(state_srcs.size())
      , n_state(state_srcs.size())
      , n_f(param_srcs.size())
      , param_dim(param_dim)
      , param_grad(*this)
   {
      for (auto &src : state_srcs)
      {
         evaluator.Add(src);
      }
      std::vector<Evaluator::param_t> param_list(param_srcs);
      int deduced_param_dim = 0;
      for (auto &src : param_list)
      {
         deduced_param_dim += Evaluator::GetSize(src);
      }
      MFEM_VERIFY(deduced_param_dim == param_dim,
                  "ParametrizedFunctional: "
                  "param_dim must match the total size of parameters");

      for (auto &func : param_funcs)
      {
         param_coeffs.push_back(std::make_unique<DifferentiableCoefficient>(*func));
         for (auto &param : param_list)
         {
            param_coeffs.back()->AddInput(param);
         }
         evaluator.Add(param_coeffs.back().get());
      }
   }
   void ProcessParameters(ElementTransformation &Tr,
                          const IntegrationPoint &ip) const override
   {
      for (int i=n_state; i<n_state + n_f; i++)
      { evaluator.Eval(i, Tr, ip); }
   }
};

class SiMPLTopopt : public IterativeSolver
{
   ParametrizedFunctional &objective;
   ParametrizedFunctional &state_energy;
   const int dim;
   const int numMaterials;
   Array<GridFunction*> states;
   GridFunction &rho;
   bool skip_adjoint = false;
public:
   SiMPLTopopt(ParametrizedFunctional &objective,
               ParametrizedFunctional &state_energy,
               int dim, int numMaterials,
               Array<GridFunction*> states,
               GridFunction &rho)
      : objective(objective)
      , state_energy(state_energy)
      , dim(dim)
      , numMaterials(numMaterials)
      , states(states)
      , rho(rho)
   { }
#ifdef MFEM_USE_MPI
   SiMPLTopopt(ParametrizedFunctional &objective,
               ParametrizedFunctional &state_energy,
               int dim, int numMaterials,
               Array<GridFunction*> states,
               ParGridFunction &rho)
      : IterativeSolver(rho.ParFESpace()->GetComm())
      , objective(objective)
      , state_energy(state_energy)
      , dim(dim)
      , numMaterials(numMaterials)
      , states(states)
      , rho(rho)
   { }
#endif
   void SkipAdjoint(bool skip=true) { skip_adjoint = skip; }
   void Mult(const Vector &load, Vector &y) const override
   {
   }
};

class ParametrizedCompliance : public ParametrizedFunctional
{
   const int dim;
   const int numMaterials;
   const real_t &lambda;
   const real_t &mu;
public:
   ParametrizedCompliance(int dim,
                          ADFunction &lambda,
                          ADFunction &mu,
                          VectorGradientGridFunction &grad_disp,
                          GridFunction &rho)
      : ParametrizedFunctional(dim*dim, rho.FESpace()->GetVDim(),
   {&grad_disp}, {&lambda, &mu}, {&rho})
   , dim(dim), numMaterials(lambda.n_input)
   , lambda(*(evaluator.val.GetData() + dim*dim))
   , mu(*(evaluator.val.GetData() + dim*dim+1))
   {}

   AD_IMPL(T, V, M, gradu,
   {
      T div_sqrd = T();
      for (int i=0; i<dim; i++) { div_sqrd += gradu[i*dim + i]; }
      div_sqrd = div_sqrd*div_sqrd;
      T symm_sqrd = T();
      for (int i=0; i<dim; i++)
      {
         for (int j=0; j<dim; j++)
         {
            T symm = 0.5*(gradu[i*dim + j] + gradu[j*dim + i]);
            symm_sqrd += symm*symm;
         }
      }
      return 0.5*lambda*div_sqrd + mu*symm_sqrd;
   });
};

};
