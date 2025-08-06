#include "pg.hpp"
namespace mfem
{
PGStepSizeRule::PGStepSizeRule(int rule_type,
                               real_t alpha0, real_t max_alpha,
                               real_t ratio, real_t ratio2)
   : rule_type(static_cast<RuleType>(rule_type))
   , max_alpha(max_alpha), alpha0(alpha0), ratio(ratio), ratio2(ratio2)
{
   MFEM_VERIFY(rule_type < RuleType::INVALID,
               "PGStepSizeRule: Invalid rule type");
   MFEM_VERIFY(alpha0 > 0, "PGStepSizeRule: alpha0 must be positive");
   MFEM_VERIFY(max_alpha >= alpha0,
               "PGStepSizeRule: max_alpha must be greater than or equal to alpha0");
   if (rule_type == RuleType::CONSTANT)
   {
   }
   else if (rule_type == RuleType::POLY)
   {
      MFEM_VERIFY(ratio > 0, "PGStepSizeRule: ratio must be positive for POLY rule");
   }
   else if (rule_type == RuleType::EXP)
   {
      MFEM_VERIFY(ratio > 1,
                  "PGStepSizeRule: ratio must be greater than 1 for EXP rule");
   }
   else if (rule_type == RuleType::DOUBLE_EXP)
   {
      MFEM_VERIFY(ratio > 1 && ratio2 > 1,
                  "PGStepSizeRule: ratio and ratio2 must be greater than 1 for DOUBLE_EXP rule");
   }
}

real_t PGStepSizeRule::Get(int iter) const
{
   real_t alpha = alpha0;
   switch (rule_type)
   {
      case RuleType::CONSTANT:
         break;
      case RuleType::POLY:
         alpha *= std::pow(iter+1, ratio);
         break;
      case RuleType::EXP:
         alpha *= std::pow(ratio, iter);
         break;
      case RuleType::DOUBLE_EXP:
         alpha *= std::pow(ratio, std::pow(ratio2, iter));
         break;
      default:
         break;
   }
   return std::min(alpha, max_alpha);
}

real_t ADPGFunctional::operator()(const Vector &x) const
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
ADReal_t ADPGFunctional::operator()(const ADVector &x) const
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
AD2Real_t ADPGFunctional::operator()(const AD2Vector &x) const
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

}
