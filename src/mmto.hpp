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
      : ADFunction(0), p(simp_exp)
   {
      evaluator.Add(E);
      this->E = &evaluator.val.GetBlock(0);
      n_input = this->E->Size();
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

class SubADFunction : public ADFunction
{
   ADFunction &f;
   int start, end;
   SubADFunction(ADFunction &f, std::vector<ADFunction*> input_func,
                 std::vector<Evaluator::param_t> params)
      : ADFunction(f.n_input), f(f), start(0), end(f.n_input)
   { }
};

};
