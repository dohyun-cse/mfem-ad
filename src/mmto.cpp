#include "mmto.hpp"
namespace mfem
{
void ParametrizedFunctional::ParamGradient::Eval(Vector &J,
      ElementTransformation &Tr,
      const IntegrationPoint &ip)
{
   J.SetSize(vdim);
   J = 0.0;
   Evaluator &evaluator = parent.evaluator;
   const int n_state = parent.n_state;
   const int state_dim = parent.n_input;
   const int param_dim = parent.param_dim;
   // This will evaluate parameters with given parameter sources
   // States are not evaluated
   parent.ProcessParameters(Tr, ip);
   // Now we evaluate the states
   for (int i=0; i<parent.n_state; i++)
   {
      evaluator.Eval(i, Tr, ip);
   }
   // Make a reference to the state vector
   const Vector state(evaluator.val, 0, state_dim);
   // Now we evaluate the parameter gradient using param_coeffs
   for (int i=0; i<parent.param_coeffs.size(); i++)
   {
      parent.param_coeffs[i]->Gradient().Eval(dfdc, Tr, ip);
      // save the current f_i value. It should be scalar.
      real_t val = evaluator.val.GetBlock(i + n_state)[0];
      for (int j=0; j<param_dim; j++)
      {
         evaluator.val.GetBlock(i + n_state)[0] = dfdc[j];
         J[j] += parent(state);
      }
      // Reset the value to the original one
      evaluator.val.GetBlock(i + n_state)[0] = val;
   }
}
};
