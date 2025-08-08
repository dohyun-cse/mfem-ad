#include "ad_native.hpp"

namespace mfem
{

void ADFunction::Gradient(const Vector &x, ElementTransformation &Tr,
                          const IntegrationPoint &ip,
                          Vector &J) const
{
   ProcessParameters(Tr, ip);
   Gradient(x, J);
}
void ADFunction::Gradient(const Vector &x, Vector &J) const
{
   MFEM_ASSERT(x.Size() == n_input,
               "ADFunction::Gradient: x.Size() must match n_input");
   J.SetSize(x.Size());
   ADVector x_ad(x);
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
   ProcessParameters(Tr, ip);
   Hessian(x, H);
}

void ADFunction::Hessian(const Vector &x, DenseMatrix &H) const
{
   MFEM_ASSERT(x.Size() == n_input,
               "ADFunction::Hessian: x.Size() must match n_input");
   H.SetSize(x.Size(), x.Size());
   AD2Vector x_ad(x);
   for (int i=0; i<x.Size(); i++) // Loop for the first derivative
   {
      x_ad[i].value.gradient = 1.0;
      for (int j=0; j<=i; j++)
      {
         x_ad[j].gradient.value = 1.0;
         AD2Real_t result = (*this)(x_ad);
         H(j, i) = result.gradient.gradient;
         H(i, j) = result.gradient.gradient;
         x_ad[j].gradient.value = 0.0; // Reset gradient for next iteration
      }
      x_ad[i].value.gradient = 0.0;
   }
}

void ADVectorFunction::Gradient(const Vector &x, DenseMatrix &J) const
{
   MFEM_ASSERT(x.Size() == n_input,
               "ADVectorFunction::Gradient: x.Size() must match n_input");
   ADVector x_ad(x);
   ADVector Fx(n_output);
   J.SetSize(n_output, n_input);
   for (int i=0; i<x.Size(); i++)
   {
      x_ad[i].gradient = 1.0;
      Fx = ADReal_t();
      (*this)(x_ad, Fx);
      for (int j=0; j<n_output; j++)
      {
         J(j,i) = Fx[j].gradient;
      }
      x_ad[i].gradient = 0.0; // Reset gradient for next iteration
   }
}

void ADVectorFunction::Hessian(const Vector &x, DenseTensor &H) const
{
   MFEM_ASSERT(x.Size() == n_input,
               "ADVectorFunction::Gradient: x.Size() must match n_input");
   AD2Vector x_ad(x);
   AD2Vector Fx(n_output);
   H.SetSize(n_input, n_input, n_output);
   for (int i=0; i<n_input; i++) // Loop for the first derivative
   {
      x_ad[i].value.gradient = 1.0;
      for (int j=0; j<=i; j++)
      {
         x_ad[j].gradient.value = 1.0;
         Fx = AD2Real_t();
         (*this)(x_ad, Fx);
         for (int k=0; k<n_output; k++)
         {
            H(j, i, k) = Fx[k].gradient.gradient;
            H(i, j, k) = Fx[k].gradient.gradient;
         }
         x_ad[j].gradient.value = 0.0; // Reset gradient for next iteration
      }
      x_ad[i].value.gradient = 0.0;
   }
}

DifferentiableCoefficient& DifferentiableCoefficient::AddInput(Coefficient &cf)
{
   MFEM_VERIFY(idx < f.n_input,
               "DifferentiableCoefficient: Too many input variables added. "
               "n_input=" << f.n_input << ", idx=" << idx);
   cfs.push_back(&cf);
   cfs_idx.push_back(idx++);
   x.SetSize(idx);
   return *this;
}

DifferentiableCoefficient& DifferentiableCoefficient::AddInput(
   VectorCoefficient &vcf)
{
   MFEM_VERIFY(idx < f.n_input,
               "DifferentiableCoefficient: Too many input variables added. "
               "n_input=" << f.n_input << ", idx=" << idx);
   v_cfs.push_back(&vcf);
   v_cfs_idx.push_back(idx);
   idx += vcf.GetVDim();
   x.SetSize(idx);
   return *this;
}

DifferentiableCoefficient& DifferentiableCoefficient::AddInput(GridFunction &gf)
{
   MFEM_VERIFY(idx < f.n_input,
               "DifferentiableCoefficient: Too many input variables added. "
               "n_input=" << f.n_input << ", idx=" << idx);
   gfs.push_back(&gf);
   gfs_idx.push_back(idx);
   idx += gf.FESpace()->GetVDim();
   x.SetSize(idx);
   return *this;
}

DifferentiableCoefficient& DifferentiableCoefficient::AddInput(
   QuadratureFunction &qf)
{
   MFEM_VERIFY(idx < f.n_input,
               "DifferentiableCoefficient: Too many input variables added. "
               "n_input=" << f.n_input << ", idx=" << idx);
   qfs.push_back(&qf);
   qfs_idx.push_back(idx);
   idx += qf.GetVDim();
   x.SetSize(idx);
   return *this;
}


void DifferentiableCoefficient::EvalInput(ElementTransformation &T,
      const IntegrationPoint &ip) const
{
   Vector x_view;
   for (int i=0; i<cfs.size(); i++)
   {
      x[cfs_idx[i]] = cfs[i]->Eval(T, ip);
   }
   for (int i=0; i<v_cfs.size(); i++)
   {
      x_view.SetDataAndSize(x.GetData() + v_cfs_idx[i],
                            v_cfs[i]->GetVDim());
      v_cfs[i]->Eval(x_view, T, ip);
   }
   for (int i=0; i<gfs.size(); i++)
   {
      x_view.SetDataAndSize(x.GetData() + gfs_idx[i],
                            gfs[i]->FESpace()->GetVDim());
      gfs[i]->GetVectorValue(T, ip, x_view);
   }
   for (int i=0; i<qfs.size(); i++)
   {
      x_view.SetDataAndSize(x.GetData() + qfs_idx[i],
                            qfs[i]->GetVDim());
      qfs[i]->GetValues(T.ElementNo, ip.index, x_view);
   }
}

Lagrangian Lagrangian::AddEqConstraint(ADFunction &constraint,
                                       real_t target)
{
   n_input++;
   eq_con.push_back(&constraint);
   int numCon = eq_con.size();
   eq_rhs.SetSize(numCon);
   eq_rhs[numCon - 1] = target;
   return *this;
}

void Lagrangian::ProcessParameters(ElementTransformation &Tr,
                                   const IntegrationPoint &ip) const
{
   objective.ProcessParameters(Tr, ip);
   for (auto *con : eq_con) { con->ProcessParameters(Tr, ip); }
}

ALFunctional ALFunctional::AddEqConstraint(ADFunction &constraint,
      real_t target)
{
   eq_con.push_back(&constraint);
   int numCon = eq_con.size();
   eq_rhs.SetSize(numCon);
   lambda.SetSize(numCon);

   eq_rhs[numCon - 1] = target;
   lambda[numCon - 1] = 0.0;
   return *this;
}

void ALFunctional::SetLambda(const Vector &lambda)
{
   MFEM_VERIFY(lambda.Size() == this->lambda.Size(),
               "ALFunctional: lambda size mismatch");
   this->lambda = lambda;
}

void ALFunctional::SetPenalty(real_t mu)
{
   MFEM_VERIFY(mu >= 0.0, "ALFunctional: mu must be non-negative");
   this->penalty = mu;
}

void ALFunctional::ProcessParameters(ElementTransformation &Tr,
                                     const IntegrationPoint &ip) const
{
   objective.ProcessParameters(Tr, ip);
   for (auto *con : eq_con) { con->ProcessParameters(Tr, ip); }
}

} // namespace mfem
