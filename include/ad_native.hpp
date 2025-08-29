#pragma once

#include "mfem.hpp"
#include "miniapps/autodiff/tadvector.hpp"
#include "miniapps/autodiff/taddensemat.hpp"

namespace mfem
{
template <typename T>
constexpr auto type_name()
{
#if defined(__clang__)
   return std::string_view(__PRETTY_FUNCTION__);
#elif defined(__GNUC__)
   return std::string_view(__PRETTY_FUNCTION__);
#elif defined(_MSC_VER)
   return std::string_view(__FUNCSIG__);
#else
   return std::string_view("unknown");
#endif
}
template <typename value_type, typename gradient_type, typename other_type>
MFEM_HOST_DEVICE
inline future::dual<value_type, gradient_type> max(
   future::dual<value_type, gradient_type> a, other_type b);

inline real_t max(const real_t a, const real_t b) { return std::max(a,b); }

template <typename value_type, typename gradient_type, typename other_type>
MFEM_HOST_DEVICE
inline future::dual<value_type, gradient_type> min(
   future::dual<value_type, gradient_type> a, other_type b);

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

class Evaluator
{
   // To add a new parameter type,
   // implement GetSize() and Eval() method
public:
   using param_t = std::variant<
                   real_t, Vector, DenseMatrix, // pass by value
                   const real_t*, const Vector*, const DenseMatrix*, // pass by pointer
                   Coefficient*, VectorCoefficient*, MatrixCoefficient*,
                   const GridFunction*, const ParGridFunction*,
                   const QuadratureFunction*>;
private:
   Array<int> offsets;
   std::vector<param_t> params;

   mutable Vector loc_vec_val;
   mutable DenseMatrix loc_mat_val;

public:
   mutable BlockVector val;
   mutable Array<bool> owns;
   Evaluator(): offsets{0} {}
   Evaluator(int capacity)
      : offsets{0}
   {
      val.SetSize(capacity);
      val.SetSize(0);
   }
   virtual ~Evaluator();
   // Add a parameter to the evaluator
   int Add(param_t param, bool eval_owns = false);
   int Add(Vector &v)
   {
      if (dynamic_cast<GridFunction*>(&v) || dynamic_cast<QuadratureFunction*>(&v))
      {
         MFEM_WARNING("Adding GridFunction or QuadratureFunction by value, instead of its pointer. "
                      "This result in the whole GridFunction value will be used at each quadrature point, "
                      "which is likely not what you want. "
                      "Use Add(const GridFunction*) instead.");
      }
      return Add((param_t)v);
   }
   int Add(const Vector &v)
   {
      if (dynamic_cast<const GridFunction*>(&v) ||
          dynamic_cast<const QuadratureFunction*>(&v))
      {
         MFEM_WARNING("Adding GridFunction or QuadratureFunction by value, instead of its pointer. "
                      "This result in the whole GridFunction value will be used at each quadrature point, "
                      "which is likely not what you want. "
                      "Use Add(const GridFunction*) instead.");
      }
      return Add((param_t)v);
   }
   // Replace a parameter at index i with a new parameter
   // The output size of param should match the size of the old parameter
   void Replace(size_t i, param_t param);
   param_t Get(size_t i) const
   {
      MFEM_VERIFY(i >= 0 && i < params.size(),
                  "Evaluator::Get: index out of range");
      return params[i];
   }

   // Evaluate all parameters at once
   // and return the block vector
   const BlockVector &Eval(ElementTransformation &Tr,
                           const IntegrationPoint &ip) const
   {
      for (int i=0; i<params.size(); i++)
      { Eval(i, Tr, ip); }
      return val;
   }

   // Evaluate the parameter at index i
   // this will update the val block vector, and return the corresponding block
   const Vector& Eval(int i, ElementTransformation &Tr,
                      const IntegrationPoint &ip) const;
   static int GetSize(const param_t &param);
   int GetSize(size_t i) const
   {
      MFEM_VERIFY(i >= 0 && i < offsets.Size() - 1,
                  "Evaluator::GetSize: index out of range");
      return offsets[i+1] - offsets[i];
   }
   void Project(QuadratureFunction &qf)
   {
      const int vdim = offsets.Last();
      qf.SetVDim(vdim);

      QuadratureSpaceBase &qspace = *qf.GetSpace();
      Vector qf_view(qf.GetData(), vdim);
      for (int i=0; i<qspace.GetNE(); i++)
      {
         ElementTransformation &Tr = *qspace.GetTransformation(i);
         const IntegrationRule &ir = qspace.GetIntRule(i);
         for (int j=0; j<ir.GetNPoints(); j++)
         {
            const IntegrationPoint &ip = ir.IntPoint(j);
            qf_view = Eval(Tr, ip);
            qf_view.SetData(qf.GetData() + vdim);
         }
      }
      return;
   }
   int GetVDim() const { return offsets.Last(); }
};

class EvaluatorCF : public Coefficient
{
   Evaluator &evaluator;
   int idx;
   const real_t &val;
public:
   EvaluatorCF(Evaluator &evaluator_, int outer_idx=0, int inner_idx=0)
      : evaluator(evaluator_)
      , idx(outer_idx)
      , val(evaluator.val.GetBlock(outer_idx)(inner_idx)) {}
   real_t Eval(ElementTransformation &Tr, const IntegrationPoint &ip) override
   {
      evaluator.Eval(idx, Tr, ip);
      return val;
   }
};
class EvaluatorVCF : public VectorCoefficient
{
   Evaluator &evaluator;
   int idx;
public:
   EvaluatorVCF(Evaluator &evaluator, int idx=-1)
      : VectorCoefficient(idx == -1 ? evaluator.GetVDim() :
                          evaluator.val.GetBlock(idx).Size())
      , evaluator(evaluator)
      , idx(idx)
   { }

   void Eval(Vector &V, ElementTransformation &Tr,
             const IntegrationPoint &ip) override
   {
      if (idx == -1) { V = evaluator.Eval(Tr, ip); }
      else { V = evaluator.Eval(idx, Tr, ip); }
   }
};
class EvaluatorMCF : public MatrixCoefficient
{
   Evaluator &evaluator;
   int idx;
   const DenseMatrix val;
public:
   EvaluatorMCF(Evaluator &evaluator, int h, int w, int idx=0)
      : MatrixCoefficient(h, w)
      , evaluator(evaluator)
      , idx(idx)
      , val(evaluator.val.GetBlock(idx).GetData(), h, w)
   {
      MFEM_VERIFY(evaluator.val.GetBlock(idx).Size() == h*w,
                  "EvaluatorMCF: size mismatch");
   }
   void Eval(DenseMatrix &M, ElementTransformation &Tr,
             const IntegrationPoint &ip) override
   {
      evaluator.Eval(idx, Tr, ip);
      M = val;
   }
};

class ADVectorFunction;
class ADFunction
{
protected:

   int AddParameter(Evaluator::param_t param)
   { return evaluator.Add(param); }

   void ReplaceParameter(int i, Evaluator::param_t param)
   { evaluator.Replace(i, param); }

   Evaluator evaluator;
public:
   virtual void ProcessParameters(ElementTransformation &Tr,
                                  const IntegrationPoint &ip) const
   { ProcessParameters(evaluator.Eval(Tr, ip)); }
   virtual void ProcessParameters(const BlockVector &param_val) const
   { }

   const int n_input;
   ADFunction(int n_input): n_input(n_input) {}
   // Constructor with capacity for evaluator.
   // This is useful when the parameter size is known in advance,
   // so that we can get references to the parameters at construction time.
   ADFunction(int n_input, int capacity)
      : n_input(n_input), evaluator(capacity)
   {
      MFEM_ASSERT(n_input > 0, "ADFunction: n_input must be positive");
   }
   // default evaluator
   virtual real_t operator()(const Vector &x) const
   { MFEM_ABORT("Not implemented. Use AD_IMPL macro to implement all path"); }
   virtual real_t operator()(const Vector &x, ElementTransformation &Tr,
                             const IntegrationPoint &ip) const
   { ProcessParameters(Tr, ip); return (*this)(x); }

   // default Jacobian evaluator
   virtual ADReal_t operator()(const ADVector &x) const
   { MFEM_ABORT("Not implemented. Use MAKE_AD_FUNCTOR macro to create derived structure"); }

   // default Hessian evaluator
   virtual AD2Real_t operator()(const AD2Vector &x) const
   { MFEM_ABORT("Not implemented. Use MAKE_AD_FUNCTOR macro to create derived structure"); }

   // Evaluate the gradient, using forward mode autodiff
   virtual void Gradient(const Vector &x, ElementTransformation &Tr,
                         const IntegrationPoint &ip, Vector &J,
                         int start=0, int end=-1) const;
   virtual void Gradient(const Vector &x, Vector &J,
                         int start=0, int end=-1) const;
   // Evaluate the Hessian, using forward over forward autodiff
   // The Hessian assumed to be symmetric.
   virtual void Hessian(const Vector &x, ElementTransformation &Tr,
                        const IntegrationPoint &ip,
                        DenseMatrix &H,
                        int istart=0, int iend=-1,
                        int jstart=0, int jend=-1) const;
   virtual void Hessian(const Vector &x, DenseMatrix &H,
                        int istart=0, int iend=-1,
                        int jstart=0, int jend=-1) const;
};

// We currently only support Jacobian.
// To consistent with ADFunction, which returns
// evaluate: scalar, Gradient: vector, Hessian: matrix,
// we overrode the Gradient for evaulation, and Hessian for Jacobian
// To be used with ADNonlinearFormIntegrator or ADBlockNonlinearFormIntegrator,
// n_input and n_output must be the same.
class ADVectorFunction : public ADFunction
{
public:
   int n_output;
   ADVectorFunction(int n_input, int n_output)
      : ADFunction(n_input), n_output(n_output)
   {
      MFEM_ASSERT(n_input > 0 && n_output > 0,
                  "ADVectorFunction: n_input and n_output must be positive");
   }

   void operator()(const Vector &x, ElementTransformation &Tr,
                   const IntegrationPoint &ip,
                   Vector &F) const
   { ProcessParameters(Tr, ip); (*this)(x, F); }

   // Derived struct should implement the following methods.
   // Use AD_VEC_IMPL macro to implement them.
   virtual void operator()(const Vector &x, Vector &F) const = 0;
   virtual void operator()(const ADVector &x, ADVector &F) const = 0;
   virtual void operator()(const AD2Vector &x, AD2Vector &F) const = 0;

   void Gradient(const Vector &x, ElementTransformation &Tr,
                 const IntegrationPoint &ip, DenseMatrix &J) const
   { ProcessParameters(Tr, ip); Gradient(x, J); }

   void Gradient(const Vector &x, DenseMatrix &J) const;

   void Hessian(const Vector &x, ElementTransformation &Tr,
                const IntegrationPoint &ip,
                DenseTensor &H) const
   { ProcessParameters(Tr, ip); Hessian(x, H); }

   void Hessian(const Vector &x, DenseTensor &H) const;

   // To support ADNonlinearFormIntegrator and ADVectorNonlinearFormIntegrator
   void Gradient(const Vector &x, ElementTransformation &Tr,
                 const IntegrationPoint &ip, Vector &F, int start=0,
                 int end=-1) const override final
   {
      MFEM_VERIFY(start == 0 && end == -1,
                  "ADVectorFunction::Gradient: start and end must be 0 and -1");
      (*this)(x, Tr, ip, F);
   }

   void Gradient(const Vector &x, Vector &F,
                 int start = 0, int end = -1) const override final
   {
      MFEM_VERIFY(start == 0 && end == -1,
                  "ADVectorFunction::Gradient: start and end must be 0 and -1");
      (*this)(x, F);
   }

   // To support ADNonlinearFormIntegrator and ADVectorNonlinearFormIntegrator
   void Hessian(const Vector &x, ElementTransformation &Tr,
                const IntegrationPoint &ip,
                DenseMatrix &J,
                int istart=0, int iend=-1,
                int jstart=0, int jend=-1) const override final
   {
      MFEM_VERIFY(istart == 0 && iend == -1 && jstart == 0 && jend == -1,
                  "ADVectorFunction::Hessian: istart, iend, jstart, jend must be 0, -1, 0, -1");
      this->Gradient(x, Tr, ip, J);
   }

   void Hessian(const Vector &x, DenseMatrix &J,
                int istart=0, int iend=-1,
                int jstart=0, int jend=-1) const override final
   {
      MFEM_VERIFY(istart == 0 && iend == -1 && jstart == 0 && jend == -1,
                  "ADVectorFunction::Hessian: istart, iend, jstart, jend must be 0, -1, 0, -1");
      this->Gradient(x, J);
   }

   real_t operator()(const Vector &x) const override final
   {
      MFEM_ABORT("ADVectorFunction::operator(): This method should not be called. "
                 "Use ADVectorFunction::operator(const Vector &x, Vector &F) instead.");
   }
   ADReal_t operator()(const ADVector &x) const override final
   {
      MFEM_ABORT("ADVectorFunction::operator(): This method should not be called. "
                 "Use ADVectorFunction::operator(const ADVector &x, ADVector &F) instead.");
   }
   AD2Real_t operator()(const AD2Vector &x) const override final
   {
      MFEM_ABORT("ADVectorFunction::operator(): This method should not be called. "
                 "Use ADVectorFunction::operator(const AD2Vector &x, AD2Vector &F) instead.");
   }
};

class ADGradientFunction : public ADVectorFunction
{
protected:
public:
   ADFunction &f;
   ADGradientFunction(ADFunction& f):
      ADVectorFunction(f.n_input, f.n_input), f(f)
   {}
   void operator()(const Vector &x, Vector &F) const override
   {
      f.Gradient(x, F);
   }
   void Gradient(const Vector &x, DenseMatrix &J) const
   {
      f.Hessian(x, J);
   }
   void ProcessParameters(ElementTransformation &Tr,
                          const IntegrationPoint &ip) const override
   {
   }
};

class DifferentiableCoefficient : public Coefficient
{
private:
   int idx; // index of the next input variable

   class GradientCoefficient : public VectorCoefficient
   {
      DifferentiableCoefficient &c;
      int start, end;
   public:
      GradientCoefficient(int dim, DifferentiableCoefficient &c)
         : VectorCoefficient(dim), c(c), start(0), end(dim) { }
      void SetRange(int s, int e) { start = s; end = e; vdim = e - s; }
      void Eval(Vector &J, ElementTransformation &T,
                const IntegrationPoint &ip) override
      {
         return c.f.Gradient(c.evaluator.Eval(T, ip), T, ip, J, start, end);
      }
   };

   friend class GradientCoefficient;
   GradientCoefficient grad_cf;

   class HessianCoefficient : public MatrixCoefficient
   {
      DifferentiableCoefficient &c;
      int istart, iend, jstart, jend;
   public:
      HessianCoefficient(int dim, DifferentiableCoefficient &c)
         : MatrixCoefficient(dim), c(c), istart(0), iend(dim), jstart(0), jend(dim) { }
      void SetRange(int is, int ie, int js, int je)
      {
         istart = is; iend = ie; height = ie - is;
         jstart = js; jend = je; width = je - js;
      }
      void Eval(DenseMatrix &H, ElementTransformation &T,
                const IntegrationPoint &ip) override
      { return c.f.Hessian(c.evaluator.Eval(T, ip), T, ip, H, istart, iend, jstart, jend); }
   };

   friend class HessianCoefficient;
   HessianCoefficient hess_cf;

protected:
   Evaluator evaluator;

   ADFunction &f;
public:
   DifferentiableCoefficient(ADFunction &f)
      : f(f), idx(0)
      , grad_cf(f.n_input, *this)
      , hess_cf(f.n_input, *this)
   {}
   DifferentiableCoefficient &AddInput(Evaluator::param_t param, bool owns = false)
   {
      evaluator.Add(param, owns);
      return *this;
   }
   int GetNumInputs() const { return evaluator.val.NumBlocks(); }

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override
   { return f(evaluator.Eval(T, ip), T, ip); }

   GradientCoefficient& Gradient(int start=0, int end=-1)
   {
      if (end == -1) { end = f.n_input; }
      grad_cf.SetRange(start, end);
      return grad_cf;
   }
   HessianCoefficient& Hessian(int istart=0, int iend=-1, int jstart=0,
                               int jend=-1)
   {
      if (iend == -1) { iend = f.n_input; }
      if (jend == -1) { jend = f.n_input; }
      hess_cf.SetRange(istart, iend, jstart, jend);
      return hess_cf;
   }

protected:
};
class DifferentiableVectorCoefficient : public VectorCoefficient
{
private:
   int idx; // index of the next input variable

   class GradientCoefficient : public MatrixCoefficient
   {
      DifferentiableVectorCoefficient &c;
      Vector fval;
   public:
      GradientCoefficient(int in_dim, int out_dim, DifferentiableVectorCoefficient &c)
         : MatrixCoefficient(out_dim, in_dim), c(c), fval(out_dim) { }
      bool transpose = false;
      void Eval(DenseMatrix &J, ElementTransformation &T,
                const IntegrationPoint &ip) override
      {
         c.f.Gradient(c.evaluator.Eval(T, ip), T, ip, J);
         if (transpose) { J.Transpose(); }
      }
   };

   friend class GradientCoefficient;
   GradientCoefficient grad_cf;

protected:
   Evaluator evaluator;

   ADVectorFunction &f;
public:
   DifferentiableVectorCoefficient(ADVectorFunction &f)
      : VectorCoefficient(f.n_output), f(f), idx(0)
      , grad_cf(f.n_input, f.n_output, *this)
   {}
   DifferentiableVectorCoefficient &AddInput(Evaluator::param_t param,
                                             bool owns = false)
   {
      evaluator.Add(param, owns);
      return *this;
   }

   void Eval(Vector &fx, ElementTransformation &T,
             const IntegrationPoint &ip) override
   { f(evaluator.Eval(T, ip), T, ip, fx); }

   GradientCoefficient& Gradient(bool transpose=false)
   {
      grad_cf.transpose = transpose;
      return grad_cf;
   }

protected:
};

// Macro to generate type-varying implementation for ADFunction.
// See, DiffusionEnergy, ..., for example of usage.
// @param SCALAR is the name of templated scalar type
// @param VEC is the name of templated vector type
// @param MAT is the name of templated matrix type
// @param var is the input variable name
// @param body is the main function body. Use T() to create T-typed 0.
#define AD_IMPL(SCALAR, VEC, MAT, var, body)                                           \
   using ADFunction::operator();                                                       \
   real_t operator()(const Vector &var) const override                                 \
   {                                                                                   \
      MFEM_ASSERT(var.Size() == n_input,                                               \
                 "ADFunction::operator(): var.Size()=" << var.Size()                   \
                  <<  " must match n_input=" << n_input)                               \
      using SCALAR = real_t;                                                           \
      using VEC = Vector;                                                              \
      using MAT = DenseMatrix;                                                         \
      body                                                                             \
   }                                                                                   \
                                                                                       \
   ADReal_t operator()(const ADVector &var) const override                             \
   {                                                                                   \
      MFEM_ASSERT(var.Size() == n_input,                                               \
                 "ADFunction::operator(): var.Size()=" << var.Size()                   \
                  <<  " must match n_input=" << n_input)                               \
      using SCALAR = ADReal_t;                                                         \
      using VEC = ADVector;                                                            \
      using MAT = ADMatrix;                                                            \
      body                                                                             \
   }                                                                                   \
                                                                                       \
   AD2Real_t operator()(const AD2Vector &var) const override                           \
   {                                                                                   \
      MFEM_ASSERT(var.Size() == n_input,                                               \
                 "ADFunction::operator(): var.Size()=" << var.Size()                   \
                  <<  " must match n_input=" << n_input)                               \
      using SCALAR = AD2Real_t;                                                        \
      using VEC = AD2Vector;                                                           \
      using MAT = AD2Matrix;                                                           \
      body                                                                             \
   }


// Macro to generate type-varying implementation for ADVectorFunction.
// @param SCALAR is the name of templated scalar type
// @param VEC is the name of templated vector type
// @param MAT is the name of templated matrix type
// @param var is the input variable name
// @param result is the output variable name
// @param body is the main function body. Use T() to create T-typed 0.
#define AD_VEC_IMPL(SCALAR, VEC, MAT, var, result, body)                               \
   using ADVectorFunction::operator();                                                 \
   using ADVectorFunction::Gradient;                                                   \
   using ADVectorFunction::Hessian;                                                    \
                                                                                       \
   void operator()(const Vector &var, Vector &result) const override                   \
   {                                                                                   \
      MFEM_ASSERT(var.Size() == n_input,                                               \
                 "ADFunction::operator(): var.Size()=" << var.Size()                   \
                  <<  " must match n_input=" << n_input)                               \
      using SCALAR = real_t;                                                           \
      using VEC = Vector;                                                              \
      using MAT = DenseMatrix;                                                         \
      body                                                                             \
   }                                                                                   \
                                                                                       \
   void operator()(const ADVector &var, ADVector &result) const override               \
   {                                                                                   \
      MFEM_ASSERT(var.Size() == n_input,                                               \
                 "ADFunction::operator(): var.Size()=" << var.Size()                   \
                  <<  " must match n_input=" << n_input)                               \
      using SCALAR = ADReal_t;                                                         \
      using VEC = ADVector;                                                            \
      using MAT = ADMatrix;                                                            \
      body                                                                             \
   }                                                                                   \
                                                                                       \
   void operator()(const AD2Vector &var, AD2Vector &result) const override             \
   {                                                                                   \
      MFEM_ASSERT(var.Size() == n_input,                                               \
                 "ADFunction::operator(): var.Size()=" << var.Size()                   \
                  <<  " must match n_input=" << n_input)                               \
      using SCALAR = AD2Real_t;                                                        \
      using VEC = AD2Vector;                                                           \
      using MAT = AD2Matrix;                                                           \
      body                                                                             \
   }

class MassEnergy : public ADFunction
{
public:
   MassEnergy(int n_var)
      : ADFunction(n_var)
   {}
   AD_IMPL(T, V, M, x, return 0.5*(x*x););
};
class DiffusionEnergy : public ADFunction
{
   const int dim;
   mutable const Vector *K;
public:
   DiffusionEnergy(int dim)
      : ADFunction(dim), dim(dim)
   {}
   DiffusionEnergy(int dim, Evaluator::param_t K)
      : DiffusionEnergy(dim)
   { SetK(K); }

   void ProcessParameters(ElementTransformation &Tr,
                          const IntegrationPoint &ip) const override
   { K = &evaluator.Eval(Tr, ip); }

   void SetK(Evaluator::param_t param)
   {
      int i = AddParameter(param);
      int size = evaluator.val.GetBlock(i).Size();
      MFEM_VERIFY(size == 1 || size == n_input || size == n_input*n_input,
                  "Incorrect size for K. Dimension is " << n_input << "but K has size " << size);
   }

   AD_IMPL(T, V, M, gradu,
   {
      const int dim = gradu.Size();
      const int Kdim = K->Size();
      // No diffusion coefficient, ||grad u||^2
      if (Kdim == 0) { return 0.5*(gradu*gradu); }
      // Scalar diffusion coefficient, ||K^{1/2} grad u||^2
      if (Kdim == 1) { return 0.5*(*K)[0]*(gradu*gradu); }
      // Vector diffusion coefficient, ||diag(K)^{1/2} grad u||^2
      if (Kdim == dim)
      {
         T result = T();
         for (int i=0; i<dim; i++)
         {
            result += (*K)[i]*gradu[i]*gradu[i];
         }
         return 0.5*result;
      }
      // Matrix diffusion coefficient, ||K^{1/2} grad u||^2
      if (Kdim == dim*dim)
      {
         DenseMatrix Kmat(K->GetData(), dim, dim);
         T result = T();
         for (int j=0; j<dim; j++)
         {
            for (int i=0; i<dim; i++)
            {
               result += Kmat(i,j)*gradu[i]*gradu[j];
            }
         }
         return 0.5*result;
      }
      MFEM_ABORT("DiffusionEnergy: K must be a scalar, vector of size dim, "
                 "or matrix of size dim x dim");
      return T();
   });
};

class DiffEnergy : public ADFunction
{
   const ADFunction &energy;
   mutable const Vector *target;
public:
   DiffEnergy(const ADFunction &energy)
      : ADFunction(energy.n_input)
      , energy(energy)
   { }

   DiffEnergy(const ADFunction &energy, Evaluator::param_t other)
      : DiffEnergy(energy)
   {
      int i = AddParameter(other);
      MFEM_VERIFY(evaluator.val.GetBlock(0).Size() == n_input,
                  "DiffEnergy: The provided target has the wrong size. "
                  "Expected " << n_input << ", got " << evaluator.val.GetBlock(0).Size());
   }

   void SetTarget(Evaluator::param_t &target)
   {
      if (evaluator.val.NumBlocks() == 1)
      { evaluator.Replace(0, target); }
      else
      { evaluator.Add(target); }
      MFEM_VERIFY(evaluator.val.GetBlock(0).Size() == n_input,
                  "DiffEnergy: The provided target has the wrong size. "
                  "Expected " << n_input << ", got " << evaluator.val.GetBlock(0).Size());
   }

   void ProcessParameters(const BlockVector &x) const override
   {
      target = &x.GetBlock(0);
   }

   AD_IMPL(T, V, M, x,
   {
      V diff(x);
      for (int i=0; i<n_input; i++)
      { diff[i] -= (*target)[i]; }
      return energy(diff);
   });
};

class LinearElasticityEnergy : public ADFunction
{
   const int dim;
   real_t &lambda;
   real_t &mu;
public:
   void ProcessParameters(ElementTransformation &Tr,
                          const IntegrationPoint &ip) const override
   {
      evaluator.Eval(Tr, ip);
   }
   LinearElasticityEnergy(int dim, Evaluator::param_t lambda,
                          Evaluator::param_t mu, int offset=0)
      : ADFunction(dim*dim, 2)
      , dim(dim)
      , lambda(*(evaluator.val.GetData() + offset))
      , mu(*(evaluator.val.GetData() + evaluator.GetSize(mu) + offset))
   {
      int lambda_idx = evaluator.Add(lambda);
      int mu_idx = evaluator.Add(mu);
      MFEM_VERIFY(lambda_idx == 0,
                  "LinearElasticityEnergy: lambda must be the first parameter");
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

// Lagrangian functional
// f(x) + sum lambda[i]*c[i](x)
class Lagrangian : public ADFunction
{
private:
   enum { OBJONLY=-2, FULL=-1, CON=0};
   int eval_mode =
      FULL; // -2: objective, -1: full Lagrangian, >=0: constraint comp

   ADFunction &objective; // f(x)

   std::vector<ADFunction*> eq_con; // c[i](x)
   Vector eq_rhs; // c[i](x) = con_target[i]
public:

   Lagrangian(ADFunction &objective, const int n_eq_con)
      : ADFunction(objective.n_input+n_eq_con)
      , objective(objective)
   {}

   Lagrangian AddEqConstraint(ADFunction &constraint,
                              real_t target = 0.0);
   Lagrangian SetEqRHS(int idx, real_t target) { eq_rhs[idx] = target; return *this; }

   // return f(x) + sum lambda[i]*c[i](x)
   void FullMode() { this->eval_mode = FULL; }
   // return f(x)
   void ObjectiveMode() { this->eval_mode = OBJONLY; }
   // return c[i](x)
   void EqConstraintMode(int comp)
   {
      MFEM_VERIFY(comp >= 0 && comp < eq_con.size(),
                  "ALFunctional: comp must be in [0, n_input)");
      this->eval_mode = comp;
   }

   void ProcessParameters(ElementTransformation &Tr,
                          const IntegrationPoint &ip) const override;

   AD_IMPL(T, V, M, x_and_lambda,
   {
      const V x(x_and_lambda.GetData(), objective.n_input);
      const V lambda(x_and_lambda.GetData() + objective.n_input,
                     eq_con.size());
      if (eval_mode >= 0) { return (*eq_con[eval_mode])(x); }

      T result = objective(x);
      if (eval_mode == OBJONLY) { return result; } // only objective
      for (int i=0; i<eq_con.size(); i++) { result += (*eq_con[i])(x)*lambda[i]; }
      return result;
   });
};

// Augmented Lagrangian functional
class ALFunctional : public ADFunction
{
private:
   enum { OBJONLY=-2, FULLAL=-1, CON=0};
   int al_eval_mode = FULLAL; // -2: objective, -1: full AL, >=0: constraint comp

   ADFunction &objective; // f(x)

   std::vector<ADFunction*> eq_con; // c[i](x)
   Vector eq_rhs; // c[i](x) = con_target[i]
   Vector lambda; // Lagrange multipliers
   real_t penalty=1.0; // penalty
public:

   ALFunctional(ADFunction &objective)
      : ADFunction(objective.n_input)
      , objective(objective)
   {}

   ALFunctional AddEqConstraint(ADFunction &constraint,
                                real_t target = 0.0);
   ALFunctional SetEqRHS(int idx, real_t target) { eq_rhs[idx] = target; return *this; }

   void SetLambda(const Vector &lambda);
   const Vector &GetLambda() const { return lambda; }
   Vector &GetLambda() { return lambda; }

   void SetPenalty(real_t mu);
   real_t GetPenalty() const {return penalty; }
   real_t &GetPenalty() { return penalty; }

   // Full AL mode: f(x) + sum lambda[i]*c[i](x) + mu/2 * sum c[i](x)^2
   void ALMode() { this->al_eval_mode = FULLAL; }
   // Objective mode: f(x)
   void ObjectiveMode() { this->al_eval_mode = OBJONLY; }
   // Constraint mode: c[i](x)
   void EqConstraintMode(int comp)
   {
      MFEM_VERIFY(comp >= 0 && comp < eq_con.size(),
                  "ALFunctional: comp must be in [0, n_input)");
      this->al_eval_mode = comp;
   }

   void ProcessParameters(ElementTransformation &Tr,
                          const IntegrationPoint &ip) const override;

   AD_IMPL(T, V, M, x,
   {
      if (al_eval_mode >= 0) { return evalAL<T>(x, al_eval_mode); }

      T result = objective(x);
      if (al_eval_mode == OBJONLY) { return result; } // only objective

      for (int i=0; i<eq_con.size(); i++) { result += evalAL<T>(x, i); }

      return result;
   });

private:
   // Evaluate lambda*c(x) + (mu/2)*c(x)^2
   template <typename T, typename V>
   T evalAL(V &x, int idx) const
   {
      T cx = (*eq_con[idx])(x) - eq_rhs[idx];
      if (al_eval_mode >= 0) { return cx; } // if non-negative, only c(x)
      return cx*(lambda[idx] + penalty*0.5*cx);
   }
};

/// @brief ADFunction with reduced input by fixing a portion of the input
/// Currently, we only support [0, offset) or [offset, n_input)
class ReducedADFunction : public ADFunction
{
private:
   ADFunction &f;
   Evaluator fixed_input;
   mutable Vector full_J;
   mutable DenseMatrix full_H;
   const int offset;
   bool fix_after;
public:
   void ProcessParameters(ElementTransformation &Tr,
                          const IntegrationPoint &ip) const override
   {
      f.ProcessParameters(Tr, ip); // process paraent's parameters
      fixed_input.Eval(Tr, ip); // process fixed input
   }
   void ProcessParameters(const BlockVector &param_val) const override
   { f.ProcessParameters(param_val); }
   ReducedADFunction(ADFunction &org_f,
                     std::initializer_list<Evaluator::param_t> fixed_srcs,
                     int offset_, bool fix_after_)
      : ADFunction(fix_after_ ? offset_ : org_f.n_input - offset_)
      , f(org_f)
      , offset(offset_)
      , fix_after(fix_after_)
   {
      for (auto &src : fixed_srcs)
      {
         fixed_input.Add(src);
      }
      MFEM_VERIFY(fixed_input.val.Size() + n_input == f.n_input,
                  "ReducedADFunction: fixed input size + n_input must equal to original function input size");
   }
   AD_IMPL(T, V, M, x,
   {
      V full_x(f.n_input);
      if (fix_after)
      {
         for (int i=0; i<n_input; i++)
         { full_x[i] = x[i]; }
         for (int i=0; i<fixed_input.val.Size(); i++)
         { full_x[i+offset] = fixed_input.val[i]; }
      }
      else
      {
         for (int i=0; i<fixed_input.val.Size(); i++)
         { full_x[i] = fixed_input.val[i]; }
         for (int i=0; i<n_input; i++)
         { full_x[i+offset] = x[i]; }
      }
      return f(full_x);
   });
   void Gradient(const Vector &x, Vector &J, int start=0,
                 int end=-1) const override
   {
      if (end == -1) { end = n_input; }
      MFEM_VERIFY(start != 0 && end != n_input,
                  "ReducedADFunction::Gradient: Partial gradient not implemented yet");
      Vector full_x(f.n_input);
      if (fix_after)
      {
         for (int i=0; i<n_input; i++)
         { full_x[i] = x[i]; }
         for (int i=0; i<fixed_input.val.Size(); i++)
         { full_x[i+offset] = fixed_input.val[i]; }
      }
      else
      {
         for (int i=0; i<fixed_input.val.Size(); i++)
         { full_x[i] = fixed_input.val[i]; }
         for (int i=0; i<n_input; i++)
         { full_x[i+offset] = x[i]; }
      }
      f.Gradient(full_x, J, fix_after ? 0 : offset, fix_after ? offset : f.n_input);
   }
   void Hessian(const Vector &x, DenseMatrix &H,
                int istart=0, int iend=-1,
                int jstart=0, int jend=-1) const override
   {
      if (iend == -1) { iend = n_input; }
      if (jend == -1) { jend = n_input; }
      MFEM_VERIFY(istart != 0 && iend != n_input && jstart != 0 && jend != n_input,
                  "ReducedADFunction::Hessian: Partial Hessian not implemented yet");
      Vector full_x(f.n_input);
      if (fix_after)
      {
         for (int i=0; i<n_input; i++)
         { full_x[i] = x[i]; }
         for (int i=0; i<fixed_input.val.Size(); i++)
         { full_x[i+offset] = fixed_input.val[i]; }
      }
      else
      {
         for (int i=0; i<fixed_input.val.Size(); i++)
         { full_x[i] = fixed_input.val[i]; }
         for (int i=0; i<n_input; i++)
         { full_x[i+offset] = x[i]; }
      }
      int start = fix_after ? 0 : offset;
      int end = fix_after ? offset : f.n_input;
      f.Hessian(full_x, H, start, end, start, end);
   }
};

/// @brief ADFunction with [state, parameters] input, providing methods to
/// get reduced functions for state or parameters only.
/// @see, ReducedADFunction, ParametrizedADFunction::GetStateFunction,
/// ParametrizedADFunction::GetParamFunction
class ParametrizedADFunction : public ADFunction
{
protected:
   const int parameter_begin;
public:
   ParametrizedADFunction(int n_input, int param_begin)
      : ADFunction(n_input), parameter_begin(param_begin)
   {
      MFEM_ASSERT(n_input > 0, "ADFunction: n_input must be positive");
      MFEM_ASSERT(param_begin >= 0 && param_begin < n_input,
                  "ADFunction: param_begin must be in [0, n_input)");
   }

   std::unique_ptr<ReducedADFunction> GetStateFunction(
      std::initializer_list<Evaluator::param_t> param_srcs)
   {
      return std::make_unique<ReducedADFunction>(*this, param_srcs,
                                                 n_input, parameter_begin);
   }

   std::unique_ptr<ReducedADFunction> GetParamFunction(
      std::initializer_list<Evaluator::param_t> state_srcs)
   {
      return std::make_unique<ReducedADFunction>(*this, state_srcs,
                                                 n_input, 0);
   }
};

/// Linear elasticity energy with parameterized Young's modulus
class ParametrizedLinearElasticityEnergy : public ParametrizedADFunction
{
   const int dim;
   const real_t nu; // fixed Poisson's ratio
public:
   ParametrizedLinearElasticityEnergy(int dim, const real_t nu)
      : ParametrizedADFunction(dim*dim + 1, dim*dim)
      , dim(dim)
      , nu(nu)
   {}
   AD_IMPL(T, V, M, gradu_and_E,
   {
      const V gradu(gradu_and_E.GetData(), dim*dim);
      const T E = gradu_and_E[dim*dim];
      const T lambda = E*nu / ((1 + nu)*(1 - 2*nu));
      const T mu = E / (2*(1 + nu));
      T divnorm = T();
      for (int i=0; i<dim; i++) { divnorm += gradu[i*dim + i]; }
      divnorm = divnorm*divnorm;
      T symm_h1_norm = T();
      for (int i=0; i<dim; i++)
      {
         for (int j=0; j<dim; j++)
         {
            T symm = 0.5*(gradu[i*dim + j] + gradu[j*dim + i]);
            symm_h1_norm += symm*symm;
         }
      }
      return 0.5*lambda*divnorm + mu*symm_h1_norm;
   })
};

class ParametrizedIsotropicDiffusionEnergy : public ParametrizedADFunction
{
   const int dim;
public:
   ParametrizedIsotropicDiffusionEnergy(int dim)
      : ParametrizedADFunction(dim + 1, dim)
      , dim(dim)
   {}
   AD_IMPL(T, V, M, gradu_and_k,
   {
      const V gradu(gradu_and_k.GetData(), dim);
      const T k = gradu_and_k[dim];
      T result = T();
      for (int i=0; i<dim; i++)
      {
         result += gradu[i]*gradu[i];
      }
      return 0.5*k*result;
   })
};
// ------------------------------------------------------------------------------
// Implement dual max/min
// ------------------------------------------------------------------------------
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
}
