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
                   real_t, Vector, DenseMatrix,
                   const real_t*, const Vector*, const DenseMatrix*,
                   Coefficient*, VectorCoefficient*, MatrixCoefficient*,
                   const GridFunction*, const QuadratureFunction*>;
private:
   Array<int> offsets;
   std::vector<param_t> params;

   mutable Vector loc_vec_val;
   mutable DenseMatrix loc_mat_val;

   int GetSize(const param_t &param) const;

public:
   mutable BlockVector val;
   Evaluator(): offsets{0} {}
   Evaluator(int capacity)
      : offsets{0}
   {
      val.SetSize(capacity);
      val.SetSize(0);
   }
   // Add a parameter to the evaluator
   int Add(param_t &param);
   // Replace a parameter at index i with a new parameter
   // The output size of param should match the size of the old parameter
   void Replace(size_t i, param_t &param);

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
};

struct ADFunction
{
protected:
   Evaluator evaluator;

   int AddParameter(Evaluator::param_t param)
   { return evaluator.Add(param); }

   void ReplaceParameter(int i, Evaluator::param_t param)
   { evaluator.Replace(i, param); }

public:
   virtual void ProcessParameters(ElementTransformation &Tr,
                                  const IntegrationPoint &ip) const
   { evaluator.Eval(Tr, ip); }

   int n_input;
   ADFunction(int n_input)
      : n_input(n_input) { }
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
                         const IntegrationPoint &ip, Vector &J) const;
   virtual void Gradient(const Vector &x, Vector &J) const;
   // Evaluate the Hessian, using forward over forward autodiff
   // The Hessian assumed to be symmetric.
   virtual void Hessian(const Vector &x, ElementTransformation &Tr,
                        const IntegrationPoint &ip,
                        DenseMatrix &H) const;
   virtual void Hessian(const Vector &x, DenseMatrix &H) const;
};

// We currently only support Jacobian.
// To consistent with ADFunction, which returns
// evaluate: scalar, Gradient: vector, Hessian: matrix,
// we overrode the Gradient for evaulation, and Hessian for Jacobian
// To be used with ADNonlinearFormIntegrator or ADBlockNonlinearFormIntegrator,
// n_input and n_output must be the same.
struct ADVectorFunction : public ADFunction
{

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
                 const IntegrationPoint &ip, Vector &F) const override final
   { (*this)(x, Tr, ip, F); }

   void Gradient(const Vector &x, Vector &F) const override final
   { (*this)(x, F); }

   // To support ADNonlinearFormIntegrator and ADVectorNonlinearFormIntegrator
   void Hessian(const Vector &x, ElementTransformation &Tr,
                const IntegrationPoint &ip,
                DenseMatrix &J) const override final
   { this->Gradient(x, Tr, ip, J); }

   void Hessian(const Vector &x, DenseMatrix &J) const override final
   { this->Gradient(x, J); }

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

class DifferentiableCoefficient : public Coefficient
{
private:
   int idx; // index of the next input variable

   class GradientCoefficient : public VectorCoefficient
   {
      DifferentiableCoefficient &c;
   public:
      GradientCoefficient(int dim, DifferentiableCoefficient &c)
         : VectorCoefficient(dim), c(c) { }
      void Eval(Vector &J, ElementTransformation &T,
                const IntegrationPoint &ip) override
      {
         const BlockVector &x = c.EvalInput(T, ip);
         c.f.Gradient(x, T, ip, J);
      }
   };

   friend class GradientCoefficient;
   GradientCoefficient grad_cf;

   class HessianCoefficient : public MatrixCoefficient
   {
      DifferentiableCoefficient &c;
   public:
      HessianCoefficient(int dim, DifferentiableCoefficient &c)
         : MatrixCoefficient(dim), c(c) { }
      void Eval(DenseMatrix &H, ElementTransformation &T,
                const IntegrationPoint &ip) override
      {
         const BlockVector &x = c.EvalInput(T, ip);
         c.f.Hessian(x, T, ip, H);
      }
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
   DifferentiableCoefficient &AddInput(Evaluator::param_t param)
   { evaluator.Add(param); return *this; }

   real_t Eval(ElementTransformation &T,
               const IntegrationPoint &ip) override
   {
      const BlockVector &x = EvalInput(T, ip);
      return f(x, T, ip);
   }

   GradientCoefficient& Gradient() { return grad_cf; }
   HessianCoefficient& Hessian() { return hess_cf; }

protected:
   const BlockVector& EvalInput(
      ElementTransformation &T,
      const IntegrationPoint &ip) const
   { return evaluator.Eval(T, ip); }
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

struct MassEnergy : public ADFunction
{
   MassEnergy(int n_var)
      : ADFunction(n_var)
   {}
   AD_IMPL(T, V, M, x, return 0.5*(x*x););
};
struct DiffusionEnergy : public ADFunction
{
   const int dim;
   mutable const Vector *K;
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

struct DiffEnergy : public ADFunction
{
   const ADFunction &energy;
   mutable const Vector *target;
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
      {
         evaluator.Replace(0, target);
      }
      else
      {
         evaluator.Add(target);
      }
      MFEM_VERIFY(evaluator.val.GetBlock(0).Size() == n_input,
                  "DiffEnergy: The provided target has the wrong size. "
                  "Expected " << n_input << ", got " << evaluator.val.GetBlock(0).Size());
   }

   void ProcessParameters(ElementTransformation &Tr,
                          const IntegrationPoint &ip) const override
   {
      MFEM_ASSERT(other != nullptr,
                  "DiffEnergy: other is not set. Use SetTarget() to set it.");
      energy.ProcessParameters(Tr, ip);
      target = &evaluator.Eval(Tr, ip);
   }

   AD_IMPL(T, V, M, x,
   {
      V diff(x);
      for (int i=0; i<n_input; i++)
      { diff[i] -= (*target)[i]; }
      return energy(diff);
   });
};

struct LinearElasticityEnergy : public ADFunction
{
   const int dim;
   real_t &lambda;
   real_t &mu;
   void ProcessParameters(ElementTransformation &Tr,
                          const IntegrationPoint &ip) const override
   {
      evaluator.Eval(Tr, ip);
   }
   LinearElasticityEnergy(int dim, Evaluator::param_t lambda,
                          Evaluator::param_t mu)
      : ADFunction(dim*dim, 2)
      , dim(dim)
      , lambda(*evaluator.val.GetData())
      , mu(*(evaluator.val.GetData()+1))
   {
      evaluator.Add(lambda);
      evaluator.Add(mu);
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
struct Lagrangian : public ADFunction
{
private:
   enum { OBJONLY=-2, FULL=-1, CON=0};
   int eval_mode =
      FULL; // -2: objective, -1: full Lagrangian, >=0: constraint comp

   ADFunction &objective; // f(x)

   std::vector<ADFunction*> eq_con; // c[i](x)
   Vector eq_rhs; // c[i](x) = con_target[i]
public:

   Lagrangian(ADFunction &objective)
      : ADFunction(objective.n_input)
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

private:
};

// Augmented Lagrangian functional
struct ALFunctional : public ADFunction
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
// ----------------------------------------------------------------
// Operator overloading for ADFunction arithmetics
// Using shared_ptr to manage memory.
// Not ideal, but good enough for now.
// ----------------------------------------------------------------
struct ProductADFunction : public ADFunction
{
   const std::shared_ptr<ADFunction> f1;
   const std::shared_ptr<ADFunction> f2;

   ProductADFunction(const std::shared_ptr<ADFunction> &f1,
                     const std::shared_ptr<ADFunction> &f2)
      : ADFunction(0), f1(f1), f2(f2)
   {
      MFEM_ASSERT(f1.get() != nullptr && f2.get() != nullptr,
                  "ProductADFunction: f1 and f2 must not be null");
      MFEM_ASSERT(f1->n_input == f2->n_input,
                  "ProductADFunction: f1 and f2 must have the same n_input");
      n_input = f1->n_input; // Set n_input to the common input size
   }

   void ProcessParameters(ElementTransformation &Tr,
                          const IntegrationPoint &ip) const override
   {
      f1->ProcessParameters(Tr, ip);
      f2->ProcessParameters(Tr, ip);
   }

   real_t operator()(const Vector &x) const override
   { return (*f1)(x)*(*f2)(x); }
   real_t operator()(const Vector &x, ElementTransformation &Tr,
                     const IntegrationPoint &ip) const override
   { return (*f1)(x, Tr, ip) * (*f2)(x, Tr, ip); }
   ADReal_t operator()(const ADVector &x) const override
   { return (*f1)(x) * (*f2)(x); }
   AD2Real_t operator()(const AD2Vector &x) const override
   { return (*f1)(x) * (*f2)(x); }
};
inline std::shared_ptr<ADFunction>
operator*(const std::shared_ptr<ADFunction> &f1,
          const std::shared_ptr<ADFunction> &f2)
{ return std::make_shared<ProductADFunction>(f1, f2); }

struct ScaledADFunction : public ADFunction
{
   const std::shared_ptr<ADFunction> f1;
   real_t a;

   ScaledADFunction(const std::shared_ptr<ADFunction> &f1,
                    real_t a)
      : ADFunction(0), f1(f1), a(a)
   {
      MFEM_ASSERT(f1.get() != nullptr,
                  "ProductADFunction: f1 and f2 must not be null");
      n_input = f1->n_input; // Set n_input to the common input size
   }

   void ProcessParameters(ElementTransformation &Tr,
                          const IntegrationPoint &ip) const override
   {
      f1->ProcessParameters(Tr, ip);
   }

   real_t operator()(const Vector &x) const override
   { return (*f1)(x)*a; }
   real_t operator()(const Vector &x, ElementTransformation &Tr,
                     const IntegrationPoint &ip) const override
   { return (*f1)(x, Tr, ip) * a; }
   ADReal_t operator()(const ADVector &x) const override
   { return (*f1)(x) * a; }
   AD2Real_t operator()(const AD2Vector &x) const override
   { return (*f1)(x) * a; }
};
inline std::shared_ptr<ADFunction>
operator*(const std::shared_ptr<ADFunction> &f1,
          real_t a)
{ return std::make_shared<ScaledADFunction>(f1, a); }
inline std::shared_ptr<ADFunction>
operator*(real_t a,
          const std::shared_ptr<ADFunction> &f1)
{ return std::make_shared<ScaledADFunction>(f1, a); }
inline std::shared_ptr<ADFunction>
operator/(const std::shared_ptr<ADFunction> &f1,
          real_t a)
{ return std::make_shared<ScaledADFunction>(f1, 1.0/a); }

struct SumADFunction : public ADFunction
{
   const std::shared_ptr<ADFunction> f1;
   const std::shared_ptr<ADFunction> f2;
   real_t b; // scaling factor for f2

   SumADFunction(const std::shared_ptr<ADFunction> &f1,
                 const std::shared_ptr<ADFunction> &f2, real_t b)
      : ADFunction(0), f1(f1), f2(f2)
   {
      MFEM_ASSERT(f1.get() != nullptr && f2.get() != nullptr,
                  "ProductADFunction: f1 and f2 must not be null");
      MFEM_ASSERT(f1->n_input == f2->n_input,
                  "ProductADFunction: f1 and f2 must have the same n_input");
      n_input = f1->n_input; // Set n_input to the common input size
   }

   void ProcessParameters(ElementTransformation &Tr,
                          const IntegrationPoint &ip) const override
   {
      f1->ProcessParameters(Tr, ip);
      f2->ProcessParameters(Tr, ip);
   }

   real_t operator()(const Vector &x) const override
   { return (*f1)(x) + b*(*f2)(x); }
   real_t operator()(const Vector &x, ElementTransformation &Tr,
                     const IntegrationPoint &ip) const override
   { return (*f1)(x, Tr, ip) + b*(*f2)(x, Tr, ip); }
   ADReal_t operator()(const ADVector &x) const override
   { return (*f1)(x) + b*(*f2)(x); }
   AD2Real_t operator()(const AD2Vector &x) const override
   { return (*f1)(x) + b* (*f2)(x); }
};
inline std::shared_ptr<ADFunction>
operator+(const std::shared_ptr<ADFunction> &f1,
          const std::shared_ptr<ADFunction> &f2)
{ return std::make_shared<SumADFunction>(f1, f2, 1.0); }
inline std::shared_ptr<ADFunction>
operator-(const std::shared_ptr<ADFunction> &f1,
          const std::shared_ptr<ADFunction> &f2)
{ return std::make_shared<SumADFunction>(f1, f2, -1.0); }

struct ShiftedADFunction : public ADFunction
{
   const std::shared_ptr<ADFunction> f1;
   real_t b; // scaling factor for f2

   ShiftedADFunction(const std::shared_ptr<ADFunction> &f1, real_t b)
      : ADFunction(0), f1(f1), b(b)
   {
      MFEM_ASSERT(f1.get() != nullptr,
                  "ProductADFunction: f1 must not be null");
      n_input = f1->n_input; // Set n_input to the common input size
   }

   void ProcessParameters(ElementTransformation &Tr,
                          const IntegrationPoint &ip) const override
   {
      f1->ProcessParameters(Tr, ip);
   }

   real_t operator()(const Vector &x) const override
   { return (*f1)(x) + b; }
   real_t operator()(const Vector &x, ElementTransformation &Tr,
                     const IntegrationPoint &ip) const override
   { return (*f1)(x, Tr, ip) + b; }
   ADReal_t operator()(const ADVector &x) const override
   { return (*f1)(x) + b; }
   AD2Real_t operator()(const AD2Vector &x) const override
   { return (*f1)(x) + b; }
};
inline std::shared_ptr<ADFunction>
operator+(const std::shared_ptr<ADFunction> &f1, real_t b)
{ return std::make_shared<ShiftedADFunction>(f1, b); }
inline std::shared_ptr<ADFunction>
operator+(real_t b,const std::shared_ptr<ADFunction> &f1)
{ return std::make_shared<ShiftedADFunction>(f1, b); }

inline std::shared_ptr<ADFunction>
operator-(const std::shared_ptr<ADFunction> &f1, real_t b)
{ return std::make_shared<ShiftedADFunction>(f1, -b); }

struct QuatiendADFunction : public ADFunction
{
   const std::shared_ptr<ADFunction> f1;
   const std::shared_ptr<ADFunction> f2;

   QuatiendADFunction(const std::shared_ptr<ADFunction> &f1,
                      const std::shared_ptr<ADFunction> &f2)
      : ADFunction(0), f1(f1), f2(f2)
   {
      MFEM_ASSERT(f1.get() != nullptr && f2.get() != nullptr,
                  "QuatiendADFunction: f1 and f2 must not be null");
      MFEM_ASSERT(f1->n_input == f2->n_input,
                  "QuatiendADFunction: f1 and f2 must have the same n_input");
      n_input = f1->n_input; // Set n_input to the common input size
   }

   void ProcessParameters(ElementTransformation &Tr,
                          const IntegrationPoint &ip) const override
   {
      f1->ProcessParameters(Tr, ip);
      f2->ProcessParameters(Tr, ip);
   }

   real_t operator()(const Vector &x) const override
   { return (*f1)(x) / (*f2)(x); }
   real_t operator()(const Vector &x, ElementTransformation &Tr,
                     const IntegrationPoint &ip) const override
   { return (*f1)(x, Tr, ip) / (*f2)(x, Tr, ip); }
   ADReal_t operator()(const ADVector &x) const override
   { return (*f1)(x) / (*f2)(x); }
   AD2Real_t operator()(const AD2Vector &x) const override
   { return (*f1)(x) / (*f2)(x); }
};
inline std::shared_ptr<ADFunction>
operator/(const std::shared_ptr<ADFunction> &f1,
          const std::shared_ptr<ADFunction> &f2)
{ return std::make_shared<QuatiendADFunction>(f1, f2); }

struct ReciprocalADFunction : public ADFunction
{
   const std::shared_ptr<ADFunction> f1;
   real_t a;

   ReciprocalADFunction(const std::shared_ptr<ADFunction> &f1, real_t a)
      : ADFunction(0), f1(f1)
   {
      MFEM_ASSERT(f1.get() != nullptr,
                  "ReciprocalADFunction: f1 must not be null");
      n_input = f1->n_input; // Set n_input to the common input size
   }

   void ProcessParameters(ElementTransformation &Tr,
                          const IntegrationPoint &ip) const override
   {
      f1->ProcessParameters(Tr, ip);
   }

   real_t operator()(const Vector &x) const override
   { return a / (*f1)(x); }
   real_t operator()(const Vector &x, ElementTransformation &Tr,
                     const IntegrationPoint &ip) const override
   { return a / (*f1)(x, Tr, ip); }
   ADReal_t operator()(const ADVector &x) const override
   { return a / (*f1)(x); }
   AD2Real_t operator()(const AD2Vector &x) const override
   { return a / (*f1)(x); }
};
inline std::shared_ptr<ADFunction>
operator/(real_t a, const std::shared_ptr<ADFunction> &f1)
{ return std::make_shared<ReciprocalADFunction>(f1, a); }

struct ReferenceConstantADFunction : public ADFunction
{
   real_t &a;

   ReferenceConstantADFunction(real_t &a, int n_input)
      : ADFunction(n_input)
      , a(a)
   { }
   real_t operator()(const Vector &x, ElementTransformation &Tr,
                     const IntegrationPoint &ip) const override
   { return a; }

   // default Jacobian evaluator
   real_t operator()(const Vector &x) const override
   { return a; }
   ADReal_t operator()(const ADVector &x) const override
   { return ADReal_t{a, 0.0}; }

   // default Hessian evaluator
   AD2Real_t operator()(const AD2Vector &x) const override
   { return AD2Real_t{a, 0.0}; }

   void Gradient(const Vector &x, Vector &J) const override
   {
      J.SetSize(x.Size());
      J = 0.0; // Gradient is zero for constant function
   }
   void Hessian(const Vector &x, DenseMatrix &H) const override
   {
      H.SetSize(x.Size(), x.Size());
      H = 0.0; // Hessian is zero for constant function
   }
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
