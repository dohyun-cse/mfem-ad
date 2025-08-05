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
#define MAKE_AD_FUNCTION(name, SCALAR, VEC, MAT, var, param, body)            \
struct name : public ADFunction                                               \
{                                                                             \
   name(int n, int n_param=0)                                                 \
      : ADFunction(n, n_param) { }                                            \
                                                                              \
   real_t operator()(const Vector &var, const Vector &param) const            \
   {                                                                          \
      MFEM_ASSERT(var.Size() == n_input,                                      \
                 "ADFunction::operator(): var.Size() must match n_input")     \
      MFEM_ASSERT(param.Size() == n_param,                                    \
                 "ADFunction::operator(): var.Size() must match n_input")     \
      using SCALAR = real_t;                                                 \
      using VEC = Vector;                                                     \
      using MAT = DenseMatrix;                                                \
      body                                                                    \
   }                                                                          \
                                                                              \
   ADReal_t operator()(const ADVector &var, const Vector &param) const        \
   {                                                                          \
      MFEM_ASSERT(var.Size() == n_input,                                      \
                 "ADFunction::operator(): var.Size() must match n_input")     \
      MFEM_ASSERT(param.Size() == n_param,                                    \
                 "ADFunction::operator(): var.Size() must match n_input")     \
      using SCALAR = ADReal_t;                                                \
      using VEC = ADVector;                                                   \
      using MAT = ADMatrix;                                                   \
      body                                                                    \
   }                                                                          \
                                                                              \
   AD2Real_t operator()(const AD2Vector &var, const Vector &param) const      \
   {                                                                          \
      MFEM_ASSERT(var.Size() == n_input,                                      \
                 "ADFunction::operator(): var.Size() must match n_input")     \
      MFEM_ASSERT(param.Size() == n_param,                                    \
                 "ADFunction::operator(): var.Size() must match n_input")     \
      using SCALAR = AD2Real_t;                                               \
      using VEC = AD2Vector;                                                  \
      using MAT = AD2Matrix;                                                  \
      body                                                                    \
   }                                                                          \
};

// @brief ADQuadEvalMode is an enumeration for the evaluation modes of the ADEnergy class.
// For example, if you want to evaluate the value and gradient of the function, you can use
// constexpr auto mode = ADEval::VALUE | ADEval::GRAD;
enum class ADEval
{
   NONE = 0, // nothing (defualt)
   VALUE  = 1 << 1, // u(T, ip) is needed
   GRAD   = 1 << 2, // grad u(T, ip) is needed
   // Hessian = 1 << 3 // In future, we may want to support Hessian evaluation.

   VECTOR = 1 << 4, // vector-valued function

};

constexpr ADEval operator|(ADEval a, ADEval b)
{
   return static_cast<ADEval>(static_cast<int>(a) | static_cast<int>(b));
}
constexpr ADEval operator&(ADEval a, ADEval b)
{
   return static_cast<ADEval>(static_cast<int>(a) & static_cast<int>(b));
}
inline constexpr bool hasFlag(ADEval mode, ADEval flag)
{
   return (mode & flag) == flag;
}

template <bool is_param_cf, ADEval... modes>
class ADBlockNonlinearFormIntegrator;

template <bool is_param_cf, ADEval mode>
class ADNonlinearFormIntegrator : public NonlinearFormIntegrator
{
protected:
   ADFunction &f;

private:
   int vdim;
   Vector x, jac;
   DenseMatrix H, Hx;

   // only if ADEvalInput::VECTOR. Each column corresponds to a vector component
   DenseMatrix xmat, jacMat, Hs, Hxsub;
   DenseMatrix elfun_matview, elvectmat, partelmat;

   DenseMatrix allshapes; // all shapes, [?shape, ?dshape]
   Vector shape, shape1, shape2;
   DenseMatrix vshape, vshape1, vshape2;
   DenseMatrix dshape, gshape1, gshape2;
   Vector nor;
   Vector param;
   std::unique_ptr<VectorCoefficient> owned_param_cf;
   VectorCoefficient *param_cf = nullptr;

   Vector face_param;
   // DenseMatrix d2shape, d2shape1, d2shape2; // for hessian. Not implemented yet.
public:
   ADNonlinearFormIntegrator(ADFunction &f, IntegrationRule *ir = nullptr)
      : NonlinearFormIntegrator(ir), f(f), vdim(1) {}
   ADNonlinearFormIntegrator(ADFunction &f, int vdim,
                             IntegrationRule *ir = nullptr)
      : NonlinearFormIntegrator(ir), f(f), vdim(vdim) {}
   ADNonlinearFormIntegrator(ADFunction &f, const Vector &param,
                             IntegrationRule *ir = nullptr)
      : ADNonlinearFormIntegrator(f, ir)
   {
      MFEM_VERIFY(!is_param_cf,
                  "ADNonlinearFormIntegrator: Expected parameter coefficients");
      MFEM_VERIFY(param.Size() == f.n_param,
                  "ADNonlinearFormIntegrator: param.Size() must match n_param");
      this->param = param;
   }

   ADNonlinearFormIntegrator(ADFunction &f, VectorCoefficient &param_cf,
                             IntegrationRule *ir = nullptr)
      : ADNonlinearFormIntegrator(f, ir)
   {
      MFEM_VERIFY(is_param_cf,
                  "ADNonlinearFormIntegrator: Expected constant parameter");
      MFEM_VERIFY(param_cf.GetVDim() == f.n_param,
                  "ADNonlinearFormIntegrator: param_cf.GetVDim() must match n_param");
      this->param_cf = &param_cf;
   }

   ADNonlinearFormIntegrator(ADFunction &f, Coefficient &param_cf,
                             IntegrationRule *ir = nullptr)
      : ADNonlinearFormIntegrator(f, ir)
   {
      MFEM_VERIFY(is_param_cf,
                  "ADNonlinearFormIntegrator: Expected constant parameter");
      MFEM_VERIFY(f.n_param == 1,
                  "ADNonlinearFormIntegrator: f takes more than one parameter, but only one coefficient is given");
      auto vec_cf = std::make_unique<VectorArrayCoefficient>(1);
      vec_cf->Set(0, &param_cf, false);
      this->owned_param_cf = std::move(vec_cf);
      this->param_cf = this->owned_param_cf.get();
   }

   ADNonlinearFormIntegrator(ADFunction &f, int vdim, const Vector &param,
                             IntegrationRule *ir = nullptr)
      : ADNonlinearFormIntegrator(f, vdim, ir)
   {
      MFEM_VERIFY(!is_param_cf,
                  "ADNonlinearFormIntegrator: Expected parameter coefficients");
      MFEM_VERIFY(param.Size() == f.n_param,
                  "ADNonlinearFormIntegrator: param.Size() must match n_param");
      this->param = param;
   }

   ADNonlinearFormIntegrator(ADFunction &f, int vdim, VectorCoefficient &param_cf,
                             IntegrationRule *ir = nullptr)
      : ADNonlinearFormIntegrator(f, vdim, ir)
   {
      MFEM_VERIFY(is_param_cf,
                  "ADNonlinearFormIntegrator: Expected constant parameter");
      MFEM_VERIFY(param_cf.GetVDim() == f.n_param,
                  "ADNonlinearFormIntegrator: param_cf.GetVDim() must match n_param");
      this->param_cf = &param_cf;
   }

   ADNonlinearFormIntegrator(ADFunction &f, int vdim, Coefficient &param_cf,
                             IntegrationRule *ir = nullptr)
      : ADNonlinearFormIntegrator(f, vdim, ir)
   {
      MFEM_VERIFY(is_param_cf,
                  "ADNonlinearFormIntegrator: Expected constant parameter");
      MFEM_VERIFY(f.n_param == 1,
                  "ADNonlinearFormIntegrator: f takes more than one parameter, but only one coefficient is given");
      auto vec_cf = std::make_unique<VectorArrayCoefficient>(1);
      vec_cf->Set(0, &param_cf, false);
      this->owned_param_cf = std::move(vec_cf);
      this->param_cf = this->owned_param_cf.get();
   }

   // post-setter for parameter
   void SetParameter(const Vector &param)
   {
      MFEM_VERIFY(!is_param_cf,
                  "ADNonlinearFormIntegrator: Expected parameter coefficients");
      this->param = param;
   }

   // post-setter for parameter coefficient
   void SetParameter(VectorCoefficient &param_cf)
   {
      MFEM_VERIFY(is_param_cf,
                  "ADNonlinearFormIntegrator: Expected constant parameter");
      this->param_cf = &param_cf;
   }

   // post-setter for parameter coefficient with scalar coefficient
   void SetParameter(Coefficient &param_cf)
   {
      MFEM_VERIFY(is_param_cf,
                  "ADNonlinearFormIntegrator: Expected constant parameter");
      MFEM_VERIFY(f.n_param == 1,
                  "ADNonlinearFormIntegrator: f takes more than one parameter, but only one coefficient is given");
      auto vec_cf = std::make_unique<VectorArrayCoefficient>(1);
      vec_cf->Set(0, &param_cf, false);
      this->owned_param_cf = std::move(vec_cf);
      this->param_cf = this->owned_param_cf.get();
   }

   const IntegrationRule* GetDefaultIntegrationRule(
      const FiniteElement& trial_fe, const FiniteElement& test_fe,
      const ElementTransformation& trans) const override
   {
      int order = std::max(trial_fe.GetOrder(), test_fe.GetOrder());
      return &IntRules.Get(trans.GetGeometryType(), order*2 + 2);
   }

   /// Compute the local energy
   real_t GetElementEnergy(const FiniteElement &el,
                           ElementTransformation &Tr,
                           const Vector &elfun) override;

   /// Perform the local action of the NonlinearFormIntegrator
   void AssembleElementVector(const FiniteElement &el,
                              ElementTransformation &Tr,
                              const Vector &elfun, Vector &elvect) override;

   /// Assemble the local gradient matrix
   void AssembleElementGrad(const FiniteElement &el,
                            ElementTransformation &Tr,
                            const Vector &elfun, DenseMatrix &elmat) override;

   /// @brief Perform the local action of the NonlinearFormIntegrator resulting
   /// from a face integral term.
   void AssembleFaceVector(const FiniteElement &el1,
                           const FiniteElement &el2,
                           FaceElementTransformations &Tr,
                           const Vector &elfun, Vector &elvect) override;


   /// @brief Assemble the local action of the gradient of the
   /// NonlinearFormIntegrator resulting from a face integral term.
   void AssembleFaceGrad(const FiniteElement &el1,
                         const FiniteElement &el2,
                         FaceElementTransformations &Tr,
                         const Vector &elfun, DenseMatrix &elmat) override;


protected:

   // Initialize shapes to [?value_shapes, ?grad_shapes]
   // and make value_shapes and grad_shapes reference to
   // allshapes.
   inline static int InitInputShapes(const FiniteElement &el,
                                     ElementTransformation &Tr,
                                     DenseMatrix &shapes,
                                     Vector &value_shapes,
                                     DenseMatrix &grad_shapes);

   // Calculate parameter, shape, dshape at the given integration point
   inline static void CalcInputShapes(const FiniteElement &el,
                                      ElementTransformation &Tr,
                                      const IntegrationPoint &ip,
                                      VectorCoefficient *parameter_cf,
                                      Vector &parameter,
                                      Vector &value_shapes,
                                      DenseMatrix &grad_shapes);
   template <bool is_param_cf_, ADEval... modes>
   friend class ADBlockNonlinearFormIntegrator;
private:
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

template <bool is_param_cf, ADEval... modes>
class ADBlockNonlinearFormIntegrator : public BlockNonlinearFormIntegrator
{
public:
   const IntegrationRule *IntRule = nullptr;

protected:
   constexpr static size_t numSpaces = sizeof...(modes);
   static constexpr std::array<ADEval, sizeof...(modes)> modes_arr = {modes...};
   ADFunction &f;
   const IntegrationRule* GetIntegrationRule(
      const FiniteElement& trial_fe, const FiniteElement& test_fe,
      const ElementTransformation& trans) const;

   /** @brief Returns an integration rule based on the arguments and
              internal state. (Version for identical trial_fe and test_fe)

       @see GetIntegrationRule(const FiniteElement*, const FiniteElement*,
            const ElementTransformation*)
   */
   const IntegrationRule* GetIntegrationRule(
      const FiniteElement& el,
      const ElementTransformation& trans) const;

private:
   Array<int> vdim;
   Vector x, jac;
   std::vector<Vector> xvar, jacVar;
   DenseMatrix H;
   DenseMatrix Hsub;
   DenseMatrix Hx;
   DenseMatrix Hxsub;

   // only if ADEvalInput::VECTOR. Each column corresponds to a vector component
   std::vector<DenseMatrix> xmat, jacVarMat, Hs;
   std::vector<DenseMatrix> elfun_matview, elvectmat, partelmat;

   std::vector<DenseMatrix> allshapes; // all shapes, [?shape, ?dshape]
   std::vector<Vector> shape, shape1, shape2;
   std::vector<DenseMatrix> vshape, vshape1, vshape2;
   std::vector<DenseMatrix> dshape, gshape1, gshape2;
   Vector nor;
   Vector param;
   // owned cf
   std::unique_ptr<VectorCoefficient> owned_param_cf;
   VectorCoefficient *param_cf = nullptr;
   Vector face_param;
   // DenseMatrix d2shape, d2shape1, d2shape2; // for hessian. Not implemented yet.
public:
   ADBlockNonlinearFormIntegrator(ADFunction &f,
                                  const IntegrationRule *ir = nullptr)
      : IntRule(ir), f(f), vdim(numSpaces)
      , allshapes(numSpaces)
      , xvar(numSpaces), jacVar(numSpaces)
      , Hx(numSpaces)
      , xmat(numSpaces), jacVarMat(numSpaces)
      , Hs(numSpaces), Hxsub(numSpaces)
      , elfun_matview(numSpaces), elvectmat(numSpaces)
      , partelmat(numSpaces)
      , shape(numSpaces), shape1(numSpaces), shape2(numSpaces)
      , vshape(numSpaces), vshape1(numSpaces), vshape2(numSpaces)
      , dshape(numSpaces), gshape1(numSpaces), gshape2(numSpaces)
   { vdim = 1; }

   ADBlockNonlinearFormIntegrator(ADFunction &f, std::initializer_list<int> vdim,
                                  const IntegrationRule *ir = nullptr)
      : ADBlockNonlinearFormIntegrator(f, ir), vdim(vdim)
   {}

   ADBlockNonlinearFormIntegrator(ADFunction &f, const Array<int> &vdim,
                                  const IntegrationRule *ir = nullptr)
      : ADBlockNonlinearFormIntegrator(f, ir)
   { this->vdim = vdim; }

   ADBlockNonlinearFormIntegrator(ADFunction &f, const Vector &param,
                                  const IntegrationRule *ir = nullptr)
      : ADBlockNonlinearFormIntegrator(f, ir)
   {
      MFEM_VERIFY(!is_param_cf,
                  "ADBlockNonlinearFormIntegrator: Expected parameter coefficients");
      MFEM_VERIFY(param.Size() == f.n_param,
                  "ADBlockNonlinearFormIntegrator: param.Size() must match n_param");
      this->param = param;
   }

   ADBlockNonlinearFormIntegrator(ADFunction &f, VectorCoefficient &param_cf,
                                  const IntegrationRule *ir = nullptr)
      : ADBlockNonlinearFormIntegrator(f, ir)
   {
      MFEM_VERIFY(is_param_cf,
                  "ADBlockNonlinearFormIntegrator: Expected constant parameter");
      MFEM_VERIFY(param_cf.GetVDim() == f.n_param,
                  "ADBlockNonlinearFormIntegrator: param_cf.GetVDim() must match n_param");
      this->param_cf = &param_cf;
   }

   ADBlockNonlinearFormIntegrator(ADFunction &f, Coefficient &param_cf,
                                  const IntegrationRule *ir = nullptr)
      : ADBlockNonlinearFormIntegrator(f, ir)
   {
      MFEM_VERIFY(is_param_cf,
                  "ADBlockNonlinearFormIntegrator: Expected constant parameter");
      MFEM_VERIFY(f.n_param == 1,
                  "ADBlockNonlinearFormIntegrator: f takes more than one parameter, but only one coefficient is given");
      auto vec_cf = std::make_unique<VectorArrayCoefficient>(1);
      vec_cf->Set(0, &param_cf, false);
      this->owned_param_cf = std::move(vec_cf);
      this->param_cf = this->owned_param_cf.get();
   }

   ADBlockNonlinearFormIntegrator(ADFunction &f, const Array<int> &vdim,
                                  const Vector &param,
                                  const IntegrationRule *ir = nullptr)
      : ADBlockNonlinearFormIntegrator(f, vdim, ir)
   {
      MFEM_VERIFY(!is_param_cf,
                  "ADBlockNonlinearFormIntegrator: Expected parameter coefficients");
      MFEM_VERIFY(param.Size() == f.n_param,
                  "ADBlockNonlinearFormIntegrator: param.Size() must match n_param");
      this->param = param;
   }

   ADBlockNonlinearFormIntegrator(ADFunction &f, const Array<int> &vdim,
                                  VectorCoefficient &param_cf,
                                  const IntegrationRule *ir = nullptr)
      : ADBlockNonlinearFormIntegrator(f, vdim, ir)
   {
      MFEM_VERIFY(is_param_cf,
                  "ADBlockNonlinearFormIntegrator: Expected constant parameter");
      MFEM_VERIFY(param_cf.GetVDim() == f.n_param,
                  "ADBlockNonlinearFormIntegrator: param_cf.GetVDim() must match n_param");
      this->param_cf = &param_cf;
   }

   ADBlockNonlinearFormIntegrator(ADFunction &f, const Array<int> &vdim,
                                  Coefficient &param_cf,
                                  const IntegrationRule *ir = nullptr)
      : ADBlockNonlinearFormIntegrator(f, vdim, ir)
   {
      MFEM_VERIFY(is_param_cf,
                  "ADBlockNonlinearFormIntegrator: Expected constant parameter");
      MFEM_VERIFY(f.n_param == 1,
                  "ADBlockNonlinearFormIntegrator: f takes more than one parameter, but only one coefficient is given");
      auto vec_cf = std::make_unique<VectorArrayCoefficient>(1);
      vec_cf->Set(0, &param_cf, false);
      this->owned_param_cf = std::move(vec_cf);
      this->param_cf = this->owned_param_cf.get();
   }

   // post-setter for parameter
   void SetParameter(const Vector &param)
   {
      MFEM_VERIFY(!is_param_cf,
                  "ADBlockNonlinearFormIntegrator: Expected parameter coefficients");
      this->param = param;
   }

   // post-setter for parameter coefficient
   void SetParameter(VectorCoefficient &param_cf)
   {
      MFEM_VERIFY(is_param_cf,
                  "ADBlockNonlinearFormIntegrator: Expected constant parameter");
      this->param_cf = &param_cf;
   }

   // post-setter for parameter coefficient with scalar coefficient
   void SetParameter(Coefficient &param_cf)
   {
      MFEM_VERIFY(is_param_cf,
                  "ADBlockNonlinearFormIntegrator: Expected constant parameter");
      MFEM_VERIFY(f.n_param == 1,
                  "ADBlockNonlinearFormIntegrator: f takes more than one parameter, but only one coefficient is given");
      auto vec_cf = std::make_unique<VectorArrayCoefficient>(1);
      vec_cf->Set(0, &param_cf, false);
      this->owned_param_cf = std::move(vec_cf);
      this->param_cf = this->owned_param_cf.get();
   }

   virtual void SetIntRule(const IntegrationRule *ir)
   { IntRule = ir; }

   /** @brief Prescribe a fixed IntegrationRule to use. Sets the NURBS patch
              integration rule to null.

       @see SetIntRule(const IntegrationRule*)
   */
   void SetIntegrationRule(const IntegrationRule &ir) { SetIntRule(&ir); }

   /** @brief Directly return the IntRule pointer (possibly null) without
       checking for NURBS patch rules or falling back on a default. */
   const IntegrationRule *GetIntRule() const { return IntRule; }

   /** @brief Equivalent to GetIntRule, but retained for backward
       compatibility with applications. */
   const IntegrationRule *GetIntegrationRule() const { return GetIntRule(); }


   /// Compute the local energy
   real_t GetElementEnergy(const Array<const FiniteElement *> &el,
                           ElementTransformation &Tr,
                           const Array<const Vector*> &elfun) override;

   /// Perform the local action of the NonlinearFormIntegrator
   void AssembleElementVector(const Array<const FiniteElement *>&el,
                              ElementTransformation &Tr,
                              const Array<const Vector *>&elfun,
                              const Array<Vector *>&elvect) override;

   /// Assemble the local gradient matrix
   void AssembleElementGrad(const Array<const FiniteElement *>&el,
                            ElementTransformation &Tr,
                            const Array<const Vector *>&elfun,
                            const Array2D<DenseMatrix *>&elmat) override;

   /// @brief Perform the local action of the NonlinearFormIntegrator resulting
   /// from a face integral term.
   void AssembleFaceVector(const Array<const FiniteElement *>&el1,
                           const Array<const FiniteElement *>&el2,
                           FaceElementTransformations &Tr,
                           const Array<const Vector *>&elfun,
                           const Array<Vector *>&elvect) override;


   /// @brief Assemble the local action of the gradient of the
   /// NonlinearFormIntegrator resulting from a face integral term.
   void AssembleFaceGrad(const Array<const FiniteElement *>&el1,
                         const Array<const FiniteElement *>&el2,
                         FaceElementTransformations &Tr,
                         const Array<const Vector *>&elfun,
                         const Array2D<DenseMatrix *>&elmat) override;


protected:

   const IntegrationRule* GetIntegrationRule(
      const Array<const FiniteElement *>& trial_fe,
      const Array<const FiniteElement *>& test_fe,
      const ElementTransformation& trans) const
   {
      if (IntRule) { return IntRule; }
      return GetDefaultIntegrationRule(trial_fe, test_fe, trans);
   }

   const IntegrationRule* GetIntegrationRule(
      const Array<const FiniteElement *>& el,
      const ElementTransformation& trans) const
   {
      if (IntRule) { return IntRule; }
      return GetDefaultIntegrationRule(el, el, trans);
   }

   virtual const IntegrationRule* GetDefaultIntegrationRule(
      const Array<const FiniteElement *>& trial_fe,
      const Array<const FiniteElement *>& test_fe,
      const ElementTransformation& trans) const
   {
      int order = 0;
      for (int i=0; i<trial_fe.Size(); i++)
      {
         order = std::max(order, trial_fe[i]->GetOrder());
      }
      for (int i=0; i<test_fe.Size(); i++)
      {
         order = std::max(order, test_fe[i]->GetOrder());
      }
      return &IntRules.Get(trans.GetGeometryType(), order*2 + 2);
   }

   static std::array<int, sizeof...(modes)> InitInputShapes(
                                    const Array<const FiniteElement *>& el,
                                    ElementTransformation &Tr,
                                    std::vector<DenseMatrix> &shapes,
                                    std::vector<Vector> &value_shapes,
                                    std::vector<DenseMatrix> &grad_shapes);

   static void CalcInputShapes(
      const Array<const FiniteElement *>& el,
      ElementTransformation &Tr,
      const IntegrationPoint &ip,
      VectorCoefficient *parameter_cf,
      Vector &parameter,
      std::vector<Vector> &value_shapes,
      std::vector<DenseMatrix> &grad_shapes);

private:
};

}
