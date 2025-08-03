#pragma once

#include "mfem.hpp"
#include "miniapps/autodiff/tadvector.hpp"
#include "miniapps/autodiff/taddensemat.hpp"

namespace mfem
{

// First order dual
typedef future::dual<real_t, real_t> ADReal_t;
typedef TAutoDiffVector<ADReal_t> ADVector;
typedef TAutoDiffDenseMatrix<ADReal_t> ADMatrix;

// second order dual
typedef future::dual<ADReal_t, ADReal_t> AD2Real_t;
typedef TAutoDiffVector<AD2Real_t> AD2Vector;
typedef TAutoDiffDenseMatrix<AD2Real_t> AD2Matrix;

inline void MultAv(const DenseMatrix &A, const Vector &v, Vector &Av)
{
   Av.SetSize(A.Height());
   A.Mult(v, Av);
}

struct ADFunction
{
   int n_input;
   int n_param;
   ADFunction(int n_input, int n_param)
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
   void Gradient(const Vector &x, const Vector &param, Vector &J) const;
   // Evaluate the Hessian, using forward over forward autodiff
   void Hessian(const Vector &x, const Vector &param, DenseMatrix &H) const;
};

// Make Autodiff Functor
// @param name will be the name of the structure
// @param T is the templated scalar type
// @param VEC is the templated vector type
// @param MAT is the templated matrix type
// @param param is additional parameter name (will not be differentiated)
// @param body is the main function body. Use T() to create 0 T-typed value.
#define MAKE_AD_FUNCTION(name, T, VEC, MAT, var, param, body)                 \
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
      using T = real_t;                                                       \
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
      using T = ADReal_t;                                                     \
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
      using T = AD2Real_t;                                                    \
      using VEC = AD2Vector;                                                  \
      using MAT = AD2Matrix;                                                  \
      body                                                                    \
   }                                                                          \
};

// @brief ADQuadEvalMode is an enumeration for the evaluation modes of the ADEnergy class.
// For example, homogeneous diffusion can be evaluated with mode=GRADIENT.
// heterogeneous diffusion can be evaluated with mode=PARAM | GRAD
enum class ADEvalInput
{
   NONE = 0, // nothing (defualt)

   // Parameters
   PARAM  = 1 << 0, // parameter is space-dependent
   NORMAL = 1 << 2, // requires normal as parameter (param will be [param, normal])

   VECTOR = 1 << 3, // vector-valued function
   VALUE  = 1 << 4, // u(T, ip) is needed
   GRAD   = 1 << 5, // grad u(T, ip) is needed

   // Hessian = 1 << 5 // In future, we may want to support Hessian evaluation.
};

constexpr ADEvalInput operator|(ADEvalInput a, ADEvalInput b)
{
   return static_cast<ADEvalInput>(static_cast<int>(a) | static_cast<int>(b));
}
constexpr ADEvalInput operator&(ADEvalInput a, ADEvalInput b)
{
   return static_cast<ADEvalInput>(static_cast<int>(a) & static_cast<int>(b));
}
inline constexpr bool hasFlag(ADEvalInput mode, ADEvalInput flag)
{
   return (mode & flag) == flag;
}

template <ADEvalInput mode>
class ADNonlinearFormIntegrator : public NonlinearFormIntegrator
{
protected:
   ADFunction &f;
private:
   int vdim;
   Vector x, j;
   DenseMatrix H, Hx;

   // only if ADEvalInput::VECTOR. Each column corresponds to a vector component
   DenseMatrix xmat, jmat;
   DenseTensor Hxmat;
   DenseMatrix elfun_matview, elvectmat;

   DenseMatrix allshapes; // all shapes, [?shape, ?dshape]
   Vector shape, shape1, shape2;
   DenseMatrix vshape, vshape1, vshape2;
   DenseMatrix dshape, gshape1, gshape2;
   Vector nor;
   Vector param;
   VectorCoefficient *param_cf;
   Vector face_param;
   // DenseMatrix d2shape, d2shape1, d2shape2; // for hessian. Not implemented yet.
public:
   ADNonlinearFormIntegrator(ADFunction &f): f(f), vdim(1) {}
   ADNonlinearFormIntegrator(ADFunction &f, int vdim): f(f), vdim(vdim)
   {
      MFEM_VERIFY((ADEvalInput::VECTOR & mode) == mode,
                  "ADNonlinearFormIntegrator: vdim is only valid for vector-valued functions");
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
   return (kappa(0)*0.5)*(gradu*gradu);
});
}
