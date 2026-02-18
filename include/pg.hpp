#pragma once
#include "mfem.hpp"
#include "ad_native.hpp"
#include "tools.hpp"
#include "diffobj.hpp"

namespace mfem
{

template <typename T>
inline void Mult(const DenseMatrix &A, const TAutoDiffVector<T> &x,
                 TAutoDiffVector<T> &y)
{
   y.SetSize(A.Height());
   y = T();
   for (int j = 0; j < A.Width(); j++)
   {
      const T xj = x[j];
      for (int i = 0; i < A.Height(); i++)
      {
         y[i] += A(i,j) * xj;
      }
   }
}
inline void Mult(const DenseMatrix &A, const Vector &x, Vector &y)
{ A.Mult(x, y); }

template <typename T>
inline void MultTranspose(const DenseMatrix &A, const TAutoDiffVector<T> &x,
                          TAutoDiffVector<T> &y)
{
   y.SetSize(A.Width());
   for (int j = 0; j < A.Width(); j++)
   {
      y[j] = T();
      for (int i = 0; i < A.Height(); i++)
      {
         y[j] += A(i, j) * x[i];
      }
   }
}

inline void MultTranspose(const DenseMatrix &A, const Vector &x, Vector &y)
{ A.MultTranspose(x, y); }


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
class ADEntropy : public ADFunction
{
public:
   ADEntropy(int n)
      : ADFunction(n) { }
   ADEntropy(int n, int eval_capacity)
      : ADFunction(n, eval_capacity) { }
};

template <typename T>
std::vector<T*> uniquevec2ptrvec(std::vector<std::unique_ptr<T>> &vec)
{
   std::vector<T*> ptrs(vec.size());
   for (int i=0; i<vec.size(); i++)
   {
      ptrs[i] = vec[i].get();
   }
   return ptrs;
}


// Construct augmented energy for proximal Galerkin
// psi =
// L(u, psi) = f(u) + (1/alpha)(u*(psi-psi_k) - E^*(psi))
// Equivalently, L(u, lambda) = f(u) + (u*lambda - E^*(alpha*lambda + psi_k))
// so that
// dL/du = df/du + (1/alpha)(psi-psi_k)
// dL/dpsi = (1/alpha)(u - dE^*(psi))
// When primal is not full vector, set primal_begin
// The parameter should be [org_param, entropy_param, alpha, psi_k]
class ADPGFunctional : public ADFunction
{
protected:
   ADFunction &f;
   std::vector<ADEntropy*> dual_entropy;
   std::vector<int> primal_idx;
   std::vector<int> dual_idx;
   std::vector<int> entropy_size;
   mutable const BlockVector *latent_k;
   mutable Vector jac;
   mutable DenseMatrix hess;
   mutable real_t alpha;
   std::unique_ptr<VectorCoefficient> owned_cf;
   static int GetEntropySize(const std::vector<ADEntropy*> &dual_entropy)
   {
      int size = 0;
      for (const auto &entropy : dual_entropy)
      {
         size += entropy->width;
      }
      return size;
   }

public:
   ADPGFunctional(ADFunction &f, ADEntropy &dual_entropy, Evaluator::param_t alpha,
                  int idx=0)
      : ADFunction(f.width + dual_entropy.width, 1)
      , f(f), dual_entropy{&dual_entropy}
      , primal_idx(1)
      , dual_idx(1)
      , entropy_size(1)
   {
      evaluator.Add(alpha);
      this->primal_idx[0] = idx;
      entropy_size[0] = dual_entropy.width;
      MFEM_VERIFY(f.width >= this->primal_idx[0] + entropy_size[0],
                  "ADPGFunctional: f.width must not exceed "
                  "primal_begin + dual_entropy.width:"
                  << f.width << " >= " << width);
      dual_idx[0] = f.width;
   }
   ADPGFunctional(ADFunction &f, ADEntropy &dual_entropy,
                  Evaluator::param_t alpha,
                  GridFunction &latent_k, int idx=0)
      : ADPGFunctional(f, dual_entropy, alpha, idx)
   {
      evaluator.Add(&latent_k);
   }
   // Multiple entropies
   ADPGFunctional(ADFunction &f, std::vector<ADEntropy*> dual_entropy_,
                  std::vector<int> &primal_begin, Evaluator::param_t alpha)
      : ADFunction(f.width + GetEntropySize(dual_entropy_), 1)
      , f(f), dual_entropy(std::move(dual_entropy_))
      , primal_idx(primal_begin)
      , dual_idx(dual_entropy.size())
      , entropy_size(dual_entropy.size())
      , alpha(*evaluator.val.GetBlock(0).GetData())
   {
      evaluator.Add(alpha);
      int dual_entropy_size = 0;
      int max_primal_index = 0;
      for (int i=0; i<dual_entropy.size(); i++)
      {
         dual_entropy_size += dual_entropy[i]->width;
         max_primal_index = std::max(max_primal_index,
                                     primal_begin[i] + dual_entropy[i]->width);
      }
      MFEM_VERIFY(f.width >= max_primal_index,
                  "ADPGFunctional: f.width must be larger than "
                  "primal_begin[i] + dual_entropy.width[i] for all i");
   }

   ADPGFunctional(ADFunction &f, std::vector<ADEntropy*> dual_entropy,
                  std::vector<GridFunction*> latent_k_gf, std::vector<int> &primal_begin,
                  Evaluator::param_t alpha)
      : ADPGFunctional(f, std::move(dual_entropy), primal_begin, alpha)
   {
      MFEM_VERIFY(latent_k_gf.size() == this->dual_entropy.size(),
                  "ADPGFunctional: latent_k must have the same size as dual_entropy: "
                  << latent_k_gf.size() << " != " << dual_entropy.size());
      MFEM_VERIFY(latent_k_gf.size() == primal_begin.size(),
                  "ADPGFunctional: latent_k must have the same size as primal_begin"
                  << latent_k_gf.size() << " != " << primal_begin.size());
      for (int i=0; i<latent_k_gf.size(); i++)
      {
         MFEM_VERIFY(latent_k_gf[i] != nullptr,
                     "ADPGFunctional: latent_k_gf[" << i << "] is null");
         evaluator.Add(latent_k_gf[i]);
      }
   }
   // Multiple entropies
   ADPGFunctional(ADFunction &f,
                  std::vector<std::unique_ptr<ADEntropy>> &dual_entropy,
                  std::vector<int> &primal_begin, Evaluator::param_t alpha)
      : ADPGFunctional(f, uniquevec2ptrvec(dual_entropy), primal_begin, alpha)
   {}
   ADPGFunctional(ADFunction &f,
                  std::vector<std::unique_ptr<ADEntropy>> &dual_entropy,
                  std::vector<std::unique_ptr<GridFunction>> &latent_k_gf,
                  std::vector<int> primal_begin, Evaluator::param_t alpha)
      : ADPGFunctional(f, uniquevec2ptrvec(dual_entropy),
                       uniquevec2ptrvec(latent_k_gf), primal_begin, alpha)
   {}

   const GridFunction& GetPrevLatent(int i) const;

   ADFunction &GetObjective() const
   { return f; }

   ADEntropy &GetEntropy() const
   {
      MFEM_VERIFY(dual_entropy.size() == 1,
                  "ADPGFunctional: GetEntropy() can only be called when there is a single entropy");
      return *dual_entropy[0];
   }
   const std::vector<ADEntropy*> &GetEntropies() const
   { return dual_entropy; }

   real_t GetAlpha() const { return alpha; }

   void ProcessParameters(ElementTransformation &Tr,
                          const IntegrationPoint &ip) const override
   {
      for (int i=0; i<dual_entropy.size(); i++)
      {
         dual_entropy[i]->ProcessParameters(Tr, ip);
      }
      f.ProcessParameters(Tr, ip);
      latent_k = &evaluator.Eval(Tr, ip);
      alpha = evaluator.val[0];
   }

   AD_IMPL(T, V, M, x_psi,
   {
      // variables
      const V x(x_psi.GetData(), f.width);
      V psi;

      // evaluate mixed value
      T cross_entropy = T();
      T dual_entropy_sum = T();
      for (int i=0; i<entropy_size.size(); i++)
      {
         psi.SetDataAndSize(x_psi.GetData() + dual_idx[i], entropy_size[i]);
         const Vector &psi_k = latent_k->GetBlock(i+1);
         for (int j=0; j<entropy_size[i]; j++)
         {
            cross_entropy += x[primal_idx[i] + j]*(psi[j] - psi_k[j]);
         }
         dual_entropy_sum += (*dual_entropy[i])(psi);
      }
      return f(x) + (cross_entropy - dual_entropy_sum)/alpha;
   });
};

class ADLambdaPGFunctional : public ADPGFunctional
{
   using ADPGFunctional::ADPGFunctional;

   AD_IMPL(T, V, M, x_lambda,
   {
      // variables
      const V x(x_lambda.GetData(), f.width);
      V lambda;
      V latent;

      // evaluate mixed value
      T cross_entropy = T();
      T dual_entropy_sum = T();
      for (int i=0; i<entropy_size.size(); i++)
      {
         lambda.SetDataAndSize(x_lambda.GetData() + dual_idx[i], entropy_size[i]);
         for (int j=0; j<entropy_size[i]; j++)
         {
            cross_entropy += x[primal_idx[i] + j]*lambda[j];
         }
         latent = latent_k->GetBlock(i+1);
         latent.Add(alpha, lambda);
         dual_entropy_sum += (*dual_entropy[i])(latent);
      }
      return f(x) + cross_entropy - dual_entropy_sum/alpha;
   });
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
class ShannonEntropy : public ADEntropy
{
protected:
   const real_t &bound;
   int sign;
public:
   ShannonEntropy(Evaluator::param_t bound, int sign=1)
      : ADEntropy(1, 1)
      , bound(*evaluator.val.GetData())
      , sign(sign)
   {
      evaluator.Add(bound);
      MFEM_VERIFY(sign == 1 || sign == -1,
                  "ShannonEntropy: sign must be 1 or -1");
      MFEM_VERIFY(evaluator.val.GetBlock(0).Size() == 1,
                  "ShannonEntropy: The provided bound has the wrong size. "
                  "Expected 1, got " << evaluator.val.GetBlock(0).Size());
   }
   AD_IMPL(T, V, M, x, return exp(x[0]*sign) - sign*bound*x[0]; );
};

// Dual entropy for (negative) Fermi-Dirac with [lower, upper] bounds
class FermiDiracEntropy : public ADEntropy
{
protected:
   const real_t &upper_bound;
   const real_t &lower_bound;
   mutable real_t shift;
   mutable real_t scale;
public:
   FermiDiracEntropy(Evaluator::param_t lower_bound,
                     Evaluator::param_t upper_bound)
      : ADEntropy(1, 2)
      , upper_bound(*evaluator.val.GetData())
      , lower_bound(*(evaluator.val.GetData()+1))
   {
      evaluator.Add(lower_bound);
      evaluator.Add(upper_bound);
      MFEM_VERIFY(evaluator.val.GetBlock(0).Size() == 1,
                  "FermiDiracEntropy: The provided bound has the wrong size. "
                  "Expected 1, got " << evaluator.val.GetBlock(0).Size());
      MFEM_VERIFY(evaluator.val.GetBlock(1).Size() == 1,
                  "FermiDiracEntropy: The provided bound has the wrong size. "
                  "Expected 1, got " << evaluator.val.GetBlock(1).Size());
   }
   void ProcessParameters(const BlockVector &x) const override
   {
      shift = lower_bound;
      scale = upper_bound - shift;
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
class HellingerEntropy : public ADEntropy
{
   const real_t &scale;
public:
   HellingerEntropy(int dim, Evaluator::param_t bound)
      : ADEntropy(dim, 1)
      , scale(*evaluator.val.GetData())
   {
      evaluator.Add(bound);
      MFEM_VERIFY(evaluator.val.GetBlock(0).Size() == 1,
                  "HellingerEntropy: The provided bound has the wrong size. "
                  "Expected 1, got " << evaluator.val.GetBlock(0).Size());
   }
   void ProcessParameters(const BlockVector &x) const override
   {
      MFEM_ASSERT(scale > 0, "HellingerEntropy: bound must be positive");
   }
   AD_IMPL(T, V, M, x, return sqrt(1 + (x*x)*(scale*scale)););
};

// Dual entropy for (negative) Simplex entropy with
// x_i >= 0 sum_i x_i = bound
// Also known as cateborical entropy or multinomial Shannon entropy
class SimplexEntropy : public ADEntropy
{
   const real_t &scale;
public:
   SimplexEntropy(int width, Evaluator::param_t bound)
      : ADEntropy(width, 1), scale(*evaluator.val.GetData())
   {
      evaluator.Add(bound);
      MFEM_VERIFY(evaluator.val.GetBlock(0).Size() == 1,
                  "SimplexEntropy: The provided bound has the wrong size. "
                  "Expected 1, got " << evaluator.val.GetBlock(0).Size());
   }

   void ProcessParameters(const BlockVector &x) const override
   {
      MFEM_ASSERT(scale >= 0, "SimplexEntropy: bound must be non-negative");
   }
   AD_IMPL(T, V, M, x,
   {
      T maxval = x[0];
      for (int i=1; i<x.Size(); i++) { maxval = max(maxval, x[i]); }

      T sum_exp = T();
      for (int i=0; i<x.Size(); i++)
      {
         sum_exp += exp(x[i]-maxval);
      }
      return scale*(maxval + log(sum_exp));
   });
};

// Simplex entropy with right corner at the origin.
class RightSimplexEntropy : public ADEntropy
{
   const real_t &scale;
public:
   RightSimplexEntropy(int width, Evaluator::param_t bound)
      : ADEntropy(width, 1), scale(*evaluator.val.GetData())
   {
      evaluator.Add(bound);
      MFEM_VERIFY(evaluator.val.GetBlock(0).Size() == 1,
                  "RightSimplexEntropy: The provided bound has the wrong size. "
                  "Expected 1, got " << evaluator.val.GetBlock(0).Size());
   }

   void ProcessParameters(const BlockVector &x) const override
   {
      MFEM_ASSERT(scale >= 0, "SimplexEntropy: bound must be non-negative");
   }
   AD_IMPL(T, V, M, x,
   {
      T sum_exp = exp(x[0]);
      for (int i=1; i<x.Size(); i++)
      {
         sum_exp += exp(x[i]);
      }
      return scale*(1.0 + log(sum_exp));
   });
};

class PolytopalEntropy : public ADEntropy
{
   DenseMatrix vertices; // V - v_0, Shifted vertex
   Vector v0; // corresponding to the first vertex
   RightSimplexEntropy simplex_entropy;
public:
   PolytopalEntropy(const DenseMatrix &V)
      :ADEntropy(V.Height(), 1)
      , vertices(V.Height(), V.Width()-1)
      , simplex_entropy(V.Width(), 1.0)
   {
      V.GetColumn(0, v0);
      vertices.CopyCols(V, 1, V.Width());
      Vector vert;
      for (int i=0; i<vertices.Width(); i++)
      {
         vertices.GetColumnReference(i, vert);
         vert -= v0;
      }
   }

   AD_IMPL(T, V, M, x,
   {
      V Vx(vertices.Width());
      Mult(vertices, x, Vx);
      return simplex_entropy(Vx) + (x*v0);
   })
};

class AffineTranslatedEntropy : public ADEntropy
{
   const ADEntropy &base_entropy;
   DenseMatrix A_T;
   Vector b;

public:
   AffineTranslatedEntropy(
      const ADEntropy &base, const DenseMatrix &A_, const Vector &b_)
      : ADEntropy(A_.Height()), base_entropy(base), A_T(A_), b(b_)
   {
      MFEM_VERIFY(A_.Width() == base.width,
                  "AffineTranslatedEntropy: A width must match base entropy width: "
                  << A_.Width() << " != " << base.width);
      A_T.Transpose();
   }
   AD_IMPL(T, V, M, x,
   {
      V Ax(A_T.Height());
      Mult(A_T, x, Ax);
      return base_entropy(Ax) + (x*b);
   })
};


class PGPreconditioner : public Solver
{
private:
   ParFiniteElementSpace *fes;
   ParGridFunction &psi;
   real_t &alpha;
   DifferentiableCoefficient entropy_cf;
   MatrixCoefficient &entropy_hessian_cf;
   IdentityMatrixCoefficient identity_cf;
   MatrixSumCoefficient entropy_prec_cf;
   std::unique_ptr<HypreBoomerAMG> stiffness_prec;
   std::unique_ptr<CGSolver> stiffness_solver;
   const Array<int> *offsets;
   Vector diag;
   Vector mass_diag;
   mutable Vector z;
   const HypreParMatrix *B;
   bool is_dg = true;
   std::unique_ptr<BilinearForm> inv_mass_form;
   std::unique_ptr<HypreParMatrix> inv_mass;
   std::unique_ptr<BilinearForm> mass_form;
   std::unique_ptr<HypreParMatrix> mass;
   std::unique_ptr<CGSolver> mass_solver;
   std::unique_ptr<HypreBoomerAMG> mass_prec;
public:
   PGPreconditioner(ParGridFunction &psik,
                    ParGridFunction &psi,
                    ADEntropy &entropy,
                    real_t &alpha)
      : fes(psik.ParFESpace())
      , psi(psi)
      , alpha(alpha)
      , entropy_cf(entropy)
      , entropy_hessian_cf(entropy_cf.Hessian())
      , identity_cf(entropy.width)
      , entropy_prec_cf(entropy_hessian_cf, identity_cf, -1.0, -1.0)
   {
      entropy_cf.AddInput(&psi);
      if ((is_dg = fes->IsDGSpace()))
      {
         inv_mass_form = NewBilinearForm(*fes);
         inv_mass_form->AddDomainIntegrator(
            new InverseIntegrator(
               new VectorMassIntegrator(entropy_prec_cf)));
      }
      else
      {
         mass_form = NewBilinearForm(*fes);
         mass_form->AddDomainIntegrator(
            new VectorMassIntegrator(entropy_prec_cf));
      }
   }

   void SetOperator(const Operator &op) override
   {
      if (is_dg)
      {
         entropy_prec_cf.SetAlpha(-1.0 / alpha);
         entropy_prec_cf.SetBeta(-1.0 / alpha/alpha);
         inv_mass_form->Update();
         inv_mass_form->Assemble();
         inv_mass_form->Finalize();
         inv_mass.reset(static_cast<ParBilinearForm&>
                        (*inv_mass_form).ParallelAssemble());
      }
      else
      {
         entropy_prec_cf.SetAlpha(1.0 / alpha);
         entropy_prec_cf.SetBeta(1.0 / alpha/alpha);
         mass_form->Update();
         auto * mat = mass_form->LoseMat();
         if (mat) { delete mat; }
         mass_form->Assemble();
         mass_form->Finalize();
         mass.reset(static_cast<ParBilinearForm&>
                    (*mass_form).ParallelAssemble());
         mass_prec = std::make_unique<HypreBoomerAMG>(*mass);
         mass_prec->SetPrintLevel(0);
      }
      auto * blocks = dynamic_cast<const BlockOperator*>(&op);
      MFEM_VERIFY(blocks != nullptr, "Not a BlockOperator");

      offsets = &blocks->RowOffsets();
      MFEM_VERIFY(offsets->Size() == 3, "Only two blocks supported");

      const HypreParMatrix * A = dynamic_cast<const HypreParMatrix*>
                                 (&blocks->GetBlock(0,0));
      MFEM_VERIFY(A != nullptr, "Not a HypreParMatrix");
      stiffness_prec = std::make_unique<HypreBoomerAMG>(*A);
      stiffness_prec->SetPrintLevel(0);
   }

   void Mult(const Vector &b, Vector &x) const override
   {
      MFEM_VERIFY(b.Size() == x.Size(),
                  "PGPreconditioner: b and x must have the same size");
      MFEM_VERIFY(offsets->Last() == b.Size(),
                  "PGPreconditioner: offsets size does not match b size");

      const BlockVector bblock(b.GetData(), *offsets);
      const Vector &b_primal = bblock.GetBlock(0);
      const Vector &b_dual = bblock.GetBlock(1);

      BlockVector xblock(x.GetData(), *offsets);
      Vector &x_primal = xblock.GetBlock(0);
      Vector &x_dual = xblock.GetBlock(1);

      stiffness_prec->Mult(b_primal, x_primal);
      if (is_dg)
      {
         inv_mass->Mult(b_dual, x_dual);
      }
      else
      {
         mass_prec->Mult(b_dual, x_dual);
         x_dual.Neg();
      }
   }
};

class PetscOperatorWrapper : public Operator
{
protected:
   MPI_Comm comm;
   Operator &op;
   Operator::Type mtype;
   mutable std::unique_ptr<PetscParMatrix> petsc_matrix;
public:
   PetscOperatorWrapper(MPI_Comm comm, Operator &op,
                        Operator::Type mtype = Operator::Type::PETSC_MATAIJ)
      : Operator(op.Height(), op.Width()), comm(comm), op(op), mtype(mtype)
   { }

   void Mult(const Vector &x, Vector &y) const override
   {
      op.Mult(x, y);
   }

   Operator &GetGradient(const Vector &x) const override
   {
      auto &grad = op.GetGradient(x);
      petsc_matrix = std::make_unique<PetscParMatrix>(comm, &grad, mtype);
      return *petsc_matrix;
   }
};

class NewtonLinearSolverMonitor : public IterativeSolverController
{
protected:
   /// The last IterativeSolver to which this controller was attached.
   const class IterativeSolver *iter_solver;
#ifdef MFEM_USE_PETSC
   PetscLinearSolver *petsc_solver;
#endif
   IterativeSolver *mfem_solver;

   int numIterations=0;
   int prefix=0;
   bool is_root = true;
   bool converged = false;

public:
#ifdef MFEM_USE_PETSC
   NewtonLinearSolverMonitor(PetscLinearSolver &linear_solver)
      : petsc_solver(&linear_solver)
   {
      is_root = Mpi::Root();
   }
#endif
   NewtonLinearSolverMonitor(IterativeSolver &linear_solver)
      : mfem_solver(&linear_solver)
   {
#ifdef MFEM_USE_MPI
      is_root = Mpi::Root();
#endif
   }

   void SetPrefix(size_t i) { prefix = i; }

   virtual void Reset()
   {
      converged = false;
      numIterations = 0;
   }

   /// Monitor the solution vector r
   virtual void MonitorResidual(int it, real_t norm, const Vector &r,
                                bool final)
   {
      if (final && is_root)
      {
         for (int i=0; i<prefix; i++) { out << " "; }
         out << "Average Linear Solver Iterations: " << (numIterations /
                                                         (it + 1.)) << std::endl;
         numIterations = 0;
         return;
      }
#ifdef MFEM_USE_PETSC
      if (petsc_solver) { numIterations += petsc_solver->GetNumIterations(); }
#endif
      if (mfem_solver) { numIterations += mfem_solver->GetNumIterations(); }
   }
};

class BregmanDykstra : public IterativeSolver
{
private:
   QuadratureSpace &qspace;
   ADEntropy &entropy;
   const int vdim;
   mutable DifferentiableCoefficient entropy_cf;
   VectorCoefficient &primal_cf;
   mutable QuadratureFunction latent;
   mutable QuadratureFunction latent_k;
   mutable QuadratureFunction gradient;
   mutable QuadratureFunction latent_freeze;
   std::vector<ADFunction*> constraints;
   mutable std::vector<std::unique_ptr<DifferentiableCoefficient>> constraint_cfs;
   std::vector<real_t> targets;
   mutable std::vector<std::unique_ptr<QuadratureFunction>> qi;
   int max_sub_iter;
   real_t lower=-1e04, upper=1e04;
public:
   BregmanDykstra(QuadratureSpace &qspace_, ADEntropy &entropy)
      : qspace(qspace_)
      , entropy(entropy), vdim(entropy.width)
      , entropy_cf(entropy), primal_cf(entropy_cf.Gradient())
      , latent(qspace, vdim), latent_k(qspace, vdim)
      , gradient(qspace, vdim)
   {
      entropy_cf.AddInput(&latent);
      constraints.clear();
#ifdef MFEM_USE_MPI
      ParMesh *mesh = dynamic_cast<ParMesh*>(qspace.GetMesh());
      if (mesh) { SetComm(mesh->GetComm()); }
#endif
      rel_tol = 1e-08;
      abs_tol = 1e-08;
      max_iter = 1000;
      max_sub_iter = 1000;
      width = height = vdim*qspace.GetSize();
   }
   void SetMaxSubIter(int n) { max_sub_iter = n; }

   void AddConstraint(ADFunction &constraint, real_t target=0.0)
   {
      MFEM_VERIFY(constraint.width == vdim,
                  "Constraint input dimension does not match.");
      constraints.push_back(&constraint);
      constraint_cfs.push_back(
         std::make_unique<DifferentiableCoefficient>(constraint));
      constraint_cfs.back()->AddInput(&primal_cf);
      targets.push_back(target);

      qi.push_back(std::make_unique<QuadratureFunction>(qspace, vdim));
   }

   /// @brief Project latent0 onto the intersection of the convex sets defined by the constraints
   /// latent0 and opt_latent can be the same vector
   void Mult(const Vector &latent0, Vector &opt_latent) const override
   {
      MFEM_VERIFY(latent0.Size() == width,
                  "Input vector size does not match operator width.");
      if (constraints.size() == 0) { opt_latent = latent0; return; }
      opt_latent.SetSize(height);
      const QuadratureFunction x_qf(&qspace, latent0.GetData(), vdim);
      latent.SetData(opt_latent.GetData());
      latent = latent0;
      Vector constraint_val(constraints.size());
      Vector qi_norm(constraints.size());
      for (int j=0; j<constraints.size(); j++)
      {
         constraint_val[j] = qspace.Integrate(*constraint_cfs[j]) - targets[j];
         *qi[j] = 0.0;
      }
      num_const_eval = 0;
      for (final_iter=0; final_iter<max_iter; final_iter++)
      {
         for (int j=0; j<constraints.size(); j++)
         {
            latent_k = latent;
            real_t cval = qspace.Integrate(*constraint_cfs[j]) - targets[j];
            constraint_cfs[j]->Gradient().Project(gradient);
            real_t target = dot(gradient, primal_cf) - cval;
            TangentProjection(j, target);
            latent_k -= latent;
            *qi[j] += latent_k;
         }
         for (int j=0; j<constraints.size(); j++)
         {
            constraint_val[j] = qspace.Integrate(*constraint_cfs[j]) - targets[j];
            qi_norm[j] = Dot(*qi[j], *qi[j]);
         }
         if (constraint_val.Max() < abs_tol)
         {
            break;
         }
      }
      if (print_level > 0)
      {
         out << "BregmanDykstra: Completed in " << final_iter+1
             << " iterations with " << num_const_eval
             << " constraint evaluations. Max constraint violation = "
             << constraint_val.Max() << std::endl;
      }
   }
   int NumConstraintEvals() const { return num_const_eval; }
   // Set the search interval [new_lower, new_upper]
   // If the bounds do not bracket the root, they will be shifted.
   void SetBracket(real_t new_lower, real_t new_upper)
   {
      MFEM_VERIFY(new_lower < new_upper,
                  "BregmanDykstra: lower bound must be less than upper bound");
      lower = new_lower;
      upper = new_upper;
   }
   // Set the search interval [-new_bound, new_bound]
   // If the bounds do not bracket the root, they will be shifted.
   void SetBracket(real_t new_bound)
   {
      MFEM_VERIFY(new_bound > 0,
                  "BregmanDykstra: lower bound must be less than upper bound");
      lower = -new_bound;
      upper = new_bound;
   }

private:
   using IterativeSolver::Dot;
   mutable int num_const_eval=0;

   void TangentProjection(const int i, const real_t target) const
   {
      real_t c, fc;
      int n, side = 0;
      latent_freeze = latent;
      auto f = [&](real_t x)
      {
         num_const_eval++;
         add(latent_freeze, x, gradient, latent);
         return qspace.Integrate(*constraint_cfs[i]) - target;
      };
      real_t f0 = f(0.0);
      if (f0 < abs_tol) { return; }
      real_t a = lower, b = upper;
      real_t fa = f(a);
      real_t fb = f(b);
      if (print_level > 1)
      {
         out << "  f(0) = " << f0
             << " f(" << a << ") = " << fa
             << " f(" << b << ") = " << fb
             << std::endl;
      }
      /// Shift bounds if necessary
      if (fa * fb > 0)
      {
         if (print_level > 1)
         {
            out << "  Initial bounds do not bracket root, shifting..." << std::endl;
         }
         real_t diff = b - a;
         if (fa > 0) // both positive
         {
            while (fa > 0)
            {
               b = a; fb = fa;
               a -= diff;
               fa = f(a);
            }
         }
         else // both negative
         {
            while (fb < 0)
            {
               a = b; fa = fb;
               b += diff;
               fb = f(b);
            }
         }
         if (print_level > 1)
         {
            out << "  f(0) = " << f0 << " f(a) = " << fa << " f(b) = " << fb << std::endl;
         }
      }


      for (n = 0; n < max_sub_iter; n++)
      {
         if (fabs(b - a) < rel_tol * fabs(b + a)) { break; }
         c = (fa * b - fb * a) / (fa - fb);

         fc = f(c);

         if (fabs(fc) < abs_tol) { break; }

         if (fc * fb > 0)
         {
            /* fc and fb have same sign, copy c to b */
            b = c; fb = fc;
            if (side == -1)
            {
               fa /= 2;
            }
            side = -1;
         }
         else if (fa * fc > 0)
         {
            /* fc and fa have same sign, copy c to a */
            a = c; fa = fc;
            if (side == +1)
            {
               fb /= 2;
            }
            side = +1;
         }
         else
         {
            break;
         }
      }
   }
};
class BregmanDykstraDG : public IterativeSolver
{
private:
   QuadratureSpace &qspace;
   FiniteElementSpace &fes;
   ADEntropy &entropy;
   const int vdim;
   mutable DifferentiableCoefficient entropy_cf;
   VectorCoefficient &primal_cf;
   mutable GridFunction latent;
   mutable GridFunction latent_k;
   mutable GridFunction gradient;
   mutable QuadratureFunction gradient_qf;
   mutable GridFunction latent_freeze;
   L2toQF &l2toqf;
   std::vector<ADFunction*> constraints;
   mutable std::vector<std::unique_ptr<DifferentiableCoefficient>> constraint_cfs;
   std::vector<real_t> targets;
   mutable std::vector<std::unique_ptr<GridFunction>> qi;
   int max_sub_iter;
   real_t lower=-1e04, upper=1e04;
public:
   BregmanDykstraDG(QuadratureSpace &qspace_, FiniteElementSpace &fes_,
                    L2toQF &map,
                    ADEntropy &entropy_)
      : qspace(qspace_)
      , fes(fes_)
      , entropy(entropy_)
      , vdim(entropy.width)
      , entropy_cf(entropy), primal_cf(entropy_cf.Gradient())
      , latent(&fes), latent_k(&fes)
      , latent_freeze(&fes)
      , gradient(&fes)
      , gradient_qf(qspace_, vdim)
      , l2toqf(map)
   {
      MFEM_VERIFY(fes.GetVDim() == vdim,
                  "BregmanDykstraDG: fes vdim does not match entropy width");
      MFEM_VERIFY(fes.GetOrdering() == Ordering::byNODES,
                  "BregmanDykstraDG: fes must have byNODES ordering");
      entropy_cf.AddInput(&latent);
#ifdef MFEM_USE_MPI
      ParMesh *mesh = dynamic_cast<ParMesh*>(qspace.GetMesh());
      if (mesh) { SetComm(mesh->GetComm()); }
#endif
      rel_tol = 1e-08;
      abs_tol = 1e-08;
      max_iter = 1000;
      max_sub_iter = 1000;
      width = height = fes.GetTrueVSize();
   }
   void SetMaxSubIter(int n) { max_sub_iter = n; }

   void AddConstraint(ADFunction &constraint, real_t target=0.0)
   {
      MFEM_VERIFY(constraint.width == vdim,
                  "Constraint input dimension does not match.");
      constraints.push_back(&constraint);
      constraint_cfs.push_back(
         std::make_unique<DifferentiableCoefficient>(constraint));
      constraint_cfs.back()->AddInput(&primal_cf);
      targets.push_back(target);

      qi.push_back(std::make_unique<GridFunction>(&fes));
   }

   /// @brief Project latent0 onto the intersection of the convex sets defined by the constraints
   /// latent0 and opt_latent can be the same vector
   void Mult(const Vector &latent0, Vector &opt_latent) const override
   {
      MFEM_VERIFY(latent0.Size() == width,
                  "Input vector size does not match operator width.");
      Vector zero_vec(vdim);
      zero_vec = 0.0;
      VectorConstantCoefficient zero_cf(zero_vec);
      opt_latent.SetSize(height);
      latent.SetData(opt_latent.GetData());
      latent = latent0;
      Vector constraint_val(constraints.size());
      Vector qi_norm(constraints.size());
      for (int j=0; j<constraints.size(); j++)
      {
         constraint_val[j] = qspace.Integrate(*constraint_cfs[j]) - targets[j];
         *qi[j] = 0.0;
      }
      num_const_eval = 0;
      for (final_iter=0; final_iter<max_iter; final_iter++)
      {
         for (int j=0; j<constraints.size(); j++)
         {
            latent_k = latent;
            real_t cval = qspace.Integrate(*constraint_cfs[j]) - targets[j];
            constraint_cfs[j]->Gradient().Project(gradient_qf);
            l2toqf.AdjointMult(gradient_qf, gradient);
            real_t target = dot(GetComm(), gradient, primal_cf, qspace) - cval;
            TangentProjection(j, target);
            latent_k -= latent;
            *qi[j] += latent_k;
         }
         for (int j=0; j<constraints.size(); j++)
         {
            constraint_val[j] = qspace.Integrate(*constraint_cfs[j]) - targets[j];
            qi_norm[j] = qi[j]->ComputeL2Error(zero_cf);
         }
         if (constraint_val.Max() < abs_tol)
         {
            break;
         }
      }
      if (print_level > 0)
      {
         out << "BregmanDykstra: Completed in " << final_iter+1
             << " iterations with " << num_const_eval
             << " constraint evaluations. Max constraint violation = "
             << constraint_val.Max() << std::endl;
      }
   }
   int NumConstraintEvals() const { return num_const_eval; }
   // Set the search interval [new_lower, new_upper]
   // If the bounds do not bracket the root, they will be shifted.
   void SetBracket(real_t new_lower, real_t new_upper)
   {
      MFEM_VERIFY(new_lower < new_upper,
                  "BregmanDykstra: lower bound must be less than upper bound");
      lower = new_lower;
      upper = new_upper;
   }
   // Set the search interval [-new_bound, new_bound]
   // If the bounds do not bracket the root, they will be shifted.
   void SetBracket(real_t new_bound)
   {
      MFEM_VERIFY(new_bound > 0,
                  "BregmanDykstra: lower bound must be less than upper bound");
      lower = -new_bound;
      upper = new_bound;
   }

private:
   using IterativeSolver::Dot;
   mutable int num_const_eval;

   void TangentProjection(const int i, const real_t target) const
   {
      real_t c, fc;
      int n, side = 0;
      latent_freeze = latent;
      auto f = [&](real_t x)
      {
         num_const_eval++;
         add(latent_freeze, x, gradient, latent);
         return qspace.Integrate(*constraint_cfs[i]) - target;
      };
      real_t f0 = f(0.0);
      if (f0 < abs_tol) { return; }
      real_t a = lower, b = upper;
      real_t fa = f(a);
      real_t fb = f(b);
      if (print_level > 1)
      {
         out << "  f(0) = " << f0
             << " f(" << a << ") = " << fa
             << " f(" << b << ") = " << fb
             << std::endl;
      }
      /// Shift bounds if necessary
      if (fa * fb > 0)
      {
         if (print_level > 1)
         {
            out << "  Initial bounds do not bracket root, shifting..." << std::endl;
         }
         real_t diff = b - a;
         if (fa > 0) // both positive
         {
            while (fa > 0)
            {
               b = a; fb = fa;
               a -= diff;
               fa = f(a);
            }
         }
         else // both negative
         {
            while (fb < 0)
            {
               a = b; fa = fb;
               b += diff;
               fb = f(b);
            }
         }
         if (print_level > 1)
         {
            out << "  f(0) = " << f0 << " f(a) = " << fa << " f(b) = " << fb << std::endl;
         }
      }


      for (n = 0; n < max_sub_iter; n++)
      {
         if (fabs(b - a) < rel_tol * fabs(b + a)) { break; }
         c = (fa * b - fb * a) / (fa - fb);

         fc = f(c);

         if (fabs(fc) < abs_tol) { break; }

         if (fc * fb > 0)
         {
            /* fc and fb have same sign, copy c to b */
            b = c; fb = fc;
            if (side == -1)
            {
               fa /= 2;
            }
            side = -1;
         }
         else if (fa * fc > 0)
         {
            /* fc and fa have same sign, copy c to a */
            a = c; fa = fc;
            if (side == +1)
            {
               fb /= 2;
            }
            side = +1;
         }
         else
         {
            break;
         }
      }
   }
};


} // namespace mfem
