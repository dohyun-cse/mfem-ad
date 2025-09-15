#pragma once
#include "mfem.hpp"
// #include "pg.hpp"
#include "tools.hpp"
#include "ad_intg.hpp"
#include "diffobj.hpp"

namespace mfem
{
/**
 * @brief A Helmholtz filter operator for topology optimization.
 *
 * This class implements a filter based on solving a component-wise Helmholtz-type
 * partial differential equation:
 *              (I - ε²Δ)u_filtered = u_unfiltered in Ω
 *                            du/dn = 0 on ∂Ω
 *                               ε² = r² / 12.
 *
 * The operator is self-adjoint, so the AdjointMult method is identical to the
 * forward Mult method.
 *
 */
class HelmholtzFilter : public ForwardBackwardOperator
{
private:
   QuadratureSpace &qspace;
   const int vdim;
   std::unique_ptr<HypreParMatrix> helmholtz_op;
   std::unique_ptr<HypreParMatrix> helmholtz_elim_op;
   CGSolver helmholtz_solver;
   HypreBoomerAMG helmholtz_prec;
   mutable QuadratureFunction target_qf;
   mutable ParLinearForm helmholtz_lf;
   mutable Vector helmholtz_rhs;
   mutable Vector helmholtz_sol;
   mutable ParGridFunction helmholtz_gf;
   Array<int> ess_tdofs;
   BlockVector *sol_tvec;
   mutable bool forward_mode = true;
public:
   HelmholtzFilter(QuadratureSpace &qs,
                   ParFiniteElementSpace &fes,
                   const real_t filter_radius,
                   const int vdim = 1,
                   Array<int> ess_bdr=Array<int>(),
                   BlockVector* sol_tvec=nullptr)
      : ForwardBackwardOperator(qs.GetSize()*vdim)
      , qspace(qs)
      , vdim(vdim)
      , helmholtz_solver(fes.GetComm())
      , helmholtz_prec()
      , target_qf(qs)
      , helmholtz_lf(&fes)
      , helmholtz_rhs(fes.GetTrueVSize())
      , helmholtz_sol(fes.GetTrueVSize())
      , helmholtz_gf(&fes)
      , sol_tvec(sol_tvec)
   {
      MFEM_VERIFY(fes.GetVDim() == 1,
                  "Helmholtz filter base space must be scalar.");
      MFEM_VERIFY(dynamic_cast<const H1_FECollection*>(fes.FEColl()) != nullptr,
                  "Helmholtz filter base space must be H1.");
      ParBilinearForm helmholtz(&fes);
      ConstantCoefficient eps_cf(filter_radius*filter_radius/12.0);
      if (ess_bdr.Size() > 0) { fes.GetEssentialTrueDofs(ess_bdr, ess_tdofs); }
      else { ess_tdofs.SetSize(0); }
      helmholtz.AddDomainIntegrator(new MassIntegrator());
      helmholtz.AddDomainIntegrator(new DiffusionIntegrator(eps_cf));
      helmholtz.Assemble();
      helmholtz.Finalize();
      helmholtz_op.reset(helmholtz.ParallelAssemble());
      helmholtz_elim_op.reset(helmholtz_op->EliminateRowsCols(ess_tdofs));
      helmholtz_prec.SetPrintLevel(0);
      helmholtz_solver.SetRelTol(1e-08);
      helmholtz_solver.SetAbsTol(0);
      helmholtz_solver.SetMaxIter(1e04);
      helmholtz_solver.SetPrintLevel(0);
      helmholtz_solver.SetOperator(*helmholtz_op);

      helmholtz_lf.AddDomainIntegrator(new DomainQLFIntegrator(target_qf));
   }
   std::string Name() const override { return std::string("HelmholtzFilter"); }

   /**
    * @brief Applies the Helmholtz filter to the input vector `x`.
    *
    * @param[in] x Input vector representing the unfiltered field defined on the quadrature space.
    * @param[out] y Output vector representing the filtered field.
    */
   void Mult(const Vector &x, Vector &y) const override
   {
      MFEM_VERIFY(x.Size() == this->Width(),
                  "Input vector size does not match operator width.");
      y.SetSize(this->Height());
      QuadratureFunction x_qf(&qspace, x.GetData(), vdim);
      QuadratureFunction y_qf(&qspace, y.GetData(), vdim);
      real_t* sol_data;
      const int N = helmholtz_sol.Size();
      helmholtz_sol.StealData(&sol_data);
      for (int i=0; i<vdim; i++)
      {
         ExtractComponent(x_qf, i, target_qf);
         helmholtz_lf.Assemble();
         helmholtz_lf.ParallelAssemble(helmholtz_rhs);
         if (sol_tvec && forward_mode)
         {
            helmholtz_sol.SetDataAndSize(sol_tvec->GetBlock(i).GetData(),
                                         sol_tvec->GetBlock(i).Size());
            helmholtz_op->EliminateBC(*helmholtz_elim_op, ess_tdofs, helmholtz_sol,
                                      helmholtz_rhs);
         }
         else
         {
            helmholtz_sol.SetDataAndSize(sol_data, N);
            helmholtz_sol = 0.0;
            helmholtz_rhs.SetSubVector(ess_tdofs, 0.0);
         }
         helmholtz_solver.Mult(helmholtz_rhs, helmholtz_sol);
         helmholtz_gf.SetFromTrueDofs(helmholtz_sol);
         target_qf.ProjectGridFunction(helmholtz_gf);
         SetComponent(target_qf, i, y_qf);
      }
      helmholtz_sol.SetDataAndSize(sol_data, N);
      helmholtz_sol.MakeDataOwner();
   }

   /**
    * @brief Solves the adjoint system.
    * @details Since the Helmholtz operator is self-adjoint, this operation is
    * identical to the forward `Mult` operation.
    *
    * @param[in] x Input vector (right-hand side of the adjoint system).
    * @param[out] y Output vector (solution of the adjoint system).
    */
   void AdjointMult(const Vector &dJdy, Vector &dJdx) const override
   {
      forward_mode = false;
      Mult(dJdy, dJdx);
      forward_mode = true;
   }
};

// forward: Interpolate L2 to Quadrature space
// backward: Project Quadrature to L2 space
class L2Projector : public ForwardBackwardOperator
{
private:
   QuadratureSpace &qspace;
   mutable GridFunction gf;
   mutable QuadratureFunction qf;
   mutable LinearForm lf;
   BilinearForm l2_inv_bf;
   Array<int> gf_toffsets;
   Array<int> qf_toffsets;
   const int vdim;
public:
   L2Projector(QuadratureSpace &qs,
               ParFiniteElementSpace &fes,
               const int vdim = 1)
      : ForwardBackwardOperator(qs.GetSize()*vdim, fes.GetTrueVSize()*vdim)
      , qspace(qs)
      , gf(&fes)
      , qf(qspace)
      , l2_inv_bf(&fes)
      , gf_toffsets(vdim+1)
      , vdim(vdim)
   {
      MFEM_VERIFY(fes.GetVDim() == 1,
                  "L2Projector base space must be scalar.");
      MFEM_VERIFY(dynamic_cast<const L2_FECollection*>(fes.FEColl()) != nullptr,
                  "L2Projector base space must be L2.");
      gf_toffsets = fes.GetTrueVSize();
      gf_toffsets[0] = 0;
      gf_toffsets.PartialSum();
      qf_toffsets = qs.GetSize();
      qf_toffsets[0] = 0;
      qf_toffsets.PartialSum();
      l2_inv_bf.AddDomainIntegrator(new InverseIntegrator(new MassIntegrator()));
      l2_inv_bf.Assemble();
      lf.AddDomainIntegrator(new DomainQLFIntegrator(qf));
   }
   std::string Name() const override { return std::string("HelmholtzFilter"); }

   /**
    * @brief Interpolate from L2 space to quadrature space.
    *
    * @param[in] x Input vector in L2 space (ordering = NODES, x_1, .., x_N, y_1, ...)
    * @param[out] y Output vector in quadrature space (ordering = VDIM, x_1, y_1, ...,)
    */
   void Mult(const Vector &x, Vector &y) const override
   {
      MFEM_VERIFY(x.Size() == this->Width(),
                  "Input vector size does not match operator width.");
      y.SetSize(this->Height());
      BlockVector x_block(x.GetData(), gf_toffsets);
      BlockVector y_block(y.GetData(), qf_toffsets);
      QuadratureFunction y_qf(&qspace, y.GetData(), vdim);
      for (int i=0; i<vdim; i++)
      {
         gf.SetData(x_block.GetBlock(i).GetData());
         qf.ProjectGridFunction(gf);
         SetComponent(qf, i, y_qf);
      }
   }

   /**
    * @brief Solves the adjoint system.
    * @details Since the Helmholtz operator is self-adjoint, this operation is
    * identical to the forward `Mult` operation.
    *
    * @param[in] x Input vector (right-hand side of the adjoint system).
    * @param[out] y Output vector (solution of the adjoint system).
    */
   void AdjointMult(const Vector &dJdy, Vector &dJdx) const override
   {
      MFEM_VERIFY(dJdy.Size() == this->Height(),
                  "Input vector size does not match operator height.");
      dJdx.SetSize(this->Width());
      BlockVector dJdx_block(dJdx.GetData(), gf_toffsets);
      QuadratureFunction dJdy_qf(&qspace, dJdy.GetData(), vdim);
      for (int i=0; i<vdim; i++)
      {
         ExtractComponent(dJdy_qf, i, qf);
         lf.Assemble();
         l2_inv_bf.Mult(lf, dJdx_block.GetBlock(i));
      }
   }
};

// @brief SIMP: E = E0 + sum_{i=1}^n E_i * eta_i^p
class SIMPFunction : public ADVectorFunction
{
private:
   real_t p;
   const DenseMatrix &E;
public:
   /* @brief SIMP material interpolation function
    * Each column of E corresponds to a material property.
   *
   *  @param p The SIMP exponent
   *  @param E Material Properties, size (num_properties, num_materials)
   */
   SIMPFunction(const real_t p, const DenseMatrix &E)
      : ADVectorFunction(E.Height(), E.Width()), p(p), E(E)
   { }
   void SetSIMPExponent(const real_t new_p) { p = new_p; }
   AD_VEC_IMPL(T, V, M, x, result,
   {
      result.SetSize(E.Height());
      for (int property_idx=0; property_idx<E.Height(); property_idx++)
      {
         result[property_idx] = E(property_idx, 0)*x[0];
         for (int i=1; i<E.Width(); i++)
         {
            result[property_idx] += E(property_idx, i)*pow(x[i], p);
         }
      }
   });
};
// class SIMPStdFunction : public ADVectorFunction
// {
// private:
//    real_t p;
//    const DenseMatrix &E;
// public:
//    /* @brief SIMP material interpolation function
//     * Each column of E corresponds to a material property.
//    *
//    *  @param p The SIMP exponent
//    *  @param E Material Properties, size (num_properties, num_materials)
//    */
//    SIMPStdFunction(const real_t p, const DenseMatrix &E)
//       : ADVectorFunction(E.Height(), E.Width()), p(p), E(E)
//    { }
//    void SetSIMPExponent(const real_t new_p) { p = new_p; }
//    AD_VEC_IMPL(T, V, M, x, result,
//    {
//       result.SetSize(E.Height());
//       for (int property_idx=0; property_idx<height; property_idx++)
//       {
//          result[property_idx] = E(property_idx, width-1);
//          for (int i=width-1; i>=1; i--)
//          {
//             T w = pow(x[i], p);
//             result[property_idx] = (1.0 - w)*E(property_idx, i-1) + w*result[property_idx];
//          }
//       }
//    });
// };

// @brief SIMP: E = E0 + sum_{i=1}^n E_i * eta_i^p
class SIMPBlendFunction : public ADVectorFunction
{
private:
   real_t p;
   const DenseMatrix &E;
   Vector sum_E;
public:
   /* @brief SIMP material interpolation function
    * Each column of E corresponds to a material property.
   *
   *  @param p The SIMP exponent
   *  @param E Material Properties, size (num_properties, num_materials)
   */
   SIMPBlendFunction(const real_t p, const DenseMatrix &E)
      : ADVectorFunction(E.Height(), E.Width()), p(p), E(E)
   {
      sum_E.SetSize(E.Height());
      sum_E = 0.0;
      for (int i=0; i<E.Width(); i++)
      {
         for (int j=0; j<E.Height(); j++)
         {
            sum_E[j] += E(j, i);
         }
      }
   }
   void SetSIMPExponent(const real_t new_p) { p = new_p; }
   AD_VEC_IMPL(T, V, M, x, result,
   {
      result.SetSize(E.Height());
      for (int i=0; i<E.Height(); i++)
      {
         result[i] = E(i, width-1);
         T w = x[width-1];
         for (int j=1; j<width; j++)
         {
            int idx = width - j - 1;
            T l = x(idx);
            T t = w / (l + w);
            result[i] = (1.0 - pow(t,p)) * E(i, idx) + pow(t,p) * result[i];
            w = l + w;
         }
      }
   });
};

class WeightedSumFunction : public ADFunction
{
private:
   const Vector &weights;
public:
   WeightedSumFunction(const Vector &weights)
      : ADFunction(weights.Size()), weights(weights)
   { }
   AD_IMPL(T, V, M, x,
   {
      T result = {};
      for (int i=0; i<weights.Size(); i++)
      {
         result += weights[i]*x[i];
      }
      return result;
   });
};

/** @brief A parametrized state solver
 *
 * This class represents a parametrized state solver that maps a set of
 * material properties defined at quadrature points to a state vector.
 * The forward operation solves a PDE to compute the state given the
 * material properties, while the adjoint operation computes the sensitivity
 *
 * The derived class must implement the pure virtual methods:
 * - SetupPDE: to setup the PDE system based on the material properties.
 * - Mult: to solve the PDE and compute the state.
 * - AdjointMult: to compute the adjoint solution and sensitivities.
 *
 */
class ParametrizedStateSolver : public ForwardBackwardOperator
{

protected:
   Array<FiniteElementSpace*> fespaces;
   QuadratureSpace &qspace;
   int numProperites;
   bool is_linear;

   mutable QuadratureFunction properties;
   ParametrizedADFunction &full_energy;
   DifferentiableCoefficient full_energy_cf; // E(P; u)
   int E_offset;
   std::unique_ptr<ReducedADFunction> state_energy; // E(P_fixed; u)
   MatrixCoefficient &d2E_dPdu; // d^2E/dPdu(P_fixed; u_fixed)(v)
   mutable std::vector<std::unique_ptr<GridFunction>> state_gfs;
   Array<int> true_offsets;
   mutable BlockVector state_tvec;
   mutable BlockVector state_lvec;
   mutable std::unique_ptr<BlockNonlinearForm> state_form; // a(P; u, v)
   mutable std::unique_ptr<ParMonolithicBlockNonlinearFormWrapper> mono_state_form;
   mutable BlockVector state_loads;
   mutable std::unique_ptr<Solver> pde_solver;
   bool solve_adjoint;
   // adjoint states. If solve_adjoint is false, it's data points corresponding primal states
   mutable std::vector<std::unique_ptr<GridFunction>> adjoint_gfs;
   mutable BlockVector adjoint_tvec;
   mutable BlockVector adjoint_lvec;
   mutable Evaluator adjoint_evaluator;
   // adjoint pde solver. If null, assume pde is self-adjoint and use pde_solver
   // This solver should solve MultTranspose
   mutable std::unique_ptr<Solver> pde_adjoint_solver;
   // Setup E_cf's input and adjoint_evaluator
   virtual void SetupCFInput() = 0;
   virtual void SetupPDE() = 0;
   virtual void SetupSolvers() = 0;
public:
   ParametrizedStateSolver(Array<FiniteElementSpace*> spaces,
                           QuadratureSpace &parameter_qspace,
                           ParametrizedADFunction &energy,
                           int numProperties_,
                           bool solve_adjoint_pde)
      : ForwardBackwardOperator(GetTrueOffsets(spaces).Last(),
                                parameter_qspace.GetSize()*numProperties_)
      , fespaces(spaces)
      , qspace(parameter_qspace)
      , numProperites(numProperties_)
      , properties(&qspace, nullptr, numProperites)
      , full_energy(energy)
      , full_energy_cf(full_energy)
      , E_offset(energy.width - numProperties_)
      , state_energy(full_energy.GetStateFunction({&properties}))
   , d2E_dPdu(full_energy_cf.Hessian(E_offset, -1, 0, E_offset))
   , solve_adjoint(solve_adjoint_pde)
   {
      true_offsets = GetTrueOffsets(spaces);
      Array<int> offsets = GetOffsets(spaces);
      state_tvec.Update(true_offsets);
      state_tvec = 0.0;
      state_lvec.Update(offsets);
      state_lvec = 0.0;
      for (int i=0; i<spaces.Size(); i++)
      {
         state_gfs.push_back(NewGridFunction(
                                *spaces[i], state_lvec.GetBlock(i)));
      }
      if (solve_adjoint)
      {
         adjoint_tvec.Update(true_offsets);
         adjoint_lvec.Update(offsets);
         adjoint_tvec = 0.0;
         adjoint_lvec = 0.0;
      }
      else
      {
         adjoint_tvec.Update(state_tvec.GetData(), true_offsets);
         adjoint_lvec.Update(state_lvec.GetData(), offsets);
      }
      for (int i=0; i<spaces.Size(); i++)
      {
         adjoint_gfs.push_back(NewGridFunction(
                                  *spaces[i], adjoint_lvec.GetBlock(i)));
      }
   }
   BlockNonlinearForm &GetStateForm()
   {
      if (!state_form) { SetupPDE(); }
      return *state_form;
   }

   std::vector<std::unique_ptr<GridFunction>>& GetStateGridFunctions()
   {
      return state_gfs;
   }

   void Mult(const Vector &parameter, Vector &state) const override
   {
      MFEM_VERIFY(parameter.Size() == Width(),
                  "Input parameter vector size does not match operator width.");
      MFEM_VERIFY(state_form != nullptr,
                  "ParametrizedStateSolver::Mult: PDE form is not set up.");
      MFEM_VERIFY(pde_solver != nullptr,
                  "ParametrizedStateSolver::Mult: PDE solver is not set up.");
      properties.SetData(parameter.GetData());

      Operator *state_op;
      if (mono_state_form) { state_op = mono_state_form.get(); }
      else { state_op = state_form.get(); }
      if (is_linear)
      {
         pde_solver->SetOperator(state_op->GetGradient(state_tvec));
      }
      else
      {
         pde_solver->SetOperator(*state_op);
      }
      pde_solver->Mult(state_loads, state_tvec);
      for (int i=0; i<state_gfs.size(); i++)
      {
         state_gfs[i]->SetFromTrueDofs(state_tvec.GetBlock(i));
      }
      state = state_tvec;
   }

protected:
   void AdjointMult(const Vector &dJdy, Vector &dJdx) const override
   {
      MFEM_VERIFY(dJdy.Size() == Height(),
                  "Input dJdy vector size does not match operator height.");
      MFEM_VERIFY(full_energy_cf.GetNumInputs() > 0,
                  "ParametrizedStateSolver::AdjointMult: E_cf inputs are not set up.");
      MFEM_VERIFY(adjoint_evaluator.GetVDim() > 0,
                  "ParametrizedStateSolver::AdjointMult: adjoint_evaluator is not set up.");
      properties = *x_ptr;

      if (solve_adjoint)
      {
         Operator *state_op;
         if (mono_state_form) { state_op = mono_state_form.get(); }
         else { state_op = state_form.get(); }
         if (pde_adjoint_solver)
         {
            pde_adjoint_solver->SetOperator(state_op->GetGradient(state_tvec));
            pde_adjoint_solver->Mult(dJdy, adjoint_tvec);
         }
         else
         {
            pde_solver->SetOperator(state_op->GetGradient(state_tvec));
            pde_solver->Mult(dJdy, adjoint_tvec);
         }
         MFEM_VERIFY(adjoint_tvec.CheckFinite() == 0,
                     "ParametrizedStateSolver::AdjointMult: adjoint_tvec is not finite.");
         for (int i=0; i<adjoint_gfs.size(); i++)
         {
            adjoint_gfs[i]->SetFromTrueDofs(adjoint_tvec.GetBlock(i));
         }
      }
      // Compute d^2a/dPdu(P, u) * adjoint_solution
      dJdx.SetSize(Width());
      EvaluatorVCF adjoint_vcf(adjoint_evaluator);
      MatrixVectorProductCoefficient sensitivity_cf(d2E_dPdu, adjoint_vcf);
      QuadratureFunction dJdx_qf(&qspace, dJdx.GetData(), numProperites);
      sensitivity_cf.Project(dJdx_qf);
      dJdx_qf.Neg();
   }
};

class ParametrizedElasticitySolver : public ParametrizedStateSolver
{
protected:
   Array<int> state_ess_tdofs;
   HypreBoomerAMG state_prec;
   void SetupCFInput() override
   {
      full_energy_cf.AddInput(new VectorGradientGridFunctionCoefficient(
                                 *state_gfs[0]), true);
      full_energy_cf.AddInput(&properties);
      adjoint_evaluator.Add(new VectorGradientGridFunctionCoefficient(
                               *adjoint_gfs[0]), true);
   }
   void SetupPDE() override
   {
      state_form = NewBlockNonlinearForm(fespaces);
      state_form->AddDomainIntegrator(
         new ADBlockNonlinearFormIntegrator<ADEval::VECTOR | ADEval::GRAD>(*state_energy,
                                                                           &qspace.GetIntRule(0)));
      Array<Vector*> load(1);
      load[0] = &state_loads;
      Array<Array<int>*> state_ess_tdof_list(1);
      state_ess_tdof_list[0] = &state_ess_tdofs;

      state_form->SetEssentialTrueDofs(state_ess_tdof_list, load);
      mono_state_form = std::make_unique<ParMonolithicBlockNonlinearFormWrapper>(
                           static_cast<ParBlockNonlinearForm&>(*state_form));
      is_linear = true;
   }
   void SetupSolvers() override
   {
      MFEM_VERIFY(dynamic_cast<ParBlockNonlinearForm*>(state_form.get()) != nullptr,
                  "ParametrizedElasticitySolver::SetupSolvers: state_form is not a ParBlockNonlinearForm.");
      mono_state_form = std::make_unique<ParMonolithicBlockNonlinearFormWrapper>(
                           dynamic_cast<ParBlockNonlinearForm&>(*state_form));
      auto solver = std::make_unique<CGSolver>(static_cast<ParFiniteElementSpace*>
                                               (fespaces[0])->GetComm());
      state_prec.SetPrintLevel(0);
      solver->SetPreconditioner(state_prec);
      solver->SetRelTol(1e-08);
      solver->SetAbsTol(0.0);
      solver->SetMaxIter(1e04);
      solver->SetPrintLevel(0);
      solver->iterative_mode = true;
      pde_solver = std::move(solver);
   }
public:
   ParametrizedElasticitySolver(FiniteElementSpace &space,
                                QuadratureSpace &parameter_qspace,
                                Array<int> &ess_tdofs,
                                Vector &load,
                                ParametrizedLinearElasticityEnergy &energy,
                                bool solve_adjoint_pde)
      : ParametrizedStateSolver(Array<FiniteElementSpace*> {&space}, parameter_qspace,
   energy, 1, solve_adjoint_pde)
   , state_ess_tdofs(ess_tdofs)
   {
      state_loads.Update(load.GetData(), true_offsets);
      SetupCFInput();
      SetupPDE();
      SetupSolvers();
   }
};
class DirectionalHookesLawBdrIntegrator : public BlockNonlinearFormIntegrator
{
   // properties
private:
   VectorCoefficient *direction;
   real_t k;
   mutable Vector shape;
   mutable DenseMatrix grad_elmat;

protected:
public:
   // methods
private:
protected:
public:
   DirectionalHookesLawBdrIntegrator(const real_t k,
                                     VectorCoefficient *direction)
      : k(k), direction(direction) {}
   void AssembleFaceVector(const Array<const FiniteElement *>&el1,
                           const Array<const FiniteElement *>&dummy_el2,
                           FaceElementTransformations &Tr,
                           const Array<const Vector *> &elfuns,
                           const Array<Vector*> &elvects) override
   {
      Array2D<DenseMatrix*> elmats(1,1);
      elmats(0,0) = &grad_elmat;
      AssembleFaceGrad(el1, dummy_el2, Tr, elfuns, elmats);
      Vector &elvect = *elvects[0];
      elvect.SetSize(grad_elmat.Height());
      grad_elmat.Mult(*elfuns[0], elvect);
   }
   void AssembleFaceGrad(const Array<const FiniteElement *>&el1,
                         const Array<const FiniteElement *>&dummy_el2,
                         FaceElementTransformations &Tr,
                         const Array<const Vector *> &dummy_elfun,
                         const Array2D<DenseMatrix *> &elmats) override
   {
      const FiniteElement &el = *el1[0];
      DenseMatrix &elmat = *elmats(0,0);
      real_t kw;
      int dim = el.GetDim();
      int ndof = el.GetDof();
      Vector dir(dim), nor(dim);

      shape.SetSize(ndof);
      elmat.SetSize(ndof*dim);
      elmat = 0.0;

      const IntegrationRule *ir = &IntRules.Get(Tr.GetGeometryType(),
                                                el.GetOrder()*2);
      MFEM_ASSERT(Tr.Elem2 == NULL, "DirectionalHookesLawBdrIntegrator "
                  "only supports boundary faces.");

      for (int p = 0; p < ir->GetNPoints(); p++)
      {
         const IntegrationPoint &ip = ir->IntPoint(p);

         // Set the integration point in the face and the neighboring elements
         Tr.SetAllIntPoints(&ip);

         // Access the neighboring elements' integration points
         // Note: eip2 will only contain valid data if Elem2 exists
         const IntegrationPoint &eip1 = Tr.GetElement1IntPoint();

         el.CalcPhysShape(*Tr.Elem1, shape);

         direction->Eval(dir, *Tr.Elem1, eip1);

         kw = k*ip.weight;
         for (int d2=0; d2<dim; d2++)
         {
            for (int j = 0; j < ndof; j++)
            {
               for (int d1 = 0; d1 < dim; d1++)
               {
                  for (int i = 0; i < ndof; i++)
                  {
                     elmat(i+d1*ndof, j+d2*ndof) += kw * dir[d1] * dir[d2] * shape(i) * shape(j);
                  }
               }
            }
         }
      }
   }
};


};
