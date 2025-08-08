/// Remap objectives
///
/// TODO:
/// It is very tidius to manage artifacts when adding/subtracting...
/// There must be a better design :(
/// See, MakeConstraints and operators in ADFunction.
#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "src/logger.hpp"
#include "src/pg.hpp"
#include "src/dof_pg.hpp"
#include "src/tools.hpp"
#include <cmath>

using namespace std;
using namespace mfem;

struct VolumeFunctional : public ADFunction
{
   VolumeFunctional(int n_input) : ADFunction(n_input) {}
   AD_IMPL(T, V, M, x, return x[0];);
};

struct MassFunctional : public ADFunction
{
   MassFunctional(int n_input) : ADFunction(n_input) {}
   AD_IMPL(T, V, M, x, return x[0]*x[1];);
};

struct InternalEnergyFunctional : public ADFunction
{
   InternalEnergyFunctional(int n_input) : ADFunction(n_input) {}
   AD_IMPL(T, V, M, x, return x[0]*x[1]*x[2];);
};

struct TotalEnergyFunctional : public ADFunction
{
   TotalEnergyFunctional(int n_input) : ADFunction(n_input) {}
   AD_IMPL(T, V, M, x,
   {
      T result = T();
      for (int i=3; i<x.Size(); i++)
      {
         result += x[i]*x[i];
      }
      result = result*0.5;
      result = result + x[2];
      result = result*(x[0]*x[1]);
      return result;
   });
};

struct MomentumFunctional : public ADFunction
{
   int comp;
   MomentumFunctional(int n_input, int comp) : ADFunction(n_input), comp(comp) {}
   AD_IMPL(T, V, M, x, return x[0]*x[1]*x[3+comp];);
};

template <int opt_type>
struct RemapALFunctional : public ADFunction
{
   int al_eval_mode = -1; // -1: full AL, 0: objective, >0: constraint comp
   int dim;
   real_t domain_volume;
   MassEnergy l2_energy;
   DiffEnergy l2_diff_sqrd;
   VolumeFunctional volume;
   MassFunctional mass;
   InternalEnergyFunctional internal_energy;
   TotalEnergyFunctional total_energy;
   MomentumFunctional momentum[3];
   std::vector<ADFunction*> constraints;
   Vector con_target; // will be divided by domain volume as this is point-functional
   const Vector &lambda;
   const real_t &mu;

   RemapALFunctional(VectorCoefficient &x0, Vector &con_target,
                     Vector &lambda, real_t &mu, real_t domain_volume)
      : ADFunction(x0.GetVDim())
      , dim(con_target.Size()) // dim is only necessary for opt_type == 3
      , domain_volume(domain_volume)
      , l2_energy(n_input)
      , l2_diff_sqrd(l2_energy, x0)
      , volume(n_input), mass(n_input)
      , internal_energy(n_input)
      , total_energy(n_input)
      , momentum{ MomentumFunctional(n_input,0),
                  MomentumFunctional(n_input,1),
                  MomentumFunctional(n_input,2) }
      , con_target(con_target)
      , lambda(lambda)
      , mu(mu)
   {
      if (opt_type >= 0) { constraints.push_back(&volume); }
      if (opt_type >= 1) { constraints.push_back(&mass); }
      if (opt_type == 2) { constraints.push_back(&internal_energy); }
      else if (opt_type > 2) { constraints.push_back(&total_energy); }
      if (opt_type == 3)
      {
         if (dim >= 1) { constraints.push_back(&momentum[0]); }
         if (dim >= 2) { constraints.push_back(&momentum[1]); }
         if (dim >= 3) { constraints.push_back(&momentum[2]); }
      }
      con_target *= 1.0 / domain_volume;
   }

   void SetTarget(Vector &con_target)
   {
      MFEM_VERIFY(con_target.Size() == this->con_target.Size(),
                  "RemapALFunctional: con_target size mismatch");
      this->con_target = con_target;
      this->con_target *= 1.0 / domain_volume;
   }

   // evaluate the Augmented Lagrangian functional
   void ALMode() { this->al_eval_mode = -1; }
   // evaluate only the objective part
   void ObjectiveMode() { this->al_eval_mode = 0; }
   // evaluate only the specific component of the constraint
   // That is, it will return c(x)
   void ConstraintMode(int comp)
   {
      MFEM_VERIFY(comp >= 0 && comp < lambda.Size(),
                  "RemapALFunctional: comp must be in [0, n_input)");
      this->al_eval_mode = comp + 1;

   }

   void ProcessParameters(ElementTransformation &Tr,
                          const IntegrationPoint &ip) const override
   {
      if (al_eval_mode <= 0) { l2_diff_sqrd.ProcessParameters(Tr, ip); }
   }

   // Evaluate lambda*c(x) + (mu/2)*c(x)^2
   template <typename T, typename V>
   T evalAL(V &x, int idx) const
   {
      T cx = (*constraints[idx])(x) - con_target[idx];
      if (al_eval_mode > 0) { return cx; }
      return cx*(lambda[idx] + mu*0.5*cx);
   }

   AD_IMPL(T, V, M, x,
   {
      if (int comp = al_eval_mode > 0)
      {
         return evalAL<T>(x, comp);
      }
      T result = l2_diff_sqrd(x); // (1/2)*||x-x0||^2
      if (al_eval_mode == 0) { return result; }
      for (int i = 0; i < constraints.size(); i++)
      {
         result += evalAL<T>(x, i);
      }
      return result;
   });
};

int main(int argc, char *argv[])
{
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();
   MPI_Comm comm = MPI_COMM_WORLD;

   int order = 3;
   int ref_levels = 3;
   bool visualization = false;
   bool paraview = false;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element polynomial degree");
   args.AddOption(&ref_levels, "-r", "--ref", "Refinement levels");
   args.AddOption(&visualization, "-vis", "--visualization",
                  "-no-vis", "--no-visualization",
                  "Enable visualization, default is false");
   args.AddOption(&paraview, "-pv", "--paraview",
                  "-no-pv", "--no-paraview",
                  "Enable Paraview Export. Default is false");
   args.ParseCheck();
   if (myid != 0) { out.Disable(); }

   // Mesh mesh = rhs_fun_circle
   Mesh ser_mesh = Mesh::MakeCartesian2D(10, 10,
                                         Element::QUADRILATERAL);
   const int dim = ser_mesh.Dimension();
   for (int i = 0; i < ref_levels; i++)
   {
      ser_mesh.UniformRefinement();
   }
   ParMesh mesh(MPI_COMM_WORLD, ser_mesh);
   real_t domain_volume = 0.0;
   for (int i=0; i<mesh.GetNE(); i++) { domain_volume += mesh.GetElementVolume(i); }

   L2_FECollection l2_fec(order, dim);
   H1_FECollection h1_fec(order, dim);

   QuadratureSpace qspace(&mesh, order);
   const IntegrationRule &ir = qspace.GetIntRule(0);
   // Convert QuadratureSpace to ParFiniteElementSpace
   // shapes and nodes will not be used. only indexing purpose
   auto qfespace_and_fec = QSpaceToFESpace(qspace);
   ParFiniteElementSpace &qf_fes = static_cast<ParFiniteElementSpace&>
                                   (*std::get<0>(qfespace_and_fec));
   MFEM_VERIFY(qspace.GetSize() == qf_fes.GetTrueVSize(),
               "QuadratureSpace and ParFiniteElementSpace sizes differ: "
               << qspace.GetSize() << " != " << qf_fes.GetTrueVSize()
              );
   ParFiniteElementSpace l2_fes(&mesh, &l2_fec);
   ParFiniteElementSpace h1_fes(&mesh, &h1_fec);

   Array<ParFiniteElementSpace*> fespaces{&qf_fes, &qf_fes, &l2_fes};
   for (int i=0; i<dim; i++)
   {
      fespaces.Append(&h1_fes);
   }
   const int numVars = fespaces.Size();
   Array<int> offsets = GetTrueOffsets(fespaces);
   Array<ParFiniteElementSpace*> all_fespaces(fespaces);
   all_fespaces.Append(fespaces);
   Array<int> all_offsets = GetTrueOffsets(all_fespaces);
   BlockVector x_and_psi(all_offsets);
   BlockVector x_vec(x_and_psi, offsets);
   BlockVector psi_vec(x_and_psi.GetData() + offsets[numVars], offsets);
   BlockVector x0_vec(offsets), latent_k_vec(offsets);

   VectorArrayCoefficient x0_cf(numVars);
   std::vector<std::unique_ptr<GridFunction>> x(numVars);
   std::vector<std::unique_ptr<GridFunction>> latent(numVars);

   std::vector<std::unique_ptr<GridFunction>> x0(numVars);
   std::vector<std::unique_ptr<GridFunction>> x_min(numVars);
   std::vector<std::unique_ptr<GridFunction>> x_max(numVars);
   std::vector<std::unique_ptr<GridFunction>> latent_k(numVars);
   std::vector<std::unique_ptr<ADEntropy>> dual_entropies(numVars);
   std::vector<int> primal_begin(numVars);
   for (int i=0; i<numVars; i++)
   {
      auto fes = fespaces[i];

      x0[i] = std::make_unique<ParGridFunction>(fes);
      x0[i]->MakeTRef(fespaces[i], x0_vec, offsets[i]);
      (*x0[i]) = 0.5;
      x0[i]->SetTrueVector();
      latent_k[i] = std::make_unique<ParGridFunction>(fes);
      latent_k[i]->MakeTRef(fespaces[i], latent_k_vec, offsets[i]);
      *latent_k[i] = 0.0;
      latent_k[i]->SetTrueVector();

      x_min[i] = std::make_unique<ParGridFunction>(fes);
      *x_min[i] = 0.0; x_min[i]->SetTrueVector();
      x_max[i] = std::make_unique<ParGridFunction>(fes);
      *x_max[i] = 1.0; x_max[i]->SetTrueVector();

      x0_cf.Set(i, new GridFunctionCoefficient(x0[i].get()), true);
      dual_entropies[i] = std::make_unique<FermiDiracEntropy>(
                             *x_min[i], *x_max[i]);

      // x and latent's T-vector will be set later
      x[i] = std::make_unique<ParGridFunction>(fes);
      x[i]->MakeTRef(fespaces[i], x_vec, offsets[i]);
      *x[i] = *x0[i];
      x[i]->SetTrueVector();
      latent[i] = std::make_unique<ParGridFunction>(fes);
      latent[i]->MakeTRef(fespaces[i], psi_vec, offsets[i]);
      *latent[i] = 0.0;
      latent[i]->SetTrueVector();

      primal_begin[i] = i;
   }

   Vector constraints_rhs(numVars);
   Vector lambda(numVars);
   lambda = 0.0;
   real_t mu(1e-02);
   Vector con_target(numVars);
   con_target = 0.0;
   // lambda and mu are passed by reference.
   // Changing them outside will change teh functional, too.
   RemapALFunctional<4> AL_functional(x0_cf, con_target, lambda, mu,
                                      domain_volume);
   MassEnergy bare_obj_energy(numVars);
   DiffEnergy obj_energy(bare_obj_energy, x0_cf);
   ADPGFunctional alpg_functional(AL_functional, dual_entropies, latent_k,
                                  primal_begin);

   ParBlockNonlinearForm bnlf(all_fespaces);
   // For ADDofPGnlfi, we only set primal mode
   constexpr ADEval qfmode = ADEval::QVALUE;
   constexpr ADEval gfmode = ADEval::VALUE;
   switch (dim)
   {
      case 2:
         bnlf.AddDomainIntegrator(new
                                  ADDofPGNonlinearFormIntegrator<qfmode, qfmode, gfmode, gfmode, gfmode>
                                  (alpg_functional, &ir));
         break;
      case 3:
         bnlf.AddDomainIntegrator(new
                                  ADDofPGNonlinearFormIntegrator<qfmode, qfmode, gfmode, gfmode, gfmode, gfmode>
                                  (alpg_functional, &ir));
         break;
      default:
         MFEM_ABORT("Unsupported dimension: " << dim);
   }

   MUMPSMonoSolver lin_solver(MPI_COMM_WORLD);
   NewtonSolver solver(MPI_COMM_WORLD);
   solver.SetSolver(lin_solver);
   solver.SetOperator(bnlf);
   IterativeSolver::PrintLevel print_level;
   print_level.Iterations().Summary();
   solver.SetPrintLevel(print_level);
   solver.SetAbsTol(1e-09);
   solver.SetRelTol(0.0);
   solver.SetMaxIter(20);
   solver.iterative_mode = true;

   GLVis glvis("localhost", 19916, 400, 350, 4);
   for (int i=0; i<numVars; i++)
   {
      glvis.Append(*x[i], std::to_string(i).c_str(), "Rjclmm");
   }

   for (int i=0; i<numVars; i++)
   {
      AL_functional.ConstraintMode(i);
      con_target[i] += bnlf.GetEnergy(x_and_psi) + 1e-06;
   }
   AL_functional.SetTarget(con_target);

   Vector dummy;
   for (int it_PG=0; it_PG<100; it_PG++)
   {
      out << "PG iteration " << it_PG + 1 << std::endl;
      // update alpha
      alpg_functional.SetAlpha(1.0);
      // Set proximal center
      latent_k_vec = psi_vec;
      for (int i=0; i<numVars; i++) { latent_k[i]->SetFromTrueVector(); }

      for (int it_AL=0; it_AL<100; it_AL++)
      {
         out << "  AL iteration " << it_AL + 1 << std::endl;
         // solve AL subproblem
         AL_functional.ALMode();
         solver.Mult(dummy, x_and_psi);
         for (int i=0; i<numVars; i++)
         {
            x[i]->SetFromTrueVector();
            latent[i]->SetFromTrueVector();
         }
         out << "    lambda = ";
         for (int i=0; i<numVars; i++)
         {
            AL_functional.ConstraintMode(i);
            real_t cval = bnlf.GetEnergy(x_and_psi);
            out << "    Constraint " << i << " : " << cval << std::endl;
            lambda[i] += mu*cval;
            out << lambda[i] << ", ";
         }
         out << std::endl;
      }
      glvis.Update();
   }
   return 0;
}
