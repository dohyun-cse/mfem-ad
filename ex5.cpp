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


enum RemapConType
{
   IND, IND_RHO, IND_RHO_E, HYDRO
};
std::vector<std::unique_ptr<ADFunction>>
MakeConstraints(int con_type, int dim=-1)
{
   std::vector<std::unique_ptr<ADFunction>> constraints;
   int numVars;
   if (con_type < HYDRO) { numVars = con_type; }
   else { numVars = con_type + dim - 1; }

   if (con_type >= IND) { constraints.push_back(std::make_unique<VolumeFunctional>(numVars)); }
   if (con_type >= IND_RHO) { constraints.push_back(std::make_unique<MassFunctional>(numVars)); }
   if (con_type >= IND_RHO_E) { constraints.push_back(std::make_unique<InternalEnergyFunctional>(numVars)); }
   if (con_type == HYDRO)
   {
      for (int d=0; d<dim; d++)
      {
         constraints.push_back(std::make_unique<MomentumFunctional>(numVars, d));
      }
   }
   return std::move(constraints);
}

int main(int argc, char *argv[])
{
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();
   MPI_Comm comm = MPI_COMM_WORLD;

   int order = 2;
   int ref_levels = 2;
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
   HYPRE_BigInt glb_siz = 0;
   for (int i=0; i<fespaces.Size(); i++)
   {
      auto fes_size = fespaces[i]->GlobalTrueVSize();
      glb_siz += fes_size;
      out << "FESpace " << i << ": " << fes_size << std::endl;
   }
   out << "Total global size: " << glb_siz << std::endl;
   const int numVars = fespaces.Size();
   Array<int> offsets = GetTrueOffsets(fespaces);
   Array<ParFiniteElementSpace*> all_fespaces(fespaces);
   all_fespaces.Append(fespaces);
   Array<int> all_offsets = GetTrueOffsets(all_fespaces);
   BlockVector x_and_psi(all_offsets);
   BlockVector x_tvec(x_and_psi, offsets);
   BlockVector psi_tvec(x_and_psi.GetData() + offsets[numVars], offsets);
   BlockVector x0_tvec(offsets), latent_k_tvec(offsets);

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
      x0[i]->MakeTRef(fespaces[i], x0_tvec, offsets[i]);
      (*x0[i]) = 0.5;
      x0[i]->SetTrueVector();
      latent_k[i] = std::make_unique<ParGridFunction>(fes);
      latent_k[i]->MakeTRef(fespaces[i], latent_k_tvec, offsets[i]);
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
      x[i]->MakeTRef(fespaces[i], x_tvec, offsets[i]);
      *x[i] = *x0[i];
      x[i]->SetTrueVector();
      latent[i] = std::make_unique<ParGridFunction>(fes);
      latent[i]->MakeTRef(fespaces[i], psi_tvec, offsets[i]);
      *latent[i] = 0.0;
      latent[i]->SetTrueVector();

      primal_begin[i] = i;
   }

   std::vector<std::unique_ptr<ADFunction>> constraints
      = MakeConstraints(HYDRO, dim);
   MassEnergy bare_obj_energy(numVars);
   DiffEnergy obj_energy(bare_obj_energy, x0_cf);
   ALFunctional AL_functional(obj_energy);
   for (int i=0; i<constraints.size(); i++)
   {
      AL_functional.AddEqConstraint(*constraints[i]);
   }
   ADPGFunctional alpg_functional(AL_functional, dual_entropies, latent_k, primal_begin);

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

   Vector con_target(numVars);
   for (int i=0; i<numVars; i++)
   {
      AL_functional.EqConstraintMode(i);
      con_target[i] = bnlf.GetEnergy(x_and_psi) - 1e-06;
      AL_functional.SetEqRHS(i, con_target[i]);
   }
   Vector &lambda = AL_functional.GetLambda();
   real_t &mu = AL_functional.GetPenalty();
   mu = 0.05;

   Vector dummy;
   AL_functional.ObjectiveMode();
   // Initialize psi.
   solver.Mult(dummy, x_and_psi);
   for (int it_PG=0; it_PG<100; it_PG++)
   {
      out << "PG iteration " << it_PG + 1 << std::endl;
      // update alpha
      alpg_functional.SetAlpha(1.0);
      // Set proximal center
      latent_k_tvec = psi_tvec;
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
         out << "    lambda and C = ";
         for (int i=0; i<numVars; i++)
         {
            AL_functional.EqConstraintMode(i);
            real_t cval = bnlf.GetEnergy(x_and_psi);
            lambda[i] += mu*cval;
            out << "(" << lambda[i] << ", " << cval << "), ";
         }
         out << std::endl;
      }
      glvis.Update();
   }
   return 0;
}
