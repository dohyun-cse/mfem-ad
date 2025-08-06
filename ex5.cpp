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
#include "src/ad_intg.hpp"
#include "src/tools.hpp"
#include "src/pg.hpp"

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
   int dim;
   MassEnergy l2_energy;
   DiffEnergy l2_diff_sqrd;
   VolumeFunctional volume;
   MassFunctional mass;
   InternalEnergyFunctional internal_energy;
   TotalEnergyFunctional total_energy;
   MomentumFunctional momentum[3];
   Vector con_target; // will be divided by domain volume as this is point-functional
   const Vector &lambda;
   const real_t &mu;

   RemapALFunctional(VectorCoefficient &x0, Vector &con_target,
                     Vector &lambda, real_t &mu, real_t domain_volume)
      : ADFunction(x0.GetVDim())
      , dim(opt_type == 3 ? con_target.Size() - 3 :
            -1) // dim is only necessary for opt_type == 3
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
   { con_target *= 1.0 / domain_volume; }

   void ProcessParameters(ElementTransformation &Tr,
                          const IntegrationPoint &ip) const override
   {
      l2_diff_sqrd.ProcessParameters(Tr, ip);
      /* Constraints don't need to process parameters
      volume.ProcessParameters(Tr, ip);
      mass.ProcessParameters(Tr, ip);
      internal_energy.ProcessParameters(Tr, ip);
      total_energy.ProcessParameters(Tr, ip);
      for (int i=0; i<dim; i++) { momentum[i].ProcessParameters(Tr, ip); }
      */
   }

   // Evaluate lambda*c(x) + (mu/2)*c(x)^2
   template <typename T, typename V>
   T evalAL(const ADFunction &c, V &x, int idx) const
   {
      T cx = c(x) - con_target[idx];
      return cx*(lambda[idx] + mu*0.5*cx);
   }

   AD_IMPL(T, V, M, x,
   {
      T result = l2_diff_sqrd(x); // (1/2)*||x-x0||^2
      if constexpr (opt_type >= 0) { result += evalAL<T>(volume, x, 0); }
      if constexpr (opt_type >= 1) { result += evalAL<T>(mass, x, 1); }
      if constexpr (opt_type == 2) { result += evalAL<T>(internal_energy, x, 2); }
      else if constexpr (opt_type > 2) { result += evalAL<T>(total_energy, x, 2); }
      if constexpr (opt_type == 3)
      {
         for (int i=0; i<dim; i++) { result += evalAL<T>(momentum[i], x, 3 + i); }
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
   auto qfespace_and_fec = QSpaceToFESpace(qspace);
   ParFiniteElementSpace &qf_fes = static_cast<ParFiniteElementSpace&>
                                   (*std::get<0>(qfespace_and_fec));
   ParFiniteElementSpace l2_fes(&mesh, &l2_fec);
   ParFiniteElementSpace h1_fes(&mesh, &h1_fec);

   Array<ParFiniteElementSpace*> fespaces{&qf_fes, &qf_fes, &l2_fes, &h1_fes, &h1_fes};
   const int numVars = fespaces.Size();

   VectorArrayCoefficient x0_cf(numVars);
   std::vector<std::unique_ptr<GridFunction>> x(numVars);
   std::vector<std::unique_ptr<GridFunction>> x0(numVars);
   std::vector<std::unique_ptr<GridFunction>> x_min(numVars);
   std::vector<std::unique_ptr<GridFunction>> x_max(numVars);
   std::vector<std::unique_ptr<GridFunction>> latent_k(numVars);
   std::vector<std::unique_ptr<FermiDiracEntropy>> dual_entropies(numVars);
   for (int i=0; i<numVars; i++)
   {
      auto fes = fespaces[i];
      x[i] = std::make_unique<GridFunction>(fes);
      *x[i] = 0.0; x[i]->SetTrueVector();
      x0[i] = std::make_unique<GridFunction>(fes);
      *x0[i] = 0.0; x0[i]->SetTrueVector();
      x_min[i] = std::make_unique<GridFunction>(fes);
      *x_min[i] = 0.0; x_min[i]->SetTrueVector();
      x_max[i] = std::make_unique<GridFunction>(fes);
      *x_max[i] = 1.0; x_max[i]->SetTrueVector();
      latent_k[i] = std::make_unique<GridFunction>(fes);
      *latent_k[i] = 0.0; latent_k[i]->SetTrueVector();

      x0_cf.Set(i, new GridFunctionCoefficient(x0[i].get()), true);
      dual_entropies[i] = std::make_unique<FermiDiracEntropy>(
                             *x_min[i], *x_max[i]);
   }
   Vector constraints_rhs(numVars);
   Vector lambda(numVars);
   lambda = 0.0;
   real_t mu(0);
   Vector target(numVars);
   target = 0.0;
   target = 0.0;
   // lambda and mu are passed by reference.
   // Changing them outside will change teh functional, too.
   RemapALFunctional<4> AL_functional(x0_cf, target, lambda, mu, domain_volume);
   std::vector<std::unique_ptr<ADPGFunctional>> energies(numVars);
   // Chain PG functionals
   for (int i=0; i<numVars; i++)
   {
      if (i == 0) { energies[i] = std::make_unique<ADPGFunctional>(AL_functional, *dual_entropies[i], *latent_k[i]); }
      else { energies[i] = std::make_unique<ADPGFunctional>(*energies[i-1], *dual_entropies[i], *latent_k[i]); }
   }

   Vector xvec(AL_functional.n_input);
   xvec = 0.0;
   auto &Tr = *mesh.GetElementTransformation(0);
   const IntegrationPoint &ip = IntRules.Get(Tr.GetGeometryType(), 0).IntPoint(0);
   AL_functional(xvec, Tr, ip);

   return 0;
}
