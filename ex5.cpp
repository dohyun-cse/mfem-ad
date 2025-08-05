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

std::tuple<std::unique_ptr<ADFunction>, std::vector<std::unique_ptr<ADFunction>>>
MakeConstraints(const Vector &target, Vector &lambda, real_t &mu, const int dim,
                std::vector<std::unique_ptr<GridFunction>> &x_min,
                std::vector<std::unique_ptr<GridFunction>> &x_max,
                std::vector<std::unique_ptr<GridFunction>> &latent_k)
{
   std::vector<std::unique_ptr<ADFunction>> temps;

   const int numVars = 3 + dim;
   std::vector<std::unique_ptr<ADFunction>> const_list;
   const_list.emplace_back(std::make_unique<VolumeFunctional>(numVars));
   const_list.emplace_back(std::make_unique<MassFunctional>(numVars));
   const_list.emplace_back(std::make_unique<TotalEnergyFunctional>(numVars));
   for (int i=0; i<dim; i++)
   {
      const_list.emplace_back(std::make_unique<MomentumFunctional>(numVars, i));
   }
   std::vector<std::unique_ptr<ShiftedADFunction>> constraints(numVars);
   for (int i=0; i<numVars; i++)
   {
      constraints[i] = std::make_unique<ShiftedADFunction>(*const_list[i],
                       - target[i]);
   }

   auto mu_ad = std::make_unique<ReferenceConstantADFunction>(mu, numVars);
   auto half_mu_ad = std::make_unique<ScaledADFunction>(*mu_ad,0.5);

   std::vector<std::unique_ptr<ReferenceConstantADFunction>> lambda_ref(numVars);
   std::vector<std::unique_ptr<ProductADFunction>> lagrangians(numVars);
   std::vector<std::unique_ptr<ProductADFunction>> sqrd_constraints(numVars);
   std::vector<std::unique_ptr<ProductADFunction>> penalties(numVars);
   for (int i=0; i<numVars; i++)
   {
      lambda_ref[i] = std::make_unique<ReferenceConstantADFunction>(lambda[i],
                      numVars);
      lagrangians[i] = std::make_unique<ProductADFunction>(*lambda_ref[i],
                       *constraints[i]);
      sqrd_constraints[i] = std::make_unique<ProductADFunction>(*constraints[i],
                            *constraints[i]);
      penalties[i] = std::make_unique<ProductADFunction>(*half_mu_ad,
                     *sqrd_constraints[i]);
   }

   auto mass_energy = std::make_unique<MassEnergy>(numVars);
   std::vector<std::unique_ptr<SumADFunction>> TotalLagrangian(numVars);
   std::vector<std::unique_ptr<SumADFunction>> TotalAL(numVars);
   for (int i=0; i<numVars; i++)
   {
      if (i == 0)
      {
         TotalLagrangian[i] = std::make_unique<SumADFunction>(*mass_energy,
                              *lagrangians[i]);
      }
      else
      {
         TotalLagrangian[i] = std::make_unique<SumADFunction>(*TotalAL[i-1],
                              *lagrangians[i]);
      }
      TotalAL[i] = std::make_unique<SumADFunction>(*TotalLagrangian[i],
                   *penalties[i]);
   }
   std::vector<std::unique_ptr<FermiDiracEntropy>> entropy_list(numVars);
   std::vector<std::unique_ptr<ADPGEnergy>> PGAL_energy_list(numVars);
   for (int i=0; i<numVars; i++)
   {
      entropy_list[i] = std::make_unique<FermiDiracEntropy>(*x_min[i], *x_max[i]);
      if (i == 0)
      {
         PGAL_energy_list[i] = std::make_unique<ADPGEnergy>(ADPGEnergy(
                                  *TotalAL[numVars-1], *entropy_list[i], *latent_k[i], i));
      }
      else
      {
         PGAL_energy_list[i] = std::make_unique<ADPGEnergy>(*PGAL_energy_list[i-1],
                               *entropy_list[i], *latent_k[i], i);
      }
   }
   for (auto &c : const_list) { temps.push_back(std::move(c)); }
   for (auto &c : constraints) { temps.push_back(std::move(c)); }
   temps.push_back(std::move(mu_ad));
   temps.push_back(std::move(half_mu_ad));
   for (auto &l : lambda_ref) { temps.push_back(std::move(l)); }
   for (auto &l : lagrangians) { temps.push_back(std::move(l)); }
   for (auto &p : sqrd_constraints) { temps.push_back(std::move(p)); }
   for (auto &p : penalties) { temps.push_back(std::move(p)); }
   temps.push_back(std::move(mass_energy));
   for (auto &L : TotalLagrangian) { temps.push_back(std::move(L)); }
   for (auto &AL : TotalAL) { temps.push_back(std::move(AL)); }
   for (auto &e : entropy_list) { temps.push_back(std::move(e)); }
   for (auto &p : PGAL_energy_list) { temps.push_back(std::move(p)); }

   std::unique_ptr<ADFunction> final_func = std::move(temps.back());
   temps.pop_back();

   return {std::move(final_func), std::move(temps)};
}


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

   std::vector<std::unique_ptr<GridFunction>> x;
   std::vector<std::unique_ptr<GridFunction>> x_min;
   std::vector<std::unique_ptr<GridFunction>> x_max;
   std::vector<std::unique_ptr<GridFunction>> latent_k;
   for (auto *fes : fespaces)
   {
      x.emplace_back(std::make_unique<GridFunction>(fes));
      x_min.emplace_back(std::make_unique<GridFunction>(fes));
      x_max.emplace_back(std::make_unique<GridFunction>(fes));
      latent_k.emplace_back(std::make_unique<GridFunction>(fes));
      *x.back() = 0.0;
      *x_min.back() = 0.0;
      *x_max.back() = 1.0;
      *latent_k.back() = 0.0;
   }
   Vector const_targets(numVars);
   Vector lambda(numVars);
   real_t mu;

   auto obj_and_temp = MakeConstraints(const_targets, lambda, mu, dim, x_min,
                                       x_max, latent_k);
   ADFunction &pg_energy = *std::get<0>(obj_and_temp);

   Vector xvec(pg_energy.n_input);
   xvec = 0.0;

   auto &Tr = *mesh.GetElementTransformation(0);
   const IntegrationPoint &ip = IntRules.Get(Tr.GetGeometryType(), 0).IntPoint(0);
   pg_energy(xvec, Tr, ip);

   return 0;
}
