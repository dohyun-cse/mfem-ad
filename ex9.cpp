/// Example 9: Topology Optimization
#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "src/logger.hpp"
#include "src/ad_intg.hpp"
#include "src/mmto.hpp"

using namespace std;
using namespace mfem;


int main(int argc, char *argv[])
{
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();
   MPI_Comm comm = MPI_COMM_WORLD;
   // file name to be saved
   std::stringstream filename;
   filename << "ad-elasticity-";

   int order = 1;
   int ref_levels = 3;
   bool visualization = false;
   bool paraview = false;
   real_t radius = 0.05;

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
   Mesh ser_mesh = Mesh::MakeCartesian2D(2, 1,
                                         Element::QUADRILATERAL, false, 2.0, 1.0);
   const int dim = ser_mesh.Dimension();
   for (int i = 0; i < ref_levels; i++)
   {
      ser_mesh.UniformRefinement();
   }
   ParMesh mesh(MPI_COMM_WORLD, ser_mesh);
   ser_mesh.Clear();

   Vector load_point{1.9, 0.5};
   VectorFunctionCoefficient load_cf(dim, [dim, load_point](const Vector &x,
                                     Vector &y)
   {
      y.SetSize(dim); y = 0.0;
      if (x.DistanceTo(load_point) < 0.05)
      { y(dim-1) = -1.0; }
   });
   Array<int> is_bdr_ess(mesh.bdr_attributes.Max());
   is_bdr_ess = 0;
   is_bdr_ess[3] = 1;

   Array<int> is_ctrl_bdr_ess(is_bdr_ess);
   is_ctrl_bdr_ess = 0;

   Vector E{0.0, 1.0, 3.0};
   Vector nu{0.0, 0.3, 0.3};
   Vector density{0.0, 1.0, 1.3};
   MFEM_VERIFY(E.Size() == nu.Size() && E.Size() == density.Size(),
               "E, nu and density must have the same size");
   const int numMaterials = E.Size();

   Vector lambda(E.Size());
   Vector mu(E.Size());
   for (int i=0; i<E.Size(); i++)
   {
      lambda[i] = E[i]*nu[i] / ((1.0 + nu[i])*(1.0 - 2.0*nu[i]));
      mu[i] = E[i] / (2.0*(1.0 + nu[i]));
   }
   SIMPFunction lambda_ad(lambda, 3.0);
   SIMPFunction mu_ad(mu, 3.0);
   SimplexEntropy entropy(numMaterials, 1.0);

   H1_FECollection state_fec(order, dim);
   L2_FECollection ctrl_fec(order-1, dim);
   H1_FECollection fltr_fec(order, dim);
   ParFiniteElementSpace state_fes(&mesh, &state_fec, dim);
   ParFiniteElementSpace ctrl_fes(&mesh, &ctrl_fec, E.Size());
   ParFiniteElementSpace ctrl_scalar_fes(&mesh, &ctrl_fec);
   ParFiniteElementSpace fltr_fes(&mesh, &fltr_fec, E.Size());
   ParFiniteElementSpace fltr_scalar_fes(&mesh, &fltr_fec);
   QuadratureSpace qspace(&mesh, 2*order+1);

   Array<int> ess_tdof_list;
   state_fes.GetEssentialTrueDofs(is_bdr_ess, ess_tdof_list, -1);

   ParGridFunction displacement(&state_fes);
   displacement = 0.0;
   ParGridFunction psi(&ctrl_fes);
   psi = 0.0;
   ParGridFunction feta(&fltr_fes);
   feta = 0.0;
   feta.SetTrueVector();
   std::vector<ParGridFunction> psi_i(numMaterials);
   std::vector<ParGridFunction> feta_i(numMaterials);
   for (int i=0; i<numMaterials; i++)
   {
      psi_i[i].MakeRef(&ctrl_scalar_fes, psi, i*ctrl_scalar_fes.GetVSize());
      feta_i[i].MakeRef(&fltr_scalar_fes, feta, i*fltr_scalar_fes.GetVSize());
   }

   DifferentiableCoefficient eta_cf(entropy);
   eta_cf.AddInput(&psi);
   VectorConstantCoefficient rho_vec_cf(density);
   InnerProductCoefficient rho_cf(rho_vec_cf, eta_cf.Gradient());

   DifferentiableCoefficient lambda_cf(lambda_ad);
   lambda_cf.AddInput(&feta);
   VectorCoefficient &dlambda_cf = lambda_cf.Gradient();

   DifferentiableCoefficient mu_cf(mu_ad);
   mu_cf.AddInput(&feta);
   VectorCoefficient &dmu_cf = mu_cf.Gradient();

   ParBilinearForm state_form(&state_fes);
   state_form.AddDomainIntegrator(new ElasticityIntegrator(lambda_cf, mu_cf));
   ParLinearForm load(&state_fes);
   load.AddDomainIntegrator(new VectorDomainLFIntegrator(load_cf));
   load.Assemble();
   std::unique_ptr<Vector> load_vec(load.ParallelAssemble());

   ParBilinearForm filter_form(&fltr_scalar_fes);
   ConstantCoefficient eps_cf(radius*radius/std::pow(2*std::sqrt(3.0), 2.0));
   filter_form.AddDomainIntegrator(new DiffusionIntegrator(eps_cf));
   filter_form.AddDomainIntegrator(new MassIntegrator);
   filter_form.Assemble();
   filter_form.Finalize();
   std::unique_ptr<HypreParMatrix> filter_mat(filter_form.ParallelAssemble());
   ParLinearForm filter_targs(&fltr_fes);
   filter_targs.AddDomainIntegrator(new VectorDomainLFIntegrator(
                                       eta_cf.Gradient()));

   VectorGradientGridFunction grad_displacement(displacement);
   std::vector<std::unique_ptr<LinearElasticityEnergy>> energies(numMaterials);
   std::vector<std::unique_ptr<DifferentiableCoefficient>> energy_cf(numMaterials);
   std::vector<std::unique_ptr<ParLinearForm>> denergy_forms(numMaterials);
   std::vector<std::unique_ptr<Vector>> energy_vecs(numMaterials);
   for (int i=0; i<numMaterials; i++)
   {
      // not ideal, but works for now
      // it will evaluate numMaterials*numMaterials times the same thing
      // at quadrature points
      energies[i] = std::make_unique<LinearElasticityEnergy>(dim, &dlambda_cf,
                    &dmu_cf, i);
      energy_cf[i] = std::make_unique<DifferentiableCoefficient>(*energies[i]);
      energy_cf[i]->AddInput(&grad_displacement);
      denergy_forms[i] = std::make_unique<ParLinearForm>(&fltr_scalar_fes);
      denergy_forms[i]->AddDomainIntegrator(new DomainLFIntegrator(*energy_cf[i]));
      energy_vecs[i] = std::make_unique<Vector>(fltr_scalar_fes.GetTrueVSize());
   }

   GLVis glvis("localhost", 19916, 400, 350, 4);
   glvis.Append(lambda_cf, qspace, "lambda", "Rjc");
   glvis.Append(rho_cf, qspace, "rho", "Rjc");
   glvis.Update();
   return 0;
}
