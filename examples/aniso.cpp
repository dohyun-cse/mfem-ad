/// Example 3: AD Linear Elasticity with Vector FE
#include "ad_intg.hpp"
#include "ad_native.hpp"
#include "logger.hpp"
#include "mfem.hpp"
#include <linalg/hypre.hpp>

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
   int ref_levels = 0;
   bool visualization = false;
   bool paraview = false;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
   args.AddOption(&ref_levels, "-r", "--ref", "Refinement levels");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable visualization, default is false");
   args.AddOption(&paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                  "Enable Paraview Export. Default is false");
   args.ParseCheck();

   // Mesh mesh = rhs_fun_circle
   Mesh ser_mesh =
      Mesh::MakeCartesian2D(30, 10, Element::QUADRILATERAL, false, 3.0, 1.0);
   ParMesh mesh(MPI_COMM_WORLD, ser_mesh);
   const int dim = mesh.Dimension();
   for (int i = 0; i < ref_levels; i++)
   {
      mesh.UniformRefinement();
   }
   VectorFunctionCoefficient load_cf(dim, [dim](const Vector &x, Vector &y)
   {
      y = 0.0;
      y(dim - 1) = 1.0;
   });

   H1_FECollection fec(order, dim);
   ParFiniteElementSpace fes_state(&mesh, &fec, dim);
   ParFiniteElementSpace fes_scalar(&mesh, &fec);
   ParFiniteElementSpace fes_filter(&mesh, &fec, 4);

   ParGridFunction u(&fes_state);
   u = 0.0;
   HypreParVector x(&fes_state), b(&fes_state);
   u.GetTrueDofs(x);
   ParGridFunction ux(&fes_scalar, u.GetData());
   ParGridFunction uy(&fes_scalar, u.GetData() + fes_scalar.GetVSize());

   Array<int> is_bdr_ess(mesh.bdr_attributes.Max());
   is_bdr_ess = 0;
   is_bdr_ess[3] = 1;
   Array<int> ess_tdof_list;
   fes_state.GetEssentialTrueDofs(is_bdr_ess, ess_tdof_list);

   ParGridFunction feta(&fes_filter);
   // eta = [v, s, a, b]
   Vector eta_val({0.0, 1.0, 0.0, 0.0});
   VectorConstantCoefficient eta_cf(eta_val);
   feta.ProjectCoefficient(eta_cf);

   real_t E_v(1e-06), nu_v(0.3); // void material property
   real_t E_s(1.0), nu_s(0.3); // isotropic solid material property
   real_t E_x(0.5), E_y(2.0), nu_xy(0.3); // orthotropic solid material property
   // E_x/(1-nu_xy*nu_yx)), E_y/nu_xy, 0
   // E_x/nu_yx, E_y/(1-nu_xy*nu_yx)), 0
   // 0,                0,             G_xy
   real_t nu_yx = nu_xy * E_y / E_x;
   real_t G_xy = sqrt(E_x * E_y) / (2 * (1.0 + sqrt(nu_xy * nu_yx)));
   ParametrizedAnisoLinElasticity2DEnergy energy(E_v, nu_v, E_s, nu_s, E_x, E_y,
                                                 nu_xy, G_xy, feta);


   ParLinearForm load(&fes_state);
   load.AddDomainIntegrator(new VectorDomainLFIntegrator(load_cf));
   load.Assemble();
   load.ParallelAssemble(b);

   ParNonlinearForm nlf(&fes_state);

   nlf.SetGradientType(Operator::Type::Hypre_ParCSR);

   nlf.AddDomainIntegrator(
      new ADNonlinearFormIntegrator<ADEval::GRAD | ADEval::VECTOR>(energy));


   HypreParMatrix &A = static_cast<HypreParMatrix&>(nlf.GetGradient(x));
   A.EliminateRowsCols(ess_tdof_list, x, b);

   // 14. Define and apply a parallel PCG solver for A X = B with the BoomerAMG
   //     preconditioner from hypre.
   HypreBoomerAMG amg(A);
   amg.SetSystemsOptions(dim, true);
   amg.SetPrintLevel(0);
   HyprePCG pcg(A);
   pcg.SetTol(1e-8);
   pcg.SetMaxIter(1e04);
   pcg.SetPrintLevel(0);
   pcg.SetPreconditioner(amg);
   pcg.Mult(b, x);
   u.SetFromTrueDofs(x);

   GLVis glvis("localhost", 19916);
   glvis.Append(u, "ux", "Rjc");
   glvis.Append(ux, "ux", "Rjc");
   glvis.Append(uy, "uy", "Rjc");
   glvis.Update();
}
