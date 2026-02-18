/// Example 3: AD Linear Elasticity with Vector FE
#include "mfem.hpp"
#include "logger.hpp"
#include "ad_intg.hpp"

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

   int order = 2;
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

   // Mesh mesh = rhs_fun_circle
   Mesh ser_mesh = Mesh::MakeCartesian2D(30, 10,
                                         Element::QUADRILATERAL, false, 3.0, 1.0);
   ParMesh mesh(MPI_COMM_WORLD, ser_mesh);
   const int dim = mesh.Dimension();
   for (int i = 0; i < ref_levels; i++)
   {
      mesh.UniformRefinement();
   }
   VectorFunctionCoefficient load_cf(dim, [dim](const Vector &x, Vector &y)
   {
      y = 0.0;
      y(dim-1) = 1.0;
   });

   H1_FECollection fec(order, dim);
   ParFiniteElementSpace fes(&mesh, &fec, dim);
   ParFiniteElementSpace fes_scalar(&mesh, &fec);

   Array<int> is_bdr_ess(mesh.bdr_attributes.Max());
   is_bdr_ess = 0;
   is_bdr_ess[3] = 1;
   Array<int> ess_tdof_list;
   fes.GetEssentialTrueDofs(is_bdr_ess, ess_tdof_list);

   real_t E(1.0), nu(0.3);
   real_t lambda(E*nu / (1.0 - nu*nu));
   real_t mu(E / (2.0 * (1.0 + nu)));
   LinearElasticityEnergy energy(dim, lambda, mu);

   ParNonlinearForm nlf(&fes);
   nlf.AddDomainIntegrator(
      new ADNonlinearFormIntegrator<ADEval::GRAD | ADEval::VECTOR>(energy));
   nlf.SetEssentialBC(is_bdr_ess);

   ParLinearForm load(&fes);
   load.AddDomainIntegrator(new VectorDomainLFIntegrator(load_cf));
   load.Assemble();
   HypreParVector x(&fes), b(&fes);
   load.ParallelAssemble(b);

   ParGridFunction u(&fes); u = 0.0;
   u.GetTrueDofs(x);
   ParGridFunction ux(&fes_scalar, u.GetData());
   ParGridFunction uy(&fes_scalar, u.GetData() + fes_scalar.GetVSize());

   CGSolver lin_solver(comm);
   HypreBoomerAMG prec;
   prec.SetPrintLevel(0);
   lin_solver.SetPreconditioner(prec);
   lin_solver.SetPrintLevel(0);
   lin_solver.SetRelTol(1e-09);
   lin_solver.SetAbsTol(1e-12);
   lin_solver.SetMaxIter(1e08);

   NewtonSolver solver(comm);
   solver.SetSolver(lin_solver);
   solver.SetOperator(nlf);
   solver.SetPrintLevel(2);
   b.SetSubVector(ess_tdof_list, 0.0);
   solver.Mult(b, x);
   u.SetFromTrueDofs(x);

   GLVis glvis("localhost", 19916);
   glvis.Append(ux, "ux", "Rjc");
   glvis.Append(uy, "uy", "Rjc");
   glvis.Update();
   return 0;
}
