/// Example 6: AD Diffusion
#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "src/logger.hpp"
#include "src/ad_intg.hpp"
#include "src/pg.hpp"
#include "src/tools.hpp"

using namespace std;
using namespace mfem;

struct DamProblem2D : public ADVectorFunction
{
   real_t slope;
   DamProblem2D(real_t a, real_t y)
      : ADVectorFunction(3, 3)
   {
      MFEM_VERIFY(a > 0 && y >= 0,
                  "DamProblem2D: a must be positive and y must be non-negative");
      slope = a / y; // slope of the dam
   }

   AD_VEC_IMPL(T, V, M, u_gradu, F,
   {
      const T* gradu = &u_gradu[1];
      F.SetSize(3);
      F[0] = 0; // value of u. No gradient
      F[1] = gradu[0] - slope*gradu[1];
      F[2] = gradu[1] + slope*gradu[0];
   });
};

int main(int argc, char *argv[])
{
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();
   MPI_Comm comm = MPI_COMM_WORLD;
   // file name to be saved
   std::stringstream filename;
   filename << "ad-dam-";

   int order = 1;
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

   Mesh ser_mesh("./data/sloped_rectangle.mesh");
   const int dim = ser_mesh.Dimension();
   for (int i = 0; i < ref_levels; i++)
   {
      ser_mesh.UniformRefinement();
   }
   ParMesh mesh(comm, ser_mesh);
   FunctionCoefficient load_cf([](const Vector &x)
   {
      return 1.0;
   });

   H1_FECollection h1_fec(order+1, dim);
   L2_FECollection l2_fec(order-1, dim);
   ParFiniteElementSpace h1_fes(&mesh, &h1_fec);
   ParFiniteElementSpace l2_fes(&mesh, &l2_fec);
   Array<ParFiniteElementSpace*> fespaces{&h1_fes, &l2_fes};

   const int numBdrAttr = mesh.bdr_attributes.Max();
   Array<int> is_bdr_ess_u(numBdrAttr), is_bdr_ess_psi(numBdrAttr);
   is_bdr_ess_u = 1; is_bdr_ess_psi = 0;
   Array<Array<int>*> is_bdr_ess{&is_bdr_ess_u, &is_bdr_ess_psi};

   Array<int> offsets = GetOffsets(fespaces);
   Array<int> toffsets = GetTrueOffsets(fespaces);
   BlockVector x(offsets), b(offsets);
   BlockVector tx(toffsets), tb(toffsets);
   tx = 0.0; tb = 0.0;

   ParGridFunction u(&h1_fes, x, offsets[0]), psi(&l2_fes, x, offsets[1]),
                   psi_k(&l2_fes);
   u.MakeTRef(&h1_fes, x, offsets[0]); psi.MakeTRef(&l2_fes, x, offsets[1]);
   u.SetFromTrueVector(); psi.SetFromTrueVector();

   DamProblem2D obj_functional(1.0, 1.0);
   ShannonEntropy entropy(0.0);
   ADPGFunctional pg_functional(obj_functional, entropy, psi_k);

   ParBlockNonlinearForm nlf(fespaces);
   nlf.AddDomainIntegrator(new
                           ADBlockNonlinearFormIntegrator<ADEval::GRAD | ADEval::VALUE, ADEval::VALUE>
                           (pg_functional));

   ParLinearForm load(&h1_fes, b.GetData());
   load.AddDomainIntegrator(new DomainLFIntegrator(load_cf));
   load.Assemble();
   load.ParallelAssemble(tb.GetBlock(0));

   Array<Vector*> rhs({&tb.GetBlock(0), &tb.GetBlock(1)});
   nlf.SetEssentialBC(is_bdr_ess, rhs);

   NewtonSolver solver(comm);
   MUMPSMonoSolver lin_solver(comm);
   solver.SetSolver(lin_solver);
   solver.SetOperator(nlf);
   solver.SetAbsTol(1e-10);
   solver.SetRelTol(1e-10);
   solver.iterative_mode = true;
   IterativeSolver::PrintLevel print_level;
   print_level.iterations = true;
   solver.SetPrintLevel(print_level);

   DifferentiableCoefficient entropy_cf(entropy);
   entropy_cf.AddInput(psi);
   VectorCoefficient &Upsi_cf = entropy_cf.Gradient();

   QuadratureSpace visspace(&mesh, order+3);
   QuadratureFunction Upsi(&visspace);

   Upsi_cf.Project(Upsi);

   GLVis glvis("localhost", 19916);
   glvis.Append(u, "u", "Rjc");
   glvis.Append(Upsi, "U(psi)", "Rjc");
   glvis.Update();

   for (int i=0; i<100; i++)
   {
      pg_functional.SetAlpha(std::pow(i+1., 2.));
      psi_k = psi;
      psi_k.SetTrueVector();

      solver.Mult(tb, tx);
      u.SetFromTrueVector();
      psi.SetFromTrueVector();
      Upsi_cf.Project(Upsi);
      glvis.Update();
   }


   return 0;
}
