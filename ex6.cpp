/// Example 6: Advection-Diffusion problem
#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "src/logger.hpp"
#include "src/ad_intg.hpp"
#include "src/pg.hpp"
#include "src/tools.hpp"

using namespace std;
using namespace mfem;

struct MixedDiffusionFunctional : public ADFunction
{
public:
   MixedDiffusionFunctional(int dim)
      : ADFunction(dim + 1 + 1) // q, div q, u
   {}

   AD_IMPL(T, V, M, q_divq_u,
   {
      const T* q_divq_u_data = q_divq_u.GetData();
      const int dim = q_divq_u.Size() - 2;
      const T* q = q_divq_u_data;
      const T& div_q = *(q_divq_u_data + dim);
      const T& u = *(q_divq_u_data + dim + 1);
      T result = T();
      for (int i=0; i<dim; i++)
      {
         result += q[i]*q[i];
      }
      result = result*0.5 - u*div_q;
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
   // file name to be saved
   std::stringstream filename;
   filename << "ad-mixed-diffusion-";

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

   // Mesh mesh = rhs_fun_circle
   Mesh ser_mesh = Mesh::MakeCartesian2D(10, 10,
                                         Element::QUADRILATERAL);
   const int dim = ser_mesh.Dimension();
   for (int i = 0; i < ref_levels; i++)
   {
      ser_mesh.UniformRefinement();
   }
   ParMesh mesh(comm, ser_mesh);
   // Setup boundary conditions. All Dirichlet boundary conditions
   // -> Weakly imposed
   int num_bdr_attr = mesh.bdr_attributes.Max();
   Array<int> is_bdr_ess_q(num_bdr_attr);
   is_bdr_ess_q = 0;
   Array<int> is_bdr_ess_u(num_bdr_attr);
   is_bdr_ess_u = 0;
   Array<Array<int>*> is_bdr_ess{&is_bdr_ess_q, &is_bdr_ess_u};
   // RHS
   FunctionCoefficient load_cf([](const Vector &x)
   {
      return 2*M_PI * M_PI * std::sin(M_PI * x(0)) * std::sin(M_PI * x(1));
   });

   RT_FECollection q_fec(order, dim);
   L2_FECollection u_fec(order, dim);
   L2_FECollection latent_fec(order-1, dim);
   ParFiniteElementSpace q_fes(&mesh, &q_fec);
   ParFiniteElementSpace u_fes(&mesh, &u_fec);
   Array<ParFiniteElementSpace*> fespaces{&q_fes, &u_fes};
   Array<int> offsets = GetOffsets(fespaces);
   Array<int> true_offsets = GetTrueOffsets(fespaces);
   BlockVector x(offsets), b(offsets);
   BlockVector tx(true_offsets), tb(true_offsets);
   x = 0.0;
   ParGridFunction q(&q_fes, x, offsets[0]), u(&u_fes, x, offsets[1]);
   q.MakeTRef(&q_fes, tx.GetBlock(0).GetData());
   u.MakeTRef(&u_fes, tx.GetBlock(1).GetData());
   ParLinearForm load(&u_fes, b.GetBlock(1).GetData());
   load.AddDomainIntegrator(new DomainLFIntegrator(load_cf));
   load.Assemble();
   load.ParallelAssemble(tb.GetBlock(1));
   tb.GetBlock(1).Neg(); // energy functional changes the sign

   MixedDiffusionFunctional energy(dim);
   // FermiDiracEntropy entropy(0.0, 1.0);
   // ADPGFunctional pg_functional(energy, entropy, dim+1);

   ParBlockNonlinearForm nlf(fespaces);
   constexpr auto q_mode = ADEval::VALUE | ADEval::DIV | ADEval::VECFE;
   constexpr auto u_mode = ADEval::VALUE;
   nlf.AddDomainIntegrator(new
                           ADBlockNonlinearFormIntegrator<q_mode, u_mode>
                           (energy));
   Array<Vector*> dummies(2);
   nlf.SetEssentialBC(is_bdr_ess, dummies);

   NewtonSolver solver(comm);
   MUMPSMonoSolver lin_solver(comm);
   solver.SetSolver(lin_solver);
   solver.SetOperator(nlf);
   solver.SetAbsTol(1e-10);
   solver.SetRelTol(1e-10);
   IterativeSolver::PrintLevel print_level;
   print_level.iterations = 1;
   solver.SetPrintLevel(print_level);
   solver.SetMaxIter(10);
   solver.Mult(tb, tx);
   q.SetFromTrueVector();
   u.SetFromTrueVector();

   GLVis glvis("localhost", 19916);
   glvis.Append(q, "q", "Rjc");
   glvis.Append(u, "u", "Rjc");
   glvis.Update();

   return 0;
}
