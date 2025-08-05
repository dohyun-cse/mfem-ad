/// Example 5: AD Obstacle Problem with PG
#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "src/logger.hpp"
#include "src/ad_intg.hpp"
#include "src/tools.hpp"

using namespace std;
using namespace mfem;


MAKE_AD_FUNCTION(ObstacleEnergy, T, V, M, x, dummy,
{
   T result = {};
   // First component is u. Others are grad u
   for (int i=1; i<x.Size(); i++)
   {
      result += x[i]*x[i];
   }
   return result*0.5;
});

int main(int argc, char *argv[])
{
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();
   MPI_Comm comm = MPI_COMM_WORLD;
   // file name to be saved
   std::stringstream filename;
   filename << "ad-diffusion";

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

   const int numBdrAttr = mesh.bdr_attributes.Max();
   Array<int> is_bdr_ess1(numBdrAttr);
   is_bdr_ess1 = 1;
   Array<int> is_bdr_ess2(numBdrAttr);
   is_bdr_ess2 = 0;
   Array<Array<int>*> is_bdr_ess{&is_bdr_ess1, &is_bdr_ess2};
   FunctionCoefficient load_cf([](const Vector &x)
   {
      return 2*M_PI * M_PI * std::sin(M_PI * x(0)) * std::sin(M_PI * x(1));
   });
   ObstacleEnergy obj_energy(1 + dim);
   FermiDiracEntropy entropy(1, 2);
   ADPGEnergy pg_energy(obj_energy, entropy);

   H1_FECollection h1_fec(order+1, dim);
   L2_FECollection l2_fec(order-1, dim);
   ParFiniteElementSpace h1_fes(&mesh, &h1_fec);
   ParFiniteElementSpace l2_fes(&mesh, &l2_fec);
   QuadratureSpace visspace(&mesh, order+3);
   const IntegrationRule &ir = IntRules.Get(Geometry::Type::SQUARE, 3*order + 3);

   Array<int> ess_tdof_list;
   h1_fes.GetEssentialTrueDofs(is_bdr_ess1, ess_tdof_list);

   Array<int> offsets(3);
   offsets[0] = 0;
   offsets[1] = h1_fes.GetTrueVSize();
   offsets[2] = l2_fes.GetTrueVSize();
   offsets.PartialSum();
   BlockVector x_and_psi(offsets);

   ParGridFunction x(&h1_fes), psi(&l2_fes);
   QuadratureFunction x_mapped(&visspace);
   MappedGridFunctionCoefficient x_mapped_cf(&psi, [](const real_t x) { return std::exp(x); });
   ParGridFunction psik(psi);

   x.MakeTRef(&h1_fes, x_and_psi.GetBlock(0).GetData());
   x = 0.0; x.SetTrueVector();

   psi.MakeTRef(&l2_fes, x_and_psi.GetBlock(1).GetData());
   psi = 0.0; psi.SetTrueVector();

   ConstantCoefficient alpha(1.0);
   GridFunctionCoefficient psik_cf(&psik);
   VectorArrayCoefficient params(4);
   params.Set(0, new ConstantCoefficient(0.0), true); // lower bound
   params.Set(1, new ConstantCoefficient(0.5), true); // upper bound
   params.Set(2, &psik_cf, false); // psi
   params.Set(3, &alpha, false); // step_size

   Array<ParFiniteElementSpace*> fespaces{&h1_fes, &l2_fes};
   ParBlockNonlinearForm bnlf(fespaces);
   constexpr ADEval u_mode = ADEval::VALUE | ADEval::GRAD;
   constexpr ADEval psi_mode = ADEval::VALUE;
   bnlf.AddDomainIntegrator(
      new ADBlockNonlinearFormIntegrator<true, u_mode, psi_mode>(
         pg_energy, params, &ir)
   );

   BlockVector rhs(offsets);
   ParLinearForm b(&h1_fes);
   b.AddDomainIntegrator(new DomainLFIntegrator(load_cf));
   b.Assemble();
   b.ParallelAssemble(rhs.GetBlock(0));
   rhs.GetBlock(0).SetSubVector(ess_tdof_list, 0.0);
   rhs.GetBlock(1) = 0.0;

   Array<Vector*> rhs_list{&rhs.GetBlock(0), &rhs.GetBlock(1)};
   bnlf.SetEssentialBC(is_bdr_ess, rhs_list);


   MUMPSMonoSolver lin_solver(MPI_COMM_WORLD);
   NewtonSolver solver(MPI_COMM_WORLD);
   solver.SetSolver(lin_solver);
   solver.SetOperator(bnlf);
   IterativeSolver::PrintLevel print_level;
   print_level.iterations = true;
   solver.SetPrintLevel(print_level);
   solver.SetAbsTol(1e-09);
   solver.SetRelTol(0.0);
   solver.iterative_mode = true;

   GLVis glvis("localhost", 19916, 400, 350, 2);
   glvis.Append(x, "x", "Rjclmm");
   // glvis.Append(x_mapped, "U(psi)", "Rjclmm");
   glvis.Update();

   for (int i=0; i<100; i++)
   {
      out << "PG iteration " << i + 1 << std::endl;
      alpha.constant = 1.0;
      psik = psi;
      psik.SetTrueVector();
      solver.Mult(rhs, x_and_psi);
      if (solver.GetNumIterations() == 0)
      {
         out << "  Newton Converged without iterations. Terminating." << std::endl;
         out << "PG Converged in " << i + 1
             << " with final residual: " << solver.GetFinalNorm() << std::endl;
         break;
      }
      else
      {
         out << "  Newton converged in " << solver.GetNumIterations()
             << " with residual " << solver.GetFinalNorm() << std::endl;
      }
      x.SetFromTrueVector();
      psi.SetFromTrueVector();
      x_mapped_cf.Project(x_mapped);
      glvis.Update();
   }

   return 0;
}
