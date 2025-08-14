/// Example 4: AD Obstacle Problem with PG
#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "src/logger.hpp"
#include "src/ad_intg.hpp"
#include "src/tools.hpp"
#include "src/pg.hpp"

using namespace std;
using namespace mfem;


struct GradientObstacleEnergy : public ADFunction
{
   GradientObstacleEnergy(int dim) : ADFunction(dim) {}
   AD_IMPL(T, V, M, x,
   {
      return x*x*0.5;
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
   filename << "ad-obstacle-";
   int rule_type = PGStepSizeRule::RuleType::CONSTANT;
   real_t max_alpha = 1e06;
   real_t alpha0 = 1.0;
   real_t ratio = 1.0;
   real_t ratio2 = 1.0;

   int order = 2;
   int ref_levels = 3;
   bool visualization = false;
   bool paraview = false;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element polynomial degree");
   args.AddOption(&ref_levels, "-r", "--ref", "Refinement levels");
   args.AddOption(&rule_type, "-rule", "--rule",
                  "Step size rule type: 0=CONSTANT, 1=POLY, 2=EXP, 3=DOUBLE_EXP");
   args.AddOption(&max_alpha, "-ma", "--max-alpha",
                  "Maximum step size for PG method");
   args.AddOption(&alpha0, "-a0", "--alpha0",
                  "Initial step size for PG method");
   args.AddOption(&ratio, "-ar", "--alpha-ratio",
                  "Ratio for step size rule (POLY, EXP, DOUBLE_EXP)");
   args.AddOption(&ratio2, "-ar2", "--alpha-ratio2",
                  "Second ratio for DOUBLE_EXP step size rule");
   args.AddOption(&visualization, "-vis", "--visualization",
                  "-no-vis", "--no-visualization",
                  "Enable visualization, default is false");
   args.AddOption(&paraview, "-pv", "--paraview",
                  "-no-pv", "--no-paraview",
                  "Enable Paraview Export. Default is false");
   args.ParseCheck();
   if (myid != 0) { out.Disable(); }
   MFEMInitializePetsc(NULL,NULL,"./src/pgpetsc",NULL);

   PGStepSizeRule alpha_rule(rule_type, alpha0, max_alpha, ratio, ratio2);

   // Mesh mesh = rhs_fun_circle
   Mesh ser_mesh = Mesh::MakeCartesian2D(10, 10,
                                         Element::TRIANGLE);
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
      return 15*std::pow(std::sin(M_PI*x[0]), 2.0);
   });
   GradientObstacleEnergy obj_energy(dim);

   H1_FECollection primal_fec(order, dim);
   H1_FECollection latent_fec(order-1, dim);
   ParFiniteElementSpace primal_fes(&mesh, &primal_fec);
   ParFiniteElementSpace latent_fes(&mesh, &latent_fec, dim);
   QuadratureSpace visspace(&mesh, order+3);

   Array<int> offsets(3);
   offsets[0] = 0;
   offsets[1] = primal_fes.GetTrueVSize();
   offsets[2] = latent_fes.GetTrueVSize();
   offsets.PartialSum();
   BlockVector x_and_latent(offsets);
   x_and_latent = 0.0;

   ParGridFunction u(&primal_fes), latent(&latent_fes);
   u.SetFromTrueDofs(x_and_latent.GetBlock(0));
   latent.SetFromTrueDofs(x_and_latent.GetBlock(1));

   ParGridFunction latent_k(latent);
   latent_k = 0.0; latent_k.SetTrueVector();

   FunctionCoefficient bound([](const Vector &x)
   { return 0.1 + 0.2*x[0] + 0.4*x[1]; });
   HellingerEntropy entropy(dim, &bound);
   ADPGFunctional pg_functional(obj_energy, entropy, latent_k);
   DifferentiableCoefficient entropy_cf(entropy);
   entropy_cf.AddInput(&latent);
   VectorNormCoefficient x_mapped_cf(entropy_cf.Gradient());

   GradientGridFunctionCoefficient gradu_cf(&u);
   VectorNormCoefficient gradu_norm_cf(gradu_cf);

   ConstantCoefficient zero_cf(0.0);
   ParGridFunction lambda(latent), lambda_prev(latent);
   lambda = 0.0; lambda.SetTrueVector();
   VectorGridFunctionCoefficient lambda_prev_cf(&lambda_prev);
   VectorGridFunctionCoefficient lambda_cf(&lambda);
   VectorNormCoefficient lambda_norm_cf(lambda_cf);
   BooleanCoefficient active_set(lambda_norm_cf, [](real_t val) { return val < 1e-06; });

   Array<ParFiniteElementSpace*> fespaces{&primal_fes, &latent_fes};
   ParBlockNonlinearForm bnlf(fespaces);
   constexpr ADEval u_mode = ADEval::GRAD;
   constexpr ADEval psi_mode = ADEval::VALUE | ADEval::VECTOR;
   bnlf.AddDomainIntegrator(
      new ADBlockNonlinearFormIntegrator<u_mode, psi_mode>(
         pg_functional)
   );

   BlockVector rhs(offsets);
   ParLinearForm b(&primal_fes);
   b.AddDomainIntegrator(new DomainLFIntegrator(load_cf));
   b.Assemble();
   b.ParallelAssemble(rhs.GetBlock(0));
   rhs.GetBlock(1) = 0.0;

   {
      Array<Vector*> rhs_list{&rhs.GetBlock(0), &rhs.GetBlock(1)};
      bnlf.SetEssentialBC(is_bdr_ess, rhs_list);
   }
   PetscOperatorWrapper petsc_bnlf(comm, bnlf, Operator::Type::PETSC_MATAIJ);

   real_t alpha;
   // PGPreconditioner prec(latent_k, latent, entropy, alpha);
   PetscLinearSolver lin_solver(comm);
   lin_solver.SetPrintLevel(0);
   NewtonSolver solver(comm);
   IterativeSolver::PrintLevel print_level;
   solver.SetPrintLevel(2);
   solver.SetAbsTol(1e-09);
   solver.SetRelTol(0.0);
   solver.SetMaxIter(20);
   solver.iterative_mode = true;
   solver.SetSolver(lin_solver);
   solver.SetOperator(petsc_bnlf);

   GLVis glvis("localhost", 19916, 400, 350, 4);
   glvis.Append(u, "x", "Rjclmm");
   glvis.Append(x_mapped_cf, visspace, "|U(psi)|", "RjclQmm");
   glvis.Append(gradu_norm_cf, visspace, "|gradu|", "RjclQmm");
   glvis.Append(active_set, visspace, "active set",
                "Rjclmm autoscale off valuerange 0 1");

   real_t lambda_diff = infinity();
   for (int i=0; i<100; i++)
   {
      alpha = alpha_rule.Get(i);
      out << "PG iteration " << i + 1 << " with alpha=" << alpha << std::endl;
      pg_functional.SetAlpha(alpha);
      latent_k = latent;
      latent_k.SetTrueVector();

      latent.Add(alpha, lambda);
      solver.Mult(rhs, x_and_latent);
      if (!solver.GetConverged())
      {
         out << "Newton Failed to converge in " << solver.GetNumIterations() <<
             std::endl;
         break;
      }

      u.SetFromTrueDofs(x_and_latent.GetBlock(0));
      latent.SetFromTrueDofs(x_and_latent.GetBlock(1));

      glvis.Update();

      subtract(latent, latent_k, lambda);
      lambda *= 1.0 / alpha;
      if (i > 0) { lambda_diff = lambda.ComputeL1Error(lambda_prev_cf); }
      if (lambda_diff < 1e-8)
      {
         out << "  The dual variable, (psi - psi_k)/alpha, converged" << std::endl;
         out << "PG Converged in " << i + 1
             << " with final Lambda difference: " << lambda_diff << std::endl;
         break;
      }
      else
      {
         out << "  Newton converged in " << solver.GetNumIterations()
             << " with residual " << solver.GetFinalNorm() << std::endl;
         out << "  Lambda difference: " << lambda_diff << std::endl;
      }
      lambda_prev = lambda;
   }
   return 0;
}
