/// Example 4: AD Eikonal equation with PG
#include "mfem.hpp"
#include "logger.hpp"
#include "ad_intg.hpp"
#include "tools.hpp"
#include "pg.hpp"

using namespace std;
using namespace mfem;


struct EikonalEnergy : public ADFunction
{
   EikonalEnergy() : ADFunction(1) {}
   AD_IMPL(T, V, M, x,
   {
      return x[0];
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
   filename << "ad-eikonal-";
   int rule_type = PGStepSizeRule::RuleType::CONSTANT;
   real_t max_alpha = 1e04;
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

   PGStepSizeRule alpha_rule(rule_type, alpha0, max_alpha, ratio, ratio2);

   // Mesh mesh = rhs_fun_circle
   Mesh ser_mesh = Mesh::MakeCartesian2D(2, 2,
                                         Element::QUADRILATERAL);
   const int dim = ser_mesh.Dimension();
   for (int i = 0; i < ref_levels; i++)
   {
      ser_mesh.UniformRefinement();
   }
   ParMesh mesh(comm, ser_mesh);

   const int numBdrAttr = mesh.bdr_attributes.Max();
   Array<int> is_bdr_ess1(numBdrAttr);
   is_bdr_ess1 = 1;
   Array<int> is_bdr_ess2(numBdrAttr);
   is_bdr_ess2 = 0;
   Array<Array<int>*> is_bdr_ess{&is_bdr_ess1, &is_bdr_ess2};
   EikonalEnergy obj_energy;

   H1_FECollection primal_fec(order, dim);
   RT_FECollection latent_fec(order, dim);
   ParFiniteElementSpace primal_fes(&mesh, &primal_fec);
   ParFiniteElementSpace latent_fes(&mesh, &latent_fec);
   QuadratureSpace visspace(&mesh, order+3);
   const IntegrationRule &ir = IntRules.Get(Geometry::Type::SQUARE, 3*order + 3);

   Array<int> ess_tdof_list;
   primal_fes.GetEssentialTrueDofs(is_bdr_ess1, ess_tdof_list);

   Array<int> offsets(3);
   offsets[0] = 0;
   offsets[1] = primal_fes.GetTrueVSize();
   offsets[2] = latent_fes.GetTrueVSize();
   offsets.PartialSum();
   BlockVector x_and_latent(offsets);

   ParGridFunction x(&primal_fes), latent(&latent_fes);
   ParGridFunction latent_k(latent);

   x = 0.0; x.ParallelAssemble(x_and_latent.GetBlock(0));
   latent = 0.0; latent.ParallelAssemble(x_and_latent.GetBlock(1));
   latent_k = 0.0; latent_k.SetTrueVector();

   FunctionCoefficient obstacle_cf([](const Vector &x)
   {
      return std::max(-x[0], -0.5);

   });
   HellingerEntropy entropy(1, 1.0);

   DifferentiableCoefficient entropy_cf(entropy);
   entropy_cf.AddInput(&latent);
   VectorCoefficient &u_cf = entropy_cf.Gradient();

   real_t alpha;
   ADPGFunctional pg_functional(obj_energy, entropy, &alpha, latent_k);

   ParGridFunction lambda(latent), lambda_prev(latent);
   lambda = 0.0;
   GridFunctionCoefficient lambda_prev_cf(&lambda_prev);

   Array<ParFiniteElementSpace*> fespaces{&primal_fes, &latent_fes};
   ParBlockNonlinearForm bnlf(fespaces);
   constexpr ADEval u_mode = ADEval::VALUE;
   constexpr ADEval latent_mode = ADEval::VECFE | ADEval::DIV;
   bnlf.AddDomainIntegrator(
      new ADBlockNonlinearFormIntegrator<u_mode, latent_mode>(
         pg_functional, &ir)
   );

   BlockVector rhs(offsets);
   rhs = 0.0;

   Array<Vector*> rhs_list{&rhs.GetBlock(0), &rhs.GetBlock(1)};
   bnlf.SetEssentialBC(is_bdr_ess, rhs_list);
   ParMonolithicBlockNonlinearFormWrapper bnlf_wrapper(bnlf);

#ifndef MFEM_USE_MUMPS
#error "MUMPS is required to run this example."
#endif
   MUMPSSolver lin_solver(comm);
   NewtonSolver solver(comm);
   solver.SetSolver(lin_solver);
   solver.SetOperator(bnlf_wrapper);
   IterativeSolver::PrintLevel print_level;
   solver.SetPrintLevel(print_level);
   solver.SetAbsTol(0.0);
   solver.SetRelTol(1e-07);
   solver.SetMaxIter(20);
   solver.iterative_mode = true;

   GLVis glvis("localhost", 19916, 400, 350, 3);
   glvis.Append(x, "u", "Rjclmm");
   glvis.Append(u_cf, visspace, "U(psi)", "RjclQmm");
   glvis.Append(lambda, "lambda", "Rjclmm");
   glvis.Update();

   real_t lambda_diff = infinity();
   for (int i=0; i<100; i++)
   {
      alpha = alpha_rule.Get(i);
      out << "PG iteration " << i + 1 << " with alpha=" << alpha << std::endl;
      latent_k = latent;
      latent_k.SetTrueVector();

      solver.Mult(rhs, x_and_latent);

      if (!solver.GetConverged())
      {
         out << "Newton Failed to converge in " << solver.GetNumIterations() <<
             std::endl;
      }
      x.SetFromTrueDofs(x_and_latent.GetBlock(0));
      latent.SetFromTrueDofs(x_and_latent.GetBlock(1));

      glvis.Update();

      subtract(latent, latent_k, lambda);
      lambda *= 1.0 / pg_functional.GetAlpha();

      if ((lambda_diff = lambda.ComputeL1Error(lambda_prev_cf)) < 1e-8)
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
