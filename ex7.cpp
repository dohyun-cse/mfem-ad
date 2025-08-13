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
class VectorNormCoefficient : public Coefficient
{
private:
   VectorCoefficient &vc;
   Vector v;
public:
   VectorNormCoefficient(VectorCoefficient &vc): vc(vc), v(vc.GetVDim()) {}
   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      vc.Eval(v, T, ip);
      return v.Norml2();
   }
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
      return 15*std::pow(std::sin(M_PI*x[0]), 2.0);
   });
   GradientObstacleEnergy obj_energy(dim);

   H1_FECollection primal_fec(order, dim);
   L2_FECollection latent_fec(order-2, dim);
   ParFiniteElementSpace primal_fes(&mesh, &primal_fec);
   ParFiniteElementSpace latent_fes(&mesh, &latent_fec, dim);
   QuadratureSpace visspace(&mesh, order+3);
   const IntegrationRule &ir = IntRules.Get(Geometry::Type::SQUARE, 3*order + 3);
   const IntegrationRule* irs[Geometry::NUM_GEOMETRIES];
   irs[Geometry::Type::SQUARE] = &ir;

   Array<int> ess_tdof_list;
   primal_fes.GetEssentialTrueDofs(is_bdr_ess1, ess_tdof_list);

   Array<int> offsets(3);
   offsets[0] = 0;
   offsets[1] = primal_fes.GetTrueVSize();
   offsets[2] = latent_fes.GetTrueVSize();
   offsets.PartialSum();
   BlockVector x_and_psi(offsets);

   ParGridFunction u(&primal_fes), psi(&latent_fes);
   ParGridFunction psik(psi);

   u = 0.0; u.ParallelAssemble(x_and_psi.GetBlock(0));

   psi = 0.0; psi.ParallelAssemble(x_and_psi.GetBlock(1));

   FunctionCoefficient bound([](const Vector &x)
   { return 0.1 + 0.2*x[0] + 0.4*x[1]; });
   HellingerEntropy entropy(dim, &bound);
   ADPGFunctional pg_functional(obj_energy, entropy, psik);
   DifferentiableCoefficient entropy_cf(entropy);
   entropy_cf.AddInput(&psi);
   VectorNormCoefficient x_mapped_cf(entropy_cf.Gradient());
   SumCoefficient active_set_cf(x_mapped_cf, bound, 1.0, -1.0);
   QuadratureFunction x_mapped(&visspace);
   x_mapped = 0.0;

   ConstantCoefficient zero_cf(0.0);
   ParGridFunction lambda(psi), lambda_prev(psi);
   VectorGridFunctionCoefficient lambda_prev_cf(&lambda_prev);

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
   rhs.GetBlock(0).SetSubVector(ess_tdof_list, 0.0);
   rhs.GetBlock(1) = 0.0;

   Array<Vector*> rhs_list{&rhs.GetBlock(0), &rhs.GetBlock(1)};
   bnlf.SetEssentialBC(is_bdr_ess, rhs_list);


   MUMPSMonoSolver lin_solver(MPI_COMM_WORLD);
   NewtonSolver solver(MPI_COMM_WORLD);
   solver.SetSolver(lin_solver);
   solver.SetOperator(bnlf);
   IterativeSolver::PrintLevel print_level;
   solver.SetPrintLevel(print_level);
   solver.SetAbsTol(1e-09);
   solver.SetRelTol(0.0);
   solver.SetMaxIter(20);
   solver.iterative_mode = true;

   GLVis glvis("localhost", 19916, 400, 350, 2);
   glvis.Append(u, "x", "Rjclmm");
   glvis.Append(x_mapped, "|U(psi)| - obs", "RjclQmm");
   glvis.Update();

   real_t lambda_diff = infinity();
   for (int i=0; i<100; i++)
   {
      real_t alpha = alpha_rule.Get(i);
      out << "PG iteration " << i + 1 << " with alpha=" << alpha << std::endl;
      pg_functional.SetAlpha(alpha);
      psik = psi;
      psik.SetTrueVector();
      solver.Mult(rhs, x_and_psi);
      if (!solver.GetConverged())
      {
         out << "Newton Failed to converge in " << solver.GetNumIterations() <<
             std::endl;
         break;
      }

      u.SetFromTrueDofs(x_and_psi.GetBlock(0));
      psi.SetFromTrueDofs(x_and_psi.GetBlock(1));
      subtract(psi, psik, lambda);
      lambda *= 1.0 / pg_functional.GetAlpha();
      if (i > 0) { lambda_diff = lambda.ComputeL1Error(lambda_prev_cf, irs); }
      lambda_prev = lambda;

      active_set_cf.Project(x_mapped);
      glvis.Update();
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
   }
   return 0;
}
