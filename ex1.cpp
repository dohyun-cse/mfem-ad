/// Example 1: AD Diffusion
#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "src/logger.hpp"
#include "src/ad_intg.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // file name to be saved
   std::stringstream filename;
   filename << "ad-diffusion";

   int order = 1;
   int ref_levels = 1;
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

   Mesh mesh = Mesh::MakeCartesian2D(10, 10,
                                     Element::QUADRILATERAL);
   const int dim = mesh.Dimension();
   for (int i = 0; i < ref_levels; i++)
   {
      mesh.UniformRefinement();
   }
   FunctionCoefficient load_cf([](const Vector &x)
   {
      return 2*M_PI * M_PI * std::sin(M_PI * x(0)) * std::sin(M_PI * x(1));
   });

   H1_FECollection fec(order, dim);
   FiniteElementSpace fes(&mesh, &fec);
   Array<int> ess_tdof_list;
   fes.GetBoundaryTrueDofs(ess_tdof_list);

   DiffusionEnergy energy(dim);

   NonlinearForm nlf(&fes);
   nlf.AddDomainIntegrator(new ADNonlinearFormIntegrator<ADEval::GRAD>(energy));
   nlf.SetEssentialTrueDofs(ess_tdof_list);
   LinearForm load(&fes);
   load.AddDomainIntegrator(new DomainLFIntegrator(load_cf));
   load.Assemble();
   load.SetSubVector(ess_tdof_list, 0.0);

   GridFunction x(&fes);
   x = 0.0;
   NewtonSolver solver;
   UMFPackSolver lin_solver;
   solver.SetSolver(lin_solver);
   solver.SetOperator(nlf);
   solver.SetAbsTol(1e-10);
   solver.SetRelTol(1e-10);
   IterativeSolver::PrintLevel print_level;
   print_level.iterations = true;
   solver.SetPrintLevel(print_level);
   solver.Mult(load, x);
   GLVis glvis("localhost", 19916);
   glvis.Append(x, "x", "Rjc");
   glvis.Update();
   FunctionCoefficient exact_sol([](const Vector &x)
   {
      return std::sin(M_PI * x(0)) * std::sin(M_PI * x(1));
   });
   real_t err = x.ComputeL2Error(exact_sol);
   out << "Error: " << err << std::endl;
   return 0;
}
