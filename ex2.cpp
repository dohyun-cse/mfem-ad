#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "src/logger.hpp"
#include "src/ad-native.hpp"

using namespace std;
using namespace mfem;

MAKE_AD_FUNCTION(MinimalSurfaceEnergy, T, V, M, gradu, dummy,
{
   T h1_norm(gradu*gradu);
   return sqrt(h1_norm + 1.0)*h1_norm * 0.5;
});

int main(int argc, char *argv[])
{
   // file name to be saved
   std::stringstream filename;
   filename << "ad-diffusion";

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
   Mesh mesh = Mesh::MakeCartesian2D(10, 10,
                                     Element::QUADRILATERAL);
   const int dim = mesh.Dimension();
   for (int i = 0; i < ref_levels; i++)
   {
      mesh.UniformRefinement();
   }
   FunctionCoefficient load_cf([](const Vector &x)
   {
      real_t theta = std::atan2(x(1)-0.5, x(0)-0.5);
      return std::sin(5*theta);
   });

   H1_FECollection fec(order, dim);
   FiniteElementSpace fes(&mesh, &fec);
   Array<int> is_bdr_ess(mesh.bdr_attributes.Max());
   is_bdr_ess = 1;

   MinimalSurfaceEnergy energy(dim);

   NonlinearForm nlf(&fes);
   nlf.AddDomainIntegrator(new ADNonlinearFormIntegrator<ADEvalInput::GRAD>
                           (energy));
   nlf.SetEssentialBC(is_bdr_ess);

   GridFunction x(&fes);
   x = 0.0;
   x.ProjectBdrCoefficient(load_cf, is_bdr_ess);
   NewtonSolver solver;
   UMFPackSolver lin_solver;
   solver.SetSolver(lin_solver);
   solver.SetOperator(nlf);
   solver.SetAbsTol(1e-10);
   solver.SetRelTol(1e-10);
   solver.SetPrintLevel(3);
   Vector dummy(0);
   solver.Mult(dummy, x);
   GLVis glvis("localhost", 19916);
   glvis.Append(x, "x", "Rjc");
   glvis.Update();
   return 0;
}
