#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "src/logger.hpp"
#include "src/ad-native.hpp"

using namespace std;
using namespace mfem;

MAKE_AD_FUNCTION(BlockTestEnergy, T, V, M, x, dummy,
{
   return x[0]*x[1]*x[2];
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
   VectorFunctionCoefficient load_cf(dim, [dim](const Vector &x, Vector &y)
   {
      y.SetSize(dim);
      y = 1.0;
   });

   H1_FECollection fec(order, dim);
   FiniteElementSpace fes(&mesh, &fec, dim);
   Array<int> is_bdr_ess(mesh.bdr_attributes.Max());
   is_bdr_ess = 0;
   is_bdr_ess[3] = 1;
   Array<int> ess_tdof_list;
   fes.GetEssentialTrueDofs(is_bdr_ess, ess_tdof_list);

   Vector lame({1.0, 1.0}); // Lame parameters: lambda, mu
   LinearElasticityEnergy energy(dim*dim, 2);

   NonlinearForm nlf(&fes);
   nlf.AddDomainIntegrator(new ADNonlinearFormIntegrator<
                           false, ADEval::GRAD | ADEval::VECTOR>(energy, dim, lame));
   nlf.SetEssentialBC(is_bdr_ess);
   LinearForm load(&fes);
   load.AddDomainIntegrator(new VectorDomainLFIntegrator(load_cf));
   load.Assemble();
   load.SetSubVector(ess_tdof_list, 0.0);


   FiniteElementSpace fes2(&mesh, &fec);
   Array<FiniteElementSpace*> fespaces{&fes2, &fes2};
   BlockNonlinearForm bnlf(fespaces);
   bnlf.AddDomainIntegrator(
      new ADBlockNonlinearFormIntegrator<false, ADEval::GRAD, ADEval::GRAD>(
         energy, lame));
   Array<Array<int>*> is_bdr_ess2{&is_bdr_ess, &is_bdr_ess};
   Array<Vector*> dummies(2);
   bnlf.SetEssentialBC(is_bdr_ess2, dummies);

   GridFunction x(&fes);
   x = 0.0;
   SparseMatrix mymat = static_cast<SparseMatrix&>(nlf.GetGradient(x));
   NewtonSolver solver;
   UMFPackSolver lin_solver;
   solver.SetSolver(lin_solver);
   solver.SetOperator(nlf);
   solver.SetAbsTol(1e-10);
   solver.SetRelTol(1e-10);
   IterativeSolver::PrintLevel pt;
   pt.iterations = true;
   solver.SetPrintLevel(pt);
   solver.Mult(load, x);

   Vector v(x.Size()), v2(x.Size());
   nlf.Mult(x, v);
   bnlf.Mult(x, v2);
   out << v.DistanceTo(v2) << std::endl;

   GLVis glvis("localhost", 19916);
   glvis.Append(x, "x", "Rjc");
   glvis.Update();
   return 0;
}
