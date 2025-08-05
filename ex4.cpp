/// Example 4: AD Linear Elasticity with multiple Scalar FE
#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "src/logger.hpp"
#include "src/ad-native.hpp"

using namespace std;
using namespace mfem;

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

   FiniteElementSpace fes_scalar(&mesh, &fec);
   Array<FiniteElementSpace*> fespaces(dim);
   fespaces = &fes_scalar;
   BlockNonlinearForm bnlf(fespaces);
   if (dim == 2)
   {
      bnlf.AddDomainIntegrator(
         new ADBlockNonlinearFormIntegrator<false, ADEval::GRAD, ADEval::GRAD>(
            energy, lame));
   }
   else
   {
      bnlf.AddDomainIntegrator(
         new ADBlockNonlinearFormIntegrator<false, ADEval::GRAD, ADEval::GRAD, ADEval::GRAD>
         (
            energy, lame));
   }
   Array<Array<int>*> is_bdr_ess2(dim);
   is_bdr_ess2 = &is_bdr_ess;
   Array<Vector*> dummies(dim); dummies = nullptr;
   bnlf.SetEssentialBC(is_bdr_ess2, dummies);

   LinearForm load(&fes);
   load.AddDomainIntegrator(new VectorDomainLFIntegrator(load_cf));
   load.Assemble();
   load.SetSubVector(ess_tdof_list, 0.0);

   Array<int> offsets(dim+1);
   offsets = fes_scalar.GetVSize();
   offsets[0] = 0;
   offsets.PartialSum();

   BlockVector X(offsets);
   GridFunction x(&fes, X, 0);
   NewtonSolver solver;
   CGSolver lin_solver;
   lin_solver.SetRelTol(1e-10);
   lin_solver.SetAbsTol(1e-10);
   lin_solver.SetMaxIter(1e06);
   solver.SetSolver(lin_solver);
   solver.SetOperator(bnlf);
   solver.SetAbsTol(1e-10);
   solver.SetRelTol(1e-10);
   IterativeSolver::PrintLevel pt;
   pt.iterations = true;
   solver.SetPrintLevel(pt);
   solver.Mult(load, X);

   GLVis glvis("localhost", 19916);
   glvis.Append(x, "x", "Rjc");
   glvis.Update();
   Hypre::Finalize();
   Mpi::Finalize();
   return 0;
}
