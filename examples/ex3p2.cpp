#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "ad_intg.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 2. Parse command-line options.
   bool use_ad = true;
   int order = 1;
   bool visualization = 3;
   const char *device_config = "cpu";
   int par_ref_levels = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&use_ad, "-ad", "--use-ad", "-no-ad",
                  "--use-integrator",
                  "Use AD-based nonlinear form integrator, or the standard integrator. Default is true.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (myid == 0) { device.Print(); }

   // 4. Read the (serial) mesh from the given mesh file on all processors.  We
   //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
   //    and volume meshes with the same code.
   // Mesh *mesh = new Mesh(mesh_file, 1, 1);
   Mesh mesh = Mesh::MakeCartesian2D(30, 10, Element::QUADRILATERAL, false, 3.0,
                                     1.0);
   int dim = mesh.Dimension();

   // 7. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   for (int l = 0; l < par_ref_levels; l++)
   {
      pmesh.UniformRefinement();
   }

   // 8. Define a parallel finite element space on the parallel mesh. Here we
   //    use vector finite elements, i.e. dim copies of a scalar finite element
   //    space. We use the ordering by vector dimension (the last argument of
   //    the FiniteElementSpace constructor) which is expected in the systems
   //    version of BoomerAMG preconditioner. For NURBS meshes, we use the
   //    (degree elevated) NURBS space associated with the mesh nodes.
   H1_FECollection fec(order, dim);
   ParFiniteElementSpace fespace(&pmesh, &fec, dim);
   HYPRE_BigInt size = fespace.GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl
           << "Assembling: " << flush;
   }

   // 9. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, the boundary conditions are defined by
   //    marking only boundary attribute 1 from the mesh as essential and
   //    converting it to a list of true dofs.
   Array<int> ess_tdof_list, ess_bdr(pmesh.bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[3] = 1;
   fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // 10. Set up the parallel linear form b(.) which corresponds to the
   //     right-hand side of the FEM linear system. In this case, b_i equals the
   //     boundary integral of f*phi_i where f represents a "pull down" force on
   //     the Neumann part of the boundary and phi_i are the basis functions in
   //     the finite element fespace. The force is defined by the object f, which
   //     is a vector of Coefficient objects. The fact that f is non-zero on
   //     boundary attribute 2 is indicated by the use of piece-wise constants
   //     coefficient for its last component.
   Vector f_vec(dim);
   f_vec = 0.0;
   f_vec(dim-1) = 1.0;
   VectorConstantCoefficient f(f_vec);

   ParLinearForm b(&fespace);
   b.AddDomainIntegrator(new VectorDomainLFIntegrator(f));
   if (myid == 0)
   {
      cout << "r.h.s. ... " << flush;
   }
   b.Assemble();

   // 11. Define the solution vector x as a parallel finite element grid
   //     function corresponding to fespace. Initialize x with initial guess of
   //     zero, which satisfies the boundary conditions.
   ParGridFunction x(&fespace);
   x = 0.0;

   // 12. Set up the parallel bilinear form a(.,.) on the finite element space
   //     corresponding to the linear elasticity integrator with piece-wise
   //     constants coefficient lambda and mu.
   real_t E(1.0), nu(0.3);
   real_t lambda = E*nu / (1.0 - nu*nu);
   real_t mu = E / (2.0 * (1.0 + nu));

   // real_t lambda(1.0);
   // real_t mu(0.0);

   ConstantCoefficient lambda_func(lambda);
   ConstantCoefficient mu_func(mu);

   if (use_ad)
   {
      LinearElasticityEnergy energy(dim, lambda, mu);

      ParNonlinearForm nlf(&fespace);

      nlf.SetGradientType(Operator::Type::Hypre_ParCSR);

      nlf.AddDomainIntegrator(
         new ADNonlinearFormIntegrator<ADEval::GRAD | ADEval::VECTOR>(energy));

      HypreParVector B(&fespace), X(&fespace);
      x.GetTrueDofs(X);
      b.ParallelAssemble(B);

      HypreParMatrix &A = static_cast<HypreParMatrix&>(nlf.GetGradient(X));
      A.EliminateRowsCols(ess_tdof_list, X, B);

      // 14. Define and apply a parallel PCG solver for A X = B with the BoomerAMG
      //     preconditioner from hypre.
      HypreBoomerAMG amg(A);
      amg.SetSystemsOptions(dim, true);
      HyprePCG pcg(A);
      pcg.SetTol(1e-8);
      pcg.SetMaxIter(1e04);
      pcg.SetPrintLevel(2);
      pcg.SetPreconditioner(amg);
      pcg.Mult(B, X);
      x.SetFromTrueDofs(X);
   }
   else
   {
      ParBilinearForm a(&fespace);
      a.AddDomainIntegrator(new ElasticityIntegrator(lambda_func, mu_func));

      // 13. Assemble the parallel bilinear form and the corresponding linear
      //     system, applying any necessary transformations such as: parallel
      //     assembly, eliminating boundary conditions, applying conforming
      //     constraints for non-conforming AMR, static condensation, etc.
      if (myid == 0) { cout << "matrix ... " << flush; }
      a.Assemble();
      HypreParMatrix A;
      HypreParVector B(&fespace), X(&fespace);
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
      if (myid == 0)
      {
         cout << "done." << endl;
         cout << "Size of linear system: " << A.GetGlobalNumRows() << endl;
      }

      // 14. Define and apply a parallel PCG solver for A X = B with the BoomerAMG
      //     preconditioner from hypre.
      HypreBoomerAMG amg(A);
      amg.SetSystemsOptions(dim, true);
      HyprePCG pcg(A);
      pcg.SetTol(1e-8);
      pcg.SetMaxIter(1e04);
      pcg.SetPrintLevel(2);
      pcg.SetPreconditioner(amg);
      pcg.Mult(B, X);
      x.SetFromTrueDofs(X);
   }


   // 15. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.

   // 18. Send the above data by socket to a GLVis server.  Use the "n" and "b"
   //     keys in GLVis to visualize the displacements.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh << x << flush;
   }

   return 0;
}
