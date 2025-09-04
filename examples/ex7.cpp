/// Example 5: AD Gradeint Obstacle Problem with PG
#include "mfem.hpp"
#include "logger.hpp"
// #include "ad_intg.hpp"
#include "tools.hpp"
#include "pg.hpp"
#include "diffobj.hpp"
#include "mmto.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();
   MPI_Comm comm = MPI_COMM_WORLD;
   // file name to be saved
   std::stringstream filename;
   filename << "ad-mmto-cantilever-";
   // int rule_type = PGStepSizeRule::RuleType::CONSTANT;
   // real_t max_alpha = 1e06;
   // real_t alpha0 = 1.0;
   // real_t ratio = 1.0;
   // real_t ratio2 = 1.0;
   // bool use_iterative = false;

   bool prefilter = true;
   bool postfilter = false;

   int order = 1;
   int ref_levels = 0;
   bool visualization = true;
   bool paraview = true;

   OptionsParser args(argc, argv);
   args.AddOption(&order, "-o", "--order",
                  "Finite element polynomial degree");
   args.AddOption(&ref_levels, "-r", "--ref", "Refinement levels");
   // args.AddOption(&rule_type, "-rule", "--rule",
   //                "Step size rule type: 0=CONSTANT, 1=POLY, 2=EXP, 3=DOUBLE_EXP");
   // args.AddOption(&max_alpha, "-ma", "--max-alpha",
   //                "Maximum step size for PG method");
   // args.AddOption(&alpha0, "-a0", "--alpha0",
   //                "Initial step size for PG method");
   // args.AddOption(&ratio, "-ar", "--alpha-ratio",
   //                "Ratio for step size rule (POLY, EXP, DOUBLE_EXP)");
   // args.AddOption(&ratio2, "-ar2", "--alpha-ratio2",
   //                "Second ratio for DOUBLE_EXP step size rule");
   args.AddOption(&visualization, "-vis", "--visualization",
                  "-no-vis", "--no-visualization",
                  "Enable visualization, default is false");
   args.AddOption(&paraview, "-pv", "--paraview",
                  "-no-pv", "--no-paraview",
                  "Enable Paraview Export. Default is false");
   // args.AddOption(&use_iterative, "-gmres", "--preconditioned-gmres",
   //                "-mumps", "--MUMPS",
   //                "Use preconditioned GMRES or MUMPS as linear solver. Default is MUMPS");
   args.ParseCheck();
   if (myid != 0) { out.Disable(); }
   MFEM_VERIFY(order >= 1, "Order must be at least 1.");

   const int numMaterials = 5;
   DenseMatrix E(1, numMaterials);
   Vector rho{0.0, 1.0, 1.3, 1.5, 1.7};
   Vector mass_bound(numMaterials); // 0: total, i>0: individual
   mass_bound = mfem::infinity();
   rho(0) = 0.0; E(0,0) = 1e-08;
   // mass_bound(0) = 1.5;
   for (int i=1; i<numMaterials; i++)
   {
      E(0, i) = i;
      // rho(i) = 1.0 + (i*0.3);
      mass_bound(i) = 1.5 / (numMaterials-1);
   }
   std::vector<Vector> rho_list;
   rho_list.push_back(Vector(numMaterials));
   rho_list.back() = rho;
   for (int i = 1; i < numMaterials; i++)
   {
      rho_list.push_back(Vector(numMaterials));
      rho_list.back() = 0.0;
      rho_list.back()[i] = rho[i];
   }
   // const Vector rho_1{0.0, 1.0, 0.0};
   // const Vector rho_2{0.0, 0.0, 1.3};
   const real_t nu = 0.3;

   MFEM_VERIFY(rho.Size() == numMaterials,
               "Young's modulus and density size do not match.");

   Mesh ser_mesh = Mesh::MakeCartesian2D(6, 2, Element::Type::QUADRILATERAL, false,
                                         3.0, 1.0);
   const int dim = ser_mesh.Dimension();
   for (int i = 0; i < ref_levels; i++)
   {
      ser_mesh.UniformRefinement();
   }
   ParMesh mesh(comm, ser_mesh);
   ser_mesh.Clear();

   Vector load_center{2.9, 0.5};
   real_t load_radius = 0.1;
   VectorFunctionCoefficient state_load_cf(dim, [load_center,
                                                 load_radius](const Vector &x, Vector &f)
   {
      f.SetSize(x.Size());
      f = 0.0;
      f[x.Size()-1] = x.DistanceTo(load_center) < load_radius ? -1.0 : 0.0;
   });

   Array2D<int> ess_bdr(dim, 4);
   {
      Array<int> curr_bdr(ess_bdr.NumCols());
      for (int i=0; i<dim; i++)
      {
         curr_bdr.MakeRef(ess_bdr.GetRow(i), ess_bdr.NumCols());
         curr_bdr = 0;
         curr_bdr[3] = 1;
      }
   }

   H1_FECollection state_fec(order, dim);
   H1_FECollection filter_fec(order, dim);

   ParFiniteElementSpace state_fes(&mesh, &state_fec, dim);
   ParFiniteElementSpace filter_fes(&mesh, &filter_fec);
   QuadratureSpace qspace(&mesh, order*2 + 1);

   Array<int> state_ess_tdofs;
   GetEssentialTrueDofs(state_fes, ess_bdr, state_ess_tdofs);

   ParGridFunction state_gf(&state_fes);
   Vector state_tvec(state_fes.GetTrueVSize());
   state_gf = 0.0;
   state_tvec = 0.0;
   QuadratureFunction latent(qspace, numMaterials);
   QuadratureFunction indicator(qspace, numMaterials);
   QuadratureFunction material(qspace);
   QuadratureFunction gradient(qspace, numMaterials);
   QuadratureFunction latent_k(qspace, numMaterials);
   QuadratureFunction indicator_k(qspace, numMaterials);
   QuadratureFunction gradient_k(qspace, numMaterials);
   QuadratureFunction dfdP(qspace);
   QuadratureFunction diff_indicator(qspace, numMaterials);
   latent = log(rho.Sum() / numMaterials * 0.5);
   gradient = 0.0;

   // visualize each component
   std::vector<std::unique_ptr<QuadratureFunction>> indicator_list;
   std::vector<std::unique_ptr<QuadratureFunction>> gradient_list;
   for (int i = 0; i < numMaterials; i++)
   {
      indicator_list.push_back(std::make_unique<QuadratureFunction>(qspace));
      gradient_list.push_back(std::make_unique<QuadratureFunction>(qspace));
   }

   SimplexEntropy entropy(numMaterials, 1.0);
   DifferentiableCoefficient entropy_cf(entropy);
   entropy_cf.AddInput(&latent);
   VectorCoefficient &indicator_cf = entropy_cf.Gradient();

   CompositeOperator material_op(qspace.GetSize(),
                                 numMaterials*qspace.GetSize());
   std::unique_ptr<HelmholtzFilter> indicator_filter;
   std::unique_ptr<HelmholtzFilter> property_filter;
   if (prefilter)
   {
      indicator_filter = std::make_unique<HelmholtzFilter>(
                            qspace, filter_fes, 0.05, numMaterials);
      material_op.AddOperation(*indicator_filter);
   }
   ForwardBackwardADVectorOperator simp_op(new SIMPFunction(3.0, E), qspace);
   material_op.AddOperation(simp_op);
   if (postfilter)
   {
      property_filter = std::make_unique<HelmholtzFilter>(
                           qspace, filter_fes, 0.05, 1);
      material_op.AddOperation(*property_filter);
   }

   ParametrizedLinearElasticityEnergy elasticity_energy(dim, nu);

   ParLinearForm load(&state_fes);
   load.AddDomainIntegrator(new VectorDomainLFIntegrator(state_load_cf));
   Vector load_dual_tvec(state_fes.GetTrueVSize());
   load.Assemble();
   load.ParallelAssemble(load_dual_tvec);
   BregmanDykstra projector(qspace, entropy);
   std::vector<std::unique_ptr<ADFunction>> constraints;
   for (int i=0; i<numMaterials; i++)
   {
      if (IsFinite(mass_bound[i]))
      {
         constraints.push_back(std::make_unique<WeightedSumFunction>(rho_list[i]));
         projector.AddConstraint(*constraints.back(), mass_bound[i]);
      }
   }
   WeightedSumFunction density_all(rho);
   DifferentiableCoefficient density_cf(density_all);
   density_cf.AddInput(&indicator);

   ParametrizedElasticitySolver state_solver(state_fes, qspace, state_ess_tdofs,
                                             load_dual_tvec, elasticity_energy, false);
   auto f = [&]()
   {
      indicator_cf.Project(indicator);
      material_op.Mult(indicator, material);
      state_solver.Mult(material, state_tvec);
      state_gf.SetFromTrueDofs(state_tvec);
      return InnerProduct(comm, state_tvec, load_dual_tvec);
   };
   auto grad_f = [&](Vector &g)
   {
      state_solver.GetGradient(material).MultTranspose(load_dual_tvec, dfdP);
      material_op.GetGradient(indicator).MultTranspose(dfdP, g);
   };
   projector.Mult(latent, latent);
   real_t obj, obj_k;
   obj = f();
   std::unique_ptr<GLVis> glvis;
   std::unique_ptr<ParaViewDataCollection> paraview_dc;
   if (visualization) { glvis = std::make_unique<GLVis>("localhost", 19916, 400, 350, 4); }
   if (paraview) { paraview_dc = std::make_unique<ParaViewDataCollection>("ParaView/Topopt", &mesh); }
   density_cf.Project(*indicator_list[0]);
   if (glvis) { glvis->Append(*indicator_list[0], "density", "Rjmm*********"); }
   if (paraview_dc) { paraview_dc->RegisterQField("density", indicator_list[0].get()); }
   if (paraview_dc)
   {
      string name;
      for (int i=1; i<numMaterials; i++)
      {
         name.clear();
         name.append("indicator_");
         name.append(std::to_string(i));
         ExtractComponent(indicator, i, *indicator_list[i]);
         paraview_dc->RegisterQField(name.c_str(), indicator_list[i].get());
      }
      paraview_dc->RegisterQField("material", &material);
   }
   if (glvis)
   { glvis->Append(material, "material", "Rjmm*********"); }
   if (paraview_dc)
   {
      paraview_dc->SetCycle(0);
      paraview_dc->SetTime(0.0);
      paraview_dc->Save();
   }

   real_t step_size = 1.0;
   for (int i=0; i<500; i++)
   {
      indicator_k = indicator;
      latent_k = latent;
      gradient_k = gradient;
      obj_k = obj;
      out << "Step " << i << " with " << step_size << std::endl;
      grad_f(gradient);
      for (int j=0; j<20; j++)
      {
         out << "  Armijo trial " << j << " with step size " << step_size << std::endl;
         add(latent_k, -step_size, gradient, latent);
         projector.Mult(latent, latent);
         out << "    Projection finished after " << projector.GetNumIterations() <<
             " iterations." << std::endl;
         obj = f();
         subtract(indicator, indicator_k, diff_indicator);
         real_t diffval = dot(comm, diff_indicator, gradient);
         real_t suff_decr = obj_k + 1e-03*diffval;
         out <<"  Objective = " << obj << "<?" << suff_decr << " = " << obj_k <<
             "+1e-03*" << diffval << std::endl;
         if (obj <= suff_decr && diffval < 0.0) { break; }
         step_size *= 0.5;
      }
      density_cf.Project(*indicator_list[0]);
      for (int i=0; i<numMaterials; i++)
      {
         if (i > 0) { ExtractComponent(indicator, i, *indicator_list[i]); }
         ExtractComponent(gradient, i, *gradient_list[i]);
      }
      if (glvis) { glvis->Update(); }
      if (paraview_dc)
      {
         paraview_dc->SetCycle(i+1);
         paraview_dc->SetTime(i+1);
         paraview_dc->Save();
      }
      if (std::sqrt(dot(comm, diff_indicator, diff_indicator)) < 1e-08)
      {
         out << "Converged." << std::endl;
         break;
      }

      latent_k -= latent;
      indicator_k -= indicator;
      gradient_k -= gradient;
      real_t latent_primal = dot(comm, latent_k, indicator_k);
      real_t gradient_primal = dot(comm, gradient_k, indicator_k);
      step_size = fabs(latent_primal / gradient_primal);
   }

   return 0;
}
