/*
This code is a template for serial implementation
*/

#include "mfem.hpp"
#include <fstream>
#include <iostream>

/*
include file goes here
*/
#include "src/logger.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   // file name to be saved
   std::stringstream filename;
   filename << "template";
   // initialize MPI
   Mpi::Init();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();
   MPI_Comm comm = MPI_COMM_WORLD;

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

   Mesh ser_mesh = Mesh::MakeCartesian2D(10, 10,
                                     Element::QUADRILATERAL);
   const int dim = ser_mesh.Dimension();
   for (int i = 0; i < ref_levels; i++)
   {
      ser_mesh.UniformRefinement();
   }
   ParMesh mesh(comm, ser_mesh);
   ser_mesh.Clear();

   L2_FECollection fec(order, 2);
   ParFiniteElementSpace fes(&mesh, &fec);

   ParGridFunction u_gf(&fes);
   FunctionCoefficient u_cf([](const Vector &x) { return std::sin(M_PI*x(0))*std::sin(M_PI*x(1)); });
   u_gf.ProjectCoefficient(u_cf);

   std::unique_ptr<GLVis> glvis;
   if (visualization)
   {
      GLVis glvis("localhost", 19916, 400, 350, 4);
      glvis.Append(u_gf, "u", "Rjc");
      glvis.Update();
   }

   std::unique_ptr<ParaViewDataCollection> paraview_dc;
   if (paraview)
   {
      paraview_dc.reset(new mfem::ParaViewDataCollection("ParaView", &mesh));
      paraview_dc->SetPrefixPath(filename.str().c_str());
      paraview_dc->SetLevelsOfDetail(order < 2 ? order : order + 3);
      paraview_dc->SetHighOrderOutput(true);
      paraview_dc->SetDataFormat(VTKFormat::BINARY);
      paraview_dc->RegisterField("u", &u_gf);
      paraview_dc->Save();
   }

   return 0;
}
