#include "logger.hpp"

namespace mfem
{

TableLogger::TableLogger(std::ostream &os)
   : os(os), w(14), var_name_printed(false), isRoot(true)
{
#ifdef MFEM_USE_MPI
   isRoot = mfem::Mpi::IsInitialized() ? mfem::Mpi::Root() : true;
#endif
}

void TableLogger::Append(const std::string name, double &val)
{
   names.push_back(name);
   data_double.push_back(&val);
   data_order.push_back(dtype::DOUBLE);
}

void TableLogger::Append(const std::string name, int &val)
{
   names.push_back(name);
   data_int.push_back(&val);
   data_order.push_back(dtype::INT);
}

void TableLogger::Print(bool print_varname)
{
   if (isRoot)
   {
      if (!var_name_printed || print_varname)
      {
         for (auto &name : names)
         {
            os << std::setw(w) << std::setfill(' ') << name << ",\t";
         }
         os << "\b\b";
         os << std::endl;
         if (!var_name_printed && file && file->is_open())
         {
            for (int i=0; i<names.size() - 1; i++)
            {
               *file << std::setw(w) << std::setfill(' ') << names[i] << ",\t";
            }
            *file << std::setw(w) << std::setfill(' ') << names.back() << std::endl;
         }
         var_name_printed = true;
      }
      int i(0), i_double(0), i_int(0);
      for (int i=0; i<data_order.size(); i++)
      {
         auto d = data_order[i];
         switch (d)
         {
            case dtype::DOUBLE:
            {
               os << std::setw(w) << *data_double[i_double];
               if (file && file->is_open())
               {
                  *file << std::setprecision(8) << std::scientific << std::setw(w)
                        << std::setfill(' ') << *data_double[i_double];
               }
               i_double++;
               break;
            }
            case dtype::INT:
            {
               os << std::setw(w) << *data_int[i_int];
               if (file && file->is_open())
               {
                  *file << std::setw(w) << std::setfill(' ') << *data_int[i_int];
               }
               i_int++;
               break;
            }
            default:
            {
               MFEM_ABORT("Unknown data type. See, TableLogger::dtype");
            }
         }
         if (i < data_order.size() - 1)
         {
            os << ",\t";
            *file << ",\t";
         }
      }
      os << std::endl;
      if (file)
      {
         *file << std::endl;
      }
   }
}

void TableLogger::SaveWhenPrint(std::string filename, std::ios::openmode mode)
{
   if (isRoot)
   {
      filename = filename.append(".csv");
      file.reset(new std::fstream);
      file->open(filename, mode);
      if (!file->is_open())
      {
         std::string msg("");
         msg += "Cannot open file ";
         msg += filename;
         MFEM_ABORT(msg);
      }
   }
}

void GLVis::Append(GridFunction *gf, QuadratureFunction *qf,
                   std::string_view window_title, std::string_view keys)
{
   MFEM_VERIFY((gf == nullptr && qf != nullptr)
               || (gf != nullptr && qf == nullptr),
               "Either GridFunction or QuadratureFunction must be provided, "
               "but not both.");
   bool is_gf = gf != nullptr;
   sockets.push_back(std::make_unique<socketstream>(hostname, port, secure));
   socketstream &socket = *sockets.back();
   if (!socket.is_open() || !socket.good())
   {
      return;
   }
   socket.precision(8);
   gfs.Append(gf);
   qfs.Append(qf);

   Mesh *mesh;
   if (is_gf) { mesh = gf->FESpace()->GetMesh(); }
   else { mesh = qf->GetSpace()->GetMesh(); }
   meshes.Append(mesh);

   cfs.Append(nullptr);
   vcfs.Append(nullptr);

   parallel.Append(false);
   myrank.Append(0);
   nrrank.Append(1);
#ifdef MFEM_USE_MPI
   if (ParMesh *pmesh = dynamic_cast<ParMesh*>(mesh))
   {
      parallel.Last() = true;
      myrank.Last() = pmesh->GetMyRank();
      nrrank.Last() = pmesh->GetNRanks();
      MPI_Barrier(pmesh->GetComm());
      socket << "parallel " << nrrank.Last() << " " << myrank.Last() <<
                "\n";
   }
#endif
   socket << "solution\n" << *mesh;

   if (is_gf) { socket << *gf; }
   else { socket << *qf; }

   if (!keys.empty())
   {
      socket << "keys " << keys << " ";
      bool hasQ=false;
      if (!is_gf)
      {
         size_t end_pos = std::min(keys.find(' '), keys.find('\n'));
         std::string_view actual_keys = keys.substr(0, end_pos);
         if (actual_keys.find('Q') != std::string_view::npos) { hasQ = true; }
      }
      qfkey_has_Q.Append(hasQ);
   }
   if (!window_title.empty())
   {
      socket << "window_title '" << window_title <<"' ";
   }
   int row = (sockets.size() - 1) / nrWinPerRow;
   int col = (sockets.size() - 1) % nrWinPerRow;
   socket << " window_geometry "
          << w*col << " " << h*row << " "
          << w << " " << h;
   socket << std::endl;
}

void GLVis::Append(Coefficient &cf, QuadratureSpace &qs,
                   std::string_view window_title,
                   std::string_view keys)
{
   owned_qfs.push_back(std::make_unique<QuadratureFunction>(qs));
   Append(nullptr, owned_qfs.back().get(), window_title, keys);
   cfs.Last() = &cf;
}

void GLVis::Append(VectorCoefficient &cf, QuadratureSpace &qs,
                   std::string_view window_title,
                   std::string_view keys)
{
   owned_qfs.push_back(std::make_unique<QuadratureFunction>(qs, cf.GetVDim()));
   Append(nullptr, owned_qfs.back().get(), window_title, keys);
   vcfs.Last() = &cf;
}

void GLVis::Update()
{
   for (int i=0; i<sockets.size(); i++)
   {
      if (!sockets[i]->is_open() || !sockets[i]->good())
      {
         continue;
      }
#ifdef MFEM_USE_MPI
      if (parallel[i])
      {
         MPI_Comm comm = static_cast<ParMesh*>(meshes[i])->GetComm();
         MPI_Barrier(comm);
         *sockets[i] << "parallel " << nrrank[i] << " " << myrank[i] <<
                        "\n";
      }
#endif
      if (gfs[i])
      {
         *sockets[i] << "solution\n" << *meshes[i] << *gfs[i];
      }
      else if (qfs[i])
      {
         if (cfs[i] != nullptr) { cfs[i]->Project(*qfs[i]); }
         else if (vcfs[i] != nullptr) { vcfs[i]->Project(*qfs[i]); }
         std::cout << qfs[i]->CheckFinite() << std::endl;
         MFEM_VERIFY(qfs[i]->CheckFinite() == 0,
                     "QuadratureFunction has non-finite entries");
         *sockets[i] << "quadrature\n" << *meshes[i] << *qfs[i];
         // NOTE: GLVis seems to have a bug with Q key
         // It does not restore interpolation type after update.
         // So, we cycle through all interpolation types to restore it.
         if (qfkey_has_Q[i]) { *sockets[i] << "keys 'QQQ'"; }
      }
      else
      {
         MFEM_ABORT("Unknown data type. See, GLVis::Update");
      }
      *sockets[i] << std::endl;
      if (parallel[i])
      {
#ifdef MFEM_USE_MPI
         MPI_Comm comm = static_cast<ParMesh*>(meshes[i])->GetComm();
         MPI_Barrier(comm);
#endif
      }
   }
}

} // namespace mfem
