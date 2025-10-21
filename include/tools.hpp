#pragma once
#include "mfem.hpp"

namespace mfem
{
class MappedGridFunctionCoefficient : public Coefficient
{
private:
   GridFunction *gf;
   std::function<real_t(const real_t)> map_func;
public:
   MappedGridFunctionCoefficient(GridFunction *gf_,
                                 std::function<real_t(const real_t)> map_func_)
      : gf(gf_), map_func(map_func_) {  }
   virtual real_t Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      return map_func(gf->GetValue(T.ElementNo, T.GetIntPoint()));
   }
};
class MappedVectorGridFunctionCoefficient : public VectorCoefficient
{
private:
   GridFunction *gf;
   Vector gf_val;
   std::function<void(const Vector&, Vector&)> map_func;
public:
   MappedVectorGridFunctionCoefficient(const int dim,
                                       GridFunction *gf_,
                                       std::function<void(const Vector&, Vector&)> map_func_)
      : VectorCoefficient(dim), gf(gf_), map_func(map_func_)
   {
      MFEM_VERIFY(gf_ != nullptr, "MappedVectorGridFunctionCoefficient: gf_ is null");
      gf_val.SetSize(gf_->VectorDim());
   }
   void Eval(Vector &V, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      gf->GetVectorValue(T.ElementNo, T.GetIntPoint(), gf_val);
      map_func(gf_val, V);
   }
};
class VectorGradientGridFunctionCoefficient : public MatrixCoefficient
{
private:
   GridFunction &gf;
public:
   VectorGradientGridFunctionCoefficient(GridFunction &gf)
      : MatrixCoefficient(gf.FESpace()->GetVDim(),
                          gf.FESpace()->GetMesh()->SpaceDimension()), gf(gf)
   {}

   void Eval(DenseMatrix &grad, ElementTransformation &T,
             const IntegrationPoint &ip) override
   { gf.GetVectorGradient(T, grad); }
};

inline std::unique_ptr<GridFunction>
NewGridFunction(FiniteElementSpace &fes)
{
#ifdef MFEM_USE_MPI
   if (ParFiniteElementSpace *pfes =
          dynamic_cast<ParFiniteElementSpace*>(&fes))
   {
      return std::make_unique<ParGridFunction>(pfes);
   }
#endif
   return std::make_unique<GridFunction>(&fes);
}
inline std::unique_ptr<GridFunction>
NewGridFunction(FiniteElementSpace &fes, real_t *data)
{
#ifdef MFEM_USE_MPI
   if (ParFiniteElementSpace *pfes =
          dynamic_cast<ParFiniteElementSpace*>(&fes))
   {
      return std::make_unique<ParGridFunction>(pfes, data);
   }
#endif
   return std::make_unique<GridFunction>(&fes, data);
}
inline std::unique_ptr<GridFunction>
NewGridFunction(FiniteElementSpace &fes, Vector &lvec)
{
#ifdef MFEM_USE_MPI
   if (ParFiniteElementSpace *pfes =
          dynamic_cast<ParFiniteElementSpace*>(&fes))
   {
      return std::make_unique<ParGridFunction>(pfes, lvec);
   }
#endif
   return std::make_unique<GridFunction>(&fes, lvec);
}

inline std::unique_ptr<LinearForm>
NewLinearForm(FiniteElementSpace &fes)
{
#ifdef MFEM_USE_MPI
   if (ParFiniteElementSpace *pfes =
          dynamic_cast<ParFiniteElementSpace*>(&fes))
   {
      return std::make_unique<ParLinearForm>(pfes);
   }
#endif
   return std::make_unique<LinearForm>(&fes);
}

inline std::unique_ptr<BilinearForm>
NewBilinearForm(FiniteElementSpace &fes)
{
#ifdef MFEM_USE_MPI
   if (ParFiniteElementSpace *pfes =
          dynamic_cast<ParFiniteElementSpace*>(&fes))
   {
      return std::make_unique<ParBilinearForm>(pfes);
   }
#endif
   return std::make_unique<BilinearForm>(&fes);
}

inline std::unique_ptr<MixedBilinearForm>
NewMixedBilinearForm(FiniteElementSpace &trial_fes,
                     FiniteElementSpace &test_fes)
{
#ifdef MFEM_USE_MPI
   if (ParFiniteElementSpace *trial_pfes =
          dynamic_cast<ParFiniteElementSpace*>(&trial_fes))
   {
      ParFiniteElementSpace *test_pfes = dynamic_cast<ParFiniteElementSpace*>
                                         (&test_fes);
      MFEM_VERIFY(test_pfes != nullptr,
                  "NewMixedBilinearForm: Trial is parallel, but test is not.");
      return std::make_unique<ParMixedBilinearForm>(trial_pfes, test_pfes);
   }
   MFEM_VERIFY(dynamic_cast<ParFiniteElementSpace*>(&test_fes) == nullptr,
               "NewMixedBilinearForm: Trial is not parallel, but test is.");
#endif
   return std::make_unique<MixedBilinearForm>(&trial_fes, &test_fes);
}
inline std::unique_ptr<NonlinearForm>
NewNonlinearForm(FiniteElementSpace &fes)
{
#ifdef MFEM_USE_MPI
   if (ParFiniteElementSpace *pfes =
          dynamic_cast<ParFiniteElementSpace*>(&fes))
   {
      return std::make_unique<ParNonlinearForm>(pfes);
   }
#endif
   return std::make_unique<NonlinearForm>(&fes);
}
inline std::unique_ptr<BlockNonlinearForm>
NewBlockNonlinearForm(Array<FiniteElementSpace*> &fes)
{
#ifdef MFEM_USE_MPI
   int numParallel = 0;

   Array<ParFiniteElementSpace*> pfes;
   for (auto *space : fes)
   {
      pfes.Append(dynamic_cast<ParFiniteElementSpace*>(space));
      numParallel += pfes.Last() != nullptr;
   }
   MFEM_VERIFY(numParallel == 0 || numParallel == fes.Size(),
               "NewBlockNonlinearForm: either all or none of the spaces must be parallel");
   if (numParallel == fes.Size())
   {
      return std::make_unique<ParBlockNonlinearForm>(pfes);
   }
#endif
   return std::make_unique<BlockNonlinearForm>(fes);
}

// Monolithic direct solver for block system
class MUMPSMonoSolver : public MUMPSSolver
{
private:
   std::unique_ptr<HypreParMatrix> mono;
public:
   MUMPSMonoSolver(MPI_Comm comm) : MUMPSSolver(comm) {}

   void SetOperator(const Operator &op)
   {
      const BlockOperator *bop = dynamic_cast<const BlockOperator*>(&op);
      MFEM_VERIFY(bop != nullptr, "Not a BlockOperator");
      Array2D<const HypreParMatrix*> blocks(bop->NumRowBlocks(), bop->NumColBlocks());
      for (int j=0; j<bop->NumColBlocks(); j++)
      {
         for (int i=0; i<bop->NumRowBlocks(); i++)
         {
            if (bop->IsZeroBlock(i,j)) { continue; }
            const HypreParMatrix *m =
               dynamic_cast<const HypreParMatrix*>(&bop->GetBlock(i,j));
            MFEM_VERIFY(m != nullptr, "Not a HypreParMatrix");
            blocks(i,j) = m;
         }
      }
      mono.reset(HypreParMatrixFromBlocks(blocks));
      MUMPSSolver::SetOperator(*mono);
   };
};

class ParMonolithicBlockNonlinearFormWrapper : public Operator
{
private:
   mutable std::unique_ptr<HypreParMatrix> mono;
   ParBlockNonlinearForm &op;
public:
   ParMonolithicBlockNonlinearFormWrapper(ParBlockNonlinearForm &op_)
      : Operator(op_.Height(), op_.Width()), op(op_) {}
   real_t GetEnergy(const Vector &x) const
   {
      return op.GetEnergy(x);
   }
   void Mult(const Vector &x, Vector &y) const override
   {
      op.Mult(x, y);
   }

   HypreParMatrix &GetGradient(const Vector &x) const override
   {
      BlockOperator &grad = op.GetGradient(x);
      if (grad.NumColBlocks() == 1) { return static_cast<HypreParMatrix&>(grad.GetBlock(0,0)); }

      Array2D<const HypreParMatrix*> blocks(grad.NumRowBlocks(), grad.NumColBlocks());
      for (int j=0; j<grad.NumColBlocks(); j++)
      {
         for (int i=0; i<grad.NumRowBlocks(); i++)
         {
            if (grad.IsZeroBlock(i,j)) { continue; }
            const HypreParMatrix *m =
               dynamic_cast<const HypreParMatrix*>(&grad.GetBlock(i,j));
            MFEM_VERIFY(m != nullptr, "Not a HypreParMatrix");
            blocks(i,j) = m;
         }
      }
      mono.reset(HypreParMatrixFromBlocks(blocks));
      return *mono;
   }
};

inline std::tuple<std::unique_ptr<FiniteElementSpace>, std::unique_ptr<L2_FECollection>>
      QSpaceToFESpace(QuadratureSpace &qs)
{
   Mesh *mesh = qs.GetMesh();
   const int dim = mesh->Dimension();
   Geometry::Type geom = mesh->GetTypicalElementGeometry();
   MFEM_VERIFY(geom != Geometry::TRIANGLE &&
               geom != Geometry::TETRAHEDRON &&
               geom != Geometry::PRISM &&
               geom != Geometry::PYRAMID,
               "QSpaceToFESpace: only support tensor product elements");
   std::unique_ptr<L2_FECollection> fec
      = std::make_unique<L2_FECollection> (qs.GetOrder()/2, dim);

   std::unique_ptr<FiniteElementSpace> fes;
#ifdef MFEM_USE_MPI
   ParMesh *pmesh = dynamic_cast<ParMesh*>(qs.GetMesh());
   if (pmesh) { fes = std::make_unique<ParFiniteElementSpace>(pmesh, fec.get()); }
#endif
   if (!fes) { fes = std::make_unique<FiniteElementSpace>(mesh, fec.get()); }
   return std::make_tuple(std::move(fes), std::move(fec));
}

inline Array<int> GetOffsets(const Array<FiniteElementSpace*> &fespaces)
{
   Array<int> offsets(fespaces.Size() + 1);
   offsets[0] = 0;
   for (int i=0; i<fespaces.Size(); i++)
   {
      offsets[i+1] = offsets[i] + fespaces[i]->GetVSize();
   }
   return std::move(offsets);
}
inline Array<int> GetOffsets(const std::vector<FiniteElementSpace*> &fespaces)
{
   Array<int> offsets(fespaces.size() + 1);
   offsets[0] = 0;
   for (int i=0; i<fespaces.size(); i++)
   {
      offsets[i+1] = offsets[i] + fespaces[i]->GetVSize();
   }
   return std::move(offsets);
}
inline Array<int> GetTrueOffsets(const Array<FiniteElementSpace*> &fespaces)
{
   Array<int> offsets(fespaces.Size() + 1);
   offsets[0] = 0;
   for (int i=0; i<fespaces.Size(); i++)
   {
      offsets[i+1] = offsets[i] + fespaces[i]->GetTrueVSize();
   }
   return std::move(offsets);
}
inline Array<int> GetTrueOffsets(const std::vector<FiniteElementSpace*>
                                 &fespaces)
{
   Array<int> offsets(fespaces.size() + 1);
   offsets[0] = 0;
   for (int i=0; i<fespaces.size(); i++)
   {
      offsets[i+1] = offsets[i] + fespaces[i]->GetTrueVSize();
   }
   return std::move(offsets);
}

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
      return std::sqrt(v*v);
   }
};

class BooleanCoefficient : public Coefficient
{
private:
   Coefficient &cf;
   std::function<bool(real_t)> func;
public:
   BooleanCoefficient(Coefficient &cf, std::function<bool(real_t)> func)
      : cf(cf), func(func) {}
   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      return func(cf.Eval(T, ip));
   }
};
class MatrixQuadratureFunctionCoefficient : public MatrixCoefficient
{
private:
   const QuadratureFunction &QuadF; //do not own

   mutable Vector vals;
public:
   /// Constructor with a quadrature function as input
   MatrixQuadratureFunctionCoefficient(const QuadratureFunction &qf, int h, int w)
      : MatrixCoefficient(h, w), QuadF(qf)
   {
      MFEM_VERIFY(qf.GetVDim() == h*w,
                  "MatrixQuadratureFunctionCoefficient: size mismatch");
   }

   const QuadratureFunction& GetQuadFunction() const { return QuadF; }

   using MatrixCoefficient::Eval;
   void Eval(DenseMatrix &M, ElementTransformation &T,
             const IntegrationPoint &ip) override
   {
      MFEM_ASSERT(QuadF.GetIntRule(T.ElementNo).GetNPoints() > ip.index,
                  "MatrixQuadratureFunctionCoefficient::Eval: "
                  "Integration point index out of range.");
      MFEM_ASSERT(&(QuadF.GetIntRule(T.ElementNo).IntPoint(ip.index)) == &ip,
                  "MatrixQuadratureFunctionCoefficient::Eval: "
                  "Integration point mismatch.");
      M.SetSize(height, width);
      vals.SetData(M.GetData());
      QuadF.GetValues(T.ElementNo, ip.index, vals);
   }

   void Project(QuadratureFunction &qf, bool transpose=false) override
   {
      MFEM_VERIFY(qf.GetSpace() == QuadF.GetSpace(),
                  "MatrixQuadratureFunctionCoefficient::Project: "
                  "QuadratureFunction space mismatch.");
      qf.SetVDim(QuadF.GetVDim());
      qf = this->QuadF; // copy
   }
};

class DomainQLFIntegrator : public LinearFormIntegrator
{
private:
   QuadratureSpace &qs;
   QuadratureFunction &f;
   const Vector &weights;
   Vector fvals;
   Vector shapevals;
public:
   DomainQLFIntegrator(QuadratureFunction &qf)
      : qs(static_cast<QuadratureSpace&>(*qf.GetSpace()))
      , f(qf), weights(qs.GetWeights())
   {
      MFEM_VERIFY(qf.GetVDim() == 1, "DomainQLFIntegrator: only support scalar QF");
   }
   void AssembleRHSElementVect(const FiniteElement &el,
                               ElementTransformation &Tr,
                               Vector &elvect) override
   {
      const IntegrationRule &ir = qs.GetIntRule(Tr.ElementNo);
      elvect.SetSize(el.GetDof());
      elvect = 0.0;
      shapevals.SetSize(el.GetDof());
      int offset = qs.Offset(Tr.ElementNo);
      for (int i=0; i<ir.GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir.IntPoint(i);
         el.CalcShape(ip, shapevals);
         elvect.Add(f(offset + i)*weights(offset + i), shapevals);
      }
   }
};

class DomainVectorQLFIntegrator : public LinearFormIntegrator
{
private:
   QuadratureSpace &qs;
   QuadratureFunction &f;
   const Vector &weights;
   Vector fvals;
   Vector shapevals;
public:
   DomainVectorQLFIntegrator(QuadratureFunction &qf)
      : qs(static_cast<QuadratureSpace&>(*qf.GetSpace()))
      , f(qf), weights(qs.GetWeights())
   { }
   void AssembleRHSElementVect(const FiniteElement &el,
                               ElementTransformation &Tr,
                               Vector &elvect) override
   {
      const IntegrationRule &ir = qs.GetIntRule(Tr.ElementNo);
      elvect.SetSize(el.GetDof()*f.GetVDim());
      elvect = 0.0;
      DenseMatrix elmat(elvect.GetData(), el.GetDof(), f.GetVDim());
      shapevals.SetSize(el.GetDof());
      int offset = qs.Offset(Tr.ElementNo);
      Vector fvals(f.GetVDim());
      for (int i=0; i<ir.GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir.IntPoint(i);
         el.CalcShape(ip, shapevals);
         f.GetValues(Tr.ElementNo, i, fvals);
         AddMult_a_VWt(weights(offset + i), shapevals, fvals, elmat);
      }
   }
};

inline real_t Integrate(QuadratureFunction &qf)
{
   return qf.Integrate();
}
inline real_t Integrate(Coefficient &cf, int order, Mesh &mesh)
{
   real_t integral = 0.0;
   for (int i=0; i<mesh.GetNE(); i++)
   {
      ElementTransformation *Tr = mesh.GetElementTransformation(i);
      const IntegrationRule &ir = IntRules.Get(mesh.GetElementGeometry(i), order);
      for (int j=0; j<ir.GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir.IntPoint(j);
         Tr->SetIntPoint(&ip);
         integral += cf.Eval(*Tr, ip) * Tr->Weight() * ip.weight;
      }
   }
#ifdef MFEM_USE_MPI
   // real_t integral = qf.Integrate();
   ParMesh *pmesh = dynamic_cast<ParMesh*>(&mesh);
   if (pmesh)
   {
      MPI_Allreduce(MPI_IN_PLACE, &integral, 1, MPI_DOUBLE, MPI_SUM,
                    pmesh->GetComm());
   }
#endif
   return integral;
}
inline real_t dot(MPI_Comm comm, QuadratureFunction &a, QuadratureFunction &b)
{
   MFEM_VERIFY(a.GetVDim() == b.GetVDim(), "Vector dimension mismatch.");
   MFEM_VERIFY(a.GetSpace() == b.GetSpace(), "Quadrature space mismatch.");
   QuadratureSpaceBase *qs = a.GetSpace();
   Vector a_vals, b_vals;
   real_t result = 0.0;
   const Vector &weights = qs->GetWeights();
   int ctr=0;
   for (int i=0; i<qs->GetNE(); i++)
   {
      const IntegrationRule &ir = qs->GetIntRule(i);
      for (int j=0; j<ir.GetNPoints(); j++)
      {
         a.GetValues(i, j, a_vals);
         b.GetValues(i, j, b_vals);
         result += weights[ctr++] * (a_vals * b_vals);
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                 comm);
   return result;
}
inline real_t dot(MPI_Comm comm, GridFunction &a, QuadratureFunction &b)
{
   MFEM_VERIFY(a.FESpace()->GetVDim() == b.GetVDim(),
               "Vector dimension mismatch.");
   QuadratureSpaceBase *qs = b.GetSpace();
   Vector a_vals, b_vals;
   real_t result = 0.0;
   const Vector &weights = qs->GetWeights();
   int ctr=0;
   for (int i=0; i<qs->GetNE(); i++)
   {
      const IntegrationRule &ir = qs->GetIntRule(i);
      for (int j=0; j<ir.GetNPoints(); j++)
      {
         a.GetVectorValue(i, ir[j], a_vals);
         b.GetValues(i, j, b_vals);
         result += weights[ctr++] * (a_vals * b_vals);
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                 comm);
   return result;
}
inline real_t dot(MPI_Comm comm, GridFunction &a, VectorCoefficient &b,
                  QuadratureSpace &qs)
{
   MFEM_VERIFY(a.FESpace()->GetVDim() == b.GetVDim(),
               "Vector dimension mismatch.");
   Vector a_vals, b_vals;
   real_t result = 0.0;
   const Vector &weights = qs.GetWeights();
   int ctr=0;
   for (int i=0; i<qs.GetNE(); i++)
   {
      const IntegrationRule &ir = qs.GetIntRule(i);
      auto *Tr = qs.GetMesh()->GetElementTransformation(i);
      for (int j=0; j<ir.GetNPoints(); j++)
      {
         a.GetVectorValue(i, ir[j], a_vals);
         b.Eval(b_vals, *Tr, ir[j]);;
         result += weights[ctr++] * (a_vals * b_vals);
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                 comm);
   return result;
}
inline real_t ParNormL1(MPI_Comm comm, QuadratureFunction &a)
{
   QuadratureSpaceBase *qs = a.GetSpace();
   Vector a_vals, b_vals;
   real_t result = 0.0;
   const Vector &weights = qs->GetWeights();
   int ctr=0;
   for (int i=0; i<qs->GetNE(); i++)
   {
      const IntegrationRule &ir = qs->GetIntRule(i);
      for (int j=0; j<ir.GetNPoints(); j++)
      {
         a.GetValues(i, j, a_vals);
         for (auto &v : a_vals) { result += weights[ctr] * fabs(v); }
         ctr++;
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPITypeMap<real_t>::mpi_type, MPI_SUM,
                 comm);
   return result;
}
inline real_t dot(const QuadratureFunction &a, VectorCoefficient &b)
{
   VectorQuadratureFunctionCoefficient a_cf(a);
   InnerProductCoefficient dot_cf(a_cf, b);
   return a.GetSpace()->Integrate(dot_cf);
}

/// @brief Get the essential true dofs from component-wise boundary conditions
inline void GetEssentialTrueDofs(const FiniteElementSpace &fes,
                                 const Array2D<int> &ess_bdr,
                                 Array<int> &ess_tdofs)
{
   MFEM_VERIFY(ess_bdr.NumRows() == fes.GetVDim(),
               "GetEssentialTrueDofs: ess_bdr height must match fes vdim.");
   Array<int> bdr_marker;
   Array<int> curr_tdofs;
   ess_tdofs.SetSize(0);
   for (int i=0; i<ess_bdr.NumRows(); i++)
   {
      curr_tdofs.SetSize(0);
      ess_bdr.GetRow(i, bdr_marker);
      fes.GetEssentialTrueDofs(bdr_marker, curr_tdofs, i);
      ess_tdofs.Append(curr_tdofs);
   }
}

/// @brief Set element attribute based on a mask function.
/// The mask function will be evaluated at the center of each (local) element
/// using the lowest order integration point.
/// If the mask returns true, the element attribute will be set to `attr`.
///
/// @param mesh The target mesh
/// @param attr The attribute to set
/// @param mask A function that takes a Vector (the coordinates) and returns
///            a boolean value.
/// @param update_attr_array If true, update the attribute arrays in the mesh
///                           after setting the attributes. Default is true.
inline void SetElementAttribute(Mesh &mesh, int attr,
                                std::function<bool(const Vector&)>mask,
                                bool update_attr_array=true)
{
   Vector x(mesh.SpaceDimension());
   for (int i=0; i<mesh.GetNE(); i++)
   {
      ElementTransformation &Tr = *mesh.GetElementTransformation(i);
      const IntegrationRule &ir = IntRules.Get(Tr.GetGeometryType(), 0);
      Tr.Transform(ir[0], x);
      if (mask(x))
      {
         mesh.SetBdrAttribute(i, attr);
      }
   }
   if (update_attr_array)
   {
      mesh.SetAttributes(true, false);
   }
}

/// @brief Set boundary attribute based on a mask function.
/// The mask function will be evaluated at the center of each (local) boundary face,
/// using the lowest order integration point.
/// If the mask returns true, the boundary attribute will be set to `attr`.
///
/// @param mesh The target mesh
/// @param attr The attribute to set
/// @param mask A function that takes a Vector (the coordinates) and returns
///            a boolean value.
/// @param update_attr_array If true, update the attribute arrays in the mesh
///                           after setting the attributes. Default is true.
inline void SetBoundaryAttribute(Mesh &mesh, int attr,
                                 std::function<bool(const Vector&)>mask,
                                 bool update_attr_array=true)
{
   Vector x(mesh.SpaceDimension());
   for (int i=0; i<mesh.GetNBE(); i++)
   {
      FaceElementTransformations &Tr = *mesh.GetBdrFaceTransformations(i);
      const IntegrationRule &ir = IntRules.Get(Tr.GetGeometryType(), 0);
      Tr.Transform(ir[0], x);
      if (mask(x))
      {
         mesh.SetBdrAttribute(i, attr);
      }
   }
   if (update_attr_array)
   {
      mesh.SetAttributes(false, true);
   }
}
}
