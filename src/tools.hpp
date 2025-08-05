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
};
