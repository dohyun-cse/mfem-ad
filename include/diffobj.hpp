#pragma once

#include "mfem.hpp"
#include "ad_native.hpp"
#include "ad_intg.hpp"
#include "tools.hpp"
#include "pg.hpp"

namespace mfem
{

/** @brief Sets one component of a vector QuadratureFunction from a scalar
 *         QuadratureFunction.
 *
 *  This function copies the specified component @a comp from the vector quadrature
 *  function @a in into the scalar quadrature function @a out.
 *
 *  @param[in]  in   The source vector QuadratureFunction.
 *  @param[in]  comp The component index in the input QuadratureFunction to extract.
 *  @param[out] out  The destination scalar QuadratureFunction.
 */
inline void ExtractComponent(const QuadratureFunction &in, const int comp,
                             QuadratureFunction &out)
{
   MFEM_VERIFY(in.GetVDim() > comp,
               "Input quadrature function does not have enough components.");
   MFEM_VERIFY(in.GetSpace() == out.GetSpace(),
               "Input and output quadrature functions must be defined on the "
               "same space.");
   const int vdim = in.GetVDim();
   out.SetVDim(1);
   bool use_dev = in.UseDevice() || out.UseDevice();
   const real_t* in_data = in.Read();
   real_t* out_data = out.Write();
   mfem::forall_switch(use_dev, out.Size(), [=] MFEM_HOST_DEVICE (int i)
   { out_data[i] = in_data[i*vdim + comp]; });
}

/** @brief Sets one component of a vector QuadratureFunction from a scalar
 *         QuadratureFunction.
 *
 *  This function copies the values from the scalar quadrature function @a in
 *  into the specified component @a comp of the vector quadrature function @a out.
 *
 *  @note The input function @a in must be scalar (VDim=1).
 *  @note The output function @a out must have at least `comp + 1` components.
 *  @note Both functions must be defined on the same QuadratureSpace.
 *
 *  @param[in]  in   The source scalar QuadratureFunction.
 *  @param[in]  comp The component index in the output QuadratureFunction to set.
 *  @param[out] out  The destination vector QuadratureFunction.
 */
inline void SetComponent(const QuadratureFunction &in, const int comp,
                         QuadratureFunction &out)
{
   MFEM_VERIFY(in.GetVDim() == 1,
               "Input quadrature function must be scalar.");
   MFEM_VERIFY(out.GetVDim() > comp,
               "Input quadrature function does not have enough components.");
   MFEM_VERIFY(in.GetSpace() == out.GetSpace(),
               "Input and output quadrature functions must be defined on the "
               "same space.");
   const int vdim = out.GetVDim();
   bool use_dev = in.UseDevice() || out.UseDevice();
   const real_t* in_data = in.Read();
   real_t* out_data = out.Write();
   mfem::forall_switch(use_dev, in.Size(), [=] MFEM_HOST_DEVICE (int i)
   { out_data[i*vdim + comp] = in_data[i]; });
}

/** @brief Abstract base class for operators with forward and adjoint actions.
 *
 *  This class represents a transformation y = F(x). It is designed to be used
 *  in gradient-based optimization problems where the gradient can be computed
 *  using the adjoint method.
 *
 *  Derived classes must implement the pure virtual methods:
 *  - Mult(): The forward operation, y = F(x).
 *  - AdjointMult(): The backward/adjoint operation, which computes the action
 *    of the Jacobian transpose, dJdx = (dF/dx)^T * dJdy.
 *
 *  @note This operator is stateful. The AdjointMult() operation depends on the
 *  input vector 'x' from the most recent call to GetGradient().
 *  A typical use case is:
 *  1. Call Mult(x, y) to compute the forward pass and store internal state.
 *  2. Call GetGradient(x).MultTranspose(dJdy, dJdx) to compute the corresponding
 *     backward pass. This will call AdjointMult() where `x` is stored in x_ptr
 */
class ForwardBackwardOperator : public Operator
{
public:
   ForwardBackwardOperator()
      : Operator(), grad_op(*this) {}
   ForwardBackwardOperator(int size)
      : Operator(size), grad_op(*this) {}
   ForwardBackwardOperator(int height, int width)
      : Operator(width, height), grad_op(*this) {}

   /// @brief Forward operation, y = F(x)
   virtual void Mult(const Vector &x, Vector &y) const = 0;
   Operator &GetGradient(const Vector &x) const
   {
      x_ptr = &x;
      return grad_op;
   }
protected:
   /// @brief Adjoint operation, y = (F')^T x
   /// with stored x from the last GetGradient call
   virtual void AdjointMult(const Vector &x, Vector &y) const = 0;
   mutable const Vector *x_ptr; // pointer to x from last GetGradientCall
private:
   class GradientOperator : public Operator
   {
   private:
      const ForwardBackwardOperator &op;
   public:
      GradientOperator(const ForwardBackwardOperator &parent)
         : Operator(parent.Width(), parent.Height())
         , op(parent)
      { }
      void Mult(const Vector &x, Vector &y) const override
      {
         MFEM_ABORT("Use MultTranspose for adjoint operation");
      }
      void MultTranspose(const Vector &dJdy, Vector &dJdx) const override
      {
         op.AdjointMult(dJdy, dJdx);
      }
   };
   mutable GradientOperator grad_op;
};

/**
 * @brief Represents a composition of multiple ForwardBackwardOperator instances.
 *
 * This class chains a sequence of ForwardBackwardOperator objects, say
 * A_1, A_2, ..., A_n, to form a single composite operator A = A_n * ... * A_2 * A_1.
 *
 * The forward operation is `y = A(x)`.
 * The adjoint operation computes `dJdx = J_A(x)^T * dJdy`, where the Jacobian `J_A(x)`
 *
 * is a product of the individual Jacobians: `J_A(x) = J_n * ... * J_2 * J_1`.
 * Therefore, the adjoint is `dJdx = (J_1^T * J_2^T * ... * J_n^T) * dJdy`.
 * intermediate results stored during the most recent call to Mult(). Therefore,
 * a call to Mult() must precede every call to AdjointMult() to ensure the
 * correct state is used for the gradient calculations.
 * GetGradient(x) will set x_ptr. As AdjointMult is a protected method,
 * the public interface is through GetGradient(x).MultTranspose(dJdy, dJdx).
 *
 * The class also provides methods to access these intermediate results from both
 * the forward and adjoint computations, which can be useful for debugging or
 * implementing more complex algorithms.
 */
class CompositeOperator : public ForwardBackwardOperator
{
   /// The sequence of operators to be composed.
   std::vector<std::reference_wrapper<ForwardBackwardOperator>> ops;
   /// Stores intermediate vectors from the forward pass (Mult).
   /// Except for the initial input vector x, which will be provided by the user
   /// Mult(x, y) call and GetGradient(x) call (stored in x_ptr).
   mutable std::vector<std::unique_ptr<Vector>> intermediate_inputs;
   /// Stores intermediate vectors from the adjoint pass (AdjointMult).
   /// Except for the final input vector dJdy, which will be directly returned
   /// to the user in AdjointMult(dJdy, dJdx).
   mutable std::vector<std::unique_ptr<Vector>> adjoint_results;
   int last_output_size; // to check dimension consistency
public:

   /**
    * @brief Constructs a CompositeOperator.
    *
    * @param height The output dimension of the final operator in the chain.
    * @param width The input dimension of the first operator in the chain.
    */
   CompositeOperator(int height, int width)
      : ForwardBackwardOperator(height, width), last_output_size(width)
   { }

   /**
    * @brief Adds an operator to the composition chain.
    *
    * The operators are applied in the order they are added. The input dimension
    * (Width) of the new operator `op` must match the output dimension (Height)
    * of the previously added operator.
    *
    * @param op The ForwardBackwardOperator to add to the chain. The
    *           CompositeOperator stores a reference, so the lifetime of `op`
    *           must be managed externally.
    */
   void AddOperator(ForwardBackwardOperator &op)
   {
      MFEM_VERIFY(last_output_size == op.Width(),
                  "CompositeOperator: Operator input size does not match.");
      last_output_size = op.Height();
      if (ops.empty())
      {
         intermediate_inputs.push_back(std::make_unique<Vector>(op.Width()));
         adjoint_results.push_back(std::make_unique<Vector>(op.Width()));
      }
      ops.push_back(op);
   }
   Vector &GetIntermediateOutput(const int idx)
   {
      MFEM_VERIFY(idx >= 0 && idx < intermediate_inputs.size(),
                  "CompositeOperator: Intermediate output index out of range.");
      return *intermediate_inputs[idx];
   }
   const Vector &GetIntermediateOutput(const int idx) const
   {
      MFEM_VERIFY(idx >= 0 && idx < intermediate_inputs.size(),
                  "CompositeOperator: Intermediate output index out of range.");
      return *intermediate_inputs[idx];
   }
   Vector &GetIntermediateAdjoint(const int idx)
   {
      MFEM_VERIFY(idx >= 0 && idx < adjoint_results.size(),
                  "CompositeOperator: Intermediate adjoint index out of range.");
      return *adjoint_results[idx];
   }
   const Vector &GetIntermediateAdjoint(const int idx) const
   {
      MFEM_VERIFY(idx >= 0 && idx < adjoint_results.size(),
                  "CompositeOperator: Intermediate adjoint index out of range.");
      return *adjoint_results[idx];
   }

   /**
    * @brief Applies the composite forward operator: y = (A_n * ... * A_1) * x.
    *
    * This method computes the forward pass, applying each operator in the chain
    * sequentially. The intermediate results are stored internally and are required
    * by the subsequent AdjointMult() call.
    *
    * @param x The input vector. Its size must match the operator's Width().
    * @param y The output vector. Its size will be set to the operator's Height().
    */
   void Mult(const Vector &x, Vector &y) const override
   {
      MFEM_VERIFY(x.Size() == Width(),
                  "CompositeOperator::Mult: x size does not match operator width");
      MFEM_VERIFY(last_output_size == Height(),
                  "CompositeOperator::Mult: Last operator output size does not match operator height");
      if (ops.empty()) { y = x; return; }
      const Vector *current_input = &x;
      for (int i=0; i<ops.size()-1; i++)
      {
         ops[i].get().Mult(*current_input, *intermediate_inputs[i]);
         current_input = intermediate_inputs[i].get();
      }
      y.SetSize(Height());
      ops.back().get().Mult(*current_input, y);
   }
protected:
   /**
    * @brief Applies the composite adjoint operator: dJdx = (J_1^T * ... * J_n^T) * dJdy.
    *
    * This method computes the action of the transpose of the composite operator's
    * Jacobian, where J_i is the Jacobian of the i-th operator. It uses the
    * intermediate states stored during the last call to Mult(). Therefore,
    * Mult() must be called before this method to provide the points at which
    * the Jacobians are evaluated.
    *
    * @param dJdy The input vector for the adjoint operation (e.g., gradient of a
    *             cost function with respect to the output y). Its size must
    *             match the operator's Height().
    * @param dJdx The output vector (e.g., gradient with respect to the input x).
    *             Its size will be set to the operator's Width().
    */
   void AdjointMult(const Vector &dJdy, Vector &dJdx) const override
   {
      MFEM_VERIFY(dJdy.Size() == Height(),
                  "CompositeOperator::AdjointMult: dJdy size does not match operator height");
      MFEM_VERIFY(last_output_size == Height(),
                  "CompositeOperator::AdjointMult: Last operator output size does not match operator height");
      MFEM_VERIFY(x_ptr != nullptr,
                  "CompositeOperator::AdjointMult: Must call GetGradient(x) before AdjointMult.");
      if (ops.empty()) { dJdx = dJdy; return; }
      const Vector *current_adjoint = &dJdy;
      for (int i=ops.size()-1; i>=1; i--)
      {
         ops[i].get().GetGradient(*intermediate_inputs[i-1]).MultTranspose(
            *current_adjoint, *adjoint_results[i-1]);
         current_adjoint = adjoint_results[i-1].get();
      }

      dJdx.SetSize(Width());
      ops[0].get().GetGradient(*x_ptr).MultTranspose(
         *current_adjoint, dJdx);
   }
};

/** @brief A ADFunction-based ForwardBackwardOperator
 *
 * This class implements a ForwardBackwardOperator using an ADFunction
 * where the forward operation is F(x)
 * and the adjoint operation computes F'(x)^T * dJdy using the autodiff gradient.
 *
 * The input and output are quadrature functions defined on the same QuadratureSpace.
**/
class ForwardBackwardADOperator : public ForwardBackwardOperator
{
   static ADFunction &ValidateF(ADFunction *F)
   {
      MFEM_VERIFY(F != nullptr,
                  "ForwardBackwardADOperator: ADFunction pointer is null.");
      return *F;
   }
   int input_vdim;
   std::unique_ptr<ADFunction> owned_F;
   mutable DifferentiableCoefficient F_cf;
   VectorCoefficient &dF_cf;
   mutable QuadratureFunction x_qf;
   QuadratureSpace &qspace;
public:
   /// @brief Create a ForwardBackwardADOperator
   ForwardBackwardADOperator(ADFunction &F, QuadratureSpace &qspace)
      : ForwardBackwardOperator(qspace.GetSize(),
                                qspace.GetSize()*F.n_input)
      , input_vdim(F.n_input), F_cf(F), dF_cf(F_cf.Gradient())
      , qspace(qspace), x_qf(&qspace, F.n_input)
   {
      MFEM_VERIFY(F.n_input > 0,
                  "ForwardBackwardADOperator: n_input and n_output must be positive");
      F_cf.AddInput(&x_qf);
   }
   /// @brief Create a ForwardBackwardADOperator that takes ownership of the ADFunction pointer.
   /// F should not be a nullptr
   ForwardBackwardADOperator(ADFunction *F, QuadratureSpace &qspace)
      : ForwardBackwardADOperator(ValidateF(F), qspace)
   { owned_F.reset(F); }

   void Mult(const Vector &x, Vector &y) const override
   {
      MFEM_VERIFY(x.Size() == Width(),
                  "ForwardBackwardADOperator::Mult: x size does not match operator width");
      y.SetSize(Height());
      x_qf.SetData(x.GetData());
      QuadratureFunction y_qf(&qspace, y.GetData(), 1);
      F_cf.Project(y_qf);
   }
protected:
   void AdjointMult(const Vector &dJdy, Vector &dJdx) const override
   {
      MFEM_VERIFY(dJdy.Size() == Height(),
                  "ForwardBackwardADOperator::AdjointMult: x size does not match operator height");
      x_qf.SetData(x_ptr->GetData()); // x from last GetGradient call
      dJdx.SetSize(Height());
      QuadratureFunction dJdy_qf(&qspace, dJdy.GetData());
      QuadratureFunctionCoefficient dJdy_cf(dJdy_qf);
      ScalarVectorProductCoefficient dJdx_cf(dJdy_cf, dF_cf);
      QuadratureFunction dJdx_qf(&qspace, dJdx.GetData(), input_vdim);
      dJdx_cf.Project(dJdx_qf);
   }
};

/** @brief A ADVectorFunction-based ForwardBackwardOperator
 *
 * This class implements a ForwardBackwardOperator using an ADVectorFunction
 * where the forward operation is F(x)
 * and the adjoint operation computes F'(x)^T * dJdy using the autodiff gradient.
 *
 * The input and output are quadrature functions defined on the same QuadratureSpace.
**/
class ForwardBackwardADVectorOperator : public ForwardBackwardOperator
{
   int input_vdim;
   int output_vdim;
   std::unique_ptr<ADVectorFunction> owned_F;
   mutable DifferentiableVectorCoefficient F_cf;
   MatrixCoefficient &dF_cf;
   mutable QuadratureFunction x_qf;
   QuadratureSpace &qspace;
public:
   /// @brief Create a ForwardBackwardADVectorOperator
   ForwardBackwardADVectorOperator(ADVectorFunction &F, QuadratureSpace &qspace)
      : ForwardBackwardOperator(qspace.GetSize()*F.n_output,
                                qspace.GetSize()*F.n_input)
      , input_vdim(F.n_input), output_vdim(F.n_output)
      , F_cf(F), dF_cf(F_cf.Gradient())
      , qspace(qspace), x_qf(&qspace, F.n_input)
   {
      MFEM_VERIFY(F.n_input > 0,
                  "ForwardBackwardADOperator: n_input and n_output must be positive");
      F_cf.AddInput(&x_qf);
   }
   /// @brief Create a ForwardBackwardADVectorOperator that takes ownership of the ADVectorFunction pointer.
   /// F should not be a nullptr
   ForwardBackwardADVectorOperator(ADVectorFunction *F, QuadratureSpace &qspace)
      : ForwardBackwardADVectorOperator(*F, qspace)
   { owned_F.reset(F); }

   void Mult(const Vector &x, Vector &y) const override
   {
      MFEM_VERIFY(x.Size() == Width(),
                  "ForwardBackwardADOperator::Mult: x size does not match operator width");
      y.SetSize(Height());
      x_qf.SetData(x.GetData());
      QuadratureFunction y_qf(&qspace, y.GetData(), output_vdim);
      F_cf.Project(y_qf);
   }
protected:
   void AdjointMult(const Vector &dJdy, Vector &dJdx) const override
   {
      MFEM_VERIFY(dJdy.Size() == Height(),
                  "ForwardBackwardADOperator::AdjointMult: x size does not match operator height");
      x_qf.SetData(x_ptr->GetData()); // x from last GetGradient call
      dJdx.SetSize(Height());
      QuadratureFunction dJdy_qf(&qspace, dJdy.GetData());
      VectorQuadratureFunctionCoefficient dJdy_cf(dJdy_qf);
      TransposeMatrixCoefficient dF_T(dF_cf);
      MatrixVectorProductCoefficient dJdx_cf(dF_T, dJdy_cf);
      QuadratureFunction dJdx_qf(&qspace, dJdx.GetData(), input_vdim);
      dJdx_cf.Project(dJdx_qf);
   }
};

}
