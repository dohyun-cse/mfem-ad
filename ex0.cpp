#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "src/logger.hpp"
#include "src/ad-native.hpp"

using namespace std;
using namespace mfem;

MAKE_AD_FUNCTION(MyADFunction, T, VEC, MAT, x, dummy,
{
   return sin(x(0))*exp(x(1)) + pow(x(2), 3.0);
});

void jacobian(const Vector &x, Vector &J)
{
   J.SetSize(x.Size());
   J[0] = std::cos(x(0)) * std::exp(x(1));
   J[1] = std::sin(x(0)) * std::exp(x(1));
   J[2] = 3.0 * std::pow(x(2), 2.0);
}
void hessian(const Vector &x, DenseMatrix &H)
{
   // J[0] = cos(x(0)) * exp(x(1));
   H.SetSize(x.Size(), x.Size());
   H(0, 0) = -std::sin(x(0)) * std::exp(x(1));
   H(0, 1) = std::cos(x(0)) *std::exp(x(1));
   H(0, 2) = 0.0;

   // J[1] = sin(x(0)) * exp(x(1));
   H(1, 0) = std::cos(x(0)) * std::exp(x(1));
   H(1, 1) = std::sin(x(0)) * std::exp(x(1));
   H(1, 2) = 0.0;

   // J[2] = 3.0 * pow(x(2), 2.0);
   H(2, 0) = 0.0;
   H(2, 1) = 0.0;
   H(2, 2) = 6.0 * std::pow(x(2), 1.0);
}

int main(int argc, char *argv[])
{
   MyADFunction f(3, 0);

   Vector x({0.5, 1.0, -1.0});
   Vector param({});

   // Vector jac, jac_ref;
   // f.Gradient(x, param, jac);
   // jacobian(x, jac_ref);

   DenseMatrix hess, hess_ref;
   f.Hessian(x, param, hess);
   hessian(x, hess_ref);

   out << "Value : " << f(x, param) << std::endl;

   // out << "Jacobian  : ";
   // for (auto & j : jac) { out << j << " "; }
   // out << std::endl;
   // out << "Reference : ";
   // for (auto & j : jac_ref) { out << j << " "; }
   // out << std::endl;

   out << "Hessian : " << std::endl;
   for (int i = 0; i < hess.Height(); i++)
   {
      for (int j = 0; j < hess.Width(); j++)
      {
         out << std::scientific << hess(i, j) << " ";
      }
      out << "; ";
   }
   out << std::endl;

   out << "Reference: " << std::endl;
   for (int i = 0; i < hess.Height(); i++)
   {
      for (int j = 0; j < hess.Width(); j++)
      {
         out << std::scientific << hess_ref(i, j) << " ";
      }
      out << "; ";
   }

   out << std::endl;

   return 0;
}
