// Matrix.hpp - Utilidades propias de álgebra lineal y estadística
#pragma once
#include <vector>

using Vec = std::vector<double>;
using Mat = std::vector<Vec>;  // matriz por filas: Mat[i][j] = fila i, columna j

namespace linalg {
// Devuelve la transpuesta de A (m x n -> n x m).
Mat transpose(const Mat& A);
// Producto matricial A (m x n) * B (n x p) = C (m x p).
Mat multiply(const Mat& A, const Mat& B);
// Producto matriz-vector A (m x n) * x (n) = y (m).
Vec multiply(const Mat& A, const Vec& x);
// Resuelve A x = b por eliminación gaussiana con pivoteo parcial. Lanza excepción si A es singular.
Vec solve(Mat A, Vec b);
// Devuelve [1 | X]: antepone una columna de unos (término de sesgo) a X.
Mat addBiasColumn(const Mat& X);
}  // namespace linalg

namespace stats {
// Media aritmética de un vector.
double mean(const Vec& v);
// Medias de cada columna de X.
Vec columnMeans(const Mat& X);
// Desviación estándar poblacional de cada columna (si es 0 devuelve 1 para evitar división por cero).
Vec columnStds(const Mat& X, const Vec& means);
}  // namespace stats
