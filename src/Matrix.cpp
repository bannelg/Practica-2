#include "Matrix.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace linalg {

Mat transpose(const Mat& A) {
    if (A.empty()) return {};
    for (const auto& row : A)
        if (row.size() != A[0].size())
            throw std::invalid_argument("transpose: filas de distinto tamano");
    Mat T(A[0].size(), Vec(A.size()));
    for (size_t i = 0; i < A.size(); ++i)
        for (size_t j = 0; j < A[0].size(); ++j) T[j][i] = A[i][j];
    return T;
}

Mat multiply(const Mat& A, const Mat& B) {
    if (A.empty() || B.empty() || A[0].size() != B.size())
        throw std::invalid_argument("multiply: dimensiones incompatibles");
    for (const auto& row : A)
        if (row.size() != A[0].size())
            throw std::invalid_argument("multiply: filas de distinto tamano en A");
    for (const auto& row : B)
        if (row.size() != B[0].size())
            throw std::invalid_argument("multiply: filas de distinto tamano en B");
    Mat C(A.size(), Vec(B[0].size(), 0.0));
    for (size_t i = 0; i < A.size(); ++i)
        for (size_t k = 0; k < B.size(); ++k)
            for (size_t j = 0; j < B[0].size(); ++j) C[i][j] += A[i][k] * B[k][j];
    return C;
}

Vec multiply(const Mat& A, const Vec& x) {
    if (A.empty() || A[0].size() != x.size())
        throw std::invalid_argument("multiply: dimensiones incompatibles");
    for (const auto& row : A)
        if (row.size() != A[0].size())
            throw std::invalid_argument("multiply: filas de distinto tamano");
    Vec y(A.size(), 0.0);
    for (size_t i = 0; i < A.size(); ++i)
        for (size_t j = 0; j < x.size(); ++j) y[i] += A[i][j] * x[j];
    return y;
}

Vec solve(Mat A, Vec b) {
    const size_t n = A.size();
    if (n == 0 || A[0].size() != n || b.size() != n)
        throw std::invalid_argument("solve: se requiere sistema cuadrado");
    for (const auto& row : A)
        if (row.size() != n) throw std::invalid_argument("solve: matriz no cuadrada");
    for (size_t col = 0; col < n; ++col) {
        // Pivoteo parcial: fila con mayor valor absoluto en la columna actual.
        size_t pivot = col;
        for (size_t r = col + 1; r < n; ++r)
            if (std::fabs(A[r][col]) > std::fabs(A[pivot][col])) pivot = r;
        if (std::fabs(A[pivot][col]) < 1e-12)
            throw std::runtime_error("solve: matriz singular (columnas linealmente dependientes)");
        std::swap(A[col], A[pivot]);
        std::swap(b[col], b[pivot]);
        // Eliminación hacia adelante.
        for (size_t r = col + 1; r < n; ++r) {
            double f = A[r][col] / A[col][col];
            for (size_t c = col; c < n; ++c) A[r][c] -= f * A[col][c];
            b[r] -= f * b[col];
        }
    }
    // Sustitución hacia atrás.
    Vec x(n, 0.0);
    for (size_t i = n; i-- > 0;) {
        double s = b[i];
        for (size_t j = i + 1; j < n; ++j) s -= A[i][j] * x[j];
        x[i] = s / A[i][i];
    }
    return x;
}

Mat addBiasColumn(const Mat& X) {
    if (X.empty()) throw std::invalid_argument("addBiasColumn: no hay datos");
    for (const auto& row : X)
        if (row.size() != X[0].size())
            throw std::invalid_argument("addBiasColumn: filas de distinto tamano");
    Mat R(X.size());
    for (size_t i = 0; i < X.size(); ++i) {
        R[i].reserve(X[i].size() + 1);
        R[i].push_back(1.0);
        R[i].insert(R[i].end(), X[i].begin(), X[i].end());
    }
    return R;
}

}  // namespace linalg

namespace stats {

double mean(const Vec& v) {
    if (v.empty()) throw std::invalid_argument("mean: vector vacío");
    double s = 0.0;
    for (double x : v) s += x;
    return s / static_cast<double>(v.size());
}

Vec columnMeans(const Mat& X) {
    if (X.empty() || X[0].empty())
        throw std::invalid_argument("columnMeans: no hay datos");
    for (const auto& row : X)
        if (row.size() != X[0].size())
            throw std::invalid_argument("columnMeans: filas de distinto tamano");
    Vec m(X[0].size(), 0.0);
    for (const auto& row : X)
        for (size_t j = 0; j < row.size(); ++j) m[j] += row[j];
    for (double& v : m) v /= static_cast<double>(X.size());
    return m;
}

Vec columnStds(const Mat& X, const Vec& means) {
    if (X.empty() || X[0].size() != means.size())
        throw std::invalid_argument("columnStds: dimensiones incorrectas");
    for (const auto& row : X)
        if (row.size() != means.size())
            throw std::invalid_argument("columnStds: filas de distinto tamano");
    Vec s(X[0].size(), 0.0);
    for (const auto& row : X)
        for (size_t j = 0; j < row.size(); ++j) s[j] += (row[j] - means[j]) * (row[j] - means[j]);
    for (double& v : s) {
        v = std::sqrt(v / static_cast<double>(X.size()));
        if (v < 1e-12) v = 1.0;
    }
    return s;
}

}  // namespace stats
