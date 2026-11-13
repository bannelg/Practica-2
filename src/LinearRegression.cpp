#include "LinearRegression.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

LinearRegression::LinearRegression(bool useScaling, Solver solver, double learningRate, int epochs)
    : bias_(0.0), learningRate_(learningRate), epochs_(epochs), solver_(solver),
      scaling_(useScaling), fitted_(false) {
    if (solver_ == Solver::GradientDescent &&
        (!std::isfinite(learningRate_) || learningRate_ <= 0 || epochs_ <= 0)) {
        throw std::invalid_argument("La tasa de aprendizaje y las epocas deben ser positivas");
    }
}

Mat LinearRegression::applyScaling(const Mat& X) const {
    if (!scaling_) return X;
    if (mean_.empty()) throw std::logic_error("Primero hay que ajustar el escalamiento");

    Mat S = X;
    for (auto& row : S) {
        if (row.size() != mean_.size())
            throw std::invalid_argument("Numero de columnas diferente");
        for (size_t j = 0; j < row.size(); ++j) {
            row[j] = (row[j] - mean_[j]) / std_[j];
        }
    }
    return S;
}

Mat LinearRegression::data_scaling(const Mat& X, bool refit) {
    if (X.empty() || X[0].empty())
        throw std::invalid_argument("No hay datos para escalar");
    for (const auto& row : X) {
        if (row.size() != X[0].size())
            throw std::invalid_argument("Hay filas con distinto numero de columnas");
    }
    if (refit && scaling_) {
        mean_ = stats::columnMeans(X);
        std_ = stats::columnStds(X, mean_);
    }
    return applyScaling(X);
}

void LinearRegression::fit(const Mat& X, const Vec& y) {
    fitted_ = false;
    if (X.empty() || X[0].empty() || X.size() != y.size())
        throw std::invalid_argument("X e y deben tener filas y al menos una columna");

    for (size_t i = 0; i < X.size(); ++i) {
        if (X[i].size() != X[0].size())
            throw std::invalid_argument("Hay filas con distinto numero de columnas");
        if (!std::isfinite(y[i]))
            throw std::invalid_argument("La variable de salida contiene valores invalidos");
        for (double x : X[i]) {
            if (!std::isfinite(x))
                throw std::invalid_argument("Los datos contienen valores invalidos");
        }
    }

    Mat Xs = scaling_ ? data_scaling(X, true) : X;
    if (solver_ == Solver::NormalEquation) fitNormalEquation(Xs, y);
    else fitGradientDescent(Xs, y);
    fitted_ = true;
}

// Resuelve las ecuaciones normales: (A transpuesta * A) * theta = A transpuesta * y.
void LinearRegression::fitNormalEquation(const Mat& Xs, const Vec& y) {
    Mat A = linalg::addBiasColumn(Xs);
    Mat At = linalg::transpose(A);
    Vec theta = linalg::solve(linalg::multiply(At, A), linalg::multiply(At, y));
    bias_ = theta[0];
    weights_.assign(theta.begin() + 1, theta.end());
}

// Aprende los pesos dando muchos pasos pequenos para disminuir el error.
void LinearRegression::fitGradientDescent(const Mat& Xs, const Vec& y) {
    const size_t n = Xs.size(), p = Xs[0].size();
    weights_.assign(p, 0.0);
    bias_ = 0.0;
    for (int e = 0; e < epochs_; ++e) {
        Vec gradW(p, 0.0);
        double gradB = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double pred = bias_;
            for (size_t j = 0; j < p; ++j) pred += weights_[j] * Xs[i][j];
            double err = pred - y[i];
            for (size_t j = 0; j < p; ++j) gradW[j] += err * Xs[i][j];
            gradB += err;
        }
        for (size_t j = 0; j < p; ++j) {
            weights_[j] -= learningRate_ * 2.0 * gradW[j] / n;
            if (!std::isfinite(weights_[j]))
                throw std::runtime_error("El entrenamiento no es estable: pruebe escalar los datos o bajar la tasa");
        }
        bias_ -= learningRate_ * 2.0 * gradB / n;
        if (!std::isfinite(bias_))
            throw std::runtime_error("El entrenamiento no es estable: pruebe escalar los datos o bajar la tasa");
    }
}

Vec LinearRegression::predict(const Mat& X) const {
    if (!fitted_) throw std::logic_error("Primero hay que entrenar el modelo");
    if (X.empty()) throw std::invalid_argument("No hay datos para predecir");
    Mat Xs = applyScaling(X);
    Vec yhat(Xs.size(), bias_);
    for (size_t i = 0; i < Xs.size(); ++i) {
        if (Xs[i].size() != weights_.size())
            throw std::invalid_argument("Numero de columnas diferente");
        for (size_t j = 0; j < weights_.size(); ++j) {
            if (!std::isfinite(Xs[i][j]))
                throw std::invalid_argument("La entrada contiene valores invalidos");
            yhat[i] += weights_[j] * Xs[i][j];
        }
        if (!std::isfinite(yhat[i]))
            throw std::runtime_error("La prediccion no produjo un numero valido");
    }
    return yhat;
}

LinearRegression::Metrics LinearRegression::score(const Mat& X, const Vec& y) const {
    if (y.empty() || X.size() != y.size())
        throw std::invalid_argument("score: X e y deben tener la misma cantidad de filas");
    for (double value : y) {
        if (!std::isfinite(value))
            throw std::invalid_argument("score: hay valores invalidos en y");
    }
    Vec yhat = predict(X);
    const double yMean = stats::mean(y);
    double sse = 0.0, sae = 0.0, sst = 0.0;
    for (size_t i = 0; i < y.size(); ++i) {
        double e = yhat[i] - y[i];
        sse += e * e;
        sae += std::fabs(e);
        sst += (y[i] - yMean) * (y[i] - yMean);
    }
    const double n = static_cast<double>(y.size());
    Metrics m;
    m.mse = sse / n;
    m.rmse = std::sqrt(m.mse);
    m.mae = sae / n;
    // Cuando todos los valores reales son iguales, hay que tratar el R2 aparte.
    m.r2 = (sst > 1e-12) ? 1.0 - sse / sst : (sse < 1e-12 ? 1.0 : 0.0);
    return m;
}

Vec LinearRegression::getWeightsOriginalScale() const {
    if (!fitted_) throw std::logic_error("Primero hay que entrenar el modelo");
    if (!scaling_) return weights_;
    Vec w(weights_.size());
    for (size_t j = 0; j < w.size(); ++j) w[j] = weights_[j] / std_[j];
    return w;
}

double LinearRegression::getBiasOriginalScale() const {
    if (!fitted_) throw std::logic_error("Primero hay que entrenar el modelo");
    if (!scaling_) return bias_;
    double b = bias_;
    for (size_t j = 0; j < weights_.size(); ++j) b -= weights_[j] * mean_[j] / std_[j];
    return b;
}

void LinearRegression::printLayout(std::ostream& os) const {
    const char* base = reinterpret_cast<const char*>(this);
    auto off = [&](const void* member) { return reinterpret_cast<const char*>(member) - base; };
    os << "  sizeof(LinearRegression) = " << sizeof(*this) << " bytes\n"
       << "  offset " << std::setw(3) << off(&weights_) << " | weights_       vector<double> (" << sizeof(weights_) << " B)\n"
       << "  offset " << std::setw(3) << off(&bias_) << " | bias_          double (" << sizeof(bias_) << " B)\n"
       << "  offset " << std::setw(3) << off(&mean_) << " | mean_          vector<double> (" << sizeof(mean_) << " B)\n"
       << "  offset " << std::setw(3) << off(&std_) << " | std_           vector<double> (" << sizeof(std_) << " B)\n"
       << "  offset " << std::setw(3) << off(&learningRate_) << " | learningRate_  double (" << sizeof(learningRate_) << " B)\n"
       << "  offset " << std::setw(3) << off(&epochs_) << " | epochs_        int (" << sizeof(epochs_) << " B)\n"
       << "  offset " << std::setw(3) << off(&solver_) << " | solver_        enum (" << sizeof(solver_) << " B)\n"
       << "  offset " << std::setw(3) << off(&scaling_) << " | scaling_       bool (" << sizeof(scaling_) << " B)\n"
       << "  offset " << std::setw(3) << off(&fitted_) << " | fitted_        bool (" << sizeof(fitted_) << " B)\n";
}

std::ostream& operator<<(std::ostream& os, const LinearRegression::Metrics& m) {
    os << "MSE=" << m.mse << "  RMSE=" << m.rmse << "  MAE=" << m.mae << "  R2=" << m.r2;
    return os;
}
