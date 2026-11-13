// Regresión lineal simple y múltiple.
#pragma once
#include <iosfwd>

#include "Matrix.hpp"

class LinearRegression {
public:
    enum class Solver { NormalEquation, GradientDescent };

    // Métricas de error devueltas por score().
    struct Metrics {
        double mse;   // error cuadrático medio
        double rmse;  // raíz del MSE
        double mae;   // error absoluto medio
        double r2;    // coeficiente de determinación
    };

    // useScaling: estandariza X (z-score) antes de entrenar o predecir.
    // learningRate/epochs solo se usan con Solver::GradientDescent.
    explicit LinearRegression(bool useScaling = true, Solver solver = Solver::NormalEquation,
                              double learningRate = 0.05, int epochs = 5000);

    // Entrena el modelo con X (n x p) y y (n).
    void fit(const Mat& X, const Vec& y);
    // Predice y_hat para cada fila de X.
    Vec predict(const Mat& X) const;
    // Calcula MSE, RMSE, MAE y R2 de las predicciones sobre (X, y).
    Metrics score(const Mat& X, const Vec& y) const;
    // Escala el dataset (z-score). Con refit=true recalcula y guarda media/desviación de X.
    Mat data_scaling(const Mat& X, bool refit = false);

    // Pesos y sesgo en el espacio con el que se entrenó (escalado si useScaling=true).
    const Vec& getWeights() const { return weights_; }
    double getBias() const { return bias_; }
    // Pesos y sesgo expresados en las unidades originales de los datos.
    Vec getWeightsOriginalScale() const;
    double getBiasOriginalScale() const;
    bool isFitted() const { return fitted_; }

    // Imprime sizeof y offset de cada atributo (para el análisis de memoria).
    void printLayout(std::ostream& os) const;

private:
    Mat applyScaling(const Mat& X) const;  // aplica (x - media)/desv sin modificar el estado
    void fitNormalEquation(const Mat& Xs, const Vec& y);
    void fitGradientDescent(const Mat& Xs, const Vec& y);

    Vec weights_;        // w1..wp
    double bias_;        // b
    Vec mean_;           // media por columna (escalado)
    Vec std_;            // desviación por columna (escalado)
    double learningRate_;
    int epochs_;
    Solver solver_;
    bool scaling_;
    bool fitted_;
};

std::ostream& operator<<(std::ostream& os, const LinearRegression::Metrics& m);
