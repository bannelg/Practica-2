// Pruebas de regresión lineal simple (helados) y múltiple.
#include <iomanip>
#include <iostream>

#include "Dataset.hpp"
#include "LinearRegression.hpp"

static void printVec(const std::string& label, const Vec& v) {
    std::cout << label << " [";
    for (size_t i = 0; i < v.size(); ++i) std::cout << (i ? ", " : "") << v[i];
    std::cout << "]\n";
}

// Entrena con un solver dado, imprime pesos, sesgo, predicciones y score sobre prueba.
static void runModel(const std::string& title, LinearRegression& mlr, const Dataset& train,
                     const Dataset& test, bool listPredictions) {
    std::cout << "\n--- " << title << " ---\n";
    mlr.fit(train.X, train.y);
    printVec("weights (espacio de entrenamiento):", mlr.getWeights());
    std::cout << "bias (espacio de entrenamiento): " << mlr.getBias() << "\n";
    printVec("weights (unidades originales):     ", mlr.getWeightsOriginalScale());
    std::cout << "bias (unidades originales):      " << mlr.getBiasOriginalScale() << "\n";

    Vec yhat = mlr.predict(test.X);
    if (listPredictions) {
        std::cout << std::setw(4) << "i" << std::setw(14) << "y_test" << std::setw(14) << "y_hat" << "\n";
        for (size_t i = 0; i < yhat.size(); ++i)
            std::cout << std::setw(4) << i << std::setw(14) << test.y[i] << std::setw(14) << yhat[i] << "\n";
    }
    std::cout << "score (train): " << mlr.score(train.X, train.y) << "\n";
    std::cout << "score (test):  " << mlr.score(test.X, test.y) << "\n";
}

static void experiment(const std::string& name, const std::string& path) {
    std::cout << "\n-----------------------------------------------------\n"
              << name << "  (" << path << ")\n"
              << "-----------------------------------------------------\n";
    Dataset all = loadCsv(path), train, test;
    std::cout << "Filas: " << all.X.size() << "  |  Características: " << all.featureNames.size()
              << "  |  Objetivo: " << all.targetName << "\n";
    trainTestSplit(all, 0.8, 42, train, test);
    std::cout << "Entrenamiento: " << train.X.size() << " filas  |  Prueba: " << test.X.size() << " filas\n";

    // 1) Ecuación normal con escalado.
    LinearRegression ne(true, LinearRegression::Solver::NormalEquation);
    runModel("Ecuación normal + data_scaling", ne, train, test, true);

    // 2) Descenso del gradiente con escalado.
    LinearRegression gd(true, LinearRegression::Solver::GradientDescent, 0.05, 5000);
    runModel("Descenso del gradiente + data_scaling (lr=0.05, 5000 epochs)", gd, train, test, false);

    // 3) Descenso del gradiente SIN escalado, mismo learning rate: muestra por qué importa escalar.
    LinearRegression gdRaw(false, LinearRegression::Solver::GradientDescent, 0.05, 200);
    try {
        runModel("Descenso del gradiente SIN escalado (lr=0.05, 200 epochs)", gdRaw, train, test, false);
    } catch (const std::runtime_error& e) {
        std::cout << "Nota: el modelo sin escalamiento no pudo entrenarse. " << e.what() << "\n";
    }

    // 4) Verificación: ajuste con TODOS los datos (comparable con numpy/sklearn).
    LinearRegression full(true, LinearRegression::Solver::NormalEquation);
    full.fit(all.X, all.y);
    std::cout << "\n--- Verificación: ecuación normal con el 100% de los datos ---\n";
    printVec("weights (unidades originales):", full.getWeightsOriginalScale());
    std::cout << "bias (unidades originales):   " << std::setprecision(10) << full.getBiasOriginalScale() << "\n";
    std::cout << std::setprecision(6);
}


// Extra: los datos de helados forman una parábola; con la característica adicional x^2 el MISMO modelo múltiple ajusta bien.
static void bonusPolynomial(const std::string& path) {
    std::cout << "\n--------------------------------------------------\n"
              << "EXTRA: helados con características [x, x^2] (regresión múltiple sobre datos no lineales)\n"
              << "--------------------------------------------------\n";
    Dataset all = loadCsv(path), train, test;
    for (auto& row : all.X) row.push_back(row[0] * row[0]);
    trainTestSplit(all, 0.8, 42, train, test);
    LinearRegression ne(true, LinearRegression::Solver::NormalEquation);
    runModel("Ecuación normal + data_scaling con [x, x^2]", ne, train, test, false);
}

int main(int argc, char** argv) {
    std::string dir = (argc > 1) ? argv[1] : "data";
    std::cout << std::fixed << std::setprecision(6);
    try {
        experiment("REGRESION LINEAL SIMPLE", dir + "/Ice_cream_selling_data.csv");
        experiment("REGRESION LINEAL MULTIPLE", dir + "/student_exam_scores.csv");
        bonusPolynomial(dir + "/Ice_cream_selling_data.csv");
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
