// Pruebas pequenas para comprobar que cada parte funciona.
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "Dataset.hpp"
#include "LinearRegression.hpp"

static int pruebas = 0;
static int fallas = 0;

static void comprobar(bool condicion, const char* nombre) {
    ++pruebas;
    if (condicion) std::cout << "OK: " << nombre << '\n';
    else {
        ++fallas;
        std::cout << "FALLO: " << nombre << '\n';
    }
}

static bool cerca(double a, double b, double tolerancia = 0.0001) {
    return std::fabs(a - b) < tolerancia;
}

int main() {
    try {
        // Ejemplo donde conocemos la respuesta: y = 2x + 1.
        Mat X = {{0}, {1}, {2}, {3}, {4}};
        Vec y = {1, 3, 5, 7, 9};
        LinearRegression simple;
        simple.fit(X, y);
        comprobar(cerca(simple.getWeightsOriginalScale()[0], 2), "Peso regresion simple");
        comprobar(cerca(simple.getBiasOriginalScale(), 1), "Bias regresion simple");
        comprobar(cerca(simple.predict({{5}})[0], 11), "Prediccion simple");
        comprobar(cerca(simple.score(X, y).r2, 1), "R2 regresion simple perfecta");
        comprobar(cerca(simple.score(X, y).mse, 0), "MSE regresion simple perfecta");

        // Ejemplo con dos variables: y = 4 + 2*x1 - 3*x2.
        Mat X2 = {{0, 0}, {1, 0}, {0, 1}, {1, 1}, {2, 1}, {1, 2}};
        Vec y2 = {4, 6, 1, 3, 5, 0};
        LinearRegression multiple;
        multiple.fit(X2, y2);
        comprobar(cerca(multiple.getWeightsOriginalScale()[0], 2), "Primer peso multiple");
        comprobar(cerca(multiple.getWeightsOriginalScale()[1], -3), "Segundo peso multiple");
        comprobar(cerca(multiple.getBiasOriginalScale(), 4), "Bias multiple");
        comprobar(cerca(multiple.predict({{2, 2}})[0], 2), "Prediccion multiple");

        // El otro metodo de entrenamiento debe llegar a una respuesta parecida.
        LinearRegression gradiente(true, LinearRegression::Solver::GradientDescent, 0.05, 5000);
        gradiente.fit(X2, y2);
        comprobar(cerca(gradiente.predict({{2, 2}})[0], 2), "Descenso del gradiente");

        // Si todas las respuestas son iguales, el R2 perfecto es 1.
        Vec iguales = {5, 5, 5, 5, 5};
        LinearRegression constante(true, LinearRegression::Solver::GradientDescent, 0.05, 5000);
        constante.fit(X, iguales);
        comprobar(cerca(constante.score(X, iguales).r2, 1), "R2 con respuestas constantes");

        bool error = false;
        try { simple.predict({{1, 2}}); }
        catch (const std::invalid_argument&) { error = true; }
        comprobar(error, "Detecta columnas incorrectas");

        error = false;
        try { simple.fit({}, {}); }
        catch (const std::invalid_argument&) { error = true; }
        comprobar(error, "Detecta entrenamiento vacio");

        error = false;
        try { LinearRegression sinEntrenar; sinEntrenar.predict({{1}}); }
        catch (const std::logic_error&) { error = true; }
        comprobar(error, "Detecta modelo sin entrenar");

        Dataset datos, entrenamiento, prueba;
        datos.X = X;
        datos.y = y;
        trainTestSplit(datos, 0.8, 42, entrenamiento, prueba);
        comprobar(entrenamiento.X.size() == 4 && prueba.X.size() == 1,
                  "Division de entrenamiento y prueba");

        error = false;
        try { trainTestSplit(datos, 1.0, 42, entrenamiento, prueba); }
        catch (const std::invalid_argument&) { error = true; }
        comprobar(error, "Detecta proporcion incorrecta");

        error = false;
        try {
            LinearRegression inestable(false, LinearRegression::Solver::GradientDescent, 0.05, 500);
            inestable.fit({{100}, {200}, {300}}, {1, 2, 3});
        } catch (const std::runtime_error&) { error = true; }
        comprobar(error, "Detecta entrenamiento inestable sin mostrar NaN");

        // Finalmente cargamos los archivos entregados por el profesor.
        Dataset helados = loadCsv("data/Ice_cream_selling_data.csv");
        Dataset estudiantes = loadCsv("data/student_exam_scores.csv");
        comprobar(helados.X.size() == 49 && helados.featureNames.size() == 1,
                  "Lectura del CSV de helados");
        comprobar(estudiantes.X.size() == 200 && estudiantes.featureNames.size() == 4,
                  "Lectura del CSV de estudiantes");

        // Un texto mezclado con numeros no debe pasar como dato correcto.
        {
            std::ofstream archivo("prueba_csv_temporal.csv");
            archivo << "x,y\n1abc,2\n";
        }
        error = false;
        try { loadCsv("prueba_csv_temporal.csv"); }
        catch (const std::runtime_error&) { error = true; }
        std::remove("prueba_csv_temporal.csv");
        comprobar(error, "Detecta un CSV con numeros invalidos");

        error = false;
        try { trainTestSplit(datos, 0.01, 42, entrenamiento, prueba); }
        catch (const std::invalid_argument&) { error = true; }
        comprobar(error, "No permite un conjunto de entrenamiento vacio");

        error = false;
        try { linalg::multiply(Mat{{1, 2}, {3}}, Mat{{1}, {2}}); }
        catch (const std::invalid_argument&) { error = true; }
        comprobar(error, "Detecta matrices con filas incompletas");
    } catch (const std::exception& e) {
        std::cerr << "Error inesperado en pruebas: " << e.what() << '\n';
        return 1;
    }
    std::cout << "\nResultado: " << pruebas - fallas << " de " << pruebas << " pruebas correctas.\n";
    return fallas == 0 ? 0 : 1;
}
