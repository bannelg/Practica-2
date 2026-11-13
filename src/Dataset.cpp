#include "Dataset.hpp"

#include <cmath>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

// Divide una linea usando las comas del CSV.
static std::vector<std::string> splitLine(const std::string& line) {
    std::vector<std::string> out;
    std::string cell;
    std::stringstream ss(line);
    while (std::getline(ss, cell, ',')) {
        if (!cell.empty() && cell.back() == '\r') cell.pop_back();
        out.push_back(cell);
    }
    // getline no devuelve la ultima celda si la linea termina en coma.
    if (!line.empty() && line.back() == ',') out.push_back("");
    return out;
}

// Convierte una celda a numero y revisa que no contenga textos o valores vacios.
static double readNumber(const std::string& text, size_t lineNumber) {
    size_t pos = 0;
    double value;
    try {
        value = std::stod(text, &pos);
    } catch (const std::exception&) {
        throw std::runtime_error("Dato no numerico en la fila " + std::to_string(lineNumber));
    }
    if (text.find_first_not_of(" \t\r", pos) != std::string::npos || !std::isfinite(value))
        throw std::runtime_error("Dato invalido en la fila " + std::to_string(lineNumber));
    return value;
}

Dataset loadCsv(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("No se pudo abrir: " + path);

    std::string line;
    if (!std::getline(in, line)) throw std::runtime_error("El CSV esta vacio: " + path);
    auto header = splitLine(line);
    if (header.size() < 2) throw std::runtime_error("El CSV necesita al menos 2 columnas");

    Dataset d;
    d.targetName = header.back();
    d.featureNames.assign(header.begin(), header.end() - 1);
    size_t lineNumber = 1;
    while (std::getline(in, line)) {
        ++lineNumber;
        if (line.empty() || line == "\r") continue;
        auto cells = splitLine(line);
        if (cells.size() != header.size())
            throw std::runtime_error("Cantidad incorrecta de columnas en la fila " + std::to_string(lineNumber));
        Vec row;
        for (size_t j = 0; j + 1 < cells.size(); ++j)
            row.push_back(readNumber(cells[j], lineNumber));
        d.X.push_back(row);
        d.y.push_back(readNumber(cells.back(), lineNumber));
    }
    if (d.X.empty()) throw std::runtime_error("El CSV no tiene filas con datos");
    return d;
}

void trainTestSplit(const Dataset& all, double trainRatio, unsigned seed, Dataset& train, Dataset& test) {
    const size_t n = all.X.size();
    if (!std::isfinite(trainRatio) || trainRatio <= 0 || trainRatio >= 1)
        throw std::invalid_argument("La proporcion de entrenamiento debe estar entre 0 y 1");
    if (n < 2 || n != all.y.size())
        throw std::invalid_argument("Se necesitan al menos dos filas y X e y del mismo tamano");

    const size_t nTrain = static_cast<size_t>(n * trainRatio);
    if (nTrain == 0 || nTrain == n)
        throw std::invalid_argument("La division deja entrenamiento o prueba sin filas");

    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 gen(seed);
    // Mezclamos las posiciones para usar la misma division al repetir el experimento.
    for (size_t i = n - 1; i > 0; --i) {
        size_t j = gen() % (i + 1);
        std::swap(idx[i], idx[j]);
    }

    train = test = Dataset();
    train.featureNames = test.featureNames = all.featureNames;
    train.targetName = test.targetName = all.targetName;
    for (size_t k = 0; k < n; ++k) {
        Dataset& dst = (k < nTrain) ? train : test;
        dst.X.push_back(all.X[idx[k]]);
        dst.y.push_back(all.y[idx[k]]);
    }
}
