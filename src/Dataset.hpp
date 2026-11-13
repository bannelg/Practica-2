//Lectura de cvs
#pragma once
#include <string>
#include <vector>

#include "Matrix.hpp"

struct Dataset {
    std::vector<std::string> featureNames;  // nombres de columnas de entrada
    std::string targetName;                 // nombre de la columna objetivo
    Mat X;                                  // características (n x p)
    Vec y;                                  // objetivo (n)
};

// Carga un CSV con encabezado; la última columna es la variable objetivo 
Dataset loadCsv(const std::string& path);

// Divide en entrenamiento y prueba con barajado reproducible.
void trainTestSplit(const Dataset& all, double trainRatio, unsigned seed, Dataset& train, Dataset& test);
