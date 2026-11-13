// Direcciones y tamanos de objetos en diferentes zonas de memoria.
#include <cstdio>
#include <iostream>
#include <vector>

#include "LinearRegression.hpp"

LinearRegression globalModel;
static LinearRegression staticModel;

static void mostrar(const char* nombre, const void* direccion, size_t tamano) {
    std::printf("  %-28s direccion: %p   sizeof: %zu bytes\n", nombre, direccion, tamano);
}

static void entrenar(LinearRegression& modelo) {
    Mat X = {{1, 2}, {2, 1}, {3, 5}, {4, 3}, {5, 8}};
    Vec y = {5, 6, 14, 13, 21};
    modelo.fit(X, y);
}

int main() {
    std::puts("--- Objetos globales, locales y dinamicos ---");
    LinearRegression localModel;
    LinearRegression* dynamicModel = new LinearRegression();

    mostrar("Objeto global", &globalModel, sizeof(globalModel));
    mostrar("Objeto global estatico", &staticModel, sizeof(staticModel));
    mostrar("Objeto local", &localModel, sizeof(localModel));
    mostrar("Objeto dinamico", dynamicModel, sizeof(*dynamicModel));
    mostrar("Variable puntero", &dynamicModel, sizeof(dynamicModel));

    std::puts("\n--- Por que el objeto tiene ese tamano ---");
    localModel.printLayout(std::cout);
    size_t suma = 3 * sizeof(Vec) + 2 * sizeof(double) +
                  sizeof(int) + sizeof(LinearRegression::Solver) + 2 * sizeof(bool);
    std::printf("  Suma de atributos: %zu bytes\n", suma);
    std::printf("  Tamano total: %zu bytes (incluye %zu bytes de relleno/alineacion)\n",
                sizeof(localModel), sizeof(localModel) - suma);
    std::puts("  El numero de bytes puede variar entre computadores y compiladores.");

    std::puts("\n--- Los pesos se guardan aparte del objeto ---");
    entrenar(globalModel);
    entrenar(localModel);
    entrenar(*dynamicModel);
    std::printf("  sizeof despues de entrenar: %zu bytes\n", sizeof(localModel));
    std::printf("  Datos de los pesos: %zu elementos x %zu bytes = %zu bytes\n",
                localModel.getWeights().size(), sizeof(double),
                localModel.getWeights().size() * sizeof(double));
    mostrar("Direccion de pesos", localModel.getWeights().data(),
            localModel.getWeights().size() * sizeof(double));

    std::puts("\n--- Arreglo dinamico de cinco objetos ---");
    const size_t N = 5;
    LinearRegression* arr = new LinearRegression[N];
    std::printf("  Direccion del arreglo: %p\n", static_cast<void*>(arr));
    std::printf("  Tamano de cada objeto: %zu bytes\n", sizeof(*arr));
    std::printf("  Tamano de los cinco objetos: %zu bytes\n", N * sizeof(*arr));
    std::printf("  Tamano de arr (puntero): %zu bytes\n", sizeof(arr));
    for (size_t i = 0; i < N; ++i)
        std::printf("  arr[%zu]: %p\n", i, static_cast<void*>(&arr[i]));
    std::puts("  El gestor de memoria puede reservar bytes adicionales, pero ese dato no es fijo.");

    std::puts("\n--- Vector con elementos en memoria dinamica ---");
    std::vector<LinearRegression> modelos(N);
    std::printf("  sizeof(vector): %zu bytes (estructura del vector)\n", sizeof(modelos));
    std::printf("  size: %zu; capacity: %zu\n", modelos.size(), modelos.capacity());
    std::printf("  Espacio reservado para elementos: %zu bytes\n",
                modelos.capacity() * sizeof(LinearRegression));
    std::printf("  Direccion de los elementos: %p\n", static_cast<void*>(modelos.data()));
    std::puts("  El vector libera sus datos automaticamente al salir de main.");

    delete dynamicModel;
    delete[] arr;
    std::puts("\nMemoria de new y new[] liberada con delete y delete[].");
    return 0;
}
