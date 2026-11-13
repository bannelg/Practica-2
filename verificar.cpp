
#include <iostream>
#include <vector>

enum class Solver {
    SGD,
    ADAM
};

class TuClase {
public:
    std::vector<double> weights_;
    double bias_;
    std::vector<double> mean_;
    std::vector<double> std_;
    double learningRate_;
    int epochs_;
    Solver solver_;
    bool scaling_;
    bool fitted_;
};

int main() {
    std::cout << "vector: " << sizeof(std::vector<double>) << " bytes\n";
    std::cout << "double: " << sizeof(double) << " bytes\n";
    std::cout << "int: " << sizeof(int) << " bytes\n";
    std::cout << "Solver: " << sizeof(Solver) << " bytes\n";
    std::cout << "bool: " << sizeof(bool) << " bytes\n";
    std::cout << "Clase: " << sizeof(TuClase) << " bytes\n";
}
