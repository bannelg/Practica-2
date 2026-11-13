CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -pedantic
SRC = src/Matrix.cpp src/Dataset.cpp src/LinearRegression.cpp

all: regression memory_analysis

regression: $(SRC) src/main.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

memory_analysis: $(SRC) src/memory_analysis.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

run: all
	./regression data
	./memory_analysis

asan: $(SRC) src/memory_analysis.cpp
	$(CXX) $(CXXFLAGS) -g -fsanitize=address,undefined $^ -o memory_analysis_asan
	./memory_analysis_asan

clean:
	rm -f regression memory_analysis memory_analysis_asan
