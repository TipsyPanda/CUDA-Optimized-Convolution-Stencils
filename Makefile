CXX = g++
CXXFLAGS = -O2 -std=c++20 -Isrc

all: main

main: src/main.cpp src/cpu_convolution.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	rm -f main
