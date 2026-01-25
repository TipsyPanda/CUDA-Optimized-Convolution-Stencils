# Compilers
CXX = g++
NVCC = nvcc

# Flags
CXXFLAGS = -O2 -std=c++20 -Isrc
NVCCFLAGS = -O2 -std=c++17 -Xcompiler -Wall

# CUDA architecture (adjust for your GPU)
# Common values: 50 (Maxwell), 60 (Pascal), 70 (Volta), 75 (Turing), 80 (Ampere), 86 (Ampere), 89 (Ada), 90 (Hopper)
CUDA_ARCH ?= 75
NVCCFLAGS += -arch=sm_$(CUDA_ARCH)

# Source files
CPP_SRCS = src/cpu_convolution.cpp src/filters.cpp
CU_SRCS = src/cuda_convolution.cu
MAIN_SRC = src/main.cpp

# Object files
CPP_OBJS = $(CPP_SRCS:.cpp=.o)
CU_OBJS = $(CU_SRCS:.cu=.o)

# Target
TARGET = convolution_benchmark

.PHONY: all clean cpu_only

# Default target: build with CUDA
all: $(TARGET)

# Link everything together
$(TARGET): $(CPP_OBJS) $(CU_OBJS) $(MAIN_SRC)
	$(NVCC) $(NVCCFLAGS) -Isrc $^ -o $@

# Compile C++ sources
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile CUDA sources
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -Isrc -c $< -o $@

# CPU-only build (no CUDA required)
cpu_only: $(CPP_SRCS) $(MAIN_SRC)
	$(CXX) $(CXXFLAGS) -DCPU_ONLY $^ -o $(TARGET)_cpu

clean:
	rm -f $(CPP_OBJS) $(CU_OBJS) $(TARGET) $(TARGET)_cpu

# Show help
help:
	@echo "Targets:"
	@echo "  all      - Build with CUDA support (default)"
	@echo "  cpu_only - Build CPU-only version"
	@echo "  clean    - Remove build artifacts"
	@echo ""
	@echo "Variables:"
	@echo "  CUDA_ARCH - CUDA compute capability (default: 75)"
	@echo "              Example: make CUDA_ARCH=86"
