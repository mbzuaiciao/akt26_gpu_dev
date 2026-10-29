# Build with the CUDA toolkit; override sm for the assigned GPU.
CUDADIR ?= /usr/local/cuda

# gpu architecture
# Ampere by default, with forward compatibility
sm ?= 80
NVCCFLAGS ?= -O3 -std=c++11 -lineinfo
GENCODE_SM   = -gencode arch=compute_$(sm),code=sm_$(sm)
GENCODE_COMP = -gencode arch=compute_$(sm),code=compute_$(sm)

LIB=-L${CUDADIR}/lib64 -lcublas

all:
	nvcc ${NVCCFLAGS} ${GENCODE_SM} ${GENCODE_COMP} ger.cu -o ger ${LIB}

clean:
	rm -rf ger

