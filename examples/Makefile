CU_C = nvcc

CU_CFLAGS = -g -O0 -arch=sm_35 -lineinfo -use_fast_math -rdc=true

IFLAGS = -I/opt/cuda/include/ -I../src/
LDFLAGS = -L/opt/cuda/lib64/
LDLIBS = -lm -lcuda -lcudart -lcudadevrt

all: cifar

cifar: main.o cifar.o data.o cnn.o config.o layer.o mat.o
	$(CU_C) -o $@ $^ $(CU_CFLAGS) $(LDFLAGS) $(IFLAGS) $(LDLIBS)

main.o: main.cu
	$(CU_C) -dc -o $@ $^ $(CU_CFLAGS) $(LDFLAGS) $(IFLAGS) $(LDLIBS)

cifar.o: cifar.cu
	$(CU_C) -dc -o $@ $^ $(CU_CFLAGS) $(LDFLAGS) $(IFLAGS) $(LDLIBS)

data.o: ../src/data.cu
	$(CU_C) -dc -o $@ $^ $(CU_CFLAGS) $(LDFLAGS) $(IFLAGS) $(LDLIBS)

cnn.o: ../src/cnn.cu
	$(CU_C) -dc -o $@ $^ $(CU_CFLAGS) $(LDFLAGS) $(IFLAGS) $(LDLIBS)

config.o: ../src/config.cu
	$(CU_C) -dc -o $@ $^ $(CU_CFLAGS) $(LDFLAGS) $(IFLAGS) $(LDLIBS)

layer.o: ../src/layer.cu
	$(CU_C) -dc -o $@ $^ $(CU_CFLAGS) $(LDFLAGS) $(IFLAGS) $(LDLIBS)

mat.o: ../src/mat.cu
	$(CU_C) -dc -o $@ $^ $(CU_CFLAGS) $(LDFLAGS) $(IFLAGS) $(LDLIBS)

clean:
	rm -f cifar *.o
