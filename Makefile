C_C = gcc
CU_C = nvcc

CU_CFLAGS = -g -O0 -arch=sm_35 -lineinfo -use_fast_math -rdc=true

IFLAGS = -I/opt/cuda/include/
LDFLAGS = -L/opt/cuda/lib64/
LDLIBS = -lm -lcuda -lcudart -lcudadevrt

all: cifar

cifar: main.o cifar.o cnn.o lex.o net.o def.o
	$(CU_C) -o $@ $^ $(CU_CFLAGS) $(LDFLAGS) $(IFLAGS) $(LDLIBS)

main.o: main.cu
	$(CU_C) -dc -o $@ $^ $(CU_CFLAGS) $(LDFLAGS) $(IFLAGS) $(LDLIBS)

cifar.o: cifar.cu
	$(CU_C) -dc -o $@ $^ $(CU_CFLAGS) $(LDFLAGS) $(IFLAGS) $(LDLIBS)

cnn.o: cnn.cu
	$(CU_C) -dc -o $@ $^ $(CU_CFLAGS) $(LDFLAGS) $(IFLAGS) $(LDLIBS)

net.o: net.cu
	$(CU_C) -dc -o $@ $^ $(CU_CFLAGS) $(LDFLAGS) $(IFLAGS) $(LDLIBS)

def.o: def.cu
	$(CU_C) -dc -o $@ $^ $(CU_CFLAGS) $(LDFLAGS) $(IFLAGS) $(LDLIBS)

lex.o: lex.cu
	$(CU_C) -dc -o $@ $^ $(CU_CFLAGS) $(LDFLAGS) $(IFLAGS) $(LDLIBS)

clean:
	rm -f cifar *.o
