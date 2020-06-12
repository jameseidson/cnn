CU_C = nvcc

CU_CFLAGS = -g -O0 -arch=sm_35 -lineinfo -use_fast_math -rdc=true

IFLAGS = -I/opt/cuda/include/ -I../src/
LDFLAGS = -L/opt/cuda/lib64/
LDLIBS = -lm -lcuda -lcudart -lcudadevrt

all: cifar

cifar: main.o cifar.o mat.o cnn.o lex.o net.o def.o
	$(CU_C) -o $@ $^ $(CU_CFLAGS) $(LDFLAGS) $(IFLAGS) $(LDLIBS)

main.o: main.cu
	$(CU_C) -dc -o $@ $^ $(CU_CFLAGS) $(LDFLAGS) $(IFLAGS) $(LDLIBS)

cifar.o: cifar.cu
	$(CU_C) -dc -o $@ $^ $(CU_CFLAGS) $(LDFLAGS) $(IFLAGS) $(LDLIBS)

cnn.o: ../src/cnn.cu
	$(CU_C) -dc -o $@ $^ $(CU_CFLAGS) $(LDFLAGS) $(IFLAGS) $(LDLIBS)

mat.o: ../src/mat.cu
	$(CU_C) -dc -o $@ $^ $(CU_CFLAGS) $(LDFLAGS) $(IFLAGS) $(LDLIBS)

net.o: ../src/net.cu
	$(CU_C) -dc -o $@ $^ $(CU_CFLAGS) $(LDFLAGS) $(IFLAGS) $(LDLIBS)

def.o: ../src/def.cu
	$(CU_C) -dc -o $@ $^ $(CU_CFLAGS) $(LDFLAGS) $(IFLAGS) $(LDLIBS)

lex.o: ../src/lex.cu
	$(CU_C) -dc -o $@ $^ $(CU_CFLAGS) $(LDFLAGS) $(IFLAGS) $(LDLIBS)

clean:
	rm -f cifar *.o
