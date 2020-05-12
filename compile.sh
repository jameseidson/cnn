#sm61 is supported by gtx1070
#sm35 is minimum requirement for dynamic parallelism- using for compatibility

case "$1" in
  "net")
    nvcc -I/opt/cuda/include/ -rdc=true -lcudadevrt -arch=sm_35 -o cnn net.cu main.cu cifar.cu
    ;;
  *)
    echo "Error: invalid argument"
    ;;
esac
