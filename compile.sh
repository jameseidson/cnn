#sm61 is supported by gtx1070
#sm35 is minimum requirement for dynamic parallelism- using for compatibility

case "$1" in
  "cifar")
    nvcc -I/opt/cuda/include/ -rdc=true -lcudadevrt -arch=sm_35 -o cifar def.cu net.cu cifar.cu cnn.cu main.cu 
    ;;
  *)
    echo "Error: invalid argument"
    ;;
esac
