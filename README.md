# cnn
cnn is a GPU-based, vectorized convolutional neural net library built completely from scratch in cuda c. It has zero dependencies (other than [cuda](https://developer.nvidia.com/cuda-zone), of course) and is designed to be easy to understand.

## Usage
```
git clone https://github.com/jameseidson/cnn.git
```
Then add `-I./cnn/include` to your compile flags.

### Configuration and Initialization 
Network configuration is done via an external file, an example of which can be found at `examples/example.cfg`. Network layers and their parameters are parsed sequentially in the file, where the first layer defined represents the first layer of the network.

Training data must be allocated by the user as the `Data_T` type exported by `include/data.h`. This type contains various properties of the training session in addition to two single-pointer arrays- `imgs` and `lbls`. `imgs` is a 'flattened' 4D array whose dimensions correspond respectively to the index, height, width, and depth of a training image, and `lbls` represents the ground truth classification of the image at a particular index. If you need a little help flattening your 4D data, I don't blame you- feel free to use the `FLAT4D(i, j, k, l, max_j, max_k, max_l)` macro exported from `src/mat.h`. Note that a depth of 3 (typically `red, green, blue`) is the only supported image format at this time. Also note that these arrays do need to be allocated in cuda device memory (using `cudaMalloc`) which you can do yourself by manually allocating a `Data_T` (the struct itself is stored in host memory). Or, if you're unfamiliar with cuda, just allocate as you normally would and make me do the hard work using the `CNN_data_init` and `CNN_data_free` functions (also in `include/data.h`).

After that, you're good to go!

### Other Exported Functions
All exported functions not specified above can be found in `include/cnn.h`.

- `CNN_T *CNN_init(FILE *config, Data_T *data)` creates a new network based on the parameters specified in the configuration file and the provided training data. If the configuration file contains errors or an invalid specification, the program will halt and print a (somewhat) helpful message to `stderr`.

- `void CNN_train(CNN_T *cnn, Data_T *data)` trains the network. Note that the network can be trained on a different `Data_T` than the one provided on initialization, provided that its `hgt` and `wid` attributes are the same as the original.

- `void CNN_predict(CNN_T *cnn, double *image, double *output)` classifies the input passed through `image`- a 'flattened' 3D array allocated in device memory. The output layer of the network is copied to `output`, which must be host allocated. As before, check out the `FLAT3D(i, j, k, max_j, max_k)` macro.

- `void CNN_free(CNN_T *cnn)` is pretty self explanatory. Use it when you're done.

And that's it. In the future I plan to add support for data persistence by including functions to save/load training progress to/from binary files (as I did in my [ffnn library](https://github.com/jameseidson/ffnn)), but I'm ready to move on to other projects at this very moment so we'll see if I ever end up doing that...

### Things to Note
- I wrote this in my own time as a personal project to learn more about parallel GPU programming and neural networks. I have no formal training in Data Science, Machine Learning, or AI- every bit of knowledge that went into this project came from friends who share my interests, random lectures and articles I found online, and CS/calculus/linear algebra classes. What I'm trying to say is, this is probably not the fastest, most efficient, or fully featured library out there and I'm OK with that (if that's what you're looking for, I suggest checking out [darknet](https://github.com/pjreddie/darknet), or use a python library like a sane person). I certainly learned a lot, which is really all I hoped to achieve, and I'm proud that I actually managed to write a working program. I tried to keep things as understandable as possible so that others who want to learn as I did might find this and gain some insight. With that in mind, there will be bugs...

- You probably want to put a [`cudaDeviceSetLimit(cudaLimitMallocHeapSize, LARGE_NUMBER_HERE)`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g05956f16eaa47ef3a4efee84563ccb7d) in your code because this library isn't the most space-efficient thing in the world.

- I wanted as little dependencies and as much speed as possible so I didn't do any explicit error checking- most runtime errors are left as undefined behaviour. If you get undefined behaviour, the most probable cause is insufficient device memory. Try the previous tip, and if that doesn't work, try using less memory?

- Every feature in a convolutional layer acts on all previous features in the network- for example, if you have 3 convolutional layers with 4 features each, the output will consist of 4 * 4 * 4 images. Therefore, convolutional layers with a lot of features will use a lot of memory. Maybe keep this in mind when considering the previous tip.

- The output dimension of a convolutional layer is equal to `inputDim - featureDim + 1`, and the output dimension of a pooling layer is `((inputDim - windowDim) / stride) + 1`. This is important because it means that the input to a pooling layer **must** satisfy `(inputDim - windowDim) % stride == 0`. The configuration file parser will give you an error if this condition isn't met. If you are having trouble, I suggest going through your network specification and calculating the size of each layer's output and adjusting the parameters so that the condition is satisfied. Note that other layers do not change the size of the input.

## CIFAR Example
Check out `examples/` for an implementation using the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. `cifar.h` includes functions to read the CIFAR binary, extract it's contents into a `Data_T`, extract a single image as a `double *` for prediction, and other things such as printing an individual image as a [ppm](http://netpbm.sourceforge.net/doc/ppm.html) format (for us measly humans to read). `main.cu` contains a concrete example of library usage.
