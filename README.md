# CNTK MNIST
For details, see [CNTK Tutorials](https://notebooks.azure.com/cntk/libraries/tutorials).

## Setup

### Download MNIST DataSet
``` console
$ docker run --rm -it -v $(pwd):/work -w /work microsoft/cntk:2.0-cpu-python3.5 bash -c "source /cntk/activate-cntk && python data_loader.py"
```

## Run with CPU
``` console
$ docker run --rm -it -v $(pwd):/work -w /work microsoft/cntk:2.0-cpu-python3.5 bash -c "source /cntk/activate-cntk && python ConvNet_MNIST.py"
```

## Run with GPU
``` console
$ nvidia-docker run --rm -it -v $(pwd):/work -w /work microsoft/cntk bash -c "source /cntk/activate-cntk && python ConvNet_MNIST.py"
```

## Run with Multiple GPU

### Build CNTK with 1bit-SGD Image
```
$ wget https://raw.githubusercontent.com/Microsoft/CNTK/master/Tools/docker/CNTK-GPU-1bit-Image/Dockerfile
$ docker build . -t cntk-1bit --build-arg ENABLE_1BIT_SGD=true
```

``` console
$ nvidia-docker run --rm -it -v $(pwd):/work -w /work cntk-1bit bash -c "source /cntk/activate-cntk && mpiexec -n $GPU_COUNT python ConvNet_MNIST_Distributed.py"
```
