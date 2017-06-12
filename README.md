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
