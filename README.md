# CNTK MNIST

## Setup

### Download MNIST DataSet
``` console
$ docker run --rm -it -v $(pwd):/work -w /work microsoft/cntk:2.0-cpu-python3.5 python data_loader.py
```

## Run `cnn.py`
``` console
$ docker run --rm -it -v $(pwd):/work -w /work microsoft/cntk:2.0-cpu-python3.5 python cnn.py
```
