# STC-Seg
Solve the Puzzle of Instance Segmentation in Videos


## Dependencies

* Python 3.7
* Pytorch 1.4
* Detectron2 `(9eb4831)`
* torchvision, opencv, cudatoolkit

This repo was tested with Python 3.7.10, PyTorch 1.4.0, cuDNN 7.6, and CUDA 10.0. But it should be runnable with more recent PyTorch>=1.4 versions.

You can use anaconda or miniconda to install those dependencies:
```bach
conda create -n STC-Seg-pytorch python=3.7 pytorch=1.4 torchvision opencv cudatoolkit=10.0
conda activate STC-Seg-pytorch
```


Use the `git clone` command to download **Detectron2** source code from the official github repository.
Then switch the **Detectron2** into the old version with commit id **9eb4831** and install it:
```bash
cd detectron2
git checkout -f 9eb4831
cd ..
python -m pip install -e detectron2
```
More details please see [docs/Install_Detectron2.md](docs/Install_Detectron2.md)

## Installation

Please build the STC-Seg with:
```bash
cd STC-Seg
python setup.py build develop
```

If any error occurs in STC-Seg installation, please remove the `build` folder before restart.


## Inference

1. Please run this script to get the trained STC-Seg models:

```bash
python tools/download_models.py
```

2. Please download examples of video frame sequences:

```shell
python tools/download_examples.py
```

3. Run the demo with those examples (under `inputs` folder):
```bash
bash run.sh inputs
```

4. Results will be saved under `results` folder.