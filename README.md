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


First use the `git clone` command to download **Detectron2** source code from the official github repository.
Then switch the **Detectron2** into the old version with commit id **9eb4831** and install it:
```bash
cd detectron2
git checkout -f 9eb4831
cd ..
python -m pip install -e detectron2
```


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

2. Please put video frame sequences in an input folder (e.g. `inputs`):

```shell
inputs
├── Sequence_1
│   ├── Frame_1.png
│   ├── Frame_2.png
│   ├── Frame_3.png
│   └── ...
│
├── Sequence_2
│   ├── Frame_1.png
│   ├── Frame_2.png
│   ├── Frame_3.png
│   └── ...
│
└── ...
```

**[Note]** You can found some examples in our `inputs` folder.

3. Run the demo with:
```bash
./run.sh inputs
```