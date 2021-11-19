# Install Detectron2

Our method is on top of [Detectron2](https://github.com/facebookresearch/detectron2) framework. You can install it in any place.

Please use Detectron2 with commit id [9eb4831](https://github.com/facebookresearch/detectron2/commit/9eb4831f742ae6a13b8edb61d07b619392fb6543) if you have any issues related to Detectron2:

1. Clone the source code from github:

```bash
git clone https://github.com/facebookresearch/detectron2/tree/main/docs
```

2. Switch to the [old version](https://github.com/facebookresearch/detectron2/commit/9eb4831f742ae6a13b8edb61d07b619392fb6543) with commit id **9eb4831**:

```bash
cd detectron2
git checkout -f 9eb4831
```

3. Install the Detectron2 from the source code:

```bash
cd ..
python -m pip install -e detectron2
```

