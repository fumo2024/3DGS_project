# Active 3D Gaussian Splatting 
This is a course project designed for deep generative models, with the goal of reproducing the results of the 3DGS paper and implementing certain improvements upon it.

This repository is built based on the [official repository of 3DGS](https://github.com/graphdeco-inria/gaussian-splatting/).

## Get Started
### Cloning the Repository
first clone this repository, unlike the original 3DGS repository, the `diff-gaussian-rasterization` and `simple-knn` libraries are already in the repo, not as git submodules.
```plain
submodules/diff-gaussian-rasterization
submodules/simple-knn
```
### Install Dependencies
running the code needs some dependencies, you can install them with the following command:
```bash
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
conda install nvidia/label/cuda-11.8.0::cuda # optional, for nvcc toolkits
```
You also need to install two customized packages `diff-gaussian-rasterization`, `simple-knn`:
```bash
# remember to specify the cuda library path if some cuda header is missing
cd submodules/diff-gaussian-rasterization
pip install -e .

# remember to specify the cuda library path if some cuda header is missing
cd submodules/simple-knn
pip install -e .
```
### Data Preparation

The data downloading and processing are the same with the original 3DGS. Please refer to [here](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#running) for more details. If you want to run SteepGS on your own dataset, please refer to [here](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#processing-your-own-scenes) for the instructions.


## Running

The simplest way to use and evaluate ActiveGS is through the following commands:

```shell
python train.py -s <path to COLMAP or NeRF Synthetic dataset> -m <path to checkpoint> --no_gui --eval # Train with train/test split
python render.py -m <path to trained model> # Generate renderings
python metrics.py -m <path to trained model> # Compute error metrics on renderings
```

## Acknowledgements
During my work on this project, I have received invaluable support and resources from the following sources:
| name | link |
| ---- | ---- |
| 3DGS | [https://github.com/graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) |
| SteepGS | [https://github.com/facebookresearch/SteepGS](https://github.com/facebookresearch/SteepGS) |
|...|...|

## License
This code is released under the Gaussian-Splatting license. (see [LICENSE](LICENSE.md)).