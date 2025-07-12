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

Some users may encounter significant issues during dependency installation, primarily due to the use of an older CUDA version. Possible solutions include installing the corresponding CUDA version or using a clean virtual machine. For Windows users, you may refer to
[this video](https://www.youtube.com/watch?v=UXtuigy_wYc) for guidance.

As for me, I used a pre-configured 3DGS Docker image from the AutoDL community and ran the experimental part of this project on a cloud server with a 24GB RTX 4090 GPU, which helped me avoid most dependency-related problems.

### Data Preparation

The data downloading and processing are the same with the original 3DGS. Please refer to [here](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#running) for more details. If you want to run SteepGS on your own dataset, please refer to [here](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#processing-your-own-scenes) for the instructions.

Here is a brief description of the basic data processing pipeline:

- Data Collection – Capture a video using a camera, then use FFmpeg to split it into individual frames (images) and save them in a specified directory structure.

- Preprocessing – Run the convert.py script to process the data. This script uses COLMAP to estimate camera poses and generate an initial sparse point cloud.

- Training – Execute train.py to train the 3D Gaussian Splatting (3DGS) model. I need `--eval` flag to split the dataset into training and testing sets, which is required for the evaluation of the model.

- Rendering & Evaluation

```shell
python train.py -s <path to COLMAP or NeRF Synthetic dataset> -m <path to checkpoint> --no_gui --eval # Train with train/test split
python render.py -m <path to trained model> # Generate renderings
python metrics.py -m <path to trained model> # Compute error metrics on renderings
```

Anyway, the data for experiments in this project is the same as the original 3DGS, and other data sources like sensors are also welcome.

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