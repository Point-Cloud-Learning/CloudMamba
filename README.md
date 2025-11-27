<div align="center">

# CloudMamba: Grouped Selective State Spaces for Point Cloud Analysis

</div>

## Abstract
Due to the long-range modeling ability and linear complexity property, Mamba has attracted considerable attention in point cloud analysis. Despite some interesting progress, related work still suffers from imperfect point cloud serialization, insufficient high-level geometric perception, and overfitting of the selective state space model (S6) at the core of Mamba. To this end, we resort to an SSM-based point cloud network termed CloudMamba to address the above challenges. Specifically, we propose sequence expanding and sequence merging, where the former serializes points along each axis separately and the latter serves to fuse the corresponding higher-order features causally inferred from different sequences, enabling unordered point sets to adapt more stably to the causal nature of Mamba without parameters. Meanwhile, we design chainedMamba that chains the forward and backward processes in the parallel bidirectional Mamba, capturing high-level geometric information during scanning. In addition, we propose a grouped selective state space model (GS6) via parameter sharing on S6, alleviating the overfitting problem caused by the computational mode in S6. Experiments on various point cloud tasks validate CloudMamba's ability to achieve state-of-the-art results with significantly less complexity.

## Overview

<div align="center">

<img src="./Assets/Pipeline.png"/>

</div>

## Install

This codebase was tested with the following environment configurations. It may work with other versions.
- Ubuntu 22.04
- CUDA 11.8
- Python 3.8
- PyTorch 2.0.0 + cu118

We recommend using Anaconda for the installation process:
```shell 
# Clone the repository
$ git clone https://github.com/Point-Cloud-Learning/CloudMamba.git
$ cd CloudMamba

# Create virtual env and install PyTorch
$ conda create -n CloudMamba python=3.8
$ conda activate CloudMamba
(CloudMamba) $ conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install basic required packages
(CloudMamba) $ pip install -r requirements.txt

# PointNet++
(CloudMamba) $ pip install pointnet2_ops_lib/.
```

## Acknowledgement

This project is based on Mamba ([paper](https://arxiv.org/abs/2312.00752), [code](https://github.com/state-spaces/mamba)), 
Causal-Conv1d ([code](https://github.com/Dao-AILab/causal-conv1d)), 
PointMamba([paper](https://arxiv.org/abs/2402.10739), [code](https://github.com/LMD0311/PointMamba)),
PCM([paper](https://arxiv.org/abs/2403.00762), [code](https://github.com/SkyworkAI/PointCloudMamba)),
Mamba3D([paper](https://arxiv.org/abs/2404.14966), [code](https://github.com/xhanxu/Mamba3D)). 
Thanks for their wonderful works.

## Citation

If you find this repository useful in your research, please consider giving a star ⭐ and a citation
```BibTeX
@Inproceedings{CloudMamba,
  author =       "K. L. Qu and P. Gao and Q. Dai and Z. Z. Ye and R. Ye and Y. H. Sun",
  title =        "CloudMamba: Grouped Selective State Spaces for Point Cloud Analysis",
  booktitle =    "Proc. AAAI Conference on Artificial Intelligence (AAAI)",
  address =      "Singapore EXPO",
  year =         2026
}
```

## LICENSE

PointMLP is under the Apache-2.0 license.
