### :book: High-resolution Depth Maps Imaging via Attention-based Hierarchical Multi-modal Fusion (IEEE TIP 2022)

> [[Paper](https://https://arxiv.org/abs/2104.01530)] 
> [Zhiwei Zhong](https://github.com/zhwzhong), [Xianming Liu](http://homepage.hit.edu.cn/xmliu?lang=en), [Junjun Jiang](https://scholar.google.com/citations?user=WNH2_rgAAAAJ&hl=en), [Debin Zhao](https://scholar.google.com/citations?user=QXyj0hkAAAAJ&hl=en) ,[Xiangyang Ji](https://ieeexplore.ieee.org/author/37271425200)<br>Harbin Institute of Technology, Tsinghua University

#### Abstract

Depth map records distance between the viewpoint and objects in the scene, which plays a critical role in many realworld applications. However, depth map captured by consumergrade RGB-D cameras suffers from low spatial resolution. Guided depth map super-resolution (DSR) is a popular approach to address this problem, which attempts to restore a highresolution (HR) depth map from the input low-resolution (LR) depth and its coupled HR RGB image that serves as the guidance. The most challenging issue for guided DSR is how to correctly select consistent structures and propagate them, and properly handle inconsistent ones. In this paper, we propose a novel attention-based hierarchical multi-modal fusion (AHMF) network for guided DSR. Specifically, to effectively extract and combine relevant information from LR depth and HR guidance, we propose a multi-modal attention based fusion (MMAF) strategy for hierarchical convolutional layers, including a feature enhancement block to select valuable features and a feature recalibration block to unify the similarity metrics of modalities with different appearance characteristics. Furthermore, we propose a bi-directional hierarchical feature collaboration (BHFC) module to fully leverage low-level spatial information and high-level structure information among multi-scale features. Experimental results show that our approach outperforms state-of-the-art methods in terms of reconstruction accuracy, running speed and memory efficiency.


---

This repository is an official PyTorch implementation of the paper "**High-resolution Depth Maps Imaging via Attention-based Hierarchical Multi-modal Fusion**"

### :wrench: Dependencies and Installation

- Python >= 3.5 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.2(https://pytorch.org/
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

### Installation

1. Clone repo

   ```bash
   git https://github.com/zhwzhong/AHMF.git
   cd AHMF
   ```

2. Install dependent packages

   ```bash
   pip install -r requirements.txt
   ```

### Train

You can also train by yourself:

```
 The training codes can be found at https://github.com/zhwzhong/Guided-Depth-Map-Super-resolution-A-Survey.
```



### Quick Test (We provide the trained models for test. The trained models can be can be found at:

链接:https://pan.baidu.com/s/1hhYhvUC6BfF5ceJX2MIs4g  密码:0cmf

链接:https://pan.baidu.com/s/1Y2eNss_Vp2eVZWTq5mrxLQ  密码:4y0d.

```
python test.py
```


:e-mail: Contact

If you have any question, please email `zhwzhong@hit.edu.cn` 

### Cititation
@ARTICLE{9642435,
  author={Zhong, Zhiwei and Liu, Xianming and Jiang, Junjun and Zhao, Debin and Chen, Zhiwen and Ji, Xiangyang},
  journal={IEEE Transactions on Image Processing}, 
  title={High-Resolution Depth Maps Imaging via Attention-Based Hierarchical Multi-Modal Fusion}, 
  year={2022},
  volume={31},
  number={},
  pages={648-663},
  doi={10.1109/TIP.2021.3131041}}


