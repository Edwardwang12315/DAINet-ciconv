<p align="center">
  <h1 align="center">Boosting Object Detection with Zero-Shot Day-Night Domain Adaptation
</h1>
  <p align="center">
    <a href="https://zpdu.github.io/">Zhipeng Du</a>
    ·
    <a href="https://sites.google.com/site/miaojingshi/home">Miaojing Shi</a>
    ·
    <a href="https://jiankangdeng.github.io/">Jiankang Deng</a>
  </p>
  
## **上传记录**
- 7.21


PyTorch implementation of **Boosting Object Detection with Zero-Shot Day-Night Domain Adaptation**. (CVPR 2024) [[Page](https://zpdu.github.io/DAINet_page/) | [Paper](https://arxiv.org/abs/2312.01220)]

![overview](./assets/overview.png)



## 🔨 To-Do List

1. - [x] release the code regarding the proposed model and losses.
3. - [x] release the evaluation code, and the pretrained models.

3. - [x] release the training code.

## :rocket: Installation

Begin by cloning the repository and setting up the environment:

```
git clone https://github.com/ZPDu/DAI-Net.git
cd DAI-Net

conda create -y -n dainet python=3.7
conda activate dainet

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt
```

## :notebook_with_decorative_cover: Training

#### Data and Weight Preparation

- Download the WIDER Face Training & Validation images at [WIDER FACE](http://shuoyang1213.me/WIDERFACE/).
- Obtain the annotations of [training set](https://github.com/daooshee/HLA-Face-Code/blob/main/train_code/dataset/wider_face_train.txt) and [validation set](https://github.com/daooshee/HLA-Face-Code/blob/main/train_code/dataset/wider_face_val.txt).
- Download the [pretrained weight](https://drive.google.com/file/d/1MaRK-VZmjBvkm79E1G77vFccb_9GWrfG/view?usp=drive_link) of Retinex Decomposition Net.
- Prepare the [pretrained weight](https://drive.google.com/file/d/1whV71K42YYduOPjTTljBL8CB-Qs4Np6U/view?usp=drive_link) of the base network.

Organize the folders as:

```
.
├── utils
├── weights
│   ├── decomp.pth
│   ├── vgg16_reducedfc.pth
├── dataset
│   ├── wider_face_train.txt
│   ├── wider_face_val.txt
│   ├── WiderFace
│   │   ├── WIDER_train
│   │   └── WIDER_val
```

#### Model Training

To train the model, run

```
python -m torch.distributed.launch --nproc_per_node=$NUM_OF_GPUS$ train.py
CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 train.py

```

## :notebook: Evaluation​

On Dark Face:

- 谷歌盘链接下载指令：
  ```bash
  gg gdown https://drive.google.com/uc?id=1BdkYLGo7PExJEMFEjh28OeLP4U1Zyx30 (DarkFaceZSDA.pth)
  ```
- Download the testing samples from [UG2+ Challenge](https://codalab.lisn.upsaclay.fr/competitions/8494?secret_key=cae604ef-4bd6-4b3d-88d9-2df85f91ea1c).
- Download the checkpoints: [DarkFaceZSDA](https://drive.google.com/file/d/1BdkYLGo7PExJEMFEjh28OeLP4U1Zyx30/view?usp=drive_link) (28.0) or [DarkFaceFS](https://drive.google.com/file/d/1ykiyAaZPl-mQDg_lAclDktAJVi-WqQaC/view?usp=drive_link) (52.9, finetuned with full supervision).
- Set (1) the paths of testing samples & checkpoint, (2) whether to use a multi-scale strategy, and run test.py.
- Submit the results for benchmarking. ([Detailed instructions](https://codalab.lisn.upsaclay.fr/competitions/8494?secret_key=cae604ef-4bd6-4b3d-88d9-2df85f91ea1c)).

On ExDark:

- Our experiments are based on the codebase of [MAET](https://github.com/cuiziteng/ICCV_MAET). You only need to replace the checkpoint with [ours](https://drive.google.com/file/d/1g74-aRdQP0kkUe4OXnRZCHKqNgQILA6r/view?usp=drive_link) for evaluation.

# 调试记录
## 2025.1.22
- test输出只有预测txt文件，补充了把预测框绘制出来的步骤
- 简单筛选了一下，置信度小于0.3的不显示，效果很好
- 以上测试用的是作者提供的权重文件，只适用于人脸检测
- _C.TOP_K = 20时，mAP=14.19
- _C.TOP_K = 750时，mAP=14.21

## 2025.4.10
- 完美收敛的结果应该是
- ->> pal1 conf loss:1.4184 || pal1 loc loss:0.6319
- ->> pal2 conf loss:1.1226 || pal2 loc loss:0.8053
- ->> mutual loss:0.0051 || enhanced loss:0.0348
- 训练的结果还有一段距离
- ->> pal1 conf loss:1.3814 || pal1 loc loss:2.4703
- ->> pal2 conf loss:2.0561 || pal2 loc loss:2.3194
- ->> mutual loss:0.0049 || enhanced loss:0.0627

## 2025.4.15
- 直接训练ref部分，测试这个模块能否实现效果
  - 方案一：去除检测模块，直接训练vgg2和decoder
