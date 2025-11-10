DSFD基于预训练用DarkFace调优做60个epoch
# 8.26
- 基于预训练在DarkFace上调优60个epoch
  - 8.28完成 记录权重文件为 dsfd_finetuneBEST.pth
  - 8.29完成测试 result 评估 62.00


```python
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train.py
```

```python
# 单通道边缘图显示方法
image = inv_out[ 0 ].detach().cpu().numpy().squeeze()  # 维度 [H, W]

# 归一化到对称范围
vmax = np.max( np.abs( image ) )
image_normalized = image / vmax  # 范围[-1, 1]

# 使用红蓝颜色映射可视化
plt.imshow( image_normalized , cmap = 'RdBu' , vmin = -1 , vmax = 1 )
plt.axis( 'off' )
plt.colorbar( label = 'Edge Strength (Red: Positive, Blue: Negative)' )
plt.show()
# 保存图像到文件
plt.savefig( f'ciconv.png' , bbox_inches = 'tight' , pad_inches = 0 , dpi = 800 )
```

```python
# 查询cpu核心数，用来调整worker数——一般为核心数的0.5-0.75
print(f'cpus num = {os.cpu_count()}')
```

```bash
tmux窗口下，ctrl + b 再加 [ 可以启动复制模式,鼠标任意滚动
或者 输入指令 tmux set -g mouse on
```


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
  
## 上传记录
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
