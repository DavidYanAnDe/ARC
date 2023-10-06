# Efficient Adaptation of Large Vision Transformer via Adapter Re-Composing (ARC)

This repo is the official implementation of our NeurIPS2023 paper "Efficient Adaptation of Large Vision Transformer via Adapter Re-Composing" ([arXiv](https://arxiv.org/abs/)). 




## Usage

### Data preparation

- FGVC & vtab-1k

You can follow [VPT](https://github.com/KMnP/vpt) to download them. 

Since the original [vtab dataset](https://github.com/google-research/task_adaptation/tree/master/task_adaptation/data) is processed with tensorflow scripts and the processing of some datasets is tricky, we also upload the extracted vtab-1k dataset in [onedrive](https://shanghaitecheducn-my.sharepoint.com/:f:/g/personal/liandz_shanghaitech_edu_cn/EnV6eYPVCPZKhbqi-WSJIO8BOcyQwDwRk6dAThqonQ1Ycw?e=J884Fp) for your convenience. You can download from here and then use them with our [vtab.py](https://github.com/dongzelian/SSF/blob/main/data/vtab.py) directly. (Note that the license is in [vtab dataset](https://github.com/google-research/task_adaptation/tree/master/task_adaptation/data)).

### Install

- Clone this repo:

```bash
git clone https://github.com/DavidYanAnDe/ARC.git
cd ARC
```

- Install requirements:

```bash
pip install -r requirements.txt
```



### Pre-trained model preparation

- For pre-trained ViT, Swin-B models on ImageNet-21K. You can also manually download them from [ViT](https://github.com/google-research/vision_transformer),[Swin Transformer](https://github.com/microsoft/Swin-Transformer).



### Train ARC model

To fine-tune a pre-trained ViT model on vtab, run:

```bash
python vtab_ARC_train.py
```

To fine-tune a pre-trained ViT model on FGVC, run:

```bash
python FGVC_ARC_train.py
```

To specify the ARC or ARC_att, change the model setting at:

```bash
parser.add_argument("--tuning_mode", choices=["ARC_att", "ARC"], default= "ARC",  help="tuning mode,can be ARC_att or ARC")
```



### Citation
If this project is helpful for you, you can cite our paper:
```


```


### Acknowledgement
The code is built upon [timm](https://github.com/jeonsworld/ViT-pytorch). The processing of the vtab-1k dataset refers to [vpt](https://github.com/KMnP/vpt), [vtab github repo](https://github.com/google-research/task_adaptation/tree/master/task_adaptation/data), and [NOAH](https://github.com/ZhangYuanhan-AI/NOAH).
