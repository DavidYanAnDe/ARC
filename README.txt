Prepare:

1.Pre-trained models:

The pre-trained weights files of ViT models are provided by VPT(github.com/kmnp/vpt), we show the links bellow:
https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz
https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_16.npz
https://storage.googleapis.com/vit_models/imagenet21k/ViT-H_14.npz

The pre-trained weights files of Swin models can be downloaded automaticly.

2.Datasets:
Also refer to the VPT(github.com/kmnp/vpt).
We utilize the following link to download VTAB-1k datasets:
https://github.com/luogen1996/RepAdapter/issues/2

For FGVC datasets, we use the same training datasets and test datasets splits of VPT. You can get the splits files from this link:
https://drive.google.com/drive/folders/1mnvxTkYxmOr2W9QjcgS64UBpoJ4UmKaM
The datasets of FGVC you can download from the official website provided by github.com/kmnp/vpt.


Files Framework:
1.Data_process: includes data preprocessing files.
2.Model: contains ViT、 ARC_ViT、ARC_swin_b different variantes.
3.Utils: consists of tools functions.

vtab_ARC_train.py: The training file for VTAB-1k datasets on ViT variants.
In the main function of this file, you con config the paras for training
--data_path: the path where you store the dataset
--name: the name you want for the output model
--dataset: the download task you wanna train
--tuning_mode: you can select two varints of our ARC mothods: ARC_att or ARC
--pretrained_dir: the pre-trained model


FGVC_ARC_train.py: The training file for FGVC datasets on ViT variants.
In the main function of this file, you con config the paras for training
--data_path: the path where you store the dataset
--name: the name you want for the output model
--dataset: the download task you wanna train
--tuning_mode: you can select two varints of our ARC mothods: ARC_att or ARC
--pretrained_dir: the pre-trained model

train_ARC_swin.py: The training file for VTAB-1k datasets on Swin-B.