import torch

'''
    frozen the original parameters pf pre-trained models.
    only active the learnable parameters of ARC modules and head layer.
'''

def ViT_ARC_Frozen(model):
    for name, para in model.named_parameters():
        if "head" in name:
            para.requires_grad = True
        elif "adapter" in name:
            para.requires_grad = True
        elif "down_projection" in name:
            para.requires_grad = True
        else:
            para.requires_grad = False

def Swin_ARC_Frozen(model):
    for name, para in model.named_parameters():
        if "head" in name:
            para.requires_grad = True
        elif "attn_u" in name:
            para.requires_grad = True
        elif "mlp_u" in name:
            para.requires_grad = True
        elif "lambda" in name:
            para.requires_grad = True
        else:
            para.requires_grad = False
