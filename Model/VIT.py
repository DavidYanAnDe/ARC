import copy
import logging
import math

import torch.nn as nn
from torch.nn.functional import gelu, relu
from torch.nn import  CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
import torch
from torch.nn.modules.utils import _pair
from Utils.tools import np2th
from os.path import join as pjoin


logger = logging.getLogger(__name__)

# 定义MLP需要用到的激活函数
ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu}

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FNN_0 = "MlpBlock_3/Dense_0"
FNN_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

# 图像输入的embedding层
class Embedding(nn.Module):
    def __init__(self, config, img_size, in_channel = 3):
        super(Embedding, self).__init__()

        # image_size(n,n)    patch_size(t, t)
        image_size = _pair(img_size)
        patch_size = config.patches['size']
        num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])

        # 将[H,W,3]转换为[224/16, 224/16, hidden_size]
        self.projective = Conv2d(in_channels=in_channel,
                                    out_channels=config.hidden_size,
                                    kernel_size=patch_size,
                                    stride=patch_size)

        # 可学习的cls, cls维度[1, 1, hidden_dim]
        self.cls = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        # 可学习的position_embedding, [1, patch_num+1, hidden_dim]
        self.position_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))

        # 输入encode前有一个dropout
        self.dropout = Dropout(config.transformer["drop_rate"])

    def forward(self, input_image):
        # [B, hidden_size, patch_size, patch_size]
        projective_embedding = self.projective(input_image)
        # 展开，并且调换维度，变成[B, num_patches, hidden_size]
        projective_embedding = projective_embedding.flatten(2).transpose(-1, -2)

        # num_patches, 加入class_token. [B, num_patches, hidden_size]和[1, 1, hidden_size]
        # 首先将cls变为[B, 1, hidden_size]
        batch_size = input_image.shape[0]
        cls = self.cls.expand(batch_size, -1, -1)
        # 拼接cls和projective_embedding, 变为[B, num_patches+1, hidden_size]
        embedding_features = torch.concat([cls, projective_embedding], dim=1)

        output_features = embedding_features + self.position_embedding
        output_features = self.dropout(output_features)
        return output_features

# encoder中的自注意力层
class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()

        hidden_dim = config.hidden_size
        # 一共多少个注意力头
        self.head_num = config.transformer["num_heads"]
        # 每个头多少维度
        self.head_hidden_dim = hidden_dim // self.head_num
        self.all_head_size = self.head_hidden_dim * self.head_num

        # q,k,v三个参数矩阵
        self.query = Linear(hidden_dim, self.all_head_size)
        self.key = Linear(hidden_dim, self.all_head_size)
        self.value = Linear(hidden_dim, self.all_head_size)

        # 计算注意力头的归一化操作，对每一行进行归一化，也就是列归一化，
        self.softmax = Softmax(dim=-1)
        # 自注意力后需要一个project矩阵，进行特征的转换
        self.projective = Linear(hidden_dim, hidden_dim)
        # dropout在attention输出后
        self.attn_dropout = Dropout(config.transformer["attention_drop_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_drop_rate"])

    def transpose_head(self, mix_metric):
        # size可以看成是一种tuple,tuple相加是维度的相加
        # new_x_shape可以看作是 [batch_size,n,H,n/H] （四维数组）
        new_x_shape = mix_metric.size()[:-1] + (self.head_num, self.head_hidden_dim)
        # *只改变形状，不改变内存地址
        mix_metric = mix_metric.view(*new_x_shape)
        # permute调整顺序
        return mix_metric.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        # hidden_states维度 [Batch_size, num_patches, hidden_dim]，计算出混合q,k,v
        mix_query_layer = self.query(hidden_states)
        mix_key_layer = self.key(hidden_states)
        mix_value_layer = self.value(hidden_states)

        # 分离注意力头，从[Batch_size, num_patches, hidden_dim]到[Batch_size, head_num, num_patches, head_dim]
        query_layer = self.transpose_head(mix_query_layer)
        key_layer = self.transpose_head(mix_key_layer)
        value_layer = self.transpose_head(mix_value_layer)

        # 计算attention score，注意需要归一化以及softmax
        attention_metric = torch.matmul(query_layer, key_layer.transpose(-2,-1))
        attention_metric = attention_metric / math.sqrt(self.head_hidden_dim)
        attention_probs = self.softmax(attention_metric)
        # dropout注意力矩阵
        attention_probs = self.attn_dropout(attention_probs)
        # 计算注意力矩阵m和value的乘积，[B, head_num, patch_num, head_dim]
        attention_features = torch.matmul(attention_probs, value_layer)

        # 对attention_features进行重新构建，恢复[batch_size,n,d]的格式
        attention_features = attention_features.permute(0, 2, 1, 3).contiguous()
        new_output_features_shape = attention_features.size()[:-2] + (self.all_head_size, )
        attention_feature = attention_features.view(*new_output_features_shape)

        # 输出前需要加一层映射
        output_features = self.projective(attention_feature)
        output_features = self.proj_dropout(output_features)

        return output_features

# encoder里面的MLP层
class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        # 两层MLP
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        # 中间有一层激活函数
        self.activation = ACT2FN["gelu"]
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        # MLP输出前需要加一层dropout
        self.dropout = Dropout(config.transformer["drop_rate"])

    def forward(self, input_features):
        # 逐层输出计算特征
        output_features = self.fc1(input_features)
        output_features = self.activation(output_features)
        output_features = self.dropout(output_features)
        output_features = self.fc2(output_features)
        output_features = self.dropout(output_features)

        return output_features

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self.hidden_size = config.hidden_size
        self.attention = Attention(config)
        self.mlp = MLP(config)
        # layer norm在输入attention和mlp前
        self.att_layer_norm = LayerNorm(self.hidden_size, eps=1e-6)
        self.fnn_layer_norm = LayerNorm(self.hidden_size, eps=1e-6)

    def forward(self, hidden_states):
        # 残差网络的特征
        res_att_feature = hidden_states
        # 输入前先进行layer norm
        hidden_states = self.att_layer_norm(hidden_states)
        attention_output = self.attention(hidden_states)
        attention_output = attention_output + res_att_feature

        res_fnn_feature = attention_output
        fnn_output = self.fnn_layer_norm(attention_output)
        fnn_output = self.mlp(fnn_output)
        fnn_output = fnn_output + res_fnn_feature
        return fnn_output

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attention.query.weight.copy_(query_weight)
            self.attention.key.weight.copy_(key_weight)
            self.attention.value.weight.copy_(value_weight)
            self.attention.projective.weight.copy_(out_weight)
            self.attention.query.bias.copy_(query_bias)
            self.attention.key.bias.copy_(key_bias)
            self.attention.value.bias.copy_(value_bias)
            self.attention.projective.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FNN_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FNN_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FNN_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FNN_1, "bias")]).t()

            self.mlp.fc1.weight.copy_(mlp_weight_0)
            self.mlp.fc2.weight.copy_(mlp_weight_1)
            self.mlp.fc1.bias.copy_(mlp_bias_0)
            self.mlp.fc2.bias.copy_(mlp_bias_1)

            self.att_layer_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.att_layer_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.fnn_layer_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.fnn_layer_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))

class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        # encoder层的数量
        self.layer_num = config.transformer["num_layers"]
        # encoder按照存储每一层
        self.layer = nn.ModuleList()
        # 添加每一层
        for _ in range(self.layer_num):
            encoder = Encoder(config)
            # 这里需要深度拷贝
            self.layer.append(copy.deepcopy(encoder))

        # 最后有一个layer_norm输出
        self.layer_norm = LayerNorm(config.hidden_size, eps=1e-6)
    def forward(self, input_feature):
        for layer_encoder in self.layer:
            input_feature = layer_encoder(input_feature)

        output = self.layer_norm(input_feature)
        return output

class HeadMLP(nn.Module):
    def __init__(self, config, num_classes):
        super(HeadMLP, self).__init__()
        self.head = Linear(config.hidden_size, num_classes)

    def forward(self, input_features):
        # input维度[B, num_patches, hidden_dim], 获取cls的表征就行，也就是[B, 0, hidden_dim]
        # cls_token维度[B, hidden_dim]
        cls_token = input_features[:, 0]

        # output维度[B, num_classes]
        output = self.head(cls_token)
        return output

class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False):
        super(VisionTransformer, self).__init__()
        self.zero_head = zero_head
        self.num_classes = num_classes

        self.embedding = Embedding(config, img_size)
        self.transformer = Transformer(config)

        self.head_mlp = HeadMLP(config, num_classes)

    def forward(self, image, labels=None):
        image_projective = self.embedding(image)
        output_representation = self.transformer(image_projective)
        # 只需要使用到cls这一个token，因此需要提取出来
        prediction = self.head_mlp(output_representation)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(prediction.view(-1, self.num_classes), labels.view(-1))
            return loss
        else:
            return prediction

    # 加载预训练模型
    def load_from(self, weights):
        with torch.no_grad():
            # 初始化头部，或者加载预训练参数
            if self.zero_head:
                nn.init.zeros_(self.head_mlp.head.weight)
                nn.init.zeros_(self.head_mlp.head.bias)
            else:
                self.head_mlp.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head_mlp.head.weight.copy_(np2th(weights["head/bias"]).t())

            # 加载embedding参数
            self.embedding.projective.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.embedding.projective.bias.copy_(np2th(weights["embedding/bias"]))
            self.embedding.cls.copy_(np2th(weights["cls"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.embedding.position_embedding
            if posemb.size() == posemb_new.size():
                self.embedding.position_embedding.copy_(posemb)

            # 加载transformer最后输出的参数
            self.transformer.layer_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.layer_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            for bname, encoder in self.transformer.named_children():
                for uname, unit in encoder.named_children():
                    unit.load_from(weights, n_block=uname)

