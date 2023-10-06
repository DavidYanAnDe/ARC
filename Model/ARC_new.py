import copy
import logging
import math

import torch.nn as nn
from torch.nn.functional import gelu, relu
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
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
    def __init__(self, config, img_size, in_channel=3):
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
        self.dropout = Dropout(config.transformer["mlp_drop_rate"])

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

        self.head_num = config.transformer["num_heads"]

        self.head_hidden_dim = hidden_dim // self.head_num
        self.all_head_size = self.head_hidden_dim * self.head_num


        self.query = Linear(hidden_dim, self.all_head_size)
        self.key = Linear(hidden_dim, self.all_head_size)
        self.value = Linear(hidden_dim, self.all_head_size)


        self.softmax = Softmax(dim=-1)

        self.projective = Linear(hidden_dim, hidden_dim)

        self.attn_dropout = Dropout(config.transformer["attention_drop_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_drop_rate"])

    def transpose_head(self, mix_metric):
        new_x_shape = mix_metric.size()[:-1] + (self.head_num, self.head_hidden_dim)

        mix_metric = mix_metric.view(*new_x_shape)

        return mix_metric.permute(0, 2, 1, 3)

    def forward(self, hidden_states):

        mix_query_layer = self.query(hidden_states)
        mix_key_layer = self.key(hidden_states)
        mix_value_layer = self.value(hidden_states)


        query_layer = self.transpose_head(mix_query_layer)
        key_layer = self.transpose_head(mix_key_layer)
        value_layer = self.transpose_head(mix_value_layer)


        attention_metric = torch.matmul(query_layer, key_layer.transpose(-2, -1))
        attention_metric = attention_metric / math.sqrt(self.head_hidden_dim)
        attention_probs = self.softmax(attention_metric)

        attention_probs = self.attn_dropout(attention_probs)
        attention_features = torch.matmul(attention_probs, value_layer)

        attention_features = attention_features.permute(0, 2, 1, 3).contiguous()
        new_output_features_shape = attention_features.size()[:-2] + (self.all_head_size,)
        attention_feature = attention_features.view(*new_output_features_shape)


        output_features = self.projective(attention_feature)
        output_features = self.proj_dropout(output_features)

        return output_features


class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()

        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])

        self.activation = ACT2FN["gelu"]
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)

        self.dropout = Dropout(config.transformer["mlp_drop_rate"])

    def forward(self, input_features):

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

        self.att_layer_norm = LayerNorm(self.hidden_size, eps=1e-6)
        self.fnn_layer_norm = LayerNorm(self.hidden_size, eps=1e-6)

        self.att_scale = nn.Parameter(torch.empty(1, config.adapter_dim))
        self.att_scale_bias = nn.Parameter(torch.empty(self.hidden_size))

        self.mlp_scale = nn.Parameter(torch.empty(1, config.adapter_dim))
        self.mlp_scale_bias = nn.Parameter(torch.empty(self.hidden_size))

        nn.init.zeros_(self.att_scale)
        nn.init.zeros_(self.att_scale_bias)
        nn.init.xavier_uniform_(self.mlp_scale)
        nn.init.zeros_(self.mlp_scale_bias)

        self.adapter_drop = nn.Dropout(config.adapter_dropout)

    def forward(self, hidden_states, left_u_att, right_v_att, left_u_mlp, right_v_mlp):
        res_att_feature = hidden_states
        hidden_states = self.att_layer_norm(hidden_states)
        
        attention_uv_output = torch.matmul(hidden_states, left_u_att * self.att_scale)
        attention_uv_output = self.adapter_drop(attention_uv_output)
        attention_uv_output = torch.matmul(attention_uv_output, right_v_att) + self.att_scale_bias
        attention_uv_output = hidden_states + attention_uv_output
        
        attention_output = self.attention(attention_uv_output)
        attention_output = attention_output + res_att_feature

        res_fnn_feature = attention_output
        fnn_output = self.fnn_layer_norm(attention_output)
        
        mlp_uv_output = torch.matmul(fnn_output, left_u_mlp * self.mlp_scale)
        mlp_uv_output = self.adapter_drop(mlp_uv_output)
        mlp_uv_output = torch.matmul(mlp_uv_output, right_v_mlp) + self.mlp_scale_bias
        mlp_uv_output = mlp_uv_output + fnn_output
        
        fnn_output = self.mlp(mlp_uv_output)
        fnn_output = fnn_output + res_fnn_feature
        return fnn_output

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()

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
        self.layer_num = config.transformer["num_layers"]
        self.layer = nn.ModuleList()
        for _ in range(self.layer_num):
            encoder = Encoder(config)
            self.layer.append(copy.deepcopy(encoder))

        self.layer_norm = LayerNorm(config.hidden_size, eps=1e-6)

        self.left_u_att = nn.Parameter(torch.empty(config.hidden_size, config.adapter_dim))
        self.left_u_mlp = nn.Parameter(torch.empty(config.hidden_size, config.adapter_dim))

        nn.init.xavier_uniform_(self.left_u_mlp)
        nn.init.xavier_uniform_(self.left_u_att)


    def forward(self, input_feature):
        for layer_encoder in self.layer:
            input_feature = layer_encoder(input_feature, self.left_u_att, self.left_u_att.t(), self.left_u_mlp,
                                          self.left_u_mlp.t())

        output = self.layer_norm(input_feature)
        return output

class HeadMLP(nn.Module):
    def __init__(self, config, num_classes):
        super(HeadMLP, self).__init__()
        self.head = Linear(config.hidden_size, num_classes)

    def forward(self, input_features):
        cls_token = input_features[:, 0]

        output = self.head(cls_token)
        return output


class ARCVisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False):
        super(ARCVisionTransformer, self).__init__()
        self.zero_head = zero_head
        self.num_classes = num_classes

        self.embedding = Embedding(config, img_size)
        self.transformer = Transformer(config)

        self.head_mlp = HeadMLP(config, num_classes)

    def forward(self, image, labels=None):
        image_projective = self.embedding(image)
        output_representation = self.transformer(image_projective)
        prediction = self.head_mlp(output_representation)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(prediction.view(-1, self.num_classes), labels.view(-1))
            return loss
        else:
            return prediction

    def load_from(self, weights):
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.head_mlp.head.weight)
                nn.init.zeros_(self.head_mlp.head.bias)
            else:
                self.head_mlp.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head_mlp.head.weight.copy_(np2th(weights["head/bias"]).t())

            self.embedding.projective.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.embedding.projective.bias.copy_(np2th(weights["embedding/bias"]))
            self.embedding.cls.copy_(np2th(weights["cls"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.embedding.position_embedding
            if posemb.size() == posemb_new.size():
                self.embedding.position_embedding.copy_(posemb)

            self.transformer.layer_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.layer_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            for bname, encoder in self.transformer.named_children():
                for uname, unit in encoder.named_children():
                    unit.load_from(weights, n_block=uname)
