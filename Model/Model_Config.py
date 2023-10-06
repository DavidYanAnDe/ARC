import ml_collections


# 获取VIT_B_16模型
def get_b16_config():
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_drop_rate = 0.0
    config.transformer.mlp_drop_rate = 0.0
    # 按照什么输出进行分类
    config.classifier = 'token'
    config.adapter_dim = 50
    config.adapter_dropout = 0.0

    return config


def get_l16_config():
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1024
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 4096
    config.transformer.num_heads = 16
    config.transformer.num_layers = 24
    config.transformer.attention_drop_rate = 0.0
    config.transformer.mlp_drop_rate = 0.0
    # 按照什么输出进行分类
    config.classifier = 'token'
    config.adapter_dim = 50
    config.adapter_dropout = 0.0

    return config


def get_h14_config():
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1280
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 5120
    config.transformer.num_heads = 16
    config.transformer.num_layers = 32
    config.transformer.attention_drop_rate = 0.0
    config.transformer.mlp_drop_rate = 0.0
    # 按照什么输出进行分类
    config.classifier = 'token'
    config.adapter_dim = 50
    config.adapter_dropout = 0.0

    return config


CONFIGS = {
    "ViT-B_16": get_b16_config(),
    "ViT-L_16": get_l16_config(),
    "ViT-H_14": get_h14_config(),
}