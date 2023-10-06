import ml_collections


def get_caltech101_config():
    config = ml_collections.ConfigDict()
    config.Num_Classes = 102
    config.CropSize = 224

    config.Num_workers = 4
    config.pin_memory = True

    config.NUM_GPUS = 1
    return config


def get_cifar_config():
    config = ml_collections.ConfigDict()
    config.Num_Classes = 100
    config.CropSize = 224

    config.Num_workers = 4
    config.pin_memory = True

    config.NUM_GPUS = 1
    return config


def get_clevr_count_config():
    config = ml_collections.ConfigDict()
    config.Num_Classes = 8
    config.CropSize = 224

    config.Num_workers = 4
    config.pin_memory = True

    config.NUM_GPUS = 1
    return config

def get_clevr_dist_config():
    config = ml_collections.ConfigDict()
    config.Num_Classes = 6
    config.CropSize = 224

    config.Num_workers = 4
    config.pin_memory = True

    config.NUM_GPUS = 1
    return config

def get_diabetic_retinopathy_config():
    config = ml_collections.ConfigDict()
    config.Num_Classes = 5
    config.CropSize = 224

    config.Num_workers = 4
    config.pin_memory = True

    config.NUM_GPUS = 1
    return config

def get_dmlab_config():
    config = ml_collections.ConfigDict()
    config.Num_Classes = 6
    config.CropSize = 224

    config.Num_workers = 4
    config.pin_memory = True

    config.NUM_GPUS = 1
    return config

def get_dsprites_loc_config():
    config = ml_collections.ConfigDict()
    config.Num_Classes = 16
    config.CropSize = 224

    config.Num_workers = 4
    config.pin_memory = True

    config.NUM_GPUS = 1
    return config

def get_dsprites_ori_config():
    config = ml_collections.ConfigDict()
    config.Num_Classes = 16
    config.CropSize = 224

    config.Num_workers = 4
    config.pin_memory = True

    config.NUM_GPUS = 1
    return config

def get_dtd_config():
    config = ml_collections.ConfigDict()
    config.Num_Classes = 47
    config.CropSize = 224

    config.Num_workers = 4
    config.pin_memory = True

    config.NUM_GPUS = 1
    return config

def get_eurosat_config():
    config = ml_collections.ConfigDict()
    config.Num_Classes = 10
    config.CropSize = 224

    config.Num_workers = 4
    config.pin_memory = True

    config.NUM_GPUS = 1
    return config

def get_kitti_config():
    config = ml_collections.ConfigDict()
    config.Num_Classes = 4
    config.CropSize = 224

    config.Num_workers = 4
    config.pin_memory = True

    config.NUM_GPUS = 1
    return config

def get_oxford_flowers102_config():
    config = ml_collections.ConfigDict()
    config.Num_Classes = 102
    config.CropSize = 224

    config.Num_workers = 4
    config.pin_memory = True

    config.NUM_GPUS = 1
    return config

def get_oxford_iiit_pet_config():
    config = ml_collections.ConfigDict()

    config.Num_Classes = 37
    config.CropSize = 224

    config.Num_workers = 4
    config.pin_memory = True

    config.NUM_GPUS = 1
    return config

def get_patch_camelyon_config():
    config = ml_collections.ConfigDict()
    config.Num_Classes = 2
    config.CropSize = 224

    config.Num_workers = 4
    config.pin_memory = True

    config.NUM_GPUS = 1
    return config

def get_resisc45_config():
    config = ml_collections.ConfigDict()
    config.Num_Classes = 45
    config.CropSize = 224

    config.Num_workers = 4
    config.pin_memory = True

    config.NUM_GPUS = 1
    return config

def get_smallnorb_azi_config():
    config = ml_collections.ConfigDict()

    config.Num_Classes = 18
    config.CropSize = 224

    config.Num_workers = 4
    config.pin_memory = True

    config.NUM_GPUS = 1
    return config

def get_smallnorb_ele_config():
    config = ml_collections.ConfigDict()

    config.Num_Classes = 9
    config.CropSize = 224

    config.Num_workers = 4
    config.pin_memory = True

    config.NUM_GPUS = 1
    return config

def get_sun397_config():
    config = ml_collections.ConfigDict()

    config.Num_Classes = 397
    config.CropSize = 224

    config.Num_workers = 4
    config.pin_memory = True

    config.NUM_GPUS = 1
    return config

def get_svhn_config():
    config = ml_collections.ConfigDict()
    config.Num_Classes = 10
    config.CropSize = 224

    config.Num_workers = 4
    config.pin_memory = True

    config.NUM_GPUS = 1
    return config


DATA_CONFIGS ={
    "caltech101": get_caltech101_config(),
    "cifar": get_cifar_config(),
    "clevr_count": get_clevr_count_config(),
    "clevr_dist": get_clevr_dist_config(),
    "diabetic_retinopathy": get_diabetic_retinopathy_config(),
    "dmlab": get_dmlab_config(),
    "dsprites_loc": get_dsprites_loc_config(),
    "dsprites_ori": get_dsprites_ori_config(),
    "dtd": get_dtd_config(),
    "eurosat": get_eurosat_config(),
    "kitti": get_kitti_config(),
    "oxford_flowers102": get_oxford_flowers102_config(),
    "oxford_iiit_pet": get_oxford_iiit_pet_config(),
    "patch_camelyon": get_patch_camelyon_config(),
    "resisc45": get_resisc45_config(),
    "smallnorb_azi": get_smallnorb_azi_config(),
    "smallnorb_ele": get_smallnorb_ele_config(),
    "sun397": get_sun397_config(),
    "svhn": get_svhn_config()
}