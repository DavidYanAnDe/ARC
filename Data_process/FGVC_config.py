import ml_collections

def get_dogs_config():
    config = ml_collections.ConfigDict()
    config.Name = "StanfordDogs"
    config.Num_Classes = 120
    config.CropSize = 224

    config.Num_workers = 4
    config.pin_memory = True

    config.NUM_GPUS = 1
    return config


def get_flowers_config():
    config = ml_collections.ConfigDict()
    config.Name = "OxfordFlowers"
    config.Num_Classes = 102
    config.CropSize = 224

    config.Num_workers = 4
    config.pin_memory = True

    config.NUM_GPUS = 1
    return config


def get_CUB_config():
    config = ml_collections.ConfigDict()
    config.Name = "CUB_200_2011"
    config.Num_Classes = 200
    config.CropSize = 224

    config.Num_workers = 4
    config.pin_memory = True

    config.NUM_GPUS = 1
    return config


def get_Cars_config():
    config = ml_collections.ConfigDict()
    config.Name = "StanfordCars"
    config.Num_Classes = 196
    config.CropSize = 224

    config.Num_workers = 4
    config.pin_memory = True

    config.NUM_GPUS = 1
    return config


def get_NABirds_config():
    config = ml_collections.ConfigDict()
    config.Name = "NABirds"
    config.Num_Classes = 555
    config.CropSize = 224

    config.Num_workers = 4
    config.pin_memory = True

    config.NUM_GPUS = 1
    return config

DATA_CONFIGS ={
    "CUB_200_2011": get_CUB_config(),
    "NABirds": get_NABirds_config(),
    "OxfordFlowers": get_flowers_config(),
    "StanfordCars": get_Cars_config(),
    "StanfordDogs": get_dogs_config()
}