#!/usr/bin/env python3

"""Data_process loader."""
import logging
import torch

from torch.utils.data.sampler import RandomSampler, SequentialSampler

from Data_process.json_dataset import CUB200Dataset, CarsDataset, DogsDataset, FlowersDataset, NabirdsDataset

logger = logging.getLogger(__name__)

_DATASET_CATALOG = {
    "CUB_200_2011": CUB200Dataset,
    'OxfordFlowers': FlowersDataset,
    'StanfordCars': CarsDataset,
    'StanfordDogs': DogsDataset,
    "NABirds": NabirdsDataset,
}


def _construct_loader(cfg, split, batch_size, shuffle, drop_last, data_path):
    """Constructs the data loader for the given dataset."""
    dataset_name = cfg.Name

    assert (
        dataset_name in _DATASET_CATALOG.keys()
    ), "Dataset '{}' not supported".format(dataset_name)
    dataset = _DATASET_CATALOG[dataset_name](cfg, split, data_path)

    # Create a sampler for training
    if "train" in split:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=cfg.Num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=drop_last,
    )
    return loader


def construct_train_loader(cfg, batch_size, data_path):
    """Train loader wrapper."""
    drop_last = False

    return _construct_loader(
        cfg=cfg,
        split="train",
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        data_path=data_path
    )

def construct_trainval_loader(cfg, batch_size, data_path):
    """Train loader wrapper."""
    drop_last = False
    return _construct_loader(
        cfg=cfg,
        split="trainval",
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        data_path=data_path
    )

def construct_test_loader(cfg, batch_size, data_path):
    """Test loader wrapper."""
    return _construct_loader(
        cfg=cfg,
        split="test",
        batch_size= batch_size,
        shuffle=False,
        drop_last=False,
        data_path=data_path
    )


def construct_val_loader(cfg, batch_size, data_path):
    """Validation loader wrapper."""
    return _construct_loader(
        cfg=cfg,
        split="val",
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        data_path=data_path
    )
