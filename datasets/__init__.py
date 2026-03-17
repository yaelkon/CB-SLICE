"""
Utility functions for data loading.
"""

from torch.utils.data import DataLoader

from .isic.isic_dataset import get_ISIC_dataloader
from .mnist_sum.mnist_sum_dataset import get_MNIST_SUM_dataset
from .waterbirds.waterbirds_dataset import get_Waterbirds_dataloader
from .celebA.celebA_dataset import get_CelebA_dataset
from .metaShiftCatDog.metaShiftCatDog_dataset import get_MetaShiftCatDog_dataloader


def get_data(config_base, config, gen):
    """
    Parse the configuration file and return the relevant dataset loaders.

    This function parses the provided configuration file and returns the appropriate dataset loaders based on the
    specified dataset type. It also sets the data path based on the hostname or the configuration file if working
    locally and on a cluster. The function supports synthetic datasets, CUB, CIFAR-10, and CIFAR-100 datasets.

    Args:
        config_base (dict): The base configuration dictionary.
        config (dict): The data configuration dictionary containing dataset and data path information.
        gen (object): A generator object to control the randomness of the data loader.

    Returns:
        tuple: A tuple containing the training data loader, validation data loader, and test data loader.
    """

    if config.dataset == "mnist_sum":
        print("MNIST_SUM DATASET")
        trainset, validset, testset = get_MNIST_SUM_dataset(
            config,
        )
    elif config.dataset == "Waterbirds":
        trainset, validset, testset = get_Waterbirds_dataloader(
            config,
        )
    elif config.dataset == "CelebA":
        trainset, validset, testset = get_CelebA_dataset(
            config,
            use_pickle=True,
        )
    elif config.dataset == "MetaShiftCatDog":
        trainset, validset, testset = get_MetaShiftCatDog_dataloader(
            config,
        )

    elif config.dataset == "ISIC":
        trainset, validset, testset = get_ISIC_dataloader(config)
    else:
        NotImplementedError("ERROR: Dataset not supported!")

    config = config_base
    train_loader = DataLoader(
        trainset,
        batch_size=config.model.train_batch_size,
        shuffle=True,
        num_workers=config.workers,
        pin_memory=True,
        generator=gen,
        drop_last=True,
    )
    val_loader = DataLoader(
        validset,
        batch_size=config.model.val_batch_size,
        shuffle=True,
        num_workers=config.workers,
        pin_memory=True,
        generator=gen,
    )
    test_loader = DataLoader(
        testset,
        batch_size=config.model.val_batch_size,
        shuffle=False,
        num_workers=config.workers,
        generator=gen,
    )

    return train_loader, val_loader, test_loader
