"""
MetaShift dataset loader with concept labels. 

This module provides a custom DataLoader for the MetaShift dataset, including concept labels for training, validation, and testing.
The dataset is preprocessed with transformations.
"""

import os
import pickle
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class MetaShiftCatDog_DatasetGenerator(Dataset):
    """MetaShift Dataset object"""

    def __init__(
        self,
        data,
        stage,
        n_concepts,
        transform=None,
        subpopulations=None,
    ):

        super(MetaShiftCatDog_DatasetGenerator, self).__init__()
        self.dataset_name = "MetaShiftCatDog"
        self.stage = stage
        self.data = data
        self.n_concepts = n_concepts
        self.transform = transform

        self.classes = ["cat", "dog"]
        self.shift_attributes = ["indoor", "outdoor"]
        self.label_enum = {'cat': 0, 'dog': 1}
        self.concepts_semantics = CONCEPT_SEMANTICS

        self.subpopulations_dict = self._create_subpopulations_dict(subpopulations)

    def _create_subpopulations_dict(self, subpopulations):
        if subpopulations is None:
            return None
        subpopulations_dict = {}
        for i, s in enumerate(subpopulations):
            s = tuple(s)
            subpopulations_dict[s] = i
        return subpopulations_dict
    
    def get_population_idx(self, label, shift_attribute):
        """
        Get the subpopulation index based on class and shift attribute indices.
        """
        if self.subpopulations_dict is None:
            return None
        # Create a tuple of the class and shift attribute indices
        subpopulation_tuple = (label, shift_attribute)
        return self.subpopulations_dict[subpopulation_tuple], subpopulation_tuple[0] + "::" + subpopulation_tuple[1]
 
    def _add_shifted_concepts(self, concepts, shift_attribute):
        # Add the shifted concepts to the concepts array
        # [1, 0] - indoor
        # [0, 1] - outdoor
        if shift_attribute == "indoor":
            concepts = np.concatenate([concepts, [1, 0]])
        else:
            concepts = np.concatenate([concepts, [0, 1]])
        return concepts
    
    def __getitem__(self, index):
        # Gets an element of the dataset
        img_data = self.data[index]
        img_path = img_data["image_path"]
        img = Image.open(img_path).convert("RGB")
        # imageData = imageData.resize((224, 224))
        image_label = img_data["label"]
        y = self.label_enum[image_label]
        
        shift_attribute = img_data["group"]
        concepts = self._add_shifted_concepts(img_data["concept_annotations"], shift_attribute)

        population_idx, population_name = self.get_population_idx(image_label, shift_attribute)
        if self.transform is not None:
            img = self.transform(img)
        
        return {
            "img_code": index,
            "labels": y,
            "features": img,
            "concepts": concepts,
            "extra_dict": {
                "population_idx": population_idx,
                "population_name": population_name,
                "img_path": img_path,
            },
        }

    def __len__(self):
        return len(self.data)


def train_test_split_MetaShiftCatDog(root_dir):
    """Performs train-validation-test split on the MetaShift dataset."""
    data_train = pickle.load(open(os.path.join(root_dir, "train_metadata.pkl"), "rb"))
    data_val = pickle.load(open(os.path.join(root_dir, "val_metadata.pkl"), "rb"))
    
    return data_train, data_val


def get_MetaShiftCatDog_dataloader(config):
    """Returns a dictionary of data loaders for the MetaShift dataset, for the training, validation, and test sets."""
    data_train, data_val = train_test_split_MetaShiftCatDog(
        root_dir=config.data_path,
    )
    
    resol = 299
    resized_resol = int(resol * 256 / 224)

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(resized_resol),
            transforms.Resize(size=(224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # implicitly divides by 255
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.CenterCrop(resized_resol),
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),  # implicitly divides by 255
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image_datasets = {
        "train": MetaShiftCatDog_DatasetGenerator(
            data_train,
            stage="train",
            transform=train_transform,
            subpopulations=config.subpopulations,
            n_concepts=config.num_concepts,
        ),
        "val": MetaShiftCatDog_DatasetGenerator(
            data_val,
            stage="val",
            transform=test_transform,
            subpopulations=config.subpopulations,
            n_concepts=config.num_concepts,
        ),
        "test": MetaShiftCatDog_DatasetGenerator(
            data_val,
            stage="test",
            transform=test_transform,
            subpopulations=config.subpopulations,
            n_concepts=config.num_concepts,
        ),
    }

    return image_datasets["train"], image_datasets["val"], image_datasets["test"]



CONCEPT_SEMANTICS = [
        "Long snout",
        "Short snout",
        "Floppy ears",
        "Upright ears",
        "Round eyes",
        "Slit pupils",
        "Curled tail",
        "Straight tail",
        "Stocky body",
        "Slim body",
        "Wide muzzle",
        "Narrow muzzle",
        "Large nose",
        "Small nose",
        "Broad paws",
        "Small paws",
        "Stocky body",
        "Slim body",
        "Wide muzzle",
        "Narrow muzzle",
        "Large nose",
        "Small nose",
        "Broad paws",
        "Small paws",
        "Short, dense fur",
        "Fine, soft fur",
        "Simple or spotted coat",
        "Striped or marbled coat",
        "Short whiskers",
        "Long whiskers",
        "Expressive face",
        "Neutral face",
        "Square or upright posture",
        "Crouched or perched posture",
        "Indoor",
        "Outdoor",
]