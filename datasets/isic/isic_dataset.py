
# datasets/isic/isic_dataset.py

import os
import pandas as pd
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode



class ISIC_DatasetGenerator(Dataset):
    
    def __init__(self, data_df, stage, concept_cols, root_data_path, transform=None, subpopulations=None):

        super().__init__()
        self.dataset_name = "ISIC"
        self.stage = stage
        self.data = data_df.reset_index(drop=True)
        self.concept_cols = concept_cols
        self.transform = transform
        self.root_data_path = root_data_path
        self.concepts_semantics = concept_cols  # optional
        self.subpopulations_dict = self._create_subpopulations_dict(subpopulations)
        self.spurious_cols = ["dark_corner","gel_border","ruler","ink","patches","hair","gel_bubble"]
        self.active_spurious = None  # set via dataset config if needed



    def __len__(self):
        return len(self.data)

    def _create_subpopulations_dict(self, subpopulations):
        if subpopulations is None:
            return None
        d = {}
        for i, s in enumerate(subpopulations):
            d[tuple(s)] = i   
        return d

    def get_population_idx(self, y, concept_name):
        if self.subpopulations_dict is None:
            return None, None
        label_name = "malignant" if y == 1 else "benign"
        key = (label_name, concept_name)
        if key not in self.subpopulations_dict:
            return None, None
        idx = self.subpopulations_dict[key]
        pop_name = f"{label_name}::{concept_name}"
        return idx, pop_name



    def __getitem__(self, index):
        row = self.data.iloc[index]

        img_path = os.path.join(self.root_data_path, row["image_path"])
        img = Image.open(img_path).convert("RGB")

        y = int(row["y"])      
        c = row[self.concept_cols].to_numpy(dtype=np.int64)

        sp = row[self.spurious_cols].to_numpy(dtype=np.int64)

        hair = int(row["hair"])
        gel_border  = int(row["gel_border"]) 
        gel_bubble = int(row["gel_bubble"])
        dark_corner = int(row["dark_corner"])
        ruler = int(row["ruler"])
        ink = int(row["ink"])
        patches = int(row["patches"])

        active_spurious = getattr(self, 'active_spurious', None)
        if active_spurious is not None:
            val = int(row[active_spurious])
            concept_name = active_spurious if val == 1 else f"no_{active_spurious}"
        else:
            active = [name for name, v in zip(self.spurious_cols, sp) if v == 1]
            concept_name = active[0] if len(active) > 0 else "no_spurious"

        population_idx, population_name = self.get_population_idx(y, concept_name)

        if self.transform is not None:
            img = self.transform(img)

        return {
            "img_code": index,
            "labels": y,
            "features": img,
            "concepts": c,
            "extra_dict": {
                "population_idx": population_idx if population_idx is not None else 0,
                "population_name": population_name if population_name is not None else "none",
                "img_path": img_path,
                "image_id": row["image_id"],
            },
        }


class ResizeLongestSidePadSquare:
    def __init__(self, size: int, fill=0, interpolation=InterpolationMode.BILINEAR):
        self.size = size
        self.fill = fill
        self.interpolation = interpolation

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = self.size / max(w, h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        img = TF.resize(img, [new_h, new_w], interpolation=self.interpolation)

        pad_w = self.size - new_w
        pad_h = self.size - new_h
        # left, top, right, bottom
        padding = [pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2]
        img = TF.pad(img, padding, fill=self.fill)
        return img


def get_ISIC_dataloader(config):

    train_df = pd.read_csv(os.path.join(config.root_data_path, config.train_csv))
    val_df   = pd.read_csv(os.path.join(config.root_data_path, config.val_csv))
    test_df  = pd.read_csv(os.path.join(config.root_data_path, config.test_csv))

    concept_cols = list(config.concept_cols)
    IMAGENET_MEAN = [0.485, 0.456, 0.406] 
    IMAGENET_STD = [0.229, 0.224, 0.225]

    IMAGE_SIZE = 1024  

    train_transform = transforms.Compose([
        ResizeLongestSidePadSquare(IMAGE_SIZE, fill=0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    test_transform = transforms.Compose([
        ResizeLongestSidePadSquare(IMAGE_SIZE, fill=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


    active_spurious = config.get("active_spurious", None)

    train_dataset = ISIC_DatasetGenerator(
        data_df=train_df,
        stage="train",
        concept_cols=concept_cols,
        root_data_path=config.root_data_path,
        transform=train_transform,
        subpopulations=config.subpopulations,
    )
    train_dataset.active_spurious = active_spurious

    val_dataset = ISIC_DatasetGenerator(
        data_df=val_df,
        stage="val",
        concept_cols=concept_cols,
        root_data_path=config.root_data_path,
        transform=test_transform,
        subpopulations=config.subpopulations,
    )
    val_dataset.active_spurious = active_spurious

    test_dataset = ISIC_DatasetGenerator(
        data_df=test_df,
        stage="test",
        concept_cols=concept_cols,
        root_data_path=config.root_data_path,
        transform=test_transform,
        subpopulations=config.subpopulations,
    )
    test_dataset.active_spurious = active_spurious

    return train_dataset, val_dataset, test_dataset