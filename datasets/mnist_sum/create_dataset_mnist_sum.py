from typing import Optional, Dict
from torchvision.datasets import MNIST
import numpy as np
import torch
import random
from tqdm import tqdm


class MNIST_Dataloader(MNIST):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, train=train, transform=transform, download=False)

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return {
            "img_code": index,
            "labels": img,
            "features": target,
            "concepts": None,
        }


def create_mnist_sum_dataset(
        digits_range: np.ndarray, 
        root_path: str, 
        save_path: str, 
        stage: str = 'train',
        n_samples: int = 60000, 
        required_distribution: Optional[np.ndarray] = None,
        minority_attribute: Optional[Dict[tuple, dict]] = None,
        data_filename: str = "mnist_sum_concepts.npz",
        seed=42,
        ):
    
    assert stage in ["train", "val", "test"], "Stage should be either 'train', 'val' or 'test'"
    is_train = stage in ["train", "val"]

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    minority_attribute_proportion = {}
    if minority_attribute is not None:
        for key in minority_attribute.keys():
            minority_attribute_proportion[key] = 0.0

    n_dists = len(digits_range) ** 2
    dist_hist = np.zeros((len(digits_range), len(digits_range))) 

    if required_distribution is None:
        equal_dist = (n_samples // n_dists)
        required_distribution = np.ones((len(digits_range), len(digits_range))) * equal_dist
    
    # Create a dataset for the specified digits
    dataloader = MNIST(
        root=root_path,
        train=is_train,
        download=False,
    )

    imgs, labels, concepts, attributes, origin_concepts, origin_labels = [], [], [], [], [], []
    n_created_train = 0
    img1, img2 = None, None

    pbar = tqdm(total=n_samples, desc="Creating training dataset")
    while n_created_train < n_samples:
        # Get a random sample of images and labels from the dataset
        img, target = dataloader[np.random.randint(len(dataloader))]

        # Check if the target is in the specified digits range
        if target not in digits_range:
            continue

        # If we have two images, create a new image by concatenating them
        if img1 is None:
            img1 = img
            target1 = target
            continue

        elif img2 is None:
            img2 = img
            target2 = target
        
        else:
            raise ValueError("img1 and img2 should be None at this point")

        if dist_hist[target1, target2] >= required_distribution[target1, target2]:
            img1 = img2 = None
            # print("Skipping pair due to histogram limit")
            # print(dist_hist)
            continue

        # Create a new image by concatenating the two images
        img = np.concatenate((img1, img2), axis=1)
        img = np.stack((img, ) * 3, axis=-1)  # Convert to RGB
        label = target1 + target2
        orig_label = label
        concept = np.zeros(2 * len(digits_range))
        concept[target1] = 1
        concept[len(digits_range) + target2] = 1
        origin_concept = concept.copy()

        if minority_attribute is not None:
            attribute = 'none'
            if (target1, target2) in minority_attribute:
                minority_proportion = minority_attribute[(target1, target2)]["proportion"]
                apply_attribute = np.random.binomial(n=1, p=minority_proportion)
                 
                if apply_attribute:
                    attribute = minority_attribute[(target1, target2)]["attribute"]
                    minority_attribute_proportion[(target1, target2)] += 1
                    if attribute == "red":
                        # Apply the red attribute to the image
                        digits_indices = np.where(img[:, :, 0] > 0)
                        img[digits_indices[0], digits_indices[1], 1:] = 0
                    
                    elif attribute == "colorful":
                        # Turn the digits to be colorful
                        digits_indices = np.where(img[:, :, 0] > 0)
                        random_colors = np.random.randint(1, 255, size=(len(digits_indices[0]), 3))
                        img[digits_indices[0], digits_indices[1], :] = random_colors
                        # Turn the background to white
                        background_indices = np.where(img[:, :, 0] == 0)
                        img[background_indices[0], background_indices[1], :] = 255
                    elif attribute == "concept_noise":
                        concept = np.zeros(2 * len(digits_range))
                        # Randomly select two different digits from the digits range
                        t1 = target1
                        while t1 == target1:
                            t1 = np.random.randint(0, len(digits_range))
                        t2 = target2
                        while t2 == target2:
                            t2 = np.random.randint(0, len(digits_range))
                        concept[t1] = 1
                        concept[len(digits_range) + t2] = 1

                    elif attribute == "label_noise":
                        # Randomly select a new label for the image
                        label = np.random.randint(digits_range[0], digits_range[-1] * 2 + 1)
                        while label == orig_label:
                            label = np.random.randint(digits_range[0], digits_range[-1] * 2 + 1)
                    else:
                        raise ValueError(f"Unknown attribute {attribute} for index {target1}-{target2}")
            attributes.append(attribute)

        imgs.append(img)
        labels.append(label)
        concepts.append(concept)
        origin_concepts.append(origin_concept)
        origin_labels.append(orig_label)

        dist_hist[target1, target2] += 1
        img1, img2 = None, None
        n_created_train += 1
        pbar.update(1)

    # Convert lists to numpy arrays
    imgs = np.array(imgs)
    labels = np.array(labels)
    concepts = np.array(concepts)
    origin_concepts = np.array(origin_concepts)
    origin_labels = np.array(origin_labels)

    # Save the training dataset
    full_save_path = save_path + f"{stage}_{data_filename}"

    print(f"Saving dataset to {full_save_path}")
    if len(attributes) > 0:
        # Convert attributes to numpy array
        attributes = np.array(attributes)
        print(f"Saving dataset with attributes:")
        for key, value in minority_attribute_proportion.items():
            attribute = minority_attribute[key]["attribute"]
            print(f"Minority {key}-{attribute}: {value / required_distribution[key[0], key[1]]} samples")
        np.savez(
            full_save_path,
            imgs=imgs,
            labels=labels,
            concepts=concepts,
            origin_concepts=origin_concepts,
            origin_labels=origin_labels,
            attributes=attributes,
            allow_pickle=True,
        )
    else:
        np.savez(
            full_save_path,
            imgs=imgs,
            labels=labels,
            concepts=concepts,
            origin_concepts=origin_concept,
            origin_labels=origin_labels,
            allow_pickle=True,
        )

    pbar.close()
    print(f"Subpopulation histogram: {dist_hist}")


if __name__ == "__main__":
    digits_range = np.array([0, 1, 2, 3])
    save_path = "/home/yk449/datasets/mnist/mnist_sum/1_11_2025/"
    root_path = "/home/yk449/datasets/mnist/"
    
    data_filename = "concepts-(2,2):rare_01-(1,1):red_09-(0,3):red_01_samples:3775.npz"

    stage = 'val'
    if stage == 'train':
        print("Creating training dataset")
        n_samples = 3775
        required_distribution = np.array([[250, 250, 250, 250],
                                          [250, 250, 250, 250],
                                          [250, 250, 25, 250],
                                          [250, 250, 250, 250],
                                          ])

        minority_attribute = {
        (1, 1): {
            "attribute": "red",
            "proportion": 0.9,
        },
        (0, 3): {
            "attribute": "red",
            "proportion": 0.1,
        },
        }
        create_mnist_sum_dataset(
            digits_range=digits_range,
            root_path=root_path,
            save_path=save_path,
            stage=stage,
            n_samples=n_samples,
            required_distribution=required_distribution,
            minority_attribute=minority_attribute,
            data_filename=data_filename,
            seed=42
            )
    else:
        print("Creating testing dataset")
        n_samples = 3775
        required_distribution = np.array([[250, 250, 250, 250],
                                          [250, 250, 250, 250],
                                          [250, 250, 25, 250],
                                          [250, 250, 250, 250],
                                          ])
        minority_attribute = {
        (1, 1): {
            "attribute": "red",
            "proportion": 0.9,
        },
        (0, 3): {
            "attribute": "red",
            "proportion": 0.1,
        },
        }

        create_mnist_sum_dataset(
            digits_range=digits_range,
            required_distribution=required_distribution,
            root_path=root_path,
            save_path=save_path,
            stage=stage,
            n_samples=n_samples,
            data_filename=data_filename,
            minority_attribute=minority_attribute,
            seed=123,
            )
