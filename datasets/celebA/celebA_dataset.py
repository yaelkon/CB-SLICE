import os
import pandas as pd
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class CelebA_Dataset(Dataset):
    """ CelebA Dataset object that loads from preprocessed pickle files"""
    
    def __init__(
        self,
        root_dir,
        stage,
        class_attribute="Male",
        transform=None,
        subpopulations=None,
        use_pickle=True,
        selected_concepts=None,
        ):

        self.dataset_name = "CelebA"
        self.root_dir = os.path.join(root_dir, "CelebA")
        self.stage = stage
        self.data_path = os.path.join(self.root_dir, self.stage)
        self.class_attribute = class_attribute
        self.transform = transform
        self.subpopulations = subpopulations
        self.use_pickle = use_pickle
        self.selected_concepts = selected_concepts
        
        if selected_concepts is not None:
            self.concept_semantics = [CONCEPT_SEMANTICS[s] for s in selected_concepts]
        else:
            self.concept_semantics = [
                CONCEPT_SEMANTICS[i] for i, s in enumerate(CONCEPT_SEMANTICS) if s != class_attribute
            ]

        if use_pickle:
            self.data, self.labels, self.concepts = self._load_from_pickle()
        else:
            self.data, self.labels, self.concepts = self._create_data()

        self.subpopulations_dict, self.subpopulations_class_dict = self._create_subpopulations()

    def _create_subpopulations(self):
        if self.subpopulations is None:
            return None, None

        subpopulations_class_dict = {}
        subpopulations_dict = {}
        for i, s in enumerate(self.subpopulations):
            c = s[0]
            a = s[1]
            a_v = s[2]

            c_name = self.class_attribute if c == 1 else "Not " + self.class_attribute
            a_name = self.concept_semantics[a] if a_v == 1 else "Not " + self.concept_semantics[a]
            
            if c not in subpopulations_class_dict:
                subpopulations_class_dict[c] = {}
            if a not in subpopulations_class_dict[c]:
                subpopulations_class_dict[c][a] = {}
            subpopulations_class_dict[c][a][a_v] = i
            subpopulations_dict[c_name + "::" + a_name] = i

        return subpopulations_dict, subpopulations_class_dict

    def _load_from_pickle(self):
        """Load preprocessed data from pickle file"""
        pickle_file = os.path.join(*[self.root_dir, "CelebA_processed", f"celeba_{self.stage}_data.pkl"])
        
        if not os.path.exists(pickle_file):
            print(f"Pickle file not found: {pickle_file}")
            print("Falling back to original data creation method...")
            return self._create_data()
        
        print(f"Loading {self.stage} data from pickle file: {pickle_file}")
        with open(pickle_file, 'rb') as f:
            data_dict = pickle.load(f)
        
        # Verify class attribute matches
        if data_dict.get('class_attribute') != self.class_attribute:
            print(f"Warning: Class attribute mismatch. Pickle has "
                  f"'{data_dict.get('class_attribute')}', expected "
                  f"'{self.class_attribute}'")
            print("Falling back to original data creation method...")
            return self._create_data()
        
        return data_dict['data'], data_dict['labels'], data_dict['concepts']

    def _create_data(self):
        """Original data creation method (fallback)"""
        data = []
        labels = []
        concepts = []
        attributes_df = pd.read_csv(os.path.join(self.root_dir, "list_attr_celeba.csv"))
        assert self.class_attribute in attributes_df.columns, (
            "Class attribute must be in the attributes dataframe, "
            "Received: " + self.class_attribute
        )
        
        # Convert all columns except 'image_id' to 0/1
        cols_to_convert = [col for col in attributes_df.columns if col != "image_id"]
        attributes_df[cols_to_convert] = attributes_df[cols_to_convert].map(
            lambda x: 1 if x == 1 else 0
        )

        # iterate over the data_path
        for file in os.listdir(self.data_path):
            if file.endswith(".jpg"):
                data.append(file)
                concept_row = attributes_df.loc[attributes_df["image_id"] == file]
                labels.append(concept_row[self.class_attribute].values[0])
                # For each image, get all concept values except the class_attribute
                concept_values = concept_row.drop(
                    columns=[self.class_attribute, "image_id"]
                ).values.squeeze()
                concepts.append(concept_values)
        
        return data, labels, concepts
    
    def get_subpopulation_idx(self, label, concept):
        if self.subpopulations_dict is None:
            return -1, -1

        subpop_dict = self.subpopulations_class_dict[label]
        attribute_idx = list(subpop_dict.keys())[0]
        attribute_value = concept[attribute_idx]
        subpop_idx = subpop_dict[attribute_idx][attribute_value]

        for name, idx in self.subpopulations_dict.items():
            if idx == subpop_idx:
                subpop_name = name
                break

        return subpop_idx, subpop_name

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, self.data[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        
        label = self.labels[idx]
        concept = np.array(self.concepts[idx], dtype=np.int64)

        if self.selected_concepts is not None:
            concept = concept[self.selected_concepts]

        subpop_idx, subpop_name = self.get_subpopulation_idx(label, concept)

        return {
            "img_code": idx,
            "labels": label,
            "features": image,
            "concepts": concept,
            "extra_dict": {
                "population_idx": subpop_idx,
                "population_name": subpop_name,
                "img_path": img_path,
            },
        }
           
    def __len__(self):
        return len(self.data)


def get_CelebA_dataset(config, use_pickle=True):
    """Get CelebA dataset with optional pickle optimization"""
    if config.selected_concepts is not None:
        assert len(config.selected_concepts) == config.num_concepts, "Selected concepts must be the same length as the number of concepts"

    train_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.RandomRotation(degrees=20),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # Datasets
    image_datasets = {
        "train": CelebA_Dataset(
            root_dir=config.data_path,
            stage="train",
            class_attribute=config.class_attribute,
            transform=train_transform,
            subpopulations=config.subpopulations,
            use_pickle=use_pickle,
            selected_concepts=config.selected_concepts,
        ),
        "val": CelebA_Dataset(
            root_dir=config.data_path,
            stage="val",
            class_attribute=config.class_attribute,
            transform=test_transform,
            subpopulations=config.subpopulations,
            use_pickle=use_pickle,
            selected_concepts=config.selected_concepts,
        ),
        "test": CelebA_Dataset(
            root_dir=config.data_path,
            stage="test",
            class_attribute=config.class_attribute,
            transform=test_transform,
            subpopulations=config.subpopulations,
            use_pickle=use_pickle,
            selected_concepts=config.selected_concepts,
        ),
    }

    return (
        image_datasets["train"],
        image_datasets["val"],
        image_datasets["test"],
    )


CONCEPT_SEMANTICS = [
    '5_o_Clock_Shadow', # 0
    'Arched_Eyebrows', # 1
    'Attractive', # 2
    'Bags_Under_Eyes', # 3
    'Bald', # 4
    'Bangs', # 5
    'Big_Lips', # 6
    'Big_Nose', # 7
    'Black_Hair', # 8
    'Blond_Hair', # 9
    'Blurry', # 10
    'Brown_Hair', # 11
    'Bushy_Eyebrows', # 12
    'Chubby', # 13
    'Double_Chin', # 14
    'Eyeglasses', # 15
    'Goatee', # 16
    'Gray_Hair', # 17
    'Heavy_Makeup', # 18
    'High_Cheekbones', # 19
    # 'Male', # 20
    'Mouth_Slightly_Open', # 21
    'Mustache', # 22
    'Narrow_Eyes', # 23
    'No_Beard', # 24
    'Oval_Face', # 25
    'Pale_Skin', # 26
    'Pointy_Nose', # 27
    'Receding_Hairline', # 28
    'Rosy_Cheeks', # 29
    'Sideburns', # 30
    'Smiling', # 31
    'Straight_Hair', # 32
    'Wavy_Hair', # 33
    'Wearing_Earrings', # 34
    'Wearing_Hat', # 35
    'Wearing_Lipstick', # 36
    'Wearing_Necklace', # 37
    'Wearing_Necktie', # 38
    'Young', # 39
]


def denormalize_tensor(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize a tensor that was normalized with ImageNet statistics.
    
    Args:
        tensor: Normalized tensor of shape [C, H, W]
        mean: Original mean used for normalization
        std: Original std used for normalization
    
    Returns:
        Denormalized tensor of shape [H, W, C]
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    
    # Denormalize
    tensor = tensor * std + mean
    
    # Clamp values to [0, 1] range
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert from [C, H, W] to [H, W, C]
    tensor = tensor.permute(1, 2, 0)
    
    return tensor


def plot_preprocessed_samples(config, num_samples=6, save_path=None):
    """
    Plot random preprocessed samples from train, val, and test sets for debugging.
    For each sample, also display the concept semantics (names) for which the value is 1.
    
    Args:
        config: Configuration object containing data_path, class_attribute, etc.
        num_samples: Number of samples to plot from each set
        save_path: Optional path to save the plot
    """
    # Get datasets
    train_dataset, val_dataset, test_dataset = get_CelebA_dataset(config, use_pickle=True)
    # Print the number of samples in each dataset
    print(f"Number of samples in train dataset: {len(train_dataset)}")
    print(f"Number of samples in val dataset: {len(val_dataset)}")
    print(f"Number of samples in test dataset: {len(test_dataset)}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, num_samples, figsize=(15, 9))
    fig.suptitle('Preprocessed Samples from Train/Val/Test Sets', fontsize=16)
    
    datasets = [
        (train_dataset, "Train"),
        (val_dataset, "Validation"), 
        (test_dataset, "Test")
    ]
    
    # Ensure CONCEPT_SEMANTICS is a list in the correct order
    concept_semantics_list = list(CONCEPT_SEMANTICS)
    
    for row, (dataset, title) in enumerate(datasets):
        # Get random indices
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        
        for col, idx in enumerate(indices):
            # Get sample
            sample = dataset[idx]
            image = sample['features']
            label = sample['labels']
            concept = sample['concepts']
            
            # Denormalize the image
            if isinstance(image, torch.Tensor):
                image_denorm = denormalize_tensor(image)
                image_np = image_denorm.numpy()
            else:
                image_np = image

            # Get concept names with value 1
            if isinstance(concept, np.ndarray):
                concept_indices = np.where(concept == 1)[0]
            else:
                concept_indices = [i for i, v in enumerate(concept) if v == 1]
            concept_names = [concept_semantics_list[i] for i in concept_indices]
            concept_str = ', '.join(concept_names)

            # Plot image
            axes[row, col].imshow(image_np)
            # Split concept_str into lines of max 5 concepts per line
            concept_names_split = concept_names
            concept_lines = [', '.join(concept_names_split[i:i+5]) for i in range(0, len(concept_names_split), 5)]
            concept_str_multiline = '\n'.join(concept_lines)
            axes[row, col].set_title(f'{title}\nLabel: {label}\nConcepts: {concept_str_multiline}', fontsize=8)
            axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    

if __name__ == "__main__":
    # Example usage for debugging
    from omegaconf import OmegaConf
    np.random.seed(42)
    # Load config
    config = OmegaConf.load("configs/data/celebA.yaml")
    
    # Plot samples
    plot_preprocessed_samples(config, num_samples=2, save_path="debug_samples.png")
