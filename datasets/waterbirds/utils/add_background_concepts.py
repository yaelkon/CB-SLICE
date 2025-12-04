import os
import numpy as np
import pandas as pd
import pickle

seed = 42
np.random.seed(seed)

split_index_dict = {"train": 0, "val": 1, "test": 2}

metadata_path = os.path.expanduser("~/datasets/CUB/Waterbirds/processed/metadata_v3.csv")
output_dir = os.path.expanduser("~/datasets/CUB/Waterbirds/processed/")
pickle_path = os.path.expanduser("~/datasets/CUB/Waterbirds/processed/train_v3.pkl")
saving_name = "train_v3.pkl"

split = saving_name.split("_")[0]
split_index = split_index_dict[split]
dataset_df = pd.DataFrame(pd.read_pickle(pickle_path))
metadata_df = pd.read_csv(metadata_path)
metadata_df = metadata_df[metadata_df["split"] == split_index]

places = metadata_df["place_filename"].str.split("/")
places = places.str.get(2).values
dataset_df["background_concept_semantic"] = places

dataset_df.to_pickle(os.path.join(output_dir, saving_name))
print(f"Dataset saved to {os.path.join(output_dir, saving_name)}")