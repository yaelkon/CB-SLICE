import os
import numpy as np
import pandas as pd
import pickle

seed = 42
np.random.seed(seed)

folder_dir = os.path.expanduser("~/datasets/CUB/Waterbirds/processed")
metadata_name = "metadata_v3"
dataset_name = "val_v3"


output_dir = folder_dir
metadata_path = os.path.join(folder_dir, f"{metadata_name}.csv")
dataset_path = os.path.join(folder_dir, f"{dataset_name}.pkl")

val_dataset_df = pd.DataFrame(pd.read_pickle(dataset_path))
metadata_df = pd.read_csv(metadata_path)

waterbird_on_land_df = val_dataset_df[(val_dataset_df["y"] == 1) & (val_dataset_df["background_label"] == 0)].copy()
landbirds_on_water_df = val_dataset_df[(val_dataset_df["y"] == 0) & (val_dataset_df["background_label"] == 1)].copy()

print(f"Number of waterbirds on land samples before reduction: {len(waterbird_on_land_df)}")
print(f"Number of landbirds on water samples before reduction: {len(landbirds_on_water_df)}")
print("----")

waterbird_on_land_percentage = 0.06
landbird_on_water_percentage = 0.06

waterbirds_indices_to_drop = []
landbirds_indices_to_drop = []

for bird_type, group in waterbird_on_land_df.groupby("bird_type"):
    n_samples = len(group)
    n_to_drop = int(round(n_samples * (1 - waterbird_on_land_percentage)))

    if n_to_drop > 0:
        indices_to_drop = np.random.choice(group.index, size=n_to_drop, replace=False)
        waterbirds_indices_to_drop.extend(indices_to_drop)
        print(f"Dropping {n_to_drop}/{n_samples} samples from "
                  f"bird_type '{bird_type}' from validation")

print(f"Total waterbirds on land samples to drop: {len(waterbirds_indices_to_drop)}")
print("----")
for bird_type, group in landbirds_on_water_df.groupby("bird_type"):
    n_samples = len(group)
    n_to_drop = int(round(n_samples * (1 - landbird_on_water_percentage)))
   
    if n_to_drop > 0:
        indices_to_drop = np.random.choice(group.index, size=n_to_drop, replace=False)
        landbirds_indices_to_drop.extend(indices_to_drop)
        print(f"Dropping {n_to_drop}/{n_samples} samples from "
                  f"bird_type '{bird_type}' from validation")

print(f"Total landbirds on water samples to drop: {len(landbirds_indices_to_drop)}")
print("----")

waterbirds_images_to_drop = val_dataset_df.iloc[waterbirds_indices_to_drop]["image_path"]
landbirds_images_to_drop = val_dataset_df.iloc[landbirds_indices_to_drop]["image_path"]

val_dataset_df = val_dataset_df.drop(index=waterbirds_indices_to_drop)
val_dataset_df = val_dataset_df.drop(index=landbirds_indices_to_drop)

print(f"Number of waterbirds on land samples: {len(val_dataset_df[(val_dataset_df['y'] == 1) & (val_dataset_df['background_label'] == 0)])}")
print(f"Number of landbirds on water samples: {len(val_dataset_df[(val_dataset_df['y'] == 0) & (val_dataset_df['background_label'] == 1)])}")

# Save updated validation set
val_dataset_df.to_pickle(os.path.join(output_dir, f"{dataset_name}_reduced.pkl"))
# Update metadata
metadata_df.loc[metadata_df['img_filename'].isin(waterbirds_images_to_drop), 'split'] = -1  # -1 = removed
metadata_df.loc[metadata_df['img_filename'].isin(landbirds_images_to_drop), 'split'] = -1  # -1 = removed

# Save updated metadata
metadata_df.to_csv(os.path.join(output_dir, f"{metadata_name}_reduced.csv"), index=False)