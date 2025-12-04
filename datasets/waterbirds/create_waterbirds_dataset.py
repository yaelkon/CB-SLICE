import os
import pandas as pd
import pickle
import numpy as np


# Function to load concept labels
def load_concepts(concepts_path):
    concept_files = [f for f in os.listdir(concepts_path) if f.endswith(".pkl")]
    concepts = {}
    for file in concept_files:
        with open(os.path.join(concepts_path, file), "rb") as f:
            data = pickle.load(f)
            for d in data:
                image_name = d["img_path"].split("/")[-2:]
                image_name = "/".join(image_name)
                concepts[image_name] = d
    return concepts


def redistribute_test_to_validation(metadata_df, concept_labels,
                                    test_to_val_percentage, random_seed=42):
    """
    Redistribute a percentage of test samples to validation set.

    Args:
        metadata_df: DataFrame with the original metadata
        concept_labels: Dictionary with concept labels for each image
        test_to_val_percentage: Percentage (0-100) of test samples to move to validation
        random_seed: Random seed for reproducibility

    Returns:
        Updated metadata DataFrame with new split assignments
    """
    np.random.seed(random_seed)

    # Create a copy of metadata to avoid modifying the original
    updated_metadata = metadata_df.copy()

    # Get test samples
    test_samples = updated_metadata[updated_metadata['split'] == 2].copy()

    if len(test_samples) == 0:
        print("No test samples found!")
        return updated_metadata

    # Add bird_type information to test samples
    test_samples['bird_type'] = test_samples['img_filename'].map(
        lambda x: concept_labels.get(x, {}).get('class_label', 'unknown')
    )

    # Group by bird_type and sample the specified percentage from each group
    samples_to_move = []

    for bird_type, group in test_samples.groupby('bird_type'):
        n_samples = len(group)
        n_to_move = int(np.ceil(n_samples * test_to_val_percentage))

        if n_to_move > 0:
            # Randomly select samples to move
            indices_to_move = np.random.choice(
                group.index, size=n_to_move, replace=False)
            samples_to_move.extend(indices_to_move)

            print(f"Moving {n_to_move}/{n_samples} samples from "
                  f"bird_type '{bird_type}' to validation")

    # Update split assignments
    updated_metadata.loc[samples_to_move, 'split'] = 1  # 1 = validation

    print(f"Total samples moved from test to validation: "
          f"{len(samples_to_move)}")

    return updated_metadata


def main():
    # Parse command line arguments
    test_to_val_percentage = 1
    random_seed = 42
    print(f"Moving {test_to_val_percentage}% of test samples to validation set")
    # Paths
    metadata_path = os.path.expanduser("~/datasets/CUB/Waterbirds/waterbird/metadata.csv")
    concepts_path = os.path.expanduser("~/datasets/CUB/CUB_processed/class_attr_data_10/")
    output_dir = os.path.expanduser("~/datasets/CUB/Waterbirds/processed/")
    postfix = "_v3"
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Load metadata
    metadata = pd.read_csv(metadata_path)
    # Load concept labels
    concept_labels = load_concepts(concepts_path)

    # Redistribute test samples to validation if requested
    if test_to_val_percentage > 0:
        metadata = redistribute_test_to_validation(
            metadata, concept_labels, test_to_val_percentage, random_seed
        )

    # Save updated metadata
    updated_metadata_path = os.path.join(output_dir, f"metadata{postfix}.csv")
    metadata.to_csv(updated_metadata_path, index=False)
    print(f"Updated metadata saved to: {updated_metadata_path}")

    # Prepare splits
    splits = {0: "train", 1: "val", 2: "test"}
    split_data = {name: [] for name in splits.values()}
    data_stats = []

    for _, row in metadata.iterrows():
        split = row["split"]
        image_path = row["img_filename"]

        # Collect concept labels for the current image
        image_concepts = concept_labels[image_path]["attribute_label"]
        bird_type = concept_labels[image_path]["class_label"]
        # Create data object
        data_object = {
            "y": row["y"],
            "bird_type": bird_type,
            "image_path": image_path,
            "background_label": row["place"],
            "concepts": image_concepts,
            "split": splits[split],
        }
        data_stats.append(data_object)
        split_data[splits[split]].append(data_object)

    data_stats_df = pd.DataFrame(data_stats)
    for split_name, data in split_data.items():
        data = pd.DataFrame(data)
        output_path = os.path.join(output_dir, f"{split_name}{postfix}.pkl")
        data.to_pickle(output_path)
        print(f"{split_name.capitalize()} dataset saved to {output_path}")

    # Print distribution statistics
    print("\n=== Dataset Distribution ===")
    class_distribution = data_stats_df.groupby(['split', 'y']).size().unstack(
        fill_value=0)
    print("Class distribution by split:")
    print(class_distribution)

    bkg_class_distribution = data_stats_df.groupby(
        ['split', 'background_label', 'y']).size().unstack(fill_value=0)
    print("\nBackground-Class distribution by split:")
    print(bkg_class_distribution)

    bird_type_bkg_distribution = data_stats_df.groupby(
        ['bird_type', 'background_label']).size().unstack(fill_value=0)
    print("\nBird type-Background distribution:")
    print(bird_type_bkg_distribution)

    bird_type_split_distribution = data_stats_df.groupby(
        ['bird_type', 'split']).size().unstack(fill_value=0)
    print("\nBird type-Split distribution:")
    print(bird_type_split_distribution)

    bird_type_bkg_split_distribution = data_stats_df.groupby(
        ['bird_type', 'background_label', 'split']).size().unstack(fill_value=0)
    print("\nBird type-Background-Split distribution:")
    print(bird_type_bkg_split_distribution)


if __name__ == "__main__":
    main()
