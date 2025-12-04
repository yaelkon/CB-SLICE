#!/usr/bin/env python3
"""
Script to preprocess CelebA dataset and save as pickle files for faster loading.
This script creates pickle files containing data, labels, and concepts for train, val, and test splits.
"""

import os
import pandas as pd
import pickle
import argparse


def preprocess_celeba_data(root_dir, class_attribute="Male", output_dir=None):
    """
    Preprocess CelebA dataset and save as pickle files.

    Args:
        root_dir (str): Root directory containing CelebA dataset
        class_attribute (str): Class attribute to use for labels
        output_dir (str): Directory to save pickle files (default: same as root_dir)
    """
    if output_dir is None:
        output_dir = root_dir

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Paths
    celebA_dir = root_dir
    attributes_file = os.path.join(celebA_dir, "list_attr_celeba.csv")

    print(f"Loading attributes from: {attributes_file}")
    attributes_df = pd.read_csv(attributes_file)

    # Validate class attribute
    assert class_attribute in attributes_df.columns, (
        f"Class attribute must be in the attributes dataframe, "
        f"Received: {class_attribute}"
    )

    # Convert all columns except 'image_id' to 0/1
    print("Converting attributes to binary format...")
    cols_to_convert = [
        col for col in attributes_df.columns if col != "image_id"
    ]
    attributes_df[cols_to_convert] = attributes_df[cols_to_convert].map(
        lambda x: 1 if x == 1 else 0
    )

    # Create a lookup dictionary for faster access
    print("Creating attribute lookup dictionary...")
    attr_lookup = {}
    for _, row in attributes_df.iterrows():
        image_id = row["image_id"]
        label = row[class_attribute]
        concept_values = row.drop([class_attribute, "image_id"]).values
        attr_lookup[image_id] = (label, concept_values)

    # Process each split
    splits = ["train", "val", "test"]

    for split in splits:
        print(f"\nProcessing {split} split...")
        data_path = os.path.join(celebA_dir, split)

        if not os.path.exists(data_path):
            print(f"Warning: {split} directory not found at {data_path}, "
                  f"skipping...")
            continue

        data = []
        labels = []
        concepts = []

        # Get all jpg files in the split directory
        jpg_files = [f for f in os.listdir(data_path) if f.endswith(".jpg")]
        print(f"Found {len(jpg_files)} images in {split} split")

        # Process each image
        for i, file in enumerate(jpg_files):
            if i % 1000 == 0 and i > 0:
                print(f"  Processed {i}/{len(jpg_files)} images...")

            # Check if image has attributes
            if file in attr_lookup:
                data.append(file)
                label, concept_values = attr_lookup[file]
                labels.append(label)
                concepts.append(concept_values)
            else:
                print(f"  Warning: No attributes found for {file}")

        print(f"  Final count: {len(data)} images with attributes")

        # Save as pickle file
        output_file = os.path.join(output_dir, f"celeba_{split}_data.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump({
                'image_id': data,
                'labels': labels,
                'concepts': concepts,
                'class_attribute': class_attribute,
                'split': split
            }, f)

        print(f"  Saved {split} data to: {output_file}")

    print("\nPreprocessing complete!")


def main():    
    root_dir = "/home/yk449/datasets/CelebA/"
    class_attribute = "Male"
    output_dir = "/home/yk449/datasets/CelebA/CelebA_processed/"

    preprocess_celeba_data(
        root_dir=root_dir,
        class_attribute=class_attribute,
        output_dir=output_dir
    )


if __name__ == "__main__":
    main()
