import os
import pandas as pd


print("CelebA dataset statistics:")
train_attributes_path = os.path.expanduser("~/datasets/CelebA/CelebA_processed/celeba_train_data.pkl")
train_attributes_df = pd.DataFrame(pd.read_pickle(train_attributes_path))

val_attributes_path = os.path.expanduser("~/datasets/CelebA/CelebA_processed/celeba_val_data.pkl")
val_attributes_df = pd.DataFrame(pd.read_pickle(val_attributes_path))

test_attributes_path = os.path.expanduser("~/datasets/CelebA/CelebA_processed/celeba_test_data.pkl")
test_attributes_df = pd.DataFrame(pd.read_pickle(test_attributes_path))
# rename the column "data" to "image_id"
train_attributes_df.rename(columns={"data": "image_id"}, inplace=True)
val_attributes_df.rename(columns={"data": "image_id"}, inplace=True)
test_attributes_df.rename(columns={"data": "image_id"}, inplace=True)

attributes_list_path = os.path.expanduser("~/datasets/CelebA/list_attr_celeba.csv")
attributes_list_df = pd.read_csv(attributes_list_path)
# change -1 to 0
attributes_list_df.replace(-1, 0, inplace=True)

joint_train_df = pd.merge(train_attributes_df, attributes_list_df, on="image_id")
joint_val_df = pd.merge(val_attributes_df, attributes_list_df, on="image_id")
joint_test_df = pd.merge(test_attributes_df, attributes_list_df, on="image_id")

# Compute blond/non-blond male/female counts for each split
def get_blond_counts(df, split_name):
    # Blond = 1, Male = 1
    blond_male = df[(df["Male"] == 1) & (df["Blond_Hair"] == 1)].shape[0]
    nonblond_male = df[(df["Male"] == 1) & (df["Blond_Hair"] == 0)].shape[0]
    blond_female = df[(df["Male"] == 0) & (df["Blond_Hair"] == 1)].shape[0]
    nonblond_female = df[(df["Male"] == 0) & (df["Blond_Hair"] == 0)].shape[0]
    total = blond_male + nonblond_male + blond_female + nonblond_female
    # Avoid division by zero
    def pct(count):
        return (count / total * 100) if total > 0 else 0.0
    return pd.DataFrame([{
        "blond_male": blond_male,
        "nonblond_male": nonblond_male,
        "blond_female": blond_female,
        "nonblond_female": nonblond_female,
        "blond_male_pct": pct(blond_male),
        "nonblond_male_pct": pct(nonblond_male),
        "blond_female_pct": pct(blond_female),
        "nonblond_female_pct": pct(nonblond_female),
        "split": split_name
    }])

counts_train = get_blond_counts(joint_train_df, "train")
counts_val = get_blond_counts(joint_val_df, "val")
counts_test = get_blond_counts(joint_test_df, "test")

split_blond_counts_df = pd.concat([counts_train, counts_val, counts_test], ignore_index=True)

print(split_blond_counts_df)

print("MetaShiftCatDog dataset statistics:")
train_attributes_path = os.path.expanduser("~/datasets/MetaShift/Cat-Dog-indoor-outdoor/train_metadata.pkl")
train_attributes_df = pd.DataFrame(pd.read_pickle(train_attributes_path))

val_attributes_path = os.path.expanduser("~/datasets/MetaShift/Cat-Dog-indoor-outdoor/val_metadata.pkl")
val_attributes_df = pd.DataFrame(pd.read_pickle(val_attributes_path))

def get_indoor_outdoor_counts(df, split_name):
    indoor_cat = df[(df["group"] == "indoor") & (df["label"] == "cat")].shape[0]
    outdoor_cat = df[(df["group"] == "outdoor") & (df["label"] == "cat")].shape[0]
    indoor_dog = df[(df["group"] == "indoor") & (df["label"] == "dog")].shape[0]
    outdoor_dog = df[(df["group"] == "outdoor") & (df["label"] == "dog")].shape[0]
    total = indoor_cat + outdoor_cat + indoor_dog + outdoor_dog
    # Avoid division by zero
    def pct(count):
        return (count / total * 100) if total > 0 else 0.0
    return pd.DataFrame([{
        "indoor_cat": indoor_cat,
        "outdoor_cat": outdoor_cat,
        "indoor_dog": indoor_dog,
        "outdoor_dog": outdoor_dog,
        "indoor_cat_pct": pct(indoor_cat),
        "outdoor_cat_pct": pct(outdoor_cat),
        "indoor_dog_pct": pct(indoor_dog),
        "outdoor_dog_pct": pct(outdoor_dog),
        "split": split_name
    }])

counts_train = get_indoor_outdoor_counts(train_attributes_df, "train")
counts_val = get_indoor_outdoor_counts(val_attributes_df, "val")

split_indoor_outdoor_counts_df = pd.concat([counts_train, counts_val], ignore_index=True)
print(split_indoor_outdoor_counts_df)

print("MetaShiftCatDog Train/Val Equal Distribution dataset statistics:")
img_id_to_group_path = os.path.expanduser("~/datasets/MetaShift/Cat-Dog-indoor-outdoor/train-val-equal-distribution/imageID_to_group.pkl")
img_id_to_group_df = pd.read_pickle(img_id_to_group_path)

# get the list of all train and val image IDs from folder
train_dir = os.path.expanduser("~/datasets/MetaShift/Cat-Dog-indoor-outdoor/train-val-equal-distribution/train")
val_dir = os.path.expanduser("~/datasets/MetaShift/Cat-Dog-indoor-outdoor/train-val-equal-distribution/val_out_of_domain")
train_image_ids_cat = os.listdir(os.path.join(train_dir, "cat"))
train_image_ids_dog = os.listdir(os.path.join(train_dir, "dog"))
train_image_ids = train_image_ids_cat + train_image_ids_dog
val_image_ids_cat = os.listdir(os.path.join(val_dir, "cat"))
val_image_ids_dog = os.listdir(os.path.join(val_dir, "dog"))
val_image_ids = val_image_ids_cat + val_image_ids_dog

def get_indoor_outdoor_counts(image_ids, split_name):
    indoor_cat = 0
    outdoor_cat = 0
    indoor_dog = 0
    outdoor_dog = 0
    total = len(image_ids)
    for image_id in image_ids:
        image_id = image_id.split(".")[0]
        group = img_id_to_group_df[image_id][0]
        if group == 'cat(indoor)':
            indoor_cat += 1
        elif group == 'cat(outdoor)':
            outdoor_cat += 1
        elif group == 'dog(indoor)':
            indoor_dog += 1
        elif group == 'dog(outdoor)':
            outdoor_dog += 1
    return pd.DataFrame([{
        "indoor_cat": indoor_cat,
        "outdoor_cat": outdoor_cat,
        "indoor_dog": indoor_dog,
        "outdoor_dog": outdoor_dog,
        "indoor_cat_pct": indoor_cat / total * 100,
        "outdoor_cat_pct": outdoor_cat / total * 100,
        "indoor_dog_pct": indoor_dog / total * 100,
        "outdoor_dog_pct": outdoor_dog / total * 100,
        "split": split_name
    }])

counts_train = get_indoor_outdoor_counts(train_image_ids, "train")
counts_val = get_indoor_outdoor_counts(val_image_ids, "val")
split_indoor_outdoor_counts_df = pd.concat([counts_train, counts_val], ignore_index=True)
print(split_indoor_outdoor_counts_df)

a = 0
