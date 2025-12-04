import os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, homogeneity_score, completeness_score
from omegaconf import DictConfig
from PIL import Image


def precision_at_k(population_ids, slices_preds, slices_probs=None, k=10, save_dir=None):
    unique_populations = np.unique(population_ids)

    precision_at_k = {}
    for pop_id in unique_populations:
        p_indices = np.where(population_ids == pop_id)[0]
        if len(p_indices) == 0:
            continue
        # Get the predictions and probabilities for the current population
        p_slice_preds = slices_preds[p_indices]

        unique_slices = np.unique(p_slice_preds)
        top_k_true_pop = np.ones(k, dtype=int) * pop_id
        best_precision = 0
        precision_at_k[pop_id] = {"cluster_id": -1, "precision": 0, "frequency": 0}
        # Iterate through each unique slice in the population
        for slice_id in unique_slices:
            # # Get the indices of the current slice in the population
            # p_k_inds = np.where(p_slice_preds == slice_id)[0]
            # if len(p_k_inds) < k:
            #     continue

            # Get predictions and probabilities for the current slice
            s_indices = np.where(slices_preds == slice_id)[0]
            if len(s_indices) < k:
                print(f"Slice {slice_id} has less than {k} samples, skipping.")
                continue

            if slices_probs is None:
                s_probs = np.ones(len(s_indices))
            else:
                s_probs = slices_probs[s_indices, slice_id]
            s_pop = population_ids[s_indices]

            # sort by probabilities
            sorted_indices = np.argsort(s_probs)[::-1]
            top_k_indices = sorted_indices[:k]
            top_k_pop = s_pop[top_k_indices]

            precision = precision_score(top_k_true_pop, top_k_pop, average='micro')

            # Calculate the frequency of the population in the slice
            s_p_frequency = np.sum(s_pop == pop_id) / len(s_pop)
            if precision > best_precision:
                best_precision = precision
                precision_at_k[pop_id]["cluster_id"] = slice_id
                precision_at_k[pop_id]["precision"] = best_precision
                precision_at_k[pop_id]["frequency"] = s_p_frequency

            elif precision == best_precision and s_p_frequency > precision_at_k[pop_id]["frequency"]:
                precision_at_k[pop_id]["cluster_id"] = slice_id
                precision_at_k[pop_id]["precision"] = best_precision
                precision_at_k[pop_id]["frequency"] = s_p_frequency

    # precision_at_k["average_precision"] = np.mean([v["precision"] for v in precision_at_k.values() if "precision" in v])
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        precision_df = pd.DataFrame.from_dict(precision_at_k, orient='index')
        precision_df.to_csv(os.path.join(save_dir, f"precision_at_{k}.csv"), index_label='population_id')

    return precision_at_k


def purity_score(labels, cluster_ids):
    """
    Calculate the purity score for the clusters.
    """
    n_samples = len(labels)
    unique_clusters = np.unique(cluster_ids)
    purity_score = 0
    for c in unique_clusters:
        c_mask = cluster_ids == c
        c_labels = labels[c_mask]
        # Get the most frequent label in the cluster counts and add it to the purity score
        c_purity = np.max(np.bincount(c_labels)) / n_samples
        purity_score += c_purity
    return purity_score


def calc_error_slices_purity_score(
    error_slice_index: list[int],
    population_indices: np.ndarray,
    slice_preds: np.ndarray,
) -> float:
    """
    Calculate the purity score for the error slices.
    """
    err_slice_mask = np.isin(slice_preds, error_slice_index)
    return purity_score(population_indices[err_slice_mask], slice_preds[err_slice_mask])


def calculate_error_slices_homogeneity_score(
    error_slice_index: list[int],
    population_indices: np.ndarray,
    slice_preds: np.ndarray,
) -> float:
    """
    Calculate the homogeneity score for the error slices.
    """
    err_slice_mask = np.isin(slice_preds, error_slice_index)
    return homogeneity_score(population_indices[err_slice_mask], slice_preds[err_slice_mask])
    

def calc_error_slices_perpopulation_completeness_score(
    target_population_index: np.ndarray,
    population_indices: np.ndarray,
    slice_preds: np.ndarray,
    n_slices: int,
) -> float:
    """
    Calculate the completeness score for the target population.
    """
    per_population_completeness_score = []
    for p in target_population_index:
        p_mask = population_indices == p
        # Compute the probability of the target population being in each slice
        p_slice_mask = slice_preds[p_mask]
        p_slices = np.unique(p_slice_mask)
        p_probs = np.array([np.sum(p_slice_mask == s) / np.sum(p_mask) for s in p_slices])
        # Compute the entropy of this distribution
        p_entropy = -np.sum(p_probs * np.log(p_probs))
        normalized_entropy = p_entropy / np.log(n_slices)

        per_population_completeness_score.append(1 - normalized_entropy)
    return np.mean(per_population_completeness_score)


def calculate_underperforming_populations_completeness_score(
    underperforming_population_index: list[int],
    population_indices: np.ndarray,
    slice_preds: np.ndarray,
) -> float:
    """
    Calculate the completeness score for the underperforming populations.
    """
    underperforming_population_mask = np.isin(population_indices, underperforming_population_index)
    return completeness_score(population_indices[underperforming_population_mask], slice_preds[underperforming_population_mask])


def find_slice_key_concepts(
    gmm_eval_dict: pd.DataFrame,
    experiment_config: DictConfig,
    max_rep_concepts: int = 3,
    semantic_concepts: str = "mnist_sum_c",
    save_dir: str = None,
    save_rep_images: bool = True,
) -> dict:
    """
    Find the key concepts for the errorslices.
    """
    save_dir = os.path.join(save_dir, "slice_representatives")
    os.makedirs(save_dir, exist_ok=True)
    
    concept_semantics_dict = {
        "mnist_sum_c": MNIST_SUM_CONCEPTS_SEMANTICS_C,
        "mnist_sum_t": MNIST_SUM_CONCEPTS_SEMANTICS_T,
        "cub": BIRD_TYPES,
        "celebA_gender": CELEBA_CONCEPTS_SEMANTICS,
        "waterbirds_with_background": WATERBIRDS_WITH_BACKGROUND_CONCEPTS_SEMANTICS,
        "metashift_cat_dog": METASHIFT_CAT_DOG_CONCEPTS_SEMANTICS,
    }

    concept_semantics = concept_semantics_dict[semantic_concepts]

    if experiment_config.model.gmm_params.filtered_concepts is not None:
        concept_semantics = [concept_semantics[i] for i in experiment_config.model.gmm_params.filtered_concepts]

    # Find the key concepts for each slice based on the average Expected Change in the Cluster Assignment Probabilities (EcCP)
    unique_slices = np.unique(gmm_eval_dict['GMM_stats:cluster_id'])
    slice_representatives_dict = {}
    
    for slice_id in unique_slices:
        slice_members_inds = np.where(gmm_eval_dict['GMM_stats:cluster_id'] == slice_id)[0]
        slice_members_probs = gmm_eval_dict['GMM_stats:cluster_probs'][slice_members_inds, slice_id]

        # Calculate the average Expected Change in the Cluster Assignment Probabilities (ECCA)
        slice_ecca = np.average(gmm_eval_dict['GMM_stats:ecca_score'][slice_members_inds], weights=slice_members_probs, axis=0) # [n_concepts]
        # Find the top max_rep_concepts concepts with the highest EcCP that above the average EcCCA
        top_concepts = np.where(slice_ecca > slice_ecca.mean())[0]
        top_concepts_ecca = slice_ecca[top_concepts]
        ordered_concepts_inds = np.argsort(top_concepts_ecca)[::-1]
        top_concepts = top_concepts[ordered_concepts_inds][:max_rep_concepts]
        top_concepts_ecca = top_concepts_ecca[ordered_concepts_inds][:max_rep_concepts]
        
        slice_members_concept_preds = gmm_eval_dict['c_scores'][slice_members_inds][:, top_concepts] # [n_members, n_top_concepts]

        # Based on the assignment probability of the slice members, determine whether 
        # each top_concepts appears or absent in the slice
        top_concepts_probs = np.average(slice_members_concept_preds, weights=slice_members_probs, axis=0)
        top_concepts_appearances = top_concepts_probs > 0.5
        top_concepts_semantic_prefixes = ["No_" if not appears else "" for appears in top_concepts_appearances]
        top_concept_semantics = [top_concepts_semantic_prefixes[i] + concept_semantics[c] for i, c in enumerate(top_concepts)]
        print(f"Top {max_rep_concepts} concepts for slice {slice_id}: {top_concept_semantics}")
        print(f"Top {max_rep_concepts} concepts for slice {slice_id}: {top_concepts}")

        # GT concepts representation for the slice
        top_gt_concepts_representation = gmm_eval_dict['concepts'][slice_members_inds][:, top_concepts] # [n_members, n_top_concepts]
        top_gt_concepts_probs = np.average(top_gt_concepts_representation, weights=slice_members_probs, axis=0)
        top_gt_concepts_appearances = top_gt_concepts_probs > 0.5
        top_gt_concepts_semantic_prefixes = ["No_" if not appears else "" for appears in top_gt_concepts_appearances]
        top_gt_concept_semantics = [top_gt_concepts_semantic_prefixes[i] + concept_semantics[c] for i, c in enumerate(top_concepts)]
        print(f"Top {max_rep_concepts} GT concepts for slice {slice_id}: {top_gt_concept_semantics}")

        # Label prediction for the slice
        slice_class_preds = gmm_eval_dict['y_preds'][slice_members_inds]
        slice_class_pred = np.argmax(np.bincount(slice_class_preds, weights=slice_members_probs))
        # slice_class_pred = round(slice_class_pred_probs)
        print(f"Slice class prediction for slice {slice_id}: {slice_class_pred}")

        # GT label for the slice
        slice_gt_labels = gmm_eval_dict['labels'][slice_members_inds]
        slice_gt_label = np.argmax(np.bincount(slice_gt_labels, weights=slice_members_probs))
        print(f"Slice GT label for slice {slice_id}: {slice_gt_label}")

        # Population id as representative for the slice
        slice_population_idx = gmm_eval_dict['population_idx'][slice_members_inds]
        slice_population_idx_counts = np.bincount(slice_population_idx, weights=slice_members_probs)
        slice_population_idx_representative = np.argmax(slice_population_idx_counts)
        print(f"Slice population id as representative for slice {slice_id}: {slice_population_idx_representative}")
        rep_pop_loc = np.where(slice_population_idx == slice_population_idx_representative)[0][0]
        slice_population_name_representative = gmm_eval_dict['population_name'][slice_members_inds][rep_pop_loc]
        print(f"Slice population name as representative for slice {slice_id}: {slice_population_name_representative}")
       
        # Compute the frequency of the representative population in the slice
        slice_population_idx_frequency = np.sum(slice_population_idx == slice_population_idx_representative) / len(slice_population_idx)
        print(f"Slice population frequency for slice {slice_id}: {slice_population_idx_frequency}")

        # Select the top-10 images with the highest assignment probability
        top_images_inds = np.argsort(slice_members_probs)[::-1][:20]
        top_images = gmm_eval_dict['img_id'][slice_members_inds][top_images_inds]
        top_images_probs = slice_members_probs[top_images_inds]
        
        top_images_paths = None
        if "img_path" in gmm_eval_dict:
            top_images_paths = gmm_eval_dict['img_path'][slice_members_inds][top_images_inds]

        if save_rep_images:
            slice_save_dir = os.path.join(save_dir, f"slice_{slice_id}")
            os.makedirs(slice_save_dir, exist_ok=True)
            for img_id, img_path in zip(top_images[:5], top_images_paths[:5]):
                img = Image.open(img_path)
                img.save(os.path.join(slice_save_dir, f"{img_id}.png"))

        slice_representatives_dict[slice_id] = {
            "slice_id": slice_id,
            "population_idx": slice_population_idx_representative,
            "population_name": slice_population_name_representative,
            "population_frequency": slice_population_idx_frequency,
            "pred_class": slice_class_pred,
            "gt_class": slice_gt_label,
            "key_pred_concepts": top_concept_semantics,
            "key_pred_concepts_score": top_concepts_probs,
            "key_gt_concepts": top_gt_concept_semantics,
            "key_gt_concepts_score": top_gt_concepts_probs,
            "top_concepts_ecca": top_concepts_ecca,
            "representative_images": top_images,
            "representative_images_probs": top_images_probs,
            "representative_images_paths": top_images_paths,
        }

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        slice_representatives_df = pd.DataFrame.from_dict(slice_representatives_dict, orient='index')
        slice_representatives_df.to_csv(os.path.join(save_dir, f"slice_representatives.csv"), index=False)

    return slice_representatives_dict


def create_slice_stats_df(
        df,
        threshold=0.9,
        num_rep_concepts=1,
        semantic_concepts="mnist_sum_c",
        save_dir=None,
        ):

    if semantic_concepts == "mnist_sum_c":
        CONCEPT_SEMANTICS = MNIST_SUM_CONCEPTS_SEMANTICS_C
    elif semantic_concepts == "mnist_sum_t":
        CONCEPT_SEMANTICS = MNIST_SUM_CONCEPTS_SEMANTICS_T
    elif semantic_concepts == "cub":
        CONCEPT_SEMANTICS = BIRD_TYPES
    elif semantic_concepts == "celebA_gender":
        CONCEPT_SEMANTICS = CELEBA_CONCEPTS_SEMANTICS
    elif semantic_concepts == "waterbirds_with_background":
        CONCEPT_SEMANTICS = WATERBIRDS_WITH_BACKGROUND_CONCEPTS_SEMANTICS
    elif semantic_concepts == "metashift_cat_dog":
        CONCEPT_SEMANTICS = METASHIFT_CAT_DOG_CONCEPTS_SEMANTICS
    else:
        raise ValueError(f"Unknown semantic_concepts: {semantic_concepts}")
        
    img_id = df['img_id']
    population_idx = df['population_idx']
    slices = df['GMM_stats:cluster_id']
    slice_probs = df['GMM_stats:cluster_probs']
    targets = df['labels']
    y_preds = df['y_preds']
    concepts = df['concepts']
    c_pred_probs = df['c_scores']
    population_names = df['population_name']
    origin_labels = df['origin_label'] if 'origin_label' in df else None
    
    n_images = len(img_id)
    unique_slices = np.unique(slices)
    slice_stats = {}

    for cluster_id in unique_slices:
        # Get the indices of the current cluster
        k_indices = np.where(slices == cluster_id)[0]
        if len(k_indices) == 0:
            Warning(f"Cluster {cluster_id} is empty, skipping.")
            continue

        k_slice_probs = slice_probs[k_indices, cluster_id]
        k_img_id = img_id[k_indices]
        k_labels = targets[k_indices]
        k_concepts = concepts[k_indices]
        k_preds = y_preds[k_indices]
        k_c_pred_probs = c_pred_probs[k_indices]
        k_population_idx = population_idx[k_indices]
        k_population_names = population_names[k_indices]
        k_origin_labels = origin_labels[k_indices] if origin_labels is not None else None
        
        # Filter out the indices with probabilities below the threshold
        # Sort the indices based on probabilities and take the top k
        # sorted_indices = np.argsort(k_slice_probs)[::-1]
        # top_k_indices = sorted_indices[:k]
        top_indices = np.where(k_slice_probs >= threshold)[0]
        if len(top_indices) == 0:
            Warning(f"Cluster {cluster_id} has no samples above the threshold {threshold}, skipping.")
            continue

        top_labels = k_labels[top_indices]
        top_preds = k_preds[top_indices]

        # Gather cluster statisctics
        rep_label = np.argmax(np.bincount(top_labels))
        rep_pred = np.argmax(np.bincount(top_preds))
        # Adjust represantatives according to rep_label and rep_pred
        adj_rep_inds = np.where((top_labels == rep_label) & (top_preds == rep_pred))[0]
        if len(adj_rep_inds) > 0:
            # Filter and sort the top indices based on the adjusted representative indices
            top_indices = top_indices[adj_rep_inds]
        top_indices = top_indices[np.argsort(k_slice_probs[top_indices])][::-1]
        top_img_id = k_img_id[top_indices]
        top_concepts = k_concepts[top_indices].astype(int)
        # top_c_preds = (k_c_pred_probs[top_indices] > 0.5).astype(int)
        top_c_pred_probs = k_c_pred_probs[top_indices]
        top_populations = k_population_idx[top_indices]
        top_population_names = k_population_names[top_indices]
        top_origin_labels = k_origin_labels[top_indices] if k_origin_labels is not None else None
        
        # Calculate the most appearing concepts
        top_concepts_freq = top_concepts.sum(axis=0)
        rep_concept_inds = np.argsort(top_concepts_freq)[::-1][:num_rep_concepts]
        # Remove concepts that are not present in the cluster
        rep_concept_inds = rep_concept_inds[top_concepts_freq[rep_concept_inds] > 0]
        rep_concept_semantics = [CONCEPT_SEMANTICS[i] for i in rep_concept_inds]
        # rep_concept = np.zeros_like(top_concepts[0], dtype=int)
        # rep_concept[rep_concept_inds] = 1
        # k_rep_c_pred = [np.argmax(np.bincount(top_k_c_probs[:, i])) for i in range(top_k_c_probs.shape[1])]
        c_pred_frequency = top_c_pred_probs.sum(axis=0)
        # num_rep_concepts_p = num_rep_concepts
        # # If there are less present concepts than num_rep_concepts, adjust the number
        # if len(np.where(c_pred_frequency > 0)[0]) < num_rep_concepts:
        #     num_rep_concepts_p = len(np.where(c_pred_frequency > 0)[0])
        rep_c_pred_inds = np.argsort(c_pred_frequency)[::-1][:num_rep_concepts]
        rep_c_pred_inds = rep_c_pred_inds[c_pred_frequency[rep_c_pred_inds] > 0]
        rep_c_pred_semantics = [CONCEPT_SEMANTICS[i] for i in rep_c_pred_inds]
        # rep_c_pred = np.zeros_like(top_c_preds[0], dtype=int)
        # rep_c_pred[rep_c_pred_inds] = 1
        # Get the most frequent population
        rep_pop = np.argmax(np.bincount(top_populations))
        pop_loc = np.where(rep_pop == top_populations)[0][0]
        rep_pop_name = top_population_names[pop_loc]
        rep_origin_label = None
        if top_origin_labels is not None:
            rep_origin_label = np.argmax(np.bincount(top_origin_labels))
        error_rate = np.mean(k_labels != k_preds)

        slice_stats[cluster_id] = {
            "cluster_id": cluster_id,
            "population_idx": rep_pop,
            "population_name": rep_pop_name,
            "imgs": top_img_id,
            "label": rep_label,
            "origin_label": rep_origin_label,
            "y_pred": rep_pred,
            "concept": rep_concept_inds,
            "concept_semantics": rep_concept_semantics,
            "concept_frequency": top_concepts_freq[rep_concept_inds] / len(k_indices),
            "c_pred": rep_c_pred_inds,
            "c_pred_semantics": rep_c_pred_semantics,
            "c_pred_frequency": c_pred_frequency[rep_c_pred_inds] / len(k_indices),
            "error_rate": error_rate,
            "population_size": round(len(k_indices) / n_images, 4),
            "n_members": len(k_indices),
            "n_top_members": len(top_indices),
        }

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        slice_stats_df = pd.DataFrame.from_dict(slice_stats, orient='index')
        slice_stats_df.to_csv(os.path.join(save_dir, f"slice_stats_at_{threshold}.csv"), index=False)
    
    return slice_stats


def calc_homogeinity(labels, cluster_ids):
    """
    Calculate the homogenity of the predictions.
    Homogenity is defined as the ratio of correct predictions to the total number of predictions.
    """
    if len(labels) == 0:
        return None
    
    homogeinity_dict = {}
    homogeinity_dict["total"] = homogeneity_score(labels, cluster_ids)
    
    unique_clusters = np.unique(cluster_ids)
    for c in unique_clusters:
        ll = labels[cluster_ids == c]
        lc = cluster_ids[cluster_ids == c]

        homogeinity_dict[c] = homogeneity_score(ll, lc)
        
    return homogeinity_dict


MNIST_SUM_CONCEPTS_SEMANTICS_C = [
    "0-left",
    "1-left",
    "2-left",
    "3-left",
    "0-right",
    "1-right",
    "2-right",
    "3-right",
    "red",
    # "(0, 0)",
    # "(0, 1)",
    # "(0, 2)",
    # "(0, 3)",
    # "(0, 3, 'red')",
    # "(1, 0)",
    # "(1, 1)",
    # "(1, 1, 'red')",
    # "(1, 2)",
    # "(1, 3)",
    # "(2, 0)",
    # "(2, 1)",
    # "(2, 2)",
    # "(2, 3)",
    # "(3, 0)",
    # "(3, 1)",
    # "(3, 2)",
    # "(3, 3)",
]

MNIST_SUM_CONCEPTS_SEMANTICS_T = [
# Population names
    "(0, 0)",
    "(0, 1)",
    "(0, 2)",
    "(0, 2, 'red')",
    "(0, 3)",
    "(1, 0)",
    "(1, 1)",
    "(1, 1, 'red')",
    "(1, 2)",
    "(1, 3)",
    "(1, 3, 'label_noise')",
    "(2, 0)",
    "(2, 0, 'red')",
    "(2, 1)",
    "(2, 2)",
    "(2, 2, 'label_noise')",
    "(2, 3)",
    "(2, 3, 'red')",
    "(3, 0)",
    "(3, 1)",
    "(3, 1, 'label_noise')",
    "(3, 2)",
    "(3, 2, 'red')",
    "(3, 3)",
] 

# CUB Class names
BIRD_TYPES = [
    "Black_footed_Albatross",
    "Laysan_Albatross",
    "Sooty_Albatross",
    "Groove_billed_Ani",
    "Crested_Auklet",
    "Least_Auklet",
    "Parakeet_Auklet",
    "Rhinoceros_Auklet",
    "Brewer_Blackbird",
    "Red_winged_Blackbird",
    "Rusty_Blackbird",
    "Yellow_headed_Blackbird",
    "Bobolink",
    "Indigo_Bunting",
    "Lazuli_Bunting",
    "Painted_Bunting",
    "Cardinal",
    "Spotted_Catbird",
    "Gray_Catbird",
    "Yellow_breasted_Chat",
    "Eastern_Towhee",
    "Chuck_will_Widow",
    "Brandt_Cormorant",
    "Red_faced_Cormorant",
    "Pelagic_Cormorant",
    "Bronzed_Cowbird",
    "Shiny_Cowbird",
    "Brown_Creeper",
    "American_Crow",
    "Fish_Crow",
    "Black_billed_Cuckoo",
    "Mangrove_Cuckoo",
    "Yellow_billed_Cuckoo",
    "Gray_crowned_Rosy_Finch",
    "Purple_Finch",
    "Northern_Flicker",
    "Acadian_Flycatcher",
    "Great_Crested_Flycatcher",
    "Least_Flycatcher",
    "Olive_sided_Flycatcher",
    "Scissor_tailed_Flycatcher",
    "Vermilion_Flycatcher",
    "Yellow_bellied_Flycatcher",
    "Frigatebird",
    "Northern_Fulmar",
    "Gadwall",
    "American_Goldfinch",
    "European_Goldfinch",
    "Boat_tailed_Grackle",
    "Eared_Grebe",
    "Horned_Grebe",
    "Pied_billed_Grebe",
    "Western_Grebe",
    "Blue_Grosbeak",
    "Evening_Grosbeak",
    "Pine_Grosbeak",
    "Rose_breasted_Grosbeak",
    "Pigeon_Guillemot",
    "California_Gull",
    "Glaucous_winged_Gull",
    "Heermann_Gull",
    "Herring_Gull",
    "Ivory_Gull",
    "Ring_billed_Gull",
    "Slaty_backed_Gull",
    "Western_Gull",
    "Anna_Hummingbird",
    "Ruby_throated_Hummingbird",
    "Rufous_Hummingbird",
    "Green_Violetear",
    "Long_tailed_Jaeger",
    "Pomarine_Jaeger",
    "Blue_Jay",
    "Florida_Jay",
    "Green_Jay",
    "Dark_eyed_Junco",
    "Tropical_Kingbird",
    "Gray_Kingbird",
    "Belted_Kingfisher",
    "Green_Kingfisher",
    "Pied_Kingfisher",
    "Ringed_Kingfisher",
    "White_breasted_Kingfisher",
    "Red_legged_Kittiwake",
    "Horned_Lark",
    "Pacific_Loon",
    "Mallard",
    "Western_Meadowlark",
    "Hooded_Merganser",
    "Red_breasted_Merganser",
    "Mockingbird",
    "Nighthawk",
    "Clark_Nutcracker",
    "White_breasted_Nuthatch",
    "Baltimore_Oriole",
    "Hooded_Oriole",
    "Orchard_Oriole",
    "Scott_Oriole",
    "Ovenbird",
    "Brown_Pelican",
    "White_Pelican",
    "Western_Wood_Pewee",
    "Sayornis",
    "American_Pipit",
    "Whip_poor_Will",
    "Horned_Puffin",
    "Common_Raven",
    "White_necked_Raven",
    "American_Redstart",
    "Geococcyx",
    "Loggerhead_Shrike",
    "Great_Grey_Shrike",
    "Baird_Sparrow",
    "Black_throated_Sparrow",
    "Brewer_Sparrow",
    "Chipping_Sparrow",
    "Clay_colored_Sparrow",
    "House_Sparrow",
    "Field_Sparrow",
    "Fox_Sparrow",
    "Grasshopper_Sparrow",
    "Harris_Sparrow",
    "Henslow_Sparrow",
    "Le_Conte_Sparrow",
    "Lincoln_Sparrow",
    "Nelson_Sharp_tailed_Sparrow",
    "Savannah_Sparrow",
    "Seaside_Sparrow",
    "Song_Sparrow",
    "Tree_Sparrow",
    "Vesper_Sparrow",
    "White_crowned_Sparrow",
    "White_throated_Sparrow",
    "Cape_Glossy_Starling",
    "Bank_Swallow",
    "Barn_Swallow",
    "Cliff_Swallow",
    "Tree_Swallow",
    "Scarlet_Tanager",
    "Summer_Tanager",
    "Artic_Tern",
    "Black_Tern",
    "Caspian_Tern",
    "Common_Tern",
    "Elegant_Tern",
    "Forsters_Tern",
    "Least_Tern",
    "Green_tailed_Towhee",
    "Brown_Thrasher",
    "Sage_Thrasher",
    "Black_capped_Vireo",
    "Blue_headed_Vireo",
    "Philadelphia_Vireo",
    "Red_eyed_Vireo",
    "Warbling_Vireo",
    "White_eyed_Vireo",
    "Yellow_throated_Vireo",
    "Bay_breasted_Warbler",
    "Black_and_white_Warbler",
    "Black_throated_Blue_Warbler",
    "Blue_winged_Warbler",
    "Canada_Warbler",
    "Cape_May_Warbler",
    "Cerulean_Warbler",
    "Chestnut_sided_Warbler",
    "Golden_winged_Warbler",
    "Hooded_Warbler",
    "Kentucky_Warbler",
    "Magnolia_Warbler",
    "Mourning_Warbler",
    "Myrtle_Warbler",
    "Nashville_Warbler",
    "Orange_crowned_Warbler",
    "Palm_Warbler",
    "Pine_Warbler",
    "Prairie_Warbler",
    "Prothonotary_Warbler",
    "Swainson_Warbler",
    "Tennessee_Warbler",
    "Wilson_Warbler",
    "Worm_eating_Warbler",
    "Yellow_Warbler",
    "Northern_Waterthrush",
    "Louisiana_Waterthrush",
    "Bohemian_Waxwing",
    "Cedar_Waxwing",
    "American_Three_toed_Woodpecker",
    "Pileated_Woodpecker",
    "Red_bellied_Woodpecker",
    "Red_cockaded_Woodpecker",
    "Red_headed_Woodpecker",
    "Downy_Woodpecker",
    "Bewick_Wren",
    "Cactus_Wren",
    "Carolina_Wren",
    "House_Wren",
    "Marsh_Wren",
    "Rock_Wren",
    "Winter_Wren",
    "Common_Yellowthroat",
]

CELEBA_CONCEPTS_SEMANTICS = [
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
    'Mouth_Slightly_Open', # 20
    'Mustache', # 21
    'Narrow_Eyes', # 22
    'No_Beard', # 23
    'Oval_Face', # 24
    'Pale_Skin', # 25
    'Pointy_Nose', # 26
    'Receding_Hairline', # 27
    'Rosy_Cheeks', # 28
    'Sideburns', # 29
    'Smiling', # 30
    'Straight_Hair', # 31
    'Wavy_Hair', # 32
    'Wearing_Earrings', # 33
    'Wearing_Hat', # 34
    'Wearing_Lipstick', # 35
    'Wearing_Necklace', # 36
    'Wearing_Necktie', # 37
    'Young', # 38
]

WATERBIRDS_WITH_BACKGROUND_CONCEPTS_SEMANTICS = [
    "has_bill_shape::dagger", # 0
    "has_bill_shape::hooked_seabird", # 1
    "has_bill_shape::all-purpose", # 2
    "has_bill_shape::cone", # 3
    "has_wing_color::brown", # 4
    "has_wing_color::grey", # 5
    "has_wing_color::yellow", # 6
    "has_wing_color::black", # 7
    "has_wing_color::white", # 8
    "has_wing_color::buff", # 9
    "has_upperparts_color::brown", # 10
    "has_upperparts_color::grey", # 11
    "has_upperparts_color::yellow", # 12
    "has_upperparts_color::black", # 13
    "has_upperparts_color::white", # 14
    "has_upperparts_color::buff", # 15
    "has_underparts_color::brown", # 16
    "has_underparts_color::grey", # 17
    "has_underparts_color::yellow", # 18
    "has_underparts_color::black", # 19
    "has_underparts_color::white", # 20
    "has_underparts_color::buff", # 21
    "has_breast_pattern::solid", # 22
    "has_breast_pattern::striped", # 23
    "has_breast_pattern::multi-colored", # 24
    "has_back_color::brown", # 25
    "has_back_color::grey", # 26
    "has_back_color::yellow", # 27
    "has_back_color::black", # 28
    "has_back_color::white", # 29
    "has_back_color::buff", # 30
    "has_tail_shape::notched_tail", # 31
    "has_upper_tail_color::brown", # 32
    "has_upper_tail_color::grey", # 33 
    "has_upper_tail_color::black", # 34
    "has_upper_tail_color::white", # 35
    "has_upper_tail_color::buff", # 36
    "has_head_pattern::eyebrow", # 37
    "has_head_pattern::plain", # 38
    "has_breast_color::brown", # 39
    "has_breast_color::grey", # 40
    "has_breast_color::yellow", # 41
    "has_breast_color::black", # 42
    "has_breast_color::white", # 43
    "has_breast_color::buff", # 44
    "has_throat_color::grey", # 45
    "has_throat_color::yellow", # 46
    "has_throat_color::black", # 47
    "has_throat_color::white", # 48
    "has_throat_color::buff", # 49
    "has_eye_color::black", # 50
    "has_bill_length::about_the_same_as_head", # 51
    "has_bill_length::shorter_than_head", # 52
    "has_forehead_color::blue", # 53
    "has_forehead_color::brown", # 54
    "has_forehead_color::grey", # 55
    "has_forehead_color::yellow", # 56
    "has_forehead_color::black", # 57
    "has_forehead_color::white", # 58
    "has_under_tail_color::brown", # 59
    "has_under_tail_color::grey", # 60
    "has_under_tail_color::black", # 61
    "has_under_tail_color::white", # 62
    "has_under_tail_color::buff", # 63
    "has_nape_color::brown", # 64
    "has_nape_color::grey", # 65
    "has_nape_color::yellow", # 66
    "has_nape_color::black", # 67
    "has_nape_color::white", # 68
    "has_nape_color::buff", # 69
    "has_belly_color::brown", # 70
    "has_belly_color::grey", # 71
    "has_belly_color::yellow", # 72
    "has_belly_color::black", # 73
    "has_belly_color::white", # 74
    "has_belly_color::buff", # 75
    "has_wing_shape::rounded-wings", # 76
    "has_wing_shape::pointed-wings", # 77
    "has_size::small_(5_-_9_in)", # 78
    "has_size::medium_(9_-_16_in)", # 79
    "has_size::very_small_(3_-_5_in)", # 80
    "has_shape::duck-like", # 81
    "has_shape::perching-like", # 82
    "has_back_pattern::solid", # 83
    "has_back_pattern::striped", # 84
    "has_back_pattern::multi-colored", # 85
    "has_tail_pattern::solid", # 86
    "has_tail_pattern::striped", # 87
    "has_tail_pattern::multi-colored", # 88
    "has_belly_pattern::solid", # 89
    "has_primary_color::brown", # 90
    "has_primary_color::grey", # 91
    "has_primary_color::yellow", # 92
    "has_primary_color::black", # 93
    "has_primary_color::white", # 94
    "has_primary_color::buff", # 95
    "has_leg_color::grey", # 96
    "has_leg_color::black", # 97
    "has_leg_color::buff", # 98
    "has_bill_color::grey", # 99
    "has_bill_color::black", # 100
    "has_bill_color::buff", # 101
    "has_crown_color::blue", # 102
    "has_crown_color::brown", # 103
    "has_crown_color::grey", # 104
    "has_crown_color::yellow", # 105
    "has_crown_color::black", # 106
    "has_crown_color::white", # 107
    "has_wing_pattern::solid", # 108
    "has_wing_pattern::spotted", # 109
    "has_wing_pattern::striped", # 1110
    "has_wing_pattern::multi-colored", # 111
    "bamboo_forest", # 112
    "forest", # 113
    "ocean", # 114
    "lake", # 115
    ]

METASHIFT_CAT_DOG_CONCEPTS_SEMANTICS = [
    "Long snout", # 0
    "Short snout", # 1
    "Floppy ears", # 2
    "Upright ears", # 3
    "Round eyes", # 4
    "Slit pupils", # 5  
    "Curled tail", # 6
    "Straight tail", # 7
    "Stocky body", # 8
    "Slim body", # 9
    "Wide muzzle", # 10
    "Narrow muzzle", # 11
    "Large nose", # 12
    "Small nose", # 13
    "Broad paws", # 14
    "Small paws", # 15
    "Short, dense fur", # 16
    "Fine, soft fur", # 17
    "Simple or spotted coat", # 18
    "Striped or marbled coat", # 19
    "Short whiskers", # 20
    "Long whiskers", # 21
    "Expressive face", # 22
    "Neutral face", # 23
    "Square or upright posture", # 24
    "Crouched or perched posture", # 25
    "Indoor", # 26
    "Outdoor", # 27
]


if __name__ == "__main__":
    a = 0