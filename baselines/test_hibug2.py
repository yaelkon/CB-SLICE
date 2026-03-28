import os
import numpy as np
import pandas as pd

from utils.plotting import plot_slices
from utils.metrics import (
    precision_at_k,
    calc_error_slices_purity_score,
    calc_error_slices_perpopulation_completeness_score,
)
from utils.utils import data_preprocessing

# default hyperparameters from HiBug2-main/config.py
SLICE_LEN = 3            # max number of concepts combined per slice
MIN_DATA_PROPORTION = 0  # min fraction of class data a slice must cover


# step22_get_slices.py 
# given a concept name and value: 
# it finds all images that match and computes their avg correctness out of 1
def _get_matched_imgs(imgs, labels, correctness, attr, tag):
    matched, n_correct = [], 0
    for img in imgs:
        if labels[img][attr] == tag:
            matched.append(img)
            n_correct += correctness[img]
    acc = -1 if not matched else n_correct / len(matched)
    return matched, acc

# checks if 2 slices can be combined. 
# 2 slices can combine if they differ by exactly one concept 
def _check_slice_match(slice1, slice2):
    diff1 = list(slice1.items() - slice2.items())
    diff2 = list(slice2.items() - slice1.items())
    if len(diff1) == 1 and len(diff2) == 1 and diff1[0][0] != diff2[0][0]:
        return diff2[0][0], diff2[0][1]
    return None, None

#Layer 1: every single concept value that has below-average accuracy
#Layer 2: pairs of concept values
#Layer 3: triples
# it only keeps combinations where accuracy is lower than all its "parent" slices and covers at least MIN_DATA_PROPORTION of the data
def _enumerate_slices(attrs_tags, labels, correctness, n_total):
    """Enumerate all concept-combination slices with below-average accuracy."""
    root_acc = sum(correctness.values()) / len(correctness)
    all_slices, base_slices = [], []
    # Layer 1: single-concept slices
    for attr, value in attrs_tags.items():
        for tag in value['tags']:
            matched, acc = _get_matched_imgs(list(labels.keys()), labels, correctness, attr, tag)
            proportion = len(matched) / n_total
            if 0 <= acc <= root_acc and proportion >= MIN_DATA_PROPORTION:
                base_slices.append({'slice': {attr: tag}, 'imgs': matched,
                                    'proportion': proportion, 'acc': acc})
    all_slices += base_slices
    # Layers 2 to SLICE_LEN: combine slices
    for _ in range(SLICE_LEN - 1):
        if not base_slices:
            break
        new_slices = []
        for i in range(len(base_slices)):
            for j in range(i, len(base_slices)):
                attr, tag = _check_slice_match(base_slices[i]['slice'], base_slices[j]['slice'])
                if attr is not None:
                    matched, acc = _get_matched_imgs(
                        base_slices[i]['imgs'], labels, correctness, attr, tag)
                    proportion = len(matched) / n_total
                    if 0 <= acc <= base_slices[i]['acc'] and \
                            acc <= base_slices[j]['acc'] and \
                            proportion >= MIN_DATA_PROPORTION:
                        new_slices.append({
                            'slice': {**base_slices[i]['slice'], **{attr: tag}},
                            'imgs': matched, 'proportion': proportion, 'acc': acc,
                        })
        all_slices += new_slices
        base_slices = new_slices
    return all_slices


# convert CB-SLICE data arrays into HiBug2 Step 2 dicts
# it takes concepts and splits everything by class
# each concept becomes an attribute named concept_0...concept_8
# each sample gets tagged "yes" or "no" per concept
# correctness becomes 1 if y_pred == label else 0

def _build_hibug2_inputs(data):
    concepts = data['concepts']   
    labels = data['labels']       
    y_preds = data['y_preds']     
    N, n_concepts = concepts.shape
    attr_names = [f"concept_{i}" for i in range(n_concepts)]
    classes = np.unique(labels)

    tags, img_labels, correctness = {}, {}, {}
    for c in classes:
        c_str = str(c)
        tags[c_str] = {a: {'tags': ['yes', 'no'], 'explanation': a} for a in attr_names}
        img_labels[c_str] = {}
        correctness[c_str] = {}

    for i in range(N):
        c_str = str(labels[i])
        img_labels[c_str][str(i)] = {
            attr_names[j]: 'yes' if concepts[i, j] == 1 else 'no'
            for j in range(n_concepts)
        }
        correctness[c_str][str(i)] = int(y_preds[i] == labels[i])

    return tags, img_labels, correctness, classes, attr_names

# purpose: it converts HiBug2 output (slice dicts), per-sample slice IDs that the metrics expect
#assign each sample to a slice ID based on its concept values
#samples that match no slice in their class get a per-class catch all ID
def _assign_slices(concepts, labels, top_slices_by_class, attr_names):

    attr_to_idx = {name: i for i, name in enumerate(attr_names)}
    assignments = np.full(len(labels), -1, dtype=int)
    offset = 0

    for c_str in sorted(top_slices_by_class.keys()):
        slices = top_slices_by_class[c_str]
        class_indices = np.where(labels == int(c_str))[0]
        for idx in class_indices:
            row = concepts[idx]
            for s_id, slice_def in enumerate(slices):
                if all(
                    (row[attr_to_idx[a]] == 1) == (t == 'yes')
                    for a, t in slice_def.items()
                ):
                    assignments[idx] = offset + s_id
                    break
            if assignments[idx] == -1:
                assignments[idx] = offset + len(slices)  # catch all for this class
        offset += len(slices) + 1  # +1 for the catch all

    return assignments, offset



def main(val_data, test_data, config=None):
    val_data = data_preprocessing(val_data)
    test_data = data_preprocessing(test_data)
    n_slices = config['n_slices']

    print("Error Rate:", np.mean(test_data['y_preds'] != test_data['labels']))

    # build HiBug2 Step 2 inputs from val data
    tags, img_labels, correctness, classes, attr_names = _build_hibug2_inputs(val_data)

    # enumerate slices per class and keep top-n by lowest accuracy (worst-performing first)
    top_slices_by_class = {}
    for c_str in tags:
        n_total = len(img_labels[c_str])
        if n_total == 0:
            top_slices_by_class[c_str] = []
            continue
        all_slices = _enumerate_slices(
            tags[c_str], img_labels[c_str], correctness[c_str], n_total)
        sorted_slices = sorted(all_slices, key=lambda s: s['acc'])
        top_slices_by_class[c_str] = [s['slice'] for s in sorted_slices[:n_slices]]

    # assign test samples to slices based on their concept values
    slice_labels, total_slices = _assign_slices(
        test_data['concepts'], test_data['labels'], top_slices_by_class, attr_names
    )

    test_pop_idx = test_data['population_idx']
    error_mask = test_data['y_preds'] != test_data['labels']

    # plot_slices(
    #     embeddings=test_data['embeddings'][error_mask],
    #     slices=slice_labels[error_mask],
    #     saving_path=config['saving_path'],
    #     dim_reduction_method='tsne',
    #     fig_name='error_slices_tsne_plot_hibug2',
    #     title="Predicted Error Slices Visualisation (HiBug2)",
    # )

    for k in config['k']:
        precision_at_k_results = precision_at_k(
            population_ids=test_pop_idx,
            slices_preds=slice_labels,
            slices_probs=None,
            k=k,
            save_dir=config['saving_path'],
        )
        print(f"Precision at {k} results:\n", precision_at_k_results)
        avg_p = np.mean([precision_at_k_results[p]['precision'] for p in precision_at_k_results])
        print(f"Average Precision at {k}: {avg_p:.4f}")

    representative_slices = precision_at_k(
        population_ids=test_pop_idx,
        slices_preds=slice_labels,
        slices_probs=None,
        k=10,
        save_dir=None,
    )
    stats_dict = {}
    error_slices = [
        representative_slices[pop_idx]['cluster_id']
        for pop_idx in representative_slices
        if pop_idx in config['error_pop_inds']
    ]
    assert -1 not in error_slices, "Error slices should not contain -1"

    stats_dict['error_slices_purity_score'] = calc_error_slices_purity_score(
        error_slice_index=error_slices,
        population_indices=test_pop_idx,
        slice_preds=slice_labels,
    )
    stats_dict['underperforming_populations_completeness_score'] = \
        calc_error_slices_perpopulation_completeness_score(
            target_population_index=config['error_pop_inds'],
            population_indices=test_pop_idx,
            slice_preds=slice_labels,
            n_slices=n_slices,
        )

    stats_path = os.path.join(config['saving_path'], 'metric_scores.csv')
    pd.DataFrame(stats_dict, index=[0]).to_csv(stats_path, index=False)
    print("Done!")


if __name__ == "__main__":
    experiment_path = "../../data/MNIST/MNIST-SUM/CBM_Joint/"
    evaluation_folder_name = "Evaluations/"
    val_df_path = os.path.join(experiment_path, evaluation_folder_name, "val_eval_df.pkl")
    test_df_path = os.path.join(experiment_path, evaluation_folder_name, "val_eval_df.pkl")
    
    n_slices_list = [4]

    for s in n_slices_list:
        base_saving_path = os.path.join(
            experiment_path, evaluation_folder_name, "new_slices", f"hibug2_k={s}"
        )

        os.makedirs(base_saving_path, exist_ok=True)

        val_df = pd.read_pickle(val_df_path)
        test_df = pd.read_pickle(test_df_path)

        for seed in [42, 77, 666, 123, 58]:
            saving_path_seed = os.path.join(base_saving_path, f"seed_{seed}")
            os.makedirs(saving_path_seed, exist_ok=True)

            config = {
                "n_slices": s,
                "saving_path": saving_path_seed,
                "seed": seed,
                "k": [5, 10],
                "error_pop_inds": [6, 12],
            }

            main(val_data=val_df, test_data=test_df, config=config)
