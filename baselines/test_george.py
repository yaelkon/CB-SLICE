
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from umap import UMAP

from utils.plotting import plot_slices
from utils.metrics import (
    precision_at_k,
    calc_error_slices_purity_score,
    calc_error_slices_perpopulation_completeness_score,
)
from utils.utils import data_preprocessing


def main(val_data=None, test_data=None, config=None):

    val_data = data_preprocessing(val_data)
    test_data = data_preprocessing(test_data)

    error_mask = test_data['y_preds'] != test_data['labels']
    print("Error Rate: ", np.mean(test_data['y_preds'] != test_data['labels']))
    
    # First use UMAP for dimensionality reduction
    umap_model = UMAP(
        n_components=2,
        n_neighbors=10,  # Hyperparameter as was used in the paper
        min_dist=0,  # Hyperparameter as was used in the paper
        random_state=config['seed'],
    )
    val_embeddings_reduced = umap_model.fit_transform(val_data['embeddings'])
    test_embeddings_reduced = umap_model.transform(test_data['embeddings'])

    # Train the clustering algorithm
    classes = np.unique(test_data['labels'])
    gmm_slicers = {}
    all_test_indices, all_test_slices, all_test_slice_probs, all_test_init_k = [], [], [], []
    current_init_k = 0
    for c in classes:
        gmm_slicer = GaussianMixture(n_components=config['n_slices'], random_state=config['seed'])
        # Fit the KMeans model on the validation embeddings
        val_c_embeddings = val_embeddings_reduced[val_data['labels'] == c]
        val_cluster_labels = gmm_slicer.fit_predict(val_c_embeddings)

        # Predict the slices for the test data
        test_c_ids = np.where(test_data['labels'] == c)[0]
        test_c_embeddings = test_embeddings_reduced[test_c_ids]
        slices = gmm_slicer.predict(test_c_embeddings)
        # Get the probability of the assigned slice for each sample
        slice_probs = gmm_slicer.predict_proba(test_c_embeddings)
        slice_init_k = np.ones(len(test_c_ids)) * current_init_k
        # slice_probs = slice_probs[np.arange(len(test_c_ids)), slices]
        
        slices += current_init_k  # Offset the slice labels by the current init_k

        all_test_indices.extend(test_c_ids)
        all_test_slices.extend(slices)
        all_test_slice_probs.extend(slice_probs)
        all_test_init_k.extend(slice_init_k)

        gmm_slicers[c] = gmm_slicer
        current_init_k += config['n_slices']

    all_test_indices = np.array(all_test_indices)
    all_test_slices = np.array(all_test_slices)
    all_test_init_k = np.array(all_test_init_k, dtype=np.int32)
    all_test_slice_probs = np.array(all_test_slice_probs)
    # Convert the slice probabilities to a 2D array containing the probabilities for each slice for each sample
    slice_probs = np.zeros((len(all_test_indices), config['n_slices'] * len(classes)))
    row_idx = np.arange(len(all_test_indices))[:, None]
    col_idx = (all_test_init_k[:, None] + np.arange(config['n_slices'])[None, :]).astype(int)
    slice_probs[row_idx, col_idx] = all_test_slice_probs
    
    inds_order = np.argsort(all_test_indices)
    slice_labels = all_test_slices[inds_order]
    slice_probs = slice_probs[inds_order]

    test_pop_idx = test_data['population_idx']

    for k in config["k"]:
        precision_at_k_results = precision_at_k(
            population_ids=test_pop_idx,
            slices_preds=slice_labels,
            slices_probs=slice_probs,
            k=k,
            save_dir=config["saving_path"],
        )

        print(f"Precision at {k} results:\n", precision_at_k_results)
        average_precision = np.mean(list(precision_at_k_results[pop]["precision"] for pop in precision_at_k_results.keys()))
        print(f"Average Precision at {k}: {average_precision:.4f}")

    representative_slices = precision_at_k(
        population_ids=test_pop_idx,
        slices_preds=slice_labels,
        slices_probs=slice_probs,
        k=10,
        save_dir=None,
    )
    stats_dict = {}
    error_slices = [representative_slices[pop_idx]["cluster_id"] for pop_idx in representative_slices.keys() if pop_idx in config['error_pop_inds']]
    assert -1 not in error_slices, "Error slices should not contain -1"

    stats_dict["error_slices_purity_score"] = calc_error_slices_purity_score(
        error_slice_index=error_slices,
        population_indices=test_pop_idx,
        slice_preds=slice_labels,
    )
    stats_dict["underperforming_populations_completeness_score"] = calc_error_slices_perpopulation_completeness_score(
        target_population_index=config['error_pop_inds'],
        population_indices=test_pop_idx,
        slice_preds=slice_labels,
        n_slices=config['n_slices'],
    )

    # Save homogeneity scores
    stats_path = os.path.join(config["saving_path"], "metric_scores.csv")
    pd.DataFrame(stats_dict, index=[0]).to_csv(stats_path, index=False)
    
    # Plot Only Erroneous Samples
    # plot_slices(
    #     embeddings=test_data['embeddings'][error_mask],
    #     targets=test_data['labels'][error_mask],
    #     pred_labels=test_data['y_preds'][error_mask],
    #     slices=slice_labels[error_mask],
    #     saving_path=config["saving_path"],
    #     dim_reduction_method='tsne',
    #     fig_name='error_slices_tsne_plot_spotlight',
    #     title="Predicted Error Slices Visualisation",
    # )
    print("Done!")

if __name__ == "__main__":
    
    experiment_path = "/homes/ea685/new_baseline/data/MNIST/MNIST-SUM/CBM_Joint/"
    evaluation_folder_name = "Evaluations"
    val_df_path = os.path.join(experiment_path, evaluation_folder_name, "val_eval_df.pkl")
    test_df_path = os.path.join(experiment_path, evaluation_folder_name, "val_eval_df.pkl")

    n_slices = [4]
    for s in n_slices:
        saving_path = os.path.join(*[experiment_path, evaluation_folder_name, "new_slices", f"george_k={s}"])
        parent_path = os.path.dirname(saving_path)
        if not os.path.exists(parent_path):
            print(f"Creating saving directory: {parent_path}")
            os.mkdir(parent_path)
        if not os.path.exists(saving_path):
            print(f"Creating saving directory: {saving_path}")
            os.mkdir(saving_path)
        
        val_df = pd.read_pickle(val_df_path)
        test_df = pd.read_pickle(test_df_path)

        for seed in [42, 77, 666, 123, 58]:
            saving_path_seed = saving_path + f"/seed_{seed}"
            if not os.path.exists(saving_path_seed):
                print(f"Creating saving directory: {saving_path_seed}")
                os.mkdir(saving_path_seed)
            config = {
                "n_slices": s,
                "saving_path": saving_path_seed,
                'seed': seed,
                'k': [5, 10],
                'error_pop_inds': [6, 12],
            }

            main(
                val_data=val_df,
                test_data=test_df,
                config=config,
            )
