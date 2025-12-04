
import os
import pandas as pd
import numpy as np
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from omegaconf import DictConfig
from collections import defaultdict
from utils.plotting import plot_slices
from utils.metrics import (
    precision_at_k,
    calc_error_slices_purity_score,
    calc_error_slices_perpopulation_completeness_score,
    find_slice_key_concepts,
)
from utils.utils import data_preprocessing


def main(data, meta_config, experiment_config: DictConfig):
    processed_data = data_preprocessing(data)
    targets = processed_data['labels']
    embeddings = processed_data['embeddings']
    slices = processed_data['GMM_stats:cluster_id']
    slice_probs = processed_data['GMM_stats:cluster_probs']
    population_idx = processed_data['population_idx']
    # population_names = processed_data['population_name']
    pred_labels = processed_data['y_preds']  # Ensure pred_labels is a 2D array
    
    # print("Population Names: ", np.unique(population_names))
    print("Error Rate: ", np.mean(pred_labels != targets))
    error_mask = pred_labels != targets

    for k in meta_config["k"]:
        precision_at_k_results = precision_at_k(
            population_ids=population_idx,
            slices_preds=slices,
            slices_probs=slice_probs,
            k=k,
            save_dir=meta_config["saving_path"],
        )

        print(f"Precision at {k} results:\n", precision_at_k_results)
        average_precision = np.mean(list(precision_at_k_results[pop]["precision"] for pop in precision_at_k_results.keys()))
        print(f"Average Precision at {k}: {average_precision:.4f}")

    representative_slices = precision_at_k(
        population_ids=population_idx,
        slices_preds=slices,
        slices_probs=slice_probs,
        k=10,
        save_dir=None,
    )

    stats_dict = {}
    error_slices = [representative_slices[pop_idx]["cluster_id"] for pop_idx in representative_slices.keys() if pop_idx in meta_config['error_pop_inds']]
    assert -1 not in error_slices, "Error slices should not contain -1"

    stats_dict["error_slices_purity_score"] = calc_error_slices_purity_score(
        error_slice_index=error_slices,
        population_indices=population_idx,
        slice_preds=slices,
    )
    stats_dict["underperforming_populations_completeness_score"] = calc_error_slices_perpopulation_completeness_score(
        target_population_index=meta_config['error_pop_inds'],
        population_indices=population_idx,
        slice_preds=slices,
        n_slices=experiment_config.model.gmm_params.n_clusters,
    )

    # Save homogeneity scores
    stats_path = os.path.join(meta_config["saving_path"], "metric_scores.csv")
    pd.DataFrame(stats_dict, index=[0]).to_csv(stats_path, index=False)
    

    find_slice_key_concepts(
        gmm_eval_dict=processed_data,
        experiment_config=experiment_config,
        max_rep_concepts=meta_config["n_representative_concepts"],
        semantic_concepts=meta_config["semantic_concepts"],
        save_dir=meta_config["saving_path"],
    )

    # Plot error slices
    plot_slices(
        embeddings=embeddings[error_mask],
        targets=targets[error_mask], 
        pred_labels=pred_labels[error_mask],
        slices=slices[error_mask], 
        saving_path=meta_config["saving_path"], 
        dim_reduction_method='tsne',
        fig_name='error_slices_tsne_plot_gmm',
        title="Predicted Error Slices Visualisation",
        # fig_size=(4,4),
    )

    plot_slices(
        embeddings=embeddings[error_mask],
        targets=targets[error_mask], 
        pred_labels=pred_labels[error_mask],
        slices=population_idx[error_mask], 
        saving_path=meta_config["saving_path"], 
        # slices_names=population_names[error_mask],
        dim_reduction_method='tsne',
        fig_name='populations_error_tsne_plot',
        title="Population Error Slices Visualization",
        plot_legend=False,
        fig_size=(4, 4),
    )

    print("Done!")

if __name__ == "__main__":

    unify_results = False

    experiments_path = [
        './experiments/cbm/Waterbirds/20251204-110018_CBM_debugging_code_flow/GMM/20251204-113817_GMM_debugging_code_flow/',
    ]

    for exp_path in experiments_path:
        val_df_path = os.path.join(exp_path, "MixtureSlicer", "val_gmm_eval_df.pkl")
        saving_path = os.path.join(exp_path, "MixtureSlicer", "concept_aware_slices")
        if not os.path.exists(saving_path):
            print(f"Creating saving directory: {saving_path}")
            os.mkdir(saving_path)
        
        val_df = pd.read_pickle(val_df_path)
        
        with open(os.path.join(exp_path, "config.yaml"), "r") as f:
            experiment_config = yaml.load(f, Loader=yaml.FullLoader)
            experiment_config = DictConfig(experiment_config)

        config = {
            "saving_path": saving_path,
            "k": [5, 10],  # Number of top slices to consider for precision at k 
            # "representative_threshold": 0.7,  # Threshold for representative slices
            "n_representative_concepts": 5,
            "error_pop_inds": [1, 2],
            "semantic_concepts": "waterbirds_with_background",
        }

        main(
            data=val_df,
            meta_config=config,
            experiment_config=experiment_config,
        )
    
    if unify_results:
        # Gather all results that are stored in MixtureSlicer/concept_aware_slices/ and save them in a single folder
        result_subpath = os.path.join("MixtureSlicer", "concept_aware_slices")
        suffix_to_remove = experiments_path[0].split("_")[-1]
        new_experiments_path = experiments_path[0].replace(suffix_to_remove, "")[:-1]
        all_results_path = os.path.join(new_experiments_path, result_subpath)
        
        if not os.path.exists(all_results_path):
            print(f"Creating saving directory: {all_results_path}")
            os.makedirs(all_results_path, exist_ok=True)

        all_results = []
        for result_path in experiments_path:
            experiment_seed = result_path.split("_")[-1].replace("/", "").replace(":", "_")
            # Create a new folder for the experiment seed
            experiment_seed_path = os.path.join(*[new_experiments_path, result_subpath, experiment_seed])
            # Copy the results to the new folder
            shutil.copytree(os.path.join(result_path, result_subpath), experiment_seed_path)
