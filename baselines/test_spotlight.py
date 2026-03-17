from domino import SpotlightSlicer


import os
import pandas as pd
import numpy as np
import torch

from utils.plotting import plot_slices
from utils.metrics import (
    precision_at_k,
    calc_error_slices_purity_score,
    calc_error_slices_perpopulation_completeness_score,
)
from utils.utils import data_preprocessing


def set_seed(seed):
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main(val_data=None, test_data=None, config=None):
    val_data = data_preprocessing(val_data)
    test_data = data_preprocessing(test_data)

    error_mask = test_data['y_preds'] != test_data['labels']
    print("Error Rate: ", np.mean(test_data['y_preds'] != test_data['labels']))
    
    set_seed(config.get("seed", 123))

    slicer = SpotlightSlicer(
        n_slices=config["n_slices"],
        n_steps=config.get("n_steps", 5),
        spotlight_size=config.get("spotlight_size", 0.02),
    )

    val_loss = val_data['target_loss'] if 'target_loss' in val_data else val_data['loss']
    test_loss = test_data['target_loss'] if 'target_loss' in test_data else test_data['loss']
    
    slicer.fit(data=None, embeddings=val_data['embeddings'], losses=val_loss)
    slices = slicer.predict(data=None, embeddings=test_data['embeddings'], losses=test_loss)
    slice_probs = slicer.predict_proba(data=None, embeddings=test_data['embeddings'], losses=test_loss)
    slice_labels = slices

    test_population_idx = test_data['population_idx']
    test_targets = test_data['labels']
    test_pred_labels = test_data['y_preds']
    test_embeddings = test_data['embeddings']

    for k in config["k"]:
        precision_at_k_results = precision_at_k(
            population_ids=test_population_idx,
            slices_preds=slice_labels,
            slices_probs=slice_probs,
            k=k,
            save_dir=config["saving_path"],
        )

        print(f"Precision at {k} results:\n", precision_at_k_results)
        average_precision = np.mean(list(precision_at_k_results[pop]["precision"] for pop in precision_at_k_results.keys()))
        print(f"Average Precision at {k}: {average_precision:.4f}")

    stats_dict = {}
    representative_slices = precision_at_k(
        population_ids=test_population_idx,
        slices_preds=slice_labels,
        slices_probs=slice_probs,
        k=10,
        save_dir=None,
    )
    error_slices = [representative_slices[pop_idx]["cluster_id"] for pop_idx in representative_slices.keys() if pop_idx in config['error_pop_inds']]
    assert -1 not in error_slices, "Error slices should not contain -1"
    stats_dict["error_slices_homogeneity_score"] = calc_error_slices_purity_score(
        error_slice_index=error_slices,
        population_indices=test_population_idx,
        slice_preds=slice_labels,
    )
    stats_dict["underperforming_populations_completeness_score"] = calc_error_slices_perpopulation_completeness_score(
        target_population_index=config['error_pop_inds'],
        population_indices=test_population_idx,
        slice_preds=slice_labels,
        n_slices=config['n_slices'],
    )

    # Save homogeneity scores
    stats_path = os.path.join(config["saving_path"], "metric_scores.csv")
    pd.DataFrame(stats_dict, index=[0]).to_csv(stats_path, index=False)

    # # Plot Only Erroneous Samples
    # plot_slices(
    #     embeddings=test_embeddings[error_mask],
    #     targets=test_targets[error_mask],
    #     pred_labels=test_pred_labels[error_mask],
    #     slices=slice_labels[error_mask],
    #     saving_path=config["saving_path"],
    #     dim_reduction_method='tsne',
    #     fig_name='error_slices_tsne_plot_spotlight',
    #     title="Predicted Error Slices Visualisation",
    # )
    print("Done!")

if __name__ == "__main__":
    
    experiment_path = "./experiments/cbm/ISIC/isic_cbm_seq_c22_1024_concept-only/20260116-231121_isic_cbm_vanilla/cbm/ISIC/train_cbm_with_full_spur_feb/20260221-100735_cbm_isic_hard_sgd_bs32_lr0.01_wd0.001_seed42-train_cbm_with_full_spur_feb/"
    evaluation_folder_name = "Evaluations"
    val_df_path = os.path.join(experiment_path, evaluation_folder_name, "val_eval_df.pkl")
    test_df_path = os.path.join(experiment_path, evaluation_folder_name, "val_eval_df.pkl")

    val_df = pd.read_pickle(val_df_path)
    test_df = pd.read_pickle(test_df_path)

    for s in [2]:
        saving_path = os.path.join(*[experiment_path, evaluation_folder_name, "Slices", f"spotlight_k={s}"])
        parent_path = os.path.dirname(saving_path)
        if not os.path.exists(parent_path):
            print(f"Creating saving directory: {parent_path}")
            os.mkdir(parent_path)
        if not os.path.exists(saving_path):
            print(f"Creating saving directory: {saving_path}")
            os.mkdir(saving_path)



        for seed in [42, 77, 666, 123, 58]:
            s_saving_path = os.path.join(saving_path, f"seed_{seed}")
            if not os.path.exists(s_saving_path):
                print(f"Creating saving directory: {s_saving_path}")
                os.mkdir(s_saving_path)
            config = {
                "n_slices": s,
                "n_steps": 5,
                "saving_path": s_saving_path,
                'k': [5, 10],
                "seed": seed,
                "error_pop_inds": [0, 3],
                "spotlight_size": 0.02,
            }

            main(
                val_data=val_df,
                test_data=test_df,
                config=config,
            )
