
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

from utils.plotting import plot_slices
from utils.metrics import (
    precision_at_k,
    calc_error_slices_purity_score,
    calc_error_slices_perpopulation_completeness_score,
)
from utils.utils import data_preprocessing


def main(val_data, test_data, config=None):
    val_data = data_preprocessing(val_data)
    test_data = data_preprocessing(test_data)
    val_embeddings = val_data['embeddings']
    test_embeddings = test_data['embeddings']
    test_targets = test_data['labels']
    test_pred_labels = test_data['y_preds']
    test_population_idx = test_data['population_idx']

    print("Error Rate: ", np.mean(test_pred_labels != test_targets))

    k_means_slicer = KMeans(n_clusters=config['n_slices'], random_state=config['seed'])
    _ = k_means_slicer.fit_predict(val_embeddings)
    slices = k_means_slicer.predict(test_embeddings)
    # Plot error slices
    error_mask = test_pred_labels != test_targets

    # # Plot Only Erroneous Samples
    # plot_slices(
    #     embeddings=test_embeddings[error_mask],
    #     targets=test_targets[error_mask],
    #     pred_labels=test_pred_labels[error_mask],
    #     slices=slices[error_mask],
    #     saving_path=config["saving_path"],
    #     dim_reduction_method='tsne',
    #     fig_name='error_slices_tsne_plot_kmeans',
    #     title="Predicted Error Slices Visualisation",
    # )

    for k in config["k"]:
        precision_at_k_results = precision_at_k(
            population_ids=test_population_idx,
            slices_preds=slices,
            slices_probs=None,
            k=k,
            save_dir=config["saving_path"],
        )

        print(f"Precision at {k} results:\n", precision_at_k_results)
        average_precision = np.mean(list(precision_at_k_results[pop]["precision"] for pop in precision_at_k_results.keys()))
        print(f"Average Precision at {k}: {average_precision:.4f}")

    stats_dict = {}
    representative_slices = precision_at_k(
        population_ids=test_population_idx,
        slices_preds=slices,
        slices_probs=None,
        k=10,
        save_dir=None,
    )
    error_slices = [representative_slices[pop_idx]["cluster_id"] for pop_idx in representative_slices.keys() if pop_idx in config['error_pop_inds']]
    assert -1 not in error_slices, "Error slices should not contain -1"
    stats_dict["error_slices_homogeneity_score"] = calc_error_slices_purity_score(
        error_slice_index=error_slices,
        population_indices=test_population_idx,
        slice_preds=slices,
    )
    stats_dict["underperforming_populations_completeness_score"] = calc_error_slices_perpopulation_completeness_score(
        target_population_index=config['error_pop_inds'],
        population_indices=test_population_idx,
        slice_preds=slices,
        n_slices=config['n_slices'],
    )
    # Save statistics
    stats_path = os.path.join(config["saving_path"], "metric_scores.csv")
    pd.DataFrame(stats_dict, index=[0]).to_csv(stats_path, index=False)

    print("Done!")


if __name__ == "__main__":
    
    experiment_path = "./experiments/cbm/ISIC/isic_cbm_seq_c22_1024_concept-only/20260116-231121_isic_cbm_vanilla/cbm/ISIC/train_cbm_with_full_spur_feb/20260221-100735_cbm_isic_hard_sgd_bs32_lr0.01_wd0.001_seed42-train_cbm_with_full_spur_feb/"
    evaluation_folder_name = "Evaluations"
    val_df_path = os.path.join(experiment_path, evaluation_folder_name, "val_eval_df.pkl")
    test_df_path = os.path.join(experiment_path, evaluation_folder_name, "val_eval_df.pkl")
    
    n_slices = [2]
    for s in n_slices:
        saving_path = os.path.join(*[experiment_path, evaluation_folder_name, "Slices", f"kmeans_k={s}"])
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
                "k": [5, 10],
                "error_pop_inds": [0, 3],
            }

            main(
                val_data=val_df,
                test_data=test_df,
                config=config,
            )
