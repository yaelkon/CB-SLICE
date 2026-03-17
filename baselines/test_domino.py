
import os
import pandas as pd
import numpy as np
import torch
import clip
import PIL
from PIL import Image
from tqdm import tqdm
from domino import DominoSlicer
from utils.plotting import plot_slices
from utils.metrics import (
    precision_at_k,
    calc_error_slices_purity_score,
    calc_error_slices_perpopulation_completeness_score,
)
from utils.utils import data_preprocessing


def main(val_data=None, config=None):

    error_mask = val_data['y_preds'] != val_data['labels']
    print("Error Rate: ", np.mean(error_mask))

    domino = DominoSlicer(
                n_slices=config['n_slices'],
                n_mixture_components=config["n_mixture_components"],
                covariance_type=config["covariance_type"],
                n_pca_components=config["n_pca_components"],
                init_params=config["init_params"],
                random_state=config["seed"],  # Set a random state for reproducibility
                y_log_likelihood_weight=config["y_log_likelihood_weight"],
                y_hat_log_likelihood_weight=config["y_hat_log_likelihood_weight"],
        )

    domino.fit(data=None, embeddings=val_data['clip_image_embeddings'], targets=val_data['labels'], pred_probs=val_data["y_scores"])
    slices = domino.predict(data=None, embeddings=val_data['clip_image_embeddings'], targets=val_data['labels'], pred_probs=val_data["y_scores"])
    slice_probs = domino.predict_proba(data=None, embeddings=val_data['clip_image_embeddings'], targets=val_data['labels'], pred_probs=val_data["y_scores"])
    slice_labels = np.argmax(slices, axis=1)

    test_pop_idx = val_data['population_idx']
    test_labels = val_data['labels']
    test_pred_labels = val_data['y_preds']
    test_embeddings = val_data['embeddings']
        
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

    stats_dict = {}
    representative_slices = precision_at_k(
        population_ids=test_pop_idx,
        slices_preds=slice_labels,
        slices_probs=slice_probs,
        k=10,
        save_dir=None,
    )
    error_slices = [representative_slices[pop_idx]["cluster_id"] for pop_idx in representative_slices.keys() if pop_idx in config['error_pop_inds']]
    assert -1 not in error_slices, "Error slices should not contain -1"
    
    stats_dict["error_slices_homogeneity_score"] = calc_error_slices_purity_score(
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
    
    # # Plot Only Erroneous Samples
    # plot_slices(
    #     embeddings=test_embeddings[error_mask],
    #     targets=test_labels[error_mask],
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

    val_df = pd.read_pickle(val_df_path)
    val_data = data_preprocessing(val_df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Add clip embeddings to val_data_df and test_data_df
    model, preprocess = clip.load("ViT-B/32", device=device)
    clip_image_embeddings_list = []

    if "img_path" in val_data:
        field_name = "img_path"
        val_data["img_path"] = list(val_data["img_path"])
    elif "img" in val_data:
        field_name = "img"
    else:
        raise ValueError("No image path or image found in data")
        
    for img_sample in tqdm(val_data[field_name], desc="Adding CLIP embeddings to val_data"):
        if field_name == "img_path":
            image = PIL.Image.open(img_sample)
        else:
            image = Image.fromarray(img_sample)

        image = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            clip_image_features = model.encode_image(image)
        clip_image_embeddings_list.append(clip_image_features.squeeze(0).cpu().numpy())
    
    clip_image_embeddings = np.array(clip_image_embeddings_list)
    val_data["clip_image_embeddings"] = clip_image_embeddings

    n_slices = [2]
    for s in n_slices:
        saving_path = os.path.join(*[experiment_path, evaluation_folder_name, "Slices", f"domino_pca_weights:40_k={s}"])
        parent_path = os.path.dirname(saving_path)
        if not os.path.exists(parent_path):
            print(f"Creating saving directory: {parent_path}")
            os.mkdir(parent_path)
        if not os.path.exists(saving_path):
            print(f"Creating saving directory: {saving_path}")
            os.mkdir(saving_path)

        for seed in [42, 77, 666, 123, 58]:
            saving_path_seed = saving_path + f"/seed_{seed}"
            if not os.path.exists(saving_path_seed):
                print(f"Creating saving directory: {saving_path_seed}")
                os.mkdir(saving_path_seed)
            config = {
                "n_slices": s,
                "n_mixture_components": s + 10,
                "covariance_type": "diag",
                "n_pca_components": 128,
                "init_params": "kmeans",
                "saving_path": saving_path_seed,
                'seed': seed,
                'k': [5, 10],
                'error_pop_inds': [0, 3],
                'y_log_likelihood_weight': 40,
                'y_hat_log_likelihood_weight': 40,
                # [2, 6, 12],
                'evaluate_erroneous_only': False,
            }

            main(
                val_data=val_data,
                config=config,
            )
