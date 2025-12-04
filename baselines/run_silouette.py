import numpy as np
import torch
import clip
import PIL
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from domino import MixtureSlicer, SpotlightSlicer
from utils import data_preprocessing, converter
from umap import UMAP
import pandas as pd

algos = ["kmeans", "domino", "spotlight"]
clusters = [2,4, 6, 8, 10, 12, 14, 16, 18, 20]
df_path = "./experiments/cbm/Waterbirds/Waterbirds_Attributes_with_Background/20251012-212000_Task_Baseline_wd:0.00001_lr:0.01_decrease:40_sgd/Evaluations_valEqTrainV3/val_eval_df.pkl"
df = pd.read_pickle(df_path)

data = data_preprocessing(df)
embeddings = data['embeddings']
labels = data['labels']
y_probs = data['y_scores']
loss = data['target_loss'] if 'target_loss' in data else data['loss']


if "kmeans" in algos:
        print("Running KMEANS")
        for k in clusters:
                k_means_slicer = KMeans(n_clusters=k)
                _ = k_means_slicer.fit_predict(embeddings)
                slices = k_means_slicer.predict(embeddings)
                silhouette_avg = silhouette_score(embeddings, slices)
                print(
                "For n_clusters =",
                k,
                "The average silhouette_score is :",
                silhouette_avg,
                )

if "domino" in algos:
        print("Running DOMINO")
        # Add clip embeddings to val_data_df and test_data_df
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, preprocess = clip.load("ViT-B/32", device=device)
        clip_image_embeddings_list = []
        
        if "img_path" in data:
                field_name = "img_path"
        elif "img" in data:
                field_name = "img"
        else:
                raise ValueError("No image path or image found in data")
        
        for img_sample in data[field_name]:
                if field_name == "img_path":
                        image = PIL.Image.open(img_sample)
                else:
                        image = Image.fromarray(img_sample)
                        
                image = preprocess(image).unsqueeze(0).to(device)
                with torch.no_grad():
                        clip_image_features = model.encode_image(image)
                clip_image_embeddings_list.append(clip_image_features.squeeze(0).cpu().numpy())
        
        clip_image_embeddings = np.array(clip_image_embeddings_list)
        for k in clusters:
                domino = MixtureSlicer(
                                n_slices=k,
                                n_mixture_components=k+10,
                                covariance_type="diag",
                                n_pca_components=128,
                                y_log_likelihood_weight=40,
                                y_hat_log_likelihood_weight=40,
                                max_iter=200,
                                # Set a random state for reproducibility
                        )

                domino.fit(data=None, embeddings=clip_image_embeddings, targets=labels, pred_probs=y_probs)
                slices = domino.predict(data=None, embeddings=clip_image_embeddings, targets=labels, pred_probs=y_probs)
                slice_labels = np.argmax(slices, axis=1)
                silhouette_avg = silhouette_score(clip_image_embeddings, slice_labels)
                print(
                        "For n_clusters =",
                        k,
                        "The average silhouette_score is :",
                        silhouette_avg,
                )

if "george" in algos:
        print("RUN GEORGE")
        # Train the clustering algorithm
        classes = np.unique(labels)
        # george_clusters = [c // len(classes) for c in clusters]
        umap_model = UMAP(
            n_components=2,
            n_neighbors=10,  # Hyperparameter as was used in the paper
            min_dist=0,  # Hyperparameter as was used in the paper
            random_state=42,
        )
        embeddings_reduced = umap_model.fit_transform(embeddings)
        # print(f"Checking George clusters per class: {george_clusters}")
        for k in clusters:
                silhouette_scores_per_k = []
                for c in classes:
                        c_embeddings = embeddings_reduced[labels == c]
                        george_slicer = GaussianMixture(n_components=k)
                        _ = george_slicer.fit_predict(c_embeddings)
                        slices = george_slicer.predict(c_embeddings)
                        silhouette_avg = silhouette_score(c_embeddings, slices)
                        silhouette_scores_per_k.append(silhouette_avg)
                
                print(f"For k = {k}, the average silhouette_score is : {sum(silhouette_scores_per_k)/len(silhouette_scores_per_k)}")
                
if "spotlight" in algos:
        print("RUNNING SPOTLIGHT")
        for k in clusters:
                slicer = SpotlightSlicer(
                n_slices=k,
                n_steps=5,
        )
                slicer.fit(data=None, embeddings=embeddings, losses=loss)
                slice_labels = slicer.predict(data=None, embeddings=embeddings, losses=loss)
                silhouette_avg = silhouette_score(embeddings, slice_labels)
                print(
                        "For n_clusters =",
                        k,
                        "The average silhouette_score is :",
                        silhouette_avg,
                )