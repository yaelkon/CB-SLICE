import numpy as np
import matplotlib.pyplot as plt

from umap import UMAP
from os.path import join as pjoin
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def plot_slice_population_proportions(slices, population_idx, targets, pred_labels, saving_path):
    """
    Plot a histogram showing the proportion of population indices within each slice,
    split into correct and error predictions.

    Args:
        slices (array-like): Array of slice indices for each sample.
        population_idx (array-like): Array of population indices for each sample.
        targets (array-like): Array of true labels for each sample.
        pred_labels (array-like): Array of predicted labels for each sample.
        saving_path (str): Path to save the histogram plot.
    """
    # Get unique slices and population indices
    unique_slices = np.unique(slices)
    unique_populations = np.unique(population_idx)

    # Assign a unique color to each population
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_populations)))  # Use a colormap
    population_color_map = {pop_idx: colors[i] for i, pop_idx in enumerate(unique_populations)}

    # Calculate proportions for each slice, population, and prediction correctness
    correct_proportions = []
    error_proportions = []
    for slice_idx in unique_slices:
        slice_indices = np.where(slices == slice_idx)[0]
        if len(slice_indices) < 100:
            continue
        slice_population = population_idx[slice_indices]
        slice_targets = targets[slice_indices]
        slice_pred_labels = pred_labels[slice_indices]

        total_slice_count = len(slice_indices)
        slice_correct_proportions = []
        slice_error_proportions = []

        for pop_idx in unique_populations:
            pop_indices = np.where(slice_population == pop_idx)[0]
            # if len(pop_indices) == 0:
            #     continue
            pop_targets = slice_targets[pop_indices]
            pop_pred_labels = slice_pred_labels[pop_indices]

            # Calculate correct and error proportions
            correct_count = np.sum(pop_targets == pop_pred_labels)
            error_count = np.sum(pop_targets != pop_pred_labels)
            # total_count = len(pop_indices)

            slice_correct_proportions.append(correct_count / total_slice_count if total_slice_count > 0 else 0)
            slice_error_proportions.append(error_count / total_slice_count if total_slice_count > 0 else 0)

        correct_proportions.append(slice_correct_proportions)
        error_proportions.append(slice_error_proportions)

    correct_proportions = np.array(correct_proportions)
    error_proportions = np.array(error_proportions)

    # Plot the histogram
    bar_width = 1 / len(unique_populations)  # Adjust bar width for grouped bars
    x = np.arange(len(unique_slices))  # X positions for slices

    plt.figure(figsize=(20, 10))
    for i, pop_idx in enumerate(unique_populations):
        # Plot correct predictions
        plt.bar(
            x + i * bar_width,
            correct_proportions[:, i],
            width=bar_width,
            color=population_color_map[pop_idx],
            label=f"pop {pop_idx}",
            align='center',
            # alpha=0.8
        )

        # Plot error predictions (stacked on top of correct predictions)
        plt.bar(
            x + i * bar_width,
            error_proportions[:, i],
            width=bar_width,
            color=population_color_map[pop_idx],
            edgecolor='black',
            hatch='//',  # Add hatching to distinguish errors
            # label=f"Population {pop_idx} (Error)" if i == 0 else None,
            bottom=correct_proportions[:, i],
            align='center',
            # alpha=0.8
        )

    # Add labels and legend
    plt.xlabel("Slices")
    plt.ylabel("Proportion")
    plt.title("Proportion of Population Indices Within Each Slice")
    plt.xticks(x + bar_width * (len(unique_populations) - 1) / 2, unique_slices)
    plt.legend(title="Population IDs")
    plt.tight_layout()

    # Save the plot
    plt.savefig(pjoin(saving_path, "slice_population_proportions.png"))
    plt.close()


def plot_slices(
        embeddings,
        slices,
        saving_path,
        dim_reduction_method="tsne",
        fig_name='',
        slices_names=None,
        title="Slices Visualization",
        plot_legend=False,
        fig_size=(4,4),
        perplexity=10,
        learning_rate=100,
        ):
    """
    Plot the slices using the specified dimensionality reduction method.
    """

    if dim_reduction_method == "tsne":
        reducer = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, init='pca', random_state=42, verbose=1)   
    elif dim_reduction_method == "pca":
        reducer = PCA(n_components=2,)
    elif dim_reduction_method == "umap":
        reducer = UMAP(n_components=2, n_neighbors=100, min_dist=0.1, random_state=42, verbose=True)
    else:
        raise ValueError("Invalid dimensionality reduction method")

    reduced_embeddings = reducer.fit_transform(embeddings)
# Prepare unique colors for each slice index
    unique_slices, counts = np.unique(slices, return_counts=True)
    order = np.argsort(-counts, kind="mergesort")
    sorted_slices = unique_slices[order]
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_slices)))  # Use a colormap with enough distinguishable colors
    
    # 1) representative y per cluster (use median for robustness)
    # cluster_y = {c: np.median(reduced_embeddings[slices == c, 1]) for c in unique_slices}
    # 2) sort clusters from top (largest y) to bottom (smallest y)
    # sorted_slices = sorted(unique_slices, key=lambda c: cluster_y[c], reverse=True)
    # 3) assign colors in that order (wrap around if not enough colors)
    slice_color_map = {c: colors[i] for i, c in enumerate(sorted_slices)}

    # Create the scatter plot
    plt.figure(figsize=fig_size, constrained_layout=True)
    for slice_idx in unique_slices:
        # Get indices for the current slice
        slice_indices = np.where(slices == slice_idx)[0]

        # Extract data for the current slice
        slice_embeddings = reduced_embeddings[slice_indices]
        slice_name = slices_names[slice_indices] if slices_names is not None else None
        # Plot correct samples
        plt.scatter(
            slice_embeddings[:, 0],
            slice_embeddings[:, 1],
            color=slice_color_map[slice_idx],
            marker='o',  # Circle for correct samples
            edgecolor='none',
            alpha=0.8,
            label=f"{slice_idx}" if slice_name is None else f"{slice_name[0]}",  # Use slice name if provided
        )

    # Add legend for slices
    if plot_legend:
        plt.legend(title="Slice Index", loc="lower right", fontsize='x-large', markerscale=1.8, title_fontsize='x-large')
    
    # Save the plot
    # plt.title(f"{title} - {dim_reduction_method.upper()}", size=24)

    plt.xticks([])  # Remove x-axis ticks
    plt.yticks([])  # Remove y-axis ticks
    # plt.xlabel("t-SNE 1", size=18)
    # plt.ylabel("t-SNE 2", size=18)
    if fig_name == "":
        fig_name = f"slice_{dim_reduction_method}_plot"
    
    plt.tight_layout()
    plt.savefig(pjoin(saving_path, fig_name + ".png"), dpi=300, bbox_inches='tight')
    plt.savefig(pjoin(saving_path, fig_name + ".pdf"), dpi=300, bbox_inches='tight')
    plt.close()
