"""
Utility functions for plotting.
"""

import io
import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import plotly.graph_objects as go
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
import wandb


def plot_heatmap(
    matrix, flipped_matrix, labels_names, labels_names_gt, log_name="Correlation Matrix"
):
    # Create the heatmap using Plotly
    fig = go.Figure(
        data=go.Heatmap(
            z=flipped_matrix,
            x=labels_names,  # Column labels
            y=labels_names_gt[::-1],  # Row labels
            colorscale="RdBu_r",  # Color scale similar to Seaborn's 'coolwarm'
            zmin=-1,
            zmax=1,  # Assuming correlation values range from -1 to 1
        )
    )

    # Update layout for better readability at large scales
    fig.update_layout(
        autosize=False,
        width=800,  # Maintain square aspect
        height=800,
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            scaleanchor="y",
            constrain="domain",
        ),
        yaxis=dict(showgrid=False, zeroline=False, constrain="domain"),
        margin=dict(l=10, r=10, t=90, b=10),  # Reduced margins
        plot_bgcolor="rgba(0,0,0,0)",  # Set background color to transparent
    )
    fig.update_xaxes(fixedrange=False)  # Allow x-axis to be zoomed independently
    fig.update_yaxes(fixedrange=False)  # Allow y-axis to be zoomed independently

    wandb.log({f"Interactive {log_name}": wandb.Plotly(fig)})
    # Assuming c_corr is your correlation matrix
    sns.set_theme(style="white")

    # Create a heatmap
    plt.figure(figsize=(20, 16))
    sns.heatmap(
        matrix,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
    )  # , yticklabels=concept_names_graph, xticklabels=concept_names_graph
    plt.title("Correlation Matrix Heatmap")

    # Save the heatmap to a file
    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    wandb.log({log_name: wandb.Image(Image.open(buf))})


def compute_and_plot_heatmap(
    matrix, concepts_true, concept_names_graph, config, log_name=None
):
    # Reorder CUB concepts to group colors&shapes instead of concept groups
    if config.data.dataset == "CUB":
        new_group = np.array(
            [
                (name.split(": ")[1] if not name.isdigit() else name)
                for name in concept_names_graph
            ]
        )
        unique_groups, index, counts = np.unique(
            new_group, return_counts=True, return_index=True
        )
        # reorder groups and counts to preserve order of unique groups
        unique_groups = unique_groups[np.argsort(index)]
        counts = counts[np.argsort(index)]

        # Get the indices that sort by descending frequency
        sorted_indices = np.argsort(-counts, kind="stable")

        # Use these indices to reorder the unique_groups and counts arrays
        unique_groups = unique_groups[sorted_indices]

        # Get the indices that sort new_group to fit the new order
        new_rowcol = np.argwhere((unique_groups == new_group[:, None]))
        assert (
            new_rowcol[:, 0] == np.arange(len(new_rowcol))
        ).all(), "Error in reordering"
        permutation = np.argsort(new_rowcol[:, 1], kind="stable")
        perm_matrix = matrix[permutation][:, permutation]
        perm_flipped_matrix = np.flipud(perm_matrix)
        perm_flipped_matrix = np.vstack(
            [
                np.append(
                    concepts_true[0].cpu().numpy(),
                )[permutation],
                perm_flipped_matrix,
            ]
        )
        labels_names = [concept_names_graph[i] for i in permutation]
        labels_names_new = np.append(labels_names, "Ground Truth")
        plot_heatmap(
            matrix=perm_matrix,
            flipped_matrix=perm_flipped_matrix,
            labels_names=labels_names,
            labels_names_gt=labels_names_new,
            log_name=(
                "Averaged CUB concept-ordered Correlation Matrix"
                if log_name is not None
                else "CUB concept-ordered Correlation Matrix"
            ),
        )

    # Perform clustering for permutation into cliques (except synthetic with predefined order)
    if config.data.get("sim_type") == "correlated_c":
        permutation = [i for i in range(len(concept_names_graph))]

    else:
        dist_matrix = 1 - ((matrix + matrix.T) / 2)
        linkage_matrix = linkage(dist_matrix, method="complete", optimal_ordering=True)
        dendro = dendrogram(linkage_matrix, no_plot=True)
        permutation = dendro["leaves"]

    # Apply permutation
    perm_matrix = matrix[permutation][:, permutation]
    perm_flipped_matrix = np.flipud(perm_matrix)
    perm_flipped_matrix = np.vstack(
        [
            np.append(
                concepts_true[0].cpu().numpy(),
            )[permutation],
            perm_flipped_matrix,
        ]
    )
    labels_names = [concept_names_graph[i] for i in permutation]
    labels_names_new = np.append(labels_names, "Ground Truth")

    args = {
        "matrix": perm_matrix,
        "flipped_matrix": perm_flipped_matrix,
        "labels_names": labels_names,
        "labels_names_gt": labels_names_new,
    }
    if log_name is not None:
        args["log_name"] = log_name

    plot_heatmap(**args)


class SubpoopulationPlotter:
    def __init__(
        self,
        population_metrics,
        subpopulations_str2idx,
        stage='val',
        save_path='./',
        log_scale=False,
        figsize=(30, 10)
    ):

        self.population_metrics = population_metrics
        self.subpopulations_str2idx = subpopulations_str2idx
        self.log_scale = log_scale
        self.figsize = figsize
        self.stage = stage

        if save_path == "":
            save_path = None
        else:
            save_path = os.path.join(*[save_path, "plots", self.stage])
            os.makedirs(save_path, exist_ok=True)
        self.save_path = save_path
        
    def plot(self, plot_uncertainty=False):
        self.plot_subpopulation_losses()
        self.plot_subpopulation_accuracies()
        self.plot_concept_accuracies()
        self.plot_concept_loss()
        if plot_uncertainty:
            self.plot_concept_uncertainty()
            self.plot_concept_prob_vs_uncertainty()
    
    def plot_subpopulation_losses(self):
        
        """
        Plots a bar chart of subpopulation losses for target_loss and total_loss.

        Args:
            population_metrics (dict): A dictionary containing:
                - "target_loss" (list or np.ndarray): Loss values for each subpopulation.
                - "total_loss" (list or np.ndarray): Loss values for each subpopulation.
            save_path (str): Path to save the plot. If empty, the plot is shown instead.
        """
        n_subpopulations = len(self.subpopulations_str2idx)
        x = np.arange(n_subpopulations)  # Subpopulation indices

        # Bar width
        bar_width = 0.4

        # Create a figure
        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot target_loss
        target_loss_bars = ax.bar(
            x - bar_width / 2,  # Offset for target_loss
            self.population_metrics["target_loss"],
            width=bar_width,
            label="Target Loss",
            color="C0",  # Consistent color for target_loss
        )

        # Plot total_loss
        total_loss_bars = ax.bar(
            x + bar_width / 2,  # Offset for total_loss
            self.population_metrics["total_loss"],
            width=bar_width,
            label="Total Loss",
            color="C1",  # Consistent color for total_loss
        )

        # Add value labels on bars
        for bar, value in zip(target_loss_bars, 
                              self.population_metrics["target_loss"]):
            if value > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, 
                        bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        for bar, value in zip(total_loss_bars, 
                              self.population_metrics["total_loss"]):
            if value > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, 
                        bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=8)

        # Set labels, title, and legend
        ax.set_xlabel("Subpopulation Index", fontsize=14)
        ax.set_ylabel("Loss Value", fontsize=14)
        ax.set_title("Subpopulation Losses", fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(self.subpopulations_str2idx.keys(), fontsize=12)
        ax.legend(fontsize=12)

        # Save or show the plot
        if self.save_path is not None:
            plt.savefig(self.save_path + f'/subpopulation_losses.png', format="png")

        wandb.log({f"{self.stage}/subpopulation_losses": wandb.Image(fig)})
        plt.close()

    def plot_subpopulation_accuracies(self, concept_flag=True):
        """
        Plots a bar chart of subpopulation losses for target_loss and total_loss.

        Args:
            population_metrics (dict): A dictionary containing:
                - "target_loss" (list or np.ndarray): Loss values for each subpopulation.
                - "total_loss" (list or np.ndarray): Loss values for each subpopulation.
            save_path (str): Path to save the plot. If empty, the plot is shown instead.
        """
        n_subpopulations = len(self.subpopulations_str2idx)
        x = np.arange(n_subpopulations)  # Subpopulation indices

        # Bar width
        bar_width = 0.4

        # Create a figure
        fig, ax = plt.subplots(figsize=self.figsize)

        if self.log_scale:
            ax.set_yscale('log')

        # Plot target_loss
        task_acc_bars = ax.bar(
            x - bar_width / 2,  # Offset for target_loss
            self.population_metrics["task_accuracy"],
            width=bar_width,
            label="Task Acc" if not self.log_scale else "Task Acc (log)",
            color="C0",  # Consistent color for target_loss
        )

        # Plot total_loss
        # Add value labels on bars
        for bar, value in zip(task_acc_bars, 
                              self.population_metrics["task_accuracy"]):
            if value > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, 
                        bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        if concept_flag:
            concept_acc_bars = ax.bar(
                x + bar_width / 2,  # Offset for total_loss
                self.population_metrics["complete_concept_accuracy"],
                width=bar_width,
                label="Complete Concept Acc" if not self.log_scale else "Complete Concept Acc (log)",
                color="C1",  # Consistent color for total_loss
            )
            
            for bar, value in zip(concept_acc_bars, 
                                self.population_metrics["complete_concept_accuracy"]):
                if value > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, 
                            bar.get_height() + 0.01,
                            f'{value:.3f}', ha='center', va='bottom', fontsize=8)

        # Set labels, title, and legend
        ax.set_xlabel("Subpopulation Index", fontsize=14)
        ylabel = "Task Accuracy" if not self.log_scale else "Task Accuracy (log)"
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title("Task Accuracy Across Subpopulations", fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(self.subpopulations_str2idx.keys(), fontsize=12)
        ax.legend(fontsize=12)

        fig_name = "subpopulation_accuracies" if not self.log_scale else "subpopulation_accuracies_log"
        # Save or show the plot
        if self.save_path is not None:
            plt.savefig(join(*[self.save_path, fig_name + '.png']), format="png")
        wandb.log({join(self.stage, fig_name): wandb.Image(fig)})
        plt.close()

    def plot_concept_accuracies(self):
        """
        Plots a histogram of concept accuracies for each subpopulation.

        Args:
            concept_accuracy (np.ndarray): A 2D array of shape (n_populations, n_concepts),
                                            where each entry represents the accuracy of a concept
                                            in a specific subpopulation.
            save_path (str): Path to save the plot. If empty, the plot is shown instead.
        """
        concept_accuracy = self.population_metrics["concept_accuracy"]
        n_populations, n_concepts = concept_accuracy.shape

        # Create a figure
        fig, ax = plt.subplots(figsize=self.figsize)

        if self.log_scale:
            ax.set_yscale('log')

        # Define bar width and positions
        bar_width = 0.8 / n_concepts  # Divide the bar width among concepts
        x = np.arange(n_populations)  # Subpopulation indices

        # Plot bars for each concept
        concept_bars = []
        for concept_idx in range(n_concepts):
            bars = ax.bar(
                x + concept_idx * bar_width,  # Offset each concept's bars
                concept_accuracy[:, concept_idx],  # Accuracy values for the concept
                width=bar_width,
                label=f"Concept {concept_idx}",
            )
            concept_bars.append(bars)

        # Add value labels on bars
        for concept_idx in range(n_concepts):
            for bar, value in zip(concept_bars[concept_idx], 
                                  concept_accuracy[:, concept_idx]):
                if value > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, 
                            bar.get_height() + 0.01,
                            f'{value:.3f}', ha='center', va='bottom', fontsize=6)

        # Set labels, title, and legend
        ax.set_xlabel("Subpopulation Index", fontsize=14)
        ylabel = "Accuracy" if not self.log_scale else "Accuracy (log)"
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title("Concept Accuracies Across Subpopulations", fontsize=16)
        ax.set_xticks(x + bar_width * (n_concepts - 1) / 2)  # Center ticks
        ax.set_xticklabels(self.subpopulations_str2idx.keys(), fontsize=12)
        ax.legend(title="Concepts", fontsize=12)

        fig_name = "concept_accuracies" if not self.log_scale else "concept_accuracies_log"
        # Save or show the plot
        if self.save_path is not None:
            plt.savefig(self.save_path + '/' + fig_name + '.png', format="png")
        wandb.log({f"{self.stage}/{fig_name}": wandb.Image(fig)})
        plt.close()

    def plot_concept_loss(self):
        """
        Plots a histogram of concept accuracies for each subpopulation.

        Args:
            concept_accuracy (np.ndarray): A 2D array of shape (n_populations, n_concepts),
                                            where each entry represents the accuracy of a concept
                                            in a specific subpopulation.
            save_path (str): Path to save the plot. If empty, the plot is shown instead.
        """
        concept_loss = self.population_metrics["concept_loss"]
        n_populations, n_concepts = concept_loss.shape

        # Create a figure
        fig, ax = plt.subplots(figsize=self.figsize)

        # Define bar width and positions
        bar_width = 0.8 / n_concepts  # Divide the bar width among concepts
        x = np.arange(n_populations)  # Subpopulation indices

        # Plot bars for each concept
        concept_bars = []
        for concept_idx in range(n_concepts):
            bars = ax.bar(
                x + concept_idx * bar_width,  # Offset each concept's bars
                concept_loss[:, concept_idx],  # Accuracy values for the concept
                width=bar_width,
                label=f"Concept {concept_idx}",
            )
            concept_bars.append(bars)

        # Add value labels on bars
        for concept_idx in range(n_concepts):
            for bar, value in zip(concept_bars[concept_idx], 
                                  concept_loss[:, concept_idx]):
                if value > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, 
                            bar.get_height() + 0.01,
                            f'{value:.3f}', ha='center', va='bottom', fontsize=6)

        # Set labels, title, and legend
        ax.set_xlabel("Subpopulation Index", fontsize=14)
        ax.set_ylabel("Concept Loss", fontsize=14)
        ax.set_title("Concept Loss Across Subpopulations", fontsize=16)
        ax.set_xticks(x + bar_width * (n_concepts - 1) / 2)  # Center ticks
        ax.set_xticklabels(self.subpopulations_str2idx.keys(), fontsize=12)
        ax.legend(title="Concepts", fontsize=12)

        # Save or show the plot
        if self.save_path is not None:
            plt.savefig(self.save_path + '/concept_loss.png', format="png")
        wandb.log({f"{self.stage}/concept_loss": wandb.Image(fig)})
        plt.close()

    def plot_concept_uncertainty(self):
        """
        Plots a histogram of concept accuracies for each subpopulation.

        Args:
            concept_accuracy (np.ndarray): A 2D array of shape (n_populations, n_concepts),
                                            where each entry represents the accuracy of a concept
                                            in a specific subpopulation.
            save_path (str): Path to save the plot. If empty, the plot is shown instead.
        """
        concept_uncertainty = self.population_metrics["concept_uncertainty"]
        n_populations, n_concepts = concept_uncertainty.shape

        # Create a figure
        fig, ax = plt.subplots(figsize=self.figsize)

        # Define bar width and positions
        bar_width = 0.8 / n_concepts  # Divide the bar width among concepts
        x = np.arange(n_populations)  # Subpopulation indices

        # Plot bars for each concept
        concept_bars = []
        for concept_idx in range(n_concepts):
            bars = ax.bar(
                x + concept_idx * bar_width,  # Offset each concept's bars
                concept_uncertainty[:, concept_idx],  # Accuracy values for the concept
                width=bar_width,
                label=f"Concept {concept_idx}",
            )
            concept_bars.append(bars)

        # Add value labels on bars
        for concept_idx in range(n_concepts):
            for bar, value in zip(concept_bars[concept_idx], 
                                  concept_uncertainty[:, concept_idx]):
                if value > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, 
                            bar.get_height() + 0.01,
                            f'{value:.3f}', ha='center', va='bottom', fontsize=6)

        # Set labels, title, and legend
        ax.set_xlabel("Subpopulation Index", fontsize=14)
        ax.set_ylabel("Uncertainty", fontsize=14)
        ax.set_title("Concept Uncertainty Across Subpopulations", fontsize=16)
        ax.set_xticks(x + bar_width * (n_concepts - 1) / 2)  # Center ticks
        ax.set_xticklabels(self.subpopulations_str2idx.keys(), fontsize=12)
        ax.legend(title="Concepts", fontsize=12)

        # Save or show the plot
        if self.save_path is not None:
            plt.savefig(self.save_path + '/concept_uncertainty.png', format="png")
        wandb.log({f"{self.stage}/concept_uncertainty": wandb.Image(fig)})
        plt.close()

    def plot_concept_prob_vs_uncertainty(self):    
        """
        Plots concept_prob vs concept_uncertainty for each concept.

        Args:
            concept_prob (np.ndarray): A 2D array of shape (n_samples, n_concepts),
                                    where each entry represents the probability of a concept.
            concept_uncertainty (np.ndarray): A 2D array of shape (n_samples, n_concepts),
                                            where each entry represents the uncertainty of a concept.
            save_path (str): Path to save the plot. If empty, the plot is shown instead.
        """    
        concept_prob = self.population_metrics["concept_prob"]
        concept_uncertainty = self.population_metrics["concept_uncertainty"]
        n_samples, n_concepts = concept_prob.shape

        # Create a figure
        fig, ax = plt.subplots(figsize=self.figsize)

        for concept_idx in range(n_concepts):
            # Sort by uncertainty in descending order
            sorted_indices = np.argsort(-concept_uncertainty[:, concept_idx])
            sorted_prob = concept_prob[sorted_indices, concept_idx]
            sorted_uncertainty = concept_uncertainty[sorted_indices, concept_idx]

            # Plot concept_prob vs concept_uncertainty
            ax.plot(
                sorted_uncertainty,
                sorted_prob,
                label=f"Concept {concept_idx}",
                marker="o",
            )

        # Set labels, title, and legend
        ax.set_xlabel("Concept Uncertainty", fontsize=14)
        ax.set_ylabel("Concept Probability", fontsize=14)
        ax.set_title("Concept Probability vs Uncertainty", fontsize=16)
        ax.legend(title="Concepts", fontsize=12)

        # Save or show the plot
        if self.save_path is not None:
            plt.savefig(self.save_path + '/concept_prob_vs_uncertainty.png', format="png")
        wandb.log({f"{self.stage}/concept_prob_vs_uncertainty": wandb.Image(fig)})
        plt.close()    
