#!/usr/bin/env python3
"""
Comprehensive analysis of concept statistics and correlations for CBM models.

This script:
1. Identifies erroneous examples (where CBM task predictions are wrong)
2. Analyzes concept annotations and predictions for these examples
3. Performs correlation analysis between concept misprediction and task misprediction
4. Plots histograms showing concept frequency distributions and correlations
5. Helps identify which concepts might be unnecessary for clustering
6. Provides insights into concept-task relationships for model improvement
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
from pathlib import Path
from omegaconf import DictConfig
from collections import Counter
from tqdm import tqdm
from scipy.stats import pearsonr    
import yaml
# from sklearn.metrics import precision_score, recall_score, f1_score

from models.models import create_model
from datasets import get_data
from utils.utils import reset_random_seeds


def analyze_comprehensive_concept_statistics(loader, cbm_model, device, config):
    """
    Comprehensive analysis of concept statistics and correlations for erroneous examples only.
    
    This function analyzes concept-level statistics ONLY for samples where the model's
    task prediction is incorrect (erroneous examples). It combines concept statistics
    analysis with per-sample correlation analysis to provide comprehensive insights
    into concept-task relationships. All statistics are calculated per class to capture
    class-specific error patterns.
    
    Args:
        loader: DataLoader containing the dataset
        cbm_model: Pretrained CBM model
        device: Device to run the model on
        config: Configuration object
        
    Returns:
        Dictionary containing concept statistics and correlation data per class
        (computed only on erroneous examples)
    """
    cbm_model.eval()
    
    # Get number of classes and concepts
    num_concepts = config.data.num_concepts
    num_classes = config.data.num_classes
    
    # Initialize per-class data structures
    class_data = {}
    
    for class_idx in range(num_classes):
        class_data[class_idx] = {
            # Original concept statistics (for erroneous examples only)
            'concept_annotations_counter': Counter(),
            'concept_predictions_counter': Counter(),
            'concept_errors_counter': Counter(),
            'concept_annotation_stats': np.zeros(num_concepts),
            'concept_prediction_stats': np.zeros(num_concepts),
            'concept_error_stats': np.zeros(num_concepts),
            'concept_tp': np.zeros(num_concepts),
            'concept_fp': np.zeros(num_concepts),
            'concept_fn': np.zeros(num_concepts),
            'concept_tn': np.zeros(num_concepts),
            'total_erroneous_examples': 0,
            
            # Per-sample data for correlation analysis
            'sample_data': [],
            'concept_errors_per_sample': [],
            'task_errors_per_sample': [],
            'total_samples': 0,
            
            # S-score (ECTP) data
            's_scores_per_erroneous_sample': []
        }
    
    # Global per-sample data collection (for overall analysis)
    global_sample_data = []
    global_concept_errors_per_sample = []
    global_task_errors_per_sample = []
    global_total_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Comprehensive concept analysis")):
            x = batch["features"].to(device)
            y_true = batch["labels"].to(device)
            c_true = batch["concepts"].to(device)
            
            # Get CBM predictions
            cbm_outputs = cbm_model(x, validation=True)
            s_score = cbm_model.calculate_ectp(x)
            y_pred_logits = cbm_outputs["y_pred_logits"]
            c_pred_prob = cbm_outputs["c_prob"]
            
            # Convert task predictions
            if config.data.num_classes == 2:
                y_pred = (torch.sigmoid(y_pred_logits) > 0.5).long().squeeze(-1)
            else:
                y_pred = torch.argmax(y_pred_logits, dim=-1)
            
            # Convert concept predictions
            if config.model.concept_loss == 'bce':
                c_pred = (c_pred_prob > 0.5).float()
            elif config.model.concept_loss == 'ce':
                c_pred = cbm_outputs["c_prob"].argmax(dim=-1).float()
                c_pred = torch.nn.functional.one_hot(
                    c_pred.long(), num_classes=num_concepts
                ).float()
            
            # Process each example in the batch
            for i in range(x.shape[0]):
                global_total_samples += 1
                
                # Get true class and task error
                true_class = y_true[i].item()
                task_error = 1 if y_pred[i].item() != true_class else 0
                
                # Concept errors (1 if wrong, 0 if correct for each concept)
                c_true_i = c_true[i].cpu().numpy()
                c_pred_i = c_pred[i].cpu().numpy()
                concept_errors = (c_pred_i != c_true_i).astype(int)
                
                # Store global per-sample data for overall correlation analysis
                global_sample_data.append({
                    'batch_idx': batch_idx,
                    'sample_idx': i,
                    'task_error': task_error,
                    'concept_errors': concept_errors.tolist(),
                    'num_concept_errors': np.sum(concept_errors),
                    'y_true': true_class,
                    'y_pred': y_pred[i].item(),
                    'c_true': c_true_i.tolist(),
                    'c_pred': c_pred_i.tolist()
                })
                
                global_concept_errors_per_sample.append(concept_errors)
                global_task_errors_per_sample.append(task_error)
                
                # Update per-class data
                class_data[true_class]['total_samples'] += 1
                class_data[true_class]['sample_data'].append({
                    'batch_idx': batch_idx,
                    'sample_idx': i,
                    'task_error': task_error,
                    'concept_errors': concept_errors.tolist(),
                    'num_concept_errors': np.sum(concept_errors),
                    'y_true': true_class,
                    'y_pred': y_pred[i].item(),
                    'c_true': c_true_i.tolist(),
                    'c_pred': c_pred_i.tolist()
                })
                
                class_data[true_class]['concept_errors_per_sample'].append(concept_errors)
                class_data[true_class]['task_errors_per_sample'].append(task_error)
                
                # Original concept statistics analysis (only for erroneous examples)
                if task_error == 1:  # Only analyze erroneous examples
                    class_data[true_class]['total_erroneous_examples'] += 1

                    # Store s_score for this sample
                    s_score_i = s_score[i].cpu().numpy()  # [n_concepts]
                    class_data[true_class]['s_scores_per_erroneous_sample'].append(s_score_i)
                    
                    # Analyze concept annotations (ground truth)
                    for concept_idx in range(num_concepts):
                        if c_true_i[concept_idx] == 1:
                            class_data[true_class]['concept_annotations_counter'][concept_idx] += 1
                            class_data[true_class]['concept_annotation_stats'][concept_idx] += 1
                    
                    # Analyze concept predictions
                    for concept_idx in range(num_concepts):
                        if c_pred_i[concept_idx] == 1:
                            class_data[true_class]['concept_predictions_counter'][concept_idx] += 1
                            class_data[true_class]['concept_prediction_stats'][concept_idx] += 1
                    
                    # Analyze concept prediction errors
                    for concept_idx in range(num_concepts):
                        if concept_errors[concept_idx]:
                            class_data[true_class]['concept_errors_counter'][concept_idx] += 1
                            class_data[true_class]['concept_error_stats'][concept_idx] += 1
                        
                        # Track confusion matrix components for precision/recall
                        true_label = c_true_i[concept_idx]
                        pred_label = c_pred_i[concept_idx]
                        
                        if true_label == 1 and pred_label == 1:
                            class_data[true_class]['concept_tp'][concept_idx] += 1
                        elif true_label == 0 and pred_label == 1:
                            class_data[true_class]['concept_fp'][concept_idx] += 1
                        elif true_label == 1 and pred_label == 0:
                            class_data[true_class]['concept_fn'][concept_idx] += 1
                        elif true_label == 0 and pred_label == 0:
                            class_data[true_class]['concept_tn'][concept_idx] += 1
    
    # Normalize statistics for erroneous examples per class
    for class_idx in range(num_classes):
        if class_data[class_idx]['total_erroneous_examples'] > 0:
            class_data[class_idx]['concept_annotation_stats'] = (
                class_data[class_idx]['concept_annotation_stats'] / 
                class_data[class_idx]['total_erroneous_examples']
            )
            class_data[class_idx]['concept_prediction_stats'] = (
                class_data[class_idx]['concept_prediction_stats'] / 
                class_data[class_idx]['total_erroneous_examples']
            )
            class_data[class_idx]['concept_error_stats'] = (
                class_data[class_idx]['concept_error_stats'] / 
                class_data[class_idx]['total_erroneous_examples']
            )
    
    # Convert to numpy arrays for correlation analysis (global)
    global_concept_errors_matrix = np.array(global_concept_errors_per_sample)
    global_task_errors_array = np.array(global_task_errors_per_sample)
    
    # Calculate global correlation metrics (Pearson only)
    global_pearson_correlations = []
    
    for concept_idx in range(num_concepts):
        concept_errors = global_concept_errors_matrix[:, concept_idx]
        
        # Pearson correlation
        if np.std(concept_errors) > 0 and np.std(global_task_errors_array) > 0:
            pearson_r, _ = pearsonr(concept_errors, global_task_errors_array)
        else:
            pearson_r = 0.0
        global_pearson_correlations.append(pearson_r)
    
    # Calculate per-class correlation metrics and s_score statistics
    for class_idx in range(num_classes):
        if class_data[class_idx]['total_samples'] > 0:
            concept_errors_matrix = np.array(class_data[class_idx]['concept_errors_per_sample'])
            task_errors_array = np.array(class_data[class_idx]['task_errors_per_sample'])
            
            # Calculate correlation metrics for this class (Pearson only)
            pearson_correlations = []
            
            for concept_idx in range(num_concepts):
                concept_errors = concept_errors_matrix[:, concept_idx]
                
                # Pearson correlation
                if np.std(concept_errors) > 0 and np.std(task_errors_array) > 0:
                    pearson_r, _ = pearsonr(concept_errors, task_errors_array)
                else:
                    pearson_r = 0.0
                pearson_correlations.append(pearson_r)
            
            # Store per-class correlation data
            class_data[class_idx]['pearson_correlations'] = np.array(pearson_correlations)
            class_data[class_idx]['correlation_matrix'] = np.corrcoef(concept_errors_matrix.T)
            class_data[class_idx]['task_error_rate'] = np.mean(task_errors_array)
            class_data[class_idx]['total_task_errors'] = np.sum(task_errors_array)
            class_data[class_idx]['concept_error_rates'] = np.mean(concept_errors_matrix, axis=0)
            class_data[class_idx]['total_concept_errors'] = np.sum(concept_errors_matrix, axis=0)
            
            # Calculate average s_score per concept for this class
            if len(class_data[class_idx]['s_scores_per_erroneous_sample']) > 0:
                s_scores_matrix = np.array(class_data[class_idx]['s_scores_per_erroneous_sample'])
                class_data[class_idx]['avg_s_scores'] = np.mean(s_scores_matrix, axis=0)
            else:
                class_data[class_idx]['avg_s_scores'] = np.zeros(num_concepts)
    
    # Calculate global correlation matrix and statistics
    global_correlation_matrix = np.corrcoef(global_concept_errors_matrix.T)
    global_task_error_rate = np.mean(global_task_errors_array)
    global_total_task_errors = np.sum(global_task_errors_array)
    global_concept_error_rates = np.mean(global_concept_errors_matrix, axis=0)
    global_total_concept_errors = np.sum(global_concept_errors_matrix, axis=0)
    
    # Return comprehensive results
    return {
        # Global statistics (for backward compatibility)
        'num_concepts': num_concepts,
        'num_classes': num_classes,
        
        # Global per-sample correlation data
        'global_sample_data': global_sample_data,
        'global_concept_errors_matrix': global_concept_errors_matrix,
        'global_task_errors_array': global_task_errors_array,
        'global_num_samples': global_total_samples,
        
        # Global correlation metrics (Pearson only)
        'global_pearson_correlations': np.array(global_pearson_correlations),
        'global_correlation_matrix': global_correlation_matrix,
        'global_task_error_rate': global_task_error_rate,
        'global_total_task_errors': global_total_task_errors,
        'global_concept_error_rates': global_concept_error_rates,
        'global_total_concept_errors': global_total_concept_errors,
        
        # Per-class data
        'class_data': class_data
    }




def calculate_precision_recall_f1(stats):
    """
    Calculate precision, recall, and F1 score for each concept.
    
    Args:
        stats: Dictionary containing concept statistics with confusion matrix components
        
    Returns:
        Dictionary containing precision, recall, and F1 scores for each concept
    """
    num_concepts = stats['num_concepts']
    precision_scores = np.zeros(num_concepts)
    recall_scores = np.zeros(num_concepts)
    f1_scores = np.zeros(num_concepts)
    
    for i in range(num_concepts):
        tp = stats['concept_tp'][i]
        fp = stats['concept_fp'][i]
        fn = stats['concept_fn'][i]
        
        # Calculate precision: TP / (TP + FP)
        if tp + fp > 0:
            precision_scores[i] = tp / (tp + fp)
        else:
            precision_scores[i] = 0.0  # No positive predictions
        
        # Calculate recall: TP / (TP + FN)
        if tp + fn > 0:
            recall_scores[i] = tp / (tp + fn)
        else:
            recall_scores[i] = 0.0  # No positive ground truth
        
        # Calculate F1 score: 2 * (precision * recall) / (precision + recall)
        if precision_scores[i] + recall_scores[i] > 0:
            f1_scores[i] = 2 * (precision_scores[i] * recall_scores[i]) / (precision_scores[i] + recall_scores[i])
        else:
            f1_scores[i] = 0.0
    
    return {
        'precision_scores': precision_scores,
        'recall_scores': recall_scores,
        'f1_scores': f1_scores
    }


def plot_correlation_analysis(comprehensive_data, output_dir, config):
    """
    Create comprehensive correlation analysis plots for both global and per-class data.
    
    Args:
        comprehensive_data: Dictionary containing comprehensive analysis results
        output_dir: Directory to save plots
        config: Configuration object
    """
    num_concepts = comprehensive_data['num_concepts']
    num_classes = comprehensive_data['num_classes']
    
    # Global correlation data
    global_pearson_corrs = comprehensive_data['global_pearson_correlations']
    global_concept_error_rates = comprehensive_data['global_concept_error_rates']
    global_task_error_rate = comprehensive_data['global_task_error_rate']
    global_correlation_matrix = comprehensive_data['global_correlation_matrix']
    
    # Create figure with multiple subplots for global analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Global Concept-Task Error Correlation Analysis', fontsize=16)
    
    # Plot 1: Pearson Correlations
    ax1 = axes[0, 0]
    bars1 = ax1.bar(range(num_concepts), global_pearson_corrs, alpha=0.7, color='skyblue')
    ax1.set_title('Pearson Correlations\n(Concept Errors vs Task Errors)')
    ax1.set_xlabel('Concept Index')
    ax1.set_ylabel('Correlation Coefficient')
    ax1.set_xticks(range(num_concepts))
    ax1.set_xticklabels([f"C{i}" for i in range(num_concepts)], rotation=45)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, corr in zip(bars1, global_pearson_corrs):
        if abs(corr) > 0.01:
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.01,
                     f'{corr:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Concept Error Rates vs Task Error Rate
    ax2 = axes[0, 1]
    bars2 = ax2.bar(range(num_concepts), global_concept_error_rates, alpha=0.7, color='purple')
    ax2.axhline(y=global_task_error_rate, color='red', linestyle='--', linewidth=2, 
                label=f'Task Error Rate: {global_task_error_rate:.3f}')
    ax2.set_title('Concept Error Rates vs Task Error Rate')
    ax2.set_xlabel('Concept Index')
    ax2.set_ylabel('Error Rate')
    ax2.set_xticks(range(num_concepts))
    ax2.set_xticklabels([f"C{i}" for i in range(num_concepts)], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, rate in zip(bars2, global_concept_error_rates):
        if rate > 0.001:
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.001,
                     f'{rate:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Global Correlation Matrix Heatmap
    ax3 = axes[1, 0]
    im = ax3.imshow(global_correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax3.set_title('Global Concept Error Correlation Matrix')
    ax3.set_xlabel('Concept Index')
    ax3.set_ylabel('Concept Index')
    ax3.set_xticks(range(num_concepts))
    ax3.set_yticks(range(num_concepts))
    ax3.set_xticklabels([f"C{i}" for i in range(num_concepts)])
    ax3.set_yticklabels([f"C{i}" for i in range(num_concepts)])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('Correlation Coefficient')
    
    # Add correlation values to heatmap
    for i in range(num_concepts):
        for j in range(num_concepts):
            ax3.text(j, i, f'{global_correlation_matrix[i, j]:.2f}',
                     ha="center", va="center", color="black", fontsize=8)
    
    # Plot 4: Per-Class Comparison (if multiple classes)
    ax4 = axes[1, 1]
    if num_classes > 1:
        x = np.arange(num_concepts)
        width = 0.8 / num_classes
        
        for class_idx in range(num_classes):
            class_data = comprehensive_data['class_data'][class_idx]
            if 'pearson_correlations' in class_data:
                ax4.bar(x + class_idx * width, class_data['pearson_correlations'], 
                       width, label=f'Class {class_idx}', alpha=0.7)
        
        ax4.set_title('Per-Class Pearson Correlations')
        ax4.set_xlabel('Concept Index')
        ax4.set_ylabel('Correlation Coefficient')
        ax4.set_xticks(x + width * (num_classes - 1) / 2)
        ax4.set_xticklabels([f"C{i}" for i in range(num_concepts)])
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    else:
        ax4.text(0.5, 0.5, 'Single Class Dataset', ha='center', va='center', 
                transform=ax4.transAxes, fontsize=14)
        ax4.set_title('Per-Class Analysis')
    
    plt.tight_layout()
    
    # Save the global analysis plot
    output_path = output_dir / "global_concept_task_correlation_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Global correlation analysis plot saved to: {output_path}")
    
    # Create per-class detailed plots if multiple classes
    if num_classes > 1:
        for class_idx in range(num_classes):
            class_data = comprehensive_data['class_data'][class_idx]
            if 'pearson_correlations' in class_data and class_data['total_samples'] > 0:
                fig, ax = plt.subplots(1, 1, figsize=(12, 6))
                
                pearson_corrs = class_data['pearson_correlations']
                
                x = np.arange(num_concepts)
                width = 0.5
                
                bars1 = ax.bar(x, pearson_corrs, width, label='Pearson', 
                              alpha=0.7, color='skyblue')
                
                ax.set_xlabel('Concept Index')
                ax.set_ylabel('Score')
                ax.set_title(f'Class {class_idx} Concept-Task Error Correlation Analysis')
                ax.set_xticks(x)
                ax.set_xticklabels([f"C{i}" for i in range(num_concepts)])
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
                
                plt.tight_layout()
                
                # Save per-class plot
                class_path = output_dir / f"class_{class_idx}_correlation_analysis.png"
                plt.savefig(class_path, dpi=300, bbox_inches='tight')
                print(f"Class {class_idx} correlation plot saved to: {class_path}")
                
                plt.close()
    
    plt.show()


def plot_per_class_concept_analysis(comprehensive_data, output_dir, config):
    """
    Create per-class concept analysis plots including histograms and precision-recall.
    
    Args:
        comprehensive_data: Dictionary containing comprehensive analysis results
        output_dir: Directory to save plots
        config: Configuration object
    """
    num_concepts = comprehensive_data['num_concepts']
    num_classes = comprehensive_data['num_classes']
    
    # Create per-class plots for each class
    for class_idx in range(num_classes):
        class_data = comprehensive_data['class_data'][class_idx]
        
        if class_data['total_samples'] == 0:
            continue
            
        # Calculate precision, recall, and F1 scores for this class
        pr_stats = calculate_precision_recall_f1_per_class(class_data)
        
        # Create figure with subplots for this class
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        fig.suptitle(f'Class {class_idx} Concept Analysis for Erroneous Examples', fontsize=16)
        
        # Plot 1: Concept Annotation Frequencies
        ax1 = axes[0, 0]
        annotation_counts = [class_data['concept_annotations_counter'][i] for i in range(num_concepts)]
        bars1 = ax1.bar(range(num_concepts), annotation_counts, alpha=0.7, color='skyblue')
        ax1.set_title(f'Class {class_idx} Concept Annotation Frequencies\n(Ground Truth Concepts)')
        ax1.set_xlabel('Concept Index')
        ax1.set_ylabel('Frequency in Erroneous Examples')
        ax1.set_xticks(range(num_concepts))
        ax1.set_xticklabels([f"C{i}" for i in range(num_concepts)], rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars1, annotation_counts):
            if count > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(count), ha='center', va='bottom', fontsize=8)
        
        # Plot 2: Concept Prediction Frequencies
        ax2 = axes[0, 1]
        prediction_counts = [class_data['concept_predictions_counter'][i] for i in range(num_concepts)]
        bars2 = ax2.bar(range(num_concepts), prediction_counts, alpha=0.7, color='lightcoral')
        ax2.set_title(f'Class {class_idx} Concept Prediction Frequencies\n(CBM Predicted Concepts)')
        ax2.set_xlabel('Concept Index')
        ax2.set_ylabel('Frequency in Erroneous Examples')
        ax2.set_xticks(range(num_concepts))
        ax2.set_xticklabels([f"C{i}" for i in range(num_concepts)], rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars2, prediction_counts):
            if count > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(count), ha='center', va='bottom', fontsize=8)
        
        # Plot 3: Concept Annotation Proportions
        ax3 = axes[1, 0]
        annotation_props = class_data['concept_annotation_stats']
        bars3 = ax3.bar(range(num_concepts), annotation_props, alpha=0.7, color='lightgreen')
        ax3.set_title(f'Class {class_idx} Concept Annotation Proportions\n(% of Erroneous Examples)')
        ax3.set_xlabel('Concept Index')
        ax3.set_ylabel('Proportion')
        ax3.set_xticks(range(num_concepts))
        ax3.set_xticklabels([f"C{i}" for i in range(num_concepts)], rotation=45)
        ax3.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, prop in zip(bars3, annotation_props):
            if prop > 0:
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{prop:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 4: Concept Prediction Proportions
        ax4 = axes[1, 1]
        prediction_props = class_data['concept_prediction_stats']
        bars4 = ax4.bar(range(num_concepts), prediction_props, alpha=0.7, color='orange')
        ax4.set_title(f'Class {class_idx} Concept Prediction Proportions\n(% of Erroneous Examples)')
        ax4.set_xlabel('Concept Index')
        ax4.set_ylabel('Proportion')
        ax4.set_xticks(range(num_concepts))
        ax4.set_xticklabels([f"C{i}" for i in range(num_concepts)], rotation=45)
        ax4.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, prop in zip(bars4, prediction_props):
            if prop > 0:
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{prop:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 5: Precision Scores
        ax5 = axes[2, 0]
        precision_scores = pr_stats['precision_scores']
        bars5 = ax5.bar(range(num_concepts), precision_scores, alpha=0.7, color='purple')
        ax5.set_title(f'Class {class_idx} Concept Precision Scores\n(TP / (TP + FP))')
        ax5.set_xlabel('Concept Index')
        ax5.set_ylabel('Precision')
        ax5.set_xticks(range(num_concepts))
        ax5.set_xticklabels([f"C{i}" for i in range(num_concepts)], rotation=45)
        ax5.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars5, precision_scores):
            if score > 0:
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 6: Recall Scores
        ax6 = axes[2, 1]
        recall_scores = pr_stats['recall_scores']
        bars6 = ax6.bar(range(num_concepts), recall_scores, alpha=0.7, color='brown')
        ax6.set_title(f'Class {class_idx} Concept Recall Scores\n(TP / (TP + FN))')
        ax6.set_xlabel('Concept Index')
        ax6.set_ylabel('Recall')
        ax6.set_xticks(range(num_concepts))
        ax6.set_xticklabels([f"C{i}" for i in range(num_concepts)], rotation=45)
        ax6.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars6, recall_scores):
            if score > 0:
                ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # Save the per-class plot
        class_histogram_path = output_dir / f"class_{class_idx}_concept_analysis_histograms.png"
        plt.savefig(class_histogram_path, dpi=300, bbox_inches='tight')
        print(f"Class {class_idx} histograms saved to: {class_histogram_path}")
        
        # Create comparison plots for this class
        create_class_comparison_plots(class_data, pr_stats, class_idx, output_dir, num_concepts)
        
        plt.close()
    
    # Create global comparison across classes
    create_global_class_comparison(comprehensive_data, output_dir, num_concepts, num_classes)


def calculate_precision_recall_f1_per_class(class_data):
    """
    Calculate precision, recall, and F1 score for each concept in a specific class.
    
    Args:
        class_data: Dictionary containing class-specific concept statistics
        
    Returns:
        Dictionary containing precision, recall, and F1 scores for each concept
    """
    num_concepts = len(class_data['concept_tp'])
    precision_scores = np.zeros(num_concepts)
    recall_scores = np.zeros(num_concepts)
    f1_scores = np.zeros(num_concepts)
    
    for i in range(num_concepts):
        tp = class_data['concept_tp'][i]
        fp = class_data['concept_fp'][i]
        fn = class_data['concept_fn'][i]
        
        # Calculate precision: TP / (TP + FP)
        if tp + fp > 0:
            precision_scores[i] = tp / (tp + fp)
        else:
            precision_scores[i] = 0.0  # No positive predictions
        
        # Calculate recall: TP / (TP + FN)
        if tp + fn > 0:
            recall_scores[i] = tp / (tp + fn)
        else:
            recall_scores[i] = 0.0  # No positive ground truth
        
        # Calculate F1 score: 2 * (precision * recall) / (precision + recall)
        if precision_scores[i] + recall_scores[i] > 0:
            f1_scores[i] = 2 * (precision_scores[i] * recall_scores[i]) / (precision_scores[i] + recall_scores[i])
        else:
            f1_scores[i] = 0.0
    
    return {
        'precision_scores': precision_scores,
        'recall_scores': recall_scores,
        'f1_scores': f1_scores
    }


def create_class_comparison_plots(class_data, pr_stats, class_idx, output_dir, num_concepts):
    """
    Create comparison plots for a specific class.
    
    Args:
        class_data: Dictionary containing class-specific data
        pr_stats: Dictionary containing precision/recall statistics
        class_idx: Index of the class
        output_dir: Directory to save plots
        num_concepts: Number of concepts
    """
    # Concept annotations vs predictions comparison
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    x = np.arange(num_concepts)
    width = 0.35
    
    ax.bar(x - width/2, class_data['concept_annotation_stats'], width, 
           label='Annotations', alpha=0.7, color='skyblue')
    ax.bar(x + width/2, class_data['concept_prediction_stats'], width, 
           label='Predictions', alpha=0.7, color='lightcoral')
    
    ax.set_xlabel('Concept Index')
    ax.set_ylabel('Proportion of Erroneous Examples')
    ax.set_title(f'Class {class_idx} Concept Annotations vs Predictions in Erroneous Examples')
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{i}" for i in range(num_concepts)])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the comparison plot
    comparison_path = output_dir / f"class_{class_idx}_concept_comparison.png"
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"Class {class_idx} comparison plot saved to: {comparison_path}")
    
    # Precision-recall comparison plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    x = np.arange(num_concepts)
    width = 0.25
    
    ax.bar(x - width, pr_stats['precision_scores'], width, label='Precision', 
           alpha=0.7, color='purple')
    ax.bar(x, pr_stats['recall_scores'], width, label='Recall', 
           alpha=0.7, color='brown')
    ax.bar(x + width, pr_stats['f1_scores'], width, label='F1 Score', 
           alpha=0.7, color='green')
    
    ax.set_xlabel('Concept Index')
    ax.set_ylabel('Score')
    ax.set_title(f'Class {class_idx} Concept Precision, Recall, and F1 Scores in Erroneous Examples')
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{i}" for i in range(num_concepts)])
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Save the precision-recall comparison plot
    pr_comparison_path = output_dir / f"class_{class_idx}_concept_precision_recall_comparison.png"
    plt.savefig(pr_comparison_path, dpi=300, bbox_inches='tight')
    print(f"Class {class_idx} precision-recall comparison plot saved to: {pr_comparison_path}")
    
    plt.close()


def create_global_class_comparison(comprehensive_data, output_dir, num_concepts, num_classes):
    """
    Create global comparison plots across all classes.
    
    Args:
        comprehensive_data: Dictionary containing comprehensive analysis results
        output_dir: Directory to save plots
        num_concepts: Number of concepts
        num_classes: Number of classes
    """
    if num_classes <= 1:
        return
    
    # Collect precision and recall data for all classes
    class_precision_data = []
    class_recall_data = []
    class_f1_data = []
    class_annotation_data = []
    class_prediction_data = []
    
    for class_idx in range(num_classes):
        class_data = comprehensive_data['class_data'][class_idx]
        if class_data['total_samples'] > 0:
            pr_stats = calculate_precision_recall_f1_per_class(class_data)
            class_precision_data.append(pr_stats['precision_scores'])
            class_recall_data.append(pr_stats['recall_scores'])
            class_f1_data.append(pr_stats['f1_scores'])
            class_annotation_data.append(class_data['concept_annotation_stats'])
            class_prediction_data.append(class_data['concept_prediction_stats'])
    
    if not class_precision_data:
        return
    
    # Convert to numpy arrays
    class_precision_data = np.array(class_precision_data)
    class_recall_data = np.array(class_recall_data)
    class_f1_data = np.array(class_f1_data)
    class_annotation_data = np.array(class_annotation_data)
    class_prediction_data = np.array(class_prediction_data)
    
    # Create precision comparison across classes
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    x = np.arange(num_concepts)
    width = 0.8 / num_classes
    
    for i, class_idx in enumerate(range(num_classes)):
        if comprehensive_data['class_data'][class_idx]['total_samples'] > 0:
            ax.bar(x + i * width, class_precision_data[i], width, 
                   label=f'Class {class_idx}', alpha=0.7)
    
    ax.set_xlabel('Concept Index')
    ax.set_ylabel('Precision Score')
    ax.set_title('Concept Precision Scores Across Classes')
    ax.set_xticks(x + width * (num_classes - 1) / 2)
    ax.set_xticklabels([f"C{i}" for i in range(num_concepts)])
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Save the precision comparison plot
    precision_comparison_path = output_dir / "global_class_precision_comparison.png"
    plt.savefig(precision_comparison_path, dpi=300, bbox_inches='tight')
    print(f"Global class precision comparison plot saved to: {precision_comparison_path}")
    
    # Create recall comparison across classes
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    for i, class_idx in enumerate(range(num_classes)):
        if comprehensive_data['class_data'][class_idx]['total_samples'] > 0:
            ax.bar(x + i * width, class_recall_data[i], width, 
                   label=f'Class {class_idx}', alpha=0.7)
    
    ax.set_xlabel('Concept Index')
    ax.set_ylabel('Recall Score')
    ax.set_title('Concept Recall Scores Across Classes')
    ax.set_xticks(x + width * (num_classes - 1) / 2)
    ax.set_xticklabels([f"C{i}" for i in range(num_concepts)])
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Save the recall comparison plot
    recall_comparison_path = output_dir / "global_class_recall_comparison.png"
    plt.savefig(recall_comparison_path, dpi=300, bbox_inches='tight')
    print(f"Global class recall comparison plot saved to: {recall_comparison_path}")
    
    # Create annotation vs prediction comparison across classes
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    width = 0.8 / (2 * num_classes)  # Half width for annotations and predictions
    
    for i, class_idx in enumerate(range(num_classes)):
        if comprehensive_data['class_data'][class_idx]['total_samples'] > 0:
            ax.bar(x + i * 2 * width, class_annotation_data[i], width, 
                   label=f'Class {class_idx} Annotations', alpha=0.7)
            ax.bar(x + i * 2 * width + width, class_prediction_data[i], width, 
                   label=f'Class {class_idx} Predictions', alpha=0.7)
    
    ax.set_xlabel('Concept Index')
    ax.set_ylabel('Proportion of Erroneous Examples')
    ax.set_title('Concept Annotations vs Predictions Across Classes')
    ax.set_xticks(x + width * (2 * num_classes - 1) / 2)
    ax.set_xticklabels([f"C{i}" for i in range(num_concepts)])
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Save the annotation comparison plot
    annotation_comparison_path = output_dir / "global_class_annotation_comparison.png"
    plt.savefig(annotation_comparison_path, dpi=300, bbox_inches='tight')
    print(f"Global class annotation comparison plot saved to: {annotation_comparison_path}")
    
    plt.close()


def save_comprehensive_analysis(comprehensive_data, output_dir):
    """
    Save comprehensive analysis data to various formats in the output directory.
    
    Args:
        comprehensive_data: Dictionary containing comprehensive analysis results
        output_dir: Directory to save analysis files
    """
    # Save global correlation data as JSON
    global_correlation = {
        'num_concepts': int(comprehensive_data['num_concepts']),
        'num_classes': int(comprehensive_data['num_classes']),
        'global_num_samples': int(comprehensive_data['global_num_samples']),
        'global_pearson_correlations': comprehensive_data['global_pearson_correlations'].tolist(),
        'global_correlation_matrix': comprehensive_data['global_correlation_matrix'].tolist(),
        'global_task_error_rate': float(comprehensive_data['global_task_error_rate']),
        'global_total_task_errors': int(comprehensive_data['global_total_task_errors']),
        'global_concept_error_rates': comprehensive_data['global_concept_error_rates'].tolist(),
        'global_total_concept_errors': comprehensive_data['global_total_concept_errors'].tolist()
    }
    
    json_path = output_dir / "global_correlation_analysis.json"
    with open(json_path, 'w') as f:
        json.dump(global_correlation, f, indent=2)
    print(f"Global correlation analysis saved as JSON to: {json_path}")
    
    # Create detailed CSV for global correlation analysis
    num_concepts = comprehensive_data['num_concepts']
    global_correlation_df_data = []
    
    for i in range(num_concepts):
        global_correlation_df_data.append({
            'concept_index': i,
            'pearson_correlation': comprehensive_data['global_pearson_correlations'][i],
            'concept_error_rate': comprehensive_data['global_concept_error_rates'][i],
            'total_concept_errors': comprehensive_data['global_total_concept_errors'][i]
        })
    
    global_correlation_df = pd.DataFrame(global_correlation_df_data)
    csv_path = output_dir / "global_correlation_analysis.csv"
    global_correlation_df.to_csv(csv_path, index=False)
    print(f"Global correlation analysis saved as CSV to: {csv_path}")
    
    # Save per-class data
    for class_idx in range(comprehensive_data['num_classes']):
        class_data = comprehensive_data['class_data'][class_idx]
        if class_data['total_samples'] > 0:
            # Calculate precision/recall for this class
            pr_stats = calculate_precision_recall_f1_per_class(class_data)
            
            # Save per-class comprehensive data (correlation + concept statistics)
            class_comprehensive = {
                'class_index': class_idx,
                'total_samples': int(class_data['total_samples']),
                'total_erroneous_examples': int(class_data['total_erroneous_examples']),
                'task_error_rate': float(class_data['task_error_rate']),
                'total_task_errors': int(class_data['total_task_errors']),
                
                # Concept statistics
                'concept_annotations_counter': dict(class_data['concept_annotations_counter']),
                'concept_predictions_counter': dict(class_data['concept_predictions_counter']),
                'concept_errors_counter': dict(class_data['concept_errors_counter']),
                'concept_annotation_stats': class_data['concept_annotation_stats'].tolist(),
                'concept_prediction_stats': class_data['concept_prediction_stats'].tolist(),
                'concept_error_stats': class_data['concept_error_stats'].tolist(),
                'concept_tp': class_data['concept_tp'].tolist(),
                'concept_fp': class_data['concept_fp'].tolist(),
                'concept_fn': class_data['concept_fn'].tolist(),
                'concept_tn': class_data['concept_tn'].tolist(),
                'precision_scores': pr_stats['precision_scores'].tolist(),
                'recall_scores': pr_stats['recall_scores'].tolist(),
                'f1_scores': pr_stats['f1_scores'].tolist(),
                
                # Correlation data (if available)
                'pearson_correlations': class_data.get('pearson_correlations', []).tolist() if 'pearson_correlations' in class_data else [],
                'correlation_matrix': class_data.get('correlation_matrix', []).tolist() if 'correlation_matrix' in class_data else [],
                'concept_error_rates': class_data.get('concept_error_rates', []).tolist() if 'concept_error_rates' in class_data else [],
                'total_concept_errors': class_data.get('total_concept_errors', []).tolist() if 'total_concept_errors' in class_data else [],
                
                # S-score (ECTP) data
                'avg_s_scores': class_data['avg_s_scores'].tolist()
            }
            
            class_json_path = output_dir / f"class_{class_idx}_comprehensive_analysis.json"
            with open(class_json_path, 'w') as f:
                json.dump(class_comprehensive, f, indent=2)
            print(f"Class {class_idx} comprehensive analysis saved to: {class_json_path}")
            
            # Save per-class CSV for concept statistics
            class_concept_df_data = []
            for i in range(num_concepts):
                concept_row = {
                    'concept_index': i,
                    'annotation_count': class_data['concept_annotations_counter'][i],
                    'prediction_count': class_data['concept_predictions_counter'][i],
                    'error_count': class_data['concept_errors_counter'][i],
                    'annotation_proportion': class_data['concept_annotation_stats'][i],
                    'prediction_proportion': class_data['concept_prediction_stats'][i],
                    'error_proportion': class_data['concept_error_stats'][i],
                    'true_positives': class_data['concept_tp'][i],
                    'false_positives': class_data['concept_fp'][i],
                    'false_negatives': class_data['concept_fn'][i],
                    'true_negatives': class_data['concept_tn'][i],
                    'precision': pr_stats['precision_scores'][i],
                    'recall': pr_stats['recall_scores'][i],
                    'f1_score': pr_stats['f1_scores'][i],
                    'avg_s_score': class_data['avg_s_scores'][i]
                }
                
                # Add correlation data if available
                if 'pearson_correlations' in class_data:
                    concept_row.update({
                        'pearson_correlation': class_data['pearson_correlations'][i],
                        'concept_error_rate': class_data['concept_error_rates'][i],
                        'total_concept_errors': class_data['total_concept_errors'][i]
                    })
                
                class_concept_df_data.append(concept_row)
            
            class_concept_df = pd.DataFrame(class_concept_df_data)
            class_csv_path = output_dir / f"class_{class_idx}_comprehensive_analysis.csv"
            class_concept_df.to_csv(class_csv_path, index=False)
            print(f"Class {class_idx} comprehensive analysis saved as CSV to: {class_csv_path}")
    
    # Save summary statistics
    # Calculate global concept annotation stats, prediction stats, and F1 scores
    # These are computed as the mean across all classes for each concept
    num_concepts = int(comprehensive_data['num_concepts'])
    num_classes = int(comprehensive_data['num_classes'])
    class_data_dict = comprehensive_data.get('class_data', {})

    # Initialize arrays to collect per-class stats for each concept
    annotation_stats = np.zeros((num_classes, num_concepts))
    prediction_stats = np.zeros((num_classes, num_concepts))
    f1_scores = np.zeros((num_classes, num_concepts))

    for class_idx, class_data in class_data_dict.items():
        annotation_stats[class_idx] = np.array(class_data.get('concept_annotation_stats', [0]*num_concepts))
        prediction_stats[class_idx] = np.array(class_data.get('concept_prediction_stats', [0]*num_concepts))
        # F1 scores may not be available for all classes; use zeros if missing
        pr_stats = calculate_precision_recall_f1_per_class(class_data)
        f1_scores[class_idx] = pr_stats['f1_scores']

    # Compute global stats as mean across classes for each concept
    global_concept_annotation_stats = annotation_stats.mean(axis=0)
    global_concept_prediction_stats = prediction_stats.mean(axis=0)
    global_f1_scores = f1_scores.mean(axis=0)

    # Concepts with low annotation frequency
    annotation_threshold = 0.05  # less than 5% annotation frequency
    prediction_threshold = 0.05  # less than 5% prediction frequency
    # Define F1 score thresholds
    f1_score_threshold_08 = 0.8     # F1 score higher than 0.8
    f1_score_threshold_07 = 0.7     # F1 score higher than 0.75
    f1_score_threshold_06 = 0.6     # F1 score higher than 0.6

    concepts_low_annotation = np.where(global_concept_annotation_stats < annotation_threshold)[0].tolist()
    concepts_low_prediction = np.where(global_concept_prediction_stats < prediction_threshold)[0].tolist()
    concept_low_ann_and_pred = np.intersect1d(concepts_low_annotation, concepts_low_prediction).tolist()

    # Find concepts with F1 scores higher than each threshold
    concepts_high_f1_08 = np.where(global_f1_scores > f1_score_threshold_08)[0].tolist()
    concepts_high_f1_07 = np.where(global_f1_scores > f1_score_threshold_07)[0].tolist()
    concepts_high_f1_06 = np.where(global_f1_scores > f1_score_threshold_06)[0].tolist()

    # Calculate top-k concepts by s_score for each class
    k = 20  # Number of top concepts to save
    top_k_s_score_per_class = {}
    
    for class_idx in range(num_classes):
        if 'avg_s_scores' in class_data_dict[class_idx]:
            avg_s_scores = class_data_dict[class_idx]['avg_s_scores']
            # Get indices of top-k concepts (highest s_scores)
            top_k_indices = np.argsort(avg_s_scores)[::-1][:k]  # Sort descending
            
            # Store concept index and its s_score value
            top_k_s_score_per_class[f'class_{class_idx}'] = {
                int(idx): float(avg_s_scores[idx]) for idx in top_k_indices
            }

    summary_comprehensive = {
        'num_concepts': num_concepts,
        'num_classes': num_classes,
        'global_num_samples': int(comprehensive_data['global_num_samples']),
        'global_task_error_rate': float(comprehensive_data['global_task_error_rate']),
        'global_total_task_errors': int(comprehensive_data['global_total_task_errors']),
        'avg_global_pearson_correlation': float(np.mean(comprehensive_data['global_pearson_correlations'])),
        'std_global_pearson_correlation': float(np.std(comprehensive_data['global_pearson_correlations'])),
        'concepts_with_high_global_correlation': len([c for c in comprehensive_data['global_pearson_correlations'] if abs(c) > 0.2]),
        'concepts_with_high_global_error_rate': len([c for c in comprehensive_data['global_concept_error_rates'] if c > 0.1]),
        # Save the computed statistics
        'global_concept_annotation_stats': global_concept_annotation_stats.tolist(),
        'global_concept_prediction_stats': global_concept_prediction_stats.tolist(),
        'global_f1_scores': global_f1_scores.tolist(),
        # New fields for concepts prone to be filtered out
        'concepts_low_annotation_frequency': concepts_low_annotation,
        'concepts_low_prediction_frequency': concepts_low_prediction,
        'concepts_low_annotation_and_prediction_frequency': concept_low_ann_and_pred,
        'concepts_high_f1_score_08': concepts_high_f1_08,
        'concepts_high_f1_score_07': concepts_high_f1_07,
        'concepts_high_f1_score_06': concepts_high_f1_06,
        # Top-k concepts by s_score (ECTP) for each class
        'top_k_concepts_by_s_score': top_k_s_score_per_class,
    }
    #     'num_concepts': int(comprehensive_data['num_concepts']),
    #     'num_classes': int(comprehensive_data['num_classes']),
    #     'global_num_samples': int(comprehensive_data['global_num_samples']),
    #     'global_task_error_rate': float(comprehensive_data['global_task_error_rate']),
    #     'global_total_task_errors': int(comprehensive_data['global_total_task_errors']),
    #     'avg_global_pearson_correlation': float(np.mean(comprehensive_data['global_pearson_correlations'])),
    #     'std_global_pearson_correlation': float(np.std(comprehensive_data['global_pearson_correlations'])),
    #     'concepts_with_high_global_correlation': len([c for c in comprehensive_data['global_pearson_correlations'] if abs(c) > 0.2]),
    #     'concepts_with_high_global_error_rate': len([c for c in comprehensive_data['global_concept_error_rates'] if c > 0.1])
    # }
    
    summary_path = output_dir / "comprehensive_analysis_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary_comprehensive, f, indent=2)
    print(f"Comprehensive analysis summary saved to: {summary_path}")


def save_correlation_analysis(correlation_data, per_sample_data, output_dir):
    """
    Save correlation analysis data to various formats in the output directory.
    
    Args:
        correlation_data: Dictionary containing correlation metrics
        per_sample_data: Dictionary containing per-sample data
        output_dir: Directory to save correlation files
    """
    # Save correlation data as JSON
    json_correlation = {
        'pearson_correlations': correlation_data['pearson_correlations'].tolist(),
        'correlation_matrix': correlation_data['correlation_matrix'].tolist(),
        'task_error_rate': float(correlation_data['task_error_rate']),
        'total_task_errors': int(correlation_data['total_task_errors']),
        'concept_error_rates': correlation_data['concept_error_rates'].tolist(),
        'total_concept_errors': correlation_data['total_concept_errors'].tolist(),
        'num_concepts': int(correlation_data['num_concepts']),
        'num_samples': int(per_sample_data['num_samples'])
    }
    
    json_path = output_dir / "correlation_analysis.json"
    with open(json_path, 'w') as f:
        json.dump(json_correlation, f, indent=2)
    print(f"Correlation analysis saved as JSON to: {json_path}")
    
    # Create detailed CSV for correlation analysis
    num_concepts = correlation_data['num_concepts']
    correlation_df_data = []
    
    for i in range(num_concepts):
        correlation_df_data.append({
            'concept_index': i,
            'pearson_correlation': correlation_data['pearson_correlations'][i],
            'concept_error_rate': correlation_data['concept_error_rates'][i],
            'total_concept_errors': correlation_data['total_concept_errors'][i]
        })
    
    correlation_df = pd.DataFrame(correlation_df_data)
    csv_path = output_dir / "correlation_analysis.csv"
    correlation_df.to_csv(csv_path, index=False)
    print(f"Correlation analysis saved as CSV to: {csv_path}")
    
    # Save per-sample data (subset for memory efficiency)
    sample_df_data = []
    for i, sample in enumerate(per_sample_data['sample_data'][:1000]):  # Limit to first 1000 samples
        sample_df_data.append({
            'sample_index': i,
            'task_error': sample['task_error'],
            'num_concept_errors': sample['num_concept_errors'],
            'y_true': sample['y_true'],
            'y_pred': sample['y_pred']
        })
        # Add individual concept errors
        for j, concept_error in enumerate(sample['concept_errors']):
            sample_df_data[-1][f'concept_{j}_error'] = concept_error
    
    sample_df = pd.DataFrame(sample_df_data)
    sample_csv_path = output_dir / "per_sample_errors.csv"
    sample_df.to_csv(sample_csv_path, index=False)
    print(f"Per-sample error data saved as CSV to: {sample_csv_path}")
    
    # Save summary statistics
    summary_correlation = {
        'total_samples': int(per_sample_data['num_samples']),
        'task_error_rate': float(correlation_data['task_error_rate']),
        'total_task_errors': int(correlation_data['total_task_errors']),
        'num_concepts': int(correlation_data['num_concepts']),
        'avg_pearson_correlation': float(np.mean(correlation_data['pearson_correlations'])),
        'std_pearson_correlation': float(np.std(correlation_data['pearson_correlations'])),
        'concepts_with_high_correlation': len([c for c in correlation_data['pearson_correlations'] if abs(c) > 0.2]),
        'concepts_with_high_error_rate': len([c for c in correlation_data['concept_error_rates'] if c > 0.1])
    }
    
    summary_path = output_dir / "correlation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary_correlation, f, indent=2)
    print(f"Correlation summary saved to: {summary_path}")


def print_global_correlation_statistics(comprehensive_data):
    """
    Print detailed global correlation statistics between concept errors and task errors.
    
    Args:
        comprehensive_data: Dictionary containing comprehensive analysis results
    """
    print("\n" + "="*70)
    print("GLOBAL CONCEPT-TASK ERROR CORRELATION ANALYSIS")
    print("="*70)
    
    num_concepts = comprehensive_data['num_concepts']
    global_task_error_rate = comprehensive_data['global_task_error_rate']
    global_total_task_errors = comprehensive_data['global_total_task_errors']
    
    print(f"Total samples analyzed: {comprehensive_data['global_num_samples']}")
    print(f"Global task error rate: {global_task_error_rate:.3f} ({global_total_task_errors} errors)")
    print(f"Number of concepts: {num_concepts}")
    
    print("\nGlobal Correlation Analysis by Concept:")
    print("-" * 50)
    print(f"{'Concept':<8} {'Pearson':<10} {'Error Rate':<12}")
    print("-" * 50)
    
    for i in range(num_concepts):
        pearson = comprehensive_data['global_pearson_correlations'][i]
        error_rate = comprehensive_data['global_concept_error_rates'][i]
        
        print(f"C{i:<7} {pearson:<10.3f} {error_rate:<12.3f}")
    
    # Identify most important concepts globally
    print("\n" + "="*70)
    print("MOST IMPORTANT CONCEPTS FOR TASK PREDICTION (GLOBAL)")
    print("="*70)
    
    # Sort concepts by absolute Pearson correlation
    pearson_correlations = comprehensive_data['global_pearson_correlations']
    sorted_indices = np.argsort(np.abs(pearson_correlations))[::-1]
    
    print("Top 10 most important concepts (by absolute Pearson correlation):")
    print("-" * 50)
    for rank, concept_idx in enumerate(sorted_indices[:10]):
        pearson = pearson_correlations[concept_idx]
        error_rate = comprehensive_data['global_concept_error_rates'][concept_idx]
        print(f"{rank+1:2d}. Concept {concept_idx:2d}: Pearson={pearson:.3f}, Error Rate={error_rate:.3f}")
    
    # Identify concepts with strong positive correlation
    strong_positive_corr = []
    strong_negative_corr = []
    
    for i in range(num_concepts):
        pearson = comprehensive_data['global_pearson_correlations'][i]
        if pearson > 0.3:
            strong_positive_corr.append((i, pearson))
        elif pearson < -0.3:
            strong_negative_corr.append((i, pearson))
    
    if strong_positive_corr:
        print("\nConcepts with strong positive correlation (>0.3) with task errors:")
        print("-" * 50)
        for concept_idx, corr in sorted(strong_positive_corr, key=lambda x: x[1], reverse=True):
            print(f"  Concept {concept_idx:2d}: Pearson correlation = {corr:.3f}")
    
    if strong_negative_corr:
        print("\nConcepts with strong negative correlation (<-0.3) with task errors:")
        print("-" * 50)
        for concept_idx, corr in sorted(strong_negative_corr, key=lambda x: x[1]):
            print(f"  Concept {concept_idx:2d}: Pearson correlation = {corr:.3f}")
    
    # Summary statistics
    print("\n" + "="*70)
    print("GLOBAL CORRELATION SUMMARY STATISTICS")
    print("="*70)
    
    pearson_corrs = comprehensive_data['global_pearson_correlations']
    
    print(f"Average Pearson correlation: {np.mean(pearson_corrs):.3f} ± {np.std(pearson_corrs):.3f}")
    
    print(f"\nConcepts with high correlation (|Pearson| > 0.2): {len([c for c in pearson_corrs if abs(c) > 0.2])}")
    print(f"Concepts with high error rate (>0.1): {len([c for c in comprehensive_data['global_concept_error_rates'] if c > 0.1])}")

def main(config: DictConfig):
    """
    Main function to analyze concept statistics for erroneous examples.
    """
    project_dir = Path(__file__).absolute().parent
    print("Project directory:", project_dir)
    print("Config:", config)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Reset random seeds
    gen = reset_random_seeds(config.seed)
    
    # Get data
    _, val_loader, _ = get_data(config, config.data, gen)
    # Load pretrained CBM model
    pretrained_cbm = create_model(config)
    pretrained_cbm = pretrained_cbm.to(device)
    pretrained_cbm.load_state_dict(torch.load(
        config.model.pretrained_model_path, weights_only=False
    ))
    print(f"Pretrained CBM loaded from {config.model.pretrained_model_path}")
    
    # Comprehensive analysis in a single pass
    print("\nPerforming comprehensive concept and correlation analysis...")
    comprehensive_data = analyze_comprehensive_concept_statistics(val_loader, pretrained_cbm, device, config)
    
    # Print global statistics
    print("\n" + "="*70)
    print("GLOBAL ANALYSIS RESULTS")
    print("="*70)
    print(f"Task error rate: {comprehensive_data['global_task_error_rate']:.3f}")
    
    # Print per-class statistics
    for class_idx in range(comprehensive_data['num_classes']):
        class_data = comprehensive_data['class_data'][class_idx]
        if class_data['total_samples'] > 0:
            print(f"\nClass {class_idx}:")
            if class_data['total_samples'] > 0:
                print(f"  Class error rate: {class_data['task_error_rate']:.3f}")
                
    # Print global correlation statistics
    print_global_correlation_statistics(comprehensive_data)
    
    # Create output directory
    output_dir = Path(config.experiment_dir) / "concept_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot correlation analysis
    print("\nGenerating correlation analysis plots...")
    plot_correlation_analysis(comprehensive_data, output_dir, config)
    
    # Plot concept histograms and precision-recall per class
    print("\nGenerating per-class concept analysis plots...")
    plot_per_class_concept_analysis(comprehensive_data, output_dir, config)
    
    # Save comprehensive data
    print("\nSaving comprehensive analysis data...")
    save_comprehensive_analysis(comprehensive_data, output_dir)
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    EXP_PATH = './experiments/cbm/Waterbirds/20251204-110018_CBM_debugging_code_flow/'
    with open(os.path.join(EXP_PATH, "config.yaml"), "r") as f:
        experiment_config = yaml.load(f, Loader=yaml.FullLoader)
        experiment_config = DictConfig(experiment_config)
    experiment_config.logging.mode = "disabled"
    experiment_config.model.pretrained_model_path = os.path.join(EXP_PATH, "model_last.pth")
    main(config=experiment_config)
