#!/usr/bin/env python3
"""
Script to calculate average precision for each population_id across all seeds.

Usage:
    from calc_method_statistics import main
    main(directory_path, output=None, aggregated_pop_ids=None)
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict


def calculate_average_precision(directory_path):
    """
    Calculate average precision and std for each population_id across all seeds.

    Args:
        directory_path: Path to directory containing seed_XXX subdirectories

    Returns:
        DataFrame with columns: population_id, mean_precision, std_precision,
        num_seeds
    """
    directory = Path(directory_path)

    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory_path}")

    # Find all seed directories
    seed_dirs = sorted([d for d in directory.iterdir()
                        if d.is_dir() and d.name.startswith('seed_')])

    if not seed_dirs:
        raise ValueError(
            f"No seed_XXX directories found in {directory_path}"
        )

    # Collect precision data from all seeds
    all_precisions = defaultdict(list)  # population_id -> list of precisions
    
    for seed_dir in seed_dirs:
        prec_10_path = seed_dir / 'precision_at_10.csv'

        if not prec_10_path.exists():
            print(f"Warning: {prec_10_path} not found, skipping {seed_dir.name}")
            continue

        try:
            df = pd.read_csv(prec_10_path)

            # Validate required columns
            required_cols = ['population_id', 'cluster_id', 'precision']
            if not all(col in df.columns for col in required_cols):
                print(
                    f"Warning: {prec_10_path} missing required columns, skipping"
                )
                continue

            # Group by population_id and collect precisions
            for _, row in df.iterrows():
                pop_id = row['population_id']
                precision = row['precision']
                all_precisions[pop_id].append(precision)

            print(f"Loaded data from {seed_dir.name}")

        except Exception as e:
            print(f"Error reading {prec_10_path}: {e}")
            continue

    if not all_precisions:
        raise ValueError(
            "No valid precision data found in any seed directory"
        )

    # Calculate mean and std precision for each population_id
    results = []
    for pop_id in sorted(all_precisions.keys()):
        precisions = all_precisions[pop_id]
        mean_precision = np.mean(precisions)
        std_precision = np.std(precisions, ddof=1)  # Sample std
        results.append({
            'population_id': pop_id,
            'mean_precision': mean_precision,
            'std_precision': std_precision,
            'num_seeds': len(precisions)
        })

    result_df = pd.DataFrame(results)
    return result_df, all_precisions


def calculate_aggregated_statistics(all_precisions, aggregated_pop_ids):
    """
    Calculate aggregated statistics for combined population_ids.

    Args:
        all_precisions: Dictionary mapping population_id -> list of precisions
        aggregated_pop_ids: List of lists, where each inner list contains
            population_ids to aggregate together

    Returns:
        DataFrame with columns: aggregated_pop_ids, mean_precision,
        std_precision, num_seeds
    """
    aggregated_results = []
    n_seeds = len(all_precisions[0])
    
    for pop_id_group in aggregated_pop_ids:
        # Collect all precisions from the specified population_ids
        combined_precisions = np.zeros(n_seeds)
        for pop_id in pop_id_group:
            if pop_id in all_precisions:
                combined_precisions += all_precisions[pop_id]
            else:
                print(
                    f"Warning: population_id {pop_id} not found, "
                    f"skipping in aggregation"
                )
        combined_precisions /= len(pop_id_group)

        # Calculate mean and std for combined population_ids
        mean_precision = np.mean(combined_precisions)
        std_precision = np.std(combined_precisions, ddof=1)  # Sample std

        # Format population_ids as string for display
        pop_ids_str = ','.join(map(str, sorted(pop_id_group)))

        aggregated_results.append({
            'aggregated_pop_ids': pop_ids_str,
            'mean_precision': mean_precision,
            'std_precision': std_precision,
        })

    return pd.DataFrame(aggregated_results)

def calculate_average_homogeinity_and_completeness_score_stats(directory_path):
    """
    Calculate average homogeneity and completeness score for each population_id across all seeds.

    Args:
        directory_path: Path to directory containing seed_XXX subdirectories

    Returns:
        DataFrame with columns: population_id, mean_homogeneity, std_homogeneity, mean_v_measure_score, std_v_measure_score, num_seeds
    """
    directory = Path(directory_path)

    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory_path}")

    # Find all seed directories
    seed_dirs = sorted([d for d in directory.iterdir()
                        if d.is_dir() and d.name.startswith('seed_')])

    if not seed_dirs:
        raise ValueError(
            f"No seed_XXX directories found in {directory_path}"
        )
    # Find all seed directories
    all_homogeneity_scores = []
    all_purity_scores = []
    all_completeness_scores = []

    for seed_dir in seed_dirs:
        scores_path = seed_dir / 'metric_scores.csv'

        if not scores_path.exists():
            print(f"Warning: {scores_path} not found, skipping {seed_dir.name}")
            continue

        try:
            df = pd.read_csv(scores_path)

            # Validate required columns
            required_cols = ['error_slices_homogeneity_score', 'underperforming_populations_completeness_score']
            if not all(col in df.columns for col in required_cols):
                print(
                    f"Warning: {scores_path} missing required columns, skipping"
                )
                continue

            # Group by population_id and collect v_measure_scores
            for _, row in df.iterrows():
                error_slices_homogeneity_score = row['error_slices_homogeneity_score']
                # error_slices_purity_score = row['error_slices_purity_score']
                error_slices_completeness_score = row['underperforming_populations_completeness_score']
                all_homogeneity_scores.append(error_slices_homogeneity_score)
                # all_purity_scores.append(error_slices_purity_score)
                all_completeness_scores.append(error_slices_completeness_score)
        
        except Exception as e:
            print(f"Error reading {scores_path}: {e}")
            continue

    if not all_homogeneity_scores or not all_completeness_scores:
        raise ValueError(
            "No valid homogeneity or v_measure_score data found in any seed directory"
        )

    # Calculate mean and std for combined population_ids
    mean_homogeneity_score = np.mean(all_homogeneity_scores)
    std_homogeneity_score = np.std(all_homogeneity_scores, ddof=1)  # Sample std
    # mean_purity_score = np.mean(all_purity_scores)
    # std_purity_score = np.std(all_purity_scores, ddof=1)  # Sample std
    mean_completeness_score = np.mean(all_completeness_scores)
    std_completeness_score = np.std(all_completeness_scores, ddof=1)  # Sample std

    return pd.DataFrame({
        'mean_homogeneity_score': mean_homogeneity_score,
        'std_homogeneity_score': std_homogeneity_score,
        # 'mean_purity_score': mean_purity_score,
        # 'std_purity_score': std_purity_score,
        'mean_completeness_score': mean_completeness_score,
        'std_completeness_score': std_completeness_score,
        'num_seeds': len(all_homogeneity_scores)
    }, index=[0])


def calculate_average_underperforming_population_frequency_in_rep_slice(directory_path, aggregated_pop_ids):
    """
    Calculate average underperforming populations frequency for each population_id across all seeds.

    Args:
        directory_path: Path to directory containing seed_XXX subdirectories

    Returns:
        DataFrame with columns: population_id, mean_underperforming_populations_frequency, std_underperforming_populations_frequency, num_seeds
    """

    directory = Path(directory_path)

    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory_path}")

    # Find all seed directories
    seed_dirs = sorted([d for d in directory.iterdir()
                        if d.is_dir() and d.name.startswith('seed_')])

    if not seed_dirs:
        raise ValueError(
            f"No seed_XXX directories found in {directory_path}"
        )

    results = []
    for pop_id_group in aggregated_pop_ids:
        pop_group_result = []
        for seed_dir in seed_dirs:
            prec_10_path = seed_dir / 'precision_at_10.csv'

            if not prec_10_path.exists():
                print(f"Warning: {prec_10_path} not found, skipping {seed_dir.name}")
                continue

            df = pd.read_csv(prec_10_path)

            # Validate required columns
            required_cols = ['population_id', 'cluster_id', 'frequency']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Required columns ({required_cols}) missing in {prec_10_path}. Please check the format of the CSV file.")
            
            combined_frequency = 0
            # Group by population_id and collect frequencies
            for _, row in df.iterrows():
                pop_id = row['population_id']
                if pop_id in pop_id_group:
                    combined_frequency += row['frequency']
            pop_group_result.append((combined_frequency / len(pop_id_group)))
        
        results.append({
            'aggregated_pop_ids': pop_id_group,
            'mean_frequency': np.mean(pop_group_result),
            'std_frequency': np.std(pop_group_result, ddof=1),
        })

    return pd.DataFrame(results)


def main(directory_path, output=None, aggregated_pop_ids=None):
    """
    Main function to calculate precision statistics.

    Args:
        directory_path: Path to directory containing seed_XXX subdirectories
        output: Optional output CSV file path (default: None, print to stdout)
        aggregated_pop_ids: Optional list of lists of population_ids to
            aggregate. Example: [[1, 2], [3, 4]] will calculate statistics
            for combined [1,2] and combined [3,4]

    Returns:
        0 on success, 1 on error
    """
    try:
        result_df, all_precisions = calculate_average_precision(
            directory_path
        )

        result_df_homogeneity_and_v_measure_score = calculate_average_homogeinity_and_completeness_score_stats(
            directory_path
        )

        result_df_underperforming_population_frequency = calculate_average_underperforming_population_frequency_in_rep_slice(
            directory_path, aggregated_pop_ids
        )

        # Print individual population_id statistics
        print("\nPrecision Statistics by Population ID:")
        print("=" * 60)
        print(result_df.to_string(index=False))

        # Calculate and print aggregated statistics if requested
        if aggregated_pop_ids:
            aggregated_df = calculate_aggregated_statistics(
                all_precisions, aggregated_pop_ids
            )
            if not aggregated_df.empty:
                print("\nAggregated Precision Statistics:")
                print("=" * 60)
                print(aggregated_df.to_string(index=False))

                # Save aggregated results if output specified
                if output:
                    # Save aggregated results to separate file
                    output_path = Path(output)
                    aggregated_output = (
                        output_path.parent
                        / f"{output_path.stem}_aggregated{output_path.suffix}"
                    )
                    aggregated_df.to_csv(aggregated_output, index=False)
                    print(f"\nAggregated results saved to {aggregated_output}")

        print("\nHomogeneity and V-Measure Score Statistics:")
        print("=" * 60)
        print(result_df_homogeneity_and_v_measure_score.to_string(index=False))

        # Save to CSV if output specified
        if output:
            result_df.to_csv(output, index=False)
            v_score_path = output.replace("precision_at_10_statistics.csv", "v_score_statistics.csv")
            result_df_homogeneity_and_v_measure_score.to_csv(v_score_path, index=False)
            print(f"\nIndividual results saved to {output} and {v_score_path}")
            underperforming_population_frequency_path = output.replace("precision_at_10_statistics.csv", "err_population_frequency_statistics.csv")
            result_df_underperforming_population_frequency.to_csv(underperforming_population_frequency_path, index=False)
            print(f"\nIndividual results saved to {output} and {underperforming_population_frequency_path}")
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    dir = ["domino_pca_weights:40_k=2", "kmeans_k=4", "spotlight_k=2", "george_k=3"]
    base_directory_path = "./experiments/cbm/Waterbirds/Waterbirds_Attributes_with_Background/20251012-212000_Task_Baseline_wd:0.00001_lr:0.01_decrease:40_sgd/Evaluations_valEqTrain_v3/Slices/"
    for d in dir:
        print(f"Calculating statistics for {d}")
        directory_path = os.path.join(base_directory_path, d)
        output = os.path.join(directory_path, "precision_at_10_statistics.csv")
        aggregated_pop_ids = [[1, 2]]
        main(directory_path=directory_path, output=output, aggregated_pop_ids=aggregated_pop_ids)
