import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import argparse

def load_data(file_path):
    """
    Loads the CSV file into a pandas DataFrame.
    
    Parameters:
        file_path (str): Path to the CSV file.
    
    Returns:
        DataFrame: Loaded data.
    """
    try:
        data = pd.read_csv(file_path)
        print(data)
        return data
    except Exception as e:
        print(f"Error loading file: {e}")
        return None
    
def plot_metric_distributions(data, metrics, bins=50, figsize=(15, 10)):
    """
    Plots the distributions of specified metrics from the DataFrame.
    
    Parameters:
        data (DataFrame): Processed data containing the metrics.
        metrics (list of str): List of metrics to plot (e.g., ['TP', 'TN', 'FP', 'FN']).
        bins (int): Number of bins for histograms.
        figsize (tuple): Figure size for the plots.
    """
    num_metrics = len(metrics)
    fig, axes = plt.subplots(nrows=(num_metrics + 1) // 2, ncols=2, figsize=figsize)

    # Flatten axes for easy iteration
    axes = axes.flatten() if num_metrics > 1 else [axes]
    
    for idx, metric in enumerate(metrics):
        if metric not in data.columns:
            print(f"Metric '{metric}' not found in data. Skipping...")
            continue

        # Plot the distribution
        sns.histplot(data[metric], bins=bins, kde=True, ax=axes[idx])
        axes[idx].set_title(f"Distribution of {metric}")
        axes[idx].set_xlabel(metric)
        axes[idx].set_ylabel("Frequency")
        axes[idx].grid(True)

    # Remove any empty subplots
    for idx in range(len(metrics), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.show()

def preprocess_data(data):
    """
    Preprocesses the data to clean up invalid entries (e.g., '#######').
    
    Parameters:
        data (DataFrame): The raw DataFrame.
    
    Returns:
        DataFrame: Cleaned data.
    """
    # Replace invalid entries (e.g., ######) with NaN and drop rows with NaNs
    data.replace("#######", pd.NA, inplace=True)
    data.dropna(inplace=True)
    
    # Convert columns to numeric where possible
    for col in data.columns:
        if col not in ['SUBJECT', 'LABEL']:  # Skip non-numeric columns
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    return data

def main(file_path, metrics, bins, figsize):
    # todo: load the "results.csv" file from the mia-results directory
    # todo: read the data into a list
    # todo: plot the Dice coefficients per label (i.e. white matter, gray matter, hippocampus, amygdala, thalamus)
    #  in a boxplot

    # alternative: instead of manually loading/reading the csv file you could also use the pandas package
    # but you will need to install it first ('pip install pandas') and import it to this file ('import pandas as pd')

    # Load the data
    data = load_data(file_path)
    if data is None:
        return

    # Preprocess the data
    data = preprocess_data(data)

    # Plot the metric distributions
    plot_metric_distributions(data, metrics, bins=bins, figsize=figsize)

if __name__ == "__main__":
    
    # Argument parsing
    parser = argparse.ArgumentParser(description="Plot distributions of evaluation metrics from CSV.")
    parser.add_argument("--file_path", type=str, help="Path to the CSV file containing the results.")
    parser.add_argument("--metrics", nargs="+", default=["TP", "TN", "FP", "FN"],
                        help="List of metrics to plot (default: ['TP', 'TN', 'FP', 'FN']).")
    parser.add_argument("--bins", type=int, default=50, help="Number of bins for histograms (default: 50).")
    parser.add_argument("--figsize", type=tuple, default=(15, 10), help="Figure size for the plots (default: (15, 10)).")
    
    args = parser.parse_args()

    # Pass arguments to main
    main(args.file_path, args.metrics, args.bins, args.figsize)

