import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import random
import csv
from tqdm import tqdm
import argparse
import subprocess
import sys
import time
import re
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc
from collections import Counter

# Set a colorblind-friendly palette for all plots
COLORS = ['#0173B2', '#DE8F05', '#029E73']  # Blue, Orange, Green

# Configuration - Improved defaults with fallbacks
DEFAULT_BASE_DATASET_PATH = "D:/TARUMT/DSPC/imageDataset"  # Path to your cancer dataset
RESULTS_FOLDER = "./results"
DATASETS_FOLDER = "./benchmark_datasets"

# For Windows, look for the executable in common locations
def find_executable():
    possible_paths = [
        "D:/TARUMT/DSPC/x64/Debug/DSPC.exe",
        "./x64/Debug/DSPC.exe",
        "./DSPC.exe"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None

EXECUTABLE_PATH = find_executable()

# Create folders if they don't exist
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER + "/figures", exist_ok=True)
os.makedirs(DATASETS_FOLDER, exist_ok=True)

def detect_existing_datasets(base_path):
    """Detect datasets already available in the specified path"""
    dataset_sizes = []
    pattern = re.compile(r'data(\d+)')
    
    try:
        # Check if directory exists
        if not os.path.exists(base_path):
            print(f"Warning: Base dataset path {base_path} does not exist")
            return []
            
        # List directories that match the pattern data<size>
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            if os.path.isdir(item_path):
                match = pattern.match(item)
                if match:
                    size = int(match.group(1))
                    
                    # Verify this is a valid dataset with the expected structure
                    image_folder = os.path.join(item_path, "sampled_images")
                    label_file = os.path.join(item_path, "sampled_labels.csv")
                    
                    if os.path.exists(image_folder) and os.path.exists(label_file):
                        print(f"Found valid dataset: {item} with {size} images")
                        dataset_sizes.append(size)
                    else:
                        print(f"Found dataset folder {item} but missing required files:")
                        if not os.path.exists(image_folder):
                            print(f"  - Missing: {image_folder}")
                        if not os.path.exists(label_file):
                            print(f"  - Missing: {label_file}")
        
        return sorted(dataset_sizes)
    except Exception as e:
        print(f"Error detecting existing datasets: {e}")
        return []

def load_ground_truth_from_dataset(dataset_path):
    """
    Load ground truth labels from a dataset folder containing sampled_labels.csv
    
    Args:
        dataset_path: Path to the dataset folder
        
    Returns:
        Dictionary mapping filenames to labels (1 for malignant, 0 for benign)
    """
    filename_to_label = {}
    
    # Check if dataset path exists
    if not os.path.exists(dataset_path):
        print(f"Dataset path does not exist: {dataset_path}")
        return filename_to_label
    
    # Look for sampled_labels.csv
    labels_path = os.path.join(dataset_path, 'sampled_labels.csv')
    if not os.path.exists(labels_path):
        print(f"Labels file not found: {labels_path}")
        return filename_to_label
    
    try:
        # Load the CSV file
        labels_df = pd.read_csv(labels_path)
        print(f"Found labels CSV with {len(labels_df)} entries")
        
        # Expected format: id,label,filename (may vary)
        required_columns = ['id', 'label']
        if not all(col in labels_df.columns for col in required_columns):
            print(f"Labels CSV missing required columns. Expected: {required_columns}, Found: {labels_df.columns.tolist()}")
            return filename_to_label
        
        # Determine the filename column (usually the third column)
        filename_col = 'filename'
        if filename_col not in labels_df.columns and len(labels_df.columns) >= 3:
            filename_col = labels_df.columns[2]  # Assume third column is filename
        
        if filename_col not in labels_df.columns:
            print(f"Could not identify filename column in labels CSV")
            return filename_to_label
        
        # Process each row to build the mapping
        for _, row in labels_df.iterrows():
            # Get the filename (could be with or without path)
            filename = row[filename_col]
            if not isinstance(filename, str):
                filename = str(filename)
            
            # Extract the base filename without path or extension
            base_filename = os.path.basename(filename)
            name_without_ext, _ = os.path.splitext(base_filename)
            
            # Get the label - assume 1 is malignant, 0 is benign
            label = int(row['label'])
            
            # Store mappings for various filename formats
            filename_to_label[base_filename] = label
            filename_to_label[name_without_ext] = label
            
            # Also store with different extensions for flexibility
            for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                filename_to_label[name_without_ext + ext] = label
        
        print(f"Created mapping for {len(filename_to_label)} unique filenames")
        return filename_to_label
        
    except Exception as e:
        print(f"Error loading ground truth from dataset: {e}")
        return {}

def match_predictions_with_ground_truth(predictions_df, filename_to_label):
    """
    Match prediction results with ground truth labels
    
    Args:
        predictions_df: DataFrame with prediction results
        filename_to_label: Dictionary mapping filenames to labels
        
    Returns:
        DataFrame with added True_Label column
    """
    if 'Image' not in predictions_df.columns:
        print("Error: Predictions DataFrame missing 'Image' column")
        return predictions_df
    
    # Create a copy to avoid modifying the original
    df = predictions_df.copy()
    
    # Function to match an image path with a ground truth label
    def get_label_for_image(image_path):
        # Try different variations of the filename
        filename = os.path.basename(image_path)
        name_without_ext, _ = os.path.splitext(filename)
        
        # Try exact match first
        if filename in filename_to_label:
            return filename_to_label[filename]
        
        # Try without extension
        if name_without_ext in filename_to_label:
            return filename_to_label[name_without_ext]
        
        # Try partial matches (some datasets add prefixes to filenames)
        for key in filename_to_label:
            if key in filename or filename in key:
                return filename_to_label[key]
        
        return None
    
    # Apply the matching function to each image
    df['True_Label'] = df['Image'].apply(get_label_for_image)
    
    # Report matching results
    matched_count = df['True_Label'].notna().sum()
    print(f"Successfully matched {matched_count} of {len(df)} predictions with ground truth labels")
    
    return df

def run_interactive(executable_path):
    """
    Run the DSPC.exe in interactive mode, allowing direct user input
    """
    if not os.path.exists(executable_path):
        print(f"Error: Executable not found at {executable_path}")
        return False
    
    print(f"\nLaunching {executable_path} in interactive mode...")
    print("You can interact with the program directly. Close the window when finished.")
    
    try:
        # Launch the program in a way that allows the user to interact with it
        process = subprocess.Popen(executable_path)
        
        # Wait for the process to complete - the user will interact with it directly
        process.wait()
        
        # After the process finishes, check if a benchmark.csv was created
        if os.path.exists("benchmark.csv"):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(RESULTS_FOLDER, f"benchmark_{timestamp}.csv")
            shutil.move("benchmark.csv", output_file)
            print(f"\nBenchmark results saved to {output_file}")
            
            # Generate plots immediately if requested by the user
            print("Generating plots from the benchmark results...")
            try:
                # Read the benchmark CSV file
                df = pd.read_csv(output_file)
                
                # Generate plots in the same folder
                generate_quick_plots(df, os.path.dirname(output_file))
                print(f"Plots have been saved to {RESULTS_FOLDER}/")
            except Exception as e:
                print(f"Error generating plots: {e}")
        
        return True
    except Exception as e:
        print(f"Error running in interactive mode: {e}")
        return False

def run_benchmark(executable_path, dataset_size, base_path):
    # Input sequence for automated CLI usage:
    # 0. Select dataset (dataset index)
    # 1. Run Serial
    # 2. Run OpenMP
    # 3. Run CUDA
    # 4. Show comparison (which saves benchmark.csv)
    # 6. Exit
    
    if not os.path.exists(executable_path):
        print(f"Error: Executable not found at {executable_path}")
        return False
    
    # Find the dataset index based on the size
    datasets = detect_existing_datasets(base_path)
    if not datasets:
        print(f"Error: No datasets found in {base_path}")
        return False
    
    try:
        dataset_index = datasets.index(dataset_size) + 1  # +1 because menu is 1-indexed
    except ValueError:
        print(f"Error: Dataset with size {dataset_size} not found in {datasets}")
        return False
    
    print(f"Found dataset with size {dataset_size} at index {dataset_index}")
    
    # Input text with dataset selection
    # Format:
    # 0 (select dataset)
    # <dataset_index>
    # 1 (run Serial)
    # 2 (run OpenMP)
    # 3 (run CUDA)
    # 4 (show comparison/save results)
    # 6 (exit)
    input_text = f"0\n{dataset_index}\n1\n2\n3\n4\n6\n"
    
    print(f"\nRunning benchmark on dataset size {dataset_size}...")
    
    try:
        # Run the program and automate inputs
        process = subprocess.Popen(
            executable_path, 
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = process.communicate(input=input_text, timeout=7200)  # 2-hour timeout
        
        # Debug info
        print(f"Program output (first 500 characters):")
        print(stdout[:500] + "..." if len(stdout) > 500 else stdout)
        
        if stderr:
            print(f"Warning: Program reported errors: {stderr[:200]}...")
        
        # Check if the benchmark.csv was created
        if os.path.exists("benchmark.csv"):
            # Rename it to include dataset size
            output_file = os.path.join(RESULTS_FOLDER, f"benchmark_{dataset_size}.csv")
            shutil.move("benchmark.csv", output_file)
            print(f"Benchmark results saved to {output_file}")
            
            # Generate plots immediately 
            print("Generating plots from the benchmark results...")
            try:
                df = pd.read_csv(output_file)
                plot_dir = os.path.join(RESULTS_FOLDER, f"plots_{dataset_size}")
                os.makedirs(plot_dir, exist_ok=True)
                generate_quick_plots(df, plot_dir)
                print(f"Plots have been saved to {plot_dir}/")
            except Exception as e:
                print(f"Error generating plots: {e}")
                
            return True
        else:
            print("Warning: benchmark.csv not created after program execution.")
            return False
    
    except subprocess.TimeoutExpired:
        process.kill()
        print(f"Error: Benchmark timed out for dataset size {dataset_size}")
        return False
    except Exception as e:
        print(f"Error running benchmark: {e}")
        return False
    # Input sequence for automated CLI usage:
    # 0. Select dataset (dataset index)
    # 1. Run Serial
    # 2. Run OpenMP
    # 3. Run CUDA
    # 4. Show comparison (which saves benchmark.csv)
    # 6. Exit
    
    if not os.path.exists(executable_path):
        print(f"Error: Executable not found at {executable_path}")
        return False
    
    # Find all available datasets to determine the index
    dataset_dir = os.path.dirname(executable_path)
    if not dataset_dir:
        dataset_dir = "."
        
    # Look in the parent directory for "imageDataset" folder
    dataset_base = os.path.join(os.path.dirname(dataset_dir), "imageDataset")
    if not os.path.exists(dataset_base):
        dataset_base = os.path.join(dataset_dir, "imageDataset")
        if not os.path.exists(dataset_base):
            print(f"Error: Could not find imageDataset folder")
            return False
    
    datasets = detect_existing_datasets(dataset_base)
    if not datasets:
        print(f"Error: No datasets found in {dataset_base}")
        return False
    
    # Find the dataset index based on the size
    try:
        dataset_index = datasets.index(dataset_size) + 1  # +1 because menu is 1-indexed
    except ValueError:
        print(f"Error: Dataset with size {dataset_size} not found in {datasets}")
        return False
    
    print(f"Found dataset with size {dataset_size} at index {dataset_index}")
    
    # Input text with dataset selection
    # Format:
    # 0 (select dataset)
    # <dataset_index>
    # 1 (run Serial)
    # 2 (run OpenMP)
    # 3 (run CUDA)
    # 4 (show comparison/save results)
    # 6 (exit)
    input_text = f"0\n{dataset_index}\n1\n2\n3\n4\n6\n"
    
    print(f"\nRunning benchmark on dataset size {dataset_size}...")
    
    try:
        # Run the program and automate inputs
        process = subprocess.Popen(
            executable_path, 
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = process.communicate(input=input_text, timeout=7200)  # 2-hour timeout
        
        # Debug info
        if stderr:
            print(f"Warning: Program reported errors: {stderr[:200]}...")
        
        # Check if the benchmark.csv was created
        if os.path.exists("benchmark.csv"):
            # Rename it to include dataset size
            output_file = os.path.join(RESULTS_FOLDER, f"benchmark_{dataset_size}.csv")
            shutil.move("benchmark.csv", output_file)
            print(f"Benchmark results saved to {output_file}")
            
            # Generate plots immediately 
            print("Generating plots from the benchmark results...")
            try:
                df = pd.read_csv(output_file)
                plot_dir = os.path.join(RESULTS_FOLDER, f"plots_{dataset_size}")
                os.makedirs(plot_dir, exist_ok=True)
                generate_quick_plots(df, plot_dir)
                print(f"Plots have been saved to {plot_dir}/")
            except Exception as e:
                print(f"Error generating plots: {e}")
                
            return True
        else:
            print("Warning: benchmark.csv not created. Program output:")
            print(stdout[:500] + "..." if len(stdout) > 500 else stdout)
            return False
    
    except subprocess.TimeoutExpired:
        process.kill()
        print(f"Error: Benchmark timed out for dataset size {dataset_size}")
        return False
    except Exception as e:
        print(f"Error running benchmark: {e}")
        return False

def generate_quick_plots(df, output_dir):
    """
    Generate quick plots from benchmark data
    This function is called right after a benchmark completes
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Bar chart of execution times
    plt.figure(figsize=(10, 6))
    implementations = df['Implementation'].tolist()
    times = df['Execution Time (s)'].tolist()
    
    # Use colorblind-friendly colors
    colors = [COLORS[i % len(COLORS)] for i in range(len(implementations))]
    
    bars = plt.bar(implementations, times, color=colors)
    
    # Add execution time labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}s', ha='center', va='bottom', fontsize=12)
    
    plt.xlabel('Implementation', fontsize=14)
    plt.ylabel('Execution Time (seconds)', fontsize=14)
    plt.title('Cancer Detection CNN: Execution Time Comparison', fontsize=16)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'execution_times.png'), dpi=300)
    
    # 2. Bar chart of speedups
    plt.figure(figsize=(10, 6))
    # Filter for parallel implementations
    parallel_df = df[df['Implementation'] != 'Serial']
    
    if not parallel_df.empty:
        implementations = parallel_df['Implementation'].tolist()
        speedups = parallel_df['Speedup (vs Serial)'].tolist()
        
        colors = [COLORS[i % len(COLORS)] for i in range(len(implementations))]
        
        bars = plt.bar(implementations, speedups, color=colors)
        
        # Add speedup labels on top of the bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.2f}x', ha='center', va='bottom', fontsize=12)
        
        plt.xlabel('Implementation', fontsize=14)
        plt.ylabel('Speedup vs Serial', fontsize=14)
        plt.title('Cancer Detection CNN: Speedup Comparison', fontsize=16)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'speedups.png'), dpi=300)
    
    # 3. Bar chart of accuracies
    plt.figure(figsize=(10, 6))
    implementations = df['Implementation'].tolist()
    accuracies = df['Accuracy (%)'].tolist()
    
    colors = [COLORS[i % len(COLORS)] for i in range(len(implementations))]
    
    bars = plt.bar(implementations, accuracies, color=colors)
    
    # Add accuracy labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=12)
    
    plt.xlabel('Implementation', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('Cancer Detection CNN: Accuracy Comparison', fontsize=16)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracies.png'), dpi=300)
    
    # 4. Create a summary text file
    with open(os.path.join(output_dir, 'benchmark_summary.txt'), 'w') as f:
        f.write("Cancer Detection CNN Benchmark Summary\n")
        f.write("====================================\n\n")
        
        if 'Dataset Size' in df.columns and df['Dataset Size'].nunique() == 1:
            dataset_size = df['Dataset Size'].iloc[0]
            f.write(f"Dataset size: {dataset_size} images\n\n")
        
        f.write("Execution Time Comparison:\n")
        for i, row in df.iterrows():
            f.write(f"  - {row['Implementation']}: {row['Execution Time (s)']:.2f} seconds\n")
        
        f.write("\nSpeedup Comparison:\n")
        for i, row in df.iterrows():
            if row['Implementation'] != 'Serial':
                f.write(f"  - {row['Implementation']} Speedup: {row['Speedup (vs Serial)']:.2f}x\n")
        
        f.write("\nAccuracy Comparison:\n")
        for i, row in df.iterrows():
            f.write(f"  - {row['Implementation']} Accuracy: {row['Accuracy (%)']:.2f}%\n")
        
        f.write("\nGenerated on: " + time.strftime("%Y-%m-%d %H:%M:%S"))
    """
    Generate quick plots from benchmark data
    This function is called right after a benchmark completes
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Bar chart of execution times
    plt.figure(figsize=(10, 6))
    implementations = df['Implementation'].tolist()
    times = df['Execution Time (s)'].tolist()
    
    # Use colorblind-friendly colors
    colors = [COLORS[i % len(COLORS)] for i in range(len(implementations))]
    
    bars = plt.bar(implementations, times, color=colors)
    
    # Add execution time labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}s', ha='center', va='bottom', fontsize=12)
    
    plt.xlabel('Implementation', fontsize=14)
    plt.ylabel('Execution Time (seconds)', fontsize=14)
    plt.title('Cancer Detection CNN: Execution Time Comparison', fontsize=16)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'execution_times.png'), dpi=300)
    
    # 2. Bar chart of speedups
    plt.figure(figsize=(10, 6))
    # Filter for parallel implementations
    parallel_df = df[df['Implementation'] != 'Serial']
    
    if not parallel_df.empty:
        implementations = parallel_df['Implementation'].tolist()
        speedups = parallel_df['Speedup (vs Serial)'].tolist()
        
        colors = [COLORS[i % len(COLORS)] for i in range(len(implementations))]
        
        bars = plt.bar(implementations, speedups, color=colors)
        
        # Add speedup labels on top of the bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.2f}x', ha='center', va='bottom', fontsize=12)
        
        plt.xlabel('Implementation', fontsize=14)
        plt.ylabel('Speedup vs Serial', fontsize=14)
        plt.title('Cancer Detection CNN: Speedup Comparison', fontsize=16)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'speedups.png'), dpi=300)
    
    # 3. Bar chart of accuracies
    plt.figure(figsize=(10, 6))
    implementations = df['Implementation'].tolist()
    accuracies = df['Accuracy (%)'].tolist()
    
    colors = [COLORS[i % len(COLORS)] for i in range(len(implementations))]
    
    bars = plt.bar(implementations, accuracies, color=colors)
    
    # Add accuracy labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=12)
    
    plt.xlabel('Implementation', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('Cancer Detection CNN: Accuracy Comparison', fontsize=16)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracies.png'), dpi=300)
    
    # 4. Create a summary text file
    with open(os.path.join(output_dir, 'benchmark_summary.txt'), 'w') as f:
        f.write("Cancer Detection CNN Benchmark Summary\n")
        f.write("====================================\n\n")
        
        if 'Dataset Size' in df.columns and df['Dataset Size'].nunique() == 1:
            dataset_size = df['Dataset Size'].iloc[0]
            f.write(f"Dataset size: {dataset_size} images\n\n")
        
        f.write("Execution Time Comparison:\n")
        for i, row in df.iterrows():
            f.write(f"  - {row['Implementation']}: {row['Execution Time (s)']:.2f} seconds\n")
        
        f.write("\nSpeedup Comparison:\n")
        for i, row in df.iterrows():
            if row['Implementation'] != 'Serial':
                f.write(f"  - {row['Implementation']} Speedup: {row['Speedup (vs Serial)']:.2f}x\n")
        
        f.write("\nAccuracy Comparison:\n")
        for i, row in df.iterrows():
            f.write(f"  - {row['Implementation']} Accuracy: {row['Accuracy (%)']:.2f}%\n")
        
        f.write("\nGenerated on: " + time.strftime("%Y-%m-%d %H:%M:%S"))

def collect_benchmark_files():
    """Collect all benchmark CSV files from the results folder"""
    benchmark_files = []
    
    for file in os.listdir(RESULTS_FOLDER):
        if file.startswith("benchmark_") and file.endswith(".csv"):
            try:
                # Try to get size from filename (for automatically generated files)
                size_str = file.split("_")[1].split(".")[0]
                if size_str.isdigit():
                    size = int(size_str)
                    benchmark_files.append((size, os.path.join(RESULTS_FOLDER, file)))
                else:
                    # For manually generated files (from interactive mode)
                    benchmark_files.append((0, os.path.join(RESULTS_FOLDER, file)))
            except:
                print(f"Warning: Could not parse size from filename: {file}")
    
    return sorted(benchmark_files)  # Sort by size for consistency

def parse_benchmark_results(benchmark_files):
    """Parse benchmark CSV files and combine results"""
    all_results = []
    
    for size, file_path in benchmark_files:
        try:
            df = pd.read_csv(file_path)
            
            # Add dataset size if not already present
            if 'Dataset Size' not in df.columns:
                if size > 0:
                    df['Dataset Size'] = size
                else:
                    # For files from interactive mode, try to extract from the data
                    try:
                        # Get first dataset size in the file
                        df['Dataset Size'] = df['Dataset Size'].iloc[0] if 'Dataset Size' in df.columns else 0
                    except:
                        df['Dataset Size'] = 0
            
            all_results.append(df)
        except Exception as e:
            print(f"Error parsing benchmark file {file_path}: {e}")
    
    if not all_results:
        return None
    
    # Combine all results
    combined_results = pd.concat(all_results)
    
    # Save combined results
    combined_results.to_csv(os.path.join(RESULTS_FOLDER, 'combined_results.csv'), index=False)
    
    return combined_results
    """Parse benchmark CSV files and combine results"""
    all_results = []
    
    for size, file_path in benchmark_files:
        try:
            df = pd.read_csv(file_path)
            
            # Add dataset size if not already present
            if 'Dataset Size' not in df.columns:
                if size > 0:
                    df['Dataset Size'] = size
                else:
                    # For files from interactive mode, try to extract from the data
                    try:
                        # Get first dataset size in the file
                        df['Dataset Size'] = df['Dataset Size'].iloc[0] if 'Dataset Size' in df.columns else 0
                    except:
                        df['Dataset Size'] = 0
            
            all_results.append(df)
        except Exception as e:
            print(f"Error parsing benchmark file {file_path}: {e}")
    
    if not all_results:
        return None
    
    # Combine all results
    combined_results = pd.concat(all_results)
    
    # Save combined results
    combined_results.to_csv(os.path.join(RESULTS_FOLDER, 'combined_results.csv'), index=False)
    
    return combined_results

def plot_results(results_df):
    """Create plots from the benchmark results"""
    # Create figures directory
    figures_dir = os.path.join(RESULTS_FOLDER, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # Group data by dataset size
    grouped = results_df.groupby('Dataset Size')
    
    # Get all dataset sizes
    dataset_sizes = sorted(results_df['Dataset Size'].unique())
    
    # Prepare data for plotting
    sizes = []
    serial_times = []
    omp_times = []
    cuda_times = []
    serial_accuracies = []
    omp_accuracies = []
    cuda_accuracies = []
    omp_speedups = []
    cuda_speedups = []
    
    for size in dataset_sizes:
        size_data = grouped.get_group(size)
        
        serial_data = size_data[size_data['Implementation'] == 'Serial']
        omp_data = size_data[size_data['Implementation'] == 'OpenMP']
        cuda_data = size_data[size_data['Implementation'] == 'CUDA']
        
        if len(serial_data) > 0 and len(omp_data) > 0 and len(cuda_data) > 0:
            sizes.append(size)
            serial_times.append(serial_data['Execution Time (s)'].values[0])
            omp_times.append(omp_data['Execution Time (s)'].values[0])
            cuda_times.append(cuda_data['Execution Time (s)'].values[0])
            serial_accuracies.append(serial_data['Accuracy (%)'].values[0])
            omp_accuracies.append(omp_data['Accuracy (%)'].values[0])
            cuda_accuracies.append(cuda_data['Accuracy (%)'].values[0])
            omp_speedups.append(omp_data['Speedup (vs Serial)'].values[0])
            cuda_speedups.append(cuda_data['Speedup (vs Serial)'].values[0])
    
    # Plot 1: Execution time vs Dataset size
    plt.figure(figsize=(12, 8))
    plt.plot(sizes, serial_times, 'o-', linewidth=2, markersize=8, label='Serial', color=COLORS[0])
    plt.plot(sizes, omp_times, 'o-', linewidth=2, markersize=8, label='OpenMP', color=COLORS[1])
    plt.plot(sizes, cuda_times, 'o-', linewidth=2, markersize=8, label='CUDA', color=COLORS[2])
    plt.xlabel('Dataset Size (Number of Images)', fontsize=14)
    plt.ylabel('Execution Time (seconds)', fontsize=14)
    plt.title('Cancer Detection CNN: Execution Time vs Dataset Size', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'execution_time.png'), dpi=300)
    
    # Plot 2: Speedup vs Dataset size
    plt.figure(figsize=(12, 8))
    plt.plot(sizes, omp_speedups, 'o-', linewidth=2, markersize=8, label='OpenMP Speedup', color=COLORS[1])
    plt.plot(sizes, cuda_speedups, 'o-', linewidth=2, markersize=8, label='CUDA Speedup', color=COLORS[2])
    plt.xlabel('Dataset Size (Number of Images)', fontsize=14)
    plt.ylabel('Speedup (x times)', fontsize=14)
    plt.title('Cancer Detection CNN: Speedup vs Dataset Size', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'speedup.png'), dpi=300)
    
    # Plot 3: Accuracy vs Dataset size
    plt.figure(figsize=(12, 8))
    plt.plot(sizes, serial_accuracies, 'o-', linewidth=2, markersize=8, label='Serial', color=COLORS[0])
    plt.plot(sizes, omp_accuracies, 'o-', linewidth=2, markersize=8, label='OpenMP', color=COLORS[1])
    plt.plot(sizes, cuda_accuracies, 'o-', linewidth=2, markersize=8, label='CUDA', color=COLORS[2])
    plt.xlabel('Dataset Size (Number of Images)', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('Cancer Detection CNN: Accuracy vs Dataset Size', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'accuracy.png'), dpi=300)
    
    # Plot 4: Execution time vs Dataset size (log scale)
    plt.figure(figsize=(12, 8))
    plt.plot(sizes, serial_times, 'o-', linewidth=2, markersize=8, label='Serial', color=COLORS[0])
    plt.plot(sizes, omp_times, 'o-', linewidth=2, markersize=8, label='OpenMP', color=COLORS[1])
    plt.plot(sizes, cuda_times, 'o-', linewidth=2, markersize=8, label='CUDA', color=COLORS[2])
    plt.xlabel('Dataset Size (Number of Images)', fontsize=14)
    plt.ylabel('Execution Time (seconds)', fontsize=14)
    plt.title('Cancer Detection CNN: Execution Time vs Dataset Size (Log Scale)', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'execution_time_log.png'), dpi=300)
    
    # Create a summary text file
    with open(os.path.join(figures_dir, 'benchmark_summary.txt'), 'w') as f:
        f.write("Cancer Detection CNN Benchmark Summary\n")
        f.write("====================================\n\n")
        f.write(f"Dataset sizes tested: {', '.join(map(str, sizes))}\n\n")
        
        if sizes:
            largest_size = max(sizes)
            largest_idx = sizes.index(largest_size)
            
            f.write("Execution Time Comparison:\n")
            f.write(f"  - Serial: {serial_times[largest_idx]:.2f} seconds\n")
            f.write(f"  - OpenMP: {omp_times[largest_idx]:.2f} seconds\n")
            f.write(f"  - CUDA: {cuda_times[largest_idx]:.2f} seconds\n\n")
            
            f.write("Speedup Comparison:\n")
            f.write(f"  - OpenMP Speedup: {omp_speedups[largest_idx]:.2f}x\n")
            f.write(f"  - CUDA Speedup: {cuda_speedups[largest_idx]:.2f}x\n\n")
            
            f.write("Accuracy Comparison:\n")
            f.write(f"  - Serial Accuracy: {serial_accuracies[largest_idx]:.2f}%\n")
            f.write(f"  - OpenMP Accuracy: {omp_accuracies[largest_idx]:.2f}%\n")
            f.write(f"  - CUDA Accuracy: {cuda_accuracies[largest_idx]:.2f}%\n\n")
        
        f.write("Generated on: " + time.strftime("%Y-%m-%d %H:%M:%S"))
    
    print(f"All plots have been saved to {figures_dir}/")
    """Create plots from the benchmark results"""
    # Create figures directory
    figures_dir = os.path.join(RESULTS_FOLDER, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # Group data by dataset size
    grouped = results_df.groupby('Dataset Size')
    
    # Get all dataset sizes
    dataset_sizes = sorted(results_df['Dataset Size'].unique())
    
    # Prepare data for plotting
    sizes = []
    serial_times = []
    omp_times = []
    cuda_times = []
    serial_accuracies = []
    omp_accuracies = []
    cuda_accuracies = []
    omp_speedups = []
    cuda_speedups = []
    
    for size in dataset_sizes:
        size_data = grouped.get_group(size)
        
        serial_data = size_data[size_data['Implementation'] == 'Serial']
        omp_data = size_data[size_data['Implementation'] == 'OpenMP']
        cuda_data = size_data[size_data['Implementation'] == 'CUDA']
        
        if len(serial_data) > 0 and len(omp_data) > 0 and len(cuda_data) > 0:
            sizes.append(size)
            serial_times.append(serial_data['Execution Time (s)'].values[0])
            omp_times.append(omp_data['Execution Time (s)'].values[0])
            cuda_times.append(cuda_data['Execution Time (s)'].values[0])
            serial_accuracies.append(serial_data['Accuracy (%)'].values[0])
            omp_accuracies.append(omp_data['Accuracy (%)'].values[0])
            cuda_accuracies.append(cuda_data['Accuracy (%)'].values[0])
            omp_speedups.append(omp_data['Speedup (vs Serial)'].values[0])
            cuda_speedups.append(cuda_data['Speedup (vs Serial)'].values[0])
    
    # Plot 1: Execution time vs Dataset size
    plt.figure(figsize=(12, 8))
    plt.plot(sizes, serial_times, 'o-', linewidth=2, markersize=8, label='Serial', color=COLORS[0])
    plt.plot(sizes, omp_times, 'o-', linewidth=2, markersize=8, label='OpenMP', color=COLORS[1])
    plt.plot(sizes, cuda_times, 'o-', linewidth=2, markersize=8, label='CUDA', color=COLORS[2])
    plt.xlabel('Dataset Size (Number of Images)', fontsize=14)
    plt.ylabel('Execution Time (seconds)', fontsize=14)
    plt.title('Cancer Detection CNN: Execution Time vs Dataset Size', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'execution_time.png'), dpi=300)
    
    # Plot 2: Speedup vs Dataset size
    plt.figure(figsize=(12, 8))
    plt.plot(sizes, omp_speedups, 'o-', linewidth=2, markersize=8, label='OpenMP Speedup', color=COLORS[1])
    plt.plot(sizes, cuda_speedups, 'o-', linewidth=2, markersize=8, label='CUDA Speedup', color=COLORS[2])
    plt.xlabel('Dataset Size (Number of Images)', fontsize=14)
    plt.ylabel('Speedup (x times)', fontsize=14)
    plt.title('Cancer Detection CNN: Speedup vs Dataset Size', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'speedup.png'), dpi=300)
    
    # Plot 3: Accuracy vs Dataset size
    plt.figure(figsize=(12, 8))
    plt.plot(sizes, serial_accuracies, 'o-', linewidth=2, markersize=8, label='Serial', color=COLORS[0])
    plt.plot(sizes, omp_accuracies, 'o-', linewidth=2, markersize=8, label='OpenMP', color=COLORS[1])
    plt.plot(sizes, cuda_accuracies, 'o-', linewidth=2, markersize=8, label='CUDA', color=COLORS[2])
    plt.xlabel('Dataset Size (Number of Images)', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('Cancer Detection CNN: Accuracy vs Dataset Size', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'accuracy.png'), dpi=300)
    
    # Plot 4: Execution time vs Dataset size (log scale)
    plt.figure(figsize=(12, 8))
    plt.plot(sizes, serial_times, 'o-', linewidth=2, markersize=8, label='Serial', color=COLORS[0])
    plt.plot(sizes, omp_times, 'o-', linewidth=2, markersize=8, label='OpenMP', color=COLORS[1])
    plt.plot(sizes, cuda_times, 'o-', linewidth=2, markersize=8, label='CUDA', color=COLORS[2])
    plt.xlabel('Dataset Size (Number of Images)', fontsize=14)
    plt.ylabel('Execution Time (seconds)', fontsize=14)
    plt.title('Cancer Detection CNN: Execution Time vs Dataset Size (Log Scale)', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'execution_time_log.png'), dpi=300)
    
    # Plot 5: Speedup vs Dataset size (with curve fitting)
    plt.figure(figsize=(12, 8))
    
    # Try to fit a curve to the data points
    try:
        from scipy.optimize import curve_fit
        
        def fit_curve(x, a, b, c):
            return a * np.log(b * x) + c
        
        omp_params, _ = curve_fit(fit_curve, sizes, omp_speedups)
        cuda_params, _ = curve_fit(fit_curve, sizes, cuda_speedups)
        
        x_smooth = np.linspace(min(sizes), max(sizes), 100)
        plt.plot(sizes, omp_speedups, 'o', markersize=8, label='OpenMP Data', color=COLORS[1])
        plt.plot(x_smooth, fit_curve(x_smooth, *omp_params), '-', linewidth=2, label='OpenMP Trend', color=COLORS[1], alpha=0.6)
        plt.plot(sizes, cuda_speedups, 'o', markersize=8, label='CUDA Data', color=COLORS[2])
        plt.plot(x_smooth, fit_curve(x_smooth, *cuda_params), '-', linewidth=2, label='CUDA Trend', color=COLORS[2], alpha=0.6)
    except Exception as e:
        print(f"Warning: Curve fitting failed: {e}")
        # If curve fitting fails, just plot the data points
        plt.plot(sizes, omp_speedups, 'o-', linewidth=2, markersize=8, label='OpenMP Speedup', color=COLORS[1])
        plt.plot(sizes, cuda_speedups, 'o-', linewidth=2, markersize=8, label='CUDA Speedup', color=COLORS[2])
    
    plt.xlabel('Dataset Size (Number of Images)', fontsize=14)
    plt.ylabel('Speedup (x times)', fontsize=14)
    plt.title('Cancer Detection CNN: Speedup Trend Analysis', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'speedup_trend.png'), dpi=300)
    
    # Plot 6: Bar chart comparison for largest dataset
    if sizes:
        largest_size = max(sizes)
        largest_idx = sizes.index(largest_size)
        
        # Bar chart for execution times
        plt.figure(figsize=(10, 6))
        implementations = ['Serial', 'OpenMP', 'CUDA']
        times = [serial_times[largest_idx], omp_times[largest_idx], cuda_times[largest_idx]]
        
        bars = plt.bar(implementations, times, color=COLORS)
        
        # Add execution time labels on top of the bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}s', ha='center', va='bottom', fontsize=12)
        
        plt.xlabel('Implementation', fontsize=14)
        plt.ylabel('Execution Time (seconds)', fontsize=14)
        plt.title(f'Execution Time Comparison for Dataset Size {largest_size}', fontsize=16)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'execution_comparison_bar.png'), dpi=300)
        
        # Create a summary text file
        with open(os.path.join(figures_dir, 'benchmark_summary.txt'), 'w') as f:
            f.write("Cancer Detection CNN Benchmark Summary\n")
            f.write("====================================\n\n")
            f.write(f"Dataset sizes tested: {', '.join(map(str, sizes))}\n\n")
            
            f.write("Execution Time Comparison:\n")
            f.write(f"  - Serial: {serial_times[largest_idx]:.2f} seconds\n")
            f.write(f"  - OpenMP: {omp_times[largest_idx]:.2f} seconds\n")
            f.write(f"  - CUDA: {cuda_times[largest_idx]:.2f} seconds\n\n")
            
            f.write("Speedup Comparison:\n")
            f.write(f"  - OpenMP Speedup: {omp_speedups[largest_idx]:.2f}x\n")
            f.write(f"  - CUDA Speedup: {cuda_speedups[largest_idx]:.2f}x\n\n")
            
            f.write("Accuracy Comparison:\n")
            f.write(f"  - Serial Accuracy: {serial_accuracies[largest_idx]:.2f}%\n")
            f.write(f"  - OpenMP Accuracy: {omp_accuracies[largest_idx]:.2f}%\n")
            f.write(f"  - CUDA Accuracy: {cuda_accuracies[largest_idx]:.2f}%\n\n")
            
            f.write("Generated on: " + time.strftime("%Y-%m-%d %H:%M:%S"))
    
    print(f"All plots have been saved to {figures_dir}/")

def run_prediction(executable_path, model_type, image_folder, output_file, dataset_size=None, base_path=None):
    """
    Automates the prediction process using a trained model
    
    Args:
        executable_path: Path to the DSPC.exe executable
        model_type: Type of model to use (Serial, OpenMP, or CUDA)
        image_folder: Folder containing images to predict
        output_file: Output file for predictions
        dataset_size: Size of dataset used for training (optional)
        base_path: Base path to datasets (optional)
    
    Returns:
        bool: True if prediction was successful, False otherwise
    """
    if not os.path.exists(executable_path):
        print(f"Error: Executable not found at {executable_path}")
        return False
    
    if not os.path.exists(image_folder):
        print(f"Error: Image folder not found at {image_folder}")
        return False
    
    # First determine if we need to load a trained model
    # If dataset_size is provided, we'll first load the model trained on that dataset
    need_model_loading = dataset_size is not None and base_path is not None
    
    # Build the input sequence
    input_text = ""
    
    if need_model_loading:
        # Find the dataset index
        datasets = detect_existing_datasets(base_path)
        if not datasets:
            print(f"Error: No datasets found in {base_path}")
            return False
        
        try:
            dataset_index = datasets.index(dataset_size) + 1  # +1 because menu is 1-indexed
        except ValueError:
            print(f"Error: Dataset with size {dataset_size} not found in {datasets}")
            return False
        
        print(f"Found dataset with size {dataset_size} at index {dataset_index}")
        
        # Input for selecting dataset and training model
        input_text += f"0\n{dataset_index}\n"
        
        if model_type == "Serial":
            input_text += "1\n"
        elif model_type == "OpenMP":
            input_text += "2\n"
        elif model_type == "CUDA":
            input_text += "3\n"
    
    # Add prediction commands
    # 5: Predict Cancer from Uploaded Image
    # 2: Predict Batch of Images (Folder)
    # Then the image folder path and output CSV name
    # Then 3: Back to Main Menu
    # Then 6: Exit
    input_text += f"5\n2\n{image_folder}\n{output_file}\n3\n6\n"
    
    print(f"\nRunning prediction using {model_type} model on images in {image_folder}...")
    
    try:
        # Run the program and automate inputs
        process = subprocess.Popen(
            executable_path, 
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = process.communicate(input=input_text, timeout=1800)  # 30-minute timeout
        
        # Debug info
        print(f"Program output (first 500 characters):")
        print(stdout[:500] + "..." if len(stdout) > 500 else stdout)
        
        if stderr:
            print(f"Warning: Program reported errors: {stderr[:200]}...")
        
        # Check if the output file was created
        if os.path.exists(output_file):
            print(f"Predictions saved to {output_file}")
            return True
        else:
            print("Warning: Prediction file not created after program execution.")
            return False
    
    except subprocess.TimeoutExpired:
        process.kill()
        print(f"Error: Prediction timed out")
        return False
    except Exception as e:
        print(f"Error running prediction: {e}")
        return False
        
def visualize_predictions(prediction_file, dataset_path=None, output_dir='prediction_visualizations'):
    """
    Visualize cancer prediction results with optional ground truth from a dataset
    
    Args:
        prediction_file: Path to the prediction CSV file
        dataset_path: Optional path to a dataset folder containing sampled_labels.csv
        output_dir: Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set a colorblind-friendly palette
    COLORS = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC']  # Blue, Orange, Green, Pink
    benign_color = COLORS[2]    # Green
    malignant_color = COLORS[3] # Pink/Magenta
    
    # Load prediction data
    print(f"Loading prediction data from {prediction_file}...")
    try:
        df = pd.read_csv(prediction_file)
        print(f"Loaded {len(df)} predictions")
    except Exception as e:
        print(f"Error loading prediction file: {e}")
        return
    
    # Ensure we have the expected columns
    required_columns = ['Image', 'Benign_Probability', 'Malignant_Probability', 'Prediction']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        return
    
    # Add binary prediction for easier plotting (1 for Malignant, 0 for Benign)
    df['Prediction_Binary'] = (df['Prediction'] == 'Malignant').astype(int)
    
    # If dataset path is provided, try to load ground truth
    has_ground_truth = False
    if dataset_path:
        print(f"Attempting to load ground truth from {dataset_path}")
        filename_to_label = load_ground_truth_from_dataset(dataset_path)
        
        if filename_to_label:
            # Match predictions with ground truth
            df = match_predictions_with_ground_truth(df, filename_to_label)
            has_ground_truth = df['True_Label'].notna().sum() > 0
    
    # If no ground truth from dataset, try to extract from filenames
    if not has_ground_truth:
        print("Attempting to extract ground truth from filenames...")
        try:
            # First check if a True_Class or GroundTruth column already exists
            if 'True_Class' in df.columns:
                df['True_Label'] = df['True_Class'].apply(lambda x: 1 if isinstance(x, str) and x.lower() == 'malignant' else 0)
                has_ground_truth = True
            elif 'GroundTruth' in df.columns:
                df['True_Label'] = df['GroundTruth'].apply(lambda x: 1 if isinstance(x, str) and x.lower() == 'malignant' else 0)
                has_ground_truth = True
            else:
                # Try to determine if we have ground truth by looking at filenames
                contains_benign = df['Image'].str.lower().str.contains('benign').any()
                contains_malignant = df['Image'].str.lower().str.contains('malignant').any()
                
                has_ground_truth = contains_benign and contains_malignant
                
                if has_ground_truth:
                    df['True_Label'] = df['Image'].apply(lambda x: 1 if 'malignant' in x.lower() else 0)
                    print("Ground truth extracted from filenames")
                else:
                    # As a last resort, check if there's a clear pattern in filenames
                    contains_b_prefix = df['Image'].str.lower().str.contains(r'\bb_').any()
                    contains_m_prefix = df['Image'].str.lower().str.contains(r'\bm_').any()
                    
                    if contains_b_prefix and contains_m_prefix:
                        df['True_Label'] = df['Image'].apply(
                            lambda x: 1 if re.search(r'\bm_', x.lower()) else 0
                        )
                        has_ground_truth = True
                        print("Ground truth extracted from filename prefixes")
                    else:
                        print("Note: Could not detect reliable ground truth from filenames")
        except Exception as e:
            print(f"Note: Could not extract ground truth: {e}")
    
    # 1. Plot distribution of predictions (benign vs malignant)
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x='Prediction', data=df, palette=[benign_color, malignant_color])
    plt.title('Distribution of Cancer Predictions', fontsize=16)
    plt.xlabel('Prediction', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    
    # Add count and percentage labels to bars
    total = len(df)
    for i, p in enumerate(ax.patches):
        count = p.get_height()
        percentage = 100.0 * count / total
        ax.annotate(f"{int(count)}\n({percentage:.1f}%)", 
                   (p.get_x() + p.get_width()/2., count + 5),
                   ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_distribution.png'), dpi=300)
    
    # 2. Histogram of prediction probabilities
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.histplot(df['Benign_Probability'], bins=20, kde=True, color=benign_color)
    plt.title('Distribution of Benign Probabilities', fontsize=14)
    plt.xlabel('Benign Probability', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    plt.subplot(1, 2, 2)
    sns.histplot(df['Malignant_Probability'], bins=20, kde=True, color=malignant_color)
    plt.title('Distribution of Malignant Probabilities', fontsize=14)
    plt.xlabel('Malignant Probability', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'probability_distributions.png'), dpi=300)
    
    # 3. Scatter plot of benign vs malignant probabilities
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='Benign_Probability', 
        y='Malignant_Probability', 
        hue='Prediction',
        palette={'Benign': benign_color, 'Malignant': malignant_color},
        alpha=0.6,
        data=df
    )
    plt.plot([0, 1], [1, 0], 'k--', alpha=0.5)  # Diagonal line where probabilities sum to 1
    plt.title('Benign vs Malignant Probabilities', fontsize=16)
    plt.xlabel('Benign Probability', fontsize=14)
    plt.ylabel('Malignant Probability', fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend(title='Prediction')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'probability_scatter.png'), dpi=300)
    
    # 4. Confidence distribution - histogram of max probability
    df['Confidence'] = df[['Benign_Probability', 'Malignant_Probability']].max(axis=1)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=df, 
        x='Confidence', 
        hue='Prediction',
        palette={'Benign': benign_color, 'Malignant': malignant_color},
        bins=20,
        multiple='stack'
    )
    plt.title('Model Confidence Distribution', fontsize=16)
    plt.xlabel('Confidence (Max Probability)', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'), dpi=300)
    
    # 5. Boxplot of malignant probabilities
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        x='Prediction', 
        y='Malignant_Probability', 
        data=df,
        palette={'Benign': benign_color, 'Malignant': malignant_color}
    )
    plt.title('Malignant Probability by Prediction Category', fontsize=16)
    plt.xlabel('Prediction', fontsize=14)
    plt.ylabel('Malignant Probability', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'malignant_prob_boxplot.png'), dpi=300)
    
    # 6. Threshold analysis - prediction counts at different thresholds
    thresholds = np.linspace(0, 1, 21)  # 0.0, 0.05, 0.10, ..., 1.0
    malignant_counts = []
    benign_counts = []
    
    for threshold in thresholds:
        malignant_count = (df['Malignant_Probability'] > threshold).sum()
        benign_count = len(df) - malignant_count
        
        malignant_counts.append(malignant_count)
        benign_counts.append(benign_count)
    
    plt.figure(figsize=(12, 6))
    plt.plot(thresholds, malignant_counts, '-', linewidth=2, label='Malignant', color=malignant_color)
    plt.plot(thresholds, benign_counts, '-', linewidth=2, label='Benign', color=benign_color)
    plt.axvline(x=0.5, color='k', linestyle='--', alpha=0.5, label='Default Threshold (0.5)')
    
    plt.title('Prediction Counts at Different Thresholds', fontsize=16)
    plt.xlabel('Malignant Probability Threshold', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'threshold_analysis.png'), dpi=300)
    
    # If ground truth is available, generate more metrics
    if has_ground_truth:
        metrics_df = None
        
        try:
            # Ensure True_Label is numeric
            df['True_Label'] = pd.to_numeric(df['True_Label'], errors='coerce')
            
            # Remove rows with missing ground truth
            valid_df = df.dropna(subset=['True_Label']).copy()
            valid_df['True_Label'] = valid_df['True_Label'].astype(int)
            
            # Print accuracy
            accuracy = accuracy_score(valid_df['True_Label'], valid_df['Prediction_Binary']) * 100
            print(f"Model accuracy based on ground truth: {accuracy:.2f}%")
            
            # 7. Confusion Matrix
            cm = confusion_matrix(valid_df['True_Label'], valid_df['Prediction_Binary'])
            
            plt.figure(figsize=(10, 8))
            ax = plt.subplot()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Benign', 'Malignant'],
                        yticklabels=['Benign', 'Malignant'], ax=ax)
            plt.title('Confusion Matrix', fontsize=18)
            plt.xlabel('Predicted Label', fontsize=16)
            plt.ylabel('True Label', fontsize=16)
            
            # Add accuracy text
            plt.text(0.5, -0.15, f"Accuracy: {accuracy:.2f}%", 
                    horizontalalignment='center',
                    fontsize=14, transform=ax.transAxes)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
            
            # 8. Calculate and plot precision, recall, F1 score
            precision = precision_score(valid_df['True_Label'], valid_df['Prediction_Binary'], zero_division=0)
            recall = recall_score(valid_df['True_Label'], valid_df['Prediction_Binary'], zero_division=0)
            f1 = f1_score(valid_df['True_Label'], valid_df['Prediction_Binary'], zero_division=0)
            
            # Bar chart with metrics
            plt.figure(figsize=(12, 6))
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            values = [accuracy/100, precision, recall, f1]  # Convert accuracy to same scale
            
            bars = plt.bar(metrics, values, color=['#4285F4', '#EA4335', '#FBBC05', '#34A853'])
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=12)
            
            plt.ylim(0, 1.1)  # Limit y-axis for better visualization
            plt.title('Model Performance Metrics', fontsize=18)
            plt.ylabel('Score (0-1)', fontsize=14)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'performance_metrics.png'), dpi=300)
            
            # 9. ROC Curve
            fpr, tpr, _ = roc_curve(valid_df['True_Label'], valid_df['Malignant_Probability'])
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, 'b-', linewidth=3, label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.7, linewidth=1)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=18)
            plt.xlabel('False Positive Rate', fontsize=16)
            plt.ylabel('True Positive Rate', fontsize=16)
            plt.legend(loc="lower right", fontsize=14)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300)
            
            # Create a metrics dataframe
            metrics_data = {
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'],
                'Value': [accuracy, precision*100, recall*100, f1*100, roc_auc]
            }
            metrics_df = pd.DataFrame(metrics_data)
            metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
            
            # 10. Create a metrics dashboard (all metrics in one image)
            fig = plt.figure(figsize=(15, 10))
            
            # Confusion matrix
            ax1 = fig.add_subplot(221)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                      xticklabels=['Benign', 'Malignant'],
                      yticklabels=['Benign', 'Malignant'], ax=ax1)
            ax1.set_title('Confusion Matrix', fontsize=14)
            ax1.set_xlabel('Predicted Label', fontsize=12)
            ax1.set_ylabel('True Label', fontsize=12)
            
            # ROC curve
            ax2 = fig.add_subplot(222)
            ax2.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {roc_auc:.3f}')
            ax2.plot([0, 1], [0, 1], 'k--', alpha=0.7, linewidth=1)
            ax2.set_xlim([0.0, 1.0])
            ax2.set_ylim([0.0, 1.05])
            ax2.set_title('ROC Curve', fontsize=14)
            ax2.set_xlabel('False Positive Rate', fontsize=12)
            ax2.set_ylabel('True Positive Rate', fontsize=12)
            ax2.legend(loc="lower right")
            ax2.grid(alpha=0.3)
            
            # Performance metrics
            ax3 = fig.add_subplot(223)
            bars = ax3.bar(metrics, values, color=['#4285F4', '#EA4335', '#FBBC05', '#34A853'])
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=10)
            ax3.set_ylim(0, 1.1)
            ax3.set_title('Performance Metrics', fontsize=14)
            ax3.set_ylabel('Score (0-1)', fontsize=12)
            ax3.grid(axis='y', alpha=0.3)
            
            # Distribution of predictions
            ax4 = fig.add_subplot(224)
            true_pos = ((valid_df['Prediction_Binary'] == 1) & (valid_df['True_Label'] == 1)).sum()
            false_pos = ((valid_df['Prediction_Binary'] == 1) & (valid_df['True_Label'] == 0)).sum()
            true_neg = ((valid_df['Prediction_Binary'] == 0) & (valid_df['True_Label'] == 0)).sum()
            false_neg = ((valid_df['Prediction_Binary'] == 0) & (valid_df['True_Label'] == 1)).sum()
            
            labels = ['True Positive', 'False Positive', 'True Negative', 'False Negative']
            sizes = [true_pos, false_pos, true_neg, false_neg]
            colors = ['#4CAF50', '#FF5722', '#2196F3', '#FFC107']
            explode = (0.1, 0.1, 0.1, 0.1)  # explode all slices
            
            ax4.pie(sizes, explode=explode, labels=labels, colors=colors,
                   autopct='%1.1f%%', shadow=True, startangle=90)
            ax4.set_title('Prediction Distribution', fontsize=14)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'metrics_dashboard.png'), dpi=300)
            
            # Create detailed CSV with class-wise metrics
            from sklearn.metrics import classification_report
            report = classification_report(valid_df['True_Label'], valid_df['Prediction_Binary'], 
                                         target_names=['Benign', 'Malignant'], 
                                         output_dict=True)
            
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv(os.path.join(output_dir, 'detailed_metrics.csv'))
            
        except Exception as e:
            print(f"Error generating metrics visualizations: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Note: No ground truth available for calculating accuracy metrics")
    
    # Create summary text file
    with open(os.path.join(output_dir, 'prediction_summary.txt'), 'w') as f:
        f.write("Cancer Detection CNN Prediction Summary\n")
        f.write("====================================\n\n")
        f.write(f"Total images analyzed: {len(df)}\n")
        f.write(f"Benign predictions: {(df['Prediction'] == 'Benign').sum()} ({(df['Prediction'] == 'Benign').mean()*100:.1f}%)\n")
        f.write(f"Malignant predictions: {(df['Prediction'] == 'Malignant').sum()} ({(df['Prediction'] == 'Malignant').mean()*100:.1f}%)\n")
        f.write(f"Average confidence: {df['Confidence'].mean()*100:.1f}%\n\n")
        
        # Add accuracy metrics if ground truth is available
        if has_ground_truth:
            try:
                # Use valid_df from above calculations if it exists
                if 'valid_df' not in locals():
                    valid_df = df.dropna(subset=['True_Label']).copy()
                    valid_df['True_Label'] = valid_df['True_Label'].astype(int)
                
                # Calculate metrics
                accuracy = accuracy_score(valid_df['True_Label'], valid_df['Prediction_Binary']) * 100
                
                # Calculate class-wise metrics
                TP = ((valid_df['Prediction_Binary'] == 1) & (valid_df['True_Label'] == 1)).sum()
                FP = ((valid_df['Prediction_Binary'] == 1) & (valid_df['True_Label'] == 0)).sum()
                TN = ((valid_df['Prediction_Binary'] == 0) & (valid_df['True_Label'] == 0)).sum()
                FN = ((valid_df['Prediction_Binary'] == 0) & (valid_df['True_Label'] == 1)).sum()
                
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0
                specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                f.write(f"MODEL PERFORMANCE METRICS\n")
                f.write(f"-----------------------\n")
                f.write(f"Accuracy: {accuracy:.2f}%\n\n")
                
                f.write(f"Class-wise Metrics (for Malignant class):\n")
                f.write(f"  Precision: {precision*100:.2f}%\n")
                f.write(f"  Recall/Sensitivity: {recall*100:.2f}%\n")
                f.write(f"  Specificity: {specificity*100:.2f}%\n")
                f.write(f"  F1 Score: {f1*100:.2f}%\n\n")
                
                f.write(f"Confusion Matrix:\n")
                f.write(f"  True Negative (Benign correctly classified): {TN}\n")
                f.write(f"  False Positive (Benign incorrectly classified as Malignant): {FP}\n")
                f.write(f"  False Negative (Malignant incorrectly classified as Benign): {FN}\n")
                f.write(f"  True Positive (Malignant correctly classified): {TP}\n\n")
                
                # ROC AUC
                try:
                    fpr, tpr, _ = roc_curve(valid_df['True_Label'], valid_df['Malignant_Probability'])
                    roc_auc = auc(fpr, tpr)
                    f.write(f"ROC AUC: {roc_auc:.3f}\n\n")
                except:
                    pass
            except Exception as e:
                f.write(f"Error calculating metrics: {str(e)}\n\n")
        
        f.write("\nGenerated on: " + time.strftime("%Y-%m-%d %H:%M:%S"))
    
    print(f"Visualizations saved to {output_dir}/")
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"  Total images: {len(df)}")
    print(f"  Benign predictions: {(df['Prediction'] == 'Benign').sum()} ({(df['Prediction'] == 'Benign').mean()*100:.1f}%)")
    print(f"  Malignant predictions: {(df['Prediction'] == 'Malignant').sum()} ({(df['Prediction'] == 'Malignant').mean()*100:.1f}%)")
    print(f"  Average confidence: {df['Confidence'].mean()*100:.1f}%")
    
    if has_ground_truth and 'valid_df' in locals():
        try:
            accuracy = accuracy_score(valid_df['True_Label'], valid_df['Prediction_Binary']) * 100
            precision = precision_score(valid_df['True_Label'], valid_df['Prediction_Binary'], zero_division=0)
            recall = recall_score(valid_df['True_Label'], valid_df['Prediction_Binary'], zero_division=0)
            f1 = f1_score(valid_df['True_Label'], valid_df['Prediction_Binary'], zero_division=0)
            
            print(f"\nPerformance Metrics:")
            print(f"  Accuracy: {accuracy:.2f}%")
            print(f"  Precision: {precision*100:.2f}%")
            print(f"  Recall: {recall*100:.2f}%")
            print(f"  F1 Score: {f1*100:.2f}%")
            
            if 'roc_auc' in locals():
                print(f"  ROC AUC: {roc_auc:.3f}")
        except:
            pass
    
    return

def visualize_multiple_models(prediction_files, model_names=None, dataset_path=None, output_dir='model_comparison'):
    """
    Compare and visualize metrics from multiple model implementations with optional ground truth
    
    Args:
        prediction_files: List of paths to prediction CSV files
        model_names: List of model names corresponding to each file (optional)
        dataset_path: Optional path to a dataset folder containing sampled_labels.csv
        output_dir: Directory to save comparison visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set a colorblind-friendly palette
    COLORS = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#9F7C0F', '#55A1CA']
    
    # Dictionary to store dataframes for each model
    all_models_data = {}
    
    # Load ground truth if dataset path is provided
    filename_to_label = {}
    if dataset_path:
        print(f"Attempting to load ground truth from {dataset_path}")
        filename_to_label = load_ground_truth_from_dataset(dataset_path)
    
    # Process each prediction file
    for i, file_path in enumerate(prediction_files):
        try:
            # Load the CSV file
            df = pd.read_csv(file_path)
            
            # Use provided model name or extract from filename
            if model_names and i < len(model_names):
                model_name = model_names[i]
            else:
                # Extract model name from filename (assuming format like "predictions_CUDA.csv")
                base_name = os.path.basename(file_path)
                model_name = base_name.split('_')[1].split('.')[0] if '_' in base_name else f"Model {i+1}"
            
            # Ensure required columns exist
            if not {'Image', 'Benign_Probability', 'Malignant_Probability', 'Prediction'}.issubset(df.columns):
                print(f"Warning: File {file_path} is missing required columns. Skipping.")
                continue
                
            # Add model identifier column
            df['Model'] = model_name
            
            # Add binary prediction column if not exists
            if 'Prediction_Binary' not in df.columns:
                df['Prediction_Binary'] = (df['Prediction'] == 'Malignant').astype(int)
            
            # If we have ground truth from dataset, add it
            if filename_to_label:
                df = match_predictions_with_ground_truth(df, filename_to_label)
            
            # Store in dictionary
            all_models_data[model_name] = df
            
            print(f"Loaded {len(df)} predictions from {model_name}")
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    
    if not all_models_data:
        print("No valid prediction files loaded. Exiting.")
        return
    
    # Merge all data for combined analysis
    combined_data = pd.concat(all_models_data.values())
    model_names = list(all_models_data.keys())
    
    # Check if we have ground truth from dataset or try to extract it
    has_ground_truth = False
    
    # Option 1: Already have ground truth from dataset
    if filename_to_label and 'True_Label' in combined_data.columns and combined_data['True_Label'].notna().any():
        has_ground_truth = True
        print(f"Using ground truth from dataset for performance evaluation")
    else:
        # Option 2: Look for existing ground truth column
        for col in ['True_Class', 'GroundTruth', 'True_Label']:
            if col in combined_data.columns and combined_data[col].notna().any():
                has_ground_truth = True
                # Standardize the column name
                combined_data['True_Label'] = combined_data[col]
                if col != 'True_Label':
                    combined_data['True_Label'] = combined_data['True_Label'].apply(
                        lambda x: 1 if isinstance(x, str) and x.lower() == 'malignant' else 
                                (0 if isinstance(x, str) and x.lower() == 'benign' else x))
                break
        
        # Option 3: Try to extract from filenames
        if not has_ground_truth:
            contains_benign = combined_data['Image'].str.lower().str.contains('benign').any()
            contains_malignant = combined_data['Image'].str.lower().str.contains('malignant').any()
            
            if contains_benign and contains_malignant:
                combined_data['True_Label'] = combined_data['Image'].apply(
                    lambda x: 1 if 'malignant' in x.lower() else 0)
                has_ground_truth = True
                print("Ground truth extracted from filenames")
    
    # ------------------------------------------------------------
    # 1. Compare accuracy metrics across models if ground truth is available
    # ------------------------------------------------------------
    if has_ground_truth:
        try:
            # Calculate metrics for each model
            metrics = {'Model': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1': [], 'AUC': []}
            
            for model_name, df in all_models_data.items():
                # Match ground truth information if not already present
                if 'True_Label' not in df.columns or df['True_Label'].isna().all():
                    # Find matching images and copy ground truth
                    image_to_label = combined_data[['Image', 'True_Label']].drop_duplicates().set_index('Image')['True_Label'].to_dict()
                    df['True_Label'] = df['Image'].map(image_to_label)
                
                # Ensure True_Label is numeric and drop NaN values
                df['True_Label'] = pd.to_numeric(df['True_Label'], errors='coerce')
                valid_df = df.dropna(subset=['True_Label']).copy()
                valid_df['True_Label'] = valid_df['True_Label'].astype(int)
                
                if len(valid_df) == 0:
                    print(f"Warning: No valid ground truth for {model_name}. Skipping metrics.")
                    metrics['Model'].append(model_name)
                    metrics['Accuracy'].append(float('nan'))
                    metrics['Precision'].append(float('nan'))
                    metrics['Recall'].append(float('nan'))
                    metrics['F1'].append(float('nan'))
                    metrics['AUC'].append(float('nan'))
                    continue
                
                # Calculate metrics
                accuracy = accuracy_score(valid_df['True_Label'], valid_df['Prediction_Binary'])
                precision = precision_score(valid_df['True_Label'], valid_df['Prediction_Binary'], zero_division=0)
                recall = recall_score(valid_df['True_Label'], valid_df['Prediction_Binary'], zero_division=0)
                f1 = f1_score(valid_df['True_Label'], valid_df['Prediction_Binary'], zero_division=0)
                
                try:
                    auc_score = roc_auc_score(valid_df['True_Label'], valid_df['Malignant_Probability'])
                except Exception as e:
                    print(f"Error calculating AUC for {model_name}: {e}")
                    auc_score = float('nan')
                
                metrics['Model'].append(model_name)
                metrics['Accuracy'].append(accuracy * 100)
                metrics['Precision'].append(precision * 100)
                metrics['Recall'].append(recall * 100)
                metrics['F1'].append(f1 * 100)
                metrics['AUC'].append(auc_score)
            
            # Create metrics dataframe
            metrics_df = pd.DataFrame(metrics)
            metrics_df.to_csv(os.path.join(output_dir, 'model_metrics_comparison.csv'), index=False)
            
            # Plot comparison of metrics
            plt.figure(figsize=(14, 10))
            
            # Plot different metrics as subplots
            metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
            
            for i, metric in enumerate(metric_names):
                plt.subplot(2, 3, i+1)
                bars = plt.bar(metrics_df['Model'], metrics_df[metric], 
                       color=[COLORS[i % len(COLORS)] for i in range(len(metrics_df))])
                
                # Add data labels on bars
                for bar in bars:
                    height = bar.get_height()
                    if not np.isnan(height):
                        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                                f'{height:.1f}', ha='center', va='bottom', fontsize=9)
                
                title = f"{metric} Comparison"
                if metric != 'AUC':
                    title += " (%)"
                plt.title(title)
                plt.xticks(rotation=45, ha='right')
                if metric != 'AUC':
                    plt.ylim(0, 105)  # Add headroom for percentage labels
                plt.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300)
            
            # Create a radar/spider chart for comparing all metrics
            plt.figure(figsize=(10, 8))
            
            # Prepare data for radar chart
            models = metrics_df['Model'].tolist()
            metrics_for_radar = metrics_df[metric_names].values
            
            # If AUC is in scale 0-1, convert to 0-100 for consistent scaling
            if np.max(metrics_df['AUC'].dropna()) <= 1.0:
                metrics_for_radar[:, 4] *= 100
            
            # Number of metrics
            N = len(metric_names)
            
            # What will be the angle of each axis in the plot
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Initialize the spider plot
            ax = plt.subplot(111, polar=True)
            
            # Draw one axis per variable and add labels
            plt.xticks(angles[:-1], metric_names, size=12)
            
            # Draw ylabels
            ax.set_rlabel_position(0)
            plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], color="grey", size=10)
            plt.ylim(0, 100)
            
            # Plot each model
            for i, model in enumerate(models):
                values = metrics_for_radar[i].tolist()
                
                # Handle NaN values
                values = [0 if np.isnan(v) else v for v in values]
                
                values += values[:1]  # Close the loop
                
                # Plot values
                ax.plot(angles, values, linewidth=2, linestyle='solid', label=model, color=COLORS[i % len(COLORS)])
                ax.fill(angles, values, alpha=0.1, color=COLORS[i % len(COLORS)])
            
            # Add legend
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            plt.title("Model Performance Comparison", size=20, y=1.1)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'radar_comparison.png'), dpi=300)
            
            # Plot ROC curves for all models
            plt.figure(figsize=(10, 8))
            
            for i, model_name in enumerate(model_names):
                df = all_models_data[model_name]
                
                # Ensure we have ground truth and valid data
                if 'True_Label' not in df.columns or df['True_Label'].isna().all():
                    print(f"Skipping ROC curve for {model_name} - no ground truth")
                    continue
                
                # Prepare valid data
                df['True_Label'] = pd.to_numeric(df['True_Label'], errors='coerce')
                valid_df = df.dropna(subset=['True_Label']).copy()
                valid_df['True_Label'] = valid_df['True_Label'].astype(int)
                
                if len(valid_df) == 0:
                    continue
                
                try:
                    # Calculate ROC curve
                    fpr, tpr, _ = roc_curve(valid_df['True_Label'], valid_df['Malignant_Probability'])
                    roc_auc = auc(fpr, tpr)
                    
                    # Plot ROC curve
                    plt.plot(fpr, tpr, linewidth=2, 
                             label=f'{model_name} (AUC = {roc_auc:.3f})',
                             color=COLORS[i % len(COLORS)])
                except Exception as e:
                    print(f"Could not calculate ROC curve for {model_name}: {e}")
            
            # Plot diagonal line
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=14)
            plt.ylabel('True Positive Rate', fontsize=14)
            plt.title('ROC Curves Comparison', fontsize=16)
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'roc_comparison.png'), dpi=300)
            
            # Plot confusion matrices side by side
            plt.figure(figsize=(5 * len(model_names), 4))
            
            for i, model_name in enumerate(model_names):
                df = all_models_data[model_name]
                
                # Ensure we have ground truth and valid data
                if 'True_Label' not in df.columns or df['True_Label'].isna().all():
                    # Create an empty subplot with a message
                    plt.subplot(1, len(model_names), i+1)
                    plt.text(0.5, 0.5, "No ground truth available", 
                             ha='center', va='center', fontsize=12)
                    plt.title(f"{model_name}", fontsize=14)
                    plt.xticks([])
                    plt.yticks([])
                    continue
                
                # Prepare valid data
                df['True_Label'] = pd.to_numeric(df['True_Label'], errors='coerce')
                valid_df = df.dropna(subset=['True_Label']).copy()
                valid_df['True_Label'] = valid_df['True_Label'].astype(int)
                
                if len(valid_df) == 0:
                    # Create an empty subplot with a message
                    plt.subplot(1, len(model_names), i+1)
                    plt.text(0.5, 0.5, "No valid ground truth data", 
                             ha='center', va='center', fontsize=12)
                    plt.title(f"{model_name}", fontsize=14)
                    plt.xticks([])
                    plt.yticks([])
                    continue
                
                # Calculate confusion matrix
                cm = confusion_matrix(valid_df['True_Label'], valid_df['Prediction_Binary'])
                
                plt.subplot(1, len(model_names), i+1)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['Benign', 'Malignant'],
                            yticklabels=['Benign', 'Malignant'])
                plt.title(f"{model_name} Confusion Matrix", fontsize=14)
                plt.xlabel('Predicted Label', fontsize=12)
                if i == 0:
                    plt.ylabel('True Label', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=300)
            
        except Exception as e:
            print(f"Error generating accuracy comparison plots: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No ground truth available for accuracy metrics comparison")
    
    # ------------------------------------------------------------
    # 2. Compare prediction distributions across models
    # ------------------------------------------------------------
    try:
        # Prediction counts
        plt.figure(figsize=(12, 6))
        
        # Calculate counts for each model
        counts_data = {'Model': [], 'Benign': [], 'Malignant': []}
        
        for model_name, df in all_models_data.items():
            benign_count = (df['Prediction'] == 'Benign').sum()
            malignant_count = (df['Prediction'] == 'Malignant').sum()
            
            counts_data['Model'].append(model_name)
            counts_data['Benign'].append(benign_count)
            counts_data['Malignant'].append(malignant_count)
        
        # Convert to DataFrame for plotting
        counts_df = pd.DataFrame(counts_data)
        
        # Stacked bar chart
        ax = counts_df.plot(x='Model', y=['Benign', 'Malignant'], kind='bar', stacked=True, 
                    color=[COLORS[0], COLORS[1]], figsize=(12, 6))
        
        # Add data labels
        for container in ax.containers:
            ax.bar_label(container, label_type='center', fmt='%d')
        
        plt.title('Prediction Distribution Across Models', fontsize=16)
        plt.xlabel('Model', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.legend(title='Prediction')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, 'prediction_distribution_comparison.png'), dpi=300)
        
        # Compare prediction probabilities distributions
        plt.figure(figsize=(14, 6))
        
        # Malignant probability distributions
        plt.subplot(1, 2, 1)
        for i, model_name in enumerate(model_names):
            sns.kdeplot(all_models_data[model_name]['Malignant_Probability'], 
                       label=model_name, color=COLORS[i % len(COLORS)])
        
        plt.title('Malignant Probability Distribution', fontsize=14)
        plt.xlabel('Malignant Probability', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.grid(alpha=0.3)
        plt.legend()
        
        # Benign probability distributions
        plt.subplot(1, 2, 2)
        for i, model_name in enumerate(model_names):
            sns.kdeplot(all_models_data[model_name]['Benign_Probability'], 
                       label=model_name, color=COLORS[i % len(COLORS)])
        
        plt.title('Benign Probability Distribution', fontsize=14)
        plt.xlabel('Benign Probability', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.grid(alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'probability_distributions_comparison.png'), dpi=300)
        
        # Compare confidence distributions
        plt.figure(figsize=(12, 6))
        
        for i, model_name in enumerate(model_names):
            df = all_models_data[model_name]
            df['Confidence'] = df[['Benign_Probability', 'Malignant_Probability']].max(axis=1)
            
            sns.kdeplot(df['Confidence'], label=model_name, color=COLORS[i % len(COLORS)])
        
        plt.title('Model Confidence Comparison', fontsize=16)
        plt.xlabel('Confidence (Max Probability)', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.grid(alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confidence_comparison.png'), dpi=300)
        
    except Exception as e:
        print(f"Error generating prediction distribution comparison plots: {e}")
        import traceback
        traceback.print_exc()
    
    # ------------------------------------------------------------
    # 3. Create agreement analysis
    # ------------------------------------------------------------
    try:
        if len(model_names) >= 2:
            # Create a dataset with matched cases
            agreement_data = []
            
            # Get all unique image filenames
            all_images = set()
            for df in all_models_data.values():
                all_images.update(df['Image'].tolist())
            
            # For each image, get predictions from all models
            for image in all_images:
                row = {'Image': image}
                
                for model_name in model_names:
                    df = all_models_data[model_name]
                    if image in df['Image'].values:
                        img_data = df[df['Image'] == image].iloc[0]
                        row[f"{model_name}_Prediction"] = img_data['Prediction']
                        row[f"{model_name}_Malignant_Prob"] = img_data['Malignant_Probability']
                
                # Add ground truth if available
                if has_ground_truth and 'True_Label' in combined_data.columns:
                    label_data = combined_data[combined_data['Image'] == image]
                    if not label_data.empty and not label_data['True_Label'].isna().all():
                        row['True_Label'] = label_data['True_Label'].iloc[0]
                
                agreement_data.append(row)
            
            agreement_df = pd.DataFrame(agreement_data)
            
            # Calculate agreement statistics
            num_models = len(model_names)
            
            # Check if we have full data for all models
            complete_rows = 0
            for _, row in agreement_df.iterrows():
                if sum(f"{model}_Prediction" in row.index for model in model_names) == num_models:
                    complete_rows += 1
            
            if complete_rows > 0:
                # Calculate pairwise agreement
                agreement_matrix = np.zeros((num_models, num_models))
                
                for i, model1 in enumerate(model_names):
                    for j, model2 in enumerate(model_names):
                        if i == j:
                            agreement_matrix[i, j] = 1.0  # Self-agreement is 100%
                        else:
                            # Count cases where both models have predictions
                            valid_rows = agreement_df[
                                agreement_df[f"{model1}_Prediction"].notna() & 
                                agreement_df[f"{model2}_Prediction"].notna()
                            ]
                            
                            if len(valid_rows) > 0:
                                # Count agreements
                                agreements = (valid_rows[f"{model1}_Prediction"] == 
                                             valid_rows[f"{model2}_Prediction"]).sum()
                                
                                agreement_matrix[i, j] = agreements / len(valid_rows)
                
                # Plot agreement heatmap
                plt.figure(figsize=(10, 8))
                sns.heatmap(agreement_matrix, annot=True, fmt='.2f', cmap='Blues',
                            xticklabels=model_names, yticklabels=model_names, vmin=0, vmax=1)
                plt.title('Pairwise Agreement Between Models', fontsize=16)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'model_agreement.png'), dpi=300)
                
                # Create a consensus analysis
                if num_models >= 3:
                    # Filter for rows where all models have predictions
                    complete_df = agreement_df.copy()
                    for model in model_names:
                        complete_df = complete_df[complete_df[f"{model}_Prediction"].notna()]
                    
                    if len(complete_df) > 0:
                        # Count how many models agree on each case (unanimous, majority, split)
                        agreement_counts = {'Unanimous': 0, 'Majority': 0, 'Split': 0}
                        
                        for _, row in complete_df.iterrows():
                            predictions = [row[f"{model}_Prediction"] for model in model_names]
                            
                            # Count occurrences of the most common prediction
                            from collections import Counter
                            most_common = Counter(predictions).most_common(1)[0][1]
                            
                            if most_common == num_models:
                                agreement_counts['Unanimous'] += 1
                            elif most_common > num_models / 2:
                                agreement_counts['Majority'] += 1
                            else:
                                agreement_counts['Split'] += 1
                        
                        # Plot agreement summary
                        plt.figure(figsize=(10, 6))
                        categories = list(agreement_counts.keys())
                        values = list(agreement_counts.values())
                        
                        bars = plt.bar(categories, values, color=[COLORS[i] for i in range(len(categories))])
                        
                        # Add percentages
                        total = sum(values)
                        for bar in bars:
                            height = bar.get_height()
                            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                                    f'{height} ({height/total*100:.1f}%)', ha='center', va='bottom', fontsize=12)
                        
                        plt.title('Consensus Analysis Across All Models', fontsize=16)
                        plt.ylabel('Number of Cases', fontsize=14)
                        plt.grid(axis='y', alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_dir, 'consensus_analysis.png'), dpi=300)
                        
                        # Detailed analysis of disagreement cases
                        if agreement_counts['Split'] > 0 or agreement_counts['Majority'] > 0:
                            # Add a column that identifies consensus type
                            complete_df['Consensus'] = complete_df.apply(
                                lambda row: _get_consensus_type(row, model_names), axis=1)
                            
                            # Save disagreement cases to CSV for further analysis
                            disagreement_df = complete_df[complete_df['Consensus'] != 'Unanimous']
                            if len(disagreement_df) > 0:
                                disagreement_df.to_csv(os.path.join(output_dir, 'disagreement_cases.csv'), index=False)
                                print(f"Saved {len(disagreement_df)} disagreement cases to disagreement_cases.csv")
    
    except Exception as e:
        print(f"Error generating agreement analysis: {e}")
        import traceback
        traceback.print_exc()
    
    # ------------------------------------------------------------
    # 4. Create a summary report
    # ------------------------------------------------------------
    try:
        with open(os.path.join(output_dir, 'comparison_summary.txt'), 'w') as f:
            f.write("Cancer Detection CNN Models Comparison\n")
            f.write("=====================================\n\n")
            
            f.write(f"Models compared: {', '.join(model_names)}\n")
            f.write(f"Total unique images: {len(agreement_df) if 'agreement_df' in locals() else 'N/A'}\n\n")
            
            if 'metrics_df' in locals():
                f.write("Performance Metrics Comparison:\n")
                f.write("------------------------------\n")
                f.write(metrics_df.to_string(index=False))
                f.write("\n\n")
                
                # Identify best model for each metric
                f.write("Best Model for Each Metric:\n")
                for metric in ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']:
                    # Handle NaN values
                    valid_metrics = metrics_df[~metrics_df[metric].isna()]
                    if len(valid_metrics) > 0:
                        best_idx = valid_metrics[metric].idxmax()
                        best_model = valid_metrics.loc[best_idx, 'Model']
                        best_value = valid_metrics.loc[best_idx, metric]
                        
                        if metric != 'AUC':
                            f.write(f"  - {metric}: {best_model} ({best_value:.2f}%)\n")
                        else:
                            f.write(f"  - {metric}: {best_model} ({best_value:.3f})\n")
                    else:
                        f.write(f"  - {metric}: No valid data\n")
                        
                f.write("\n")
            
            if 'agreement_counts' in locals():
                f.write("Consensus Analysis:\n")
                f.write("-----------------\n")
                total = sum(agreement_counts.values())
                for category, count in agreement_counts.items():
                    f.write(f"  - {category}: {count} cases ({count/total*100:.1f}%)\n")
                f.write("\n")
            
            if 'disagreement_df' in locals():
                f.write(f"Disagreement Analysis:\n")
                f.write("--------------------\n")
                f.write(f"Number of cases with disagreement: {len(disagreement_df)}\n")
                f.write("Disagreement cases saved to disagreement_cases.csv for detailed review\n\n")
            
            f.write("Prediction Distribution:\n")
            f.write("----------------------\n")
            for model_name, df in all_models_data.items():
                benign_count = (df['Prediction'] == 'Benign').sum()
                malignant_count = (df['Prediction'] == 'Malignant').sum()
                total = len(df)
                
                f.write(f"  - {model_name}:\n")
                f.write(f"    * Benign: {benign_count} ({benign_count/total*100:.1f}%)\n")
                f.write(f"    * Malignant: {malignant_count} ({malignant_count/total*100:.1f}%)\n")
            
            f.write("\nVisualizations:\n")
            f.write("--------------\n")
            f.write("The following visualizations have been generated:\n")
            
            if has_ground_truth:
                f.write("  - metrics_comparison.png: Bar charts comparing key metrics\n")
                f.write("  - radar_comparison.png: Radar chart of all metrics\n")
                f.write("  - roc_comparison.png: ROC curves for all models\n")
                f.write("  - confusion_matrices.png: Confusion matrices for all models\n")
            
            f.write("  - prediction_distribution_comparison.png: Distribution of predictions\n")
            f.write("  - probability_distributions_comparison.png: Probability distributions\n")
            f.write("  - confidence_comparison.png: Confidence level distributions\n")
            f.write("  - model_agreement.png: Pairwise agreement between models\n")
            if num_models >= 3 and 'agreement_counts' in locals():
                f.write("  - consensus_analysis.png: Consensus analysis across models\n")
            
            f.write("\nGenerated on: " + time.strftime("%Y-%m-%d %H:%M:%S"))
    
    except Exception as e:
        print(f"Error generating summary report: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"Model comparison analysis completed. Results saved to {output_dir}/")
    """
    Compare and visualize metrics from multiple model implementations
    
    Args:
        prediction_files: List of paths to prediction CSV files
        model_names: List of model names corresponding to each file (optional)
        output_dir: Directory to save comparison visualizations
    """
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set a colorblind-friendly palette
    COLORS = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#9F7C0F', '#55A1CA']
    
    # Dictionary to store dataframes for each model
    all_models_data = {}
    
    # Process each prediction file
    for i, file_path in enumerate(prediction_files):
        try:
            # Load the CSV file
            df = pd.read_csv(file_path)
            
            # Use provided model name or extract from filename
            if model_names and i < len(model_names):
                model_name = model_names[i]
            else:
                # Extract model name from filename (assuming format like "predictions_CUDA.csv")
                base_name = os.path.basename(file_path)
                model_name = base_name.split('_')[1].split('.')[0] if '_' in base_name else f"Model {i+1}"
            
            # Ensure required columns exist
            if not {'Image', 'Benign_Probability', 'Malignant_Probability', 'Prediction'}.issubset(df.columns):
                print(f"Warning: File {file_path} is missing required columns. Skipping.")
                continue
                
            # Add model identifier column
            df['Model'] = model_name
            
            # Add binary prediction column if not exists
            if 'Prediction_Binary' not in df.columns:
                df['Prediction_Binary'] = (df['Prediction'] == 'Malignant').astype(int)
            
            # Store in dictionary
            all_models_data[model_name] = df
            
            print(f"Loaded {len(df)} predictions from {model_name}")
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    
    if not all_models_data:
        print("No valid prediction files loaded. Exiting.")
        return
    
    # Merge all data for combined analysis
    # We'll use the image filenames as keys to match predictions across models
    combined_data = pd.concat(all_models_data.values())
    model_names = list(all_models_data.keys())
    
    # ------------------------------------------------------------
    # 1. Compare accuracy metrics across models
    # ------------------------------------------------------------
    try:
        # Check if ground truth can be extracted
        has_ground_truth = False
        
        # Look for existing ground truth column
        if any(col in combined_data.columns for col in ['True_Class', 'GroundTruth', 'True_Label']):
            has_ground_truth = True
            # Standardize the column name
            for col in ['True_Class', 'GroundTruth', 'True_Label']:
                if col in combined_data.columns:
                    combined_data['True_Label'] = combined_data[col]
                    if col != 'True_Label':
                        combined_data['True_Label'] = combined_data['True_Label'].apply(
                            lambda x: 1 if isinstance(x, str) and x.lower() == 'malignant' else 
                                     (0 if isinstance(x, str) and x.lower() == 'benign' else x))
                    break
        else:
            # Try to extract from filenames
            contains_benign = combined_data['Image'].str.lower().str.contains('benign').any()
            contains_malignant = combined_data['Image'].str.lower().str.contains('malignant').any()
            
            if contains_benign and contains_malignant:
                combined_data['True_Label'] = combined_data['Image'].apply(
                    lambda x: 1 if 'malignant' in x.lower() else 0)
                has_ground_truth = True
                print("Ground truth extracted from filenames")
                
        if has_ground_truth:
            # Calculate metrics for each model
            metrics = {'Model': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1': [], 'AUC': []}
            
            for model_name, df in all_models_data.items():
                # Match ground truth information
                if 'True_Label' not in df.columns:
                    # Find matching images and copy ground truth
                    image_to_label = combined_data[['Image', 'True_Label']].drop_duplicates().set_index('Image')['True_Label'].to_dict()
                    df['True_Label'] = df['Image'].map(image_to_label)
                
                # Calculate metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
                
                accuracy = accuracy_score(df['True_Label'], df['Prediction_Binary'])
                precision = precision_score(df['True_Label'], df['Prediction_Binary'], zero_division=0)
                recall = recall_score(df['True_Label'], df['Prediction_Binary'], zero_division=0)
                f1 = f1_score(df['True_Label'], df['Prediction_Binary'], zero_division=0)
                
                try:
                    auc_score = roc_auc_score(df['True_Label'], df['Malignant_Probability'])
                except:
                    auc_score = 0
                
                metrics['Model'].append(model_name)
                metrics['Accuracy'].append(accuracy * 100)
                metrics['Precision'].append(precision * 100)
                metrics['Recall'].append(recall * 100)
                metrics['F1'].append(f1 * 100)
                metrics['AUC'].append(auc_score)
            
            # Create metrics dataframe
            metrics_df = pd.DataFrame(metrics)
            metrics_df.to_csv(os.path.join(output_dir, 'model_metrics_comparison.csv'), index=False)
            
            # Plot comparison of metrics
            plt.figure(figsize=(14, 10))
            
            # Plot different metrics as subplots
            metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
            
            for i, metric in enumerate(metric_names):
                plt.subplot(2, 3, i+1)
                bars = plt.bar(metrics_df['Model'], metrics_df[metric], 
                       color=[COLORS[i % len(COLORS)] for i in range(len(metrics_df))])
                
                # Add data labels on bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                            f'{height:.1f}', ha='center', va='bottom', fontsize=9)
                
                title = f"{metric} Comparison"
                if metric != 'AUC':
                    title += " (%)"
                plt.title(title)
                plt.xticks(rotation=45, ha='right')
                if metric != 'AUC':
                    plt.ylim(0, 105)  # Add headroom for percentage labels
                plt.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300)
            
            # Create a radar/spider chart for comparing all metrics
            plt.figure(figsize=(10, 8))
            
            # Prepare data for radar chart
            models = metrics_df['Model'].tolist()
            metrics_for_radar = metrics_df[metric_names].values
            
            # If AUC is in scale 0-1, convert to 0-100 for consistent scaling
            if np.max(metrics_df['AUC']) <= 1.0:
                metrics_for_radar[:, 4] *= 100
            
            # Number of metrics
            N = len(metric_names)
            
            # What will be the angle of each axis in the plot
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Initialize the spider plot
            ax = plt.subplot(111, polar=True)
            
            # Draw one axis per variable and add labels
            plt.xticks(angles[:-1], metric_names, size=12)
            
            # Draw ylabels
            ax.set_rlabel_position(0)
            plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], color="grey", size=10)
            plt.ylim(0, 100)
            
            # Plot each model
            for i, model in enumerate(models):
                values = metrics_for_radar[i].tolist()
                values += values[:1]  # Close the loop
                
                # Plot values
                ax.plot(angles, values, linewidth=2, linestyle='solid', label=model, color=COLORS[i % len(COLORS)])
                ax.fill(angles, values, alpha=0.1, color=COLORS[i % len(COLORS)])
            
            # Add legend
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            plt.title("Model Performance Comparison", size=20, y=1.1)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'radar_comparison.png'), dpi=300)
            
            # Plot ROC curves for all models
            plt.figure(figsize=(10, 8))
            
            for i, model_name in enumerate(model_names):
                df = all_models_data[model_name]
                
                try:
                    # Calculate ROC curve
                    fpr, tpr, _ = roc_curve(df['True_Label'], df['Malignant_Probability'])
                    roc_auc = auc(fpr, tpr)
                    
                    # Plot ROC curve
                    plt.plot(fpr, tpr, linewidth=2, 
                             label=f'{model_name} (AUC = {roc_auc:.3f})',
                             color=COLORS[i % len(COLORS)])
                except:
                    print(f"Could not calculate ROC curve for {model_name}")
            
            # Plot diagonal line
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=14)
            plt.ylabel('True Positive Rate', fontsize=14)
            plt.title('ROC Curves Comparison', fontsize=16)
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'roc_comparison.png'), dpi=300)
            
            # Plot confusion matrices side by side
            plt.figure(figsize=(5 * len(model_names), 4))
            
            for i, model_name in enumerate(model_names):
                df = all_models_data[model_name]
                
                # Calculate confusion matrix
                cm = confusion_matrix(df['True_Label'], df['Prediction_Binary'])
                
                plt.subplot(1, len(model_names), i+1)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['Benign', 'Malignant'],
                            yticklabels=['Benign', 'Malignant'])
                plt.title(f"{model_name} Confusion Matrix", fontsize=14)
                plt.xlabel('Predicted Label', fontsize=12)
                if i == 0:
                    plt.ylabel('True Label', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=300)
            
        else:
            print("No ground truth available for accuracy metrics comparison")
    
    except Exception as e:
        print(f"Error generating accuracy comparison plots: {e}")
    
    # ------------------------------------------------------------
    # 2. Compare prediction distributions across models
    # ------------------------------------------------------------
    try:
        # Prediction counts
        plt.figure(figsize=(12, 6))
        
        # Calculate counts for each model
        counts_data = {'Model': [], 'Benign': [], 'Malignant': []}
        
        for model_name, df in all_models_data.items():
            benign_count = (df['Prediction'] == 'Benign').sum()
            malignant_count = (df['Prediction'] == 'Malignant').sum()
            
            counts_data['Model'].append(model_name)
            counts_data['Benign'].append(benign_count)
            counts_data['Malignant'].append(malignant_count)
        
        # Convert to DataFrame for plotting
        counts_df = pd.DataFrame(counts_data)
        
        # Stacked bar chart
        ax = counts_df.plot(x='Model', y=['Benign', 'Malignant'], kind='bar', stacked=True, 
                    color=[COLORS[0], COLORS[1]], figsize=(12, 6))
        
        # Add data labels
        for container in ax.containers:
            ax.bar_label(container, label_type='center', fmt='%d')
        
        plt.title('Prediction Distribution Across Models', fontsize=16)
        plt.xlabel('Model', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.legend(title='Prediction')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, 'prediction_distribution_comparison.png'), dpi=300)
        
        # Compare prediction probabilities distributions
        plt.figure(figsize=(14, 6))
        
        # Malignant probability distributions
        plt.subplot(1, 2, 1)
        for i, model_name in enumerate(model_names):
            sns.kdeplot(all_models_data[model_name]['Malignant_Probability'], 
                       label=model_name, color=COLORS[i % len(COLORS)])
        
        plt.title('Malignant Probability Distribution', fontsize=14)
        plt.xlabel('Malignant Probability', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.grid(alpha=0.3)
        plt.legend()
        
        # Benign probability distributions
        plt.subplot(1, 2, 2)
        for i, model_name in enumerate(model_names):
            sns.kdeplot(all_models_data[model_name]['Benign_Probability'], 
                       label=model_name, color=COLORS[i % len(COLORS)])
        
        plt.title('Benign Probability Distribution', fontsize=14)
        plt.xlabel('Benign Probability', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.grid(alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'probability_distributions_comparison.png'), dpi=300)
        
        # Compare confidence distributions
        plt.figure(figsize=(12, 6))
        
        for i, model_name in enumerate(model_names):
            df = all_models_data[model_name]
            df['Confidence'] = df[['Benign_Probability', 'Malignant_Probability']].max(axis=1)
            
            sns.kdeplot(df['Confidence'], label=model_name, color=COLORS[i % len(COLORS)])
        
        plt.title('Model Confidence Comparison', fontsize=16)
        plt.xlabel('Confidence (Max Probability)', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.grid(alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confidence_comparison.png'), dpi=300)
        
    except Exception as e:
        print(f"Error generating prediction distribution comparison plots: {e}")
    
    # ------------------------------------------------------------
    # 3. Create agreement analysis
    # ------------------------------------------------------------
    try:
        if len(model_names) >= 2:
            # Create a dataset with matched cases
            agreement_data = []
            
            # Get all unique image filenames
            all_images = set()
            for df in all_models_data.values():
                all_images.update(df['Image'].tolist())
            
            # For each image, get predictions from all models
            for image in all_images:
                row = {'Image': image}
                
                for model_name in model_names:
                    df = all_models_data[model_name]
                    if image in df['Image'].values:
                        img_data = df[df['Image'] == image].iloc[0]
                        row[f"{model_name}_Prediction"] = img_data['Prediction']
                        row[f"{model_name}_Malignant_Prob"] = img_data['Malignant_Probability']
                
                agreement_data.append(row)
            
            agreement_df = pd.DataFrame(agreement_data)
            
            # Calculate agreement statistics
            num_models = len(model_names)
            
            # Check if we have full data for all models
            complete_rows = 0
            for _, row in agreement_df.iterrows():
                if sum(f"{model}_Prediction" in row.index for model in model_names) == num_models:
                    complete_rows += 1
            
            if complete_rows > 0:
                # Calculate pairwise agreement
                agreement_matrix = np.zeros((num_models, num_models))
                
                for i, model1 in enumerate(model_names):
                    for j, model2 in enumerate(model_names):
                        if i == j:
                            agreement_matrix[i, j] = 1.0  # Self-agreement is 100%
                        else:
                            # Count cases where both models have predictions
                            valid_rows = agreement_df[
                                agreement_df[f"{model1}_Prediction"].notna() & 
                                agreement_df[f"{model2}_Prediction"].notna()
                            ]
                            
                            if len(valid_rows) > 0:
                                # Count agreements
                                agreements = (valid_rows[f"{model1}_Prediction"] == 
                                             valid_rows[f"{model2}_Prediction"]).sum()
                                
                                agreement_matrix[i, j] = agreements / len(valid_rows)
                
                # Plot agreement heatmap
                plt.figure(figsize=(10, 8))
                sns.heatmap(agreement_matrix, annot=True, fmt='.2f', cmap='Blues',
                            xticklabels=model_names, yticklabels=model_names, vmin=0, vmax=1)
                plt.title('Pairwise Agreement Between Models', fontsize=16)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'model_agreement.png'), dpi=300)
                
                # Create a consensus analysis
                if num_models >= 3:
                    # Filter for rows where all models have predictions
                    complete_df = agreement_df.copy()
                    for model in model_names:
                        complete_df = complete_df[complete_df[f"{model}_Prediction"].notna()]
                    
                    if len(complete_df) > 0:
                        # Count how many models agree on each case (unanimous, majority, split)
                        agreement_counts = {'Unanimous': 0, 'Majority': 0, 'Split': 0}
                        
                        for _, row in complete_df.iterrows():
                            predictions = [row[f"{model}_Prediction"] for model in model_names]
                            
                            # Count occurrences of the most common prediction
                            from collections import Counter
                            most_common = Counter(predictions).most_common(1)[0][1]
                            
                            if most_common == num_models:
                                agreement_counts['Unanimous'] += 1
                            elif most_common > num_models / 2:
                                agreement_counts['Majority'] += 1
                            else:
                                agreement_counts['Split'] += 1
                        
                        # Plot agreement summary
                        plt.figure(figsize=(10, 6))
                        categories = list(agreement_counts.keys())
                        values = list(agreement_counts.values())
                        
                        bars = plt.bar(categories, values, color=[COLORS[i] for i in range(len(categories))])
                        
                        # Add percentages
                        total = sum(values)
                        for bar in bars:
                            height = bar.get_height()
                            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                                    f'{height} ({height/total*100:.1f}%)', ha='center', va='bottom', fontsize=12)
                        
                        plt.title('Consensus Analysis Across All Models', fontsize=16)
                        plt.ylabel('Number of Cases', fontsize=14)
                        plt.grid(axis='y', alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_dir, 'consensus_analysis.png'), dpi=300)
                        
                        # Detailed analysis of disagreement cases
                        if agreement_counts['Split'] > 0 or agreement_counts['Majority'] > 0:
                            # Add a column that identifies consensus type
                            complete_df['Consensus'] = complete_df.apply(
                                lambda row: _get_consensus_type(row, model_names), axis=1)
                            
                            # Save disagreement cases to CSV for further analysis
                            disagreement_df = complete_df[complete_df['Consensus'] != 'Unanimous']
                            if len(disagreement_df) > 0:
                                disagreement_df.to_csv(os.path.join(output_dir, 'disagreement_cases.csv'), index=False)
                                print(f"Saved {len(disagreement_df)} disagreement cases to disagreement_cases.csv")
    
    except Exception as e:
        print(f"Error generating agreement analysis: {e}")
    
    # ------------------------------------------------------------
    # 4. Create a summary report
    # ------------------------------------------------------------
    try:
        with open(os.path.join(output_dir, 'comparison_summary.txt'), 'w') as f:
            f.write("Cancer Detection CNN Models Comparison\n")
            f.write("=====================================\n\n")
            
            f.write(f"Models compared: {', '.join(model_names)}\n")
            f.write(f"Total unique images: {len(agreement_df) if 'agreement_df' in locals() else 'N/A'}\n\n")
            
            if 'metrics_df' in locals():
                f.write("Performance Metrics Comparison:\n")
                f.write("------------------------------\n")
                f.write(metrics_df.to_string(index=False))
                f.write("\n\n")
                
                # Identify best model for each metric
                f.write("Best Model for Each Metric:\n")
                for metric in ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']:
                    best_idx = metrics_df[metric].idxmax()
                    best_model = metrics_df.loc[best_idx, 'Model']
                    best_value = metrics_df.loc[best_idx, metric]
                    
                    if metric != 'AUC':
                        f.write(f"  - {metric}: {best_model} ({best_value:.2f}%)\n")
                    else:
                        f.write(f"  - {metric}: {best_model} ({best_value:.3f})\n")
                        
                f.write("\n")
            
            if 'agreement_counts' in locals():
                f.write("Consensus Analysis:\n")
                f.write("-----------------\n")
                total = sum(agreement_counts.values())
                for category, count in agreement_counts.items():
                    f.write(f"  - {category}: {count} cases ({count/total*100:.1f}%)\n")
                f.write("\n")
            
            if 'disagreement_df' in locals():
                f.write(f"Disagreement Analysis:\n")
                f.write("--------------------\n")
                f.write(f"Number of cases with disagreement: {len(disagreement_df)}\n")
                f.write("Disagreement cases saved to disagreement_cases.csv for detailed review\n\n")
            
            f.write("Prediction Distribution:\n")
            f.write("----------------------\n")
            for model_name, df in all_models_data.items():
                benign_count = (df['Prediction'] == 'Benign').sum()
                malignant_count = (df['Prediction'] == 'Malignant').sum()
                total = len(df)
                
                f.write(f"  - {model_name}:\n")
                f.write(f"    * Benign: {benign_count} ({benign_count/total*100:.1f}%)\n")
                f.write(f"    * Malignant: {malignant_count} ({malignant_count/total*100:.1f}%)\n")
            
            f.write("\nVisualizations:\n")
            f.write("--------------\n")
            f.write("The following visualizations have been generated:\n")
            f.write("  - metrics_comparison.png: Bar charts comparing key metrics\n")
            f.write("  - radar_comparison.png: Radar chart of all metrics\n")
            f.write("  - roc_comparison.png: ROC curves for all models\n")
            f.write("  - confusion_matrices.png: Confusion matrices for all models\n")
            f.write("  - prediction_distribution_comparison.png: Distribution of predictions\n")
            f.write("  - probability_distributions_comparison.png: Probability distributions\n")
            f.write("  - confidence_comparison.png: Confidence level distributions\n")
            f.write("  - model_agreement.png: Pairwise agreement between models\n")
            if num_models >= 3:
                f.write("  - consensus_analysis.png: Consensus analysis across models\n")
            
            f.write("\nGenerated on: " + time.strftime("%Y-%m-%d %H:%M:%S"))
    
    except Exception as e:
        print(f"Error generating summary report: {e}")
    
    print(f"Model comparison analysis completed. Results saved to {output_dir}/")

# Helper function to determine consensus type
def _get_consensus_type(row, model_names):
    predictions = [row[f"{model}_Prediction"] for model in model_names 
                  if f"{model}_Prediction" in row and pd.notna(row[f"{model}_Prediction"])]
    
    if not predictions:
        return "Unknown"
    
    from collections import Counter
    most_common = Counter(predictions).most_common(1)[0][1]
    
    if most_common == len(predictions) and len(predictions) == len(model_names):
        return 'Unanimous'
    elif most_common > len(predictions) / 2:
        return 'Majority'
    else:
        return 'Split'
        
def monitor_csv_file(directory='.', interval=1.0):
    """
    Monitors the current directory for newly created benchmark.csv files and generates plots
    Can be used to automatically generate plots when the user selects option 4 in DSPC.exe
    """
    print("\nMonitoring for benchmark.csv files... (Press Ctrl+C to stop)")
    
    while True:
        try:
            # Check if benchmark.csv exists
            if os.path.exists(os.path.join(directory, 'benchmark.csv')):
                print("\nFound benchmark.csv file! Generating plots...")
                
                # Read the CSV file
                try:
                    df = pd.read_csv(os.path.join(directory, 'benchmark.csv'))
                    
                    # Create a timestamp-based directory for the plots
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    output_dir = os.path.join(RESULTS_FOLDER, f"plots_{timestamp}")
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Generate plots
                    generate_quick_plots(df, output_dir)
                    
                    # Make a copy of the CSV with timestamp
                    output_csv = os.path.join(RESULTS_FOLDER, f"benchmark_{timestamp}.csv")
                    shutil.copy(os.path.join(directory, 'benchmark.csv'), output_csv)
                    
                    print(f"Plots generated and saved to {output_dir}/")
                    print(f"CSV file copied to {output_csv}")
                    
                    # Remove the original benchmark.csv file to avoid processing it again
                    os.remove(os.path.join(directory, 'benchmark.csv'))
                    
                except Exception as e:
                    print(f"Error processing benchmark.csv: {e}")
            
            # Sleep for the specified interval
            time.sleep(interval)
            
        except KeyboardInterrupt:
            print("\nStopping file monitor.")
            break
        except Exception as e:
            print(f"Error in monitor: {e}")
            time.sleep(interval)

def main():
    parser = argparse.ArgumentParser(description='Benchmark CNN implementations and visualize cancer detection results')
    
    # Main operation modes
    parser.add_argument('--run', action='store_true', help='Run benchmarks on available datasets')
    parser.add_argument('--interactive', action='store_true', help='Run DSPC.exe in interactive mode')
    parser.add_argument('--plot', action='store_true', help='Plot results from benchmark files')
    parser.add_argument('--visualize', type=str, metavar='CSV_FILE', 
                        help='Visualize prediction results from a CSV file')
    parser.add_argument('--monitor', action='store_true', 
                        help='Monitor for benchmark.csv files and generate plots')
    parser.add_argument('--all', action='store_true', 
                        help='Perform all steps: generate, run, and plot')
    
    # New prediction arguments
    parser.add_argument('--predict', action='store_true', 
                        help='Generate predictions using the trained model')
    parser.add_argument('--model-type', type=str, choices=['Serial', 'OpenMP', 'CUDA'], default='CUDA', 
                        help='Type of model to use for prediction')
    parser.add_argument('--image-folder', type=str, 
                        help='Folder containing images to predict')
    parser.add_argument('--pred-output', type=str, default='predictions.csv', 
                        help='Output file for predictions')
    parser.add_argument('--dataset-size', type=int, 
                        help='Size of dataset to use for model training (needed if running predictions on a newly trained model)')

    # New argument for comparing multiple models
    parser.add_argument('--compare-models', type=str, 
                        help='Comma-separated list of prediction CSV files for comparing models')
    parser.add_argument('--model-names', type=str, 
                        help='Comma-separated list of model names (optional, default: extracted from filenames)')
    
    # Dataset integration for ground truth
    parser.add_argument('--dataset-path', type=str,
                        help='Path to dataset folder containing sampled_labels.csv for ground truth')
    
    # Additional options
    parser.add_argument('--base-path', type=str, help='Path to base dataset')
    parser.add_argument('--sizes', type=str, help='Comma-separated list of dataset sizes')
    parser.add_argument('--executable', type=str, help='Path to CNN executable')
    parser.add_argument('--output-dir', type=str, default='prediction_visualizations', 
                        help='Directory for saving prediction visualizations')
    
    args = parser.parse_args()

    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    # Set paths from arguments if provided
    base_path = args.base_path if args.base_path else DEFAULT_BASE_DATASET_PATH
    executable = args.executable if args.executable else EXECUTABLE_PATH
    
    if executable is None:
        print("Warning: Could not find DSPC.exe executable. Please specify path with --executable")
        if args.run or args.all or args.interactive:
            print("Cannot run without executable. Exiting.")
            return
    
    # Define dataset sizes to test
    if args.sizes:
        try:
            dataset_sizes = [int(s.strip()) for s in args.sizes.split(',')]
        except:
            print("Error parsing custom sizes. Using default sizes.")
            dataset_sizes = [10, 20, 50, 100, 200, 500]
    else:
        dataset_sizes = [10, 20, 50, 100, 200, 500]
    
    # Monitor mode has highest priority
    if args.monitor:
        monitor_csv_file()
        return
    
    # Interactive mode has next highest priority
    if args.interactive:
        print("\n=== Running in Interactive Mode ===")
        success = run_interactive(executable)
        if success:
            print("Interactive session completed successfully")
        else:
            print("Interactive session failed")
        return
    
    if args.predict:
        if not args.image_folder:
            print("Error: Please specify an image folder with --image-folder")
            return
            
        print("\n=== Running prediction ===")
        success = run_prediction(
            executable, 
            args.model_type, 
            args.image_folder, 
            args.pred_output,
            args.dataset_size,
            base_path if args.dataset_size else None
        )
        
        if success and args.visualize != False:  # If prediction succeeded and visualization isn't explicitly disabled
            # If user specified a specific CSV file, use that, otherwise use the prediction output
            csv_file = args.visualize if isinstance(args.visualize, str) else args.pred_output
            print(f"\n=== Visualizing prediction results from {csv_file} ===")
            visualize_predictions(csv_file, args.dataset_path, args.output_dir)
        
        return
        
    # Handle single visualization with dataset path
    if args.visualize:
        if not os.path.exists(args.visualize):
            print(f"Error: Prediction CSV file not found: {args.visualize}")
            return
        
        visualize_predictions(args.visualize, args.dataset_path, args.output_dir)
        return
    
     # Handle multi-model comparison with dataset path
    if args.compare_models:
        prediction_files = [file.strip() for file in args.compare_models.split(',')]
        
        # Validate files exist
        valid_files = []
        for file in prediction_files:
            if os.path.exists(file):
                valid_files.append(file)
            else:
                print(f"Warning: File not found: {file}")
        
        if not valid_files:
            print("Error: No valid prediction files found.")
            return
        
        # Parse model names if provided
        model_names = None
        if args.model_names:
            model_names = [name.strip() for name in args.model_names.split(',')]
        
        # Perform multi-model comparison
        visualize_multiple_models(valid_files, model_names, args.dataset_path, 
                                 args.output_dir or 'model_comparison')
        return

    # Execute steps based on arguments
    if args.all:
        print("\n=== STEP 1: Generating datasets ===")
        
        # Check if the base path exists
        if not os.path.exists(base_path):
            print(f"Error: Base dataset path does not exist: {base_path}")
            return
            
        datasets = create_datasets(dataset_sizes, base_path, base_path)
        
        if not datasets:
            print("Error: No datasets were created. Check the base dataset path.")
            return
    
    if args.all or args.run:
        print("\n=== STEP 2: Running benchmarks ===")
        
        # Check if executable exists
        if not os.path.exists(executable):
            print(f"Error: Executable not found at {executable}")
            print("Please compile your CNN program and specify the correct path with --executable")
            return
        
        # Check if any datasets are available
        available_datasets = detect_existing_datasets(base_path)
        if not available_datasets:
            print(f"Error: No datasets found in {base_path}")
            print("Please run with --generate first to create datasets")
            return
            
        print(f"Found {len(available_datasets)} datasets: {available_datasets}")
        
        # Run benchmarks on each dataset
        for size in available_datasets:
            success = run_benchmark(executable, size, base_path)
            if success:
                print(f"Successfully benchmarked dataset size {size}")
            else:
                print(f"Failed to benchmark dataset size {size}")
    
    if args.all or args.plot:
        print("\n=== STEP 3: Generating plots ===")
        
        # Collect and parse benchmark files
        benchmark_files = collect_benchmark_files()
        
        if not benchmark_files:
            print("No benchmark files found in the results folder!")
            print("Please run benchmarks first with --run")
            return
        
        print(f"Found {len(benchmark_files)} benchmark files")
        
        # Parse and combine benchmark results
        results_df = parse_benchmark_results(benchmark_files)
        
        if results_df is not None:
            # Plot results
            plot_results(results_df)
            print("Plots generated successfully in the 'results/figures' directory!")
        else:
            print("Error: Could not parse benchmark results")

if __name__ == "__main__":
    main()