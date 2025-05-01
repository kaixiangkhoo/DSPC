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
DEFAULT_BASE_DATASET_PATH = "./imageDataset"  # Path to your cancer dataset
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
            
            # Import necessary metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
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
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
    
    # Additional options
    parser.add_argument('--base-path', type=str, help='Path to base dataset')
    parser.add_argument('--sizes', type=str, help='Comma-separated list of dataset sizes')
    parser.add_argument('--executable', type=str, help='Path to CNN executable')
    parser.add_argument('--output-dir', type=str, default='prediction_visualizations', 
                        help='Directory for saving prediction visualizations')
    
    # Dataset integration for ground truth
    parser.add_argument('--dataset-path', type=str,
                        help='Path to dataset folder containing sampled_labels.csv for ground truth')
    
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
    
    # Handle single visualization with dataset path
    if args.visualize:
        if not os.path.exists(args.visualize):
            print(f"Error: Prediction CSV file not found: {args.visualize}")
            return
        
        visualize_predictions(args.visualize, args.dataset_path, args.output_dir)
        return

    # Execute steps based on arguments
    if args.all or args.run:
        print("\n=== Running benchmarks ===")
        
        # Check if executable exists
        if not os.path.exists(executable):
            print(f"Error: Executable not found at {executable}")
            print("Please compile your CNN program and specify the correct path with --executable")
            return
        
        # Check if any datasets are available
        available_datasets = detect_existing_datasets(base_path)
        if not available_datasets:
            print(f"Error: No datasets found in {base_path}")
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
        print("\n=== Generating plots ===")
        
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