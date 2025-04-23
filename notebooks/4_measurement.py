import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import json

def extract_contours_from_mask(mask):
    """
    Extract contours for disk (value 1) and cup (value 2) from the segmentation mask
    """
    # Extract optic disk (value 1)
    disk_mask = np.zeros_like(mask, dtype=np.uint8)
    disk_mask[mask >= 1] = 255  # Both disk and cup pixels
    
    # Extract optic cup (value 2)
    cup_mask = np.zeros_like(mask, dtype=np.uint8)
    cup_mask[mask == 2] = 255  # Only cup pixels
    
    # Find contours
    disk_contours, _ = cv2.findContours(disk_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cup_contours, _ = cv2.findContours(cup_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check if contours were found
    disk_contour = None
    cup_contour = None
    
    if disk_contours and len(disk_contours) > 0:
        disk_contour = max(disk_contours, key=cv2.contourArea)
    
    if cup_contours and len(cup_contours) > 0:
        cup_contour = max(cup_contours, key=cv2.contourArea)
    
    return disk_contour, cup_contour

def measure_diameters(contour):
    """
    Measure vertical and horizontal diameters using bounding rectangle and ellipse fitting
    """
    if contour is None or len(contour) < 5:
        return None, None, None, None
    
    # Method 1: Bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)
    rect_horizontal_diameter = w
    rect_vertical_diameter = h
    
    # Method 2: Ellipse fitting
    try:
        if len(contour) >= 5:  # Need at least 5 points to fit an ellipse
            ellipse = cv2.fitEllipse(contour)
            center, axes, angle = ellipse
            ellipse_horizontal_diameter = max(axes)
            ellipse_vertical_diameter = min(axes)
            return rect_horizontal_diameter, rect_vertical_diameter, ellipse_horizontal_diameter, ellipse_vertical_diameter
    except Exception as e:
        print(f"Error fitting ellipse: {e}")
    
    # If we can't fit an ellipse, return only rectangle measurements
    return rect_horizontal_diameter, rect_vertical_diameter, None, None

def calculate_cdr(cup_diameters, disk_diameters):
    """
    Calculate Cup-to-Disc Ratio (CDR) using vertical and horizontal diameters
    """
    # Return None values if either cup or disk measurements are missing
    if cup_diameters is None or disk_diameters is None:
        return None, None
    
    # Use ellipse diameters if available, otherwise use rectangle
    if cup_diameters[2] is not None and disk_diameters[2] is not None:
        # If we have valid ellipse measurements for both cup and disk
        horizontal_cdr = cup_diameters[2] / disk_diameters[2] if disk_diameters[2] > 0 else None
        vertical_cdr = cup_diameters[3] / disk_diameters[3] if disk_diameters[3] > 0 else None
    else:
        # Fall back to rectangle measurements
        # Ensure we're not dividing by zero or using None values
        horizontal_cdr = None
        vertical_cdr = None
        
        if cup_diameters[0] is not None and disk_diameters[0] is not None:
            horizontal_cdr = cup_diameters[0] / disk_diameters[0] if disk_diameters[0] > 0 else None
            
        if cup_diameters[1] is not None and disk_diameters[1] is not None:
            vertical_cdr = cup_diameters[1] / disk_diameters[1] if disk_diameters[1] > 0 else None
    
    return horizontal_cdr, vertical_cdr

def visualize_results(image_name, mask, disk_contour, cup_contour, disk_diameters, cup_diameters, output_dir):
    """
    Visualize the segmentation and measurements
    """
    # Create RGB visualization
    vis_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    # Set background to black (0,0,0)
    # Set disk to blue (0,0,255)
    vis_img[mask >= 1] = [0, 0, 255]
    # Set cup to red (255,0,0)
    vis_img[mask == 2] = [255, 0, 0]
    
    # Draw contours
    if disk_contour is not None:
        cv2.drawContours(vis_img, [disk_contour], -1, (0, 255, 0), 2)
    if cup_contour is not None:
        cv2.drawContours(vis_img, [cup_contour], -1, (0, 255, 255), 2)
    
    # Draw bounding rectangles
    if disk_contour is not None:
        x, y, w, h = cv2.boundingRect(disk_contour)
        cv2.rectangle(vis_img, (x, y), (x+w, y+h), (255, 255, 0), 2)
    
    if cup_contour is not None:
        x, y, w, h = cv2.boundingRect(cup_contour)
        cv2.rectangle(vis_img, (x, y), (x+w, y+h), (255, 0, 255), 2)
    
    # Draw ellipses if possible
    if disk_contour is not None and len(disk_contour) >= 5:
        ellipse = cv2.fitEllipse(disk_contour)
        cv2.ellipse(vis_img, ellipse, (255, 255, 255), 2)
    
    if cup_contour is not None and len(cup_contour) >= 5:
        ellipse = cv2.fitEllipse(cup_contour)
        cv2.ellipse(vis_img, ellipse, (128, 128, 128), 2)
    
    # Add measurement text
    if disk_diameters is not None and disk_diameters[2] is not None:
        text = f"Disk: {disk_diameters[2]:.1f}x{disk_diameters[3]:.1f}"
        cv2.putText(vis_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    if cup_diameters is not None and cup_diameters[2] is not None:
        text = f"Cup: {cup_diameters[2]:.1f}x{cup_diameters[3]:.1f}"
        cv2.putText(vis_img, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    horizontal_cdr, vertical_cdr = calculate_cdr(cup_diameters, disk_diameters)
    
    if horizontal_cdr is not None and vertical_cdr is not None:
        text = f"CDR (H): {horizontal_cdr:.3f}, CDR (V): {vertical_cdr:.3f}"
        cv2.putText(vis_img, text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Save visualization
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, f"{image_name}_analysis.png"), vis_img)
    
    return vis_img

def process_all_masks(masks_dir, output_dir="results/analysis"):
    """
    Process all segmentation masks in a directory
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Results storage
    results = []
    
    # Process each mask
    mask_files = list(Path(masks_dir).glob("*.png")) + list(Path(masks_dir).glob("*.tif")) + list(Path(masks_dir).glob("*.tiff"))
    
    for mask_path in mask_files:
        image_name = mask_path.stem
        print(f"Processing {image_name}...")
        
        # Read mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Error reading {mask_path}")
            continue
        
        # Debug: Check if mask contains expected values
        unique_values = np.unique(mask)
        print(f"  Unique values in mask: {unique_values}")
        
        # Extract contours
        disk_contour, cup_contour = extract_contours_from_mask(mask)
        
        if disk_contour is None:
            print(f"  Warning: No disk contour found in {image_name}")
        if cup_contour is None:
            print(f"  Warning: No cup contour found in {image_name}")
        
        # Measure diameters
        disk_diameters = measure_diameters(disk_contour)
        cup_diameters = measure_diameters(cup_contour)
        
        print(f"  Disk diameters: {disk_diameters}")
        print(f"  Cup diameters: {cup_diameters}")
        
        # Calculate CDR
        horizontal_cdr, vertical_cdr = calculate_cdr(cup_diameters, disk_diameters)
        
        print(f"  CDR (H x V): {horizontal_cdr} x {vertical_cdr}")
        
        # Visualize results
        try:
            vis_img = visualize_results(image_name, mask, disk_contour, cup_contour, 
                                    disk_diameters, cup_diameters, vis_dir)
        except Exception as e:
            print(f"  Error visualizing results: {e}")
        
        # Store results
        result = {
            'Image': image_name,
            'Disk_Width_Rect': disk_diameters[0] if disk_diameters and disk_diameters[0] is not None else None,
            'Disk_Height_Rect': disk_diameters[1] if disk_diameters and disk_diameters[1] is not None else None,
            'Disk_Width_Ellipse': disk_diameters[2] if disk_diameters and disk_diameters[2] is not None else None,
            'Disk_Height_Ellipse': disk_diameters[3] if disk_diameters and disk_diameters[3] is not None else None,
            'Cup_Width_Rect': cup_diameters[0] if cup_diameters and cup_diameters[0] is not None else None,
            'Cup_Height_Rect': cup_diameters[1] if cup_diameters and cup_diameters[1] is not None else None,
            'Cup_Width_Ellipse': cup_diameters[2] if cup_diameters and cup_diameters[2] is not None else None,
            'Cup_Height_Ellipse': cup_diameters[3] if cup_diameters and cup_diameters[3] is not None else None,
            'Horizontal_CDR': horizontal_cdr,
            'Vertical_CDR': vertical_cdr,
            'Average_CDR': (horizontal_cdr + vertical_cdr) / 2 if horizontal_cdr is not None and vertical_cdr is not None else None
        }
        
        results.append(result)
    
    # Create and save results DataFrame
    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, "cdr_measurements.csv")
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
        
        # Summary statistics
        summary = df[['Horizontal_CDR', 'Vertical_CDR', 'Average_CDR']].describe()
        summary_path = os.path.join(output_dir, "cdr_summary.csv")
        #summary.to_csv(summary_path)
        #print(f"Summary statistics saved to {summary_path}")
        
        return df
    else:
        print("No results to save")
        return None


def analyze_cdr_by_class(results_df, labels_file, output_dir="results/analysis"):
    """
    Analyze CDR distribution based on image class (Glaucoma vs Normal)
    
    Parameters:
    - results_df: DataFrame containing CDR measurements
    - labels_file: Path to JSON file with class labels
    - output_dir: Directory to save output visualizations
    
    Returns:
    - Dictionary containing statistics for each class
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load labels
    with open(labels_file, 'r') as f:
        labels = json.load(f)
    
    # Add label column to results dataframe
    results_df['Label'] = results_df['Image'].apply(lambda x: labels.get(x, "Unknown"))
    
    # Filter out rows with missing CDR values
    valid_df = results_df.dropna(subset=['Average_CDR'])
    
    # Split by class
    glaucoma_df = valid_df[valid_df['Label'] == 'Glaucoma']
    normal_df = valid_df[valid_df['Label'] == 'Normal']
    
    print(f"Number of valid glaucoma images: {len(glaucoma_df)}")
    print(f"Number of valid normal images: {len(normal_df)}")
    
    stats_dict = {}
    
    # Process glaucoma images
    if not glaucoma_df.empty:
        stats_dict['Glaucoma'] = {
            'count': len(glaucoma_df),
            'mean': glaucoma_df['Average_CDR'].mean(),
            'std': glaucoma_df['Average_CDR'].std(),
            'min': glaucoma_df['Average_CDR'].min(),
            'max': glaucoma_df['Average_CDR'].max(),
            'median': glaucoma_df['Average_CDR'].median()
        }
        
        # Create visualization
        plot_cdr_distribution(glaucoma_df, 'Glaucoma', output_dir)
    else:
        print("No valid glaucoma images found")
    
    # Process normal images
    if not normal_df.empty:
        stats_dict['Normal'] = {
            'count': len(normal_df),
            'mean': normal_df['Average_CDR'].mean(),
            'std': normal_df['Average_CDR'].std(),
            'min': normal_df['Average_CDR'].min(),
            'max': normal_df['Average_CDR'].max(),
            'median': normal_df['Average_CDR'].median()
        }
        
        # Create visualization
        plot_cdr_distribution(normal_df, 'Normal', output_dir)
    else:
        print("No valid normal images found")
    
    # Save statistics to CSV
    stats_df = pd.DataFrame(stats_dict).T
    stats_df.to_csv(os.path.join(output_dir, "cdr_class_statistics.csv"))
    
    # Create comparison plot
    if not glaucoma_df.empty and not normal_df.empty:
        plot_cdr_comparison(glaucoma_df, normal_df, output_dir)
    
    return stats_dict

def plot_cdr_distribution(df, class_name, output_dir):
    """
    Create a bar chart with normal distribution overlay for a class
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Extract CDR values
    cdr_values = df['Average_CDR'].values
    
    # Calculate statistics
    mean_cdr = np.mean(cdr_values)
    std_cdr = np.std(cdr_values)
    
    # Create bar chart
    ax.bar(df['Image'], df['Average_CDR'], color='skyblue', alpha=0.7)
    ax.set_xlabel('Image')
    ax.set_ylabel('Average CDR')
    ax.set_title(f'CDR Distribution for {class_name} Images')
    ax.tick_params(axis='x', rotation=90)
    
    # Add horizontal line for mean
    ax.axhline(y=mean_cdr, color='red', linestyle='-', label=f'Mean: {mean_cdr:.3f}')
    
    # Add lines for mean ± std
    ax.axhline(y=mean_cdr + std_cdr, color='red', linestyle='--', alpha=0.5, 
               label=f'Mean + Std: {mean_cdr + std_cdr:.3f}')
    ax.axhline(y=mean_cdr - std_cdr, color='red', linestyle='--', alpha=0.5,
               label=f'Mean - Std: {mean_cdr - std_cdr:.3f}')
    
    # Add legend
    ax.legend()
    
    # Save bar chart
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{class_name.lower()}_cdr_barchart.png"), dpi=300)
    
    # Create normal distribution plot
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    # Generate x values for the distribution curve
    x = np.linspace(max(0, mean_cdr - 3*std_cdr), min(1, mean_cdr + 3*std_cdr), 1000)
    
    # Generate normal distribution
    pdf = stats.norm.pdf(x, mean_cdr, std_cdr)
    
    # Plot the normal distribution
    ax2.plot(x, pdf, 'r-', linewidth=2, label=f'Normal Distribution\nMean={mean_cdr:.3f}, Std={std_cdr:.3f}')
    
    # Add histogram
    ax2.hist(cdr_values, bins=10, density=True, alpha=0.6, color='skyblue', label='Actual Data')
    
    # Add labels and title
    ax2.set_xlabel('Average CDR')
    ax2.set_ylabel('Density')
    ax2.set_title(f'CDR Distribution for {class_name} Images')
    
    # Add legend
    ax2.legend()
    
    # Save distribution plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{class_name.lower()}_cdr_distribution.png"), dpi=300)
    
    # Close figures to free memory
    plt.close(fig)
    plt.close(fig2)

def plot_cdr_comparison(glaucoma_df, normal_df, output_dir):
    """
    Create a comparison plot for glaucoma vs normal CDR distributions
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract CDR values
    glaucoma_cdr = glaucoma_df['Average_CDR'].values
    normal_cdr = normal_df['Average_CDR'].values
    
    # Calculate statistics
    glaucoma_mean = np.mean(glaucoma_cdr)
    glaucoma_std = np.std(glaucoma_cdr)
    normal_mean = np.mean(normal_cdr)
    normal_std = np.std(normal_cdr)
    
    # Generate x values for the distribution curves
    x = np.linspace(max(0, min(normal_mean, glaucoma_mean) - 3*max(normal_std, glaucoma_std)), 
                   min(1, max(normal_mean, glaucoma_mean) + 3*max(normal_std, glaucoma_std)), 1000)
    
    # Generate normal distributions
    glaucoma_pdf = stats.norm.pdf(x, glaucoma_mean, glaucoma_std)
    normal_pdf = stats.norm.pdf(x, normal_mean, normal_std)
    
    # Plot the distributions
    ax.plot(x, glaucoma_pdf, 'r-', linewidth=2, 
            label=f'Glaucoma (Mean={glaucoma_mean:.3f}, Std={glaucoma_std:.3f})')
    ax.plot(x, normal_pdf, 'g-', linewidth=2, 
            label=f'Normal (Mean={normal_mean:.3f}, Std={normal_std:.3f})')
    
    # Add histograms
    ax.hist(glaucoma_cdr, bins=10, density=True, alpha=0.3, color='red', label='Glaucoma Data')
    ax.hist(normal_cdr, bins=10, density=True, alpha=0.3, color='green', label='Normal Data')
    
    # Add labels and title
    ax.set_xlabel('Average CDR')
    ax.set_ylabel('Density')
    ax.set_title('CDR Distribution Comparison: Glaucoma vs Normal')
    
    # Add legend
    ax.legend()
    
    # Save comparison plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cdr_comparison.png"), dpi=300)
    
    # Boxplot comparison
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    
    # Create data for boxplot
    data = [glaucoma_cdr, normal_cdr]
    labels = ['Glaucoma', 'Normal']
    
    # Create boxplot
    bp = ax2.boxplot(data, labels=labels, patch_artist=True)
    
    # Customize boxplot colors
    for i, box in enumerate(bp['boxes']):
        if i == 0:  # Glaucoma
            box.set(facecolor='lightcoral')
        else:  # Normal
            box.set(facecolor='lightgreen')
    
    # Add scatter points for individual values
    for i, d in enumerate([glaucoma_cdr, normal_cdr]):
        y = d
        x = np.random.normal(i+1, 0.04, size=len(y))
        ax2.scatter(x, y, alpha=0.5, s=20)
    
    # Add labels and title
    ax2.set_ylabel('Average CDR')
    ax2.set_title('CDR Comparison: Glaucoma vs Normal')
    
    # Add p-value if sample sizes are sufficient
    if len(glaucoma_cdr) >= 5 and len(normal_cdr) >= 5:
        t_stat, p_value = stats.ttest_ind(glaucoma_cdr, normal_cdr, equal_var=False)
        ax2.text(0.5, 0.01, f'p-value: {p_value:.4f}', 
                 horizontalalignment='center', transform=ax2.transAxes)
    
    # Save boxplot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cdr_boxplot_comparison.png"), dpi=300)
    
    # Close figures
    plt.close(fig)
    plt.close(fig2)

if __name__ == "__main__":
    masks_dir = "results/segmented_masks"
    labels_file = "test_labels.json"

    results_df = process_all_masks(masks_dir)
    
    # Analyze CDR by class
    if results_df is not None and not results_df.empty:
        stats = analyze_cdr_by_class(results_df, labels_file)
        
        # Print summary
        print("\nSummary Statistics by Class:")
        for class_name, class_stats in stats.items():
            print(f"\n{class_name}:")
            for stat_name, stat_value in class_stats.items():
                print(f"  {stat_name}: {stat_value:.4f}" if isinstance(stat_value, float) else f"  {stat_name}: {stat_value}")