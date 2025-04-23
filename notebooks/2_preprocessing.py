import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from skimage import draw
from glob import glob
from pathlib import Path

"""
--------------------------------------------------------------
==================PREPROCESSING===============================
--------------------------------------------------------------
"""

def load_image(image_path):
    """
    Load image and convert BGR to RGB
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def apply_clahe(image):
    """
    Apply CLAHE for contrast enhancement
    
    Rationale: CLAHE enhances local contrast making the boundaries between
    optic disc/cup and surrounding retina more distinct, which improves
    segmentation accuracy.
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge the CLAHE enhanced L channel with the original A and B channels
    enhanced_lab = cv2.merge((cl, a, b))
    
    # Convert back to RGB color space
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    return enhanced_rgb

def get_image_paths():
    """
    Get image paths from both train and test directories for Glaucoma and Normal cases
    """
    image_paths = {
        'train_glaucoma': [],
        'train_normal': [],
        'test_glaucoma': [],
        'test_normal': []
    }
    
    # Training set
    train_glaucoma = glob('dataset/train/Images/Glaucoma/*.png') + glob('dataset/train/Images/Glaucoma/*.jpg')
    train_normal = glob('dataset/train/Images/Normal/*.png') + glob('dataset/train/Images/Normal/*.jpg')
    image_paths['train_glaucoma'].extend(train_glaucoma)
    image_paths['train_normal'].extend(train_normal)
    
    # Test set
    test_glaucoma = glob('dataset/test/Images/Glaucoma/*.png') + glob('dataset/test/Images/Glaucoma/*.jpg')
    test_normal = glob('dataset/test/Images/Normal/*.png') + glob('dataset/test/Images/Normal/*.jpg')
    image_paths['test_glaucoma'].extend(test_glaucoma)
    image_paths['test_normal'].extend(test_normal)
    
    return image_paths

def create_output_directories():
    """
    Create output directories for preprocessed images
    """
    base_dir = "preprocessing"
    directories = [
        f"{base_dir}/train_glaucoma",
        f"{base_dir}/train_normal",
        f"{base_dir}/test_glaucoma",
        f"{base_dir}/test_normal"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def save_preprocessed_image(image, category, filename):
    """
    Save preprocessed image to the appropriate directory
    """
    output_dir = f"preprocessing/{category}"
    output_path = os.path.join(output_dir, filename)
    
    # Convert RGB to BGR for saving with cv2
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, image_bgr)
    return output_path

def visualize_random_samples(preprocessed_images):
    """
    Visualize randomly selected samples from each category
    """
    # Randomly select 1 image from each category
    samples = []
    for category in ['train_glaucoma', 'test_glaucoma', 'train_normal', 'test_normal']:
        if preprocessed_images[category]:  # Check if category has images
            sample = np.random.choice(preprocessed_images[category])
            samples.append((category, sample))
    
    # Create figure with smaller size
    plt.figure(figsize=(10, 11))
    plt.suptitle('Preprocessed Images', fontsize=14, y=0.95)
    
    # Adjust subplot parameters for tighter spacing
    plt.subplots_adjust(hspace=0.3, wspace=0.05)
    
    for idx, (category, sample) in enumerate(samples):
        original, enhanced = sample['original'], sample['enhanced']
        filename = os.path.splitext(os.path.basename(sample['path']))[0]  # Remove file extension
        
        # Original
        plt.subplot(4, 2, idx*2 + 1)
        plt.imshow(original)
        if idx == 0:
            plt.title('Original Image', pad=5)
        plt.axis('off')
        
        # Add filename and category as row label
        if idx % 2 == 0:
            plt.text(-0.2, 0.5, f"{filename}\n({category})", 
                    transform=plt.gca().transAxes,
                    verticalalignment='center',
                    fontsize=8)
        
        # CLAHE Enhanced
        plt.subplot(4, 2, idx*2 + 2)
        plt.imshow(enhanced)
        if idx == 0:
            plt.title('CLAHE Enhanced', pad=5)
        plt.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to account for suptitle
    plt.show()

if __name__ == '__main__':
    # Create output directories
    create_output_directories()
    
    # Get all image paths
    image_paths = get_image_paths()
    
    # Initialize preprocessed images dictionary with separate train/test categories
    preprocessed_images = {
        'train_glaucoma': [],
        'train_normal': [],
        'test_glaucoma': [],
        'test_normal': []
    }
    
    # Process all images keeping train/test separation
    for category in preprocessed_images.keys():
        print(f"Processing {category} images...")
        for path in image_paths[category]:
            try:
                # Get filename without directory path
                filename = os.path.basename(path)
                
                # Load and preprocess image
                image = load_image(path)
                enhanced = apply_clahe(image)
                
                # Save preprocessed image
                saved_path = save_preprocessed_image(enhanced, category, filename)
                #print(f"Saved preprocessed image to: {saved_path}")
                
                # Store for visualization
                preprocessed_images[category].append({
                    'path': path,
                    'original': image,
                    'enhanced': enhanced
                })
            except Exception as e:
                print(f"Error processing {path}: {str(e)}")
    
    # Visualize random samples
    visualize_random_samples(preprocessed_images)
