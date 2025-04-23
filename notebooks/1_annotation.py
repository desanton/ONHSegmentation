import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from skimage import draw
from glob import glob
from pathlib import Path

def create_label_dictionaries():
    """
    Create dictionaries mapping image filenames to their classes (Glaucoma/Normal)
    and save them as JSON files.
    """
    train_labels = {}
    test_labels = {}
    
    # Process train directory
    for label in ["Glaucoma", "Normal"]:
        train_path = os.path.join('dataset', 'train', 'Images', label)
        if os.path.exists(train_path):
            for img_file in os.listdir(train_path):
                base_filename = os.path.splitext(img_file)[0]
                train_labels[base_filename] = label
                
    # Process test directory
    for label in ["Glaucoma", "Normal"]:
        test_path = os.path.join('dataset', 'test', 'Images', label)
        if os.path.exists(test_path):
            for img_file in os.listdir(test_path):
                base_filename = os.path.splitext(img_file)[0]
                test_labels[base_filename] = label
                
    # Save dictionaries as JSON
    with open("train_labels.json", "w") as f:
        json.dump(train_labels, f, indent=4)
    with open("test_labels.json", "w") as f:
        json.dump(test_labels, f, indent=4)

def load_labelme_annotation(json_file):
    """
    Load LabelMe annotation and convert to a combined mask.
    0 = background, 1 = optic disc, 2 = optic cup
    """
    with open(json_file, 'r') as f:
        annotation = json.load(f)

    img_height = annotation['imageHeight']
    img_width = annotation['imageWidth']

    # Initialize combined mask
    combined_mask = np.zeros((img_height, img_width), dtype=np.uint8)

    for shape in annotation['shapes']:
        label = shape['label']
        points = np.array(shape['points'], dtype=np.int32)
        rr, cc = draw.polygon(points[:, 1], points[:, 0])

        valid_indices = (rr >= 0) & (rr < img_height) & (cc >= 0) & (cc < img_width)

        if label == 'Optic_Cup':
            combined_mask[rr[valid_indices], cc[valid_indices]] = 2  
        elif label == 'Optic_Disc':
            
            target_rr = rr[valid_indices]
            target_cc = cc[valid_indices]
            for r, c in zip(target_rr, target_cc):
                if combined_mask[r, c] == 0:
                    combined_mask[r, c] = 1

    return combined_mask

def save_mask(mask, base_filename, is_train=True):
    """
    Save the combined mask as a single grayscale PNG.
    
    Args:
        mask: numpy array where 0=background, 1=disc, 2=cup
        base_filename: name without extension
        is_train: save to train or test folder
    """
    mask_dir = os.path.join('dataset', 'train' if is_train else 'test', 'Masks')
    os.makedirs(mask_dir, exist_ok=True)

    mask_path = os.path.join(mask_dir, f'{base_filename}_mask.png')
    cv2.imwrite(mask_path, mask)

    return mask_path

def visualize_annotation(image, combined_mask, visualize=True, save_path=None):
    """
    Visualize image with overlaid combined mask.
    
    - Red overlay for optic disc (label 1)
    - Blue overlay for optic cup (label 2)
    """
    plt.figure(figsize=(10, 5))
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Overlayed mask
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    
    # Create overlay masks
    disc_overlay = np.zeros_like(combined_mask, dtype=np.uint8)
    cup_overlay = np.zeros_like(combined_mask, dtype=np.uint8)
    
    disc_overlay[combined_mask == 1] = 255
    cup_overlay[combined_mask == 2] = 255

    plt.imshow(disc_overlay, alpha=0.5, cmap='Reds')
    plt.imshow(cup_overlay, alpha=0.5, cmap='Blues')

    plt.title('Overlay: Disc (Red), Cup (Blue)')
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    
    if visualize:
        plt.show()

def process_annotations(num_to_visualize=5):
    """
    Process all annotation files and generate masks for both train and test sets
    """
    # First ensure we have the label dictionaries
    if not os.path.exists("train_labels.json") or not os.path.exists("test_labels.json"):
        create_label_dictionaries()
        
    # Load label dictionaries
    with open("train_labels.json", "r") as f:
        train_labels = json.load(f)
    with open("test_labels.json", "r") as f:
        test_labels = json.load(f)
    
    for is_train, labels in [(True, train_labels), (False, test_labels)]:
        dataset_type = "train" if is_train else "test"
        print(f"\nProcessing {dataset_type} set annotations...")
        
        i = 0
        annotation_files = []
        for base_filename, label in labels.items():
            json_path = f'annotations/{label}/{base_filename}.json'
            if os.path.exists(json_path):
                annotation_files.append((json_path, base_filename, label))
            else:
                #print(f"Annotation file not found for {base_filename}")
                print()

        for json_file, base_filename, label in annotation_files:
                
            image_path = os.path.join('dataset', dataset_type, 'Images', label, f'{base_filename}.jpg')
            
            if not os.path.exists(image_path):
                image_path = os.path.join('dataset', dataset_type, 'Images', label, f'{base_filename}.png')
                if not os.path.exists(image_path):
                    print(f"Image not found for {base_filename}, skipping")
                    continue
            
            print(f"Processing annotation for {base_filename}")
            
            # Load and convert annotation
            mask = load_labelme_annotation(json_file)
            
            # Save masks
            save_mask(mask, base_filename, is_train=is_train)
        

            # Visualize
            toVisOrNotToVis = False
            if i < num_to_visualize:
                toVisOrNotToVis = True
            i += 1

            
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            visualize_annotation(img, mask, visualize=toVisOrNotToVis)


if __name__ == '__main__':
    create_label_dictionaries()
    process_annotations(num_to_visualize=3)
