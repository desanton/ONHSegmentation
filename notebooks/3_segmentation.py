import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import random
from glob import glob
from PIL import Image
import segmentation_models_pytorch as smp
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, Resize, ScaleIntensity, ToTensor
)
from monai.data import Dataset, DataLoader
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
import time

os.makedirs("results/segmented_masks", exist_ok=True)
os.makedirs("models", exist_ok=True)

# MODEL 1: Otsu's Thresholding
def otsu_segmentation(image_path):
    """
    Segment using Otsu's thresholding
    """
    # Load and convert to grayscale
    image = cv2.imread(image_path)
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Apply Otsu's thresholding
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Post-processing: morphological operations to refine the mask
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)  # Remove small noise
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # Fill small holes
    
    return binary

# MODEL 2: Contour Detection
def contour_segmentation(image_path):
    """
    Segment using Canny edge detection followed by contour extraction
    """
    # Load and convert to grayscale
    image = cv2.imread(image_path)
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create mask from contours
    mask = np.zeros_like(gray_image)
    if contours:
        # Sort contours by area and take the largest
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Take only the largest few contours that may represent disc/cup
        num_contours = min(3, len(contours))
        for i in range(num_contours):
            cv2.drawContours(mask, [contours[i]], -1, 255, -1)
    
    # Post-processing
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

# MODEL 3: U-Net with Pretrained Encoder
class CupDiskTransform:
    def __init__(self, image_size=(256, 256), train_mode=False):
        self.image_transform = Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            ScaleIntensity(),
            Resize(image_size),
            ToTensor()
        ])
        self.image_size = image_size
        self.train_mode = train_mode

    def __call__(self, sample):
        # Transform the image
        image = self.image_transform(sample["image"])
        
        if self.train_mode and "mask" in sample:
            # Load and prepare mask for training
            mask = np.array(Image.open(sample["mask"]).resize(self.image_size, resample=Image.NEAREST))
            mask = torch.from_numpy(mask).long()
            return {"image": image, "label": mask}
        
        return {"image": image}

def prepare_dataset():
    """
    Prepare training and validation datasets
    """
    # Define your image folders
    image_dirs = [
        "preprocessing/train_glaucoma",
        "preprocessing/train_normal"
    ]

    # Build the list of data samples
    train_data = []

    for image_dir in image_dirs:
        image_paths = glob(os.path.join(image_dir, "*.png"))

        for image_path in image_paths:
            base_name = os.path.splitext(os.path.basename(image_path))[0]  # e.g., drishtiGS_###

            # Path to corresponding mask
            mask_path = os.path.join("dataset", "train", "Masks", f"{base_name}_mask.png")

            if os.path.exists(mask_path):
                train_data.append({
                    "image": image_path,
                    "mask": mask_path
                })

    print(f"Found {len(train_data)} training images with masks")
    
    # Split into training and validation sets (80/20)
    random.shuffle(train_data)
    split_idx = int(len(train_data) * 0.8)
    train_set = train_data[:split_idx]
    val_set = train_data[split_idx:]
    
    return train_set, val_set

def finetune_unet_model(train_set, val_set, model_save_path="models/unet_finetuned.pth", 
                        num_epochs=20, batch_size=4, learning_rate=1e-4):
    """
    Fine-tune a pretrained U-Net model for cup and disk segmentation
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create datasets and dataloaders
    transform = CupDiskTransform(train_mode=True)
    train_ds = Dataset(data=train_set, transform=transform)
    val_ds = Dataset(data=val_set, transform=transform)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    # Create model with pretrained encoder
    # Best options for medical imaging segmentation:
    # - efficientnet-b0 to b7 (lighter to heavier)
    # - resnet34 or resnet50 (good balance)
    # - densenet121 (good for detail preservation)
    model = smp.Unet(
        encoder_name="efficientnet-b3",  # Better than resnet34 for medical imaging
        encoder_weights="imagenet",      # Using imagenet weights
        in_channels=3,                   # RGB input
        classes=3                        # Background, Optic Disc, Optic Cup
    ).to(device)
    
    # Use a balanced loss function since your classes are likely imbalanced
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.45, 0.45]).to(device))
    
    # Use an optimizer with learning rate scheduling
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0
        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["label"].to(device)  # [B, H, W]
            
            optimizer.zero_grad()
            outputs = model(images)  # [B, C, H, W]
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                masks = batch["label"].to(device)
                
                outputs = model(images)
                loss = loss_fn(outputs, masks)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved new best model with validation loss: {best_val_loss:.4f}")
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Time: {epoch_time:.1f}s, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    return model_save_path

def calculate_iou(pred, target, class_id):
    """
    Calculate IoU for a specific class
    pred and target are numpy arrays
    class_id is the class to calculate IoU for
    """
    # Create binary masks for the class
    pred_mask = (pred == class_id)
    target_mask = (target == class_id)
    
    # Calculate intersection and union
    intersection = np.logical_and(pred_mask, target_mask).sum()
    union = np.logical_or(pred_mask, target_mask).sum()
    
    # Handle division by zero
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union

def run_unet_segmentation(model_path="models/unet_finetuned.pth", val_set=None):
    """
    Run the U-Net model on test images and save segmentation masks
    Calculate IoU for validation images where ground truth is available
    
    Args:
        model_path: Path to the trained model
        val_set: Validation dataset (list of dictionaries with 'image' and 'mask' keys)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the model
    model = smp.Unet(
        encoder_name="efficientnet-b3",
        encoder_weights="imagenet",
        in_channels=3,
        classes=3
    ).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded fine-tuned model from {model_path}")
    else:
        print("No fine-tuned model found. Using model with ImageNet weights only.")
    
    model.eval()
    
    # Process test images (without IoU calculation since no ground truth)
    test_image_paths = sorted(glob("preprocessing/test_glaucoma/*.png") + 
                              glob("preprocessing/test_normal/*.png"))
    
    transform = CupDiskTransform()
    
    print("Processing test images...")
    for path in test_image_paths:
        # Get the base filename
        base_name = os.path.basename(path)
        
        # Transform the image
        image_tensor = transform({"image": path})["image"].unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            output = model(image_tensor)
            pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        
        # Save the prediction mask
        output_path = os.path.join("results/segmented_masks", base_name)
        Image.fromarray(pred_mask.astype(np.uint8)).save(output_path)
    
    print("Test image segmentation completed.")
    
    # Calculate IoU for validation set if provided
    if val_set:
        print("\nCalculating IoU for validation set...")
        
        # Lists to store IoU scores
        disc_ious = []
        cup_ious = []
        mean_ious = []
        
        # Transform for validation images
        val_transform = CupDiskTransform(train_mode=False)
        
        for sample in val_set:
            image_path = sample["image"]
            mask_path = sample["mask"]
            
            base_name = os.path.basename(image_path)
            
            # Transform the image
            image_tensor = val_transform({"image": image_path})["image"].unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(image_tensor)
                pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            
            # Load ground truth mask
            ground_truth = np.array(Image.open(mask_path).resize((256, 256), resample=Image.NEAREST))
            
            # Calculate IoU for optic disc (class 1)
            disc_iou = calculate_iou(pred_mask, ground_truth, 1)
            disc_ious.append(disc_iou)
            
            # Calculate IoU for optic cup (class 2)
            cup_iou = calculate_iou(pred_mask, ground_truth, 2)
            cup_ious.append(cup_iou)
            
            # Calculate mean IoU for this image
            mean_iou = (disc_iou + cup_iou) / 2
            mean_ious.append(mean_iou)
            
            print(f"Processed {base_name}: Disc IoU = {disc_iou:.4f}, Cup IoU = {cup_iou:.4f}, Mean IoU = {mean_iou:.4f}")
        
        # Calculate and report average IoU metrics
        if disc_ious:
            avg_disc_iou = sum(disc_ious) / len(disc_ious)
            avg_cup_iou = sum(cup_ious) / len(cup_ious)
            avg_mean_iou = sum(mean_ious) / len(mean_ious)
            
            print("\nValidation Set Performance Metrics:")
            print(f"Average Optic Disc IoU: {avg_disc_iou:.4f}")
            print(f"Average Optic Cup IoU: {avg_cup_iou:.4f}")
            print(f"Average Mean IoU: {avg_mean_iou:.4f}")
        else:
            print("\nNo IoU metrics could be calculated for validation set.")
    
    print("U-Net segmentation completed.")

def visualize_comparison(sample_images=5):
    """
    Visualize segmentation results for random test images using all three models
    """
    # Get test image paths
    test_image_paths = sorted(glob("preprocessing/test_glaucoma/*.png") + 
                             glob("preprocessing/test_normal/*.png"))
    
    # Randomly select images
    if len(test_image_paths) > sample_images:
        selected_images = random.sample(test_image_paths, sample_images)
    else:
        selected_images = test_image_paths
    
    # Set up the figure
    fig, axes = plt.subplots(len(selected_images), 4, figsize=(16, 4 * len(selected_images)))
    
    # Set column titles
    if len(selected_images) > 1:
        axes[0, 0].set_title('Original')
        axes[0, 1].set_title('Otsu Segmentation')
        axes[0, 2].set_title('Contour Segmentation')
        axes[0, 3].set_title('U-Net Segmentation')
    else:
        axes[0].set_title('Original')
        axes[1].set_title('Otsu Segmentation')
        axes[2].set_title('Contour Segmentation')
        axes[3].set_title('U-Net Segmentation')
    
    for i, image_path in enumerate(selected_images):
        filename = os.path.basename(image_path)
        
        # Load the original image
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # Run Otsu segmentation
        otsu_mask = otsu_segmentation(image_path)
        
        # Run contour segmentation
        contour_mask = contour_segmentation(image_path)
        
        # Load U-Net segmentation result
        unet_result_path = os.path.join("results/segmented_masks", filename)
        if os.path.exists(unet_result_path):
            unet_mask = np.array(Image.open(unet_result_path))
            # Create a colormap for different classes (0: background, 1: disc, 2: cup)
            unet_rgb = np.zeros((*unet_mask.shape, 3), dtype=np.uint8)
            unet_rgb[unet_mask == 1] = [0, 255, 0]  # Green for disc
            unet_rgb[unet_mask == 2] = [0, 0, 255]  # Blue for cup
        else:
            print(f"Warning: U-Net result not found for {filename}")
            unet_rgb = np.zeros((*original_image.shape[:2], 3), dtype=np.uint8)
        
        # Display the images
        if len(selected_images) > 1:
            row = axes[i]
            # Set row title (filename)
            row[0].set_ylabel(filename, rotation=0, size='large', labelpad=40)
            
            # Original image
            row[0].imshow(original_image)
            row[0].axis('off')
            
            # Otsu result
            row[1].imshow(original_image)
            row[1].imshow(otsu_mask, alpha=0.5, cmap='Reds')
            row[1].axis('off')
            
            # Contour result
            row[2].imshow(original_image)
            row[2].imshow(contour_mask, alpha=0.5, cmap='Greens')
            row[2].axis('off')
            
            # U-Net result
            row[3].imshow(original_image)
            row[3].imshow(unet_rgb, alpha=0.5)
            row[3].axis('off')
        else:
            # Same logic for single image case
            axes[0].imshow(original_image)
            axes[0].set_ylabel(filename, rotation=0, size='large', labelpad=40)
            axes[0].axis('off')
            
            axes[1].imshow(original_image)
            axes[1].imshow(otsu_mask, alpha=0.5, cmap='Reds')
            axes[1].axis('off')
            
            axes[2].imshow(original_image)
            axes[2].imshow(contour_mask, alpha=0.5, cmap='Greens')
            axes[2].axis('off')
            
            axes[3].imshow(original_image)
            axes[3].imshow(unet_rgb, alpha=0.5)
            axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig('segmentation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Prepare datasets
    train_set, val_set = prepare_dataset()
    
    # Fine-tune the model (this may take some time)
    model_path = finetune_unet_model(
        train_set, 
        val_set,
        num_epochs=7,  # Adjust based on how powerful your computer
        batch_size=4,   # Reduce if memory issues occur
        learning_rate=3e-4
    )
    
    run_unet_segmentation(model_path, val_set=val_set)

    visualize_comparison(sample_images=5)