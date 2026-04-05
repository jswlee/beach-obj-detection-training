#!/usr/bin/env python3
"""
Filter YOLO labels to keep only those within the ROI mask
"""
import cv2
import numpy as np
from pathlib import Path
import argparse
import yaml
import shutil
import sys

# Import the ROI mask creation function from apply_roi_mask.py
sys.path.append(str(Path(__file__).parent))
from apply_roi_mask import create_roi_mask

def filter_labels_by_roi(image_path, label_path, output_label_path, exclude_pool=True):
    """
    Filter YOLO labels to keep only those within the ROI
    
    Args:
        image_path: Path to the image
        label_path: Path to the YOLO format label file
        output_label_path: Path to save the filtered labels
        exclude_pool: Whether to exclude the pool area
    """
    # Read the image to get dimensions
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not read {image_path}")
        return False
    
    height, width = image.shape[:2]
    
    # Create ROI mask
    mask = create_roi_mask(image.shape, exclude_pool)
    
    # Read labels
    if not Path(label_path).exists():
        print(f"Label file not found: {label_path}")
        return False
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    # Filter labels
    filtered_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            try:
                # YOLO format: x, y, w, h are normalized (0-1)
                x_center = float(parts[1])
                y_center = float(parts[2])
                
                # Convert to pixel coordinates
                x_pixel = int(x_center * width)
                y_pixel = int(y_center * height)
                
                # Check if center point is in the ROI (mask value is 1)
                if mask[y_pixel, x_pixel] == 1:
                    filtered_lines.append(line)
            except (ValueError, IndexError) as e:
                print(f"Error processing label in {label_path}: {e}")
                # Keep the line if there's an error
                filtered_lines.append(line)
    
    # Write filtered labels
    Path(output_label_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_label_path, 'w') as f:
        f.writelines(filtered_lines)
    
    return True

def process_dataset(dataset_dir, output_dir):
    """
    Process a YOLO dataset to filter labels by ROI
    
    Args:
        dataset_dir: Input directory containing YOLO dataset
        output_dir: Output directory for filtered dataset
    """
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each split (train, valid, test)
    for split in ['train', 'valid', 'test']:
        split_images_dir = dataset_path / split / 'images'
        split_labels_dir = dataset_path / split / 'labels'
        
        if not split_images_dir.exists() or not split_labels_dir.exists():
            print(f"Split directory not found: {split}")
            continue
        
        # Create output directories for this split
        output_split_images_dir = output_path / split / 'images'
        output_split_labels_dir = output_path / split / 'labels'
        
        output_split_images_dir.mkdir(parents=True, exist_ok=True)
        output_split_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy images (we'll keep the masked images)
        for img_file in split_images_dir.glob('*.*'):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                shutil.copy(img_file, output_split_images_dir / img_file.name)
        
        # Process each label file
        total_labels = 0
        filtered_labels = 0
        
        for label_file in split_labels_dir.glob('*.txt'):
            # Find corresponding image
            img_stem = label_file.stem
            img_file = None
            
            for ext in ['.jpg', '.jpeg', '.png']:
                potential_img = split_images_dir / f"{img_stem}{ext}"
                if potential_img.exists():
                    img_file = potential_img
                    break
            
            if img_file is None:
                print(f"Could not find image for label {label_file}")
                continue
            
            # Filter labels
            output_label_file = output_split_labels_dir / label_file.name
            
            # Read original labels to count
            with open(label_file, 'r') as f:
                original_lines = f.readlines()
                total_labels += len(original_lines)
            
            # Filter and write new labels
            success = filter_labels_by_roi(
                img_file, 
                label_file, 
                output_label_file
            )
            
            if success:
                # Count filtered labels
                with open(output_label_file, 'r') as f:
                    filtered_lines = f.readlines()
                    filtered_labels += len(filtered_lines)
        
        print(f"{split}: Kept {filtered_labels}/{total_labels} labels ({filtered_labels/total_labels*100:.1f}% in ROI)")
    
    # Copy the data.yaml file
    yaml_file = dataset_path / 'data.yaml'
    if yaml_file.exists():
        with open(yaml_file, 'r') as f:
            data_yaml = yaml.safe_load(f)
        
        # Update paths in the YAML file
        data_yaml['path'] = 'data/roi_masked_data'
        data_yaml['train'] = 'train/images'
        data_yaml['val'] = 'valid/images'
        data_yaml['test'] = 'test/images'
        
        # Write the updated YAML file
        with open(output_path / 'data.yaml', 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"Processed YOLO dataset from {dataset_dir} to {output_dir}")
    return str(output_path / 'data.yaml')

def main():
    parser = argparse.ArgumentParser(description="Filter YOLO labels to keep only those within the ROI")
    parser.add_argument("--input", type=str, default="data/roi_masked_data",
                        help="Input directory containing the masked YOLO dataset")
    parser.add_argument("--output", type=str, default="data/roi_filtered_data",
                        help="Output directory for the filtered dataset")
    
    args = parser.parse_args()
    
    process_dataset(args.input, args.output)

if __name__ == "__main__":
    main()
