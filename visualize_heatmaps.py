import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def normalize_heatmap(heatmap):
    """Normalize heatmap to 0-255."""
    h_min = heatmap.min()
    h_max = heatmap.max()
    if h_max - h_min > 0:
        norm_h = (heatmap - h_min) / (h_max - h_min)
    else:
        norm_h = heatmap
    return (norm_h * 255).astype(np.uint8)

def colorize_heatmap(heatmap, colormap=cv2.COLORMAP_JET):
    """Apply colormap to heatmap."""
    heatmap_u8 = normalize_heatmap(heatmap)
    color_h = cv2.applyColorMap(heatmap_u8, colormap)
    return color_h

def visualize_combined_heatmaps(npy_path, out_path, num_channels=None):
    """Load npy file, compute max across channels and save."""
    try:
        data = np.load(npy_path)
        # Assuming shape is (C, H, W)
        
        if len(data.shape) == 3:
            # Combine across channels (take max value)
            if num_channels is not None:
                data = data[:num_channels]
            combined_h = np.max(data, axis=0)
        else:
            combined_h = data
            
        color_h = colorize_heatmap(combined_h)
        
        cv2.imwrite(out_path, color_h)
        return True
    except Exception as e:
        print(f"Error processing {npy_path}: {e}")
        return False

def visualize_separate_heatmaps(npy_path, out_dir, num_channels=None):
    """Save each channel separately and a combined one."""
    try:
        data = np.load(npy_path)
        base_name = Path(npy_path).stem
        
        if len(data.shape) == 3:
            channels = min(data.shape[0], num_channels) if num_channels is not None else data.shape[0]
            
            # Save separate channels
            for c in range(channels):
                color_h = colorize_heatmap(data[c])
                out_p = os.path.join(out_dir, f"{base_name}_c{c}.png")
                cv2.imwrite(out_p, color_h)
                
            # Combine and save
            combined_h = np.max(data, axis=0)
            color_h = colorize_heatmap(combined_h)
            cv2.imwrite(os.path.join(out_dir, f"{base_name}_combined.png"), color_h)
        else:
            color_h = colorize_heatmap(data)
            cv2.imwrite(os.path.join(out_dir, f"{base_name}.png"), color_h)
            
        return True
    except Exception as e:
        print(f"Error processing {npy_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Visualize .npy heatmaps.")
    parser.add_argument("--input_dir", type=str, default="dji_action4_real_wrist_jacket_occlusion_mynet/heatmaps", 
                        help="Directory containing .npy files (can have subdirs like train/val)")
    parser.add_argument("--output_dir", type=str, default="vis_heatmaps", help="Directory to save visualized images")
    parser.add_argument("--mode", type=str, choices=["combined", "separate"], default="combined", 
                        help="combined: save one image per file; separate: save each channel separately")
    parser.add_argument("--max_files", type=int, default=100, help="Maximum number of files to process")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all npy files recursively
    npy_files = []
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith('.npy'):
                npy_files.append(os.path.join(root, file))
    
    if not npy_files:
        print(f"No .npy files found in {args.input_dir}")
        return
        
    print(f"Found {len(npy_files)} files. Processing first {min(args.max_files, len(npy_files))} files...")
    
    success_count = 0
    for i, npy_path in enumerate(npy_files[:args.max_files]):
        rel_path = os.path.relpath(npy_path, args.input_dir)
        base_name = Path(npy_path).stem
        
        # Create output subdirectory structure if needed
        rel_dir = os.path.dirname(rel_path)
        out_subdir = os.path.join(args.output_dir, rel_dir)
        os.makedirs(out_subdir, exist_ok=True)
        
        if args.mode == "combined":
            out_path = os.path.join(out_subdir, f"{base_name}.png")
            if visualize_combined_heatmaps(npy_path, out_path):
                success_count += 1
        elif args.mode == "separate":
            if visualize_separate_heatmaps(npy_path, out_subdir):
                success_count += 1
                
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{min(args.max_files, len(npy_files))} files")
            
    print(f"Finished! Successfully visualized {success_count} heatmaps.")
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
