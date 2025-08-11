#!/usr/bin/env python3
"""
Script to create GIF from training epoch comparison images.
"""

import os
import glob
from PIL import Image
import argparse

def create_gif(checkpoint_dir, output_name="training_progress.gif", duration=800):
    """
    Create GIF from epoch comparison images.
    
    Args:
        checkpoint_dir: Directory containing epoch_xxx_comparison.png files
        output_name: Name of output GIF file
        duration: Duration per frame in milliseconds
    """
    
    # Find all epoch comparison images
    pattern = os.path.join(checkpoint_dir, "epoch_*_comparison.png")
    image_files = sorted(glob.glob(pattern))
    
    if not image_files:
        print(f"No epoch comparison images found in {checkpoint_dir}")
        print(f"Looking for pattern: {pattern}")
        return
    
    print(f"Found {len(image_files)} epoch images")
    
    # Load images
    images = []
    for img_file in image_files:
        print(f"Loading: {os.path.basename(img_file)}")
        img = Image.open(img_file)
        images.append(img)
    
    # Create GIF
    gif_path = os.path.join(checkpoint_dir, output_name)
    
    if images:
        # Save as GIF
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0  # Loop forever
        )
        print(f"\nGIF created successfully: {gif_path}")
        print(f"Frames: {len(images)}")
        print(f"Duration per frame: {duration}ms")
        
        # Also create a faster version
        fast_gif_path = os.path.join(checkpoint_dir, f"fast_{output_name}")
        images[0].save(
            fast_gif_path,
            save_all=True,
            append_images=images[1:],
            duration=300,  # Faster
            loop=0
        )
        print(f"Fast GIF created: {fast_gif_path}")
        
    else:
        print("No images to create GIF")

def create_loss_gif(checkpoint_dir, output_name="loss_progress.gif", duration=500):
    """
    Create GIF from loss progression images.
    """
    
    # Find all loss progression images
    pattern = os.path.join(checkpoint_dir, "loss_progression_epoch_*.png")
    image_files = sorted(glob.glob(pattern))
    
    if not image_files:
        print(f"No loss progression images found in {checkpoint_dir}")
        return
    
    print(f"Found {len(image_files)} loss progression images")
    
    # Load images
    images = []
    for img_file in image_files:
        img = Image.open(img_file)
        images.append(img)
    
    # Create GIF
    gif_path = os.path.join(checkpoint_dir, output_name)
    
    if images:
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0
        )
        print(f"Loss progression GIF created: {gif_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create training progress GIF')
    parser.add_argument('--checkpoint-dir', type=str, 
                       default='checkpoints',
                       help='Directory containing epoch images')
    parser.add_argument('--duration', type=int, default=800,
                       help='Duration per frame in milliseconds')
    
    args = parser.parse_args()
    
    checkpoint_dir = args.checkpoint_dir
    if not os.path.isabs(checkpoint_dir):
        checkpoint_dir = os.path.join(os.path.dirname(__file__), checkpoint_dir)
    
    print(f"Creating GIFs from images in: {checkpoint_dir}")
    
    # Create training comparison GIF
    create_gif(checkpoint_dir, "training_progress.gif", args.duration)
    
    # Create loss progression GIF  
    create_loss_gif(checkpoint_dir, "loss_progress.gif", args.duration)
    
    print("\nTo run after training:")
    print(f"python create_training_gif.py --checkpoint-dir {checkpoint_dir}")