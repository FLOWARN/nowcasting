#!/usr/bin/env python3
"""
Script to plot training and validation losses from CSV file.
Creates both individual plots and combined plots with various styles.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Plot training and validation losses from CSV')
    parser.add_argument('--csv-path', type=str, default='checkpoints/training_losses.csv',
                        help='Path to CSV file containing losses (default: checkpoints/training_losses.csv)')
    parser.add_argument('--output-dir', type=str, default='loss_plots_5e-5',
                        help='Directory to save plots (default: loss_plots_5e-5)')
    parser.add_argument('--style', type=str, default='seaborn-v0_8', 
                        choices=['default', 'seaborn-v0_8', 'ggplot', 'bmh'],
                        help='Plot style (default: seaborn-v0_8)')
    parser.add_argument('--figsize', type=int, nargs=2, default=[12, 8],
                        help='Figure size as width height (default: 12 8)')
    parser.add_argument('--dpi', type=int, default=150,
                        help='DPI for saved figures (default: 150)')
    parser.add_argument('--show-plots', action='store_true',
                        help='Display plots interactively (default: False)')
    return parser.parse_args()

def load_losses(csv_path):
    """Load losses from CSV file with error handling."""
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        
        # Check required columns
        required_cols = ['epoch', 'train_loss']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Handle missing validation loss
        if 'val_loss' not in df.columns:
            df['val_loss'] = np.nan
            print("Warning: No validation loss found, will plot training loss only")
        
        # Convert empty strings to NaN
        df['val_loss'] = pd.to_numeric(df['val_loss'], errors='coerce')
        
        print(f"Loaded {len(df)} epochs of training data")
        print(f"Training loss range: {df['train_loss'].min():.6f} - {df['train_loss'].max():.6f}")
        
        if not df['val_loss'].isna().all():
            valid_val = df['val_loss'].dropna()
            print(f"Validation loss range: {valid_val.min():.6f} - {valid_val.max():.6f}")
        
        return df
        
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")

def plot_losses_combined(df, output_dir, figsize, dpi):
    """Create combined training and validation loss plot."""
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot training loss
    ax.plot(df['epoch'], df['train_loss'], 'b-', linewidth=2, marker='o', 
            markersize=4, label='Training Loss', alpha=0.8)
    
    # Plot validation loss if available
    if not df['val_loss'].isna().all():
        valid_data = df.dropna(subset=['val_loss'])
        ax.plot(valid_data['epoch'], valid_data['val_loss'], 'r-', linewidth=2, 
                marker='s', markersize=4, label='Validation Loss', alpha=0.8)
    
    # Formatting
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.set_title('Training and Validation Loss Over Time', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Set integer ticks for epochs
    ax.set_xticks(df['epoch'])
    
    # Add loss values as text annotations
    for idx, row in df.iterrows():
        if idx % max(1, len(df) // 10) == 0:  # Annotate every 10th point or all if few points
            ax.annotate(f'{row["train_loss"]:.4f}', 
                       (row['epoch'], row['train_loss']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, 'training_validation_losses.png')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved combined plot: {output_path}")
    
    return fig

def plot_losses_separate(df, output_dir, figsize, dpi):
    """Create separate plots for training and validation losses."""
    
    # Training loss plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df['epoch'], df['train_loss'], 'b-', linewidth=3, marker='o', markersize=6)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Training Loss', fontsize=14)
    ax.set_title('Training Loss Over Time', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(df['epoch'])
    
    # Add trend line
    z = np.polyfit(df['epoch'], df['train_loss'], 1)
    p = np.poly1d(z)
    ax.plot(df['epoch'], p(df['epoch']), 'r--', alpha=0.7, linewidth=2, 
            label=f'Trend: {z[0]:.6f}x + {z[1]:.6f}')
    ax.legend()
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'training_loss_only.png')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved training loss plot: {output_path}")
    plt.close()
    
    # Validation loss plot (if available)
    if not df['val_loss'].isna().all():
        valid_data = df.dropna(subset=['val_loss'])
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(valid_data['epoch'], valid_data['val_loss'], 'r-', linewidth=3, marker='s', markersize=6)
        ax.set_xlabel('Epoch', fontsize=14)
        ax.set_ylabel('Validation Loss', fontsize=14)
        ax.set_title('Validation Loss Over Time', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(valid_data['epoch'])
        
        # Add trend line
        z = np.polyfit(valid_data['epoch'], valid_data['val_loss'], 1)
        p = np.poly1d(z)
        ax.plot(valid_data['epoch'], p(valid_data['epoch']), 'b--', alpha=0.7, linewidth=2,
                label=f'Trend: {z[0]:.6f}x + {z[1]:.6f}')
        ax.legend()
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'validation_loss_only.png')
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved validation loss plot: {output_path}")
        plt.close()

def plot_loss_difference(df, output_dir, figsize, dpi):
    """Plot difference between training and validation loss."""
    
    if df['val_loss'].isna().all():
        print("Skipping loss difference plot - no validation data available")
        return
    
    valid_data = df.dropna(subset=['val_loss'])
    loss_diff = valid_data['val_loss'] - valid_data['train_loss']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot difference
    colors = ['green' if x <= 0 else 'red' for x in loss_diff]
    ax.bar(valid_data['epoch'], loss_diff, color=colors, alpha=0.7, width=0.6)
    
    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Validation Loss - Training Loss', fontsize=14)
    ax.set_title('Overfitting Analysis (Val Loss - Train Loss)', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(valid_data['epoch'])
    
    # Add text explanation
    ax.text(0.02, 0.98, 'Green: Good generalization\nRed: Possible overfitting', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'loss_difference_analysis.png')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved loss difference plot: {output_path}")
    plt.close()

def create_summary_stats(df, output_dir):
    """Create and save summary statistics."""
    
    stats = {
        'Total Epochs': len(df),
        'Final Training Loss': df['train_loss'].iloc[-1],
        'Best Training Loss': df['train_loss'].min(),
        'Best Training Epoch': df.loc[df['train_loss'].idxmin(), 'epoch'],
        'Training Loss Improvement': df['train_loss'].iloc[0] - df['train_loss'].iloc[-1],
        'Training Loss Reduction %': ((df['train_loss'].iloc[0] - df['train_loss'].iloc[-1]) / df['train_loss'].iloc[0]) * 100
    }
    
    if not df['val_loss'].isna().all():
        valid_data = df.dropna(subset=['val_loss'])
        stats.update({
            'Final Validation Loss': valid_data['val_loss'].iloc[-1],
            'Best Validation Loss': valid_data['val_loss'].min(),
            'Best Validation Epoch': valid_data.loc[valid_data['val_loss'].idxmin(), 'epoch'],
            'Validation Loss Improvement': valid_data['val_loss'].iloc[0] - valid_data['val_loss'].iloc[-1],
            'Validation Loss Reduction %': ((valid_data['val_loss'].iloc[0] - valid_data['val_loss'].iloc[-1]) / valid_data['val_loss'].iloc[0]) * 100
        })
    
    # Save to file
    stats_path = os.path.join(output_dir, 'training_summary.txt')
    with open(stats_path, 'w') as f:
        f.write("TRAINING SUMMARY STATISTICS\n")
        f.write("=" * 50 + "\n\n")
        
        for key, value in stats.items():
            if isinstance(value, float):
                if 'Epoch' in key:
                    f.write(f"{key}: {value:.0f}\n")
                else:
                    f.write(f"{key}: {value:.6f}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    print(f"Saved training summary: {stats_path}")
    
    # Print to console
    print("\nTraining Summary:")
    print("-" * 40)
    for key, value in stats.items():
        if isinstance(value, float):
            if 'Epoch' in key:
                print(f"{key}: {value:.0f}")
            elif '%' in key:
                print(f"{key}: {value:.2f}%")
            else:
                print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

def main():
    args = parse_args()
    
    # Setup
    plt.style.use(args.style)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading losses from: {args.csv_path}")
    
    # Load data
    try:
        df = load_losses(args.csv_path)
    except Exception as e:
        print(f"Error: {e}")
        return
    
    print(f"Creating plots in: {args.output_dir}")
    print(f"Using style: {args.style}")
    
    # Create plots
    fig_combined = plot_losses_combined(df, args.output_dir, args.figsize, args.dpi)
    plot_losses_separate(df, args.output_dir, args.figsize, args.dpi)
    plot_loss_difference(df, args.output_dir, args.figsize, args.dpi)
    
    # Create summary statistics
    create_summary_stats(df, args.output_dir)
    
    print(f"\nCompleted! Generated plots saved in: {args.output_dir}")
    
    # Show plots if requested
    if args.show_plots:
        plt.show()
    else:
        plt.close(fig_combined)

if __name__ == "__main__":
    main()