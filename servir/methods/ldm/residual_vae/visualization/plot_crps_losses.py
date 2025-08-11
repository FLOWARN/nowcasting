import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def plot_crps_losses(csv_path=None):
    """
    Plot CRPS loss and validation CRPS loss from CSV file.
    
    Args:
        csv_path: Path to CSV file. If None, will search for crps_generative_history.csv
    """
    if csv_path is None:
        csv_files = glob.glob('experiments_generative_crps/*/crps_generative_history.csv')
        if not csv_files:
            print("No CRPS history CSV files found!")
            return
        csv_path = csv_files[0]
        print(f"Using CSV file: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    epochs = df['epoch']
    crps_loss = df['crps_loss']
    val_crps_loss = df['val_crps_loss']
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(epochs, crps_loss, 'b-', label='Training CRPS Loss', linewidth=2)
    plt.plot(epochs, val_crps_loss, 'r-', label='Validation CRPS Loss', linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('CRPS Loss')
    plt.title('CRPS Loss vs Validation CRPS Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = 'crps_losses_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    plt.show()
    
    print(f"Training CRPS Loss - Min: {crps_loss.min():.6f}, Final: {crps_loss.iloc[-1]:.6f}")
    print(f"Validation CRPS Loss - Min: {val_crps_loss.min():.6f}, Final: {val_crps_loss.iloc[-1]:.6f}")

if __name__ == "__main__":
    plot_crps_losses()