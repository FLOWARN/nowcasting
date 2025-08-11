import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_training_curves(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found")
        return
    
    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['total_loss'], 'b-', label='Total Loss', linewidth=2)
    plt.plot(df['epoch'], df['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Training curves plotted and saved as 'training_curves.png'")

if __name__ == "__main__":
    csv_path = "experiments/vae_diff_training_20250709_131353/training_history.csv"
    plot_training_curves(csv_path)