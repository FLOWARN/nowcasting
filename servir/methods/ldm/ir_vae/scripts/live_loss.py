import pandas as pd
import time
import argparse
import matplotlib.pyplot as plt
import os

def plot_live_loss(csv_path):
    last_modified_time = None  # Track the last modification time of the file

    while True:
        try:
            # Check if the file exists
            if not os.path.exists(csv_path):
                print(f"File not found: {csv_path}. Waiting...")
                time.sleep(3)
                continue

            # Check if the file has been updated
            current_modified_time = os.path.getmtime(csv_path)
            if last_modified_time == current_modified_time:
                # No new data, wait and retry
                print("No new data. Waiting...")
                time.sleep(3)
                continue

            # Update the last modified time
            last_modified_time = current_modified_time

            # Read the CSV file
            data = pd.read_csv(csv_path)

            # Create the plot
            plt.figure(figsize=(10, 6))
            plt.plot(data['epoch'], data['total_loss'], label='Total Loss', marker='o')
            # plt.plot(data['epoch'], data['recon_loss'], label='Reconstruction Loss', marker='o')
            # plt.plot(data['epoch'], data['kl_loss'], label='KL Loss', marker='o')
            plt.plot(data['epoch'], data['val_loss'], label='Validation Loss', marker='o')

            # Add labels, legend, and title
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Live Loss Plot')
            plt.grid()
            plt.legend()

            # Save the plot as a PNG file
            plt.savefig("live_loss_plot.png")
            plt.close()  # Close the plot to free memory
            print("Plot updated and saved as 'live_loss_plot.png'")

            # Wait before checking for updates
            time.sleep(3)
        except KeyboardInterrupt:
            print("Live plotting stopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(30)  # Wait before retrying

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live plot CSV data.")
    parser.add_argument("csv_path", type=str, help="Path to the CSV file.")
    args = parser.parse_args()
    
    plot_live_loss(args.csv_path)