import pandas as pd
# # import matplotlib.pyplot as plt
# # import numpy as np

# # # Load the CSV
# # df = pd.read_csv("outputs/vae_3d_experiment_20250414_173818/iteration_history.csv")

# # # Compute global iteration number
# # df['global_iteration'] = (df['epoch'] - 1) * 13 + df['iteration']  # 2 iterations per epoch

# # # Compute log10 values
# # df['log_loss'] = np.log10(df['total_loss'])
# # df['log_iter'] = np.log10(df['global_iteration'])

# # # Plot
# # plt.figure(figsize=(8, 6))
# # plt.plot(df['log_iter'], df['log_loss'], marker='o', linestyle='-')
# # plt.xlabel("log10(Iteration)")
# # plt.ylabel("log10(Loss)")
# # plt.title("Log-Log Plot of Loss vs Iteration")
# # plt.grid(True)
# # plt.tight_layout()
# # plt.savefig("log_loss_plot.png")


# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# def plot_log_loss_vs_log_iterations(csv_path):
#     """
#     Plot log loss vs log iterations from the given CSV file.

#     Args:
#         csv_path (str): Path to the CSV file containing iteration-level loss data.
#     """
#     # Read the CSV file
#     data = pd.read_csv(csv_path)

#     # Calculate global iterations
#     data['global_iteration'] = (data['epoch'] - 1) * data['iteration'].max() + data['iteration']

#     # Apply log10 transformation
#     data['log_iterations'] = data['global_iteration']
#     data['log_total_loss'] = np.log10(data['total_loss'])

#     # Plot log loss vs log iterations
#     plt.figure(figsize=(16, 12))
#     plt.plot(data['log_iterations'], data['log_total_loss'], label='Log Total Loss', marker='o', linestyle='-')

#     # Add labels, legend, and title
#     plt.xlabel('(Iterations)')
#     plt.ylabel('Log10(Total Loss)')
#     plt.title('Log Loss vs Log Iterations')
#     plt.grid(True)
#     plt.legend()

#     # Show the plot
#     plt.savefig("log_loss_vs_log_iterations.png")import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Path to your training_history.csv
csv_path = "outputs/vae_3d_experiment_20250616_163638/training_history.csv"

# Load the CSV
df = pd.read_csv(csv_path)

# Compute log10 of losses
df['log_total_loss'] = np.log10(df['total_loss'])
df['log_val_loss'] = np.log10(df['val_loss'])
df['log_epoch'] = np.log10(df['epoch'])

# Plot
plt.figure(figsize=(10, 6))
plt.plot(df['log_epoch'], df['log_total_loss'], label='Log10(Train Loss)')
plt.plot(df['log_epoch'], df['log_val_loss'], label='Log10(Val Loss)')
plt.xlabel("Log_Epoch")
plt.ylabel("log10(Loss)")
plt.title("Log10 Loss vs Log10 Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("log_loss_vs_log_epoch.png")
# plt.show()

# # Example usage
# if __name__ == "__main__":
#     csv_path = "outputs/vae_3d_experiment_20250418_005227/iteration_history.csv"
#     plot_log_loss_vs_log_iterations(csv_path)