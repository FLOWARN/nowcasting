# import os
# import torch
# import argparse
# from src.utils.config import load_config
# from src.utils.logger import setup_logging
# from src.models.vae import SimpleVAE3D
# from src.data.data_provider import IMERGDataModule
# from src.data.visualization import VAEVisualizer
# from torch.utils.data import Subset
# from torch.utils.data import DataLoader


# # from src.data.dataset import Random3DDataset
# def main():
#     # Setup argument parser
#     parser = argparse.ArgumentParser(description='Run post-training visualizations for 3D VAE')
#     parser.add_argument('--model_path', type=str, required=True,
#                        help='Path to saved model checkpoint')
#     parser.add_argument('--config', type=str, default='configs/train_config.yaml',
#                        help='Path to config file used for training')
#     parser.add_argument('--output_dir', type=str, default='outputs/post_training_analysis',
#                        help='Directory to save visualizations')
#     parser.add_argument('--num_samples', type=int, default=4,  # Changed to match batch size
#                        help='Number of samples to use for visualization')
#     parser.add_argument('--force', action='store_true',
#                        help='Force recreation of existing visualizations')
#     args = parser.parse_args()

#     # Setup logging
#     setup_logging(args.output_dir)
    
#     # Load config
#     config = load_config(args.config)
        
                
#     #IR data 

#         # event id for which the data was downloaded
#     event_id = 'WA'

#     # h5_dataset_location = '../ldm_data_loader/imerg_data.h5'

#         # as of now, we do not have IR data, so we set it None
#     ir_h5_dataset_location = '../ldm_data_loader/filled_missing_nan_ir_data.h5'

#         # this string is used to determine the kind of dataloader we need to use
#         # for processing individual events, we reccommend the user to keep this fixed
#     dataset_type = 'wa_ir'


#     data_provider =  IMERGDataModule(
#                 forecast_steps = 12,
#                 history_steps = 12,
                
#                 ir_filename = ir_h5_dataset_location,
#                 batch_size = 4,
#                 image_shape = (360, 516),
#                 normalize_data=False,
#                 dataset = dataset_type,
#                 production_mode = False,
#                 )


#     # train_data_loader = data_provider.train_dataloader()
#     # test_data_loader = data_provider.test_dataloader()
#     test_data_loader = data_provider.test_dataloader()
 
         
        
#     # train_dataset = data_provider.train_dataset  # Assuming `train_dataset` is accessible from `data_provider`
#     # train_subset = Subset(train_dataset, range(100))  # Use the first 100 samples or create as per your requirement
#     #     # Create DataLoader for the subset
#     # train_data_loader = DataLoader(
#     #         train_subset,
#     #         batch_size=2,
#     #         shuffle=False,
#     #         num_workers=90,
#     #         pin_memory=config['data']['pin_memory'],
#     #         persistent_workers=config['data']['persistent_workers']
#     #     )
        
#     # test_dataset = data_provider.test_dataset  # Assuming `val_dataset` is accessible from `data_provider`
#     # test_subset = Subset(test_dataset, range(100))  # Use the first 100 samples or create as per your requirement
#     #     # Create DataLoader for the validation subset
#     # test_data_loader = DataLoader(
#     #         test_subset,
#     #         batch_size=2,
#     #         shuffle=False,
#     #         num_workers=90,
#     #         pin_memory=config['data']['pin_memory'],
#     #         persistent_workers=config['data']['persistent_workers']
#     #     )
        
        
     
     
     
#     # # # Create dataset
#     # dataset = Random3DDataset(
#     #     num_samples=args.num_samples,
#     #     shape=config['data']['input_shape']
#     # )
    
#     # # Create dataloader
#     # train_dataloader = torch.utils.data.DataLoader(
#     #     dataset,
#     #     batch_size=6,
#     #     shuffle=True
#     # )
    
    
#     # Initialize visualizer
#     visualizer = VAEVisualizer(
#         experiment_dir=args.output_dir,
#         config=config,
        
#     )
    
#     # Run all visualizations
#     try:
#         print("\n" + "="*50)
#         print("Running post-training visualizations")
#         print(f"Model: {args.model_path}")
#         print(f"Output directory: {args.output_dir}")
#         print("="*50 + "\n")
        
#         results = visualizer.load_and_visualize(
#             model_path=args.model_path,
#             dataloader=test_data_loader,
#             device="cuda",
#             force=args.force
#         )
        
#         print("\nVisualization results:")
#         for name, path in results.items():
#             if path:
#                 print(f"- {name}: {path}")
        
#         print("\nPost-training visualization completed successfully!")
        
#     except Exception as e:
#         print(f"\nError during visualization: {str(e)}")
#         raise

# if __name__ == "__main__":
#     main()








import os
import torch
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.utils.config import load_config
from src.utils.logger import setup_logging
from src.models.vae import SimpleVAE3D
from src.data.data_provider import IMERGDataModule


def save_vae_comparisons(
    model, dataloader, output_dir, mean, std, device="cuda", samples_per_file=4, max_samples=20
):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    total_processed = 0
    file_counter = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating Reconstructions"):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]

            batch = batch.to(device).float()  # (B, C, T, H, W)
            batch = batch.permute(0, 1, 4, 2, 3)  # -> (B, C, H, T, W)

            # Normalize using loaded mean/std
            batch_norm = (batch - mean) / std

            # Run model
            recon, _, _ = model(batch_norm)

            for i in range(batch.size(0)):
                if max_samples and total_processed >= max_samples:
                    break

                if total_processed % samples_per_file == 0:
                    if total_processed > 0:
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_dir, f"comparison_{file_counter:03d}.png"), dpi=150)
                        plt.close()
                    file_counter += 1
                    fig, axes = plt.subplots(samples_per_file, 2, figsize=(10, 4 * samples_per_file))

                subplot_idx = total_processed % samples_per_file
                ax = axes if samples_per_file == 1 else axes[subplot_idx]
                mid_slice = batch.shape[2] // 2

                # Denormalize and clip to [0, 1]
                original = batch[i, 0, mid_slice].cpu().numpy()
                original = (original - original.min()) / (original.max() - original.min() + 1e-8)

                recon_image = recon[i, 0, mid_slice].cpu().numpy() * std + mean
                recon_image = (recon_image - recon_image.min()) / (recon_image.max() - recon_image.min() + 1e-8)

                ax[0].imshow(original, cmap='viridis')
                ax[0].set_title(f"Original Sample {total_processed}")
                ax[0].axis('off')

                ax[1].imshow(recon_image, cmap='viridis')
                ax[1].set_title("Reconstruction")
                ax[1].axis('off')

                total_processed += 1

        if total_processed % samples_per_file != 0:
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"comparison_{file_counter:03d}.png"), dpi=150)
            plt.close()


def main():
    parser = argparse.ArgumentParser(description='3D VAE Reconstruction Visualizer')
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved VAE checkpoint')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml', help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='outputs/post_training_analysis', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=20, help='Max samples to visualize')
    parser.add_argument('--force', action='store_true', help='Force overwrite')
    args = parser.parse_args()

    setup_logging(args.output_dir)
    config = load_config(args.config)

    # Setup dataloader
    ir_h5_dataset_location = '../ldm_data_loader/filled_missing_nan_ir_data.h5'
    data_provider = IMERGDataModule(
        forecast_steps=12,
        history_steps=12,
        ir_filename=ir_h5_dataset_location,
        batch_size=4,
        image_shape=(360, 516),
        normalize_data=False,
        dataset='wa_ir',
        production_mode=False,
    )
    test_loader = data_provider.test_dataloader()

    # Load model and checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleVAE3D(
                input_channels=config.get('data', {}).get('input_shape', [1])[0],
                latent_dim=config.get('model', {}).get('latent_dim', 32)
            ).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    mean = checkpoint.get('mean', 0.0)
    std = checkpoint.get('std', 1.0)
    if isinstance(mean, torch.Tensor):
        mean = mean.to(device)
    if isinstance(std, torch.Tensor):
        std = std.to(device)

    print(f"\nLoaded checkpoint from {args.model_path}")
    print(f"Mean: {mean}, Std: {std}\n")

    # Save comparisons
    save_vae_comparisons(
        model=model,
        dataloader=test_loader,
        output_dir=args.output_dir,
        mean=mean,
        std=std,
        device=device,
        samples_per_file=4,
        max_samples=10
    )

    print("\nVisual comparison images saved successfully!")


if __name__ == "__main__":
    main()
