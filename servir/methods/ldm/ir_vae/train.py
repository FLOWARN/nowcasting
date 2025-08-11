import argparse
from src.utils.config import load_config
from src.utils.logger import setup_logging
# from src.data.dataset import Random3DDataset
from src.models.vae import SimpleVAE3D
from src.training.trainer import VAETrainer
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import logging
from src.data.data_provider import IMERGDataModule
from torch.utils.data import Subset
from tqdm import tqdm

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='3D VAE Training Script')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                      help='Path to configuration file')
    args = parser.parse_args()
    
    try:
        # Load config
        config = load_config(args.config)
        
        # Ensure output directory exists
        os.makedirs(config['experiment']['output_dir'], exist_ok=True)
        
        # Setup logging
        setup_logging(config['experiment']['output_dir'])
        
      
        # code copied as akshay suggested to use it.
##########################################################

     
        # # Create a subset of the training dataset
        # train_dataset = data_provider.train_dataset  # Assuming `train_dataset` is accessible from `data_provider`
        # train_subset = Subset(train_dataset, range(1000))  # Use the first 100 samples or create as per your requirement
        # # Create DataLoader for the subset
        # train_data_loader = DataLoader(
        #     train_subset,
        #     batch_size=config['data']['batch_size'],
        #     shuffle=False,
        #     num_workers=config['data']['num_workers'],
        #     pin_memory=config['data']['pin_memory'],
        #     persistent_workers=config['data']['persistent_workers']
        # )

            # Example batch from train_data_loader:
        # for batch in train_data_loader:
        #     input_seq, output_seq = batch
        ## To see the shape of the input and output sequences
        ## we need to permute the input.
        
        #     # input_seq=input_seq.permute(0, 2, 1, 3, 4)
        #     print("Input shape:", input_seq.shape)
            
        #     # torch.Size([32, 8, 1, 360, 516])
        #     # print("Output shape:", output_seq.shape) # torch.Size([32, 12, 1, 360, 516])
        #     break
                
        
                
    #IR data 

        # event id for which the data was downloaded
        event_id = 'WA'

        # location of the h5 file that was generated after downloading the data
        # h5_dataset_location = '../ldm_data_loader/imerg_data.h5'

        # as of now, we do not have IR data, so we set it None
        ir_h5_dataset_location = '../ldm_data_loader/filled_missing_nan_ir_data.h5'

        # this string is used to determine the kind of dataloader we need to use
        # for processing individual events, we reccommend the user to keep this fixed
        dataset_type = 'wa_ir'


        data_provider =  IMERGDataModule(
                forecast_steps = 12,
                history_steps = 12,
                
                ir_filename = ir_h5_dataset_location,
                batch_size = 8,
                image_shape = (360, 516),
                normalize_data=False,
                dataset = dataset_type,
                production_mode = False,
                )


        train_data_loader = data_provider.train_dataloader()
        test_data_loader = data_provider.test_dataloader()
        val_data_loader = data_provider.val_dataloader()

        print("Train DataLoader created with", len(train_data_loader.dataset), "samples")
        print("Test DataLoader created with", len(test_data_loader.dataset), "samples")
        print("Validation DataLoader created with", len(val_data_loader.dataset), "samples")
        # for batch in train_data_loader:
                
        # #     print(batch)
        #     # print(batch.shape)
        #     print(batch[0].shape)
        #     print(batch[1].shape)
            
        #     break    
                    
        print("Data loading complete. Proceeding with data mean and model training...")
        
        
        
        
            
        
        # train_dataset = data_provider.train_dataset  # Assuming `train_dataset` is accessible from `data_provider`
        # train_subset = Subset(train_dataset, range(100))  # Use the first 100 samples or create as per your requirement
        # # Create DataLoader for the subset
        # train_data_loader = DataLoader(
        #     train_subset,
        #     batch_size=2,
        #     shuffle=False,
        #     num_workers=40,
        #     pin_memory=config['data']['pin_memory'],
        #     persistent_workers=config['data']['persistent_workers']
        # )
        
        # val_dataset = data_provider.val_dataset  # Assuming `val_dataset` is accessible from `data_provider`
        # val_subset = Subset(val_dataset, range(100))  # Use the first 100 samples or create as per your requirement
        # # Create DataLoader for the validation subset
        # val_data_loader = DataLoader(
        #     val_subset,
        #     batch_size=2,
        #     shuffle=False,
        #     num_workers=40,
        #     pin_memory=config['data']['pin_memory'],
        #     persistent_workers=config['data']['persistent_workers']
        # )
        
        
        
        
        # def compute_mean_std(dataloader):
        #     n_samples = 0
        #     mean = 0.0
        #     var = 0.0
        #     for batch in dataloader:
        #         data = batch[2]  # Adjust index if needed
        #         data = data.float()
        #         batch_samples = data.numel()
        #         batch_mean = data.mean()
        #         batch_var = data.var()
        #         mean += batch_mean * batch_samples
        #         var += batch_var * batch_samples
        #         n_samples += batch_samples
        #     mean /= n_samples
        #     std = (var / n_samples) ** 0.5
        #     return mean.item(), std.item()
        
        
        def compute_mean_std(dataloader, device='cuda'):
            n_pixels = 0
            sum_ = 0.0
            sum_squared = 0.0
            n_samples = 0
            print("Computing mean and std for the dataset...")
            for batch in tqdm(dataloader):
                data = batch[0].to(device).float()  # Use GPU
                sum_ += data.sum().item()  # convert to Python scalar
                sum_squared += (data ** 2).sum().item()
                n_pixels += data.numel()
                n_samples += data.shape[0]
                if n_samples % 4000 == 0:
                    print(f"Processed {n_samples} samples...")
                    break

            mean = sum_ / n_pixels
            std = ((sum_squared / n_pixels) - mean ** 2) ** 0.5

            print(f"Number of samples: {n_samples}")
            return mean, std
        # Usage before training:
        mean, std = compute_mean_std(train_data_loader)
        print(f"Dataset mean: {mean}, std: {std}")
                
        
        
        
        
        
        
        
        # # ####################################################################################################################################
        model = SimpleVAE3D(
            input_channels=config['data']['input_shape'][0],
            latent_dim=config['model']['latent_dim']
        )
        
        print("Starting training with model for IR data:")
      
        # # # # # Train
        trainer = VAETrainer(config, model, train_data_loader,mean,std,val_data_loader)
        history = trainer.train()
        
        
        
    # # Option 2: Resume training
#         trainer = VAETrainer(
#         config, 
#         model, 
#         train_data_loader,
#         mean,
#         std,
#         val_data_loader, 
        
#     #   # use you own path here dont just uncomment and use!!
#         resume_checkpoint="outputs/vae_3d_experiment_20250616_163638/checkpoints/checkpoint_epoch_19.pth"
#         )
# #    #  change yout checkpoint path according to your system here
#         history = trainer.train() 
        
        
               
    # #     # Save final model
        final_model_path = os.path.join(trainer.exp_dir, "final_model.pth")
        torch.save(model.state_dict(), final_model_path)
        logging.info(f"Saved final model to {final_model_path}")
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
