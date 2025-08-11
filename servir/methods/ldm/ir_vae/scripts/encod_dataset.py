import torch
import os
import numpy as np
from tqdm import tqdm

from src.models.vae import SimpleVAE3D
from src.data.data_provider import IMERGDataModule

def reparameterize(mu, logvar):
    """Reparameterization trick to sample from N(mu, var)"""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def encode_dataset(model, dataloader, device, output_dir,mean,std):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    all_latents = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Encoding dataset")):
            # Adjust batch index if needed (e.g., batch[0] or batch[2])
            x = batch[2].to(device)
            x = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W] if needed
            x = (x - mean) / (std + 1e-8) 
            mu, logvar = model.encode(x)
            z = reparameterize(mu, logvar)
            latents = z.cpu().numpy()
            all_latents.append(latents)

    all_latents = np.concatenate(all_latents, axis=0)
    np.save(os.path.join(output_dir, "encoded_latents.npy"), all_latents)
    print(f"Saved encoded latents to {os.path.join(output_dir, 'encoded_latents.npy')}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Encode dataset using a trained VAE model (with reparameterization)")
    parser.add_argument('--model-checkpoint', type=str, required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save encoded data')
    args = parser.parse_args()

    # Load config
    from src.utils.config import load_config
    config = load_config(args.config)

    # Setup device
    device = torch.device(config['training']['device'])

    # Load model
    model = SimpleVAE3D(
        input_channels=config['data']['input_shape'][0],
        latent_dim=config['model']['latent_dim']
    )
    checkpoint = torch.load(args.model_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

     # event id for which the data was downloaded
    event_id = 'WA'

        # location of the h5 file that was generated after downloading the data
    h5_dataset_location = '../ldm_data_loader/imerg_data.h5'
    # h5_dataset_location='none'
        # as of now, we do not have IR data, so we set it None
    ir_h5_dataset_location = 'src/data/dataset/WA_IR.h5'

        # this string is used to determine the kind of dataloader we need to use
        # for processing individual events, we reccommend the user to keep this fixed
    dataset_type = 'wa_ir'


    data_provider =  IMERGDataModule(
                forecast_steps = 12,
                history_steps = 12,
                imerg_filename = h5_dataset_location,
                ir_filename = ir_h5_dataset_location,
                batch_size = 8,
                image_shape = (360, 516),
                normalize_data=False,
                dataset = dataset_type,
                production_mode = False,
                )



   
    dataloader = data_provider.train_dataloader()
    def compute_mean_std(dataloader):
            n_samples = 0
            mean = 0.0
            var = 0.0
            for batch in dataloader:
                data = batch[2]  # Adjust index if needed
                data = data.float()
                batch_samples = data.numel()
                batch_mean = data.mean()
                batch_var = data.var()
                mean += batch_mean * batch_samples
                var += batch_var * batch_samples
                n_samples += batch_samples
            mean /= n_samples
            std = (var / n_samples) ** 0.5
            return mean.item(), std.item()

        # Usage before training:
    mean, std = compute_mean_std(dataloader)
    print(f"Dataset mean: {mean}, std: {std}")
    
    # Encode and save
    encode_dataset(model, dataloader, device, args.output_dir,mean,std)

if __name__ == "__main__":
    main()