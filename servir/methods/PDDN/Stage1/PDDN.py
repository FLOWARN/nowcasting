# code is written by Rui Wang (email:rwangbp@connect.ust.hk)
import os
import shutil
import tempfile

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from monai.config import print_config
from monai.utils import first, set_determinism
from torch.cuda.amp import GradScaler, autocast
from torch.nn import L1Loss
from tqdm import tqdm

from servir.methods.PDDN.Stage1.generative.inferers import LatentDiffusionInferer
from servir.methods.PDDN.Stage1.generative.losses import PatchAdversarialLoss, PerceptualLoss
from servir.methods.PDDN.Stage1.generative.networks.nets import DiffusionModelUNet, PatchDiscriminator
from servir.methods.PDDN.Stage1.generative.networks.schedulers import DDPMScheduler

from torch.utils.data import TensorDataset, DistributedSampler
from torch.utils.data import DataLoader
import numpy as np
from torchinfo import summary

import torch.multiprocessing as mp
import torch.distributed as dist

# from autoencoder2d.autoencoder import AutoencoderNowcast
from servir.utils.data_provider import IMERGDataModule, IMERGDataModuleLatentDim
from servir.methods.PDDN.Stage1.vae import SimpleVAE3D

print_config()

root_data_dir = "/home/aa3328/data/"
# checkpoint and save image folder
feature = "checkpoints/"
model_save_path = root_data_dir + feature
os.makedirs(model_save_path, exist_ok=True)
val_fig_save_path = root_data_dir + feature
os.makedirs(val_fig_save_path, exist_ok=True)

batch_size = 1
channel = 0  # 0 = Flair
assert channel in [0, 1, 2, 3], "Choose a valid channel"

def load_data():
    # event id for which the data was downloaded
    event_id = 'WA'

    # location of the h5 file that was generated after downloading the data
    h5_dataset_location = root_data_dir+''+str(event_id)+'.h5'

    # as of now, we do not have IR data, so we set it None
    ir_h5_dataset_location = root_data_dir+''+str(event_id)+'_IR.h5'

    # this string is used to determine the kind of dataloader we need to use
    # for processing individual events, we reccommend the user to keep this fixed
    dataset_type = 'wa_ir'


    data_provider =  IMERGDataModule(
            forecast_steps = 12,
            history_steps = 12,
            imerg_filename = h5_dataset_location,
            ir_filename = ir_h5_dataset_location,
            batch_size = batch_size,
            image_shape = (360, 516),
            normalize_data=False,
            dataset = dataset_type)

    # test_data_loader = data_provider.test_dataloader()
    # train_data_loader = data_provider.train_dataloader()
    # val_data_loader = data_provider.val_dataloader()
    return data_provider.train_dataset, data_provider.val_dataset, data_provider.test_dataset


def load_latent_data(rank):
    # event id for which the data was downloaded
    event_id = 'WA'

    # location of the h5 file that was generated after downloading the data
    h5_dataset_location = root_data_dir + '' + str(event_id) + '.h5'

    # as of now, we do not have IR data, so we set it None
    ir_h5_dataset_location = root_data_dir + '' + str(event_id) + '_IR.h5'

    # this string is used to determine the kind of dataloader we need to use
    # for processing individual events, we recommend the user to keep this fixed
    dataset_type = 'wa_ir_latent'

    data_provider = IMERGDataModule(
        forecast_steps=12,
        history_steps=12,
        imerg_filename=h5_dataset_location,
        ir_filename=ir_h5_dataset_location,
        batch_size=batch_size,
        image_shape=(360, 516),
        normalize_data=False,
        dataset=dataset_type)

    # train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    # val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    # test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)

    test_data_loader = data_provider.test_dataloader()
    train_data_loader = data_provider.train_dataloader()
    val_data_loader = data_provider.val_dataloader()

    return train_data_loader, val_data_loader, test_data_loader


def cleanup():
    dist.destroy_process_group()

def setup(rank, world_size, use_gpu = False):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    if use_gpu:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        # using CPU tensors
        dist.init_process_group("gloo", rank=1, world_size=2)

def train_ddp(rank, world_size, use_gpu = False):
    setup(rank, world_size, use_gpu )

    device = torch.device(f"cuda:{rank}")
    print(f"Using {device}")

    # imerg_data_train, imerg_data_val, imerg_data_test = load_data()

    train_loader, val_loader, test_loader = load_latent_data(rank)

    print(f"Data has been loaded")

    # train_dataset = TensorDataset(torch.tensor(imerg_data_train, dtype=torch.float32))
    # val_dataset = TensorDataset(torch.tensor(imerg_data_val, dtype=torch.float32))
    # test_dataset = TensorDataset(torch.tensor(imerg_data_test, dtype=torch.float32))


    # train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=8, persistent_workers=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=8, persistent_workers=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=8, persistent_workers=True)
    #

    # enc = encoder.SimpleConvEncoder()
    # dec = encoder.SimpleConvDecoder()
    
    # autoencoder1 = autoenc.AutoencoderKL(enc, dec)
    
    autoencoder1 = SimpleVAE3D()
    autoencoder1.to(device)

    autoencoder1 = torch.nn.parallel.DistributedDataParallel(autoencoder1, device_ids=[rank], find_unused_parameters=True)
    # checkpoint of 3d autoencoder
    # checkpoints = torch.load('')
    # autoencoder1.module.load_state_dict(checkpoints['state_dict'])

    # enc = encoder.SimpleConvEncoder(in_dim=19)
    # dec = encoder.SimpleConvDecoder(in_dim=19)
    # autoencoder2 = autoenc.AutoencoderKL(enc, dec)
    # autoencoder2.to(device)
    #
    # autoencoder2 = torch.nn.parallel.DistributedDataParallel(autoencoder2, device_ids=[rank], find_unused_parameters=True)

    for param in autoencoder1.parameters():
        param.requires_grad = False

    unet = DiffusionModelUNet(
        with_conditioning=True,
        cross_attention_dim=4,
        spatial_dims=3,
        in_channels=32,
        out_channels=32,
        num_res_blocks=2,
        # num_channels=(128, 256, 256, 256),
        num_channels=(8, 16, 16, 16),
        attention_levels=(True, True, True, True),
        # num_head_channels=(8, 8, 8, 8),
        num_head_channels=(2, 2, 2, 2),
        
    )

    unet.to(device)

    unet = torch.nn.parallel.DistributedDataParallel(unet, device_ids=[rank], find_unused_parameters=True)

    scheduler = DDPMScheduler(num_train_timesteps=4, schedule="scaled_linear_beta", beta_start=0.0001, beta_end=0.02)

    # with torch.no_grad():
    #     with autocast(enabled=True):
    #         check_data = first(train_loader)
    #         z = autoencoder1.module.encode(check_data[0][:, :1, :16].to(device))

    # print(f"Scaling factor set to {1/torch.std(z)}")
    # scale_factor = 1 / torch.std(z)
    scale_factor = 1

    inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

    optimizer_diff = torch.optim.AdamW(params=unet.parameters(), lr=1e-4, weight_decay=1e-5)

    n_epochs = 15000
    val_interval = 5
    epoch_loss_list = []
    autoencoder1.eval()
    scaler = GradScaler()

    first_batch = first(train_loader)



    # z = autoencoder1.module.encode(first_batch[0][:, :1, :16].to(device))
    # z_condition = autoencoder1.module.encode(first_batch[0][:, :1, 16:].to(device))
    # z_condition = first(train_loader)

    first_batch_condition = first_batch[0]
    first_batch_output = first_batch[1]

    print("First batch condition shape", first_batch_condition.shape)
    print("First batch output shape",first_batch_output.shape)

    def validate(epoch):
        autoencoder1.eval()
        unet.eval()
        val_batch = next(iter(val_loader))

        img_data = val_batch[0][:, :, :16].cpu()

        noise = torch.randn_like(z)
        noise = noise.to(device)
        scheduler.set_timesteps(num_inference_steps=1000)
        with torch.no_grad():
            synthetic_images = inferer.sample(
                input_noise=noise, conditioning=img_data.to(device), autoencoder_model_radar=autoencoder1.module, autoencoder_model_wrf=None, diffusion_model=unet.module, scheduler=scheduler
            )

        idx = 0
        img = synthetic_images[idx, 0].detach().cpu().numpy()  # images
        img_gt = img_data[idx, 0].detach().cpu().numpy()
        fig, axs = plt.subplots(nrows=2, ncols=16, figsize=(75, 30))
        for frame in range(32):
            if (frame // 16) == 0:
                ax = axs[frame // 16, frame % 16]
                ax.imshow(img[frame], cmap="gray")
            else:
                ax = axs[frame // 16, frame % 16]
                ax.imshow(img_gt[frame-16], cmap="gray")

        fig.savefig(os.path.join(val_fig_save_path, f"val_comparison_epoch_{epoch}_rank_{rank}.png"))

    for epoch in range(n_epochs):
        unet.train()
        epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in progress_bar:
            images = batch
            batch_condition = images[0].to(device)
            batch_output = images[1].to(device)
            print("batch condition shape", batch_condition.shape)
            print("batch output shape", batch_output.shape)

            optimizer_diff.zero_grad(set_to_none=True)

            with autocast(enabled=True):
                noise = torch.randn_like(first_batch_output).to(device)

                # timesteps = torch.randint(
                #     0, inferer.scheduler.num_train_timesteps, (batch_output.shape[0],), device=batch_output.device
                # ).long()

                timesteps = torch.randint(
                    4, 5, (batch_output.shape[0],), device=batch_output.device
                ).long()

                noise_pred = inferer(
                    inputs=batch_output, condition=batch_condition, autoencoder_model_radar=None, autoencoder_model_wrf=None, diffusion_model=unet.module, noise=noise, timesteps=timesteps
                )

                loss = F.mse_loss(noise_pred.float(), noise.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer_diff)
            scaler.update()

            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss /= world_size

            epoch_loss += loss.item()

            progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
        epoch_loss_list.append(epoch_loss / (step + 1))

        if epoch % val_interval == 0:
            validate(epoch)
            if rank == 0:
                os.makedirs(model_save_path, exist_ok=True)
                torch.save(unet.state_dict(), os.path.join(model_save_path, f"unet_model_epoch_{epoch}.pt"))

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print("World size (GPU)", world_size)
    train_ddp(0, world_size, use_gpu = True)

    # mp.spawn(train_ddp, args=(world_size,), nprocs=2, join=True)
