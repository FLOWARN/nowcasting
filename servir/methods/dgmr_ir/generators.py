import logging
from typing import List

import einops
import torch
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from torch.nn.modules.pixelshuffle import PixelShuffle
from torch.nn.utils.parametrizations import spectral_norm

from servir.methods.dgmr_ir.common import GBlock, UpsampleGBlock
from servir.methods.dgmr_ir.layers import ConvGRU

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)


class Sampler(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        forecast_steps: int = 18,
        latent_channels: int = 768,
        context_channels: int = 384,
        output_channels: int = 1,
        sample_ir = False,
        **kwargs
    ):
        """
        Sampler from the Skillful Nowcasting, see https://arxiv.org/pdf/2104.00954.pdf

        The sampler takes the output from the Latent and Context conditioning stacks and
        creates one stack of ConvGRU layers per future timestep.

        Args:
            forecast_steps: Number of forecast steps
            latent_channels: Number of input channels to the lowest ConvGRU layer
        """
        super().__init__()
        config = locals()
        config.pop("__class__")
        config.pop("self")
        self.config = kwargs.get("config", config)
        self.forecast_steps = self.config["forecast_steps"]
        latent_channels = self.config["latent_channels"]
        context_channels = self.config["context_channels"]
        output_channels = self.config["output_channels"]
        self.sample_ir = sample_ir
        
        self.convGRU1 = ConvGRU(
            input_channels=latent_channels + context_channels,
            output_channels=context_channels,
            kernel_size=3,
        )
        
        self.gru_conv_1x1 = spectral_norm(
            torch.nn.Conv2d(
                in_channels=context_channels, out_channels=latent_channels, kernel_size=(1, 1)
            )
        )
        self.g1 = GBlock(input_channels=latent_channels, output_channels=latent_channels)
        self.up_g1 = UpsampleGBlock(
            input_channels=latent_channels, output_channels=latent_channels // 2
        )

        self.convGRU2 = ConvGRU(
            input_channels=latent_channels // 2 + context_channels // 2,
            output_channels=context_channels // 2,
            kernel_size=3,
        )
        self.gru_conv_1x1_2 = spectral_norm(
            torch.nn.Conv2d(
                in_channels=context_channels // 2,
                out_channels=latent_channels // 2,
                kernel_size=(1, 1),
            )
        )
        self.g2 = GBlock(input_channels=latent_channels // 2, output_channels=latent_channels // 2)
        self.up_g2 = UpsampleGBlock(
            input_channels=latent_channels // 2, output_channels=latent_channels // 4
        )

        self.convGRU3 = ConvGRU(
            input_channels=latent_channels // 4 + context_channels // 4,
            output_channels=context_channels // 4,
            kernel_size=3,
        )
        self.gru_conv_1x1_3 = spectral_norm(
            torch.nn.Conv2d(
                in_channels=context_channels // 4,
                out_channels=latent_channels // 4,
                kernel_size=(1, 1),
            )
        )
        self.g3 = GBlock(input_channels=latent_channels // 4, output_channels=latent_channels // 4)
        self.up_g3 = UpsampleGBlock(
            input_channels=latent_channels // 4, output_channels=latent_channels // 8
        )

        self.convGRU4 = ConvGRU(
            input_channels=latent_channels // 8 + context_channels // 8,
            output_channels=context_channels // 8,
            kernel_size=3,
        )
        self.gru_conv_1x1_4 = spectral_norm(
            torch.nn.Conv2d(
                in_channels=context_channels // 8,
                out_channels=latent_channels // 8,
                kernel_size=(1, 1),
            )
        )
        self.g4 = GBlock(input_channels=latent_channels // 8, output_channels=latent_channels // 8)
        self.up_g4 = UpsampleGBlock(
            input_channels=latent_channels // 8, output_channels=latent_channels // 16
        )

        self.bn = torch.nn.BatchNorm2d(latent_channels // 16)
        self.relu = torch.nn.ReLU()
        self.conv_1x1 = spectral_norm(
            torch.nn.Conv2d(
                in_channels=latent_channels // 16,
                out_channels=4 * output_channels,
                kernel_size=(1, 1),
            )
        )

        self.depth2space = PixelShuffle(upscale_factor=2)
        
        if self.sample_ir:
        
            self.convGRU1IR = ConvGRU(
                input_channels=latent_channels + context_channels,
                output_channels=context_channels,
                kernel_size=3,
            )
            
            self.gru_conv_1x1IR = spectral_norm(
                torch.nn.Conv2d(
                    in_channels=context_channels, out_channels=latent_channels, kernel_size=(1, 1)
                )
            )
            self.g1IR = GBlock(input_channels=latent_channels, output_channels=latent_channels)
            self.up_g1IR = UpsampleGBlock(
                input_channels=latent_channels, output_channels=latent_channels // 2
            )

            self.convGRU2IR = ConvGRU(
                input_channels=latent_channels // 2 + context_channels // 2,
                output_channels=context_channels // 2,
                kernel_size=3,
            )
            self.gru_conv_1x1_2IR = spectral_norm(
                torch.nn.Conv2d(
                    in_channels=context_channels // 2,
                    out_channels=latent_channels // 2,
                    kernel_size=(1, 1),
                )
            )
            self.g2IR = GBlock(input_channels=latent_channels // 2, output_channels=latent_channels // 2)
            self.up_g2IR = UpsampleGBlock(
                input_channels=latent_channels // 2, output_channels=latent_channels // 4
            )

            self.convGRU3IR = ConvGRU(
                input_channels=latent_channels // 4 + context_channels // 4,
                output_channels=context_channels // 4,
                kernel_size=3,
            )
            self.gru_conv_1x1_3IR = spectral_norm(
                torch.nn.Conv2d(
                    in_channels=context_channels // 4,
                    out_channels=latent_channels // 4,
                    kernel_size=(1, 1),
                )
            )
            self.g3IR = GBlock(input_channels=latent_channels // 4, output_channels=latent_channels // 4)
            self.up_g3IR = UpsampleGBlock(
                input_channels=latent_channels // 4, output_channels=latent_channels // 8
            )

            self.convGRU4IR = ConvGRU(
                input_channels=latent_channels // 8 + context_channels // 8,
                output_channels=context_channels // 8,
                kernel_size=3,
            )
            self.gru_conv_1x1_4IR = spectral_norm(
                torch.nn.Conv2d(
                    in_channels=context_channels // 8,
                    out_channels=latent_channels // 8,
                    kernel_size=(1, 1),
                )
            )
            self.g4IR = GBlock(input_channels=latent_channels // 8, output_channels=latent_channels // 8)
            self.up_g4IR = UpsampleGBlock(
                input_channels=latent_channels // 8, output_channels=latent_channels // 16
            )

            self.bnIR = torch.nn.BatchNorm2d(latent_channels // 16)
            self.reluIR = torch.nn.ReLU()
            self.conv_1x1IR = spectral_norm(
                torch.nn.Conv2d(
                    in_channels=latent_channels // 16,
                    out_channels=4 * output_channels,
                    kernel_size=(1, 1),
                )
            )

            self.depth2spaceIR = PixelShuffle(upscale_factor=2)
            
        
    def forward(
        self, conditioning_states: List[torch.Tensor],
        conditioning_states_ir: List[torch.Tensor],
        latent_dim: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform the sampling from Skillful Nowcasting with GANs
        Args:
            conditioning_states: Outputs from the `ContextConditioningStack` with the 4 input states, ordered from largest to smallest spatially
            conditioning_states_ir: Outputs from the `ContextConditioningStack` for IR images with the 4 input states, ordered from largest to smallest spatially
            latent_dim: Output from `LatentConditioningStack` for input into the ConvGRUs

        Returns:
            forecast_steps-length output of images for future timesteps

        """
        # Iterate through each forecast step
        # Initialize with conditioning state for first one, output for second one
        #  each init state of shape [batch size, 24, 16, 16]
        init_states = conditioning_states
        init_states_ir = conditioning_states_ir
        latent_dim_copy = latent_dim.clone()
        
        # Expand latent dim to match batch size
        latent_dim = einops.repeat(
            latent_dim, "b c h w -> (repeat b) c h w", repeat=init_states[0].shape[0]
        )
        
        if self.sample_ir:
            # Expand latent dim to match batch size
            latent_dim_ir = einops.repeat(
                latent_dim_copy, "b c h w -> (repeat b) c h w", repeat=init_states_ir[0].shape[0]
            )
        
        # list of [batch size, context channels, 2, 2]
        hidden_states = [latent_dim] * self.forecast_steps
        if self.sample_ir:
            hidden_states_ir = [latent_dim_ir] * self.forecast_steps
        
    
        # started [batch size, 384, 2, 2]
        # Layer 4 (bottom most)
        # output shape [batch size, 192, 2, 2]
        # ConvGRU(input_channels=latent_channels + context_channels,output_channels=context_channels,kernel_size=3,)
        hidden_states = self.convGRU1(hidden_states, init_states[3])
        if self.sample_ir:
            hidden_states_ir = self.convGRU1IR(hidden_states_ir, init_states_ir[3])
        
        # output shape [batch size, 384, 2, 2]
        # torch.nn.Conv2d(in_channels=context_channels, out_channels=latent_channels, kernel_size=(1, 1))
        hidden_states = [self.gru_conv_1x1(h) for h in hidden_states]
        if self.sample_ir:
            hidden_states_ir = [self.gru_conv_1x1IR(h) for h in hidden_states_ir]
        
        # output shape [batch size, 384, 2 , 2]
        # GBlock(input_channels=latent_channels, output_channels=latent_channels)
        hidden_states = [self.g1(h) for h in hidden_states]
        if self.sample_ir:
            hidden_states_ir = [self.g1IR(h) for h in hidden_states_ir]
        
        # output shape [batch size, 192, 4, 4]
        # UpsampleGBlock(input_channels=latent_channels, output_channels=latent_channels // 2)
        hidden_states = [self.up_g1(h) for h in hidden_states]
        if self.sample_ir:
            hidden_states_ir = [self.up_g1IR(h) for h in hidden_states_ir]
        
        # Layer 3. overall output shape [batch size, 96, 8, 8]
        hidden_states = self.convGRU2(hidden_states, init_states[2])
        if self.sample_ir:
            hidden_states_ir = self.convGRU2IR(hidden_states_ir, init_states_ir[2])
        
        hidden_states = [self.gru_conv_1x1_2(h) for h in hidden_states]
        if self.sample_ir:
            hidden_states_ir = [self.gru_conv_1x1_2IR(h) for h in hidden_states_ir]
        
        hidden_states = [self.g2(h) for h in hidden_states]
        if self.sample_ir:
            hidden_states_ir = [self.g2IR(h) for h in hidden_states_ir]
        
        hidden_states = [self.up_g2(h) for h in hidden_states]
        if self.sample_ir:
            hidden_states_ir = [self.up_g2IR(h) for h in hidden_states_ir]

        # Layer 2. overall output shape [batch size, 48, 16, 16]
        hidden_states = self.convGRU3(hidden_states, init_states[1])
        if self.sample_ir:
            hidden_states_ir = self.convGRU3(hidden_states_ir, init_states_ir[1])
        
        hidden_states = [self.gru_conv_1x1_3(h) for h in hidden_states]
        if self.sample_ir:
            hidden_states_ir = [self.gru_conv_1x1_3IR(h) for h in hidden_states_ir]
        
        hidden_states = [self.g3(h) for h in hidden_states]
        if self.sample_ir:
            hidden_states_ir = [self.g3IR(h) for h in hidden_states_ir]
        
        hidden_states = [self.up_g3(h) for h in hidden_states]
        if self.sample_ir:
            hidden_states_ir = [self.up_g3IR(h) for h in hidden_states_ir]
        
        # Layer 1 (top-most). overall output shape [batch size, 24, 32, 32]
        hidden_states = self.convGRU4(hidden_states, init_states[0])
        if self.sample_ir:
            hidden_states_ir = self.convGRU4IR(hidden_states_ir, init_states_ir[0])
        
        hidden_states = [self.gru_conv_1x1_4(h) for h in hidden_states]
        if self.sample_ir:
            hidden_states_ir = [self.gru_conv_1x1_4IR(h) for h in hidden_states_ir]
        
        hidden_states = [self.g4(h) for h in hidden_states]
        if self.sample_ir:
            hidden_states_ir = [self.g4IR(h) for h in hidden_states_ir]
        
        hidden_states = [self.up_g4(h) for h in hidden_states]
        if self.sample_ir:
            hidden_states_ir = [self.up_g4IR(h) for h in hidden_states_ir]

        # Output layer. overall output shape: [batch size, 1, 64, 64]
        hidden_states = [F.relu(self.bn(h)) for h in hidden_states]
        if self.sample_ir:
            hidden_states_ir = [F.relu(self.bnIR(h)) for h in hidden_states_ir]
        
        hidden_states = [self.conv_1x1(h) for h in hidden_states]
        if self.sample_ir:
            hidden_states_ir = [self.conv_1x1IR(h) for h in hidden_states_ir]
        
        hidden_states = [self.depth2space(h) for h in hidden_states]
        if self.sample_ir:
            hidden_states_ir = [self.depth2spaceIR(h) for h in hidden_states_ir]
        

        # Convert forecasts to a torch Tensor
        # output shape: [batch size, 12, 1, 64, 64]
        forecasts = torch.stack(hidden_states, dim=1)
        if self.sample_ir:
            forecastsIR = torch.stack(hidden_states_ir, dim=1)
        
        return  forecasts


class Generator(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        conditioning_stack: torch.nn.Module,
        conditioning_stack_ir: torch.nn.Module,
        latent_stack: torch.nn.Module,
        sampler: torch.nn.Module,
    ):
        """
        Wraps the three parts of the generator for simpler calling
        Args:
            conditioning_stack:
            latent_stack:
            sampler:
        """
        super().__init__()
        self.conditioning_stack = conditioning_stack
        self.conditioning_stack_ir = conditioning_stack_ir
        self.latent_stack = latent_stack
        self.sampler = sampler
        
        # for resolution 1
        layer_1_1 = torch.nn.Conv2d(48, 36, (1,1), stride=1, padding=0)
        layer_2_1 = torch.nn.Conv2d(36, 24, (1,1), stride=1, padding=0)
        
        self.resolution_1_processor = torch.nn.Sequential(layer_1_1, torch.nn.ReLU(), layer_2_1, torch.nn.ReLU())
        
        # for resolution 2
        layer_1_2 = torch.nn.Conv2d(96, 72, (1,1), stride=1, padding=0)
        layer_2_2 = torch.nn.Conv2d(72, 48, (1,1), stride=1, padding=0)
        
        self.resolution_2_processor = torch.nn.Sequential(layer_1_2, torch.nn.ReLU(), layer_2_2, torch.nn.ReLU())
        
        # for resolution 3
        layer_1_3 = torch.nn.Conv2d(192, 144, (1,1), stride=1, padding=0)
        layer_2_3 = torch.nn.Conv2d(144, 96, (1,1), stride=1, padding=0)
        
        self.resolution_3_processor = torch.nn.Sequential(layer_1_3, torch.nn.ReLU(),layer_2_3, torch.nn.ReLU())
        
        # for resolution 4
        layer_1_4 = torch.nn.Conv2d(384, 288, (1,1), stride=1, padding=0)
        layer_2_4 = torch.nn.Conv2d(288, 192, (1,1), stride=1, padding=0)
        
        self.resolution_4_processor = torch.nn.Sequential(layer_1_4, torch.nn.ReLU(),layer_2_4, torch.nn.ReLU())
        
        self.combining_layers = [self.resolution_1_processor, self.resolution_2_processor, 
                                self.resolution_3_processor, self.resolution_4_processor]
    
    def forward(self, x, x_ir):
        conditioning_states = self.conditioning_stack(torch.cat([x, x_ir], dim=1))
        # conditioning_states_ir = self.conditioning_stack_ir(x_ir)
        
        # modified_conditioning_states = tuple([torch.cat([x, y], axis=1) for x,y in zip(conditioning_states, conditioning_states_ir)])
        # modified_conditioning_states = tuple([torch.add(x,y) for x,y in zip(conditioning_states, conditioning_states_ir)])
        
        # resolution_layer_index = 0
        # modified_conditioning_states = []
        # for x,y in zip(conditioning_states, conditioning_states_ir):
        #     modified_conditioning_states.append(self.combining_layers[resolution_layer_index](torch.cat([x,y], dim=1)))
        #     resolution_layer_index += 1
        
        latent_dim = self.latent_stack(torch.cat([x, x_ir], dim=1))
        
        # print(conditioning_states[0].shape, conditioning_states[1].shape, conditioning_states[2].shape, conditioning_states[3].shape)
        # print("=============")
        # print(conditioning_states_ir[0].shape, conditioning_states_ir[1].shape, conditioning_states_ir[2].shape, conditioning_states_ir[3].shape)
        # print("=============")
        # print(latent_dim.shape)
        # print("=============")
        # print(modified_conditioning_states[0].shape, modified_conditioning_states[1].shape, modified_conditioning_states[2].shape, modified_conditioning_states[3].shape)
        # print("=============")
        # input()
        # Ideas
        # 1. adding
        # 2. for each of the context representations, produce a new representation by passing it through an mlp 
        
        # x = torch.relu(self.sampler(conditioning_states, latent_dim))
        
        
        x = torch.relu(self.sampler(conditioning_states, None, latent_dim))
        # print("processed")
        # input()
        
        return x
