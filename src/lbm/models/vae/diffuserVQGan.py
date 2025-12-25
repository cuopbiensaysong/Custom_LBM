import torch
import torch.nn as nn
from .vqgan import VQModel as TamingVQModel

class DiffusersVQGANWrapper(nn.Module):
    """
    Adapter class to make a taming-transformers VQGAN compatible with 
    diffusers pipelines used in LBM.
    """
    def __init__(self, taming_config_path, taming_ckpt_path):
        super().__init__()
        from omegaconf import OmegaConf
        
        # Load the taming model
        config = OmegaConf.load(taming_config_path)
        self.model = TamingVQModel(**config.model.params)
        
        # Load weights
        state_dict = torch.load(taming_ckpt_path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        self.model.load_state_dict(state_dict, strict=False)
        
        # Freeze the VQGAN (LBM trains the UNet, not the VAE)
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
            
        # Define the scaling factor (critical for diffusion)
        # See Section 4.4 for how to calculate this value
        self.config = type('Config', (), {'scaling_factor': 1.0})() # TODO: check if this is correct

    def encode(self, x):
        """
        Adapts taming encode to diffusers format.
        x: Input image tensor, typically normalized to [-1, 1]
        """
        # 1. Forward pass through encoder
        h = self.model.encoder(x)
        h = self.model.quant_conv(h)
        
        # 2. Return a Distribution-like object
        # LBM script expects: vae.encode(x).latent_dist.sample()
        return TamingDistribution(h)

    def decode(self, z, return_dict=True):
        """
        Adapts taming decode to diffusers format.
        z: Latent tensor
        """
        # 1. Post-quantization convolution
        quant = self.model.post_quant_conv(z)
        
        # 2. Decoder
        dec = self.model.decoder(quant)
        
        if return_dict:
            # Return object with.sample attribute
            return type('Output', (), {'sample': dec})()
        return (dec,)

class TamingDistribution:
    """
    Mock distribution for VQGAN.
    Since VQGAN is deterministic (mostly), sample() just returns the mean.
    """
    def __init__(self, parameters):
        self.parameters = parameters
        self.mean = parameters
        self.std = torch.zeros_like(parameters) 

    def sample(self, generator=None):
        return self.parameters