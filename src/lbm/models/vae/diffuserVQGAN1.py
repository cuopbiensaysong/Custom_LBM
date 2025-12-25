import torch
import torch.nn as nn
from typing import Tuple, Optional, Union, Dict, Any
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.autoencoders.vae import DecoderOutput

# Dynamic import to handle environment variability

from .vqgan import VQModel
from omegaconf import OmegaConf, DictConfig, ListConfig


class DiagonalGaussianDistributionLike:
    """
    A dummy distribution class that adapts deterministic VQGAN outputs to the 
    probabilistic interface expected by LBM/Diffusers.
    
    This class mimics `diffusers.models.autoencoders.vae.DiagonalGaussianDistribution`.
    Since VQGAN is deterministic, the 'sample' is simply the latent tensor itself.
    """
    def __init__(self, parameters: torch.Tensor, deterministic: bool = True):
        self.parameters = parameters
        self.deterministic = deterministic
        # Mimic VAE attributes to prevent AttributeErrors in training loops
        self.mean = parameters 
        self.logvar = torch.zeros_like(parameters) - 30.0 # Effectively zero variance
        self.std = torch.zeros_like(parameters) + 1e-6

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """
        Returns the latent tensor. The generator argument is ignored as the process
        is deterministic (or quantization noise is already applied).
        """
        return self.parameters

    def mode(self) -> torch.Tensor:
        return self.parameters
        
    def kl(self, other=None) -> torch.Tensor:
        # Return zero KL divergence since we are not optimizing a variational bound here
        return torch.tensor(0.0, device=self.parameters.device)

class AutoencoderOutput:
    """
    Mimics `diffusers.models.modeling_outputs.AutoencoderKLOutput`.
    Holds the 'latent_dist' object.
    """
    def __init__(self, latent_dist: DiagonalGaussianDistributionLike):
        self.latent_dist = latent_dist

class VQGANLBMWrapper(ModelMixin, ConfigMixin):
    """
    A robust adapter class integrating a custom 'taming-transformers' VQGAN into
    the gojasper/LBM framework.
    
    This wrapper:
    1. Loads a VQGAN from a.yaml config and.ckpt checkpoint.
    2. Exposes `encode` and `decode` methods matching AutoencoderKL signatures.
    3. Handles 1-channel input adaptation (optional).
    4. Manages latent scaling.
    """
    
    config_name = "config.json" # Required by ConfigMixin

    @register_to_config
    def __init__(
        self,
        vqgan_config_path: str,
        vqgan_checkpoint_path: str = None,
        scaling_factor: float = 1.0, 
        use_quantized_latents: bool = True,
        force_input_channels: Optional[int] = None
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.use_quantized_latents = use_quantized_latents
        self.force_input_channels = force_input_channels
        # Mean: [-0.03646181896328926, 0.015901772305369377, -0.02612009085714817, -0.023668786510825157]
        # Std: [0.6274713277816772, 0.44813016057014465, 0.5477502346038818, 0.392182320356369]



        self.latents_mean = torch.tensor([-0.03646181896328926, 0.015901772305369377, -0.02612009085714817, -0.023668786510825157]) # vector of mean latents for each channel # TODO get from config 
        self.latents_std = torch.tensor([0.6274713277816772, 0.44813016057014465, 0.5477502346038818, 0.392182320356369]) # vector of std latents for each channel # TODO get from config 
        self.latents_mean = self.latents_mean.view(1, -1, 1, 1)
        self.latents_std = self.latents_std.view(1, -1, 1, 1)
        # self.latents_std = torch.tensor(self.latents_std).view(1, -1, 1, 1)
        # Load configuration
        print(f" Loading config from {vqgan_config_path}...")
        try:
            vqgan_config = OmegaConf.load(vqgan_config_path)
        except Exception as e:
            raise ValueError(f"Failed to load VQGAN config: {e}")

        # Initialize the underlying VQModel
        print(" Initializing VQModel...")
        model_params = self._extract_model_params(vqgan_config)
        self.model = VQModel(**model_params)
        
        # Load Pre-trained Weights
        if vqgan_checkpoint_path is not None:
            print(f" Loading checkpoint from {vqgan_checkpoint_path}...")
            sd = torch.load(vqgan_checkpoint_path, map_location="cpu")
            if "state_dict" in sd:
                sd = sd["state_dict"]
            
            # Flexible loading: strict=False allows ignoring discriminator weights
            # or other auxiliary components not needed for pure inference/encoding.
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            print(f" Weights Loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        
        # Freeze the VQGAN
        # In LBM, the autoencoder is a frozen perceptual compression stage.
        self.model.eval()
        self.model.requires_grad_(False)

    def encode(self, x: torch.Tensor, return_dict: bool = True) -> Union[torch.Tensor, tuple]:
        """
        Encodes the input `x` to a latent representation wrapped in a pseudo-distribution.
        Adapts 1-channel inputs if necessary.
        """
        # --- 1-Channel Input Handling Strategy ---
        # If the input is 1-channel (B, 1, H, W) but the model expects 3 (RGB),
        # we replicate the channel dimension.
        expected_channels = self.model.encoder.conv_in.in_channels
        if x.shape[1]!= expected_channels:
            # if x.shape[1] == 1 and expected_channels == 3:
            #     # Replicate grayscale to RGB
            #     x = x.repeat(1, 3, 1, 1)
            # elif self.force_input_channels is not None and x.shape[1]!= self.force_input_channels:
            #      raise ValueError(f"Input channel mismatch: Got {x.shape[1]}, expected {expected_channels} or {self.force_input_channels}")
            raise ValueError(f"Input channel mismatch: Got {x.shape[1]}, expected {expected_channels}")
        # --- VQGAN Encoding Pass ---
        # 1. Convolutional Encoder
        h = self.model.encoder(x)
        # 2. Pre-quantization Convolution (adjusts dimensions for codebook)
        h = self.model.quant_conv(h)
        
        if self.use_quantized_latents:
            # 3. Vector Quantization
            # quantize() returns: (quantized_tensor, embedding_loss, info_tuple)
            quant, _, _ = self.model.quantize(h)
            latent = quant
        else:
            # Return continuous latents (pre-codebook)
            latent = h

        if self.latents_mean is not None and self.latents_std is not None:
            latents = (latents - self.latents_mean) / self.latents_std

        # # --- Interface Adaptation ---
        # # Wrap in the dummy distribution object
        # posterior = DiagonalGaussianDistributionLike(latent)

        # if not return_dict:
        #     return (posterior,)

        # return AutoencoderOutput(posterior)
        return latent

    def decode(self, z: torch.Tensor, return_dict: bool = True) -> Union:
        """
        Decodes the latent `z` back to image space.
        """
        # Note: LBM training loops often scale latents by `scaling_factor` before
        # passing to the UNet, and unscale them before decoding. 
        # Ensure that `z` passed here is unscaled (magnitude matching codebook).
        
        # VQGAN decoder expects post-quantized embeddings.
        # We pass through post_quant_conv to map from latent dim back to decoder dim.

        if self.latents_mean is not None and self.latents_std is not None:
            z = z * self.latents_std + self.latents_mean
        
        quant = self.model.post_quant_conv(z)
        
        # Decoder pass
        dec = self.model.decoder(quant)
        
        # if not return_dict:
        #     return (dec,)
            
        # return DecoderOutput(sample=dec)
        return dec

    @property
    def dtype(self):
        """Helper to expose model dtype."""
        return self.model.encoder.conv_in.weight.dtype

    @property
    def device(self):
        """Helper to expose model device."""
        return self.model.encoder.conv_in.weight.device

    @staticmethod
    def _extract_model_params(config: DictConfig) -> Dict[str, Any]:
        """
        Support both original Hydra config structure with `model.params`
        and a simplified flat structure that lists VQModel arguments at
        the top level (like `ddconfig`, `lossconfig`, ...).
        """

        def _to_container(value):
            if isinstance(value, (DictConfig, ListConfig)):
                return OmegaConf.to_container(value, resolve=True)
            return value

        # Case 1: canonical structure config.model.params.*
        if "model" in config and "params" in config.model:
            params_cfg = config.model.params
            return _to_container(params_cfg)

        # Case 2: flattened structure; collect known parameters directly.
        fallback_keys = {
            "ddconfig",
            "lossconfig",
            "n_embed",
            "embed_dim",
            "ckpt_path",
            "ignore_keys",
            "image_key",
            "colorize_nlabels",
            "monitor",
            "remap",
            "sane_index_shape",
        }
        model_params: Dict[str, Any] = {}
        for key in fallback_keys:
            if key in config:
                model_params[key] = _to_container(config[key])

        required = {"ddconfig", "lossconfig", "n_embed", "embed_dim"}
        missing = sorted(required - model_params.keys())
        if missing:
            raise KeyError(
                "VQGAN config does not contain the required keys for VQModel "
                f"initialization: {', '.join(missing)}"
            )

        return model_params