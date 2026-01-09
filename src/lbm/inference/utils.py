import logging
import os
from typing import List, Optional

import torch
import yaml
from diffusers import FlowMatchEulerDiscreteScheduler
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from lbm.models.embedders import (
    ConditionerWrapper,
    LatentsConcatEmbedder,
    LatentsConcatEmbedderConfig,
)
from lbm.models.lbm import LBMConfig, LBMModel
from lbm.models.unets import DiffusersUNet2DCondWrapper
from lbm.models.vae import AutoencoderKLDiffusers, AutoencoderKLDiffusersConfig


def get_model(
    model_dir: str,
    save_dir: Optional[str] = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    config_path: Optional[str] = None,
) -> LBMModel:
    """Download or load a model from a directory or single weight file.

    Args:
        model_dir (str): Directory containing config + weights or a path to a single ckpt/safetensors file.
        save_dir (Optional[str]): Local path to save the model if downloading from HF Hub.
        torch_dtype (torch.dtype): Torch dtype to use.
        device (str): Target device for the model.
        config_path (Optional[str]): Explicit path to a config yaml. Required when model_dir is a single file.
    """
    # If model_dir is a remote repo id, download it first
    if not os.path.exists(model_dir):
        local_dir = snapshot_download(
            model_dir,
            local_dir=save_dir,
        )
        model_dir = local_dir

    # Determine if a single file or a directory is passed
    if os.path.isfile(model_dir):
        if config_path is None:
            raise ValueError("config_path must be provided when model_dir is a file.")
        model_root = os.path.dirname(model_dir)
        model_files = [os.path.basename(model_dir)]
    else:
        model_root = model_dir
        model_files = os.listdir(model_dir)

    # Resolve config file
    if config_path is not None:
        yaml_file_path = config_path
    else:
        yaml_candidates = [f for f in model_files if f.endswith(".yaml")]
        if len(yaml_candidates) == 0:
            raise ValueError("No yaml file found in the model directory.")
        yaml_file_path = os.path.join(model_root, yaml_candidates[0])

    # Resolve weight files
    safetensors_files = sorted(
        [f for f in model_files if f.endswith(".safetensors")]
    )
    ckpt_files = sorted([f for f in model_files if f.endswith(".ckpt")])
    if os.path.isfile(model_dir):
        if model_dir.endswith(".safetensors"):
            safetensors_files = [os.path.basename(model_dir)]
            ckpt_files = []
        elif model_dir.endswith(".ckpt"):
            ckpt_files = [os.path.basename(model_dir)]
            safetensors_files = []
        else:
            raise ValueError("Unsupported weight file. Use .ckpt or .safetensors.")

    if len(safetensors_files) == 0 and len(ckpt_files) == 0:
        raise ValueError("No safetensors or ckpt file found in the model directory")

    with open(yaml_file_path, "r") as f:
        config = yaml.safe_load(f)

    model = _get_model_from_config(**config, torch_dtype=torch_dtype)

    if len(safetensors_files) > 0:
        weight_path = os.path.join(model_root, safetensors_files[-1])
        logging.info(f"Loading safetensors file: {weight_path}")
        sd = load_file(weight_path)
        model.load_state_dict(sd, strict=True)
    elif len(ckpt_files) > 0:
        weight_path = os.path.join(model_root, ckpt_files[-1])
        logging.info(f"Loading ckpt file: {weight_path}")
        sd = torch.load(
            weight_path,
            map_location="cpu",
            weights_only=False,  # allow loading TrainingConfig globals (PyTorch 2.6+ default changed)
        )["state_dict"]
        sd = {k[6:]: v for k, v in sd.items() if k.startswith("model.")}
        model.load_state_dict(
            sd,
            strict=True,
        )
    model.to(device).to(torch_dtype)

    model.eval()

    return model


def _get_model_from_config(
    backbone_signature: str = "stabilityai/stable-diffusion-xl-base-1.0",
    vae_num_channels: int = 4,
    unet_input_channels: int = 4,
    timestep_sampling: str = "log_normal",
    selected_timesteps: Optional[List[float]] = None,
    prob: Optional[List[float]] = None,
    conditioning_images_keys: Optional[List[str]] = [],
    conditioning_masks_keys: Optional[List[str]] = [],
    source_key: str = "source_image",
    target_key: str = "source_image_paste",
    bridge_noise_sigma: float = 0.0,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    pixel_loss_type: str = "lpips",
    latent_loss_type: str = "l2",
    latent_loss_weight: float = 1.0,
    pixel_loss_weight: float = 0.0,
    torch_dtype: torch.dtype = torch.bfloat16,
    **kwargs,
):

    conditioners = []

    denoiser = DiffusersUNet2DCondWrapper(
        in_channels=unet_input_channels,  # Add downsampled_image
        out_channels=vae_num_channels,
        center_input_sample=False,
        flip_sin_to_cos=True,
        freq_shift=0,
        down_block_types=[
            "DownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
        ],
        mid_block_type="UNetMidBlock2DCrossAttn",
        up_block_types=["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"],
        only_cross_attention=False,
        block_out_channels=[320, 640, 1280],
        layers_per_block=2,
        downsample_padding=1,
        mid_block_scale_factor=1,
        dropout=0.0,
        act_fn="silu",
        norm_num_groups=32,
        norm_eps=1e-05,
        cross_attention_dim=[320, 640, 1280],
        transformer_layers_per_block=[1, 2, 10],
        reverse_transformer_layers_per_block=None,
        encoder_hid_dim=None,
        encoder_hid_dim_type=None,
        attention_head_dim=[5, 10, 20],
        num_attention_heads=None,
        dual_cross_attention=False,
        use_linear_projection=True,
        class_embed_type=None,
        addition_embed_type=None,
        addition_time_embed_dim=None,
        num_class_embeds=None,
        upcast_attention=None,
        resnet_time_scale_shift="default",
        resnet_skip_time_act=False,
        resnet_out_scale_factor=1.0,
        time_embedding_type="positional",
        time_embedding_dim=None,
        time_embedding_act_fn=None,
        timestep_post_act=None,
        time_cond_proj_dim=None,
        conv_in_kernel=3,
        conv_out_kernel=3,
        projection_class_embeddings_input_dim=None,
        attention_type="default",
        class_embeddings_concat=False,
        mid_block_only_cross_attention=None,
        cross_attention_norm=None,
        addition_embed_type_num_heads=64,
    ).to(torch_dtype)

        # Wrap conditioners and set to device
    conditioner = ConditionerWrapper(
        conditioners=conditioners,
    )
    from lbm.models.vae import VQGANLBMWrapper
    vqgan_config_path = "src/lbm/models/vae/vqgan.yaml"
    vqgan_checkpoint_path = "checkpoints/vqgan/epoch=000135.ckpt"
    vqgan = VQGANLBMWrapper(vqgan_config_path, vqgan_checkpoint_path)
    vqgan.freeze()
    vqgan = vqgan.to(torch_dtype)


    ## Diffusion Model ##
    # Get diffusion model
    config = LBMConfig(
        source_key=source_key,
        target_key=target_key,
        latent_loss_weight=latent_loss_weight,
        latent_loss_type=latent_loss_type,
        pixel_loss_type=pixel_loss_type,
        pixel_loss_weight=pixel_loss_weight,
        timestep_sampling=timestep_sampling,
        logit_mean=logit_mean,
        logit_std=logit_std,
        selected_timesteps=selected_timesteps,
        prob=prob,
        bridge_noise_sigma=bridge_noise_sigma,
    )

    sampling_noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        backbone_signature,
        subfolder="scheduler",
    )

    model = LBMModel(
        config, 
        denoiser=denoiser,
        sampling_noise_scheduler=sampling_noise_scheduler,
        vae=vqgan,
        conditioner=conditioner,
    ).to(torch_dtype)

    return model
