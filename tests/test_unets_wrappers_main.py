from typing import Optional, List
from diffusers.models import UNet2DConditionModel
from lbm.models.unets import DiffusersUNet2DCondWrapper
import torch 
from diffusers import FlowMatchEulerDiscreteScheduler, StableDiffusionXLPipeline
from lbm.trainer.utils import StateDictAdapter

from lbm.models.embedders import (
    ConditionerWrapper,
    LatentsConcatEmbedder,
    LatentsConcatEmbedderConfig,
)

backbone_signature = "stabilityai/stable-diffusion-xl-base-1.0"
vae_num_channels = 4
unet_input_channels = 4
timestep_sampling = "log_normal"
selected_timesteps = None
prob = None
conditioning_images_keys = []
conditioning_masks_keys = []
source_key = "source_image"
target_key = "source_image_paste"
mask_key = "mask"
bridge_noise_sigma = 0.0
logit_mean = 0.0
logit_std = 1.0

conditioners = []

pipe = StableDiffusionXLPipeline.from_pretrained(
        backbone_signature,
        torch_dtype=torch.bfloat16,
    )

denoiser = DiffusersUNet2DCondWrapper(
        in_channels=unet_input_channels,  # Add downsampled_image
        out_channels=vae_num_channels,
        center_input_sample=False, # ?
        flip_sin_to_cos=True, # ?
        freq_shift=0, # ?
        down_block_types=[
            "DownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
        ],
        mid_block_type="UNetMidBlock2DCrossAttn",
        up_block_types=["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"],
        only_cross_attention=False,
        block_out_channels=[320, 640, 1280], # ?
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
    ).to(torch.bfloat16)

state_dict = pipe.unet.state_dict()

del state_dict["add_embedding.linear_1.weight"]
del state_dict["add_embedding.linear_1.bias"]
del state_dict["add_embedding.linear_2.weight"]
del state_dict["add_embedding.linear_2.bias"]

# Adapt the shapes
state_dict_adapter = StateDictAdapter()
state_dict = state_dict_adapter(
    model_state_dict=denoiser.state_dict(),
    checkpoint_state_dict=state_dict,
    regex_keys=[
        r"class_embedding.linear_\d+.(weight|bias)",
        r"conv_in.weight",
        r"(down_blocks|up_blocks)\.\d+\.attentions\.\d+\.transformer_blocks\.\d+\.attn\d+\.(to_k|to_v)\.weight",
        r"mid_block\.attentions\.\d+\.transformer_blocks\.\d+\.attn\d+\.(to_k|to_v)\.weight",
    ],
    strategy="zeros",
)

denoiser.load_state_dict(state_dict, strict=True)

del pipe


if conditioning_images_keys != [] or conditioning_masks_keys != []:

    latents_concat_embedder_config = LatentsConcatEmbedderConfig(
        image_keys=conditioning_images_keys,
        mask_keys=conditioning_masks_keys,
    )
    latent_concat_embedder = LatentsConcatEmbedder(latents_concat_embedder_config)
    latent_concat_embedder.freeze()
    conditioners.append(latent_concat_embedder)

# Wrap conditioners and set to device
conditioner = ConditionerWrapper(
    conditioners=conditioners,
)
