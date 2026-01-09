import logging
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from lbm.models.lbm import LBMModel


def load_image(root_dir: str, img_path: str) -> torch.Tensor:
    """Load a numpy image file and convert to torch tensor with a channel dim."""
    full_path = os.path.join(root_dir, img_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Image not found at {full_path}")

    img = np.load(full_path, allow_pickle=True)
    img = np.expand_dims(img, axis=0).astype(np.float32) - 0.8
    img = np.expand_dims(img, axis=0)
    return torch.from_numpy(img)


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Convert a 1xHxW or CxHxW tensor to a PIL image for visualization."""
    if t.dim() == 4:
        t = t[0]
    if t.shape[0] == 1:
        t = t.repeat(3, 1, 1)
    # bring approximate range back to [0, 1] for visualization
    t = torch.clamp(t + 0.8, 0.0, 1.0)
    return ToPILImage()(t.cpu())


def _prepare_lpips_tensor(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Ensure tensor is NCHW with 3 channels for LPIPS."""
    if x.dim() == 2:
        x = x.unsqueeze(0)  # H, W -> 1, H, W
    if x.dim() == 3:
        x = x.unsqueeze(0)  # C, H, W -> 1, C, H, W
    if x.dim() != 4:
        raise ValueError(f"Unexpected tensor shape for LPIPS: {tuple(x.shape)}")
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    elif x.shape[1] != 3:
        raise ValueError(f"LPIPS expects 1 or 3 channels, got {x.shape[1]}")
    return x.to(device)


def evaluate_for_1_image(
    model: LBMModel,
    root_dir: str,
    npy_img_path: str,
    num_sampling_steps: int = 1,
    output_dir: Optional[str] = None,
):
    """Run inference for a single input image and optionally save the output."""
    img_tensor = load_image(root_dir, npy_img_path)
    output_image = inference_step(model, img_tensor, num_sampling_steps)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(npy_img_path))[0]
        output_npy_path = os.path.join(output_dir, f"{base_name}_pred.npy")
        np.save(output_npy_path, output_image[0].detach().cpu().numpy())

        pil_img = _tensor_to_pil(output_image)
        output_png_path = os.path.join(output_dir, f"{base_name}_pred.png")
        pil_img.save(output_png_path)
        logging.info(f"Saved outputs to {output_png_path} and {output_npy_path}")

    return output_image

@torch.no_grad()
def inference_step(model, img_tensor, num_sampling_steps: int = 1):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    batch = {
        model.source_key: img_tensor.to(device=device, dtype=dtype),
    }
    model.vae.to(device=device)
    # model.vae.to(dtype=dtype)
    z_source = model.vae.encode(batch[model.source_key])

    output_image = model.sample(
        z=z_source,
        num_steps=num_sampling_steps,
        conditioner_inputs=batch,
        max_samples=1,
    )

    return output_image

def evaluate_for_test_csv(
    model: LBMModel,
    test_df: pd.DataFrame,
    root_dir: str,
    save_npy_output: bool = False,
    num_sampling_steps: int = 1,
    output_dir: str = None,
    frequent_visualize: int = 10,
):
    """Evaluate the model on a test dataframe and report metrics."""
    try:
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "scikit-image is required for PSNR/SSIM computation."
        ) from exc

    try:
        import lpips
        lpips_fn = lpips.LPIPS(net="vgg").to(next(model.parameters()).device)
    except Exception:
        lpips_fn = None
        logging.warning("lpips not available; LPIPS metric will be skipped.")

    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        fid = FrechetInceptionDistance(feature=64).to(next(model.parameters()).device)
    except Exception:
        fid = None
        logging.warning("torchmetrics FID not available; FID metric will be skipped.")

    metrics: Dict[str, list] = {"psnr": [], "ssim": [], "lpips": []}
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    if save_npy_output:
        os.makedirs(os.path.join(output_dir, "npy"), exist_ok=True)
    if frequent_visualize:
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
        source_path = row["img2D_path_A"]
        target_path = row["img2D_path_B"]

        source_tensor = load_image(root_dir, source_path)
        target_tensor = load_image(root_dir, target_path)



        prediction = inference_step(model, source_tensor, num_sampling_steps=num_sampling_steps)
        # Ensure float32 on CPU for numpy/skimage
        pred = prediction[0].detach().cpu().float()
        target = target_tensor.detach().cpu().float()

        pred_np = pred.squeeze().numpy() + 0.8
        target_np = target.squeeze().numpy() + 0.8

        max_pixel_value = max(np.max(target_np), np.max(pred_np))   
        min_pixel_value = min(np.min(target_np), np.min(pred_np))
        data_range = max_pixel_value - min_pixel_value

        psnr = peak_signal_noise_ratio(target_np, pred_np, data_range=data_range)
        ssim = structural_similarity(
            target_np, pred_np, data_range=data_range, channel_axis=None
        )
        metrics["psnr"].append(psnr)
        metrics["ssim"].append(ssim)

        if lpips_fn is not None:
            lpips_device = next(lpips_fn.parameters()).device
            pred_lpips = _prepare_lpips_tensor(torch.clamp(pred, -1, 1), lpips_device)
            target_lpips = _prepare_lpips_tensor(torch.clamp(target, -1, 1), lpips_device)
            lpips_score = lpips_fn(pred_lpips, target_lpips)
            metrics["lpips"].append(lpips_score.item())

        if fid is not None:
            pred_fid = (torch.clamp((pred + 0.8) / 2, 0.0, 1.0) * 255).byte()
            target_fid = (torch.clamp((target + 0.8) / 2, 0.0, 1.0) * 255).byte()
            # print(pred_fid.shape, target_fid.shape)
            if pred_fid.shape[0] == 1:
                pred_fid = pred_fid.repeat( 3, 1, 1).unsqueeze(0)
            if target_fid.shape[0] == 1:
                target_fid = target_fid.repeat(1, 3, 1, 1)
            fid.update(pred_fid.to(fid.device), real=False)
            fid.update(target_fid.to(fid.device), real=True)

        if save_npy_output and output_dir is not None:
            base_name = os.path.splitext(os.path.basename(source_path))[0]
            
            np.save(os.path.join(output_dir, "npy", f"{base_name}_pred.npy"), pred.numpy())
            if frequent_visualize and (idx % frequent_visualize == 0):
                _tensor_to_pil(pred).save(
                    os.path.join(output_dir, "images", f"{base_name}_pred.png")
                )

    results = {
        "psnr_mean": float(np.mean(metrics["psnr"])) if metrics["psnr"] else None,
        "ssim_mean": float(np.mean(metrics["ssim"])) if metrics["ssim"] else None,
        "lpips_mean": float(np.mean(metrics["lpips"])) if metrics["lpips"] else None,
    }

    if fid is not None:
        results["fid"] = fid.compute().item()

    if output_dir is not None:
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(os.path.join(output_dir, "metrics_per_image.csv"), index=False)
        summary_df = pd.DataFrame([results])
        summary_df.to_csv(os.path.join(output_dir, "metrics_summary.csv"), index=False)
        logging.info(f"Saved metrics to {output_dir}")

    psnr_str = f"{results['psnr_mean']:.4f}" if results["psnr_mean"] is not None else "NA"
    ssim_str = f"{results['ssim_mean']:.4f}" if results["ssim_mean"] is not None else "NA"
    lpips_str = (
        f"{results['lpips_mean']:.4f}" if results["lpips_mean"] is not None else "NA"
    )
    fid_val = results.get("fid")
    fid_str = f"{fid_val:.4f}" if fid_val is not None else "NA"
    logging.info(f"PSNR: {psnr_str}, SSIM: {ssim_str}, LPIPS: {lpips_str}, FID: {fid_str}")

    return results
