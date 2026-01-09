import argparse
import logging
import os
import pandas as pd
import torch
import yaml

from lbm.inference import (
    evaluate_for_1_image,
    evaluate_for_test_csv,
    get_model,
)

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--test_csv_path", type=str, default=None)
parser.add_argument("--root_dir", type=str, required=True)
parser.add_argument("--npy_path", type=str, default=None)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--num_inference_steps", type=int, default=1)
parser.add_argument("--ckpt_path", type=str, required=True)
parser.add_argument("--config_path", type=str, required=False)
parser.add_argument("--save_npy_output", type=bool, default=False)

args = parser.parse_args()


def main():
    os.makedirs(args.output_dir, exist_ok=True)

    # Save arguments for reproducibility
    with open(os.path.join(args.output_dir, "args.yaml"), "w") as f:
        yaml.safe_dump(vars(args), f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(
        model_dir=args.ckpt_path,
        config_path=args.config_path,
        torch_dtype=torch.bfloat16,
        device=device,
    )

    if (args.test_csv_path is not None) and (args.npy_path is not None):
        raise ValueError("Either test_csv_path or npy_path must be provided")
    if (args.test_csv_path is None) and (args.npy_path is None):
        raise ValueError("Either test_csv_path or npy_path must be provided")

    if args.test_csv_path is not None:
        test_csv = pd.read_csv(args.test_csv_path)
        evaluate_for_test_csv(
            model=model,
            test_df=test_csv,
            root_dir=args.root_dir,
            save_npy_output=args.save_npy_output,
            num_sampling_steps=args.num_inference_steps,
            output_dir=args.output_dir,
            frequent_visualize=10,
        )
    
    if args.npy_path is not None:
        evaluate_for_1_image(
            model=model,
            root_dir=args.root_dir,
            save_npy_output=args.save_npy_output,
            npy_img_path=args.npy_path,
            num_sampling_steps=args.num_inference_steps,
            output_dir=args.output_dir,
        )

    # save all the args to a yaml file 
    # with open(os.path.join(args.output_dir, "args.yaml"), "w") as f:
    #     yaml.safe_dump({
    #         "test_csv_path": args.test_csv_path,
    #         "root_dir": args.root_dir,
    #         "npy_path": args.npy_path,
    #         "output_dir": args.output_dir,
    #         "num_inference_steps": args.num_inference_steps,
    #         "ckpt_path": args.ckpt_path,
    #         "config_path": args.config_path,
    #     }, f)

if __name__ == "__main__":
    main()
