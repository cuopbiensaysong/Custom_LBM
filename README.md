## Setup
To be up and running, you need first to create a virtual env with at least python3.10 installed and activate it

### With venv
```bash
python3.10 -m venv envs/lbm
source envs/lbm/bin/activate
```

### With conda
```bash
conda create -n lbm python=3.10
conda activate lbm
```

Then install the required dependencies and the repo in editable mode

```bash
pip install --upgrade pip
pip install -e .
pip install omegaconf
```

## Inference

We provide in `examples` a simple script to perform depth and normal estimation using the proposed method. 

```bash
python examples/inference/inference.py \
--model_name [depth|normals|relighting] \
--source_image path_to_your_image.jpg \
--output_path output_images
```

See the trained models on the HF Hub 🤗
- [Surface normals Checkpoint](https://huggingface.co/jasperai/LBM_normals)
- [Depth Checkpoint](https://huggingface.co/jasperai/LBM_depth)
- [Relighting Checkpoint](https://huggingface.co/jasperai/LBM_relighting)


## Training

### Data: 

To download data: 
```bash
pip install --upgrade gdown
mkdir data 
cd data
gdown --folder "https://drive.google.com/drive/folders/15Iue3eqUajySEwN9Dl3wIUKCvXdWfWRj"
gdown --id 1RWl7ZkR1hGcjMCRCf6WDNmQ5xHieRf8r
cd ..
sh delete_redundant_data.sh 
```

To train the model, you can use the following command:

```bash
python examples/training/train_lbm_surface.py examples/training/config/surface.yaml
```

*Note*: Make sure to update the relevant section of the `yaml` file to use your own data and log the results on your own [WandB](https://wandb.ai/site).


run.sh

#!/bin/bash
#SBATCH --job-name=lbm        # Job name
#SBATCH --output=./log_slurm/result/byol_cf10.txt      # Output file
#SBATCH --error=./log_slurm/error/byol_cf10.txt       # Error file
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --gpus=1                 # Number of GPUs per node
#SBATCH --nodes=1               # Số node yêu cầu
#SBATCH --cpus-per-task=20       # Số CPU cho mỗi task


python train.py --image_folder /home/user01/aiotlab/baodq/Self-Supervised-Learning-comparisons/train_lightly/datasets/cifar10