import os
import io 
import re
import sys
sys.path.extend(['./taming-transformers', './stable-diffusion', './latent-diffusion'])
import time
import torch
import hashlib
import requests
import argparse
import huggingface_hub

import numpy as np
import k_diffusion as K
import torch.nn.functional as F

from sr_utils import *
from torch import nn
from PIL import Image
from einops import rearrange
from omegaconf import OmegaConf
from subprocess import Popen
from pytorch_lightning import seed_everything
from torchvision.utils import make_grid
from requests.exceptions import HTTPError
from torchvision.transforms import functional as TF
from tqdm.notebook import tqdm, trange
from functools import partial
from IPython.display import display
from ipywidgets import widgets
from ldm.util import instantiate_from_config



save_to_drive = False 
save_location = 'stable-diffusion-upscaler/%T-%I-%P.png'
cpu = torch.device("cpu")
device = torch.device("cuda")

if save_to_drive:
	from google.colab import drive
	drive.mount('/content/drive')
	save_location = '/content/drive/MyDrive/' + save_location

json_path = 'https://models.rivershavewings.workers.dev/config_laion_text_cond_latent_upscaler_2.json'
ckpt_path = 'https://models.rivershavewings.workers.dev/laion_text_cond_latent_upscaler_2_1_00470000_slim.pth'

#model_up = make_upscaler_model(fetch(json_path), fetch(ckpt_path))
model_up = make_upscaler_model('resources/' + json_path.split('/')[-1], 
								'resources/' + ckpt_path.split('/')[-1])

# sd_model_path = download_from_huggingface("CompVis/stable-diffusion-v-1-4-original", "sd-v1-4.ckpt")
vae_840k_model_path = download_from_huggingface("stabilityai/sd-vae-ft-mse-original", 
												"vae-ft-mse-840000-ema-pruned.ckpt")
vae_560k_model_path = download_from_huggingface("stabilityai/sd-vae-ft-ema-original", 
												"vae-ft-ema-560000-ema-pruned.ckpt")

# sd_model = load_model_from_config("stable-diffusion/configs/stable-diffusion/v1-inference.yaml", sd_model_path)
vae_model_840k = load_model_from_config("latent-diffusion/models/first_stage_models/kl-f8/config.yaml", 
										vae_840k_model_path)
vae_model_560k = load_model_from_config("latent-diffusion/models/first_stage_models/kl-f8/config.yaml", 
										vae_560k_model_path)

# sd_model = sd_model.to(device)
vae_model_840k = vae_model_840k.to(device)
vae_model_560k = vae_model_560k.to(device)
model_up = model_up.to(device)

tok_up = CLIPTokenizerTransform()
text_encoder_up = CLIPEmbedder(device=device)

num_samples = 1 
batch_size = 1 
decoder = 'finetuned_840k' 
guidance_scale = 1 
noise_aug_level = 0 
noise_aug_type = 'gaussian'
sampler = 'k_dpm_adaptive' 
steps = 50 
tol_scale = 0.25 
eta = 1.0 


if 'input_image' not in globals():
	# Set a demo image on first run.
	image_path = 'https://models.rivershavewings.workers.dev/assets/sd_2x_upscaler_demo.png'
	#input_image = Image.open(fetch()).convert('RGB')
	input_image = Image.open('resources/' + image_path.split('/')[-1]).convert('RGB')



upload = widgets.FileUpload(accept='.png,.jpg,.jpeg', multiple=False)
upload.observe(on_upload)
image_widget = widgets.Image(value=pil_to_bytes(input_image), width=512, height=512)
box = widgets.VBox([upload, image_widget])
display(box)


SD_C = 4 # Latent dimension
SD_F = 8 # Latent patch size (pixels per latent)
SD_Q = 0.18215 # sd_model.scale_factor; scaling for latents in first stage models


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--seed', type=int, default=42)
	parser.add_argument('--batch_size', type=int, default=1)
	parser.add_argument('--prompt', type=str, 
						default="the temple of fire by Ross Tran and Gerardo Dottori, oil on canvas" )
	args = parser.parse_args()
	run(args, input_image)
