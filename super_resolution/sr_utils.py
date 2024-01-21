import os
import io 
import re
import sys
sys.path.extend(['./taming-transformers', './stable-diffusion', './latent-diffusion'])
import time
import torch
import hashlib
import requests
import huggingface_hub

import numpy as np
import k_diffusion as K
import torch.nn.functional as F


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



def clean_prompt(prompt):
	badchars = re.compile(r'[/\\]')
	prompt = badchars.sub('_', prompt)
	if len(prompt) > 100:
		prompt = prompt[:100] + '…'
	return prompt


def format_filename(timestamp, seed, index, prompt):
	string = save_location
	string = string.replace('%T', f'{timestamp}')
	string = string.replace('%S', f'{seed}')
	string = string.replace('%I', f'{index:02}')
	string = string.replace('%P', clean_prompt(prompt))
	return string

def save_image(image, **kwargs):
	filename = format_filename(**kwargs)
	os.makedirs(os.path.dirname(filename), exist_ok=True)
	image.save(filename)


def load_model_from_config(config, ckpt):
	print(f"Loading model from {ckpt}")
	pl_sd = torch.load(ckpt, map_location="cpu")
	sd = pl_sd["state_dict"]
	config = OmegaConf.load(config)
	model = instantiate_from_config(config.model)
	m, u = model.load_state_dict(sd, strict=False)
	model = model.to('cpu').eval().requires_grad_(False)
	return model


def fetch(url_or_path):
	if url_or_path.startswith('http:') or url_or_path.startswith('https:'):
		_, ext = os.path.splitext(os.path.basename(url_or_path))
		cachekey = hashlib.md5(url_or_path.encode('utf-8')).hexdigest()
		cachename = f'{cachekey}{ext}'
		if not os.path.exists(f'cache/{cachename}'):
			os.makedirs('tmp', exist_ok=True)
			os.makedirs('cache', exist_ok=True)
			#!curl '{url_or_path}' -o 'tmp/{cachename}'
			cmd = "!curl {} -o tmp/{}".format(url_or_path, cachename)
			os.system(cmd)
			os.rename(f'tmp/{cachename}', f'cache/{cachename}')
		return f'cache/{cachename}'
	return url_or_path


class NoiseLevelAndTextConditionedUpscaler(nn.Module):
    def __init__(self, inner_model, sigma_data=1., embed_dim=256):
        super().__init__()
        self.inner_model = inner_model
        self.sigma_data = sigma_data
        self.low_res_noise_embed = K.layers.FourierFeatures(1, embed_dim, std=2)

    def forward(self, inputs, sigma, low_res, low_res_sigma, c, **kwargs):
        cross_cond, cross_cond_padding, pooler = c
        c_in = 1 / (low_res_sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_noise = low_res_sigma.log1p()[:, None]
        c_in = K.utils.append_dims(c_in, low_res.ndim)
        low_res_noise_embed = self.low_res_noise_embed(c_noise)
        low_res_in = F.interpolate(low_res, scale_factor=2, mode='nearest') * c_in
        mapping_cond = torch.cat([low_res_noise_embed, pooler], dim=1)
        return self.inner_model(inputs, sigma, unet_cond=low_res_in, mapping_cond=mapping_cond, 
        						cross_cond=cross_cond, cross_cond_padding=cross_cond_padding, **kwargs)


def make_upscaler_model(config_path, model_path, pooler_dim=768, train=False, device='cpu'):
    config = K.config.load_config(open(config_path))
    model = K.config.make_model(config)
    model = NoiseLevelAndTextConditionedUpscaler(
        model,
        sigma_data=config['model']['sigma_data'],
        embed_dim=config['model']['mapping_cond_dim'] - pooler_dim,
    )
    ckpt = torch.load(model_path, map_location='cpu')
    model.load_state_dict(ckpt['model_ema'])
    model = K.config.make_denoiser_wrapper(config)(model)
    if not train:
        model = model.eval().requires_grad_(False)
    return model.to(device)


def download_from_huggingface(repo, filename):
	while True:
		try:
			return huggingface_hub.hf_hub_download(repo, filename)
		except HTTPError as e:
			if e.response.status_code == 401:
				# Need to log into huggingface api
				huggingface_hub.interpreter_login()
				continue
			elif e.response.status_code == 403:
				# Need to do the click through license thing
				print(f'Go here and agree to the click through license on your account: https://huggingface.co/{repo}')
				input('Hit enter when ready:')
				continue
			else:
				raise e


class CFGUpscaler(nn.Module):
    def __init__(self, model, uc, cond_scale):
        super().__init__()
        self.inner_model = model
        self.uc = uc
        self.cond_scale = cond_scale

    def forward(self, x, sigma, low_res, low_res_sigma, c):
        if self.cond_scale in (0.0, 1.0):
          # Shortcut for when we don't need to run both.
          if self.cond_scale == 0.0:
            c_in = self.uc
          elif self.cond_scale == 1.0:
            c_in = c
          return self.inner_model(x, sigma, low_res=low_res, low_res_sigma=low_res_sigma, c=c_in)
          
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        low_res_in = torch.cat([low_res] * 2)
        low_res_sigma_in = torch.cat([low_res_sigma] * 2)
        c_in = [torch.cat([uc_item, c_item]) for uc_item, c_item in zip(self.uc, c)]
        uncond, cond = self.inner_model(x_in, sigma_in, low_res=low_res_in, low_res_sigma=low_res_sigma_in, c=c_in).chunk(2)
        return uncond + (cond - uncond) * self.cond_scale


class CLIPTokenizerTransform:
    def __init__(self, version="openai/clip-vit-large-patch14", max_length=77):
        from transformers import CLIPTokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.max_length = max_length

    def __call__(self, text):
        indexer = 0 if isinstance(text, str) else ...
        tok_out = self.tokenizer(text, truncation=True, max_length=self.max_length,
                                 return_length=True, return_overflowing_tokens=False,
                                 padding='max_length', return_tensors='pt')
        input_ids = tok_out['input_ids'][indexer]
        attention_mask = 1 - tok_out['attention_mask'][indexer]
        return input_ids, attention_mask


class CLIPEmbedder(nn.Module):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda"):
        super().__init__()
        from transformers import CLIPTextModel, logging
        logging.set_verbosity_error()
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.transformer = self.transformer.eval().requires_grad_(False).to(device)

    @property
    def device(self):
        return self.transformer.device

    def forward(self, tok_out):
        input_ids, cross_cond_padding = tok_out
        clip_out = self.transformer(input_ids=input_ids.to(self.device), output_hidden_states=True)
        return clip_out.hidden_states[-1], cross_cond_padding.to(self.device), clip_out.pooler_output


def pil_to_bytes(image):
	with io.BytesIO() as fp:
		image.save(fp, format='png')
		return fp.getvalue()

def on_upload(change):
	global input_image
	if change['name'] == 'value':
		value ,= change['new'].values()
		filename = value['metadata']['name']
		assert '/' not in filename
		print(f'Upscaling {filename}')
		input_image = Image.open(io.BytesIO(value['content'])).convert('RGB')
		image_widget.value = value['content']
		image_widget.width = input_image.size[0]
		image_widget.height = input_image.size[1]


@torch.no_grad()
def condition_up(prompts):
	tok_up = CLIPTokenizerTransform()
	text_encoder_up = CLIPEmbedder(device='cuda')
	return text_encoder_up(tok_up(prompts))

@torch.no_grad()
def run(args, input_image):
	timestamp = int(time.time())
	seed = args.seed
	if not seed:
		print('No seed was provided, using the current time.')
		seed = timestamp
	print(f'Generating with seed={seed}')
	seed_everything(seed)

	uc = condition_up(args.batch_size * [""])
	c = condition_up(args.batch_size * [args.prompt])

	if decoder == 'finetuned_840k':
		vae = vae_model_840k
	elif decoder == 'finetuned_560k':
		vae = vae_model_560k

	# image = Image.open(fetch(input_file)).convert('RGB')
	image = input_image
	image = TF.to_tensor(image).to(device) * 2 - 1
	low_res_latent = vae.encode(image.unsqueeze(0)).sample() * SD_Q
	low_res_decoded = vae.decode(low_res_latent/SD_Q)

	[_, C, H, W] = low_res_latent.shape

	# Noise levels from stable diffusion.
	sigma_min, sigma_max = 0.029167532920837402, 14.614642143249512

	model_wrap = CFGUpscaler(model_up, uc, cond_scale=guidance_scale)
	low_res_sigma = torch.full([batch_size], noise_aug_level, device=device)
	x_shape = [batch_size, C, 2*H, 2*W]

	def do_sample(noise, extra_args):

		sigmas = torch.linspace(np.log(sigma_max), np.log(sigma_min), steps+1).exp().to(device)
		if sampler == 'k_euler':
			return K.sampling.sample_euler(model_wrap, noise * sigma_max, sigmas, extra_args=extra_args)
		elif sampler == 'k_euler_ancestral':
			return K.sampling.sample_euler_ancestral(model_wrap, noise * sigma_max, sigmas, extra_args=extra_args, eta=eta)
		elif sampler == 'k_dpm_2_ancestral':
			return K.sampling.sample_dpm_2_ancestral(model_wrap, noise * sigma_max, sigmas, extra_args=extra_args, eta=eta)
		elif sampler == 'k_dpm_fast':
			return K.sampling.sample_dpm_fast(model_wrap, noise * sigma_max, sigma_min, sigma_max, steps, extra_args=extra_args, eta=eta)
		elif sampler == 'k_dpm_adaptive':
			sampler_opts = dict(s_noise=1., rtol=tol_scale * 0.05, atol=tol_scale / 127.5, pcoeff=0.2, icoeff=0.4, dcoeff=0)
		return K.sampling.sample_dpm_adaptive(model_wrap, noise * sigma_max, sigma_min, sigma_max, extra_args=extra_args, eta=eta, **sampler_opts)

	image_id = 0
	for _ in range((num_samples-1)//batch_size + 1):
		if noise_aug_type == 'gaussian':
			latent_noised = low_res_latent + noise_aug_level * torch.randn_like(low_res_latent)
		elif noise_aug_type == 'fake':
			latent_noised = low_res_latent * (noise_aug_level ** 2 + 1)**0.5
		extra_args = {'low_res': latent_noised, 'low_res_sigma': low_res_sigma, 'c': c}
		noise = torch.randn(x_shape, device=device)
		up_latents = do_sample(noise, extra_args)

		pixels = vae.decode(up_latents/SD_Q) # equivalent to sd_model.decode_first_stage(up_latents)
		pixels = pixels.add(1).div(2).clamp(0,1)

		display(TF.to_pil_image(make_grid(pixels, batch_size)))
		for j in range(pixels.shape[0]):
			img = TF.to_pil_image(pixels[j])
			save_image(img, timestamp=timestamp, index=image_id, prompt=prompt, seed=seed)
			image_id += 1


