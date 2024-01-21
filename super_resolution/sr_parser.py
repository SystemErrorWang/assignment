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
from diffusers import StableDiffusionPipeline



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
	model = model.to(cpu).eval().requires_grad_(False)
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
json_name = json_path.split('/')[-1]
ckpt_name = ckpt_path.split('/')[-1]

if not os.path.exits('resources'):
	os.mkdir('resources')

cmd1 = "wget {}".format(json_path)
cmd2 = "mv {} {}".format(json_name, 'resources/' + json_name)
cmd3 = "wget {}".format(ckpt_path)
cmd4 = "mv {} {}".format(ckpt_name, 'resources/' + ckpt_name)
os.system(cmd1)
os.system(cmd2)
os.system(cmd3)
os.system(cmd4)

#model_up = make_upscaler_model(fetch(json_path), fetch(ckpt_path))
model_up = make_upscaler_model('resources/' + json_name, 'resources/' + ckpt_name)

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


prompt = "the temple of fire by Ross Tran and Gerardo Dottori, oil on canvas" 
num_samples = 1 
batch_size = 1 
decoder = 'finetuned_840k' 
guidance_scale = 1 
noise_aug_level = 0 
noise_aug_type = 'gaussian'
sampler = 'k_dpm_adaptive' 
steps = 30
tol_scale = 0.25 
eta = 1.0 
seed = 0 


if 'input_image' not in globals():
	# Set a demo image on first run.
	image_path = 'https://models.rivershavewings.workers.dev/assets/sd_2x_upscaler_demo.png'
	#input_image = Image.open(fetch()).convert('RGB')
	input_image = Image.open('resources/' + image_path.split('/')[-1]).convert('RGB')

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


upload = widgets.FileUpload(accept='.png,.jpg,.jpeg', multiple=False)
upload.observe(on_upload)
image_widget = widgets.Image(value=pil_to_bytes(input_image), width=512, height=512)
box = widgets.VBox([upload, image_widget])
display(box)


SD_C = 4 # Latent dimension
SD_F = 8 # Latent patch size (pixels per latent)
SD_Q = 0.18215 # sd_model.scale_factor; scaling for latents in first stage models

@torch.no_grad()
def condition_up(prompts):
	return text_encoder_up(tok_up(prompts))

@torch.no_grad()
def run(seed, prompt, image):
	timestamp = int(time.time())
	if not seed:
		print('No seed was provided, using the current time.')
		seed = timestamp
	print(f'Generating with seed={seed}')
	seed_everything(seed)

	uc = condition_up(batch_size * [""])
	c = condition_up(batch_size * [prompt])

	if decoder == 'finetuned_840k':
		vae = vae_model_840k
	elif decoder == 'finetuned_560k':
		vae = vae_model_560k

	# image = Image.open(fetch(input_file)).convert('RGB')
	#image = input_image
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


def main(args):
	pipeline = StableDiffusionPipeline.from_pretrained("../../AnimateDiff/models/StableDiffusion", 
													revision="fp16", torch_dtype=torch.float16).to('cuda')
	for prompt in args.prompt_list:
		full_prompt = args.prompt_common + prompt
		rand_seeds = np.random.randint(1e8, size=5)
		for rand_seed in rand_seeds:
			print('rand_seed:', rand_seed, type(rand_seed))
			generator = torch.Generator("cuda").manual_seed(int(rand_seed))
			pipeline.safety_checker = None
			image = pipeline(full_prompt, negative_prompt=args.negative_prompt,
							num_inference_steps=25, generator=generator).images[0]
			run(args.seed, full_prompt, image)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--seed', type=int, default=42)
	#parser.add_argument('--prompt', type=str, default="")
	args = parser.parse_args()
	args.prompt_common = "masterpiece, best quality, extremely detailed, high quality,"
	args.prompt_list = [
	
	"the temple of fire by Ross Tran and Gerardo Dottori, oil on canvas",
	"close up photo of a girl walking in the river, in the style of oil painting by Claude Monet",
	"the CBD full of skyscraper in morden tokyo city, in the style of japanese ukiyoe",
	"a knight is wearing bonzer armors and riding his horse with a spear in his hand, in the style of cyberpunk",
	"traditional chinese painting of a office worker working in front of the desk with a laptop",
	"the castle of disneyland rises from a cup of coffee in a romantic pink atmosphere",
	"the earth rises from the horizon of moon with a rainbow, science fiction style",
	"realistic, DLSR photo, HDR, a girl in school uniform, in the universe with stars in the background",
	"a farrari sport car made of marble sculpture by Da Vinci of , photo realistic"
	]
	args.negative_prompt = "semi-realistic, text, cropped, out of frame, worst quality, low quality, " +\
                            "blurry, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers," +\
                            " mutated hands, poorly drawn hands, poorly drawn face, mutation," +\
                            "deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, " +\
                            "cloned face, disfigured, gross proportions, malformed limbs, long neck"

	main(args)
