import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import cv2
import time
import math
import wandb
import random
import logging
import inspect
import argparse
import datetime
import itertools
import subprocess
import numpy as np

import torch
import torchvision
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.transforms as transforms


from PIL import Image
from tqdm.auto import tqdm
from PIL.ImageOps import exif_transpose
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.optim.swa_utils import AveragedModel
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import AutoTokenizer, PretrainedConfig
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module



class DreamBoothDataset(Dataset):
    def __init__(self, data_root, prompt, tokenizer, size=512,):
        #self.size = size
        self.prompt = prompt
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.name_list = [f for f in os.listdir(data_root) if 'processed' in f]
        self.num_images = len(self.name_list)
        self.image_transforms = transforms.Compose(
            [  
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),
                #transforms.RandomRotation(15, transforms.InterpolationMode.BILINEAR),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self.num_images


    def dynamic_center_crop(self, image):
        h, w, c = np.shape(image)
        if h > w:
            dh = (h-w)//2
            image = image[dh:dh+w, :, :]
        else:
            dw = (w-h)//2
            image = image[:, dw:dw+h, :]
        return image


    def __getitem__(self, index):
        example = {}
        image_name = self.name_list[index]
        image = cv2.imread(os.path.join(self.data_root, image_name))
        x0, x1, y0, y1 = image_name.split('.')[0].split('_')[1:]
        x0, x1, y0, y1 = int(x0), int(x1), int(y0), int(y1)
        h, w, c = np.shape(image)
        dx0, dx1 = np.random.randint(0, x0//2), np.random.randint(x1+(w-x1)//2, w)
        dy0, dy1 = np.random.randint(0, y0//2), np.random.randint(y1+(h-y1)//2, h)
        image_crop = image[dy0:dy1, dx0:dx1, ::-1]
        image_crop = self.dynamic_center_crop(image_crop)
        example["pixel_values"] = self.image_transforms(Image.fromarray(image_crop))
        text_inputs = self.tokenizer(self.prompt, truncation=True, padding="max_length", 
                                    max_length=self.tokenizer.model_max_length, return_tensors="pt")
        example["prompt_ids"] = text_inputs.input_ids
        return example


def collate_fn(data):
    pixel_values = torch.stack([example["pixel_values"] for example in data])
    prompt_ids = torch.stack([example["prompt_ids"] for example in data])
    return {
            "pixel_values": pixel_values, 
            "prompt_ids": prompt_ids
            }


def train(args):
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-5").to("cuda")
    if args.train_text_encoder:
        pipe.vae.requires_grad_(False)
        params_to_train = itertools.chain(pipe.unet.parameters(), pipe.text_encoder.parameters()) 
    else:
        params_to_train = pipe.unet.parameters()
        pipe.vae.requires_grad_(False)
        pipe.text_encoder.requires_grad_(False)

    optimizer = torch.optim.AdamW(params_to_train, lr=args.learning_rate, weight_decay=args.weight_decay)
    dataset = DreamBoothDataset(data_root=args.dataroot, prompt=args.prompt, tokenizer=pipe.tokenizer)
    dataloader = DataLoader(dataset, shuffle=True, collate_fn=collate_fn,
                            batch_size=args.batch_size, num_workers=args.num_workers)
    
    lr_scheduler = get_scheduler("constant", optimizer=optimizer, num_warmup_steps=args.warmup_steps,
                                    num_training_steps=args.max_iter, num_cycles=1, power=1.0,)

    global_step = 0
    scaler = GradScaler(enabled=True)
    progress_bar = tqdm(range(0, args.max_iter), initial=0, desc="Steps",)
    for epoch in range(0, args.num_epochs):
        begin = time.perf_counter()
        epoch_loss = 0
        for step, batch in enumerate(dataloader):
            load_data_time = time.perf_counter() - begin
            with autocast(enabled=True):
                with torch.no_grad():
                    pixel_values = batch["pixel_values"].cuda()
                    prompt_ids = batch["prompt_ids"].cuda()
                    latents = pipe.vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * pipe.vae.config.scaling_factor
                    encoder_hidden_states = pipe.text_encoder(prompt_ids)[0]

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, pipe.scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
        
                noise_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean") / args.accumulate_step
                # Gather the losses across all processes for logging (if we use distributed training).
                epoch_loss += loss.mean().item()
                
                # Backpropagate
                if np.mod(global_step, args.accumulate_step) == 0:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                global_step += 1
                lr_scheduler.step()
                progress_bar.update(1)

                if global_step % args.log_steps == 0:
                    print("Epoch {}, global_step {}, data_time: {}, time: {}, step_loss: {}".format(
                        epoch, global_step, load_data_time, time.perf_counter() - begin, epoch_loss/(step+1)))
                
                if global_step % args.save_steps == 0:
                    save_path = os.path.join(args.output_dir, 
                                            f"checkpoint-{str(global_step).zfill(6)}.pt")
                    if args.train_text_encoder:
                        torch.save({'unet':pipe.unet.state_dict(),
                                    'text_encoder':pipe.text_encoder.state_dict()}, 
                                    save_path)
                    else:
                        torch.save(pipe.unet.state_dict(), save_path)
                    
                
                begin = time.perf_counter()
                if global_step > args.max_iter:
                    break

    


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--accumulate_step', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--warmup_steps', type=int, default=200)
    parser.add_argument('--log_steps', type=int, default=50)
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--max_iter', type=int, default=5000)
    parser.add_argument('--learning_rate', type=float, default=5e-6)
    parser.add_argument('--weight_decay', type=float, default=2e-2)
    parser.add_argument('--dataroot', type=str, default='dataset_body')
    parser.add_argument('--output_dir', type=str, default='output_dreambooth6')
    parser.add_argument('--logging_dir', type=str, default='log_dreambooth')
    parser.add_argument('--mixed_precision', type=str, default='fp16')
    parser.add_argument('--train_text_encoder', type=bool, default=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    args.prompt = "wangxinrui, dreamboothidentifier, dreamboothkeyword"
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    train(args)
    '''

    pipe = StableDiffusionPipeline.from_pretrained("../AnimateDiff/models/StableDiffusion")
    dataset = DreamBoothDataset(data_root="dataset", prompt="wangxinrui", tokenizer=pipe.tokenizer)
    dataloader = DataLoader(dataset, shuffle=True, collate_fn=collate_fn, batch_size=4, num_workers=4)
    for batch in tqdm(dataloader):
        pass
    '''




