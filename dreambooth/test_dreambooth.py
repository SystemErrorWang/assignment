import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import argparse
import numpy as np
from PIL import Image
from compel import Compel
from diffusers import StableDiffusionPipeline


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


def test(args):
    ckpt = torch.load(args.ckpt_path)
    pipe = StableDiffusionPipeline.from_pretrained("../AnimateDiff/models/StableDiffusion")
    if args.train_text_encoder:
        pipe.unet.load_state_dict(ckpt['unet'])
        pipe.text_encoder.load_state_dict(ckpt['text_encoder'])
    else:
        pipe.unet.load_state_dict(ckpt)
    
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()
    pipe.unet.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    #compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32
    pipe = pipe.to(args.device, dtype=weight_dtype)
    pipe.safety_checker = None
    idx = args.ckpt_path.split('/')[1].split('-')[1].split('.')[0]
    with torch.no_grad():
        for _ in range(5):
            rand_seed = np.random.randint(1e8)
            prompt = [args.prompt] * args.batch_size
            negative_prompt = [args.negative_prompt] * args.batch_size
            #prompt_embeds = compel_proc(prompt)
            output = pipe(prompt=prompt, negative_prompt=negative_prompt, 
                        height=args.image_size, width=args.image_size, num_inference_steps=25, 
                        guidance_scale=7.5, generator=torch.Generator("cpu").manual_seed(rand_seed)).images
            save_name = os.path.join(args.save_path, "{}-{}-{}-{}.jpg".format(args.behavior, args.style, idx, rand_seed))
            grid = image_grid(output, rows=2, cols=2)
            grid.save(save_name)
            

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--save_path', type=str, default='result_dreambooth5')
    parser.add_argument('--ckpt_root', type=str, default='output_dreambooth5')
    parser.add_argument('--train_text_encoder', type=bool, default=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--mixed_precision', type=str, default='fp16')
    args = parser.parse_args()

    args.negative_prompt = "deformed iris, deformed pupils, semi-realistic, text, cropped, out of frame, " +\
                                "worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated," +\
                                "extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation," +\
                                "deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, " +\
                                "cloned face, disfigured, gross proportions, malformed limbs, long neck"

    behavior_list = [
                    "in front of mountain fuji, ",
                    "climbing the mountain fuji, ", 
                    "in front of tokyo tower, ",
                    "in front of the fushimi-inari shrine, ",
                    "visiting the sensoji, ",
                    "visiting the kiyomizu temple, ",
                    "walking in the beach of kamakura, ",
                    "walking on the streets of kyoto, ",
                    "walking along the railway of enoden, ",
                    "cross the streets of shibuya, ",
                    "standing in front of the shinjuku kabukicho, ",
                    ]

    style_list = [
                    "in the style of the starry sky by Van Gogh",
                    "in the style of Impressionism by Cloude Monet",
                    "in the style of ukiyoe The Great Wave of Kanagawa",
                    "traditional chinese ink painting style",
                    "in the style of Japanese animation by Shinkai Makoto",
                    "in the style of Japanese animation by Miyazaki Hayao",
                ]
    
    prefix = "masterpiece, best quality, highest detailed, extreme detailed, upper body photo, " +\
             "29 years old young man, wangxinrui, dreamboothidentifier, dreamboothkeyword, " 
    for behavior in behavior_list:
        for style in style_list:
            args.behavior, args.style = behavior, style
            args.prompt = prefix + behavior + style
            if not os.path.exists(args.save_path):
                os.mkdir(args.save_path)
            for idx in range(1, 11):
                args.ckpt_path = '{}/checkpoint-{}.pt'.format(args.ckpt_root, str(idx*500).zfill(6))
                test(args)









