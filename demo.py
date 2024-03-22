import argparse
import os
import cv2
import torch
import numpy as np
from PIL import Image
# from controlnet_aux import OpenposeDetector
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from diffusers import ControlNetModel, AutoencoderKL # StableDiffusionXLControlNetPipeline
from diffusers.utils import load_image
from diffusers import StableDiffusionXLPipeline
from diffusers.pipeline_controlnet_sdxl import StableDiffusionXLControlNetPipeline
from compel import Compel, ReturnedEmbeddingsType
# from diffusers.models.attention_processor import AttnProcessor2_0

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a pose image generation scripts.")
    parser.add_argument(
        "--normal_image_path",
        type=str,
        default=None,
        help=""
    )
    parser.add_argument(
        "--normal_background_image_path",
        type=str,
        default=None,
        help=""
    )
    parser.add_argument(
        "--depth_image_path",
        type=str,
        default=None,
        help=""
    )
    parser.add_argument(
        "--normal_controlnet_dir",
        type=str,
        default="./data/weights/controlnet_sdxl_normal_omnidata-80000/controlnet",
        help=""
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help=(
            "inference diffusion model steps."
        ),
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a group of people on the beach",
        help=""
    )
    parser.add_argument(
        "--image_prompt",
        type=str,
        # default="",
        default=None,
        help="used for ip adapter"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        # default="naked, ugly, anime, cartoon, graphic, text, painting, crayon, graphite, abstract glitch, blurry, extra limbs, poorly drawn face, poorly drawn hands, poorly drawn feet",
        default="naked, anime, cartoon, graphic, out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature.",
        help=""
    )
    parser.add_argument(
        "--controlnet_conditioning_scale",
        type=float,
        nargs="+",
        default=0.7,
        metavar=(''),
        help=""
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output_dir",
        help="the path of generated images"
    )
    parser.add_argument(
        "--sdxl_dir",
        type=str,
        default="data/weights/sdxl-base-1.0/",
        help="the path of sdxl weights"
    )
    parser.add_argument(
        "--vae_dir",
        type=str,
        default="data/weights/sdxl-vae-fp16-fix/",
        help="the path of sdxl vae weights"
    )
    parser.add_argument(
    "--use_dynamic_scale",
    action="store_true",
    )
    parser.add_argument(
    "--run_compile",
    action="store_true",
    )
    parser.add_argument(
    "--adapter_arch",
    choices=['controlnet', 't2i_adapter'],
    default='controlnet',
    )
    parser.add_argument(
    "--num_images_per_prompt",
    type=int,
    default=4,
    help="the numbers of generated images"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    condition_images = []
    controlnets = []

    if args.normal_image_path is not None:
        if args.normal_background_image_path is not None:
            normal_image = Image.open(args.normal_image_path)
            mask = cv2.imread(args.normal_image_path, cv2.IMREAD_UNCHANGED)[...,3]
            normal_background_image = load_image(args.normal_background_image_path)
            merged_image = Image.new('RGBA', normal_image.size)
            merged_image.paste(normal_background_image.convert("RGBA"), (0,0), normal_background_image.convert("RGBA") )
            merged_image.paste(normal_image, (0,0), normal_image)
            normal_image = merged_image.convert("RGB")
        else:
            normal_image = load_image(args.normal_image_path)
            # normal_image = normal_image.resize((1024, 1024))

        base_name = os.path.basename(args.normal_image_path)

        condition_images.append(normal_image)
        normal_controlnet = ControlNetModel.from_pretrained(
                # "./data/sdxl_weights/controlnet-openpose-sdxl-1.0/snapshots/4104e2c285d4e7e0ff1e426923415819ffa2bec7", 
                # "./data/sdxl_weights/controlnet-openpose-sdxl-1.0", 
                # "./logs_controlnet_sdxl_normal_5e-5/checkpoint-35000/controlnet",
                args.normal_controlnet_dir,
                # "./logs_controlnet/checkpoint-20000/controlnet/",
                use_safetensors=True,
                torch_dtype=torch.float16).to("cuda")
        controlnets.append(normal_controlnet)

    # MODEL_DIR="data/weights/sdxl-base-1.0/"
    # VAE_DIR="data/weights/sdxl-vae-fp16-fix/"
    MODEL_DIR=args.sdxl_dir
    VAE_DIR=args.vae_dir

    vae = AutoencoderKL.from_pretrained(VAE_DIR, torch_dtype=torch.float16).to("cuda")

    if len(controlnets) == 0:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            # "stabilityai/stable-diffusion-xl-base-1.0", 
            MODEL_DIR,
            torch_dtype=torch.float16, 
            variant="fp16", 
            use_safetensors=True
        ).to("cuda")

        compel = Compel(
            tokenizer=[pipe.tokenizer, pipe.tokenizer_2] , 
            text_encoder=[pipe.text_encoder, pipe.text_encoder_2], 
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, 
            requires_pooled=[False, True],
            truncate_long_prompts=False
            )

        # negative_prompt = "ugly, anime, cartoon, graphic, text, painting, crayon, graphite, abstract glitch, blurry."

    else:
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            # "stabilityai/stable-diffusion-xl-base-1.0",
            MODEL_DIR,
            controlnet=controlnets,
            vae=vae,
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16,
        ).to("cuda")
        pipe.controlnet.to(memory_format=torch.channels_last)

    # pipe = pipe.to("cuda")
    pipe.unet.to(memory_format=torch.channels_last)
    pipe.enable_model_cpu_offload()

    # if args.run_compile:
    #     print("Run torch compile")
    #     torch._dynamo.config.verbose = True
    #     pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    #     pipe.controlnet = torch.compile(pipe.controlnet, mode="reduce-overhead", fullgraph=True)
    # else:
    #     pipe.enable_model_cpu_offload()

    # pipe.unet.set_attn_processor(AttnProcessor2_0())
    generator = torch.manual_seed(2)

    if isinstance(args.controlnet_conditioning_scale, int):
        controlnet_conditioning_scale = [args.controlnet_conditioning_scale]
    else:
        controlnet_conditioning_scale = args.controlnet_conditioning_scale
    
    if len(controlnets) == 0:
        conditioning, pooled = compel(args.prompt)
        negative_conditioning, negative_pooled = compel(args.negative_prompt)
        [conditioning, negative_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, negative_conditioning])
        [pooled, negative_pooled] = compel.pad_conditioning_tensors_to_same_length([pooled, negative_pooled])
        pooled = pooled[0]
        negative_pooled = negative_pooled[0]

        images = pipe(
            # prompt=prompt_1, 
            prompt_embeds=conditioning, 
            pooled_prompt_embeds=pooled,
            negative_prompt_embeds=negative_conditioning,
            negative_pooled_prompt_embeds=negative_pooled,
            original_size=(1024, 1024), 
            target_size=(1024,1024),
            num_images_per_prompt=args.num_images_per_prompt,
            ).images

    elif args.use_dynamic_scale:
        images = pipe(
            args.prompt,
            image=condition_images,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
            negative_prompt=args.negative_prompt,
            num_images_per_prompt=args.num_images_per_prompt,
            use_dynamic_scale=args.use_dynamic_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
        ).images

    else:
        images = pipe(
            args.prompt,
            image=condition_images,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
            negative_prompt=args.negative_prompt,
            num_images_per_prompt=args.num_images_per_prompt,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
        ).images

    os.makedirs(args.output_dir, exist_ok=True)
    for idx, image_i in enumerate(images):
        image_i.save(os.path.join(args.output_dir, "{}_{}.jpg".format(base_name[:-4], idx)))