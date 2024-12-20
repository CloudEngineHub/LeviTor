import datetime
import uuid
from PIL import Image
import numpy as np
import cv2
from scipy.interpolate import interp1d, PchipInterpolator
from packaging import version
from math import inf
import copy

import torch
import torchvision
import gradio as gr
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils import load_image, export_to_video
from utils import export_to_gif
from scipy import stats

from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from depth_anything_v2.dpt import DepthAnythingV2
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    PerspectiveCameras,
    AlphaCompositor
)
from sklearn.cluster import KMeans

import os
import sys
import ast
sys.path.insert(0, os.getcwd())
from models_diffusers.controlnet_svd import ControlNetSVDModel
from models_diffusers.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from pipelines.pipeline_stable_video_diffusion_mask_control import StableVideoDiffusionPipelineControlNet
from gradio_demo.utils_drag import *

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
import transformers
import diffusers
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import AutoencoderKLTemporalDecoder
from diffusers.training_utils import EMAModel
import accelerate


import warnings
print("gr file", gr.__file__)


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_guidance_scale", type=float, default=1.0)
    parser.add_argument("--max_guidance_scale", type=float, default=3.0)
    parser.add_argument("--middle_max_guidance", type=int, default=0, choices=[0, 1])
    parser.add_argument("--with_control", type=int, default=1, choices=[0, 1])
    parser.add_argument("--controlnet_cond_scale", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default="gradio_demo/outputs", help="Path to the output video.")
    parser.add_argument("--seed", type=int, default=42, help="random seed.")
    parser.add_argument("--noise_aug", type=float, default=0.02)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--gaussian_r", type=int, default=10,)
    parser.add_argument("--width", type=int, default=288)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument('--input-size', type=int, default=512)
    # parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument(
        "--use_ema", action="store_true", help="Whether to use EMA model."
    )
    parser.add_argument("--if_id", type=int, default=1,)
    parser.add_argument("--if_depth", type=int, default=1,)
    parser.add_argument("--sam_path", type=str, default='',)
    parser.add_argument("--depthanything_path", type=str, default='',)
    parser.add_argument("--logging_dir", type=str, default="logs", help="",)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default='latest',
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    args = parser.parse_args()
    return args

args = get_args()
ensure_dirname(args.output_dir)

colors = torch.tensor([
        [240,255,240],
        [230,230,250],
        [123,104,238],
        [0,255,127],
        [255,255,0],
        [255,185,15],
        [255,106,106],
        [255,0,255],
        [145,44,238],
        [0,139,139],
        [255,211,155],
        [139,126,102],
        [255,165,0],
        [240,255,240],
        [230,230,250],
        [123,104,238],
        [0,255,127],
        [255,255,0],
        [255,185,15],
        [255,106,106],
        [255,0,255],
        [145,44,238],
        [0,139,139],
        [255,211,155],
        [139,126,102],
        [255,165,0],
        [255,250,250],
        [240,255,240],
        [230,230,250],
        [123,104,238],
        [0,255,127],
        [255,255,0],
        [255,185,15],
        [255,106,106],
        [255,0,255],
        [145,44,238],
        [0,139,139],
        [255,211,155],
        [139,126,102],
        [255,165,0],
        [255,250,250],
        [240,255,240],
        [230,230,250],
        [123,104,238],
        [0,255,127],
        [255,255,0],
        [255,185,15],
        [255,106,106],
        [255,0,255],
        [145,44,238],
        [0,139,139],
        [255,211,155],
        [139,126,102],
        [255,165,0],
        [255,250,250],
        [240,255,240],
        [230,230,250],
        [123,104,238],
        [0,255,127],
        [255,255,0],
        [255,185,15],
        [255,106,106],
        [255,0,255],
        [145,44,238],
        [0,139,139],
        [255,211,155],
        [139,126,102],
        [255,165,0],
        [255,250,250],
        [240,255,240],
        [230,230,250],
        [123,104,238],
        [0,255,127],
        [255,255,0],
        [255,185,15],
        [255,106,106],
        [255,0,255],
        [145,44,238],
        [0,139,139],
        [255,211,155],
        [139,126,102],
        [255,165,0],
        ], dtype=torch.uint8
    )

color_list = []
for i in range(20):
    color = np.concatenate([np.random.random(4)*255], axis=0)
    if color[-1] < 150:
        color[-1] = 150.0
    color_list.append(color)

# SAM
sam_checkpoint = args.sam_path
model_type = "vit_h"
device = "cuda:0"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
auto_predictor = SamAutomaticMaskGenerator(sam)

test_time = 0
SAM_labels = []

model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
depth_anything = DepthAnythingV2(**model_configs['vitl'])
depth_anything.load_state_dict(torch.load(args.depthanything_path, map_location='cpu'))
depth_anything = depth_anything.to(device).eval()
depth_for_3D_initial = None

masks_selected = []
masks_selected_tmp = None

def interpolate_trajectory(points, n_points):
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    d = [point[2] for point in points]

    t = np.linspace(0, 1, len(points))

    fx = PchipInterpolator(t, x)
    fy = PchipInterpolator(t, y)
    fd = PchipInterpolator(t, d)

    new_t = np.linspace(0, 1, n_points)

    new_x = fx(new_t)
    new_y = fy(new_t)
    new_d = fd(new_t)
    new_points = list(zip(new_x, new_y, new_d))

    return new_points


def gen_gaussian_heatmap(imgSize=200):
    circle_img = np.zeros((imgSize, imgSize), np.float32)
    circle_mask = cv2.circle(circle_img, (imgSize//2, imgSize//2), imgSize//2, 1, -1)

    isotropicGrayscaleImage = np.zeros((imgSize, imgSize), np.float32)

    for i in range(imgSize):
        for j in range(imgSize):
            isotropicGrayscaleImage[i, j] = 1 / 2 / np.pi / (40 ** 2) * np.exp(
                -1 / 2 * ((i - imgSize / 2) ** 2 / (40 ** 2) + (j - imgSize / 2) ** 2 / (40 ** 2)))

    isotropicGrayscaleImage = isotropicGrayscaleImage * circle_mask
    isotropicGrayscaleImage = (isotropicGrayscaleImage / np.max(isotropicGrayscaleImage)).astype(np.float32)
    isotropicGrayscaleImage = (isotropicGrayscaleImage / np.max(isotropicGrayscaleImage)*255).astype(np.uint8)

    return isotropicGrayscaleImage


def get_vis_image(
        target_size=(512 , 512), points=None,  side=20,
        num_frames=14,
    ):

    vis_images = []
    heatmap = gen_gaussian_heatmap()

    trajectory_list = []
    radius_list = []
    
    for index, point in enumerate(points):
        trajectories = [[int(i[0]), int(i[1])] for i in point]
        trajectory_list.append(trajectories)

        radius = side
        radius_list.append(radius)  

    if len(trajectory_list) == 0:
        vis_images = [Image.fromarray(np.zeros(target_size, np.uint8)) for _ in range(num_frames)]
        return vis_images

    for idxx, point in enumerate(trajectory_list[0]):
        new_img = np.zeros(target_size, np.uint8)
        vis_img = new_img.copy()
        
        if idxx >= args.num_frames:
            break

        for cc, (trajectory, radius) in enumerate(zip(trajectory_list, radius_list)):
            
            center_coordinate = trajectory[idxx]
            trajectory_ = trajectory[:idxx]
            side = min(radius, 50)
 
            y1 = max(center_coordinate[1] - side,0)
            y2 = min(center_coordinate[1] + side, target_size[0] - 1)
            x1 = max(center_coordinate[0] - side, 0)
            x2 = min(center_coordinate[0] + side, target_size[1] - 1)
            
            if x2-x1>3 and y2-y1>3:
                need_map = cv2.resize(heatmap, (x2-x1, y2-y1))
                new_img[y1:y2, x1:x2] = need_map.copy()
                
                if cc >= 0:
                    vis_img[y1:y2,x1:x2] = need_map.copy()
                    if len(trajectory_) == 1:
                        vis_img[trajectory_[0][1], trajectory_[0][0]] = 255
                    else:
                        for itt in range(len(trajectory_)-1):
                            cv2.line(vis_img, (trajectory_[itt][0], trajectory_[itt][1]), (trajectory_[itt+1][0], trajectory_[itt+1][1]), (255, 255, 255), 3)

        img = new_img

        # Ensure all images are in RGB format
        if len(img.shape) == 2:  # Grayscale image
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3 and img.shape[2] == 3:  # Color image in BGR format
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
            
        # Convert the numpy array to a PIL image
        # pil_img = Image.fromarray(img)
        # images.append(pil_img)
        vis_images.append(Image.fromarray(vis_img))

    return vis_images


def frames_to_video(frames_folder, output_video_path, fps=7):
    frame_files = os.listdir(frames_folder)
    # sort the frame files by their names
    frame_files = sorted(frame_files, key=lambda x: int(x.split(".")[0]))

    video = []
    for frame_file in frame_files:
        frame_path = os.path.join(frames_folder, frame_file)
        frame = torchvision.io.read_image(frame_path)
        video.append(frame)

    video = torch.stack(video)
    video = rearrange(video, 'T C H W -> T H W C')
    torchvision.io.write_video(output_video_path, video, fps=fps)


def save_gifs_side_by_side(
    batch_output,
    validation_control_images,
    output_folder,
    target_size=(512 , 512),
    duration=200,
    point_tracks=None,
):
    flattened_batch_output = batch_output
    def create_gif(image_list, gif_path, duration=100):
        pil_images = [validate_and_convert_image(img, target_size=target_size) for img in image_list]
        pil_images = [img for img in pil_images if img is not None]
        if pil_images:
            pil_images[0].save(gif_path, save_all=True, append_images=pil_images[1:], loop=0, duration=duration)

        # also save all the pil_images
        tmp_folder = gif_path.replace(".gif", "")
        print(tmp_folder)
        ensure_dirname(tmp_folder)
        tmp_frame_list = []
        for idx, pil_image in enumerate(pil_images):
            tmp_frame_path = os.path.join(tmp_folder, f"{idx}.png")
            pil_image.save(tmp_frame_path)
            tmp_frame_list.append(tmp_frame_path)
        
        # also save as mp4
        output_video_path = gif_path.replace(".gif", ".mp4")
        frames_to_video(tmp_folder, output_video_path, fps=7)

    # Creating GIFs for each image list
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    gif_paths = []

    for idx, image_list in enumerate([validation_control_images, flattened_batch_output]):

        gif_path = os.path.join(output_folder.replace("vis_gif.gif", ""), f"temp_{idx}_{timestamp}.gif")
        create_gif(image_list, gif_path)
        gif_paths.append(gif_path)

        # also save the point_tracks
        assert point_tracks is not None
        point_tracks_path = gif_path.replace(".gif", ".npy")
        np.save(point_tracks_path, point_tracks.cpu().numpy())

    # Function to combine GIFs side by side
    def combine_gifs_side_by_side(gif_paths, output_path, vertical=False):
        print(gif_paths)
        gifs = [Image.open(gif) for gif in gif_paths]

        # Assuming all gifs have the same frame count and duration
        frames = []
        for frame_idx in range(gifs[0].n_frames):
            combined_frame = None
            for gif in gifs:
                gif.seek(frame_idx)
                if combined_frame is None:
                    combined_frame = gif.copy()
                else:
                    if vertical:
                        combined_frame = get_concat_v(combined_frame, gif.copy())
                    else:
                        combined_frame = get_concat_h(combined_frame, gif.copy())
            frames.append(combined_frame)
        if output_path.endswith(".mp4"):
            video = [torchvision.transforms.functional.pil_to_tensor(frame) for frame in frames]
            video = torch.stack(video)
            video = rearrange(video, 'T C H W -> T H W C')
            torchvision.io.write_video(output_path, video, fps=7)
            print(f"Saved video to {output_path}")
        else:
            print(gifs[0].info['duration'])
            frames[0].save(output_path, save_all=True, append_images=frames[1:], loop=0, duration=duration)
        
    # Helper function to concatenate images horizontally
    def get_concat_h(im1, im2):
        # img first, heatmap second
        im1, im2 = im2, im1

        dst = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst

    # Helper function to concatenate images vertically
    def get_concat_v(im1, im2):
        dst = Image.new('RGB', (max(im1.width, im2.width), im1.height + im2.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (0, im1.height))
        return dst

    # Combine the GIFs into a single file
    combined_gif_path = output_folder
    combine_gifs_side_by_side(gif_paths, combined_gif_path)

    combined_gif_path_v = gif_path.replace(".gif", "_v.mp4")
    ensure_dirname(combined_gif_path_v.replace(".mp4", ""))
    combine_gifs_side_by_side(gif_paths, combined_gif_path_v, vertical=True)

    return combined_gif_path


# Define functions
def validate_and_convert_image(image, target_size=(512 , 512)):
    if image is None:
        print("Encountered a None image")
        return None

    if isinstance(image, torch.Tensor):
        # Convert PyTorch tensor to PIL Image
        if image.ndim == 3 and image.shape[0] in [1, 3]:  # Check for CxHxW format
            if image.shape[0] == 1:  # Convert single-channel grayscale to RGB
                image = image.repeat(3, 1, 1)
            image = image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            image = Image.fromarray(image)
        else:
            print(f"Invalid image tensor shape: {image.shape}")
            return None
    elif isinstance(image, Image.Image):
        # Resize PIL Image
        image = image.resize(target_size)
    else:
        print("Image is not a PIL Image or a PyTorch tensor")
        return None
    
    return image


class Drag:

    def __init__(self, device, args, model_length, dtype=torch.float16):
        self.device = device
        self.dtype = dtype

        logging_dir = os.path.join(args.output_dir, args.logging_dir)
        accelerator_project_config = ProjectConfiguration(
            project_dir=args.output_dir, logging_dir=logging_dir)

        kwargs_handlers = None
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with=args.report_to,
            project_config=accelerator_project_config,
            kwargs_handlers=kwargs_handlers,
        )

        if accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        # If passed along, set the training seed now.
        set_seed(217113)

        # Handle the repository creation
        if accelerator.is_main_process:
            if args.output_dir is not None:
                os.makedirs(args.output_dir, exist_ok=True)


        # Load scheduler, tokenizer and models.
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="image_encoder", revision=args.revision, variant="fp16"
        )
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant="fp16")

        unet = UNetSpatioTemporalConditionModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="unet",
            low_cpu_mem_usage=True,
            variant="fp16",
        )

        logger.info("Initializing controlnet weights from unet")
        controlnet = ControlNetSVDModel.from_unet(
            unet,
            conditioning_channels=3,
        )

        vae.requires_grad_(False)
        image_encoder.requires_grad_(False)
        unet.requires_grad_(False)

        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Move image_encoder and vae to gpu and cast to weight_dtype
        image_encoder.to(accelerator.device, dtype=weight_dtype)
        vae.to(accelerator.device, dtype=weight_dtype)
        
        # Create EMA for the unet.
        if args.use_ema:
            ema_controlnet = EMAModel(controlnet.parameters(
            ), model_cls=UNetSpatioTemporalConditionModel, model_config=unet.config)

        if args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers

                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    logger.warn(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError(
                    "xformers is not available. Make sure it is installed correctly")

        # `accelerate` 0.16.0 will have better support for customized saving
        if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
            # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
            def save_model_hook(models, weights, output_dir):
                if args.use_ema:
                    ema_controlnet.save_pretrained(os.path.join(output_dir, "controlnet_ema"))

                assert len(models) == 2, len(models)
                for i, model in enumerate(models):
                    if i == 0:
                        assert isinstance(model, UNetSpatioTemporalConditionModel)
                        # also save unet weights for convenience
                        model.save_pretrained(os.path.join(output_dir, "unet"))
                    elif i == 1:
                        assert isinstance(model, ControlNetSVDModel)
                        model.save_pretrained(os.path.join(output_dir, "controlnet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

            def load_model_hook(models, input_dir):
                if args.use_ema:
                    load_model = EMAModel.from_pretrained(os.path.join(input_dir, "controlnet_ema"), ControlNetSVDModel)
                    ema_controlnet.load_state_dict(load_model.state_dict())
                    ema_controlnet.to(accelerator.device)
                    del load_model

                for i in range(len(models)):
                    # pop models so that they are not loaded again
                    model = models.pop()

                    if isinstance(model, UNetSpatioTemporalConditionModel):
                        continue
                    else:
                        assert isinstance(model, ControlNetSVDModel)
                        # load diffusers style into model
                        load_model = ControlNetSVDModel.from_pretrained(input_dir, subfolder="controlnet")
                        model.register_to_config(**load_model.config)

                        model.load_state_dict(load_model.state_dict())
                        del load_model

            accelerator.register_save_state_pre_hook(save_model_hook)
            accelerator.register_load_state_pre_hook(load_model_hook)

        controlnet.requires_grad_(True)

        # check parameters
        if accelerator.is_main_process:
            rec_txt1 = open(os.path.join(args.output_dir, 'rec_para.txt'), 'w')
            rec_txt2 = open(os.path.join(args.output_dir, 'rec_para_train.txt'), 'w')
            for name, para in controlnet.named_parameters():
                if para.requires_grad is False:
                    rec_txt1.write(f'{name}\n')
                else:
                    rec_txt2.write(f'{name}\n')
            rec_txt1.close()
            rec_txt2.close()

        # Prepare everything with our `accelerator`.
        unet, controlnet = accelerator.prepare(
            unet, controlnet
        )

        if args.use_ema:
            ema_controlnet.to(accelerator.device)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if accelerator.is_main_process:
            accelerator.init_trackers("SVDXtend", config=vars(args))

        # Potentially load in the weights and states from a previous save
        if args.resume_from_checkpoint:
            if args.resume_from_checkpoint != "latest":
                # path = os.path.basename(args.resume_from_checkpoint)
                path = args.resume_from_checkpoint
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(args.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                accelerator.print(
                    f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                args.resume_from_checkpoint = None
            else:
                accelerator.print(f"Resuming from checkpoint {path}")
                # accelerator.load_state(os.path.join(args.output_dir, path))
                accelerator.load_state(path)

        controlnet.eval()

        # create pipeline
        if args.use_ema:
            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
            ema_controlnet.store(controlnet.parameters())
            ema_controlnet.copy_to(controlnet.parameters())
        # The models need unwrapping because for compatibility in distributed training mode.
        pipeline = StableVideoDiffusionPipelineControlNet.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            image_encoder=accelerator.unwrap_model(image_encoder),
            vae=accelerator.unwrap_model(vae),
            controlnet=accelerator.unwrap_model(controlnet),
            revision=args.revision,
            torch_dtype=weight_dtype,
        )
        pipeline = pipeline.to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)

        self.pipeline = pipeline
        self.accelerator = accelerator

        self.height = args.height
        self.width = args.width
        self.args = args
        self.model_length = model_length

    def run(self, first_frame_path, tracking_points, controlnet_cond_scale, motion_bucket_id, depth_input, points_num):
        keypoints_dir = os.path.join(args.output_dir, os.path.basename(first_frame_path).split('_')[-1].split('.')[0])
        os.makedirs(keypoints_dir, exist_ok=True)

        input_all_points = tracking_points.constructor_args['value']
        depth_input = ast.literal_eval(depth_input)
        points_num = float(points_num)

        resized_all_points = [
            [[int(e1[0]), int(e1[1])]for e1 in e] for e in input_all_points
        ]

        for i in range(len(depth_input)):
            for ids, d in enumerate(depth_input[i]):
                resized_all_points[i][ids].append(d)

        resized_all_points = [
            tuple([tuple([e1[0], e1[1], e1[2]]) for e1 in e]) for e in resized_all_points
        ]

        for idx, splited_track in enumerate(resized_all_points):
            if len(splited_track) == 0:
                warnings.warn("running without point trajectory control")
                continue

            if len(splited_track) == 1: # stationary point
                displacement_point = tuple([splited_track[0][0] + 1, splited_track[0][1] + 1])
                splited_track = tuple([splited_track[0], displacement_point])
            # interpolate the track
            splited_track = interpolate_trajectory(splited_track, self.model_length)
            splited_track = splited_track[:self.model_length]
            resized_all_points[idx] = splited_track

        # load_image
        image = cv2.imread(first_frame_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape
        assert width == self.width
        assert height == self.height
        # image = image.resize((self.width, self.height))

        vis_images = get_vis_image(
            target_size=(self.args.height, self.args.width),
            points=resized_all_points,
            num_frames=self.model_length,
            side=args.gaussian_r,
        )

        resized_all_points = torch.tensor(resized_all_points)

        # auto masks

        masks_auto = auto_predictor.generate(image)
        num_moving_points = len(masks_selected)

        # masks_selected.reverse()
        for mask in masks_selected[::-1]:
            masks_auto.append({'segmentation': mask[1], 'area': mask[1].sum()})
        # masks_auto.append({'segmentation': masks_select2[0], 'area': masks_select2.sum()})
        # masks_auto = sorted(masks_auto, key=(lambda x: x['area']), reverse=True)
        masks_auto.reverse()

        final_masks = np.zeros_like(masks_auto[0]['segmentation'])
        threshold = self.width * self.height
        threshold *= 0.001
        for ids, ann in enumerate(masks_auto):
            m = ann['segmentation'].astype(np.uint8)
            m *= (ids+1)
            if ids >= num_moving_points:
                mask_user = (final_masks > 0) & (final_masks <= num_moving_points)
                m[mask_user] = 0

            diff = ((m != 0) & (final_masks != 0))
            if diff.sum() != 0:
                mask_id_before = np.argmax(np.bincount(final_masks[diff]))
                if (final_masks == mask_id_before).sum() < ann['area']:
                    final_masks[diff] = 0
                else:
                    m[diff] = 0
            
            if (m != 0).sum() > threshold:
                final_masks = final_masks + m

        for i in range(final_masks.max()):
            seg = (final_masks == i+1)
            area = seg.sum()
            if area <= threshold:
                final_masks[seg] = 0

        num_point = (final_masks != 0).sum()
        points_all = torch.zeros((num_point, 3))
        ids_all = torch.zeros((num_point, 1))

        total_point = 0
        global depth_for_3D_initial

        for h in range(self.height):
            for w in range(self.width):
                if final_masks[h, w] != 0:
                    points_all[total_point][0] = w
                    points_all[total_point][1] = h
                    points_all[total_point][2] = depth_for_3D_initial[h, w]
                    ids_all[total_point] = final_masks[h, w]
                    total_point += 1

        # mask_colors = torch.zeros((final_masks.shape[0], final_masks.shape[1], 3), dtype=torch.uint8)
        # global colors
        # for i in range(final_masks.max()):
        #     mask_i = torch.tensor((final_masks == i+1).squeeze())
        #     if mask_i.sum() > 0:
        #         mask_colors[mask_i] = colors[i]

        # mask_colors_save = mask_colors.numpy()

        # name3 = "test_mask_color.jpg"
        # save_image = Image.fromarray(mask_colors_save)
        # save_image.save(name3)

        mask_first_frame = np.zeros((self.height, self.width), dtype=np.int32)
        for i in range(len(points_all)):
            mask_first_frame[int(points_all[i, 1]), int(points_all[i, 0])] = ids_all[i][0]

        points_ind_user = [torch.nonzero((ids_all.squeeze() == i+1)).squeeze() for i in range(num_moving_points)]
        ids_user = [ids_all[points_ind_user[i]] for i in range(num_moving_points)]
        moving_deltas = []
        for i in range(num_moving_points):
            moving_delta = []
            for p in resized_all_points[i]:
                moving_delta.append(p - resized_all_points[i][0])
            moving_deltas.append(moving_delta)

        # Create cameras

        intrinsics = [3866.378063371277, 4538.052129020386, args.width/2, args.height/2]
        fcl_screen = ((intrinsics[0], intrinsics[1]),)
        prp_screen = ((intrinsics[2], intrinsics[3]),)
        image_size = ((args.height, args.width),)

        R = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        R = torch.stack([-R[:, 0], -R[:, 1], R[:, 2]], 1)
        # T = torch.tensor([0.4275749375398189, -4.969627768772348, -0.34638224259263617]).unsqueeze(1)
        T = torch.tensor([0.0, 0.0, 0.0]).unsqueeze(1)

        new_c2w = torch.cat([R, T], 1)
        w2c = torch.linalg.inv(torch.cat((new_c2w, torch.Tensor([[0, 0, 0, 1]])), 0))
        R, T = w2c[:3, :3].permute(1, 0), w2c[:3, 3]
        R = R[None]
        T = T[None]

        cameras = PerspectiveCameras(device='cuda', focal_length=fcl_screen, principal_point=prp_screen,
                                    in_ndc=False, image_size=image_size, R=R, T=T)
        
        mask_area_max = [0 for i in range(mask_first_frame.max())]
        mask_area_min = [inf for i in range(mask_first_frame.max())]

        mask_for_num_moving = (mask_first_frame <= num_moving_points)
        mask_first_frame[mask_for_num_moving] = 0
        depth_for_3D_initial_c = copy.deepcopy(depth_for_3D_initial)
        depth_for_3D_initial_c[mask_for_num_moving] = 1.
        
        initial_image = copy.deepcopy(mask_first_frame)
        # mask_output = torch.zeros((self.model_length, 512, 288, 3), dtype=torch.int32)
        mask_output = torch.zeros((self.model_length, self.height, self.width), dtype=torch.int32)
        depth_output = torch.zeros((self.model_length, self.height, self.width), dtype=torch.float32)

        # move points
        for i in range(self.model_length):
            depth_for_3D = copy.deepcopy(depth_for_3D_initial_c)
            depth_for_3D_true = copy.deepcopy(depth_for_3D_initial_c)
            points = copy.deepcopy(points_all)

            for j in range(num_moving_points):
                # moving_scale = 1 + moving_deltas[j][i][2] * 10
                # moving_deltas[j][i][:2] *= moving_scale 
                points[points_ind_user[j], :2] += moving_deltas[j][i][:2]
            
            points[:, 0] -= (args.width / 2)
            points[:, 0] *= points[:, 2]
            points[:, 0] /= 3866.378063371277

            points[:, 1] -= (args.height / 2)
            points[:, 1] *= points[:, 2]
            points[:, 1] /= 4538.052129020386
            
            for j in range(num_moving_points):
                points[points_ind_user[j], 2] += moving_deltas[j][i][2]

            points_moving = [Pointclouds(points=points[points_ind_user[j]].unsqueeze(0), features=ids_user[j].unsqueeze(0)).cuda() for j in range(num_moving_points)]

            # Create renderer and rasterization settings
            images_out = []
            for j in range(num_moving_points):
                current_depth = float(resized_all_points[j][i][2])
                radius_current = max(0.011 - float(moving_deltas[j][i][2]), 0.011)
                raster_settings = PointsRasterizationSettings(
                    image_size=(self.height, self.width),
                    radius=radius_current,
                    points_per_pixel=10
                )
                renderer = PointsRenderer(
                    rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings),
                    compositor=AlphaCompositor()
                )

                image = renderer(points_moving[j]).squeeze().cpu().numpy()
                image = image.round()

                # mask_empty = (image != 0) & (mask_first_frame == 0)
                mask_noempty = (image != 0) & (depth_for_3D > current_depth)
                # mask_first_frame[mask_empty] = (j+1)
                mask_first_frame[mask_noempty] = (j+1)

                # for more objects process
                mask_for_imagedepth = (depth_for_3D > current_depth) & (image != 0)
                depth_for_3D[mask_for_imagedepth] = current_depth
                depth_for_3D_true[mask_for_imagedepth] = current_depth

                if len(points_all[points_ind_user[j]]) > 0:
                    top_point = points_all[points_ind_user[j]][0][:2]
                    max_id = stats.mode(initial_image[max(int(top_point[1])-5, 0):int(top_point[1]), max(int(top_point[0])-5, 0):min(int(top_point[0])+5, args.width)]).mode[0]
                    mask_first_frame[mask_for_num_moving & (mask_first_frame == 0)] = max_id
            
                images_out.append(image)
            
            mask_colors = torch.zeros((args.height, args.width, 3), dtype=torch.uint8)

            for j in range(int(mask_first_frame.max())):
                mask_i = torch.tensor((mask_first_frame == j+1).squeeze())
                if mask_i.sum() > 0:
                    mask_colors[mask_i] = colors[j % 80]

            # mask_output[i] = mask_colors
            mask_output[i] = torch.tensor(mask_first_frame, dtype=torch.int32)
            global test_time
            cv2.imwrite(os.path.join(keypoints_dir, f'test_{test_time}_mask_{i+1}.jpg'), mask_colors.numpy())

            for j in range(len(mask_area_max)):
                area = (mask_first_frame == (j+1)).sum()
                if area > mask_area_max[j]:
                    mask_area_max[j] = area
                if area < mask_area_min[j]:
                    mask_area_min[j] = area

            depth_output[i] = torch.tensor(depth_for_3D_true, dtype=torch.float32)
            
            for j in range(num_moving_points):
                mask_first_frame[images_out[j] != 0] = initial_image[images_out[j] != 0]
                
            

        # name3 = "mask_move.mp4"
        # torchvision.io.write_video(name3, mask_output, 6.0)
        # print(mask_area_max)
        # torch.save(mask_output, 'mask.pt')
        # torch.save(depth_output, 'depth.pt')

        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

        pos_array = np.zeros((mask_output[0].shape[0], mask_output[0].shape[1], 2), dtype=np.float32)
        for i in range(mask_output[0].shape[0]):
            for j in range(mask_output[0].shape[1]):
                pos_array[i][j][0] = i
                pos_array[i][j][1] = j

        all_keypoints = []
        random_seed = 217
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        # name = "keypoint.mp4"
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # video_writer = cv2.VideoWriter(name, fourcc, 5.0, (masks[0].shape[1], masks[0].shape[0]))

        initial_centers = {}
        for maskid, mask in enumerate(mask_output):
            # mask_color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            clusters = []
            for i in range(mask.max()):
                mask_i = mask == i+1
                # mask_color[mask_i] = colors[i]
                
                pos_array_i = pos_array[mask_i]

                if mask_area_max[i] == 0:
                    continue
                else:
                    k = round(mask_area_max[i] * 50 / (self.height * self.width))
                    k = max(k, 1)
                    k = min(k, 8)

                if i < num_moving_points or mask_area_min[i] == 0 or mask_area_max[i] / mask_area_min[i] > 10:
                    k = max(k, 3)

                pos_array_i = pos_array_i[::10]
                if len(pos_array_i) < k or k == 0:
                    continue

                k = int(k * points_num)
                # k = min(k, len(pos_array_i))
                k = max(1, k)
                # k = 1
                
                if i+1 not in initial_centers.keys():
                    len_per_label = len(pos_array_i) // k
                    id_list = [len_per_label // 2]
                    for _ in range(k-1):
                        id_list.append(id_list[-1] + len_per_label)
                    kmeans = KMeans(n_clusters=k, init=pos_array_i[id_list]).fit(pos_array_i)
                    initial_centers[i+1] = kmeans.cluster_centers_

                else:
                    for point in initial_centers[i+1]:
                        if point not in pos_array_i:
                            len_per_label = len(pos_array_i) // k
                            id_list = [len_per_label // 2]
                            for _ in range(k-1):
                                id_list.append(id_list[-1] + len_per_label)
                            initial_centers[i+1] = pos_array_i[id_list]
                            break

                # k = min(k, len(initial_centers[i+1]))
                # if len(initial_centers[i+1]) > len(pos_array_i):
                #     initial_centers[i+1] = initial_centers[i+1][:len(pos_array_i)]
                # k = len(initial_centers[i+1])

                kmeans = KMeans(n_clusters=k, init=initial_centers[i+1]).fit(pos_array_i)
                cluster_centers = kmeans.cluster_centers_
                for j in cluster_centers:
                    depth = round(1. / float(depth_output[maskid][int(j[0])][int(j[1])]))
                    if depth == 1:
                        depth = 0
                    clusters.append([int(j[0]), int(j[1]), i+1, depth])

            all_keypoints.append(clusters)
        
        keypoints_path = os.path.join(keypoints_dir, 'keypoints_sklearn_new.json')
        with open(keypoints_path, "w") as f:
            json.dump(all_keypoints, f)

        # ours
        validation_sample_list = [[first_frame_path, keypoints_path],]

        args.num_validation_images = len(validation_sample_list)
        logger.info(
            f"Running validation... \n Generating {args.num_validation_images} videos."
        )

        # run inference
        val_save_dir = os.path.join(args.output_dir, "validation_images")

        if not os.path.exists(val_save_dir):
            os.makedirs(val_save_dir)

        with torch.autocast(
            str(self.accelerator.device).replace(":0", ""), enabled=self.accelerator.mixed_precision == "fp16"
        ):
            for val_img_idx in range(args.num_validation_images):
                num_frames = args.num_frames
                image_val, point_conditions = validation_sample_list[val_img_idx]
                frame_indice = np.linspace(0, (num_frames - 1) * args.frame_interval, num_frames, dtype=int)
                frame_list = []
                frame_list.append(load_image(image_val))

                # original_size = frame_list[0].size
                # frame_list = [frame.resize((args.width, args.height)) for frame in frame_list]
                input_start_image = frame_list[0]
                # input_end_image = frame_list[-1]

                pred_tracks_val = point_embedding_val = None
                if point_conditions is not None:
                    with open(point_conditions, 'r') as f:
                        pred_tracks_val = json.load(f)
                        pred_tracks_val = [pred_tracks_val[frame_id] for frame_id in frame_indice]

                    if args.if_id == 0 or args.if_depth == 0:
                        for kpoints in pred_tracks_val:
                            for kpoint in kpoints:
                                if args.if_id == 0:
                                    kpoint[2] = 0
                                if args.if_depth == 0:
                                    kpoint[3] = 0


                video_frames = self.pipeline(
                    input_start_image,
                    with_control=True,
                    point_tracks=pred_tracks_val,
                    height=self.height,
                    width=self.width,
                    side=args.gaussian_r,
                    num_frames=num_frames,
                    motion_bucket_id=motion_bucket_id,
                    fps=6,
                    with_id_feature=False,
                    controlnet_cond_scale=controlnet_cond_scale,
                ).frames[0]

                out_file = os.path.join(keypoints_dir, f"test_{test_time}")
                test_time += 1

                for i in range(num_frames):
                    img = video_frames[i]
                    video_frames[i] = np.array(img)
                    # save_image = Image.fromarray(video_frames[i])
                    # save_image.save(out_file + f'_{i}.png')
                export_to_gif(video_frames, out_file, 16)

        return out_file


with gr.Blocks() as demo:
    gr.Markdown("""<h1 align="center">Generative Video</h1><br>""")

    gr.Markdown("""Gradio Demo for <a href='https://arxiv.org/abs/2403.07420'><b>: LeviTor: 3D Trajectory Oriented Image-to-Video Synthesis</b></a>. The template is inspired by DragAnything.""")

    # gr.Image(label="Generative Video Interpolation", value="assets/output.gif")
    # gr.Image(label="Generative Video Interpolation", value=None)

    gr.Markdown("""## Usage: <br>
                1. Upload images. <br>
                    1.1  Upload the start image via the "Upload Start Image" button.<br>
                2. Draw some drags.<br>
                    2.1. Click "Add SAM Point?" to determine whether the current input point is in mask(1) or not(0).<br>
                    2.2. Click "Select Area with SAM" to select the area that you want to control.<br>
                    2.3. Click "Add New Drag Trajectory" to add a new motion trajectory.<br>
                    2.4. Click "Input number of points for inference here" to choose scale of the points number for inference.<br>
                    2.5. Click "Input depth values here" to input depth value for points in trajectories.<br>
                    2.6. Click "Delete last drag" to delete the whole lastest path.<br>
                    2.7. Click "Delete last step" to delete the lastest clicked control point.<br>
                3. Generate the video (according the path) with a click on "Run" button. <br>""")
    
    # device, args, height, width, model_length
    DragAnything = Drag("cuda:0", args, 16)
    first_frame_path = gr.State()
    tracking_points = gr.State([])
    
    flag_points = gr.State()


    def reset_states(first_frame_path, tracking_points):
        # first_frame_path = gr.State()
        tracking_points = gr.State([])
        global masks_selected, masks_selected_tmp, SAM_labels
        masks_selected = []
        SAM_labels = []
        masks_selected_tmp = None

        return first_frame_path, tracking_points


    def preprocess_image(image):
        image_pil = image2pil(image.name)

        raw_w, raw_h = image_pil.size
        # resize_ratio = max(512 / raw_w, 320 / raw_h)
        # image_pil = image_pil.resize((int(raw_w * resize_ratio), int(raw_h * resize_ratio)), Image.BILINEAR)
        # image_pil = transforms.CenterCrop((320, 512))(image_pil.convert('RGB'))
        image_pil = image_pil.resize((args.width, args.height), Image.BILINEAR)

        first_frame_path = os.path.join(args.output_dir, f"first_frame_{str(uuid.uuid4())[:4]}.png")
        
        image_pil.save(first_frame_path)

        image = cv2.imread(first_frame_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        depth = depth_anything.infer_image(image, args.input_size)
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        global depth_for_3D_initial
        depth_for_3D_initial = 1.0 / depth
        depth_for_3D_initial[depth_for_3D_initial == inf] = 1.

        global test_time
        test_time = 0

        return first_frame_path, first_frame_path, gr.State([])


    def add_drag(tracking_points):
        global SAM_labels
        SAM_labels = []
        tracking_points.constructor_args['value'].append([])
        return tracking_points, 0
        
    def re_add_drag(tracking_points):
        tracking_points.constructor_args['value'][-1]=[]
        masks_selected.append(masks_selected_tmp)
        return tracking_points, 1


    def delete_last_drag(tracking_points, first_frame_path):
        tracking_points.constructor_args['value'].pop()
        transparent_background = Image.open(first_frame_path).convert('RGBA')
        w, h = transparent_background.size
        transparent_layer = np.zeros((h, w, 4))

        for track in tracking_points.constructor_args['value']:
            if len(track) > 1:
                for i in range(len(track)-1):
                    start_point = track[i]
                    end_point = track[i+1]
                    vx = end_point[0] - start_point[0]
                    vy = end_point[1] - start_point[1]
                    arrow_length = np.sqrt(vx**2 + vy**2)
                    if i == len(track)-2:
                        cv2.arrowedLine(transparent_layer, tuple(start_point), tuple(end_point), (255, 0, 0, 255), 2, tipLength=8 / arrow_length)
                    else:
                        cv2.line(transparent_layer, tuple(start_point), tuple(end_point), (255, 0, 0, 255), 2,)
            else:
                cv2.circle(transparent_layer, tuple(track[0]), 5, (255, 0, 0, 255), -1)

        transparent_layer = Image.fromarray(transparent_layer.astype(np.uint8))
        trajectory_map = Image.alpha_composite(transparent_background, transparent_layer)
        return tracking_points, trajectory_map


    def delete_last_step(tracking_points, first_frame_path):
        tracking_points.constructor_args['value'][-1].pop()
        transparent_background = Image.open(first_frame_path).convert('RGBA')
        w, h = transparent_background.size
        transparent_layer = np.zeros((h, w, 4))

        for track in tracking_points.constructor_args['value']:
            if len(track) > 1:
                for i in range(len(track)-1):
                    start_point = track[i]
                    end_point = track[i+1]
                    vx = end_point[0] - start_point[0]
                    vy = end_point[1] - start_point[1]
                    arrow_length = np.sqrt(vx**2 + vy**2)
                    if i == len(track)-2:
                        cv2.arrowedLine(transparent_layer, tuple(start_point), tuple(end_point), (255, 0, 0, 255), 2, tipLength=8 / arrow_length)
                    else:
                        cv2.line(transparent_layer, tuple(start_point), tuple(end_point), (255, 0, 0, 255), 2,)
            else:
                cv2.circle(transparent_layer, tuple(track[0]), 5, (255, 0, 0, 255), -1)

        transparent_layer = Image.fromarray(transparent_layer.astype(np.uint8))
        trajectory_map = Image.alpha_composite(transparent_background, transparent_layer)

        return tracking_points, trajectory_map


    def add_tracking_points(tracking_points, first_frame_path, flag_points, add_sam, evt: gr.SelectData):  # SelectData is a subclass of EventData
        print(f"You selected {evt.value} at {evt.index} from {evt.target}")
        tracking_points.constructor_args['value'][-1].append(evt.index)
        tracking_depths = ""

        if flag_points==1:
            transparent_background = Image.open(first_frame_path).convert('RGBA')
            w, h = transparent_background.size
            transparent_layer = 0

            for idx, mask in enumerate(masks_selected):
                color = color_list[idx]
                transparent_layer = mask[1].reshape(h, w, 1) * color.reshape(1, 1, -1) + transparent_layer

            for idx, track in enumerate(tracking_points.constructor_args['value']):
                mask = np.zeros((h, w, 3))
                color = color_list[idx+1]
                transparent_layer = mask[:, :, 0].reshape(h, w, 1) * color.reshape(1, 1, -1) + transparent_layer
            
                if len(track) > 1:
                    for i in range(len(track)-1):
                        start_point = track[i]
                        end_point = track[i+1]
                        vx = end_point[0] - start_point[0]
                        vy = end_point[1] - start_point[1]
                        arrow_length = np.sqrt(vx**2 + vy**2)
                        if i == len(track)-2:
                            cv2.arrowedLine(transparent_layer, tuple(start_point), tuple(end_point), (255, 0, 0, 255), 2, tipLength=8 / arrow_length)
                        else:
                            cv2.line(transparent_layer, tuple(start_point), tuple(end_point), (255, 0, 0, 255), 2,)
                else:
                    cv2.circle(transparent_layer, tuple(track[0]), 5, (255, 0, 0, 255), -1)
            
            transparent_layer = Image.fromarray(transparent_layer.astype(np.uint8))
            alpha_coef = 0.99
            im2_data = transparent_layer.getdata()
            new_im2_data = [(r, g, b, int(a * alpha_coef)) for r, g, b, a in im2_data]
            transparent_layer.putdata(new_im2_data)
                
            trajectory_map = Image.alpha_composite(transparent_background, transparent_layer)

            global depth_for_3D_initial
            tracking_depths = []
            for track in tracking_points.constructor_args['value']:
                tmp = []
                for point in track:
                    tmp.append(depth_for_3D_initial[point[1], point[0]])
                tracking_depths.append(tmp)

        else:
            transparent_background = Image.open(first_frame_path).convert('RGBA')
            w, h = transparent_background.size
            
            input_point = []
            global SAM_labels
            add_sam = ast.literal_eval(add_sam)
            SAM_labels.append(add_sam)

            for track in tracking_points.constructor_args['value'][-1]:
                input_point.append([track[0],track[1]])
        
            image = cv2.imread(first_frame_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictor.set_image(image)
            

            input_point = np.array(input_point)
            input_label = np.array(SAM_labels)

            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            # cv2.imwrite(args.output_dir+"/"+"mask_{}.jpg".format(len(tracking_points.constructor_args['value'])),masks[1]*255)
            global masks_selected_tmp
            masks_selected_tmp = masks
            transparent_layer = 0
            for idx, mask in enumerate(masks_selected):
                color = color_list[idx]
                transparent_layer = mask[1].reshape(h, w, 1) * color.reshape(1, 1, -1) + transparent_layer
            
            color = color_list[len(masks_selected)]
            transparent_layer = masks_selected_tmp[1].reshape(h, w, 1) * color.reshape(1, 1, -1) + transparent_layer
            
            transparent_layer = Image.fromarray(transparent_layer.astype(np.uint8))
            alpha_coef = 0.99
            im2_data = transparent_layer.getdata()
            new_im2_data = [(r, g, b, int(a * alpha_coef)) for r, g, b, a in im2_data]
            transparent_layer.putdata(new_im2_data)
            trajectory_map = Image.alpha_composite(transparent_background, transparent_layer)

        return tracking_points, trajectory_map, tracking_depths

    with gr.Row():
        with gr.Column(scale=1):
            image_upload_button = gr.UploadButton(label="Upload Start Image", file_types=["image"])
            select_area_button = gr.Button(value="Select Area with SAM")
            add_drag_button = gr.Button(value="Add New Drag Trajectory")

            depth_input = gr.Textbox(placeholder="Input depth values here")
            points_num = gr.Textbox(placeholder="Input number of points for inference here")
            add_sam = gr.Textbox(placeholder="Add SAM point?")

            tracking_depths = gr.Textbox(label="Depths for reference")
            reset_button = gr.Button(value="Reset")
            run_button = gr.Button(value="Run")
            delete_last_drag_button = gr.Button(value="Delete last drag")
            delete_last_step_button = gr.Button(value="Delete last step")
            

        with gr.Column(scale=7):
            with gr.Row():
                with gr.Column(scale=6):
                    input_image = gr.Image(
                        label="start frame",
                        interactive=True,
                        height=args.height,
                        width=args.width,
                    )
            with gr.Row():
                with gr.Column(scale=5):
                    output_video = gr.Image(
                        label="Output Video",
                        height=args.height,
                        width=args.width,
                    )

    with gr.Row():
        with gr.Column(scale=1):

            controlnet_cond_scale = gr.Slider(
                label='Control Scale', 
                minimum=0.0, 
                maximum=10, 
                step=0.1, 
                value=1.0,
            )

            motion_bucket_id = gr.Slider(
                label='Motion Bucket', 
                minimum=1, 
                maximum=180, 
                step=1, 
                value=100,
            )


    with gr.Row():
        gr.Markdown("""
            ## Citation
            ```bibtex
                title={LeviTor: 3D Trajectory Oriented Image-to-Video Synthesis}, 
                author={Hanlin Wang and Hao Ouyang and Qiuyu Wang and Wen Wang and Ka Leong Cheng and Qifeng Chen and Yujun Shen and Limin Wang}, 
                year={2024}, 
                eprint={2412.15214}, 
                archivePrefix={arXiv}, 
                primaryClass={cs.CV}}
            ```
            """)

    image_upload_button.upload(preprocess_image, image_upload_button, [input_image, first_frame_path, tracking_points])

    select_area_button.click(add_drag, tracking_points, [tracking_points, flag_points])

    add_drag_button.click(re_add_drag, tracking_points, [tracking_points, flag_points])

    delete_last_drag_button.click(delete_last_drag, [tracking_points, first_frame_path], [tracking_points, input_image])

    delete_last_step_button.click(delete_last_step, [tracking_points, first_frame_path], [tracking_points, input_image])

    reset_button.click(reset_states, [first_frame_path, tracking_points], [first_frame_path, tracking_points])

    input_image.select(add_tracking_points, [tracking_points, first_frame_path, flag_points, add_sam], [tracking_points, input_image, tracking_depths])

    run_button.click(DragAnything.run, [first_frame_path, tracking_points, controlnet_cond_scale, motion_bucket_id, depth_input, points_num], output_video)

    demo.queue().launch(server_name="0.0.0.0", share=False)
    # demo.launch(server_name="0.0.0.0", share=True)
