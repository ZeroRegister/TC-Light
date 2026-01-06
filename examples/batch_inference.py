
import argparse
import os
import torch
import torch.distributed as dist
from glob import glob
from tqdm import tqdm
from omegaconf import OmegaConf
import copy
import sys

# Add project root to sys.path to allow imports from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.VidToMe.config_utils import load_config
from utils.VidToMe.utils import init_model, seed_everything, get_frame_ids
from generate import Generator

def setup_ddp():
    """Initialize Distributed Data Parallel"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return rank, world_size, local_rank
    else:
        print("[INFO] Not running in DDP mode.")
        return 0, 1, 0

def cleanup_ddp():
    """Cleanup DDP"""
    if dist.is_initialized():
        dist.destroy_process_group()

def get_video_files(video_dir):
    """Get list of video files from directory"""
    valid_extensions = ['.mp4', '.avi', '.move', '.mkv']
    video_files = []
    for ext in valid_extensions:
        video_files.extend(glob(os.path.join(video_dir, f"*{ext}")))
    return sorted(video_files)

def get_prompt(prompt_path):
    """Read prompt from text file"""
    if os.path.exists(prompt_path):
        with open(prompt_path, 'r') as f:
            return f.read().strip()
    return None

def main():
    parser = argparse.ArgumentParser(description="Batch Inference for TC-Light Sim2Real")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing input videos")
    parser.add_argument("--prompt_dir", type=str, required=True, help="Directory containing prompt txt files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--base_config", type=str, default="configs/tclight_default.yaml", help="Path to base config file")
    parser.add_argument("--control_type", type=str, default="depth", help="ControlNet type (e.g., depth, softedge, seg, or none)")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    # === Local Model Paths ===
    parser.add_argument("--sd_model_path", type=str, default=None, help="Local path to Stable Diffusion model (e.g., /models/stable-diffusion-v1-5)")
    parser.add_argument("--controlnet_path", type=str, default=None, help="Local path to ControlNet model (e.g., /models/control_v11p_sd15_seg)")
    
    args = parser.parse_args()
    
    # Setup DDP
    rank, world_size, local_rank = setup_ddp()
    
    # Load base config
    base_config = OmegaConf.load(args.base_config)
    
    # Update config with args
    base_config.seed = args.seed
    base_config.generation.control = args.control_type
    base_config.work_dir = args.output_dir
    base_config.generation.output_path = args.output_dir
    
    # Initialize Model (One per process)
    seed_everything(base_config.seed)
    
    # Use local SD model path if provided, otherwise use config's model_key
    sd_model_to_use = args.sd_model_path if args.sd_model_path else base_config.model_key
    
    pipe, scheduler, model_key = init_model(
        base_config.device, 
        base_config.sd_version, 
        sd_model_to_use, 
        base_config.generation.control, 
        base_config.float_precision,
        controlnet_path=args.controlnet_path
    )
    base_config.model_key = model_key
    
    # Get list of videos
    all_videos = get_video_files(args.video_dir)
    
    # Shard videos for DDP
    my_videos = all_videos[rank::world_size]
    
    print(f"[Rank {rank}] Processing {len(my_videos)} videos out of {len(all_videos)} total.")
    
    for video_path in tqdm(my_videos, desc=f"Rank {rank}", position=rank):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        prompt_path = os.path.join(args.prompt_dir, f"{video_name}.txt")
        
        # Determine prompt
        prompt = get_prompt(prompt_path)
        if prompt is None:
            print(f"[Rank {rank}] [WARNING] Prompt file not found for {video_name}, using default/upsampler.")
            # If using upsampler, generate.py handles None prompt
        
        # Create a specific config for this video
        cur_config = copy.deepcopy(base_config)
        cur_config.data.rgb_path = video_path
        cur_config.generation.prompt = {video_name: prompt} # Format expected by generate.py
        
        # Prepare output dir for this video
        cur_output_dir = os.path.join(args.output_dir, video_name)
        os.makedirs(cur_output_dir, exist_ok=True)
        cur_config.generation.output_path = cur_output_dir # Update output path to subfolder
        cur_config.work_dir = cur_output_dir # Update work_dir valid for some internal saves
        cur_config.inversion.save_path = os.path.join(cur_output_dir, "latents") # Latents in subfolder
        cur_config.generation.latents_path = cur_config.inversion.save_path 

        
        # Initialize Generator for this video
        # We reuse the loaded pipe and scheduler
        generator = Generator(pipe, scheduler, cur_config)
        
        # Determine frame IDs
        frame_ids = get_frame_ids(
            cur_config.generation.frame_range, 
            getattr(cur_config.generation, 'frame_ids', None) # Handle possible missing attribute
        )
        
        # Run inference
        # The generator's __call__ expects (latent_path, output_path, frame_ids)
        # It handles data loading internally based on config.data.rgb_path
        try:
           generator(
               cur_config.generation.latents_path,
               cur_config.generation.output_path,
               frame_ids=frame_ids
           )
        except Exception as e:
            print(f"[Rank {rank}] [ERROR] Failed to process {video_name}: {e}")
            import traceback
            traceback.print_exc()

    
    cleanup_ddp()
    print(f"[Rank {rank}] Finished.")

if __name__ == "__main__":
    main()
