# Huggingface Hub validation - must be first
try:
    import huggingface_hub
    print("huggingface_hub path:", huggingface_hub.__file__)
    print("Huggingface Hub version:", huggingface_hub.__version__)
    # Check for hf_hub_download instead of deprecated cached_download
    assert hasattr(huggingface_hub, "hf_hub_download"), "hf_hub_download missing from huggingface_hub"
except Exception as e:
    raise ImportError(f"Your Python environment is broken or corrupted: {e}. Reinstall the correct huggingface_hub.")

import os
import shutil
import tarfile
import zipfile
import mimetypes
import torch
import numpy as np
import requests
from PIL import Image
from typing import List, Optional
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
from weights_downloader import WeightsDownloader
from cog_model_helpers import optimise_images
from config import config
from diffusers import WanPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video

# Environment setup
os.environ["DOWNLOAD_LATEST_WEIGHTS_MANIFEST"] = "true"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

mimetypes.add_type("image/webp", ".webp")
mimetypes.add_type("video/webm", ".webm")

# Directory setup
OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

IMAGE_TYPES = [".jpg", ".jpeg", ".png", ".webp"]
VIDEO_TYPES = [".mp4", ".mov", ".avi", ".mkv", ".webm"]

# Wan2.2 model configuration
WAN22_REPO = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
LORA_HIGH_NOISE_PATH = "https://odb96bm9wznk9riz.public.blob.vercel-storage.com/Instagirlv2.0_hinoise.safetensors"
LORA_LOW_NOISE_PATH = "https://odb96bm9wznk9riz.public.blob.vercel-storage.com/Instagirlv2.0_lownoise.safetensors"
UPLOAD_URL = "https://jerrrycans-file.hf.space/upload"

# Load example workflow if it exists
try:
    with open("examples/api_workflows/birefnet_api.json", "r") as file:
        EXAMPLE_WORKFLOW_JSON = file.read()
except FileNotFoundError:
    EXAMPLE_WORKFLOW_JSON = "{}"


class Predictor(BasePredictor):
    def setup(self, weights: Optional[str] = None):
        """Setup both ComfyUI and Wan2.2 pipeline"""
        print("Setting up Wan2.2-T2V-A14B with ComfyUI integration...")
        
        # Handle user weights if provided
        if weights and bool(weights):
            self.handle_user_weights(weights)

        # Initialize ComfyUI
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)
        
        # Setup Wan2.2 pipeline
        self.setup_wan_pipeline()
        
        print("Setup complete!")

    def setup_wan_pipeline(self):
        """Setup Wan2.2-T2V-A14B pipeline with quality optimizations"""
        print("Loading VAE with maximum quality settings...")
        vae = AutoencoderKLWan.from_pretrained(
            WAN22_REPO, 
            subfolder="vae", 
            torch_dtype=torch.float32  # Use float32 for VAE quality
        )
        
        print("Loading Wan2.2 pipeline...")
        self.pipe = WanPipeline.from_pretrained(
            WAN22_REPO,
            vae=vae,
            torch_dtype=torch.bfloat16,  # bfloat16 for main pipeline
            use_safetensors=True
        )
        self.pipe.to("cuda")
        
        # Enable memory efficient attention and CPU offload for quality
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_attention_slicing()
        if hasattr(self.pipe, 'enable_vae_slicing'):
            self.pipe.enable_vae_slicing()
        
        # Optimize scheduler for better quality
        try:
            from diffusers import FlowMatchEulerDiscreteScheduler
            self.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
                self.pipe.scheduler.config,
                timestep_spacing="trailing",
                prediction_type="flow_prediction"
            )
            print("Using optimized FlowMatchEulerDiscreteScheduler")
        except ImportError:
            print("Using default scheduler")
        
        # Download and load LoRAs
        print("Loading LoRAs...")
        self.download_and_load_loras()
        
        self.pipe.set_progress_bar_config(disable=True)

    def download_and_load_loras(self):
        """Download and load both LoRAs for Instagirl V2.0 with optimized quality settings"""
        # Download high-noise LoRA
        lora_high_path = "/tmp/lora_high_noise.safetensors"
        if not os.path.exists(lora_high_path):
            print("Downloading high-noise LoRA...")
            response = requests.get(LORA_HIGH_NOISE_PATH, stream=True)
            response.raise_for_status()
            with open(lora_high_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        # Download low-noise LoRA
        lora_low_path = "/tmp/lora_low_noise.safetensors"
        if not os.path.exists(lora_low_path):
            print("Downloading low-noise LoRA...")
            response = requests.get(LORA_LOW_NOISE_PATH, stream=True)
            response.raise_for_status()
            with open(lora_low_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        # Load both LoRAs with balanced weights for optimal quality
        try:
            self.pipe.load_lora_weights(lora_high_path, adapter_name="high_noise")
            self.pipe.load_lora_weights(lora_low_path, adapter_name="low_noise") 
            # Optimized weights: balanced for better quality and coherence
            self.pipe.set_adapters(["high_noise", "low_noise"], adapter_weights=[0.8, 1.0])
            print("LoRAs loaded with quality-optimized weights (0.8, 1.0)")
        except Exception as e:
            print(f"Error loading dual LoRAs: {e}")
            # Fallback to single LoRA
            try:
                self.pipe.load_lora_weights(lora_low_path)
                print("Loaded low-noise LoRA as fallback")
            except Exception as e2:
                print(f"LoRA loading failed entirely: {e2}")

    def handle_user_weights(self, weights: str):
        """Handle user-provided weights (from original ComfyUI implementation)"""
        if hasattr(weights, "url"):
            if weights.url.startswith("http"):
                weights_url = weights.url
            else:
                weights_url = "https://replicate.delivery/" + weights.url
        else:
            weights_url = weights

        print(f"Downloading user weights from: {weights_url}")
        WeightsDownloader.download_weights(weights_url)

    def upload_file(self, file_path: str, is_video: bool = False) -> str:
        """Upload file to specified URL with better error handling"""
        try:
            content_type = "video/mp4" if is_video else "image/png"
            with open(file_path, "rb") as f:
                files = {"file": (os.path.basename(file_path), f, content_type)}
                response = requests.post(UPLOAD_URL, files=files, timeout=300)
            
            if response.status_code == 200:
                json_response = response.json()
                if "url" in json_response:
                    return "https://jerrrycans-file.hf.space" + json_response["url"]
            
            print(f"Upload failed: {response.status_code} - {response.text}")
            return ""
        except Exception as e:
            print(f"Upload error: {e}")
            return ""

    def cleanup_directories(self):
        """Clean up temporary directories"""
        for directory in ALL_DIRECTORIES:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory, exist_ok=True)

    def predict(
        self,
        prompt: str = Input(
            description="Prompt (use Instagirl trigger for best results)",
            default="Instagirl, photorealistic portrait, natural lighting, high quality, detailed"
        ),
        negative_prompt: str = Input(
            description="Negative prompt",
            default="blurry, low quality, distorted, deformed, ugly, bad anatomy, worst quality, low resolution, jpeg artifacts, oversaturated, noise, grain"
        ),
        generate_video: bool = Input(
            description="Generate full video (unchecked = first frame only as image)",
            default=False
        ),
        width: int = Input(description="Width", default=1280, ge=256, le=2048),
        height: int = Input(description="Height", default=720, ge=256, le=2048),
        num_frames: int = Input(description="Number of frames (video only)", default=81, ge=1, le=121),
        num_inference_steps: int = Input(description="Steps", default=40, ge=20, le=100),
        guidance_scale: float = Input(description="Guidance scale", default=2.5, ge=1.0, le=8.0),
        seed: int = Input(description="Seed", default=None),
        use_comfyui_workflow: bool = Input(
            description="Use ComfyUI workflow (if False, uses direct Wan2.2 pipeline)",
            default=False
        ),
        workflow_json: str = Input(
            description="ComfyUI workflow JSON (optional)",
            default=""
        )
    ) -> List[Path]:
        
        # Clean up directories
        self.cleanup_directories()
        
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF
        
        generator = torch.Generator("cuda").manual_seed(seed)
        
        # Quality-optimized parameters for Wan2.2
        if generate_video:
            frames_to_use = num_frames
            steps_to_use = max(35, num_inference_steps)
            guidance_to_use = min(guidance_scale, 3.0)
        else:
            frames_to_use = 1
            steps_to_use = max(40, num_inference_steps)
            guidance_to_use = min(guidance_scale, 2.8)
        
        print(f"Generating with prompt: {prompt[:80]}...")
        print(f"Video mode: {generate_video}")
        print(f"Resolution: {width}x{height}")
        print(f"Frames: {frames_to_use}")
        print(f"Steps: {steps_to_use}")
        print(f"Guidance: {guidance_to_use}")
        print(f"Seed: {seed}")
        print(f"Use ComfyUI workflow: {use_comfyui_workflow}")
        
        outputs = []
        
        if use_comfyui_workflow and workflow_json:
            # Use ComfyUI workflow
            print("Using ComfyUI workflow...")
            try:
                workflow = workflow_json if workflow_json else EXAMPLE_WORKFLOW_JSON
                wf = self.comfyUI.load_workflow(workflow)
                
                # Update workflow with parameters
                if hasattr(wf, 'set_prompt'):
                    wf.set_prompt(prompt)
                if hasattr(wf, 'set_negative_prompt'):
                    wf.set_negative_prompt(negative_prompt)
                
                results = self.comfyUI.run_workflow(wf)
                
                # Process ComfyUI results
                for result in results:
                    if isinstance(result, str) and os.path.exists(result):
                        outputs.append(Path(result))
                        
            except Exception as e:
                print(f"ComfyUI workflow failed: {e}")
                print("Falling back to direct Wan2.2 pipeline...")
                use_comfyui_workflow = False
        
        if not use_comfyui_workflow:
            # Use direct Wan2.2 pipeline with quality optimizations
            print("Using direct Wan2.2 pipeline...")
            
            # Generate with quality-optimized settings
            with torch.cuda.amp.autocast(enabled=True):
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    num_frames=frames_to_use,
                    guidance_scale=guidance_to_use,
                    num_inference_steps=steps_to_use,
                    generator=generator,
                    output_type="pil"
                )
            
            print(f"Generation completed. Frames generated: {len(result.frames[0])}")
            
            if generate_video:
                # Export video with higher quality settings
                out_path = os.path.join(OUTPUT_DIR, "output.mp4")
                export_to_video(
                    result.frames[0], 
                    out_path, 
                    fps=24,
                    video_codec="libx264",
                    options={
                        "crf": "18",  # High quality
                        "preset": "slow"  # Better compression
                    }
                )
                outputs.append(Path(out_path))
                
                # Upload video
                file_url = self.upload_file(out_path, is_video=True)
                if file_url:
                    print(f"Video uploaded: {file_url}")
            else:
                # Export single frame with maximum quality
                frame = result.frames[0][0]
                
                # High-quality image processing
                if hasattr(frame, 'save'):
                    image = frame
                else:
                    if isinstance(frame, torch.Tensor):
                        frame = frame.cpu().numpy()
                    
                    if isinstance(frame, np.ndarray):
                        if frame.dtype in [np.float32, np.float64]:
                            frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
                        elif frame.dtype != np.uint8:
                            frame = frame.astype(np.uint8)
                        
                        image = Image.fromarray(frame)
                    else:
                        image = frame
                
                out_path = os.path.join(OUTPUT_DIR, "output.png")
                image.save(out_path, "PNG", optimize=True, compress_level=1)
                outputs.append(Path(out_path))
                
                # Upload image
                file_url = self.upload_file(out_path, is_video=False)
                if file_url:
                    print(f"Image uploaded: {file_url}")
        
        # Optimize images for better quality
        if outputs:
            try:
                outputs = optimise_images(outputs)
            except Exception as e:
                print(f"Image optimization failed: {e}")
        
        return outputs
