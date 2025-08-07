# Huggingface Hub compatibility check
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
import json
from PIL import Image
from typing import List, Optional
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
from weights_downloader import WeightsDownloader
from cog_model_helpers import optimise_images
from config import config
import requests


os.environ["DOWNLOAD_LATEST_WEIGHTS_MANIFEST"] = "true"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

mimetypes.add_type("image/webp", ".webp")
mimetypes.add_type("video/webm", ".webm")

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

IMAGE_TYPES = [".jpg", ".jpeg", ".png", ".webp"]
VIDEO_TYPES = [".mp4", ".mov", ".avi", ".mkv", ".webm"]

with open("examples/api_workflows/wan22_t2v_a14b_api.json", "r") as file:
    EXAMPLE_WORKFLOW_JSON = file.read()

# Constants for LoRA downloads
LORA_HIGH_NOISE_PATH = "https://odb96bm9wznk9riz.public.blob.vercel-storage.com/Instagirlv2.0_hinoise.safetensors"
LORA_LOW_NOISE_PATH = "https://odb96bm9wznk9riz.public.blob.vercel-storage.com/Instagirlv2.0_lownoise.safetensors"
UPLOAD_URL = "https://jerrrycans-file.hf.space/upload"


class Predictor(BasePredictor):
    def setup(self, weights: str):
        if bool(weights):
            self.handle_user_weights(weights)

        # Download LoRAs
        self.download_loras()

        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

    def handle_user_weights(self, weights: str):
        if hasattr(weights, "url"):
            if weights.url.startswith("http"):
                weights_url = weights.url
            else:
                weights_url = "https://replicate.delivery/" + weights.url
        else:
            weights_url = weights

        print(f"Downloading user weights from: {weights_url}")
        WeightsDownloader.download("weights.tar", weights_url, config["USER_WEIGHTS_PATH"])
        for item in os.listdir(config["USER_WEIGHTS_PATH"]):
            source = os.path.join(config["USER_WEIGHTS_PATH"], item)
            destination = os.path.join(config["MODELS_PATH"], item)
            if os.path.isdir(source):
                if not os.path.exists(destination):
                    print(f"Moving {source} to {destination}")
                    shutil.move(source, destination)
                else:
                    for root, _, files in os.walk(source):
                        for file in files:
                            if not os.path.exists(os.path.join(destination, file)):
                                print(
                                    f"Moving {os.path.join(root, file)} to {destination}"
                                )
                                shutil.move(os.path.join(root, file), destination)
                            else:
                                print(
                                    f"Skipping {file} because it already exists in {destination}"
                                )

    def download_loras(self):
        """Download LoRAs to the models directory"""
        loras_dir = os.path.join(config["MODELS_PATH"], "loras")
        os.makedirs(loras_dir, exist_ok=True)
        
        # Download high-noise LoRA
        lora_high_path = os.path.join(loras_dir, "Instagirlv2.0_hinoise.safetensors")
        if not os.path.exists(lora_high_path):
            print("Downloading high-noise LoRA...")
            try:
                response = requests.get(LORA_HIGH_NOISE_PATH, stream=True)
                response.raise_for_status()
                with open(lora_high_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("✅ High-noise LoRA downloaded")
            except Exception as e:
                print(f"❌ Failed to download high-noise LoRA: {e}")
        
        # Download low-noise LoRA
        lora_low_path = os.path.join(loras_dir, "Instagirlv2.0_lownoise.safetensors")
        if not os.path.exists(lora_low_path):
            print("Downloading low-noise LoRA...")
            try:
                response = requests.get(LORA_LOW_NOISE_PATH, stream=True)
                response.raise_for_status()
                with open(lora_low_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("✅ Low-noise LoRA downloaded")
            except Exception as e:
                print(f"❌ Failed to download low-noise LoRA: {e}")

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

    def customize_workflow(self, workflow, prompt, negative_prompt, width, height, num_frames, steps, cfg, seed):
        """Customize the workflow with user parameters"""
        import json
        
        # Update prompts
        for node_id, node in workflow.items():
            if node.get("class_type") == "CLIPTextEncode":
                if "positive" in node.get("_meta", {}).get("title", "").lower():
                    node["inputs"]["text"] = prompt
                elif "negative" in node.get("_meta", {}).get("title", "").lower():
                    node["inputs"]["text"] = negative_prompt
            
            # Update KSampler parameters
            elif node.get("class_type") == "KSampler":
                node["inputs"]["seed"] = seed
                node["inputs"]["steps"] = steps
                node["inputs"]["cfg"] = cfg
            
            # Update video dimensions and length
            elif node.get("class_type") == "EmptyHunyuanLatentVideo":
                node["inputs"]["width"] = width
                node["inputs"]["height"] = height
                node["inputs"]["length"] = num_frames

    def handle_input_file(self, input_file: Path):
        file_extension = self.get_file_extension(input_file)

        if file_extension == ".tar":
            with tarfile.open(input_file, "r") as tar:
                tar.extractall(INPUT_DIR)
        elif file_extension == ".zip":
            with zipfile.ZipFile(input_file, "r") as zip_ref:
                zip_ref.extractall(INPUT_DIR)
        elif file_extension in IMAGE_TYPES + VIDEO_TYPES:
            shutil.copy(input_file, os.path.join(INPUT_DIR, f"input{file_extension}"))
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        print("====================================")
        print(f"Inputs uploaded to {INPUT_DIR}:")
        self.comfyUI.get_files(INPUT_DIR)
        print("====================================")

    def get_file_extension(self, input_file: Path) -> str:
        file_extension = os.path.splitext(input_file)[1].lower()
        if not file_extension:
            with open(input_file, "rb") as f:
                file_signature = f.read(4)
            if file_signature.startswith(b"\x1f\x8b"):  # gzip signature
                file_extension = ".tar"
            elif file_signature.startswith(b"PK"):  # zip signature
                file_extension = ".zip"
            else:
                try:
                    with Image.open(input_file) as img:
                        file_extension = f".{img.format.lower()}"
                        print(f"Determined file type: {file_extension}")
                except Exception as e:
                    raise ValueError(
                        f"Unable to determine file type for: {input_file}, {e}"
                    )
        return file_extension

    def predict(
        self,
        prompt: str = Input(
            description="Prompt (use Instagirl trigger)",
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
        width: int = Input(description="Width", default=1280),
        height: int = Input(description="Height", default=720),
        num_frames: int = Input(description="Number of frames (video only)", default=81, ge=1, le=121),
        num_inference_steps: int = Input(description="Steps", default=40, ge=20, le=100),
        guidance_scale: float = Input(description="Guidance scale", default=2.5, ge=1.0, le=8.0),
        seed: int = Input(description="Seed", default=None),
        workflow_json: str = Input(
            description="Your ComfyUI workflow as JSON string or URL. Leave empty to use default Wan2.2 workflow.",
            default="",
        ),
        input_file: Optional[Path] = Input(
            description="Input image, video, tar or zip file. Read guidance on workflows and input files here: https://github.com/replicate/cog-comfyui. Alternatively, you can replace inputs with URLs in your JSON workflow and the model will download them."
        ),
        return_temp_files: bool = Input(
            description="Return any temporary files, such as preprocessed controlnet images. Useful for debugging.",
            default=False,
        ),
        output_format: str = optimise_images.predict_output_format(),
        output_quality: int = optimise_images.predict_output_quality(),
        randomise_seeds: bool = Input(
            description="Automatically randomise seeds (seed, noise_seed, rand_seed)",
            default=True,
        ),
        force_reset_cache: bool = Input(
            description="Force reset the ComfyUI cache before running the workflow. Useful for debugging.",
            default=False,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        self.comfyUI.cleanup(ALL_DIRECTORIES)

        # Set seed if not provided
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF

        if input_file:
            self.handle_input_file(input_file)

        workflow_json_content = workflow_json
        if workflow_json.startswith(("http://", "https://")):
            try:
                response = requests.get(workflow_json)
                response.raise_for_status()
                workflow_json_content = response.text
            except requests.exceptions.RequestException as e:
                raise ValueError(f"Failed to download workflow JSON from URL: {e}")

        wf = self.comfyUI.load_workflow(workflow_json_content or EXAMPLE_WORKFLOW_JSON)

        # Customize workflow with user parameters
        self.customize_workflow(wf, prompt, negative_prompt, width, height, 
                               num_frames if generate_video else 1, 
                               num_inference_steps, guidance_scale, seed)

        self.comfyUI.connect()

        if force_reset_cache or not randomise_seeds:
            self.comfyUI.reset_execution_cache()

        if randomise_seeds and seed is None:
            self.comfyUI.randomise_seeds(wf)

        print(f"Generating {'video' if generate_video else 'image'} with:")
        print(f"  Prompt: {prompt[:80]}...")
        print(f"  Resolution: {width}x{height}")
        print(f"  Frames: {num_frames if generate_video else 1}")
        print(f"  Steps: {num_inference_steps}")
        print(f"  Guidance: {guidance_scale}")
        print(f"  Seed: {seed}")

        self.comfyUI.run_workflow(wf)

        output_directories = [OUTPUT_DIR]
        if return_temp_files:
            output_directories.append(COMFYUI_TEMP_OUTPUT_DIR)

        files = self.comfyUI.get_files(output_directories)
        
        # Upload files and add URLs to output
        for file_path in files:
            if str(file_path).endswith(('.mp4', '.webm')):
                file_url = self.upload_file(str(file_path), is_video=True)
                if file_url:
                    print(f"Video uploaded: {file_url}")
            elif str(file_path).endswith(('.png', '.jpg', '.jpeg', '.webp')):
                file_url = self.upload_file(str(file_path), is_video=False)
                if file_url:
                    print(f"Image uploaded: {file_url}")

        optimised_files = optimise_images.optimise_image_files(
            output_format, output_quality, files
        )
        return [Path(p) for p in optimised_files]
