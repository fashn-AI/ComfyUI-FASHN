import base64
import os
import time
from io import BytesIO

import numpy as np
import requests
import torch
from comfy.utils import ProgressBar
from PIL import Image


class FASHN:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_image": ("IMAGE",),
                "garment_image": ("IMAGE",),
                "category": (["tops", "bottoms", "one-pieces"],),
            },
            "optional": {
                "flat_lay": ("BOOLEAN", {"default": False}),
                "nsfw_filter": ("BOOLEAN", {"default": True}),
                "cover_feet": ("BOOLEAN", {"default": False}),
                "adjust_hands": ("BOOLEAN", {"default": False}),
                "restore_background": ("BOOLEAN", {"default": False}),
                "restore_clothes": ("BOOLEAN", {"default": False}),
                "guidance_scale": ("FLOAT", {"default": 2.5, "min": 1.5, "max": 5.0, "step": 0.1}),
                "timesteps": ("INT", {"default": 50, "min": 20, "max": 50, "step": 1}),
                "seed": ("INT", {"default": 42}),
                "num_samples": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "fashn_api_key": ("STRING", {"multiline": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "fashn_tryon"
    CATEGORY = "FASHN AI"

    @staticmethod
    def encode_img_to_base64(img):
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"

    @staticmethod
    def maybe_resize_image(img: Image.Image, target_width: int = 384, target_height: int = 576):
        if img.width > target_width or img.height > target_height:
            img.thumbnail((target_width, target_height), resample=Image.Resampling.LANCZOS)

        return img

    @staticmethod
    def loadimage_to_pil(img_tensor_bhwc: torch.Tensor):
        img_np = img_tensor_bhwc.squeeze(0).numpy()
        return Image.fromarray((img_np * 255).astype(np.uint8))

    @staticmethod
    def pil_load_image_from_http(session, url: str) -> Image.Image:
        response = session.get(url, stream=True)
        response.raise_for_status()

        # Check if the content type is an image
        content_type = response.headers.get("Content-Type", "")
        if not content_type.startswith("image/"):
            raise ValueError(f"The URL's Content-Type is not an image. Content-Type: {content_type}")

        # Use BytesIO to create an in-memory binary stream
        img_bytes = BytesIO(response.content)
        with Image.open(img_bytes) as img:
            img.load()  # Load the image here to ensure it can be opened
            return img.copy()  # Return a copy to ensure the file pointer doesn't affect the image

    @staticmethod
    def pil_to_torch_hwc(img: Image.Image):
        img = np.array(img)
        img = torch.from_numpy(img).to(dtype=torch.float32) / 255.0
        return img

    def fashn_tryon(
        self,
        model_image,
        garment_image,
        category,
        flat_lay,
        nsfw_filter,
        cover_feet,
        adjust_hands,
        restore_background,
        restore_clothes,
        guidance_scale,
        timesteps,
        seed,
        num_samples,
        fashn_api_key=None,
    ):
        # Environment variables
        ENDPOINT_URL = os.getenv("FASHN_ENDPOINT_URL", "https://api.fashn.ai/v1")
        API_KEY = fashn_api_key or os.getenv("FASHN_API_KEY")

        if not API_KEY:
            raise ValueError("FASHN_API_KEY must be set in environment variables or provided as fashn_api_key.")

        # Progress bar
        pbar = ProgressBar(total=7 + num_samples)

        # Preprocess images
        model_image, garment_image = map(self.loadimage_to_pil, [model_image, garment_image])
        model_image, garment_image = map(self.maybe_resize_image, [model_image, garment_image])
        model_image, garment_image = map(self.encode_img_to_base64, [model_image, garment_image])

        pbar.update(1)

        # if seed is greater than 2^32, we need to convert it to a 32-bit integer
        if seed > 2**32:
            seed = int(seed & 0xFFFFFFFF)
        print(f"seed: {seed}")

        # Prepare API request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        }
        inputs = {
            "model_image": model_image,
            "garment_image": garment_image,
            "category": category,
            "flat_lay": flat_lay,
            "nsfw_filter": nsfw_filter,
            "cover_feet": cover_feet,
            "adjust_hands": adjust_hands,
            "restore_background": restore_background,
            "restore_clothes": restore_clothes,
            "guidance_scale": guidance_scale,
            "timesteps": timesteps,
            "seed": seed,
            "num_samples": num_samples,
        }

        # Make API request
        session = requests.Session()
        try:
            response = session.post(f"{ENDPOINT_URL}/run", headers=headers, json=inputs, timeout=60)
            response.raise_for_status()
            pred_id = response.json().get("id")
        except requests.exceptions.RequestException as e:
            raise Exception(f"API call failed: {str(e)} - Req Body: {inputs}") from e
        pbar.update(1)

        # Poll the status of the prediction
        start_time = time.time()
        while True:
            if time.time() - start_time > 180:  # 3 minutes timeout
                raise Exception("Maximum polling time exceeded.")

            status_response = session.get(f"{ENDPOINT_URL}/status/{pred_id}", headers=headers, timeout=60)
            status_response.raise_for_status()
            status_data = status_response.json()

            if status_data["status"] == "completed":
                break
            elif status_data["status"] not in ["starting", "in_queue", "processing"]:
                raise Exception(f"Prediction failed with id {pred_id}: {status_data.get('error')}. Inputs: {inputs}")

            pbar.update(1)
            time.sleep(3)

        # Get the result images
        pbar.update(1)
        urls = status_data["output"]
        result_imgs = []
        for output_url in urls:
            pil_img = self.pil_load_image_from_http(session, output_url)
            result_imgs.append(self.pil_to_torch_hwc(pil_img))

        session.close()

        result_tensor = torch.stack(result_imgs, dim=0)
        return (result_tensor,)


NODE_CLASS_MAPPINGS = {"FASHN": FASHN}

NODE_DISPLAY_NAME_MAPPINGS = {"FASHN": "FASHN Virtual Try-On"}
