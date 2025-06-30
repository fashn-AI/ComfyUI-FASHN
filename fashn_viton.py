import base64
import os
import time
from io import BytesIO

import numpy as np
import requests
import torch
from PIL import Image

from comfy.utils import ProgressBar


class FASHN:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_image": ("IMAGE",),
                "garment_image": ("IMAGE",),
            },
            "optional": {
                "category": (["tops", "bottoms", "one-pieces", "auto"], {"default": "auto"}),
                "mode": (["performance", "balanced", "quality"], {"default": "balanced"}),
                "garment_photo_type": (["auto", "model", "flat-lay"], {"default": "auto"}),
                "moderation_level": (["none", "permissive", "conservative"], {"default": "permissive"}),
                "segmentation_free": ("BOOLEAN", {"default": True}),
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
        img.save(buffered, format="JPEG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"

    @staticmethod
    def loadimage_to_pil(img_tensor_bhwc: torch.Tensor):
        img_np = img_tensor_bhwc.squeeze(0).numpy()
        return Image.fromarray((img_np * 255).astype(np.uint8))

    @staticmethod
    def pil_load_image_from_http(session, url: str) -> Image.Image:
        response = session.get(url, stream=True)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if not content_type.startswith("image/"):
            raise ValueError(f"The URL's Content-Type is not an image. Content-Type: {content_type}")

        img_bytes = BytesIO(response.content)
        with Image.open(img_bytes) as img:
            img.load()
            return img.copy()

    @staticmethod
    def pil_to_torch_hwc(img: Image.Image):
        img = np.array(img)
        img = torch.from_numpy(img).to(dtype=torch.float32) / 255.0
        return img

    @staticmethod
    def shorten_string(s: str, max_len: int = 50):
        return s[:max_len] + "..." if len(s) > max_len else s

    @staticmethod
    def make_api_request(session, url, headers, data=None, method="GET", max_retries=3, timeout=60):
        for attempt in range(max_retries):
            try:
                if method.upper() == "GET":
                    response = session.get(url, headers=headers, timeout=timeout)
                elif method.upper() == "POST":
                    response = session.post(url, headers=headers, json=data, timeout=timeout)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise Exception(f"API call failed after {max_retries} attempts: {str(e)}") from e
                print(f"Attempt {attempt + 1} failed. Retrying...")
                time.sleep(2)

    def fashn_tryon(
        self,
        model_image,
        garment_image,
        model_name="tryon-v1.6",
        category="auto",
        mode="balanced",
        garment_photo_type="auto",
        moderation_level="permissive",
        segmentation_free=True,
        seed=42,
        num_samples=1,
        fashn_api_key=None,
    ):
        ENDPOINT_URL = os.getenv("FASHN_ENDPOINT_URL", "https://api.fashn.ai/v1")
        API_KEY = fashn_api_key or os.getenv("FASHN_API_KEY")

        if not API_KEY:
            raise ValueError("FASHN_API_KEY must be set in environment variables or provided as fashn_api_key.")

        def process_image(image):
            if isinstance(image, str) and (image.startswith("http://") or image.startswith("https://")):
                return image
            else:
                img = self.loadimage_to_pil(image)
                return self.encode_img_to_base64(img)

        model_image = process_image(model_image)
        garment_image = process_image(garment_image)

        if seed > 2**32:
            seed = int(seed & 0xFFFFFFFF)

        # Prepare API request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        }

        inputs = {
            "model_image": model_image,
            "garment_image": garment_image,
            "category": category,
            "mode": mode,
            "garment_photo_type": garment_photo_type,
            "moderation_level": moderation_level,
            "segmentation_free": segmentation_free,
            "seed": seed,
            "num_samples": num_samples,
        }

        # Prepare API request data
        api_data = {
            "model_name": model_name,
            "inputs": inputs
        }

        # Estimate processing time and initialize progress bar
        if mode == "performance":
            base_time = 7
        elif mode == "quality":
            base_time = 19
        else:  # balanced or default
            base_time = 10

        # Estimate poll time: base_time * (n+2)/3, ensure minimum of 1s
        estimated_poll_time = max(1.0, base_time * (num_samples + 2) / 3.0)

        pbar = ProgressBar(100) # Progress bar represents percentage

        # Make API request
        session = requests.Session()
        try:
            response_data = self.make_api_request(
                session, f"{ENDPOINT_URL}/run", headers=headers, data=api_data, method="POST"
            )
            pred_id = response_data.get("id")
        except Exception as e:
            # Shorten image strings for error reporting
            error_data = api_data.copy()
            error_data["inputs"]["model_image"] = self.shorten_string(error_data["inputs"]["model_image"])
            error_data["inputs"]["garment_image"] = self.shorten_string(error_data["inputs"]["garment_image"])
            raise Exception(f"API call failed: {str(e)} - Req Body: {error_data}") from e

        # Poll the status of the prediction
        start_poll_time = time.time()
        while True:
            # Check timeout relative to polling start time
            if time.time() - start_poll_time > 180:  # 3 minutes timeout
                raise Exception("Maximum polling time exceeded.")

            try:
                status_data = self.make_api_request(
                    session, f"{ENDPOINT_URL}/status/{pred_id}", headers=headers, method="GET"
                )
            except Exception as e:
                raise Exception(f"Status check failed: {str(e)}") from e

            if status_data["status"] == "completed":
                break
            elif status_data["status"] not in ["starting", "in_queue", "processing"]:
                raise Exception(f"Prediction failed with id {pred_id}: {status_data.get('error')}. Inputs: {api_data['inputs']}")

            # Update progress bar based on elapsed time vs estimated time
            elapsed_poll_time = time.time() - start_poll_time
            # Ensure progress doesn't exceed 99% during polling to leave room for final step
            expected_progress = min(99, int((elapsed_poll_time / estimated_poll_time) * 100))
            increment = expected_progress - pbar.current
            if increment > 0:
                 pbar.update(increment)

            time.sleep(2) # Original sleep interval

        # Ensure pbar reaches 100% on successful completion
        if pbar.current < 100:
            pbar.update(100 - pbar.current)

        # Get the result images
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
