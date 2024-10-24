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
                "category": (["tops", "bottoms", "one-pieces"],),
            },
            "optional": {
                "flat_lay": ("BOOLEAN", {"default": False}),
                "nsfw_filter": ("BOOLEAN", {"default": True}),
                "cover_feet": ("BOOLEAN", {"default": False}),
                "adjust_hands": ("BOOLEAN", {"default": False}),
                "restore_background": ("BOOLEAN", {"default": False}),
                "restore_clothes": ("BOOLEAN", {"default": False}),
                "remove_garment_background": ("BOOLEAN", {"default": False}),
                "guidance_scale": ("FLOAT", {"default": 3.0, "min": 1.5, "max": 5.0, "step": 0.1}),
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

    @staticmethod
    def shorten_string(s: str, max_len: int = 50):
        return s[:max_len] + "..." if len(s) > max_len else s

    @staticmethod
    def make_api_request(session, url, headers, data=None, method='GET', max_retries=3, timeout=60):
        for attempt in range(max_retries):
            try:
                if method.upper() == 'GET':
                    response = session.get(url, headers=headers, timeout=timeout)
                elif method.upper() == 'POST':
                    response = session.post(url, headers=headers, json=data, timeout=timeout)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:  # If it's the last attempt
                    raise Exception(f"API call failed after {max_retries} attempts: {str(e)}") from e
                print(f"Attempt {attempt + 1} failed. Retrying...")
                time.sleep(2)  # Wait for 2 seconds before retrying

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
        remove_garment_background,
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
        def process_image(image):
            if isinstance(image, str) and (image.startswith("http://") or image.startswith("https://")):
                return image  # It's a URL, don't preprocess
            else:
                img = self.loadimage_to_pil(image)
                return self.encode_img_to_base64(img)

        model_image = process_image(model_image)
        garment_image = process_image(garment_image)

        pbar.update(1)

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
            "remove_garment_background": remove_garment_background,
            "guidance_scale": guidance_scale,
            "timesteps": timesteps,
            "seed": seed,
            "num_samples": num_samples,
        }

        # Make API request
        session = requests.Session()
        try:
            response_data = self.make_api_request(
                session,
                f"{ENDPOINT_URL}/run",
                headers=headers,
                data=inputs,
                method='POST'
            )
            pred_id = response_data.get("id")
        except Exception as e:
            inputs["model_image"] = self.shorten_string(inputs["model_image"])
            inputs["garment_image"] = self.shorten_string(inputs["garment_image"])
            raise Exception(f"API call failed: {str(e)} - Req Body: {inputs}") from e
        pbar.update(1)

        # Poll the status of the prediction
        start_time = time.time()
        while True:
            if time.time() - start_time > 180:  # 3 minutes timeout
                raise Exception("Maximum polling time exceeded.")

            try:
                status_data = self.make_api_request(
                    session,
                    f"{ENDPOINT_URL}/status/{pred_id}",
                    headers=headers,
                    method='GET'
                )
            except Exception as e:
                raise Exception(f"Status check failed: {str(e)}") from e

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
