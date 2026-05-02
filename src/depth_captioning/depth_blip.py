
import os
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration
from .depth_kosmos import DepthContextCreator
from .spatial_analysis import SpatialAnalyzer

class BlipCaptioner:
    def __init__(self, ckpt="Salesforce/blip-image-captioning-base", device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f"Using device for BLIP: {self.device}")
        self.processor = AutoProcessor.from_pretrained(ckpt)
        self.model = BlipForConditionalGeneration.from_pretrained(ckpt).to(self.device)

    def get_caption(self, image_array):
        image = Image.fromarray(image_array.astype("uint8"))
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        generated_ids = self.model.generate(**inputs)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return generated_text

class DepthBlipCaptioner:
    def __init__(self, ckpt="Salesforce/blip-image-captioning-base", device=None, encoder="vits", yolo_model_path="yolov8n.pt"):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"Initializing DepthContextCreator with encoder='{encoder}'...")
        # Import DepthContextCreator from existing module to reuse depth logic
        self.depth_context = DepthContextCreator(encoder=encoder)
        
        print(f"Initializing BlipCaptioner with ckpt='{ckpt}'...")
        self.captioner = BlipCaptioner(ckpt=ckpt, device=self.device)
        
        print("Initializing SpatialAnalyzer...")
        self.spatial_analyzer = SpatialAnalyzer(model_path=yolo_model_path)
        
        self.location = ["Closest", "Farthest", "Mid Range"]

    def get_caption_with_depth(self, image, top_threshold=70, bottom_threshold=30):
        # Generate depth segmented images and masks
        images, masks = self.depth_context.make_depth_context_img(
            image,
            top_threshold=top_threshold,
            bottom_threshold=bottom_threshold,
            return_masks=True,
        )
        
        # Analyze spatial relationships
        print("Analyzing spatial relationships...", flush=True)
        # Keep relations short; otherwise pairwise relations explode prompt length.
        spatial_descriptions = self.spatial_analyzer.analyze(
            np.array(image),
            masks,
            max_relations_per_layer=6,
        )
        
        full_string = ""
        for i in range(3):
            # Generate caption for each region
            caption = self.captioner.get_caption(images[i])
            spatial_info = spatial_descriptions[i]
            
            section_text = f"{self.location[i]}: {caption}"
            if spatial_info:
                section_text += f"\nSpatial Relationships: {spatial_info}"
            
            full_string += f"{section_text}\n----\n"
        return full_string

    def display_depth_images(self, image):
        # Delegate to depth_context logic if needed, or reimplement
        # Since DepthContextCreator has make_depth_context_img but not display,
        # we can just use the same logic as in depth_kosmos if needed.
        # But for now, we just implement the captioning part.
        pass
