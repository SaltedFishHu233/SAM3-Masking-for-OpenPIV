
import os
from xml.parsers.expat import model

import matplotlib.pyplot as plt
import numpy as np
from wcwidth import width

import sam3
from PIL import Image,ImageOps,ImageEnhance
from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results

import torch
import cv2


def SAM3Mask(ImgDIR, Prompt):
    """
    Process images in the specified directory using SAM3 masking.
    
    Args:
        ImgDIR: Path to the directory containing images
        Prompt: Text prompt for object detection (e.g., "Black Shape")
    """

    #Check for CUDA availability
    print(f"Is CUDA available: {torch.cuda.is_available()}")

    if not (torch.cuda.is_available()):
        print("No CUDA-compatible GPU found. Ending.")
        return None
    
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Processing images from: {ImgDIR}")
    # Add your SAM3 masking logic here

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load the model
    model = build_sam3_image_model()
    processor = Sam3Processor(model,confidence_threshold=0.5)

    #load image
    image = Image.open(ImgDIR).convert('RGB')
    
    # Get image dimensions
    depth, height, width = np.array(image).shape
    print(f"Depth: {depth}")
    print(f"Height: {height}") 
    print(f"Width: {width}")  

    # Add a white border around the image
    img_with_border = ImageOps.expand(image,border=300,fill='white')

    # Process the image with Pillow to enhance brightness
    EnhancerInstance = ImageEnhance.Brightness(img_with_border)
    ImageInput=EnhancerInstance.enhance(1)

    # Run inference with autocast for mixed precision
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        inference_state = processor.set_image(img_with_border)
        output = processor.set_text_prompt(state=inference_state, prompt=Prompt)
        masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

    # Check if any masks were detected
    if len(masks) == 0:
        print(f"No objects of the prompt detected in the image.")
        return None
    
    print(f"Detected")    

    # Print the scores for debugging
    print(scores)
    
    # Convert masks to numpy arrays
    masks_np = [mask.squeeze().cpu().numpy() for mask in masks]    
    IndivMask = masks_np[0]  # Assuming you want to use the first mask for cropping

    # Crop the mask to the original image dimensions
    CroppedMask=IndivMask[300:depth+300, 300:height+300]

    return CroppedMask






# if __name__ == "__main__":
#     image_directory = "/path/to/images"
#     SAM3Mask(image_directory)