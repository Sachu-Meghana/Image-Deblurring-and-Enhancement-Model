!pip install torch==1.12.1 torchvision==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu113
!pip install opencv-python-headless==4.6.0.66
!pip install numpy==1.23.5
!pip install pillow==9.4.0
!pip install tqdm==4.64.1
!pip install basicsr
!pip install facexlib
!pip install gfpgan
!pip install realesrgan
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet  # Note the correct capitalization

def load_enhancer(device='cuda', scale=4):
    """Load Real-ESRGAN model with correct architecture"""
    model = RRDBNet(  # Fixed the class name here
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=scale
    )
    
    model_urls = {
        2: 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
        4: 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
    }
    
    try:
        upsampler = RealESRGANer(
            scale=scale,
            model_path=model_urls[scale],
            model=model,
            device=device,
            tile=400,
            tile_pad=10,
            pre_pad=0,
            half=False
        )
        return upsampler
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        return None

def adaptive_sharpen(img, strength=0.8):
    """Fixed version with proper type handling"""
    # Convert to float32 for processing
    img_float = img.astype(np.float32) / 255.0
    
    # Create blurred version
    blurred = cv2.GaussianBlur(img_float, (0, 0), 3)
    
    # Create edge mask
    gray = cv2.cvtColor(img_float, cv2.COLOR_BGR2GRAY)
    edges = cv2.Laplacian(gray, cv2.CV_32F)
    edge_mask = np.abs(edges)
    edge_mask = cv2.normalize(edge_mask, None, 0, 1, cv2.NORM_MINMAX)
    
    # Ensure mask is 3-channel for color images
    edge_mask = np.stack([edge_mask]*3, axis=-1)
    
    # Apply sharpening
    sharpened = cv2.addWeighted(
        img_float, 1.0 + strength,
        blurred, -strength,
        0
    )
    
    # Blend based on edge strength
    result = img_float * (1 - edge_mask) + sharpened * edge_mask
    
    # Convert back to uint8
    return (np.clip(result, 0, 1) * 255).astype(np.uint8)

def enhance_clarity(img_path, output_path, upsampler):
    """Complete clarity enhancement pipeline"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error reading {img_path}")
            return False
        
        # Convert RGBA to RGB if needed
        if img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        # Stage 1: Super-resolution
        torch.cuda.empty_cache()
        img_sr, _ = upsampler.enhance(img, outscale=1)
        
        # Stage 2: Adaptive sharpening
        img_sharp = adaptive_sharpen(img_sr)
        
        # Stage 3: Contrast enhancement
        lab = cv2.cvtColor(img_sharp, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        final = cv2.cvtColor(cv2.merge([l_enhanced, a, b]), cv2.COLOR_LAB2BGR)
        
        cv2.imwrite(output_path, final)
        return True
        
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        return False

def process_folder(input_dir, output_dir, scale=4):
    """Process all images in a folder"""
    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading Real-ESRGAN {scale}x model...")
    upsampler = load_enhancer(device, scale)
    if upsampler is None:
        return
    
    valid_ext = ('.png', '.jpg', '.jpeg', '.webp')
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_ext)]
    
    if not files:
        print("No valid images found")
        return
    
    success = 0
    for filename in tqdm(files):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        if enhance_clarity(input_path, output_path, upsampler):
            success += 1
    
    print(f"Processed {success}/{len(files)} successfully")

if __name__ == "__main__":
    INPUT_DIR = "/kaggle/input/images/deblurred_defocused"
    OUTPUT_DIR = "enhanced_output"
    SCALE = 4  # Use 2 for less aggressive enhancement
    
    process_folder(INPUT_DIR, OUTPUT_DIR, SCALE)