import numpy as np
import cv2
import os
from scipy.fft import fft2, ifft2

def motion_blur_kernel(size=30, angle=0, length=20):
    """Generate directional motion blur kernel"""
    kernel = np.zeros((size, size))
    center = size // 2
    for i in range(length):
        x = int(center + (i - length//2) * np.cos(np.radians(angle)))
        y = int(center + (i - length//2) * np.sin(np.radians(angle)))
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1
    return kernel / np.sum(kernel)

def defocus_blur_kernel(size=30, radius=10):
    """Generate circular defocus blur kernel"""
    kernel = np.zeros((size, size), dtype=np.float32)
    cv2.circle(kernel, (size//2, size//2), radius, 1, -1)
    return kernel / np.sum(kernel)

def wiener_deconvolution(img, kernel, K=0.0025):
    """Color image deconvolution with Wiener filter"""
    deblurred = np.zeros_like(img, dtype=np.float32)
    for i in range(3):  # Process each channel
        channel = img[:,:,i].astype(np.float32)
        
        # Pad kernel to match image size
        pad_kernel = np.zeros_like(channel)
        kh, kw = kernel.shape
        pad_kernel[:kh, :kw] = kernel
        pad_kernel = np.roll(pad_kernel, (-kh//2, -kw//2), axis=(0,1))
        
        # Frequency domain processing
        img_fft = fft2(channel)
        kernel_fft = fft2(pad_kernel)
        deblurred_fft = (img_fft * np.conj(kernel_fft)) / (np.abs(kernel_fft)**2 + K)
        deblurred_ch = np.abs(ifft2(deblurred_fft))
        
        # Normalize and store
        deblurred[:,:,i] = cv2.normalize(deblurred_ch, None, 0, 255, cv2.NORM_MINMAX)
    return np.clip(deblurred, 0, 255).astype(np.uint8)

def post_process(img, brightness_factor=1.5, gamma=0.8):
    """Enhance brightness, contrast, and sharpness of the deblurred image."""
    # Convert to LAB and extract channels
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Brighten the L channel (Luminance)
    l = cv2.addWeighted(l, brightness_factor, np.zeros_like(l), 0, 30)  # Increase brightness

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    # Merge and convert back
    bright_lab = cv2.merge((l, a, b))
    bright_img = cv2.cvtColor(bright_lab, cv2.COLOR_LAB2BGR)

    # Gamma Correction (Boosts overall brightness)
    gamma_table = np.array([(i / 255.0) ** (1.0 / gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    bright_img = cv2.LUT(bright_img, gamma_table)

    # Apply Stronger Sharpening
    sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # Stronger sharpening filter
    sharp_img = cv2.filter2D(bright_img, -1, sharpening_kernel)

    return sharp_img

def process_folder(input_dir, output_dir, blur_type):
    """Process all images in folder with specified blur type"""
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"⚠️ Failed to load {filename}")
            continue
            
        try:
            # Select kernel based on blur type
            if blur_type == 'motion':
                kernel = motion_blur_kernel(size=35, angle=0, length=25)  # Horizontal motion
            else:  # defocus
                kernel = defocus_blur_kernel(size=35, radius=12)
            
            # Apply deconvolution
            deblurred = wiener_deconvolution(img, kernel)
            
            # Post-processing
            final = post_process(deblurred)
            
            # Save result
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, final)
            print(f"✅ Processed {filename} ({blur_type})")
            
        except Exception as e:
            print(f"❌ Failed {filename}: {str(e)}")
            # Save original as fallback
            cv2.imwrite(os.path.join(output_dir, f"original_{filename}"), img)

if __name__ == "__main__":
    # Configuration - UPDATE THESE PATHS
    DEFOCUS_INPUT = "/kaggle/input/defocused/defocused"
    MOTION_INPUT = "/kaggle/input/motion-blur/motion_blur"
    OUTPUT_DIR = "/kaggle/working/output_folder"
    
    # Process each folder
    print("Processing defocused images...")
    process_folder(DEFOCUS_INPUT, os.path.join(OUTPUT_DIR, "deblurred_defocused"), 'defocus')
    
    print("\nProcessing motion blur images...")
    process_folder(MOTION_INPUT, os.path.join(OUTPUT_DIR, "deblurred_motion"), 'motion')
    
    print("\nProcessing complete. Results saved in:")
    print(f"- Defocused: {os.path.join(OUTPUT_DIR, 'deblurred_defocused')}")
    print(f"- Motion: {os.path.join(OUTPUT_DIR, 'deblurred_motion')}")