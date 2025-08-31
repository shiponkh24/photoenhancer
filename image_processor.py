import os
import logging
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import json
import base64
from io import BytesIO

# Background removal availability flag
REMBG_AVAILABLE = False
logging.warning("Background removal feature is currently disabled due to dependency compatibility issues.")

class ImageProcessor:
    def __init__(self):
        self.supported_formats = {'PNG', 'JPEG', 'WEBP', 'GIF'}
        
    def get_image_info(self, filepath):
        """Get basic information about an image"""
        try:
            with Image.open(filepath) as img:
                return {
                    'width': img.width,
                    'height': img.height,
                    'format': img.format,
                    'mode': img.mode,
                    'size_mb': round(os.path.getsize(filepath) / (1024 * 1024), 2)
                }
        except Exception as e:
            logging.error(f"Error getting image info: {str(e)}")
            return None
    
    def upscale_image(self, input_path, output_path, scale_factor=2):
        """Upscale image using Lanczos resampling"""
        try:
            with Image.open(input_path) as img:
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'LA'):
                    # Keep transparency for PNG
                    pass
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Calculate new dimensions
                new_width = int(img.width * scale_factor)
                new_height = int(img.height * scale_factor)
                
                # Upscale using high-quality resampling
                upscaled = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Apply additional sharpening for better quality
                enhancer = ImageEnhance.Sharpness(upscaled)
                upscaled = enhancer.enhance(1.2)
                
                # Save with appropriate format
                if img.format == 'PNG' or img.mode in ('RGBA', 'LA'):
                    upscaled.save(output_path, 'PNG', optimize=True)
                else:
                    upscaled.save(output_path, 'JPEG', quality=95, optimize=True)
                
                return {'success': True}
                
        except Exception as e:
            logging.error(f"Upscaling error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def enhance_image(self, input_path, output_path, brightness=1.0, contrast=1.0, sharpness=1.0):
        """Enhance image with brightness, contrast, and sharpness adjustments"""
        try:
            with Image.open(input_path) as img:
                # Apply brightness enhancement
                if brightness != 1.0:
                    enhancer = ImageEnhance.Brightness(img)
                    img = enhancer.enhance(brightness)
                
                # Apply contrast enhancement
                if contrast != 1.0:
                    enhancer = ImageEnhance.Contrast(img)
                    img = enhancer.enhance(contrast)
                
                # Apply sharpness enhancement
                if sharpness != 1.0:
                    enhancer = ImageEnhance.Sharpness(img)
                    img = enhancer.enhance(sharpness)
                
                # Save with appropriate format
                if img.format == 'PNG' or img.mode in ('RGBA', 'LA'):
                    img.save(output_path, 'PNG', optimize=True)
                else:
                    img.save(output_path, 'JPEG', quality=95, optimize=True)
                
                return {'success': True}
                
        except Exception as e:
            logging.error(f"Enhancement error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def resize_image(self, input_path, output_path, width, height):
        """Resize image to specific dimensions"""
        try:
            with Image.open(input_path) as img:
                # Resize using high-quality resampling
                resized = img.resize((width, height), Image.Resampling.LANCZOS)
                
                # Save with appropriate format
                if img.format == 'PNG' or img.mode in ('RGBA', 'LA'):
                    resized.save(output_path, 'PNG', optimize=True)
                else:
                    resized.save(output_path, 'JPEG', quality=95, optimize=True)
                
                return {'success': True}
                
        except Exception as e:
            logging.error(f"Resize error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def remove_background(self, input_path, output_path):
        """Remove background using smart edge detection and color analysis"""
        try:
            with Image.open(input_path) as img:
                # Convert to RGBA for transparency support
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
                
                # Convert to numpy array for processing
                img_array = np.array(img)
                height, width = img_array.shape[:2]
                
                # Create a mask based on edge detection and color analysis
                mask = self._create_smart_background_mask(img_array)
                
                # Apply the mask to create transparency
                img_array[:, :, 3] = mask  # Set alpha channel
                
                # Convert back to PIL Image
                result_img = Image.fromarray(img_array, 'RGBA')
                
                # Save as PNG to preserve transparency
                result_img.save(output_path.replace(os.path.splitext(output_path)[1], '.png'), 'PNG', optimize=True)
                
                return {'success': True}
                
        except Exception as e:
            logging.error(f"Background removal error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _create_smart_background_mask(self, img_array):
        """Create a mask to identify background vs foreground using smart algorithms"""
        height, width = img_array.shape[:2]
        rgb_img = img_array[:, :, :3]  # Get RGB channels only
        
        # Method 1: Edge-based detection
        # Convert to grayscale for edge detection
        gray = np.dot(rgb_img, [0.2989, 0.5870, 0.1140])
        
        # Simple edge detection using gradient
        grad_x = np.gradient(gray, axis=1)
        grad_y = np.gradient(gray, axis=0)
        edges = np.sqrt(grad_x**2 + grad_y**2)
        
        # Method 2: Color uniformity analysis
        # Analyze corner regions to identify likely background colors
        corner_size = min(height, width) // 10
        
        corners = [
            rgb_img[:corner_size, :corner_size],  # Top-left
            rgb_img[:corner_size, -corner_size:],  # Top-right
            rgb_img[-corner_size:, :corner_size],  # Bottom-left
            rgb_img[-corner_size:, -corner_size:]  # Bottom-right
        ]
        
        # Get dominant colors from corners (likely background)
        background_colors = []
        for corner in corners:
            avg_color = np.mean(corner.reshape(-1, 3), axis=0)
            background_colors.append(avg_color)
        
        # Calculate overall background color
        bg_color = np.mean(background_colors, axis=0)
        
        # Method 3: Distance from background color
        color_distances = np.sqrt(np.sum((rgb_img - bg_color)**2, axis=2))
        color_threshold = np.percentile(color_distances, 70)  # Keep 70% as foreground
        
        # Combine methods to create final mask
        edge_threshold = np.percentile(edges, 85)  # Areas with strong edges are likely foreground
        
        # Create mask: 255 (opaque) for foreground, 0 (transparent) for background
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Mark as foreground if:
        # 1. Strong edges are present (detailed areas)
        # 2. Color is significantly different from background
        foreground_condition = (edges > edge_threshold) | (color_distances > color_threshold)
        mask[foreground_condition] = 255
        
        # Apply morphological operations to clean up the mask
        # Dilate to fill small gaps in foreground objects
        kernel_size = max(3, min(height, width) // 200)
        mask = self._morphological_operations(mask, kernel_size)
        
        # Smooth edges with a slight blur
        mask = self._smooth_mask_edges(mask)
        
        return mask
    
    def _morphological_operations(self, mask, kernel_size):
        """Apply morphological operations to clean up the mask"""
        # Simple dilation and erosion operations using numpy
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        
        # Dilation (expand white areas)
        dilated = self._dilate(mask, kernel)
        
        # Erosion (shrink white areas back)
        result = self._erode(dilated, kernel)
        
        return result
    
    def _dilate(self, image, kernel):
        """Simple dilation operation"""
        result = image.copy()
        k_h, k_w = kernel.shape
        pad_h, pad_w = k_h // 2, k_w // 2
        
        for i in range(pad_h, image.shape[0] - pad_h):
            for j in range(pad_w, image.shape[1] - pad_w):
                if np.any(image[i-pad_h:i+pad_h+1, j-pad_w:j+pad_w+1] * kernel):
                    result[i, j] = 255
        return result
    
    def _erode(self, image, kernel):
        """Simple erosion operation"""
        result = image.copy()
        k_h, k_w = kernel.shape
        pad_h, pad_w = k_h // 2, k_w // 2
        
        for i in range(pad_h, image.shape[0] - pad_h):
            for j in range(pad_w, image.shape[1] - pad_w):
                if not np.all(image[i-pad_h:i+pad_h+1, j-pad_w:j+pad_w+1] * kernel == kernel * 255):
                    result[i, j] = 0
        return result
    
    def _smooth_mask_edges(self, mask):
        """Apply simple smoothing to mask edges"""
        # Simple 3x3 averaging for edge smoothing
        smoothed = mask.copy().astype(np.float32)
        
        for i in range(1, mask.shape[0] - 1):
            for j in range(1, mask.shape[1] - 1):
                # Average with surrounding pixels
                neighborhood = mask[i-1:i+2, j-1:j+2]
                smoothed[i, j] = np.mean(neighborhood)
        
        return smoothed.astype(np.uint8)
    
    def remove_selected_area(self, input_path, output_path, mask_data):
        """Remove selected area from image using canvas mask data"""
        try:
            with Image.open(input_path) as img:
                # Convert image to RGBA for transparency support
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
                
                # Decode mask data from base64
                mask_image_data = base64.b64decode(mask_data.split(',')[1])
                mask_img = Image.open(BytesIO(mask_image_data))
                
                # Resize mask to match image dimensions
                mask_img = mask_img.resize((img.width, img.height), Image.Resampling.LANCZOS)
                
                # Convert mask to grayscale if needed
                if mask_img.mode != 'L':
                    mask_img = mask_img.convert('L')
                
                # Convert mask to numpy array
                mask_array = np.array(mask_img)
                img_array = np.array(img)
                
                # Apply mask - set alpha channel to 0 where mask is white (255)
                mask_threshold = 128  # Threshold for mask detection
                transparent_pixels = mask_array > mask_threshold
                img_array[transparent_pixels, 3] = 0  # Set alpha to 0 (transparent)
                
                # Convert back to PIL Image
                result_img = Image.fromarray(img_array, 'RGBA')
                
                # Save as PNG to preserve transparency
                result_img.save(output_path.replace(os.path.splitext(output_path)[1], '.png'), 'PNG', optimize=True)
                
                return {'success': True}
                
        except Exception as e:
            logging.error(f"Area removal error: {str(e)}")
            return {'success': False, 'error': str(e)}
