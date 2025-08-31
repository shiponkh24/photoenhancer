import os
import logging
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from rembg import remove
import json
import base64
from io import BytesIO

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
        """Remove background from image using rembg"""
        try:
            with open(input_path, 'rb') as input_file:
                input_data = input_file.read()
            
            # Remove background
            output_data = remove(input_data)
            
            # Save as PNG to preserve transparency
            with open(output_path.replace(os.path.splitext(output_path)[1], '.png'), 'wb') as output_file:
                output_file.write(output_data)
            
            return {'success': True}
            
        except Exception as e:
            logging.error(f"Background removal error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
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
