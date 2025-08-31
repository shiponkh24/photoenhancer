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
                
                # Return image data directly if no output path
                if output_path is None:
                    return self._image_to_response_data(upscaled, img.format)
                
                # Save with appropriate format (legacy support)
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
                orig_format = img.format
                
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
                
                # Return image data directly if no output path
                if output_path is None:
                    return self._image_to_response_data(img, orig_format)
                
                # Save with appropriate format (legacy support)
                if orig_format == 'PNG' or img.mode in ('RGBA', 'LA'):
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
                orig_format = img.format
                
                # Resize using high-quality resampling
                resized = img.resize((width, height), Image.Resampling.LANCZOS)
                
                # Return image data directly if no output path
                if output_path is None:
                    return self._image_to_response_data(resized, orig_format)
                
                # Save with appropriate format (legacy support)
                if orig_format == 'PNG' or img.mode in ('RGBA', 'LA'):
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
                
                # Return image data directly if no output path
                if output_path is None:
                    return self._image_to_response_data(result_img, 'PNG')
                
                # Save as PNG to preserve transparency (legacy support)
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
    
    def _image_to_response_data(self, image, original_format):
        """Convert PIL image to base64 data for client download"""
        buffer = BytesIO()
        
        # Determine output format
        if original_format == 'PNG' or image.mode in ('RGBA', 'LA'):
            format_str = 'PNG'
            image.save(buffer, format=format_str, optimize=True)
        else:
            format_str = 'JPEG'
            # Convert RGBA to RGB if saving as JPEG
            if image.mode == 'RGBA':
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[-1])
                rgb_image.save(buffer, format=format_str, quality=95, optimize=True)
            else:
                image.save(buffer, format=format_str, quality=95, optimize=True)
        
        buffer.seek(0)
        image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return {
            'success': True,
            'image_data': f"data:image/{format_str.lower()};base64,{image_data}",
            'format': format_str,
            'info': {
                'width': image.width,
                'height': image.height,
                'format': format_str,
                'mode': image.mode,
                'size_mb': round(len(buffer.getvalue()) / (1024 * 1024), 2)
            }
        }
    
    def humanize_image(self, input_path, output_path, intensity=0.7):
        """Transform AI-generated image to look more natural and human-made"""
        try:
            with Image.open(input_path) as img:
                orig_format = img.format
                
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Convert to numpy array for processing
                img_array = np.array(img, dtype=np.float32)
                height, width = img_array.shape[:2]
                
                # Apply multiple humanization techniques
                img_array = self._add_natural_grain(img_array, intensity)
                img_array = self._add_color_variations(img_array, intensity)
                img_array = self._add_lighting_imperfections(img_array, intensity)
                img_array = self._add_subtle_blur_variations(img_array, intensity)
                img_array = self._add_organic_textures(img_array, intensity)
                
                # Convert back to PIL Image
                img_array = np.clip(img_array, 0, 255).astype(np.uint8)
                result_img = Image.fromarray(img_array)
                
                # Apply post-processing effects
                result_img = self._apply_natural_sharpening(result_img, intensity)
                result_img = self._add_subtle_vignette(result_img, intensity)
                
                # Return image data directly if no output path
                if output_path is None:
                    return self._image_to_response_data(result_img, orig_format)
                
                # Save with appropriate format (legacy support)
                if orig_format == 'PNG' or img.mode in ('RGBA', 'LA'):
                    result_img.save(output_path, 'PNG', optimize=True)
                else:
                    result_img.save(output_path, 'JPEG', quality=92, optimize=True)
                
                return {'success': True}
                
        except Exception as e:
            logging.error(f"Humanization error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _add_natural_grain(self, img_array, intensity):
        """Add realistic film grain and noise"""
        height, width = img_array.shape[:2]
        
        # Generate multiple layers of noise for realistic grain
        fine_noise = np.random.normal(0, 2.5 * intensity, (height, width, 3))
        medium_noise = np.random.normal(0, 1.5 * intensity, (height//2, width//2, 3))
        medium_noise = np.repeat(np.repeat(medium_noise, 2, axis=0), 2, axis=1)[:height, :width]
        
        # Combine noise layers
        total_noise = fine_noise + 0.7 * medium_noise
        
        # Apply grain with luminance-based intensity
        gray = np.dot(img_array, [0.299, 0.587, 0.114])
        grain_mask = 1.0 - (gray / 255.0) * 0.3  # Less grain in bright areas
        grain_mask = np.stack([grain_mask] * 3, axis=2)
        
        return img_array + total_noise * grain_mask
    
    def _add_color_variations(self, img_array, intensity):
        """Add subtle color temperature variations and imperfections"""
        height, width = img_array.shape[:2]
        
        # Create smooth color temperature variations
        x = np.linspace(0, 4, width)
        y = np.linspace(0, 3, height)
        X, Y = np.meshgrid(x, y)
        
        # Generate smooth temperature map
        temp_map = np.sin(X * 0.8) * np.cos(Y * 0.6) + np.sin(X * 1.2 + 1) * np.cos(Y * 0.9 + 0.5)
        temp_map = temp_map * intensity * 8
        
        # Apply warm/cool variations
        warm_factor = 1 + temp_map * 0.03
        cool_factor = 1 - temp_map * 0.02
        
        result = img_array.copy()
        result[:, :, 0] *= np.clip(warm_factor, 0.95, 1.08)  # Red channel
        result[:, :, 2] *= np.clip(cool_factor, 0.92, 1.05)   # Blue channel
        
        # Add slight saturation variations
        saturation_map = np.sin(X * 1.5 + 2) * np.cos(Y * 1.1 + 1.5) * intensity * 0.1
        
        # Convert to HSV for saturation adjustment
        hsv_img = self._rgb_to_hsv(result)
        hsv_img[:, :, 1] *= (1 + saturation_map)
        hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1], 0, 1)
        
        return self._hsv_to_rgb(hsv_img)
    
    def _add_lighting_imperfections(self, img_array, intensity):
        """Add realistic lighting variations and soft shadows"""
        height, width = img_array.shape[:2]
        
        # Create natural lighting gradients
        center_x, center_y = width // 2, height // 2
        y, x = np.ogrid[:height, :width]
        
        # Radial gradient for natural vignetting
        distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        vignette = 1 - (distance_from_center / max_distance) * intensity * 0.15
        
        # Add directional lighting bias
        directional = np.sin(x / width * np.pi * 2) * np.cos(y / height * np.pi * 1.5) * intensity * 0.08
        lighting_map = vignette + directional
        
        # Apply lighting variations
        lighting_map = np.clip(lighting_map, 0.85, 1.15)
        lighting_map = np.stack([lighting_map] * 3, axis=2)
        
        return img_array * lighting_map
    
    def _add_subtle_blur_variations(self, img_array, intensity):
        """Add natural depth of field and focus variations"""
        height, width = img_array.shape[:2]
        
        # Create focus map with subtle variations
        center_x, center_y = width // 2 + np.random.randint(-width//6, width//6), height // 2 + np.random.randint(-height//6, height//6)
        y, x = np.ogrid[:height, :width]
        
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(width**2 + height**2) / 3
        
        # Natural focus falloff
        focus_map = 1 - np.clip(distance / max_dist, 0, 1) * intensity * 0.3
        
        # Apply subtle gaussian blur to out-of-focus areas
        from PIL import ImageFilter
        
        result = Image.fromarray(img_array.astype(np.uint8))
        blurred = result.filter(ImageFilter.GaussianBlur(radius=1.5 * intensity))
        
        result_array = np.array(result, dtype=np.float32)
        blurred_array = np.array(blurred, dtype=np.float32)
        
        # Blend based on focus map
        focus_map = np.stack([focus_map] * 3, axis=2)
        return result_array * focus_map + blurred_array * (1 - focus_map)
    
    def _add_organic_textures(self, img_array, intensity):
        """Add subtle organic texture variations"""
        height, width = img_array.shape[:2]
        
        # Generate organic texture pattern
        scale = min(height, width) // 100
        if scale < 1:
            scale = 1
            
        # Create multiple octaves of noise for organic feel
        texture = np.zeros((height, width))
        
        for octave in range(3):
            freq = 0.01 * (2 ** octave)
            amplitude = 0.5 ** octave
            
            # Generate Perlin-like noise using sine waves
            x = np.linspace(0, freq * width, width)
            y = np.linspace(0, freq * height, height)
            X, Y = np.meshgrid(x, y)
            
            octave_noise = np.sin(X * 6.28) * np.cos(Y * 6.28 + 1.57) + \
                          np.sin(X * 6.28 + 2.09) * np.cos(Y * 6.28 + 3.14) + \
                          np.sin(X * 6.28 * 1.7 + 4.18) * np.cos(Y * 6.28 * 1.3 + 5.23)
            
            texture += octave_noise * amplitude
        
        # Normalize texture
        texture = (texture - texture.min()) / (texture.max() - texture.min())
        texture = (texture - 0.5) * intensity * 6
        
        # Apply texture as luminance variation
        texture_3d = np.stack([texture] * 3, axis=2)
        return img_array + texture_3d
    
    def _apply_natural_sharpening(self, img, intensity):
        """Apply natural, non-uniform sharpening"""
        # Create unsharp mask with variations
        from PIL import ImageFilter
        
        blurred = img.filter(ImageFilter.GaussianBlur(radius=1.2))
        
        # Convert to arrays for blending
        orig_array = np.array(img, dtype=np.float32)
        blur_array = np.array(blurred, dtype=np.float32)
        
        # Create non-uniform sharpening mask
        height, width = orig_array.shape[:2]
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        X, Y = np.meshgrid(x, y)
        
        sharp_mask = 0.8 + 0.4 * np.sin(X * 6.28 * 2.3) * np.cos(Y * 6.28 * 1.7)
        sharp_mask = np.stack([sharp_mask] * 3, axis=2) * intensity
        
        # Apply unsharp mask
        sharpened = orig_array + (orig_array - blur_array) * sharp_mask * 0.3
        sharpened = np.clip(sharpened, 0, 255)
        
        return Image.fromarray(sharpened.astype(np.uint8))
    
    def _add_subtle_vignette(self, img, intensity):
        """Add natural vignette effect"""
        img_array = np.array(img, dtype=np.float32)
        height, width = img_array.shape[:2]
        
        # Create natural vignette
        center_x, center_y = width // 2, height // 2
        y, x = np.ogrid[:height, :width]
        
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        # Natural falloff
        vignette = 1 - (distance / max_distance) ** 2 * intensity * 0.25
        vignette = np.clip(vignette, 0.75, 1.0)
        vignette = np.stack([vignette] * 3, axis=2)
        
        result = img_array * vignette
        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
    
    def _rgb_to_hsv(self, rgb):
        """Convert RGB to HSV"""
        rgb = rgb / 255.0
        max_val = np.max(rgb, axis=2)
        min_val = np.min(rgb, axis=2)
        diff = max_val - min_val
        
        # Hue calculation
        h = np.zeros_like(max_val)
        mask = diff != 0
        
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        
        h[mask & (max_val == r)] = (60 * ((g - b) / diff) % 360)[mask & (max_val == r)]
        h[mask & (max_val == g)] = (60 * ((b - r) / diff) + 120)[mask & (max_val == g)]
        h[mask & (max_val == b)] = (60 * ((r - g) / diff) + 240)[mask & (max_val == b)]
        
        # Saturation
        s = np.zeros_like(max_val)
        s[max_val != 0] = diff[max_val != 0] / max_val[max_val != 0]
        
        # Value
        v = max_val
        
        return np.stack([h/360, s, v], axis=2)
    
    def _hsv_to_rgb(self, hsv):
        """Convert HSV to RGB"""
        h, s, v = hsv[:, :, 0] * 360, hsv[:, :, 1], hsv[:, :, 2]
        
        c = v * s
        x = c * (1 - np.abs((h / 60) % 2 - 1))
        m = v - c
        
        rgb_prime = np.zeros_like(hsv)
        
        # Define conditions for each hue sector
        conditions = [
            (h >= 0) & (h < 60),
            (h >= 60) & (h < 120),
            (h >= 120) & (h < 180),
            (h >= 180) & (h < 240),
            (h >= 240) & (h < 300),
            (h >= 300) & (h < 360)
        ]
        
        rgb_values = [
            [c, x, 0],
            [x, c, 0],
            [0, c, x],
            [0, x, c],
            [x, 0, c],
            [c, 0, x]
        ]
        
        for i, condition in enumerate(conditions):
            for j in range(3):
                rgb_prime[:, :, j] = np.where(condition, rgb_values[i][j], rgb_prime[:, :, j])
        
        rgb = (rgb_prime + np.stack([m, m, m], axis=2)) * 255
        return rgb
    
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
                
                # Return image data directly if no output path
                if output_path is None:
                    return self._image_to_response_data(result_img, 'PNG')
                
                # Save as PNG to preserve transparency (legacy support)
                result_img.save(output_path.replace(os.path.splitext(output_path)[1], '.png'), 'PNG', optimize=True)
                
                return {'success': True}
                
        except Exception as e:
            logging.error(f"Area removal error: {str(e)}")
            return {'success': False, 'error': str(e)}
