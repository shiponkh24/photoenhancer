import os
import logging
import time
from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
import uuid
from image_processor import ImageProcessor
import json

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev_secret_key_change_in_production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configuration
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'gif'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Initialize image processor
processor = ImageProcessor()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file selected'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file format. Supported formats: PNG, JPG, JPEG, WEBP, GIF'}), 400
        
        # Generate unique filename
        if file.filename is None:
            return jsonify({'error': 'Invalid filename'}), 400
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        file.save(filepath)
        
        # Get image info
        info = processor.get_image_info(filepath)
        
        return jsonify({
            'success': True,
            'filename': unique_filename,
            'info': info
        })
        
    except Exception as e:
        logging.error(f"Upload error: {str(e)}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/process', methods=['POST'])
def process_image():
    try:
        data = request.get_json()
        filename = data.get('filename')
        operation = data.get('operation')
        params = data.get('params', {})
        
        if not filename or not operation:
            return jsonify({'error': 'Missing filename or operation'}), 400
        
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(input_path):
            return jsonify({'error': 'Original file not found'}), 404
        
        # Generate output filename for user download
        base_name = os.path.splitext(filename)[0]
        timestamp = str(int(time.time()))
        
        # Process image in memory and return data directly
        if operation == 'upscale':
            scale_factor = params.get('scale_factor', 2)
            result = processor.upscale_image(input_path, None, scale_factor)
        elif operation == 'enhance':
            brightness = params.get('brightness', 1.0)
            contrast = params.get('contrast', 1.0)
            sharpness = params.get('sharpness', 1.0)
            result = processor.enhance_image(input_path, None, brightness, contrast, sharpness)
        elif operation == 'resize':
            width = params.get('width')
            height = params.get('height')
            if not width or not height:
                return jsonify({'error': 'Width and height required for resize'}), 400
            result = processor.resize_image(input_path, None, int(width), int(height))
        elif operation == 'remove_background':
            result = processor.remove_background(input_path, None)
        elif operation == 'remove_area':
            mask_data = params.get('mask_data')
            if not mask_data:
                return jsonify({'error': 'Mask data required for area removal'}), 400
            result = processor.remove_selected_area(input_path, None, mask_data)
        elif operation == 'humanize':
            intensity = params.get('intensity', 0.7)
            result = processor.humanize_image(input_path, None, intensity)
        else:
            return jsonify({'error': 'Invalid operation'}), 400
        
        if result['success']:
            return jsonify({
                'success': True,
                'image_data': result['image_data'],
                'filename': f"{base_name}_{operation}_{timestamp}.{result.get('format', 'jpg').lower()}",
                'info': result['info']
            })
        else:
            return jsonify({'error': result['error']}), 500
            
    except Exception as e:
        logging.error(f"Processing error: {str(e)}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/download/<filename>')
def download_file(filename):
    try:
        filepath = os.path.join(app.config['PROCESSED_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(filepath, as_attachment=True, download_name=filename)
        
    except Exception as e:
        logging.error(f"Download error: {str(e)}")
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

@app.route('/preview/<path:filename>')
def preview_file(filename):
    try:
        # Check both upload and processed folders
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        processed_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
        
        if os.path.exists(upload_path):
            return send_file(upload_path)
        elif os.path.exists(processed_path):
            return send_file(processed_path)
        else:
            return jsonify({'error': 'File not found'}), 404
            
    except Exception as e:
        logging.error(f"Preview error: {str(e)}")
        return jsonify({'error': f'Preview failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
