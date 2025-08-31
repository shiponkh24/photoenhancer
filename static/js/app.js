// Global variables
let currentFilename = null;
let currentImageInfo = null;
let processedFilename = null;
let isDrawing = false;
let drawMode = true; // true for draw, false for erase
let canvas = null;
let ctx = null;
let originalImageData = null;
let currentZoom = 1.0;
let processedImageElement = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    initializeSliders();
});

function initializeEventListeners() {
    // File upload
    const fileInput = document.getElementById('file-input');
    const uploadArea = document.getElementById('upload-area');
    
    fileInput.addEventListener('change', handleFileSelect);
    // Removed uploadArea click handler to prevent double prompt
    
    // Drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleFileDrop);
    
    // Processing buttons
    document.getElementById('upscale-btn').addEventListener('click', () => processImage('upscale'));
    document.getElementById('enhance-btn').addEventListener('click', () => processImage('enhance'));
    document.getElementById('resize-btn').addEventListener('click', () => processImage('resize'));
    document.getElementById('remove-bg-btn').addEventListener('click', () => processImage('remove_background'));
    document.getElementById('apply-selection-btn').addEventListener('click', () => processImage('remove_area'));
    document.getElementById('humanize-btn').addEventListener('click', () => processImage('humanize'));
    
    // Reset button
    document.getElementById('reset-enhance-btn').addEventListener('click', resetEnhanceSliders);
    
    // Canvas controls
    document.getElementById('draw-mode-btn').addEventListener('click', () => setCanvasMode(true));
    document.getElementById('erase-mode-btn').addEventListener('click', () => setCanvasMode(false));
    document.getElementById('clear-selection-btn').addEventListener('click', clearCanvas);
    
    // Download button
    document.getElementById('download-btn').addEventListener('click', downloadProcessedImage);
    // Mockup buttons
    const createMockupBtn = document.getElementById('create-mockup-btn');
    if (createMockupBtn) createMockupBtn.addEventListener('click', createMockup);
    const downloadMockupBtn = document.getElementById('download-mockup-btn');
    if (downloadMockupBtn) downloadMockupBtn.addEventListener('click', downloadMockup);
    document.querySelectorAll('.mockup-color').forEach(btn => btn.addEventListener('click', selectMockupColor));
    
    // Aspect ratio maintenance
    document.getElementById('new-width').addEventListener('input', maintainAspectRatio);
    document.getElementById('new-height').addEventListener('input', maintainAspectRatio);
    
    // Zoom controls
    document.getElementById('zoom-in-btn').addEventListener('click', zoomIn);
    document.getElementById('zoom-out-btn').addEventListener('click', zoomOut);
    document.getElementById('zoom-reset-btn').addEventListener('click', resetZoom);
}

function initializeSliders() {
    // Enhancement sliders
    const brightnessSlider = document.getElementById('brightness-slider');
    const contrastSlider = document.getElementById('contrast-slider');
    const sharpnessSlider = document.getElementById('sharpness-slider');
    const brushSizeSlider = document.getElementById('brush-size');
    
    brightnessSlider.addEventListener('input', (e) => {
        document.getElementById('brightness-value').textContent = e.target.value;
    });
    
    contrastSlider.addEventListener('input', (e) => {
        document.getElementById('contrast-value').textContent = e.target.value;
    });
    
    sharpnessSlider.addEventListener('input', (e) => {
        document.getElementById('sharpness-value').textContent = e.target.value;
    });
    
    brushSizeSlider.addEventListener('input', (e) => {
        document.getElementById('brush-size-value').textContent = e.target.value + 'px';
    });
    
    // Humanize intensity slider
    const humanizeSlider = document.getElementById('humanize-intensity');
    if (humanizeSlider) {
        humanizeSlider.addEventListener('input', (e) => {
            document.getElementById('humanize-intensity-value').textContent = Math.round(e.target.value * 100) + '%';
        });
    }
}

// File handling functions
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        uploadFile(file);
        // Reset the input so selecting the same file again will trigger change
        event.target.value = '';
    }
}

function handleDragOver(event) {
    event.preventDefault();
    event.currentTarget.classList.add('dragover');
}

function handleDragLeave(event) {
    event.currentTarget.classList.remove('dragover');
}

function handleFileDrop(event) {
    event.preventDefault();
    event.currentTarget.classList.remove('dragover');
    
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        uploadFile(files[0]);
    }
}

function uploadFile(file) {
    // Validate file type
    const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp', 'image/gif'];
    if (!allowedTypes.includes(file.type)) {
        showAlert('Invalid file type. Please upload PNG, JPG, JPEG, WEBP, or GIF files.', 'danger');
        return;
    }
    
    // Validate file size (50MB)
    if (file.size > 50 * 1024 * 1024) {
        showAlert('File size too large. Please upload files smaller than 50MB.', 'danger');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    // Show upload progress
    showUploadProgress();
    
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        hideUploadProgress();
        
        if (data.success) {
            currentFilename = data.filename;
            currentImageInfo = data.info;
            displayOriginalImage();
            showImageWorkspace();
        } else {
            showAlert(data.error || 'Upload failed', 'danger');
        }
    })
    .catch(error => {
        hideUploadProgress();
        console.error('Upload error:', error);
        showAlert('Upload failed. Please try again.', 'danger');
    });
}

function displayOriginalImage() {
    const img = document.getElementById('original-image');
    const info = document.getElementById('original-info');
    
    img.src = `/preview/${currentFilename}`;
    img.onload = function() {
        // Update image info
        info.innerHTML = `
            <small class="text-muted">
                ${currentImageInfo.width} × ${currentImageInfo.height} pixels | 
                ${currentImageInfo.format} | 
                ${currentImageInfo.size_mb} MB
            </small>
        `;
        
        // Initialize canvas for selection tool
        initializeSelectionCanvas();
        
        // Set default resize values
        document.getElementById('new-width').value = currentImageInfo.width;
        document.getElementById('new-height').value = currentImageInfo.height;
    };
}

function initializeSelectionCanvas() {
    canvas = document.getElementById('selection-canvas');
    ctx = canvas.getContext('2d');
    
    const img = document.getElementById('original-image');
    
    // Set canvas size to match displayed image
    const rect = img.getBoundingClientRect();
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    canvas.style.width = Math.min(500, rect.width) + 'px';
    canvas.style.height = (canvas.height * Math.min(500, rect.width) / canvas.width) + 'px';
    
    // Store original image data
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    tempCanvas.width = canvas.width;
    tempCanvas.height = canvas.height;
    tempCtx.drawImage(img, 0, 0);
    originalImageData = tempCtx.getImageData(0, 0, canvas.width, canvas.height);
    
    // Draw original image on canvas
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    
    // Add drawing event listeners
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);
    
    // Touch events for mobile
    canvas.addEventListener('touchstart', handleTouch);
    canvas.addEventListener('touchmove', handleTouch);
    canvas.addEventListener('touchend', stopDrawing);
}

// Canvas drawing functions
function startDrawing(event) {
    isDrawing = true;
    draw(event);
}

function draw(event) {
    if (!isDrawing) return;
    
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    const x = (event.clientX - rect.left) * scaleX;
    const y = (event.clientY - rect.top) * scaleY;
    
    const brushSize = parseInt(document.getElementById('brush-size').value);
    
    ctx.globalCompositeOperation = drawMode ? 'source-over' : 'destination-out';
    ctx.beginPath();
    ctx.arc(x, y, brushSize / 2, 0, 2 * Math.PI);
    ctx.fillStyle = drawMode ? 'rgba(255, 0, 0, 0.5)' : 'transparent';
    ctx.fill();
}

function stopDrawing() {
    isDrawing = false;
}

function handleTouch(event) {
    event.preventDefault();
    const touch = event.touches[0];
    const mouseEvent = new MouseEvent(event.type === 'touchstart' ? 'mousedown' : 
                                     event.type === 'touchmove' ? 'mousemove' : 'mouseup', {
        clientX: touch.clientX,
        clientY: touch.clientY
    });
    canvas.dispatchEvent(mouseEvent);
}

function setCanvasMode(isDraw) {
    drawMode = isDraw;
    document.getElementById('draw-mode-btn').classList.toggle('active', isDraw);
    document.getElementById('erase-mode-btn').classList.toggle('active', !isDraw);
    canvas.className = isDraw ? '' : 'erase-mode';
}

function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.putImageData(originalImageData, 0, 0);
}

// Image processing functions
function processImage(operation) {
    if (!currentFilename) {
        showAlert('Please upload an image first', 'warning');
        return;
    }
    
    let params = {};
    
    // Gather parameters based on operation
    switch (operation) {
        case 'upscale':
            params.scale_factor = parseInt(document.getElementById('scale-factor').value);
            break;
        case 'enhance':
            params.brightness = parseFloat(document.getElementById('brightness-slider').value);
            params.contrast = parseFloat(document.getElementById('contrast-slider').value);
            params.sharpness = parseFloat(document.getElementById('sharpness-slider').value);
            break;
        case 'resize':
            params.width = parseInt(document.getElementById('new-width').value);
            params.height = parseInt(document.getElementById('new-height').value);
            
            if (!params.width || !params.height || params.width < 1 || params.height < 1) {
                showAlert('Please enter valid width and height values', 'warning');
                return;
            }
            break;
        case 'remove_area':
            if (!canvas) {
                showAlert('Please draw on the image to select areas to remove', 'warning');
                return;
            }
            params.mask_data = canvas.toDataURL();
            break;
        case 'humanize':
            params.intensity = parseFloat(document.getElementById('humanize-intensity').value);
            break;
    }
    
    showProcessingStatus();
    
    fetch('/process', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            filename: currentFilename,
            operation: operation,
            params: params
        })
    })
    .then(response => response.json())
    .then(data => {
        hideProcessingStatus();
        
        if (data.success) {
            // Display processed image in the UI and enable manual download
            renderProcessedImage(data.image_data, data.filename);
            showProcessedImageInfo(data.info);
            showAlert(`Image processed: ${data.filename}. Click Download Result to save.`, 'success');
        } else {
            showAlert(data.error || 'Processing failed', 'danger');
        }
    })
    .catch(error => {
        hideProcessingStatus();
        console.error('Processing error:', error);
        showAlert('Processing failed. Please try again.', 'danger');
    });
}

// Legacy function - no longer used since images auto-download
function displayProcessedImage(info) {
    showProcessedImageInfo(info);
}

function downloadImage(imageData, filename) {
    // Create download link element
    const link = document.createElement('a');
    link.href = imageData;
    link.download = filename;
    
    // Trigger download
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// New: render processed image into the processed preview area and enable download button
function renderProcessedImage(imageData, filename) {
    const processedPreview = document.getElementById('processed-preview');
    const downloadSection = document.getElementById('download-section');
    const zoomControls = document.getElementById('zoom-controls');

    // Remove any previous image element
    if (processedImageElement) {
        processedImageElement.remove();
        processedImageElement = null;
    }

    // Create image element
    const img = document.createElement('img');
    img.src = imageData;
    img.alt = filename;
    img.className = 'img-fluid rounded';
    img.style.maxHeight = '500px';
    img.style.cursor = 'default';
    img.style.transition = 'transform 0.1s ease';

    processedPreview.appendChild(img);
    processedImageElement = img;

    // Show zoom controls
    if (zoomControls) {
        zoomControls.classList.remove('d-none');
    }

    // Show download section and attach filename to button dataset
    if (downloadSection) {
        downloadSection.classList.remove('d-none');
        const btn = document.getElementById('download-btn');
        btn.dataset.filename = filename;
    }

    // Make image draggable when zoomed
    makeImageDraggable(img);

    // Enable mockup controls
    const mockupControls = document.getElementById('mockup-controls');
    if (mockupControls) mockupControls.classList.remove('d-none');
}

function showProcessedImageInfo(info) {
    const infoDiv = document.getElementById('processed-info');
    const downloadSection = document.getElementById('download-section');
    const placeholder = document.getElementById('processed-placeholder');
    
    // Hide placeholder
    if (placeholder) {
        placeholder.style.display = 'none';
    }
    
    // Show processed info
    infoDiv.innerHTML = `
        <small class="text-success">
            <i class="fas fa-check-circle"></i> Image processed successfully!<br>
            ${info.width} × ${info.height} pixels | 
            ${info.format} | 
            ${info.size_mb} MB<br>
            <strong>Click "Download Result" to save the processed image</strong>
        </small>
    `;
    
    // Keep zoom controls and download section visible when a processed image is present
    const zoomControls = document.getElementById('zoom-controls');
    if (zoomControls && processedImageElement) {
        zoomControls.classList.remove('d-none');
    }
    if (downloadSection && processedImageElement) {
        downloadSection.classList.remove('d-none');
    }
}

// Mockup functionality
let mockupColor = '#ffffff';
let mockupCanvas = null;
let mockupCtx = null;

function createMockup() {
    if (!processedImageElement) {
        showAlert('No processed image to create mockup from', 'warning');
        return;
    }

    mockupCanvas = document.getElementById('mockup-canvas');
    mockupCtx = mockupCanvas.getContext('2d');

    // Load tshirt mock image if available
    const tshirtImg = new Image();
    tshirtImg.onload = () => renderMockup(tshirtImg);
    tshirtImg.onerror = () => renderMockup(null);
    tshirtImg.src = '/static/images/tshirt_mock.png';

    document.getElementById('download-mockup-btn').classList.remove('d-none');
}

function renderMockup(tshirtImg) {
    const canvas = mockupCanvas;
    const ctx = mockupCtx;
    const width = 800;
    const height = 900;
    canvas.width = width;
    canvas.height = height;

    // Draw colored t-shirt background
    ctx.fillStyle = mockupColor;
    ctx.fillRect(0, 0, width, height);

    // If tshirt image provided, draw it on top to show shape
    if (tshirtImg) {
        // Fit the tshirt mock into canvas
        const tshirtW = width * 0.95;
        const tshirtH = height * 0.95;
        const x = (width - tshirtW) / 2;
        const y = (height - tshirtH) / 2;
        ctx.drawImage(tshirtImg, x, y, tshirtW, tshirtH);
    }

    // Draw the processed image centered on the chest area
    const img = processedImageElement;
    const targetW = Math.floor(width * 0.5);
    const aspect = img.naturalWidth / img.naturalHeight;
    const targetH = Math.floor(targetW / aspect);
    const imgX = Math.floor((width - targetW) / 2);
    const imgY = Math.floor(height * 0.28);

    // Create an offscreen image to ensure CORS/dataURL works
    const drawImg = new Image();
    drawImg.onload = () => {
        // Optionally add a subtle shadow
        ctx.save();
        ctx.globalAlpha = 0.95;
        ctx.drawImage(drawImg, imgX, imgY, targetW, targetH);
        ctx.restore();
    };
    drawImg.src = img.src;
}

function selectMockupColor(e) {
    mockupColor = e.currentTarget.dataset.color || '#ffffff';
    // re-render mockup if canvas exists
    const tshirtImg = new Image();
    tshirtImg.onload = () => renderMockup(tshirtImg);
    tshirtImg.onerror = () => renderMockup(null);
    tshirtImg.src = '/static/images/tshirt_mock.png';
}

function downloadMockup() {
    if (!mockupCanvas) return showAlert('No mockup available', 'warning');
    const dataUrl = mockupCanvas.toDataURL('image/png');
    downloadImage(dataUrl, 'tshirt_mockup.png');
}

// Download handler attached to Download button
function downloadProcessedImage() {
    const btn = document.getElementById('download-btn');
    const filename = btn.dataset.filename;

    // Prefer server-provided download endpoint if available. Use fetch to get blob and trigger download.
    if (filename) {
        const downloadUrl = `/download/${encodeURIComponent(filename)}`;
        fetch(downloadUrl)
            .then(resp => {
                if (!resp.ok) throw new Error('Download failed');
                return resp.blob();
            })
            .then(blob => {
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                a.remove();
                URL.revokeObjectURL(url);
                showAlert(`Download started for ${filename}`, 'info');
            })
            .catch(err => {
                console.error('Download error:', err);
                // Fallback to data URL if available
                if (processedImageElement && processedImageElement.src) {
                    downloadImage(processedImageElement.src, filename || 'processed_image.png');
                } else {
                    showAlert('No processed image available to download', 'warning');
                }
            });
    } else if (processedImageElement && processedImageElement.src) {
        // Fallback: download blob data from data URL
        downloadImage(processedImageElement.src, filename || 'processed_image.png');
    } else {
        showAlert('No processed image available to download', 'warning');
    }
}

// Utility functions
function maintainAspectRatio(event) {
    if (!document.getElementById('maintain-aspect').checked || !currentImageInfo) return;
    
    const aspectRatio = currentImageInfo.width / currentImageInfo.height;
    const widthInput = document.getElementById('new-width');
    const heightInput = document.getElementById('new-height');
    
    if (event.target.id === 'new-width') {
        heightInput.value = Math.round(widthInput.value / aspectRatio);
    } else {
        widthInput.value = Math.round(heightInput.value * aspectRatio);
    }
}

function resetEnhanceSliders() {
    document.getElementById('brightness-slider').value = 1;
    document.getElementById('contrast-slider').value = 1;
    document.getElementById('sharpness-slider').value = 1;
    document.getElementById('brightness-value').textContent = '1.0';
    document.getElementById('contrast-value').textContent = '1.0';
    document.getElementById('sharpness-value').textContent = '1.0';
}

function showImageWorkspace() {
    document.getElementById('image-workspace').classList.remove('d-none');
}

function showUploadProgress() {
    document.getElementById('upload-progress').classList.remove('d-none');
    document.querySelector('.progress-bar').style.width = '100%';
}

function hideUploadProgress() {
    document.getElementById('upload-progress').classList.add('d-none');
    document.querySelector('.progress-bar').style.width = '0%';
}

function showProcessingStatus() {
    document.getElementById('processing-status').classList.remove('d-none');
}

function hideProcessingStatus() {
    document.getElementById('processing-status').classList.add('d-none');
}

// Zoom functionality
function zoomIn() {
    if (currentZoom < 3.0) {
        currentZoom += 0.25;
        updateZoomDisplay();
        applyZoom();
    }
}

function zoomOut() {
    if (currentZoom > 0.25) {
        currentZoom -= 0.25;
        updateZoomDisplay();
        applyZoom();
    }
}

function resetZoom() {
    currentZoom = 1.0;
    updateZoomDisplay();
    applyZoom();
    // Reset position
    if (processedImageElement) {
        processedImageElement.style.transform = 'scale(1) translate(0, 0)';
    }
}

function updateZoomDisplay() {
    const zoomLevel = document.getElementById('zoom-level');
    if (zoomLevel) {
        zoomLevel.textContent = Math.round(currentZoom * 100) + '%';
    }
}

function applyZoom() {
    if (processedImageElement) {
        // Get current translation values
        const transform = processedImageElement.style.transform;
        const translateMatch = transform.match(/translate\(([^,]+),\s*([^)]+)\)/);
        const currentTranslateX = translateMatch ? translateMatch[1] : '0px';
        const currentTranslateY = translateMatch ? translateMatch[2] : '0px';
        
        processedImageElement.style.transform = `scale(${currentZoom}) translate(${currentTranslateX}, ${currentTranslateY})`;
    }
}

// Make image draggable when zoomed
function makeImageDraggable(imageElement) {
    let isDragging = false;
    let startX, startY, initialTranslateX = 0, initialTranslateY = 0;
    
    imageElement.addEventListener('mousedown', function(e) {
        if (currentZoom > 1) {
            isDragging = true;
            startX = e.clientX;
            startY = e.clientY;
            
            // Get current translate values
            const transform = imageElement.style.transform;
            const translateMatch = transform.match(/translate\(([^,]+),\s*([^)]+)\)/);
            if (translateMatch) {
                initialTranslateX = parseFloat(translateMatch[1]);
                initialTranslateY = parseFloat(translateMatch[2]);
            } else {
                initialTranslateX = 0;
                initialTranslateY = 0;
            }
            
            imageElement.style.cursor = 'grabbing';
            e.preventDefault();
        }
    });
    
    document.addEventListener('mousemove', function(e) {
        if (isDragging && currentZoom > 1) {
            const deltaX = (e.clientX - startX) / currentZoom;
            const deltaY = (e.clientY - startY) / currentZoom;
            const newTranslateX = initialTranslateX + deltaX;
            const newTranslateY = initialTranslateY + deltaY;
            
            imageElement.style.transform = `scale(${currentZoom}) translate(${newTranslateX}px, ${newTranslateY}px)`;
        }
    });
    
    document.addEventListener('mouseup', function() {
        if (isDragging) {
            isDragging = false;
            if (processedImageElement) {
                processedImageElement.style.cursor = currentZoom > 1 ? 'move' : 'default';
            }
        }
    });
    
    // Handle mouse wheel zoom
    imageElement.addEventListener('wheel', function(e) {
        e.preventDefault();
        if (e.deltaY < 0 && currentZoom < 3.0) {
            zoomIn();
        } else if (e.deltaY > 0 && currentZoom > 0.25) {
            zoomOut();
        }
    });
}

function showAlert(message, type = 'info') {
    // Create alert element
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    alertDiv.style.cssText = 'top: 20px; right: 20px; z-index: 1055; min-width: 300px;';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(alertDiv);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 5000);
}
