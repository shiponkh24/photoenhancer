# Overview

This is an AI-powered image enhancement web application built with Flask. The application allows users to upload images and apply various enhancement operations including upscaling, background removal, resizing, and area-specific editing through an interactive canvas interface. The system provides a user-friendly web interface with drag-and-drop functionality and real-time preview capabilities.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Framework**: Vanilla JavaScript with Bootstrap for UI components
- **Styling**: Custom CSS with Bootstrap dark theme and Font Awesome icons
- **Interactive Features**: Drag-and-drop file upload, canvas-based selection tool for area editing
- **Real-time Feedback**: Progress bars for upload/processing operations and live image previews

## Backend Architecture
- **Web Framework**: Flask (Python) serving as the main application server
- **File Handling**: Secure file upload with validation and temporary storage system
- **Image Processing**: Dedicated `ImageProcessor` class handling all image operations
- **Session Management**: Flask sessions with configurable secret key
- **Middleware**: ProxyFix for handling reverse proxy headers

## Image Processing Pipeline
- **Core Library**: PIL (Pillow) for basic image operations
- **Background Removal**: rembg library for AI-powered background removal
- **Enhancement Algorithms**: Custom implementations using PIL for upscaling, sharpening, and filtering
- **Format Support**: PNG, JPG, JPEG, WEBP, and GIF with automatic format detection

## File Management System
- **Upload Directory**: Temporary storage for incoming files with unique naming
- **Processing Directory**: Separate folder for processed outputs
- **File Validation**: Extension and size limits (50MB maximum)
- **Cleanup Strategy**: Automatic file organization with secure filename handling

# External Dependencies

## Python Libraries
- **Flask**: Web application framework and routing
- **PIL/Pillow**: Core image processing and manipulation
- **rembg**: AI-powered background removal service
- **NumPy**: Numerical operations for image data processing
- **Werkzeug**: HTTP utilities and secure filename handling

## Frontend Libraries
- **Bootstrap**: UI framework with dark theme support
- **Font Awesome**: Icon library for interface elements
- **HTML5 Canvas API**: Interactive drawing and selection tools

## System Requirements
- **Python Runtime**: Flask application requiring Python environment
- **File System**: Read/write access for upload and processing directories
- **Memory**: Sufficient RAM for processing large images (up to 50MB)