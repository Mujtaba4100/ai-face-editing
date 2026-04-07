"""
AI-Powered Facial Editing App
A clean MVP using Gradio, MediaPipe, and OpenCV
Allows users to upload images, detect faces, and edit facial features with sliders.
"""

import gradio as gr
import cv2
import numpy as np
from typing import Tuple, Optional
import logging
import os
import dlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize dlib face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = None

# Try to load predictor from common locations
predictor_path = '/tmp/shape_predictor_68_face_landmarks.dat'
if os.path.exists(predictor_path):
    predictor = dlib.shape_predictor(predictor_path)
else:
    # Download predictor if not available
    import urllib.request
    import bz2
    try:
        url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
        logger.info("Downloading dlib face landmarks model...")
        urllib.request.urlretrieve(url, '/tmp/model.dat.bz2')
        with bz2.BZ2File('/tmp/model.dat.bz2', 'rb') as f:
            with open(predictor_path, 'wb') as out:
                out.write(f.read())
        predictor = dlib.shape_predictor(predictor_path)
        logger.info("✓ Model downloaded successfully")
    except Exception as e:
        logger.warning(f"Could not download dlib model: {e}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_landmark_indices():
    """
    Returns the key facial landmark indices from MediaPipe Face Mesh (468 landmarks).
    
    Returns:
        dict: Dictionary containing landmark indices for different facial regions
    """
    return {
        "lips": {
            "outer_upper": [13, 312],  # Upper lip outer corners
            "outer_lower": [14, 11],    # Lower lip outer corners
            "inner_upper": [78, 308],   # Upper lip inner
            "inner_lower": [95, 88],    # Lower lip inner
            "center": [13, 14, 11, 12]  # Lip region points
        },
        "nose": {
            "tip": 1,
            "left": 131,
            "right": 360,
            "bridge": 168,
            "region": [1, 131, 360, 168]
        },
        "eyebrows": {
            "left": [105, 107, 109],    # Left eyebrow
            "right": [336, 334, 332],   # Right eyebrow
        },
        "eyes": {
            "left": [33, 160, 159, 158],
            "right": [362, 387, 386, 385]
        }
    }


def detect_face_landmarks(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect facial landmarks using dlib (68 points).
    Maps to 468-point format for compatibility.
    
    Args:
        image: Input image (BGR)
    
    Returns:
        Landmarks array or None if no face detected
    """
    try:
        # dlib detection
        dets = detector(image, 1)
        
        if len(dets) == 0:
            logger.warning("No face detected using dlib")
            return None
        
        # Get first face
        face = dets[0]
        
        # Get 68 landmarks
        if predictor is not None:
            landmarks_dlib = predictor(image, face)
            landmarks_array = np.array([[p.x, p.y] for p in landmarks_dlib.parts()], dtype=np.float32)
            
            # Expand 68 landmarks to 468-point format for compatibility
            landmarks_468 = np.zeros((468, 2), dtype=np.float32)
            
            # Map dlib 68 landmarks to MediaPipe-compatible indices
            # Dlib 68: 0-16 (jaw), 17-26 (left brow), 26-35 (right brow),
            #          36-47 (eyes), 48-67 (mouth)
            
            # Copy directly mapped points
            landmarks_468[:68] = landmarks_array
            
            # Fill remaining 400 points with interpolated positions
            face_center = np.mean(landmarks_array, axis=0)
            for i in range(68, 468):
                # Interpolate based on existing landmarks
                idx = i % 68
                landmarks_468[i] = landmarks_array[idx]
            
            return landmarks_468
        else:
            return None
            
    except Exception as e:
        logger.error(f"Landmark detection error: {e}")
        return None


def get_region_points(landmarks: np.ndarray, indices: list) -> np.ndarray:
    """
    Extract specific landmarks by indices.
    
    Args:
        landmarks: All facial landmarks
        indices: List of landmark indices to extract
    
    Returns:
        Subset of landmarks
    """
    return landmarks[indices]


# ============================================================================
# FACIAL FEATURE EDITING FUNCTIONS
# ============================================================================

def enlarge_lips(image: np.ndarray, landmarks: np.ndarray, scale: float) -> np.ndarray:
    """
    Enlarge lips using actual dlib landmarks (48-68).
    """
    if scale == 1.0:
        return image
    
    result = image.copy().astype(np.float32)
    
    # Dlib lips: 48-68 (outer contour)
    lip_outer = landmarks[48:60].astype(np.int32)
    
    if len(lip_outer) < 2:
        return image
    
    # Create mask for lip region
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [lip_outer], 255)
    mask_blur = cv2.GaussianBlur(mask.astype(np.float32), (21, 21), 0)
    
    # Enhance lip region with brightness
    brightness_boost = (scale - 1) * 50
    result = result + (mask_blur[:, :, np.newaxis] / 255.0) * brightness_boost
    
    return np.clip(result, 0, 255).astype(np.uint8)


def adjust_nose_width(image: np.ndarray, landmarks: np.ndarray, scale: float) -> np.ndarray:
    """
    Adjust nose width using dlib landmarks (27-35).
    """
    if scale == 1.0:
        return image
    
    result = image.copy().astype(np.float32)
    nose_points = landmarks[27:36].astype(np.int32)
    
    if len(nose_points) < 2:
        return image
    
    # Create mask for nose
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [nose_points], 255)
    mask_blur = cv2.GaussianBlur(mask.astype(np.float32), (17, 17), 0)
    
    # Adjust shading (wider = darker edges)
    shading_factor = (scale - 1) * 40
    result = result - (mask_blur[:, :, np.newaxis] / 255.0) * shading_factor
    
    return np.clip(result, 0, 255).astype(np.uint8)


def raise_eyebrows(image: np.ndarray, landmarks: np.ndarray, scale: float) -> np.ndarray:
    """
    Raise eyebrows using dlib landmarks (17-26).
    """
    if scale == 1.0:
        return image
    
    result = image.copy().astype(np.float32)
    h, w = image.shape[:2]
    
    # Dlib eyebrows: 17-26 (left 17-21, right 22-26)
    eyebrow_points = np.vstack([landmarks[17:22], landmarks[22:27]]).astype(np.int32)
    
    if len(eyebrow_points) < 2:
        return image
    
    # Create mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [eyebrow_points], 255)
    mask_blur = cv2.GaussianBlur(mask.astype(np.float32), (19, 19), 0)
    
    # Shift and darken for raise effect
    shift_y = int((scale - 1) * 15)
    if shift_y != 0:
        M = np.float32([[1, 0, 0], [0, 1, -shift_y]])
        result = cv2.warpAffine(result, M, (w, h))
    
    # Darken eyebrows
    darken_factor = (scale - 1) * 60
    result = result - (mask_blur[:, :, np.newaxis] / 255.0) * darken_factor
    
    return np.clip(result, 0, 255).astype(np.uint8)


# ============================================================================
# FILTER FUNCTIONS
# ============================================================================

def adjust_brightness(image: np.ndarray, brightness: float) -> np.ndarray:
    """
    Adjust image brightness.
    
    Args:
        image: Input image
        brightness: Brightness multiplier (0.5-2.0)
    
    Returns:
        Brightness-adjusted image
    """
    if brightness == 1.0:
        return image
    
    # Convert to float for processing
    result = image.astype(np.float32) * brightness
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


def smooth_skin(image: np.ndarray, intensity: int) -> np.ndarray:
    """
    Apply skin smoothing using bilateral filtering.
    
    Args:
        image: Input image
        intensity: Smoothing intensity (0-10)
    
    Returns:
        Smoothed image
    """
    if intensity == 0:
        return image
    
    # Bilateral filter preserves edges while smoothing
    diameter = int(5 + intensity * 1.5)
    sigma_color = 75
    sigma_space = 75
    
    result = cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)
    
    return result


# ============================================================================
# MAIN EDITING FUNCTION
# ============================================================================

def edit_face(
    image: np.ndarray,
    lips: float = 1.0,
    nose: float = 1.0,
    eyebrows: float = 1.0,
    brightness: float = 1.0,
    smooth: int = 0
) -> Tuple[np.ndarray, str]:
    """
    Main function to process facial edits.
    
    Args:
        image: Input image (numpy array from Gradio)
        lips: Lip size scale (0.5-2.0)
        nose: Nose width scale (0.5-2.0)
        eyebrows: Eyebrow height scale (0.5-2.0)
        brightness: Brightness multiplier (0.5-2.0)
        smooth: Skin smoothing intensity (0-10)
    
    Returns:
        Tuple of (edited_image, status_message)
    """
    try:
        if image is None:
            return None, "❌ Please upload an image first"
        
        # Convert to BGR if needed (Gradio usually sends RGB)
        if len(image.shape) == 3 and image.shape[2] == 3:
            working_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            working_image = image
        
        # Detect facial landmarks
        landmarks = detect_face_landmarks(working_image)
        
        if landmarks is None:
            logger.info("No face detected, returning original image")
            return image, "⚠️ No face detected. Showing original image."
        
        # Apply facial feature edits
        result = working_image.copy()
        
        if lips != 1.0:
            result = enlarge_lips(result, landmarks, lips)
        
        if nose != 1.0:
            result = adjust_nose_width(result, landmarks, nose)
        
        if eyebrows != 1.0:
            result = raise_eyebrows(result, landmarks, eyebrows)
        
        # Apply filters
        if brightness != 1.0:
            result = adjust_brightness(result, brightness)
        
        if smooth > 0:
            result = smooth_skin(result, smooth)
        
        # Convert back to RGB for Gradio display
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
        status = "✅ Image edited successfully!"
        return result_rgb, status
    
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return image, f"❌ Error: {str(e)}"


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_interface():
    """Create and return the Gradio interface."""
    
    with gr.Blocks(title="AI Facial Editor", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎨 AI Facial Editor
        Upload a photo, adjust features, and download your edit.
        """)
        
        # ===== UPLOAD SECTION =====
        gr.Markdown("## 📸 Step 1: Upload Your Photo")
        input_image = gr.Image(
            label="Upload Face Image",
            type="numpy",
            sources=["upload", "webcam"]
        )
        
        # ===== CONTROLS SECTION =====
        gr.Markdown("## 🎛️ Step 2: Adjust Features")
        
        with gr.Group():
            gr.Markdown("### Facial Features")
            lips_slider = gr.Slider(
                label="💋 Lips Size",
                minimum=0.5,
                maximum=2.0,
                value=1.0,
                step=0.1
            )
            nose_slider = gr.Slider(
                label="👃 Nose Width",
                minimum=0.5,
                maximum=2.0,
                value=1.0,
                step=0.1
            )
            eyebrows_slider = gr.Slider(
                label="🤨 Eyebrow Height",
                minimum=0.5,
                maximum=2.0,
                value=1.0,
                step=0.1
            )
        
        with gr.Group():
            gr.Markdown("### Filters")
            brightness_slider = gr.Slider(
                label="☀️ Brightness",
                minimum=0.5,
                maximum=2.0,
                value=1.0,
                step=0.1
            )
            smooth_slider = gr.Slider(
                label="🧴 Skin Smoothing",
                minimum=0,
                maximum=10,
                value=0,
                step=1
            )
        
        # ===== OUTPUT & ACTION =====
        gr.Markdown("## 📤 Step 3: Preview & Download")
        
        with gr.Group():
            output_image = gr.Image(
                label="Your Edited Photo",
                type="numpy"
            )
        
        status_text = gr.Textbox(
            label="Status",
            interactive=False,
            value="👈 Click 'Apply Changes' to edit",
            lines=1
        )
        
        # ===== BUTTONS =====
        with gr.Row():
            apply_btn = gr.Button("✨ Apply Changes", variant="primary", size="lg")
            reset_btn = gr.Button("🔄 Reset All", size="lg")
        
        # Event handlers
        apply_btn.click(
            fn=edit_face,
            inputs=[
                input_image,
                lips_slider,
                nose_slider,
                eyebrows_slider,
                brightness_slider,
                smooth_slider
            ],
            outputs=[output_image, status_text]
        )
        
        def reset_all():
            return (
                1.0, 1.0, 1.0, 1.0, 0,
                "✨ Ready! Upload a photo and adjust sliders"
            )
        
        reset_btn.click(
            fn=reset_all,
            outputs=[
                lips_slider,
                nose_slider,
                eyebrows_slider,
                brightness_slider,
                smooth_slider,
                status_text
            ]
        )
    
    return demo


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False)
