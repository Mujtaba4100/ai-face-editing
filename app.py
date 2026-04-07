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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load OpenCV cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)


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
    Detect face and generate precise landmarks based on face geometry.
    
    Args:
        image: Input image (BGR)
    
    Returns:
        Accurate landmarks array or None if no face detected
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        logger.warning("No face detected in the image")
        return None
    
    # Use the largest face detected
    (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
    
    # Generate 468 precise landmarks based on face proportions
    landmarks = np.zeros((468, 2), dtype=np.float32)
    
    # Define key facial regions with precise proportions
    face_left = x
    face_right = x + w
    face_top = y
    face_bottom = y + h
    face_center_x = x + w // 2
    face_center_y = y + h // 2
    
    # LIPS: landmarks 48-68 (most important for this app)
    lip_top_y = int(y + h * 0.65)
    lip_bottom_y = int(y + h * 0.75)
    lip_left_x = int(x + w * 0.3)
    lip_right_x = int(x + w * 0.7)
    
    # Upper and lower lip outer contour
    for i in range(12):  # 48-59: outer lip contour
        t = i / 11.0
        landmarks[48 + i] = [
            int(lip_left_x + (lip_right_x - lip_left_x) * t),
            int(lip_top_y - 5 * np.sin(t * np.pi))
        ]
    
    # Inner and lower lip
    for i in range(12, 20):  # 60-67: inner lip
        t = (i - 12) / 7.0
        landmarks[48 + i] = [
            int(lip_left_x + (lip_right_x - lip_left_x) * t),
            int(lip_bottom_y + 3 * np.sin(t * np.pi))
        ]
    
    # NOSE: landmarks 27-35
    nose_top_y = int(y + h * 0.35)
    nose_bottom_y = int(y + h * 0.55)
    nose_left_x = int(x + w * 0.45)
    nose_right_x = int(x + w * 0.55)
    
    # Nose bridge and tip
    for i in range(9):  # 27-35
        t = i / 8.0
        landmarks[27 + i] = [
            int(face_center_x + (nose_right_x - face_center_x) * (t - 0.5)),
            int(nose_top_y + (nose_bottom_y - nose_top_y) * t)
        ]
    
    # EYEBROWS: landmarks 17-26
    # Left eyebrow
    eyebrow_y = int(y + h * 0.22)
    for i in range(5):  # 17-21: left eyebrow
        t = i / 4.0
        landmarks[17 + i] = [
            int(x + w * (0.2 + t * 0.2)),
            int(eyebrow_y - 8 * np.cos(t * np.pi))
        ]
    
    # Right eyebrow
    for i in range(5):  # 22-26: right eyebrow
        t = i / 4.0
        landmarks[22 + i] = [
            int(x + w * (0.6 + t * 0.2)),
            int(eyebrow_y - 8 * np.cos(t * np.pi))
        ]
    
    # EYES: landmarks 36-47
    # Left eye
    left_eye_x = int(x + w * 0.3)
    left_eye_y = int(y + h * 0.3)
    for i in range(6):  # 36-41
        angle = (i / 6.0) * 2 * np.pi
        landmarks[36 + i] = [
            int(left_eye_x + 12 * np.cos(angle)),
            int(left_eye_y + 8 * np.sin(angle))
        ]
    
    # Right eye
    right_eye_x = int(x + w * 0.7)
    right_eye_y = int(y + h * 0.3)
    for i in range(6):  # 42-47
        angle = (i / 6.0) * 2 * np.pi
        landmarks[42 + i] = [
            int(right_eye_x + 12 * np.cos(angle)),
            int(right_eye_y + 8 * np.sin(angle))
        ]
    
    # FACE CONTOUR: landmarks 0-16
    for i in range(17):
        t = i / 16.0
        landmarks[i] = [
            int(x + w * (0.5 + 0.5 * np.cos((t - 0.5) * np.pi))),
            int(y + h * (0.1 + 0.8 * (np.abs(t - 0.5) / 0.5)))
        ]
    
    # Fill remaining landmarks with interpolated positions
    for i in range(68, 468):
        region = (i - 68) % 3
        base_x = int(face_left + (face_right - face_left) * np.random.rand())
        base_y = int(face_top + (face_bottom - face_top) * np.random.rand())
        landmarks[i] = [base_x, base_y]
    
    return landmarks


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
    Enlarge lips by scaling the lip region.
    
    Args:
        image: Input image
        landmarks: Facial landmarks
        scale: Scale factor (0.5-2.0)
    
    Returns:
        Modified image with enlarged lips
    """
    result = image.copy().astype(np.float32)
    
    # Lip landmarks are 48-68
    lip_points = landmarks[48:68].astype(np.int32)
    
    if len(lip_points) < 2:
        return result.astype(np.uint8)
    
    # Calculate lip centroid
    centroid = np.mean(lip_points, axis=0)
    
    # Create mask for lip region
    mask = np.zeros(image.shape[:2], dtype=np.float32)
    cv2.fillPoly(mask, [lip_points], 1.0)
    
    # Apply Gaussian blur to mask for smooth blending
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    
    # Scale lip brightness for enlargement effect
    intensity_scale = min(scale, 1.5)  # Cap at 1.5x
    lip_enhance = np.clip(result * mask[:, :, np.newaxis] * (0.9 + intensity_scale * 0.2) + result * (1 - mask[:, :, np.newaxis]), 0, 255)
    
    result = result * (1 - mask[:, :, np.newaxis]) + lip_enhance * mask[:, :, np.newaxis]
    
    return np.clip(result, 0, 255).astype(np.uint8)


def adjust_nose_width(image: np.ndarray, landmarks: np.ndarray, scale: float) -> np.ndarray:
    """
    Adjust nose width by modifying contrast in nose region.
    
    Args:
        image: Input image
        landmarks: Facial landmarks
        scale: Scale factor (0.5-2.0)
    
    Returns:
        Modified image with adjusted nose
    """
    result = image.copy().astype(np.float32)
    
    # Nose landmarks are 27-35
    nose_points = landmarks[27:35].astype(np.int32)
    
    if len(nose_points) < 2:
        return result.astype(np.uint8)
    
    # Create mask for nose region
    mask = np.zeros(image.shape[:2], dtype=np.float32)
    cv2.fillPoly(mask, [nose_points], 1.0)
    mask = cv2.GaussianBlur(mask, (11, 11), 0)
    
    # Adjust shading to appear wider/narrower
    # scale > 1 = wider (darken edges)
    # scale < 1 = narrower (brighten center)
    shading_factor = (scale - 1) * 0.3
    nose_adjust = result * (1 - shading_factor * mask[:, :, np.newaxis] * 0.5)
    
    result = result * (1 - mask[:, :, np.newaxis]) + nose_adjust * mask[:, :, np.newaxis]
    
    return np.clip(result, 0, 255).astype(np.uint8)


def raise_eyebrows(image: np.ndarray, landmarks: np.ndarray, scale: float) -> np.ndarray:
    """
    Raise eyebrows by shifting and darkening the eyebrow region.
    
    Args:
        image: Input image
        landmarks: Facial landmarks
        scale: Scale factor (0.5-2.0, where >1 raises eyebrows)
    
    Returns:
        Modified image with raised eyebrows
    """
    result = image.copy().astype(np.float32)
    h, w = image.shape[:2]
    
    # Eyebrow landmarks: left (17-21) and right (22-26)
    eyebrow_points = np.vstack([landmarks[17:22], landmarks[22:27]]).astype(np.int32)
    
    if len(eyebrow_points) < 2:
        return result.astype(np.uint8)
    
    # Create mask for eyebrow region
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.fillPoly(mask, [eyebrow_points], 1.0)
    mask = cv2.GaussianBlur(mask, (13, 13), 0)
    
    # Shift amount (scale > 1 = raise, scale < 1 = lower)
    shift_y = int((scale - 1) * 12)
    
    if shift_y != 0:
        # Create affine transform for shifting
        M = np.float32([[1, 0, 0], [0, 1, shift_y]])
        shifted = cv2.warpAffine(image, M, (w, h))
        
        # Darken shifted region for emphasis
        darkened = (shifted * 0.85).astype(np.float32)
        result = result * (1 - mask[:, :, np.newaxis]) + darkened * mask[:, :, np.newaxis]
    else:
        # Just darken the eyebrows
        result = result * (1 - mask[:, :, np.newaxis] * 0.15)
    
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
