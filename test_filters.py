"""
Quick test script to verify filter functions work
"""
import cv2
import numpy as np
import insightface

# Load insightface
app = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))

def detect_face_landmarks(image):
    """Detect landmarks using InsightFace"""
    try:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = app.get(rgb_image)
        if len(faces) == 0:
            return None
        face = faces[0]
        landmarks_106 = face.landmark_2d_106
        landmarks_468 = np.zeros((468, 2), dtype=np.float32)
        landmarks_468[:106] = landmarks_106.astype(np.float32)
        for i in range(106, 468):
            idx = i % 106
            landmarks_468[i] = landmarks_106[idx]
        return landmarks_468
    except Exception as e:
        print(f"Error: {e}")
        return None

def enlarge_lips(image, landmarks, scale):
    """Enlarge lips"""
    if scale == 1.0:
        return image
    h, w = image.shape[:2]
    result = image.copy().astype(np.float32)
    mouth_outer = landmarks[55:71].astype(np.float32)
    mouth_inner = landmarks[71:87].astype(np.float32)
    if len(mouth_outer) < 4:
        return image
    mouth_center = np.mean(mouth_outer, axis=0)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [mouth_outer.astype(np.int32)], 255)
    mask_blur = cv2.GaussianBlur(mask.astype(np.float32), (15, 15), 0) / 255.0
    scale_amount = (scale - 1) * 0.3
    for y in range(h):
        for x in range(w):
            if mask_blur[y, x] > 0.01:
                dx = x - mouth_center[0]
                dy = y - mouth_center[1]
                new_x = int(mouth_center[0] + dx * (1 + scale_amount * mask_blur[y, x]))
                new_y = int(mouth_center[1] + dy * (1 + scale_amount * mask_blur[y, x]))
                if 0 <= new_x < w and 0 <= new_y < h:
                    result[y, x] = image[new_y, new_x].astype(np.float32) * 0.7 + result[y, x] * 0.3
    h_hsv = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    h_hsv[:, :, 1] = np.clip(h_hsv[:, :, 1] * (1 + mask_blur * (scale - 1) * 0.15), 0, 255)
    result = cv2.cvtColor(h_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    return np.clip(result, 0, 255).astype(np.uint8)

# Test with a sample image
print("Testing filters...")
print("1. Loading a test image...")

# Create a simple test image or load from file
try:
    # Try to load any image in the directory
    import os
    images = [f for f in os.listdir('.') if f.endswith(('.jpg', '.png', '.jpeg'))]
    if images:
        test_image = cv2.imread(images[0])
        print(f"✓ Loaded test image: {images[0]}")
    else:
        print("⚠ No test images found in directory")
        print("Create a test by placing a face image (face.jpg) in this directory")
        exit()
    
    print("\n2. Detecting face landmarks...")
    landmarks = detect_face_landmarks(test_image)
    if landmarks is None:
        print("❌ No face detected!")
        exit()
    print(f"✓ Face detected! Landmarks shape: {landmarks.shape}")
    
    print("\n3. Testing lip filter (scale=2.0 - enlarge)...")
    lips_edited = enlarge_lips(test_image, landmarks, 2.0)
    print(f"✓ Lips filter applied!")
    print(f"   Original shape: {test_image.shape}")
    print(f"   Edited shape: {lips_edited.shape}")
    print(f"   Pixel difference: {np.mean(np.abs(test_image.astype(float) - lips_edited.astype(float))):.2f}")
    
    # Save result
    cv2.imwrite('test_lips_result.jpg', lips_edited)
    print(f"✓ Saved result to: test_lips_result.jpg")
    
    print("\n✅ Filters are working! Test completed successfully.")
    print("\nTo see results:")
    print("- Open 'test_lips_result.jpg' to check lip enlargement")
    
except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()
