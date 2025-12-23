import cv2
import numpy as np

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow('Astral Synthesis', cv2.WINDOW_NORMAL)

print("Press 'q' to quit, 'f' for fullscreen, 'SPACE' to capture background.")

# Background for subtraction
background = None
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    
    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame, learningRate=0.01)
    
    # Clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    
    # Blur for smooth edges
    fg_mask = cv2.GaussianBlur(fg_mask, (15, 15), 0)
    
    # Create black silhouette
    silhouette = np.zeros_like(frame)
    
    # Apply mask to create silhouette effect
    mask_3channel = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
    silhouette = cv2.bitwise_and(frame, mask_3channel)
    
    # Convert to pure black silhouette
    gray_silhouette = cv2.cvtColor(silhouette, cv2.COLOR_BGR2GRAY)
    _, binary_silhouette = cv2.threshold(gray_silhouette, 10, 255, cv2.THRESH_BINARY)
    
    # Create cosmic background (deep black with stars)
    cosmic_bg = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Add sparse stars
    if np.random.random() < 0.01:
        star_x = np.random.randint(0, w)
        star_y = np.random.randint(0, h)
        cv2.circle(cosmic_bg, (star_x, star_y), 1, (200, 200, 200), -1)
    
    # Combine: Black silhouette on cosmic background
    silhouette_colored = np.zeros_like(frame)
    silhouette_colored[binary_silhouette > 0] = [0, 0, 0]  # Pure black
    
    # Add subtle glow around silhouette
    glow = cv2.GaussianBlur(binary_silhouette, (25, 25), 0)
    glow_colored = np.zeros_like(frame)
    glow_colored[:, :, 0] = glow // 3  # Blue glow
    glow_colored[:, :, 1] = glow // 4
    glow_colored[:, :, 2] = glow // 2
    
    # Final composite
    output = cosmic_bg.copy()
    output = cv2.add(output, glow_colored)
    output[binary_silhouette > 0] = [0, 0, 0]
    
    # Show original and silhouette side by side
    combined = np.hstack([frame, output])
    
    cv2.imshow('Astral Synthesis', combined)
    
    # Handle keys
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):
        cv2.setWindowProperty('Astral Synthesis', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cap.release()
cv2.destroyAllWindows()