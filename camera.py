import cv2
import numpy as np

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot access webcam")
    exit()

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Camera active. Press 'q' to quit, 'f' for fullscreen toggle.")

# Create named window
cv2.namedWindow('Astral Synthesis', cv2.WINDOW_NORMAL)

prev_frame = None
fullscreen = False

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip frame (mirror mode)
    frame = cv2.flip(frame, 1)
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    # Motion detection
    if prev_frame is not None:
        frame_diff = cv2.absdiff(prev_frame, gray)
        motion_level = float(np.mean(frame_diff))
        
        # Display motion value
        cv2.putText(frame, f"Motion: {motion_level:.1f}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    
    prev_frame = gray.copy()
    
    # Show frame
    cv2.imshow('Astral Synthesis', frame)
    
    # Handle keys
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):
        fullscreen = not fullscreen
        if fullscreen:
            cv2.setWindowProperty('Astral Synthesis', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty('Astral Synthesis', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

cap.release()
cv2.destroyAllWindows()