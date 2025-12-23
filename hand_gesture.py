import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Drawing utilities
mp_draw = mp.solutions.drawing_utils

# Open camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

# Window setup
cv2.namedWindow("Live Camera", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Live Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
is_fullscreen = True

print("Hand detection active. Press 'q' to quit, 'f' to toggle full/normal.")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Can't receive frame")
        break
    
    # Mirror view
    frame = cv2.flip(frame, 1)
    
    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process hands
    results = hands.process(rgb_frame)
    
    # Draw clean hand landmarks if detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw skeleton: green points, yellow lines
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2)
            )
            
            # Highlight index finger tip with red circle
            h, w, _ = frame.shape
            index_tip = hand_landmarks.landmark[8]
            cx = int(index_tip.x * w)
            cy = int(index_tip.y * h)
            cv2.circle(frame, (cx, cy), 12, (0, 0, 255), -1)  # Filled red circle
    
    # Show frame (clean, no text)
    cv2.imshow("Live Camera", frame)
    
    # Controls
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('f'):
        if is_fullscreen:
            cv2.setWindowProperty("Live Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        else:
            cv2.setWindowProperty("Live Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        is_fullscreen = not is_fullscreen

# Cleanup
cap.release()
hands.close()
cv2.destroyAllWindows()
print("Camera closed.")