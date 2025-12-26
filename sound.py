import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd

# --- MediaPipe Hand Setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# --- Audio Setup ---
SAMPLE_RATE = 44100  # Standard audio rate
current_freq = 220.0  # Starting frequency (A3)
current_amp = 0.0     # Starting volume (silent)
target_freq = 220.0
target_amp = 0.0

def audio_callback(outdata, frames, time, status):
    global current_freq, current_amp, target_freq, target_amp
    
    # Smoothly interpolate for calm feel
    current_freq += 0.05 * (target_freq - current_freq)
    current_amp += 0.05 * (target_amp - current_amp)
    
    t = np.linspace(0, frames / SAMPLE_RATE, frames, False)
    wave = current_amp * np.sin(2 * np.pi * current_freq * t)
    outdata[:] = wave.reshape(-1, 1)

# Start audio stream
stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback)
stream.start()

# --- Camera Setup ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

cv2.namedWindow("Live Camera", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Live Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
is_fullscreen = True

print("Sound active. Use right hand height for pitch. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    hand_detected = False
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw clean skeleton
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2)
            )
            
            # Red circle on index tip
            h, w, _ = frame.shape
            index_tip = hand_landmarks.landmark[8]
            cx = int(index_tip.x * w)
            cy = int(index_tip.y * h)
            cv2.circle(frame, (cx, cy), 12, (0, 0, 255), -1)
            
            # Use RIGHT hand for control (MediaPipe labels from your perspective)
            if results.multi_handedness:
                handedness = results.multi_handedness[results.multi_hand_landmarks.index(hand_landmarks)].classification[0].label
                if handedness == "Right":  # Your right hand
                    hand_detected = True
                    # Map Y position (lower cy = higher hand) to frequency 220–880 Hz
                    normalized_y = 1.0 - (cy / h)  # 0 at bottom, 1 at top
                    target_freq = 220 + normalized_y * 660  # 220 Hz (A3) to 880 Hz (A5)
                    target_amp = 0.3  # Gentle volume
    
    # If no right hand → fade out sound
    if not hand_detected:
        target_amp = 0.0
    
    cv2.imshow("Live Camera", frame)
    
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
stream.stop()
stream.close()
cv2.destroyAllWindows()
print("Done.")