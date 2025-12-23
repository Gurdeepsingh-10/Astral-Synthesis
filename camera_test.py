import cv2

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

cv2.namedWindow("Live Camera", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Live Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
is_fullscreen = True

print("Camera opened. Press 'q' to quit, 'f' to toggle full/normal.")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Can't receive frame")
        break
    
    # Mirror the video horizontally (1 = horizontal flip)
    frame = cv2.flip(frame, 1)
    
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

cap.release()
cv2.destroyAllWindows()
print("Camera closed.")