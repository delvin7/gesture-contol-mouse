import cv2
import numpy as np
import mediapipe as mp
import pyautogui
from IPython.display import Image

# Initialize video capture
cap = cv2.VideoCapture(0)   # capture video '0' one cam

# Initialize hand detection
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils

# Get screen size
screen_width, screen_height = pyautogui.size()

# Initialize variables for mouse movement smoothing
smoothening = 9
plocx, plocy = 0, 0
clocx, clocy = 0, 0 

# Initialize index_y outside the loop
index_y = 0

while True:
    # Read frame from video capture
    _, frame = cap.read()   # read data from cap
    
    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)
    
    # Get frame dimensions
    frame_height, frame_width, _ = frame.shape
    
    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
    
    # Detect hands in the frame
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks
    
    if hands:
        for hand in hands:
            # Draw landmarks on the frame
            drawing_utils.draw_landmarks(frame, hand)
            
            # Get hand landmarks
            landmarks = hand.landmark
            
            # Loop through landmarks
            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                
                if id == 8:  # Index finger tip
                    cv2.circle(img=frame, center=(x, y), radius=15, color=(0, 255, 255))
                    
                    # Calculate screen coordinates
                    index_x = int((screen_width - 1) * landmark.x)
                    index_y = int((screen_height - 1) * landmark.y)
                    
                    # Smooth mouse movement
                    clocx = plocx + (index_x - plocx) / smoothening
                    clocy = plocy + (index_y - plocy) / smoothening
                    pyautogui.moveTo(clocx, clocy)
                    plocx, plocy = clocx, clocy
                
                if id == 4:  # Thumb tip
                    cv2.circle(img=frame, center=(x, y), radius=15, color=(0, 255, 255))
                    thumb_x = (screen_width / frame_width) * x
                    thumb_y = (screen_height / frame_height) * y
                    print('distance : ', abs(index_y - thumb_y))
                    
                    # Check for double-click condition
                    if abs(index_y - thumb_y) < 70:
                        print('double click')
                        pyautogui.doubleClick()
                        pyautogui.sleep(1)
    
    # Display frame
    cv2.imshow('Virtual Mouse', frame)
    
    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
