import cv2
import numpy as np
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

drawing = False
prev_x, prev_y = None, None
canvas = None
color_index = 0
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
thickness = 5
eraser_thickness = 100 
eraser_mode = False
navbar_height = 50

def draw_navbar(frame):
    # options and an eraser at the top
    
    w = frame.shape[1]
    for i, color in enumerate(colors):
        cv2.rectangle(frame, (i * navbar_height, 0), ((i + 1) * navbar_height, navbar_height), color, -1)
    cv2.circle(frame, (640,25), 15, (0,255,255), -1 )
    # Draw eraser
    cv2.rectangle(frame, (w - navbar_height, 0), (w, navbar_height), (0, 0, 0), -1)
    cv2.putText(frame, "E", (w - navbar_height + 10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def check_navbar_selection(x, y, frame):
    global color_index, eraser_mode
    w = frame.shape[1]
    if y < navbar_height:
        if x < len(colors) * navbar_height:
            color_index = x // navbar_height
            eraser_mode = False
            cv2.circle(frame, (640,25), 15, colors[color_index], -1 )
        elif w - navbar_height <= x < w:
            eraser_mode = True
            cv2.putText(frame, "E", (w - navbar_height + 10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.circle(frame, (640,25), 15, (0,0,0), -1 )

# Video capture from webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #  Recognise hands
    result = hands.process(rgb_frame)

    # Draw the navbar
    draw_navbar(frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get the tip of the index finger
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            h, w, _ = frame.shape
            dist = math.hypot(index_finger_tip.x - thumb_tip.x, index_finger_tip.y - thumb_tip.y)
            x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            # a,b = int(thumb_tip.x * w), int(index_finger_tip.y * h)

            # Check navbar selection
            check_navbar_selection(x, y, frame)

            if index_finger_tip.z < 0.2 and y > 50 : # Ensure drawing starts below navbar
                drawing = True
                cv2.circle(frame, (640,25), 15, colors[color_index], -1 )
                if prev_x is None and prev_y is None:
                    prev_x, prev_y = x, y
                else:
                    if canvas is None:
                        canvas = np.zeros_like(frame)
                    if eraser_mode:
                        cv2.circle(frame, (640,25), 15, (0,0,0), -1 )
                        canvas = cv2.line(canvas, (prev_x, prev_y), (x, y), (0,0,0), eraser_thickness)
                        cv2.putText(frame, "E", (w - 50 + 10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    else :
                        if dist <= 0.08 :
                            canvas = cv2.line(canvas, (prev_x, prev_y), (x, y), colors[color_index], thickness)
                    prev_x, prev_y = x, y
            else:
                drawing = False
                prev_x, prev_y = None, None

            # Draw the hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Overlay the canvas on the frame
    if canvas is not None:
        frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Display the resulting frame
    cv2.imshow('Air Canvas', frame)

    # Press 'c' to clear the canvas
    key = cv2.waitKey(1)
    if key == ord('c'):
        canvas = None
    elif key == 27:  # Press 'Esc' to exit
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
