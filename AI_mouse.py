import cv2
import mediapipe as mp
import pyautogui

# Initialize video capture and Mediapipe Hands
cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils

# Screen dimensions
screen_width, screen_height = pyautogui.size()

# Initialize variables
index_y = 0
thumb_y = 0

while True:
    # Capture and preprocess frame
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame using Mediapipe
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark
            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                if id == 8:  # Index finger tip
                    cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 255, 255), thickness=-1)
                    index_x = screen_width / frame_width * x
                    index_y = screen_height / frame_height * y
                    pyautogui.moveTo(index_x, index_y)

                if id == 4:  # Thumb tip
                    cv2.circle(img=frame, center=(x, y), radius=10, color=(255, 0, 0), thickness=-1)
                    thumb_x = screen_width / frame_width * x
                    thumb_y = screen_height / frame_height * y

            # Debugging: Log index and thumb positions
            print(f"Index Y: {index_y}, Thumb Y: {thumb_y}, Difference: {abs(index_y - thumb_y)}")

            # Perform click if thumb and index finger tips are close
            if index_y != 0 and thumb_y != 0 and abs(index_y - thumb_y) < 20:
                pyautogui.click()
                pyautogui.sleep(0.1)  # Reduced sleep time for responsiveness

    # Show the frame
    cv2.imshow('AI Mouse', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
