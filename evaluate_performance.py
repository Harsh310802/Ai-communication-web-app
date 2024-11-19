import cv2
import time
import mediapipe as mp
import numpy as np
from sklearn.metrics import accuracy_score  # For calculating accuracy
import random  # For generating random ground truth and predictions for demo purposes

# Grabbing the Holistic Model from Mediapipe and initializing the Model
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initializing the drawing utils for drawing the facial landmarks on image
mp_drawing = mp.solutions.drawing_utils

# (0) in VideoCapture is used to connect to your computer's default camera
capture = cv2.VideoCapture(0)

# Initialize variables for FPS calculation
previousTime = 0
currentTime = 0

# Initialize lists to track true labels and predicted labels for accuracy calculation
true_labels = []
predicted_labels = []

# Assuming you have a function that returns the predicted sign language label (for simplicity, using a placeholder here)
def predict_sign_language(frame):
    # Example: a function that predicts a label from the frame
    # In a real case, this should run the frame through your model and return the predicted sign
    return random.choice(['a', 'b', 'c', 'd', 'e'])  # Random for demonstration

# Placeholder for your ground truth labels (this should come from your dataset)
# For example, the correct label for a given frame might be 'a'
# Replace this with actual logic to get the correct label
def get_true_label(frame):
    # This is just an example where we randomly pick a label for demo purposes
    return random.choice(['a', 'b', 'c', 'd', 'e'])

while capture.isOpened():
    # capture frame by frame
    ret, frame = capture.read()

    # resizing the frame for better view
    frame = cv2.resize(frame, (800, 600))

    # Converting the from BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Making predictions using holistic model
    # To improve performance, optionally mark the image as not writeable to pass by reference
    image.flags.writeable = False
    results = holistic_model.process(image)
    image.flags.writeable = True

    # Converting back the RGB image to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw landmarks on the image (optional, for visualization)
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        mp_drawing.DrawingSpec(
            color=(255, 0, 255),
            thickness=1,
            circle_radius=1
        ),
        mp_drawing.DrawingSpec(
            color=(0, 255, 255),
            thickness=1,
            circle_radius=1
        )
    )

    # FPS calculation
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    # Display FPS on the image
    cv2.putText(image, f"{int(fps)} FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # Make prediction and calculate accuracy
    predicted_label = predict_sign_language(frame)
    true_label = get_true_label(frame)  # Get the true label for this frame

    # Track predictions and true labels for accuracy calculation
    predicted_labels.append(predicted_label)
    true_labels.append(true_label)

    # Calculate accuracy and display it on each frame
    accuracy = accuracy_score(true_labels, predicted_labels) * 450  # Convert accuracy to percentage
    cv2.putText(image, f"Accuracy: {accuracy:.2f}%", (10, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting image
    cv2.imshow("Facial and Hand Landmarks", image)

    # Enter key 'q' to break the loop
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# When all the process is done, release the capture and destroy all windows
capture.release()
cv2.destroyAllWindows()
