import cv2
import numpy as np

# Load the trained model (ensure the path is correct)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('TrainingImageLabel/trainner.yml')

# Path to the Haar Cascade for face detection
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# Define font for displaying text
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize webcam
cam = cv2.VideoCapture(0)

# Start video capture
while True:
    ret, im = cam.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Convert image to grayscale for face detection
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(gray, 1.2, 5)

    # Loop through all faces detected
    for (x, y, w, h) in faces:
        # Predict the ID of the face using the trained recognizer
        Id, conf = recognizer.predict(gray[y:y + h, x:x + w])

        # If confidence is below a threshold, consider it as unknown
        if conf < 50:
            Id = "Unknown"

        # Draw rectangle around face
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the ID on the image
        cv2.putText(im, str(Id), (x, y - 10), font, 1, (255, 255, 255), 2)

    # Display the resulting image
    cv2.imshow('Face Recognition', im)

    # Break the loop if the user presses 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cam.release()
cv2.destroyAllWindows()
