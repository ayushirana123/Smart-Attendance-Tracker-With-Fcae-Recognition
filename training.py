import cv2
import os
import numpy as np
from PIL import Image

# Initialize the recognizer and face detector
recognizer = cv2.face.LBPHFaceRecognizer_create()  # Use LBPH face recognizer
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")  # Load Haar Cascade for face detection

# Function to get images and labels
def getImagesAndLabels(path):
    # Get list of all image paths in the folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []  # List to store face samples (face images)
    Ids = []  # List to store the corresponding labels (IDs)

    for imagePath in imagePaths:
        # Check for image file types (.jpg, .png, etc.)
        if imagePath.endswith('.jpg') or imagePath.endswith('.png'):
            pilImage = Image.open(imagePath).convert('L')  # Convert the image to grayscale
            imageNp = np.array(pilImage, 'uint8')  # Convert PIL image to NumPy array
            # Extract the ID from the image filename, assuming format 'name.ID.jpg'
            try:
                Id = int(os.path.split(imagePath)[-1].split(".")[1])  # Assuming 'name.ID.jpg' format
            except ValueError:
                print(f"Skipping image {imagePath} due to improper filename format.")
                continue

            # Detect faces in the image
            faces = detector.detectMultiScale(imageNp, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

            # Process each detected face
            for (x, y, w, h) in faces:
                faceSamples.append(imageNp[y:y + h, x:x + w])  # Store the detected face in faceSamples
                Ids.append(Id)  # Store the corresponding ID

                # Optionally, draw a rectangle around the face for visualization
                cv2.rectangle(imageNp, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Show the image with the detected faces
            cv2.imshow("Training on image...", imageNp)
            cv2.waitKey(10)

    # Return the face samples and corresponding IDs
    return faceSamples, Ids

# Path to the folder containing your training images
imageFolderPath = 'TrainingImage'

# Get images and labels from the folder
faces, Ids = getImagesAndLabels(imageFolderPath)

# If faces were found, train the recognizer
if len(faces) > 0:
    recognizer.train(faces, np.array(Ids))  # Train the recognizer with the collected faces and IDs
    recognizer.save('TrainingImageLabel/trainner.yml')  # Save the trained model
    print("Model training complete and saved as 'trainner.yml'")
else:
    print("No faces found in the training images")

# Close all OpenCV windows
cv2.destroyAllWindows()
