import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import euclidean_distances

# face_haar_cascade = cv2.CascadeClassifier(
#     cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# img = cv2.imread("r3.jpeg")

# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# faces_detected = face_haar_cascade.detectMultiScale(img, 1.32, 5)

# for (x, y, w, h) in faces_detected:
#     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=7)
#     plt.imshow(img)
#     plt.show()


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

img = cv2.imread("r3.jpeg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# Detect the face
rects = detector(gray, 1)
# Detect landmarks for each face
print("rects: ", rects)
for rect in rects:
    # Get the landmark points
    shape = predictor(gray, rect)
    # Convert it to the NumPy Array
    shape_np = np.zeros((68, 2), dtype="int")
    for i in range(0, 68):
        shape_np[i] = (shape.part(i).x, shape.part(i).y)
    shape = shape_np

    eucl_dist = euclidean_distances(shape, shape)
    print(eucl_dist)
    # Display the landmarks
    for i, (x, y) in enumerate(shape):
        # Draw the circle to mark the keypoint
        cv2.circle(img, (x, y), 2, (255, 0, 0), -1)

    plt.imshow(img)
    plt.show()
