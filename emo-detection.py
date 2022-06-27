import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import euclidean_distances
import tensorflow as tf
from keras.models import load_model

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

model = load_model("model_landmark.h5")

img = cv2.imread("test/a2.jpeg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

labels_class = ['confused', 'engaged',
                'frustrated', 'Lookingaway', 'bored', 'drowsy']

# Detect the face
rects = detector(gray, 1)
#print("rects: ", rects)
X = []

for rect in rects:
    # print(rect)
    # Get the landmark points
    shape = predictor(gray, rect)
    # Convert it to the NumPy Array
    shape_np = np.zeros((68, 2), dtype="int")
    for i in range(0, 68):
        shape_np[i] = (shape.part(i).x, shape.part(i).y)
    shape = shape_np

    eucl_dist = euclidean_distances(shape, shape)
    X.append(eucl_dist)

    X = np.array(X)
    print(X.shape)

    X_train = tf.expand_dims(X, axis=-1)
    print(X_train.shape)

    predictions = model.predict(X_train)
    print(predictions)

    # Display the landmarks
    for i, (x, y) in enumerate(shape):
        # Draw the circle to mark the keypoint
        cv2.circle(img, (x, y), 3, (255, 0, 0), -1)

    print("Predicted state: ", labels_class[np.argmax(predictions)])

    plt.imshow(img)
    plt.show()
