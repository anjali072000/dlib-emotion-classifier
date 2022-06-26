import cv2
import dlib
import numpy as np
from keras.preprocessing import image
from sklearn.metrics import euclidean_distances
from keras.models import load_model
import matplotlib.pyplot as plt
import tensorflow as tf

model = load_model("model_landmark.h5")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
face_haar_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


labels_class = ['confused', 'engaged',
                'frustrated', 'Lookingaway', 'bored', 'drowsy']

cap = cv2.VideoCapture(1)

while True:
    ret, test_img = cap.read()
    if not ret:
        continue
    img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x+w, y+h), (0, 255, 0), thickness=3)

    if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
        break
    rects = detector(gray_img, 1)

    for rect in rects:
        X = []
        shape = predictor(gray_img, rects[0])

        shape_np = np.zeros((68, 2), dtype="int")

        for i in range(0, 68):
            shape_np[i] = (shape.part(i).x, shape.part(i).y)
        shape = shape_np

        eucl_dist = euclidean_distances(shape, shape)
        X.append(eucl_dist)

        X = np.array(X)

        X_train = tf.expand_dims(X, axis=-1)

        predictions = model.predict(X_train)

        # Display the landmarks
        for i, (x, y) in enumerate(shape):
            # Draw the circle to mark the keypoint
            cv2.circle(test_img, (x, y), 1, (0, 0, 255), -1)

        cv2.putText(test_img, labels_class[np.argmax(predictions)] + " " + str(round(predictions[0][np.argmax(
            predictions)] * 100, 2)), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print("Predicted state: ", labels_class[np.argmax(predictions)])

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ', resized_img)

cap.release()
cv2.destroyAllWindows
