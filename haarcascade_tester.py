import cv2
import dlib
import numpy as np
from keras.preprocessing import image
from sklearn.metrics import euclidean_distances
from keras.models import load_model
import matplotlib.pyplot as plt
import tensorflow as tf
import time

model = load_model("model_landmark.h5")

predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

face_haar_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


labels_class = ['Neutral', 'engaged',
                'frustrated', 'Lookingaway', 'bored', 'drowsy']


# used to record the time when we processed last frame
prev_frame_time = 0

# used to record the time at which we processed current frame
new_frame_time = 0

cap = cv2.VideoCapture(0)

while True:
    ret, test_img = cap.read()
    if not ret:
        continue
    img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        X = []

        # cv2.rectangle(test_img, (x, y), (x+w, y+h), (0, 255, 0), thickness=3)

        shape = predictor(gray_img, dlib.rectangle(x, y, x+w, y+h))

        shape_np = np.zeros((68, 2), dtype="int")

        for i in range(0, 68):
            shape_np[i] = (shape.part(i).x, shape.part(i).y)
        shape = shape_np

        eucl_dist = euclidean_distances(shape, shape)
        X.append(eucl_dist)

        X = np.array(X)

        X_train = tf.expand_dims(X, axis=-1)

        predictions = model.predict(X_train)
        print(predictions)

        # Display the landmarks
        for i, (_x, _y) in enumerate(shape):
            # Draw the circle to mark the keypoint
            cv2.circle(test_img, (_x, _y), 1, (0, 0, 0), -1)

        cv2.putText(test_img, labels_class[np.argmax(predictions)] + " " + str(round(predictions[0][np.argmax(
            predictions)] * 100, 2)), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print("Predicted state: ", labels_class[np.argmax(predictions)])

        if (labels_class[np.argmax(predictions)] == "Neutral" or labels_class[np.argmax(predictions)] == "engaged" or labels_class[np.argmax(predictions)] == "frustrated"):

            # draw box over face
            cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)

        else:

            # draw box over face
            cv2.rectangle(image, (x, y), (w, h), (0, 0, 255), 2)

    # time when we finish processing for this frame
    new_frame_time = time.time()

    # Calculating the fps

    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time

    # converting the fps into integer
    fps = int(fps)

    # converting the fps to string so that we can display it on frame
    # by using putText function
    fps = str(fps)

    # putting the FPS count on the frame
    cv2.putText(test_img, "FPS: " + fps, (0, 25), cv2.FONT_HERSHEY_SIMPLEX,
                1, (100, 255, 0), 1, cv2.LINE_AA)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ', resized_img)

    if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows
