import mediapipe as mp
import cv2
import dlib
import numpy as np
from sklearn.metrics import euclidean_distances
from keras.models import load_model
import tensorflow as tf
import time


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

model = load_model("model_landmark.h5")

predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')


labels_class = ['Neutral', 'engaged',
                'frustrated', 'Lookingaway', 'bored', 'drowsy']


# used to record the time when we processed last frame
prev_frame_time = 0

# used to record the time at which we processed current frame
new_frame_time = 0


# For webcam input:
cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)

        # Draw the face detection annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if results.detections:
            for detection in results.detections:
                # mp_drawing.draw_detection(image, detection)
                x = detection.location_data.relative_bounding_box.xmin
                y = detection.location_data.relative_bounding_box.ymin
                width = detection.location_data.relative_bounding_box.width
                height = detection.location_data.relative_bounding_box.height

                img_shape = image.shape

                r_x = int(x * img_shape[1])
                r_y = int(y * img_shape[0])
                r_w = int(x * img_shape[1]) + int(width * img_shape[1])
                r_h = int(y * img_shape[0]) + int(height * img_shape[0])

                X = []

                shape = predictor(gray_img, dlib.rectangle(r_x, r_y, r_w, r_h))

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
                for i, (x, y) in enumerate(shape):
                    # Draw the circle to mark the keypoint
                    cv2.circle(image, (x, y), 1, (0, 0, 0), -1)

                # cv2.rectangle(image, (r_x, r_y), (r_w, r_h), (0, 255, 0), 2)

                cv2.putText(image, labels_class[np.argmax(predictions)] + " " + str(round(predictions[0][np.argmax(
                    predictions)] * 100, 2)), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                print("Predicted state: ",
                      labels_class[np.argmax(predictions)])

                if (labels_class[np.argmax(predictions)] == "Neutral" or labels_class[np.argmax(predictions)] == "engaged" or labels_class[np.argmax(predictions)] == "frustrated"):

                    # draw box over face
                    cv2.rectangle(image, (r_x, r_y),
                                  (r_w, r_h), (0, 255, 0), 2)

                else:

                    # draw box over face
                    cv2.rectangle(image, (r_x, r_y),
                                  (r_w, r_h), (0, 0, 255), 2)

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
        cv2.putText(image, "FPS: " + fps, (0, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (100, 255, 0), 1, cv2.LINE_AA)

        resized_img = cv2.resize(image, (1000, 700))
        cv2.imshow('Facial emotion analysis ', resized_img)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
