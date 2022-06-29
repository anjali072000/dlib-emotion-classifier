import cv2
import dlib
import numpy as np
import time

# used to record the time when we processed last frame
prev_frame_time = 0

# used to record the time at which we processed current frame
new_frame_time = 0

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
cap = cv2.VideoCapture(0)

while True:
    # Capture the image from the webcam
    ret, image = cap.read()

    # Convert the image color to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect the face
    rects = detector(gray, 1)
    # Detect landmarks for each face
    print("rects: ", rects)
    for rect in rects:
        # Get the landmark points
        shape = predictor(gray, rect)
        print("shape: ", shape)
        # Convert it to the NumPy Array
        shape_np = np.zeros((68, 2), dtype="int")
        print("shape_np: ", shape_np)
        for i in range(0, 68):
            shape_np[i] = (shape.part(i).x, shape.part(i).y)
        shape = shape_np

        # Display the landmarks
        for i, (x, y) in enumerate(shape):
            # Draw the circle to mark the keypoint
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

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

    # Display the image
    cv2.imshow('Landmark Detection', image)

    # Press the escape button to terminate the code
    if cv2.waitKey(10) == 27:
        break

cap.release()
