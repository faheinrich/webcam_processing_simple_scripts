import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

import mediapipe as mp


"""
Z Coordinate

The Z Coordinate is an experimental value that is calculated for every landmark. 
It is measured in "image pixels" like the X and Y coordinates, but it is not a true 3D value. 
The Z axis is perpendicular to the camera and passes between a subject's hips. 
The origin of the Z axis is approximately the center point 
between the hips (left/right and front/back relative to the camera). 
Negative Z values are towards the camera; positive values are away from it. 
The Z coordinate does not have an upper or lower bound.

Please note that while facial points have a Z coordinate, you should ignore it.
"""

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
# For webcam input:

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

cap = cv2.VideoCapture(0)
with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = pose.process(image)

        # landmarks = np.zeros((33, 3))
        # for i, lm in enumerate(results.pose_landmarks.landmark):
        #     landmarks[i, ...] = np.array([lm.x, lm.z, -lm.y])
        # # print(results.pose_landmarks.ListFields())
        # # print(landmarks)


        # ax.clear()
        # ax.scatter(*landmarks.T)
        # # plt.xlim(-0.5,1,5)
        # # plt.ylim(-3,1)
        # # plt.scatter(*landmarks.T)
        # plt.savefig('mediapose_test.png')
        # cv2.imshow("pseudo 3d", cv2.imread('mediapose_test.png')[:,:,::-1])
        





        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow('MediaPipe Pose', cv2.resize(image, (image.shape[1]*2, image.shape[0]*2)))
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()

