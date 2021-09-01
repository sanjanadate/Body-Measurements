import math

import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
import matplotlib.pyplot as plt
from IPython import get_ipython
from PIL import Image
from numpy import asarray

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def spectacles():
    mp_face_detection = mp.solutions.face_detection
    cap = cv2.VideoCapture(0)
    with mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5) as face_detection:
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
            results = face_detection.process(image)

            # Draw the face detection annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.detections:
                for detection in results.detections:
                    left = mp_face_detection.get_key_point(
                        detection, mp_face_detection.FaceKeyPoint.LEFT_EAR_TRAGION)
                    mp_drawing.draw_detection(image, detection)

                    right = mp_face_detection.get_key_point(
                        detection, mp_face_detection.FaceKeyPoint.RIGHT_EAR_TRAGION)
                    mp_drawing.draw_detection(image, detection)

            cv2.imshow('MediaPipe Face Detection', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()

    def frameSize(left, right):
        size = right.x - left.x
        if size < 121:
            return "small"
        elif size < 152:
            return "medium"
        else:
            return "large"

    print('You will need spectacles in a ',frameSize(left, right), ' size.')
    cv2.destroyAllWindows()

def walker():
    print('Walker')
    mp_holistic = mp.solutions.holistic
    mp_holistic.POSE_CONNECTIONS
    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
    mp_drawing.draw_landmarks
    landmarksList = []

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        img = cv2.imread("image1.jpg")
        cv2.imshow("Input Image", img)
        cv2.waitKey(0)

        numpydata = asarray(img)
        image = cv2.cvtColor(numpydata, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
        landmarksList.append(results.pose_landmarks)
        imageWalker = Image.fromarray(image)
        imageWalker.save("image1_walker.jpg")
        imgPoints = cv2.imread("image1_walker.jpg")
        cv2.imshow("Body with Points", imgPoints)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        calf = distance(results.pose_world_landmarks.landmark[26].x, results.pose_world_landmarks.landmark[26].y,results.pose_world_landmarks.landmark[26].z,results.pose_world_landmarks.landmark[28].x, results.pose_world_landmarks.landmark[28].y,results.pose_world_landmarks.landmark[28].z )
        thigh = distance(results.pose_world_landmarks.landmark[26].x, results.pose_world_landmarks.landmark[26].y,
                         results.pose_world_landmarks.landmark[26].z, results.pose_world_landmarks.landmark[24].x,
                         results.pose_world_landmarks.landmark[24].y, results.pose_world_landmarks.landmark[24].z)
        foot = distance(results.pose_world_landmarks.landmark[28].x, results.pose_world_landmarks.landmark[28].y,
                        results.pose_world_landmarks.landmark[28].z, results.pose_world_landmarks.landmark[30].x,
                        results.pose_world_landmarks.landmark[30].y, results.pose_world_landmarks.landmark[30].z)

        leg = calf + thigh + foot

        upper_arm = distance(results.pose_world_landmarks.landmark[13].x, results.pose_world_landmarks.landmark[13].y,
                             results.pose_world_landmarks.landmark[13].z, results.pose_world_landmarks.landmark[11].x,
                             results.pose_world_landmarks.landmark[11].y, results.pose_world_landmarks.landmark[11].z)

        forearm = distance(results.pose_world_landmarks.landmark[13].x, results.pose_world_landmarks.landmark[13].y,
                           results.pose_world_landmarks.landmark[13].z, results.pose_world_landmarks.landmark[15].x,
                           results.pose_world_landmarks.landmark[15].y, results.pose_world_landmarks.landmark[15].z)

        hand = upper_arm + forearm

        torso = distance(results.pose_world_landmarks.landmark[23].x, results.pose_world_landmarks.landmark[23].y,
                         results.pose_world_landmarks.landmark[23].z, results.pose_world_landmarks.landmark[11].x,
                         results.pose_world_landmarks.landmark[11].y, results.pose_world_landmarks.landmark[11].z)

        body = torso + leg
        feet_to_wrist = body - hand
        feet_to_wrist = "{:.2f}".format(feet_to_wrist*100)
        print('Height of the walker should be ',feet_to_wrist, ' cm.')


def standingFrame():
    mp_holistic = mp.solutions.holistic
    mp_holistic.POSE_CONNECTIONS
    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
    mp_drawing.draw_landmarks
    a = []

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        img = cv2.imread("image1.jpg")
        cv2.imshow("Input Image", img)
        cv2.waitKey(0)

        numpydata = asarray(img)
        image = cv2.cvtColor(numpydata, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
        a.append(results.pose_landmarks)
        im = Image.fromarray(image)
        im.save("image1_standing_frame.jpg")
        imgPoints = cv2.imread("image1_standing_frame.jpg")
        cv2.imshow("Body with Points", imgPoints)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print("Measurements for Standing Frame -")
        print()
        calf = distance(results.pose_world_landmarks.landmark[26].x, results.pose_world_landmarks.landmark[26].y,
                        results.pose_world_landmarks.landmark[26].z, results.pose_world_landmarks.landmark[28].x,
                        results.pose_world_landmarks.landmark[28].y, results.pose_world_landmarks.landmark[28].z)

        thigh = distance(results.pose_world_landmarks.landmark[26].x, results.pose_world_landmarks.landmark[26].y,
                         results.pose_world_landmarks.landmark[26].z, results.pose_world_landmarks.landmark[24].x,
                         results.pose_world_landmarks.landmark[24].y, results.pose_world_landmarks.landmark[24].z)

        foot = distance(results.pose_world_landmarks.landmark[28].x, results.pose_world_landmarks.landmark[28].y,
                        results.pose_world_landmarks.landmark[28].z, results.pose_world_landmarks.landmark[30].x,
                        results.pose_world_landmarks.landmark[30].y, results.pose_world_landmarks.landmark[30].z)

        leg = calf + thigh + foot
        print("Distance from Knee to Bottom of Feet = ", "{:.2f}".format((calf + foot) * 100), " cm")
        print("Distance from Hip to Bottom of Feet = ", "{:.2f}".format(leg * 100), ' cm')

        upper_arm = distance(results.pose_world_landmarks.landmark[13].x, results.pose_world_landmarks.landmark[13].y,
                             results.pose_world_landmarks.landmark[13].z, results.pose_world_landmarks.landmark[11].x,
                             results.pose_world_landmarks.landmark[11].y, results.pose_world_landmarks.landmark[11].z)

        print("Length from Shoulder to Elbow = ", "{:.2f}".format(upper_arm * 100), " cm")
        torso = distance(results.pose_world_landmarks.landmark[23].x, results.pose_world_landmarks.landmark[23].y,
                         results.pose_world_landmarks.landmark[23].z, results.pose_world_landmarks.landmark[11].x,
                         results.pose_world_landmarks.landmark[11].y, results.pose_world_landmarks.landmark[11].z)

        body = torso + leg
        print("Distance from Shoulder to Bottom of Feet = ", "{:.2f}".format(body * 100), ' cm')


# Function to find distance between two points in 3D space
def distance(x1, y1, z1, x2, y2, z2):
    return math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2) + math.pow(z2 - z1, 2) * 1.0)

def menu():
    while True:
        print('\n\nAssistive devices measurements calculator')
        print('Choose the device to take measurements for:')
        print('1. Spectacles')
        print('2. Walker')
        print('3. Standing Frame')
        print('4. Exit')
        op = int(input('Enter option: '))
        if op == 1:
            spectacles()
        elif op == 2:
            walker()
        elif op == 3:
            standingFrame()
        elif op == 4:
            print('Goodbye!')
            exit(0)
            break
        else:
            print('Invalid option, please try again')

menu()
