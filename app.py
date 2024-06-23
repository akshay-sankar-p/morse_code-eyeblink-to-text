import cv2
# import numpy as np
from scipy.spatial import distance
from collections import deque
from morse_converter import convertblinktoText
import dlib   # py -m pip install C:\Users\HP\Downloads\dlib-19.24.1-cp311-cp311-win_amd64.whl   
from imutils import face_utils
import imutils


class Detectmorse():

    def __init__(self):
        self.closedEye = 0
        self.openEye = 0
        self.str = ''
        self.finalString = []
        global L
        self.L = []

        self.final = ''
        self.pts = deque(maxlen=512)
        self.thresh = 0.25
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(r"C:\Users\HP\Desktop\story\morse_code-eyeblink-to-text\shape_predictor_68_face_landmarks.dat")

    # euclidean distance = sqrt((x2 - x1) ^ 2 + (y2 - y1) ^ 2)
    def eye_aspect_ratio(self, eye):  # [36:42] => [36, 37, 38, 39, 40, 41]
        A = distance.euclidean(eye[1], eye[5])  # 37, 41
        B = distance.euclidean(eye[2], eye[4])  # 38, 40
        C = distance.euclidean(eye[0], eye[3])  # 36, 39
        ear = (A + B) / (2.0 * C)
        return ear

    def calculate(self, image):

        image = imutils.resize(image, width=640)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        for face in faces:

            shape = self.predictor(gray, face)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[36:42]
            rightEye = shape[42: 48]
            # print(leftEye, rightEye)

            leftEAR = self.eye_aspect_ratio(leftEye)
            rightEAR = self.eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            # print(ear)

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            # print(leftEyeHull, rightEyeHull)

            cv2.drawContours(image, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(image, [rightEyeHull], -1, (0, 255, 0), 1)

            if ear < self.thresh:  
                # print("------------ closed eyes-------")
                self.closedEye += 1
                self.pts.appendleft(self.closedEye)
                self.openEye = 0
            else:
                self.openEye += 1
                self.closedEye = 0
                self.pts.appendleft(self.closedEye)
            for i in range(1, len(self.pts)):
                # print(self.openEye)
                if self.pts[i] > self.pts[i - 1]:
                    # print(self.pts[i - 1], self.pts[i])
                    # print(self.pts[i])

                    #  15 frame - 2.60 sec
                    #  30 frame - 3.25 sec
                    #  60 frame - 7.53 sec

                    # if self.pts[i] > 30 and self.pts[i] < 70:
                    if self.pts[i] > 15 and self.pts[i] < 30:
                        print("Eyes have been closed for 50 frames!")
                        self.L.append("-")
                        self.pts = deque(maxlen=512)
                        break
                    
                    # elif self.pts[i] > 15 and self.pts[i] < 30:
                    elif self.pts[i] > 7 and self.pts[i] <= 15:
                        print("Eyes have been closed for 20 frames!")
                        self.L.append(".")
                        self.pts = deque(maxlen=512)
                        break

                    elif self.pts[i] > 30 and self.pts[i] <= 45:
                        print("Eyes have been closed for 90 frames!")
                        self.L.pop()
                        self.pts = deque(maxlen=512)
                        break

                

        # if (self.L != []):
        #     print(self.L)

        if self.openEye > 35:

            self.str = convertblinktoText(''.join(self.L))

            if self.str is not None:
                print(self.str)
                self.finalString.append(self.str)
                self.final = ''.join(self.finalString)
            if self.str is not None:
                self.L = []
            self.L = []
        # # print(self.L)
        if self.openEye < 100:
            return self.final, image, False
        else:
            return self.final, image, True
        # return self.final, image
