import cv2
# import numpy as np
from scipy.spatial import distance
from collections import deque
from morse_converter import convertblinktoText
import dlib   # py -m pip install C:\Users\HP\Downloads\dlib-19.24.1-cp311-cp311-win_amd64.whl   
from imutils import face_utils
import imutils


class Detectmorse():

    # Constructor...
    def __init__(self):
        self.flag = 0
        self.openEye = 0
        self.str = ''
        self.finalString = []
        global L
        self.L = []
        # self.closed = False
        # self.timer = 0
        self.final = ''
        self.pts = deque(maxlen=512)
        self.thresh = 0.25
        # self.dot = 10
        # self.dash = 40
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(r"C:\Users\HP\Desktop\story\morse_code-eyeblink-to-text\shape_predictor_68_face_landmarks.dat")

    def eye_aspect_ratio(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def calculate(self, image):
        # decoded = cv2.imdecode(np.frombuffer(image, np.uint8), -1)
        image = imutils.resize(image, width=640)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        # print(faces)
        for face in faces:
            # x1, x2 = face.left(), face.right()
            # y1, y2 = face.top(), face.bottom()
            # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

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
                self.flag += 1
                self.pts.appendleft(self.flag)
                self.openEye = 0
            else:
                self.openEye += 1
                self.flag = 0
                self.pts.appendleft(self.flag)
            for i in range(1, len(self.pts)):
                if self.pts[i] > self.pts[i - 1]:
                    # print(self.pts[i - 1], self.pts[i])

                    #  15 frame - 2.60 sec
                    #  30 frame - 3.25 sec
                    #  60 frame - 7.53 sec

                    if self.pts[i] > 30 and self.pts[i] < 70:
                    # if self.pts[i] > 15 and self.pts[i] < 30:
                        print("Eyes have been closed for 50 frames!")
                        self.L.append("-")
                        self.pts = deque(maxlen=512)
                        break
                    elif self.pts[i] > 15 and self.pts[i] < 30:
                    # elif self.pts[i] > 7 and self.pts[i] < 15:
                        print("Eyes have been closed for 20 frames!")
                        self.L.append(".")
                        self.pts = deque(maxlen=512)
                        break

                    elif self.pts[i] > 90:
                        print("Eyes have been closed for 90 frames!")
                        self.L.pop()
                        self.pts = deque(maxlen=512)
                        break

        # if (self.L != []):
        #     print(self.L)

        if self.openEye > 35:
            # if (self.L != []):
                # print(self.L)
            self.str = convertblinktoText(''.join(self.L))

            if self.str is not None:
                print(self.str)
                self.finalString.append(self.str)
                self.final = ''.join(self.finalString)
            if self.str is not None:
                self.L = []
            self.L = []
        # # print(self.L)
        # cv2.putText(image, "Predicted :  " + self.finalString, (10, 470),
        #             cv2.FONT_HERSHEY_DUPLEX, 0.7, (52, 152, 219), 2)
        return self.final, image
