# Python version 3.10
# Import necessary modules
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import cv2
import cvzone
from HandTrackingModule import HandDetector
import numpy as np
import math

# read the webcam
cap = cv2.VideoCapture(0)
# set the width
cap.set(3, 1280)
# set the height
cap.set(3, 780)

# create object to show fps
fpsReader = cvzone.FPS()
# Create a hand tracking object
detector = HandDetector(min_detection_confidence=0.7)

# Code of control volume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volume_range = volume.GetVolumeRange()
min_volume = volume_range[0]
max_volume = volume_range[1]
my_volume = 0
my_volume_bar = 400
my_volume_percentage = 0

while True:
    success, img = cap.read()
    if success:
        fps, img = fpsReader.update(img=img, pos=(50, 80), color=(255, 0, 0), scale=1.5, thickness=2)
        img = detector.find_hands(img=img, draw=True)
        landmarks_list = detector.find_positions(img=img, draw=False)
        if len(landmarks_list) != 0:
            # print(landmarks_list[0], landmarks_list[8])
            # here I just extract the x1,x2 and y1,y2 coordinates
            x1, y1 = landmarks_list[4][1], landmarks_list[4][2]
            x2, y2 = landmarks_list[8][1], landmarks_list[8][2]
            # find the coordinates for draw a circle in the middle of the line
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            # Now draw circle on this coordinates
            cv2.circle(img=img, center=(x1, y1), radius=8, color=(255, 0, 255), thickness=cv2.FILLED)
            cv2.circle(img=img, center=(x2, y2), radius=8, color=(255, 0, 255), thickness=cv2.FILLED)
            # draw a line between the points
            cv2.line(img=img, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 255), thickness=2)
            # draw a circle in the middle of the line
            cv2.circle(img=img, center=(cx, cy), radius=8, color=(255, 0, 255), thickness=cv2.FILLED)
            # Now find out the length of the line to find out minimum and maximum values
            # so that we can control volume
            line_length = math.hypot(x2 - x1, y2 - y1)
            # print(line_length)
            # HandRange - 20 to 200
            # Volume range - -65 to 0
            # We just convert the handRange into our desired volume range
            my_volume = np.interp(x=line_length, xp=[20, 200], fp=[min_volume, max_volume])
            my_volume_bar = np.interp(x=line_length, xp=[20, 200], fp=[400, 150])
            my_volume_percentage = np.interp(x=line_length, xp=[20, 200], fp=[0, 100])
            # print(my_volume)
            # set the master volume value
            volume.SetMasterVolumeLevel(my_volume, None)

            if line_length < 20:
                cv2.circle(img=img, center=(cx, cy), radius=8, color=(0, 255, 0), thickness=cv2.FILLED)

        cv2.rectangle(img=img, pt1=(50, 150), pt2=(85, 400), color=(255, 0, 0), thickness=2)
        cv2.rectangle(img=img, pt1=(50, int(my_volume_bar)), pt2=(85, 400), color=(255,), thickness=cv2.FILLED)
        cv2.putText(img=img, text=f'{int(my_volume_percentage)} %', org=(40, 450), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=1, color=(255, 0, 0), thickness=2)

        cv2.imshow("image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        exit()

cap.release()
cv2.destroyAllWindows()
