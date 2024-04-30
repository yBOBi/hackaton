import cv2
import random as r
import numpy as np

cap = cv2.VideoCapture("./data/c1.mp4")

subtractor = cv2.createBackgroundSubtractorMOG2(history=5, varThreshold=20, detectShadows=True)

optical_flow = cv2.optflow.createOptFlow_Farneback()

i = 0

p1 = [[]]*20
np.array(p1).reshape(-1, 1, 2).astype(np.float32)

status, err = [], []
while True:
    if (i == 0):
        _, frame = cap.read()
        prev_frame = frame
        i=1
    _, frame = cap.read()
    prev_height = prev_frame.shape[0]
    prev_width = prev_frame.shape[1]
    x_coords = np.random.uniform(low=0, high=prev_width, size=(20, 1))
    y_coords = np.random.uniform(low=0, high=prev_height, size=(20, 1))
    p0 = np.concatenate((x_coords, y_coords), axis=1).reshape(-1, 1, 2).astype(np.float32)
    
    #обработка предыдущего кадра
    calc = cv2.calcOpticalFlowPyrLK(prev_frame, frame, np.array(p0), np.array(p1), np.array(status), np.array(err))
    for i in range(20):
        if(calc[2][i][0] > 5):
            print(calc[2][i][0])

    difference = cv2.absdiff(prev_frame, frame)
    #mask = subtractor.apply(frame)

    #cv2.imshow("Frame", frame)
    #cv2.imshow("Mask", mask)
    cv2.imshow("difference", difference)
    prev_frame = frame

    key = cv2.waitKey(18)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
