import cv2
import random as r
import numpy as np

cap = cv2.VideoCapture("./data/c3.mp4")

subtractor = cv2.createBackgroundSubtractorMOG2(history=1, varThreshold=200, detectShadows=True)

optical_flow = cv2.optflow.createOptFlow_Farneback()

n = 1000
i = 0
p1 = np.array(1)

lk_params = dict(winSize = (15, 15),
                 maxLevel = 2,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03))

def translateImg(img, offset_x, offset_y):
    trans_mat = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
    img = cv2.warpAffine(img, trans_mat, (img.shape[1], img.shape[0]))
    return img

while True:
    if (i == 0):
        _, frame = cap.read()
        prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_frame = cv2.GaussianBlur(prev_frame, (5,5), 0)
        #prev_frame = subtractor.apply(prev_frame)
        i=1
    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame, (5,5), 0)
    #frame = subtractor.apply(frame)

    prev_height = prev_frame.shape[0]
    prev_width = prev_frame.shape[1]
    x_coords = np.random.uniform(low=0, high=prev_width, size=(n, 1))
    y_coords = np.random.uniform(low=0, high=prev_height, size=(n, 1))
    p0 = np.concatenate((x_coords, y_coords), axis=1).reshape(-1, 1, 2).astype(np.float32)

    #https://www.youtube.com/watch?v=hfXMw2dQO4E
    p1, _st, _err = cv2.calcOpticalFlowPyrLK(prev_frame, frame, p0, None, **lk_params)
    p0r, _st, _err = cv2.calcOpticalFlowPyrLK(frame, prev_frame, p1, None, **lk_params)
    d = abs(p0-p0r).reshape(-1, 2).max(-1)
    good = d < 1

    arr0 = np.empty((n,2))
    i = 0
    for (x0, y0), (x1, y1), good_flag in zip(p1.reshape(-1, 2), p0.reshape(-1, 2), good):
        if not good_flag:
            continue
        arr0[i] = [x0-x1, y0-y1]
        i += 1

    arr = np.empty((i,2))
    for ii in range(i):
        arr[ii] = arr0[ii]
    
    sum_x, sum_y = 0, 0
    result = np.empty((1,2))
    for (x, y) in arr:
        sum_x += x
        sum_y += y
    result = [sum_x/(i), sum_y/(i)]

    sum_x, sum_y = 0, 0
    for (x, y) in arr:
        if(abs(x-result[0]) < 3):
            sum_x += x
        if(abs(y-result[1]) < 3):
            sum_y += y
    result = [sum_x/(i), sum_y/(i)]

    #if(abs(result[0]) < 0.05): result[0] = 0
    #if(abs(result[1]) < 0.05): result[1] = 0
    print(result)

    prev_frame = translateImg(prev_frame, result[0], result[1])

    if(result[0] == 0 and result[1] == 0): difference = subtractor.apply(frame)
    else: difference = cv2.absdiff(prev_frame, frame)

    #cv2.imshow("Frame", frame)
    #cv2.imshow("Mask", mask)
    cv2.imshow("difference", difference)
    prev_frame = frame
    
    key = cv2.waitKey(50)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
