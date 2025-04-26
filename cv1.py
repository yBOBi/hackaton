import cv2
import random as r
import numpy as np
from sklearn.cluster import DBSCAN

#print("Введите номер видео 1-3(ESC - закрыть видео): ", end="")
#video = input()
#if(video == "2"):
#    cap = cv2.VideoCapture("./data/c2.mp4")
#    kontr = 2.58
#elif(video == "3"):
#    cap = cv2.VideoCapture("./data/c3.mp4")
#    kontr = 2.4
#else:
#    cap = cv2.VideoCapture("./data/c1.mp4")
#    kontr = 2.75

print("Введите номер видео 1-3(ESC - закрыть видео): ", end="")
video = input()
if(video == "2"):
    cap = cv2.VideoCapture("./data/p2.mp4")
    kontr = 2.55
elif(video == "3"):
    cap = cv2.VideoCapture("./data/p3.mp4")
    kontr = 2.75
elif(video == "4"):
    cap = cv2.VideoCapture("./data/p4.mp4")
    kontr = 2.9
elif(video == "5"):
    cap = cv2.VideoCapture("./data/p5.mp4")
    kontr = 2.75
elif(video == "22"):
    cap = cv2.VideoCapture("./data/c2.mp4")
    #kontr = 2.58
    kontr = 2.88
elif(video == "33"):
    cap = cv2.VideoCapture("./data/c3.mp4")
    kontr = 2.4
elif(video == "11"):
    cap = cv2.VideoCapture("./data/c1.mp4")
    #kontr = 2.75
    kontr = 2.85
else:
    cap = cv2.VideoCapture("./data/c3.mp4")
    kontr = 2.4

fps = 60

subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=100, detectShadows=True)

optical_flow = cv2.optflow.createOptFlow_Farneback()

n = 100
jjj = 0
p1 = np.array(1)
last_min = 0
bb = 5

feature_params = dict(maxCorners = n,
                    qualityLevel = 0.5,
                    minDistance = 7,
                    blockSize = 7 )

lk_params = dict(winSize = (5, 5),
                maxLevel = 2,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03))

def translateImg(img, offset_x, offset_y):
    trans_mat = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
    img = cv2.warpAffine(img, trans_mat, (img.shape[1], img.shape[0]))
    return img

def adjust_contrast_brightness(img, contrast:float=1.0, brightness:int=0):
    brightness += int(round(255*(1-contrast)/2))
    return cv2.addWeighted(img, contrast, img, 0, brightness)

while True:
    if (jjj == 0):
        _, frame = cap.read()
        if _ is False:
            break
        prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_frame = cv2.GaussianBlur(prev_frame, (bb,bb), 0)
        #prev_frame = subtractor.apply(prev_frame)
        jjj=1
    _, frame = cap.read()
    if _ is False:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame, (bb,bb), 0)
    #frame = subtractor.apply(frame)
    frame_arrow = frame.copy()
    output = frame.copy()
    
    p0 = cv2.goodFeaturesToTrack(prev_frame, mask = None, **feature_params)

    p1, _st, _err = cv2.calcOpticalFlowPyrLK(prev_frame, frame, p0, None, **lk_params)
    p0r, _st, _err = cv2.calcOpticalFlowPyrLK(frame, prev_frame, p1, None, **lk_params)
    d = abs(p0-p0r).reshape(-1, 2).max(-1)
    good = d < 1
    #print(d)

    #print(good)

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
    if(i==0):
        i = 1
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

    #if(abs(result[0]) < 0.1 and last_min < 200000): result[0] = 0
    #if(abs(result[1]) < 0.1 and last_min < 200000): result[1] = 0
    print(result)

    prev_frame1 = translateImg(prev_frame, result[0], result[1])


    difference = cv2.absdiff(prev_frame1, frame)
    difference = adjust_contrast_brightness(difference[2:(difference.shape[0]-2), 10:(difference.shape[1]-10)],kontr,255)
    last_min = cv2.countNonZero(difference)
    for (x0, y0), (x1, y1) in zip(p1.reshape(-1, 2), p0.reshape(-1, 2)):
        pt1 = (int(x0), int(y0))
        pt2 = (int(x1), int(y1))
        cv2.arrowedLine(frame_arrow, pt1, pt2, (0,0,255), 2)

    diff_gray, image_edges = cv2.threshold(difference, 100, 255, cv2.THRESH_BINARY)

    canvas = np.zeros(difference.shape, np.uint8)
    canvas.fill(255)

    mask = np.zeros(difference.shape, np.uint8)
    mask.fill(255)

    new_background = np.zeros(difference.shape, np.uint8)
    new_background.fill(255)

    contours_draw, hierachy = cv2.findContours(image_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours_mask, hierachy = cv2.findContours(image_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


    for contour in range(len(contours_draw)):
        cv2.drawContours(canvas, contours_draw, contour, (0, 0, 0), 3)

    for contour in range(len(contours_mask)):
        if contour > 2:
            cv2.fillConvexPoly(mask, contours_mask[contour], (0, 0, 0))

        if contour > 2:
            cv2.fillConvexPoly(new_background, contours_mask[contour], (0, 255, 0))

    #print("--------------------")
    #print(contours_mask)
    #print("--------------------")

    if(len(contours_mask) != 0):
        # Найти bounding boxes для каждого контура
        bounding_boxes = [cv2.boundingRect(contour) for contour in contours_mask]

        # Подготовить данные для DBSCAN
        boxes_points = []
        for (x, y, w, h) in bounding_boxes:
            boxes_points.append([x, y])
            boxes_points.append([x + w, y + h])

        boxes_points = np.array(boxes_points, dtype=float)  # Приведение к типу float

        # Увеличение расстояния по оси X для кластеризации
        scale_x = 0.3  # Коэффициент увеличения по оси X
        boxes_points[:, 0] *= scale_x

        # Кластеризация с DBSCAN
        clustering = DBSCAN(eps=20, min_samples=1).fit(boxes_points)

        # Вернуть масштабирование координат по оси X обратно
        boxes_points[:, 0] /= scale_x
        boxes_points = boxes_points.astype(int)  # Приведение обратно к типу int

        # Создать массив объединенных прямоугольников
        merged_boxes = []
        for cluster in set(clustering.labels_):
            cluster_points = boxes_points[clustering.labels_ == cluster]
            x_min = np.min(cluster_points[:, 0])
            y_min = np.min(cluster_points[:, 1])
            x_max = np.max(cluster_points[:, 0])
            y_max = np.max(cluster_points[:, 1])
            merged_boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))


        # Нарисовать объединенные bounding boxes
        for (x, y, w, h) in merged_boxes:
            cv2.rectangle(output, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)
            if h > w:
                label = "human"
            else:
                label = "car"
            cv2.putText(output, label, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)
    prev_len = len(contours_mask)
    #mask = subtractor.apply(frame)

    cv2.imshow("Output", output)
    cv2.imshow("canvas", canvas)
    cv2.imshow('Background mask', mask)
    #cv2.imshow('New background', new_background)
    cv2.imshow("Frame_", frame_arrow)
    cv2.imshow("difference", difference)

    prev_frame = frame.copy()
    
    key = cv2.waitKey(fps)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
