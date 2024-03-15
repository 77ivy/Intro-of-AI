import os
import cv2
import matplotlib.pyplot as plt

def detect(dataPath, clf):
    """
    Please read detectData.txt to understand the format. Load the image and get
    the face images. Transfer the face images to 19 x 19 and grayscale images.
    Use clf.classify() function to detect faces. Show face detection results.
    If the result is True, draw the green box on the image. Otherwise, draw
    the red box on the image.
      Parameters:
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    # Begin your code (Part 4)
    # raise NotImplementedError("To be implemented")
    data_initial = list(open(dataPath, "r"))
    idx = 0
    while idx < len(data_initial):
        data_name, data_people = map(str, data_initial[idx].split())
        idx += 1
        data_info = []
        for i in range(int(data_people)):
            data_info.append(tuple(map(int, data_initial[idx].split())))
            idx += 1
        
        ##img_gray = cv2.imread(os.path.join("data/detect", data_name + ".jpg"), cv2.IMREAD_GRAYSCALE)
        img = cv2.imread("data/detect/" + data_name)
        img_gray = cv2.imread("data/detect/" + data_name, cv2.IMREAD_GRAYSCALE)

        green = (0, 255, 0)
        red = (0, 0, 255)

        for rd in data_info:
            
            start_point = (int(rd[0]), int(rd[1])) #(x,y)
            end_point = (int(rd[0])+int(rd[2]), int(rd[1])+int(rd[3]))

            img_crop = img_gray[int(rd[1]):int(rd[1])+int(rd[3]), int(rd[0]):int(rd[0])+int(rd[2])].copy()
            img_crop = cv2.resize(img_crop, (19, 19), interpolation=cv2.INTER_LINEAR)

            if clf.classify(img_crop):
              cv2.rectangle(img, start_point, end_point, green, thickness=3)
            else :
              cv2.rectangle(img, start_point, end_point, red, thickness=3)

        cv2.imwrite("detection_result/detection_test_"+data_name, img)
    # End your code (Part 4)