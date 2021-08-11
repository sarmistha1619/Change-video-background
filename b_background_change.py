import cv2
import numpy as np
import skimage.exposure as ske

video = cv2.VideoCapture('for_face_detection.mp4')

while(video.isOpened()):
    ret, frame = video.read()

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        range1 = (90, 255, 255)
        range2 = (0, 0, 90)
        mask = cv2.inRange(hsv, range1,range2)
        mask = 255 - mask

        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        mask = cv2.GaussianBlur(mask, (0,0), sigmaX=3, sigmaY=3, borderType=cv2.BORDER_DEFAULT)

        mask = ske.rescale_intensity(mask, in_range=(150,255), out_range=(0,255))

        final = frame.copy()
        final[mask==0]=(255,255,255)
        cv2.imshow('marvin', final)
        if cv2.waitKey(100) &0xFF == ord('t'):
            break

video.release()
cv2.destroyAllWindows()





