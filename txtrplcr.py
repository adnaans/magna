from PIL import Image
import base64
import cv2
import numpy as np

def cutChar(img, spots):
    for corners in spots:
        for i in range(corners[0][0],corners[1][0]):
            for j in range(corners[0][1],corners[1][1]):
                img[i][j][0]=255
                img[i][j][1]=255
                img[i][j][2]=255
    
    return img