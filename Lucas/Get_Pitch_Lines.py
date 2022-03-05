from PIL import Image
from PIL import ImageFilter
import sys
import random
import numpy as np
import cv2

HLS_Hue_Low_Thresh = 0
HLS_Hue_High_Thresh = 80

HLS_Light_Low_Thresh = 150
HLS_Light_High_Thresh = 255

HLS_Sat_Low_Thresh = 20
HLS_Sat_High_Thresh = 255


def Create_HLS_Img(img):

    imHLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    mask = cv2.inRange(
                        imHLS, 

                        np.array([HLS_Hue_Low_Thresh, 
                                HLS_Light_Low_Thresh, 
                                HLS_Sat_Low_Thresh]),

                        np.array([HLS_Hue_High_Thresh, 
                                HLS_Light_High_Thresh, 
                                HLS_Sat_High_Thresh])
                        )

    img = cv2.bitwise_and(img,img, mask=mask)
    img = Image.fromarray(img)

    return img



if __name__ == '__main__':

    im = cv2.imread(sys.argv[1])

    im = Create_HLS_Img(im)

    im.save(sys.argv[2])

