from pathlib import Path
import os

import cv2
import numpy as np

def is_white_pixel(pixel, threshold=175):

    if pixel[0] > threshold and pixel[1] > threshold and pixel[2] > threshold:
        return True
    
    return False

def is_red_pixel(pixel):

    if pixel[0] < 0.75*pixel[2] and pixel[1] < 0.75*pixel[2] and pixel[2] > 90:
        return True

    return False

def is_dark_pixel(pixel):

    if pixel[0] < 100 and pixel[1] < 100 and pixel[2] < 100:
        return True

    return False

if __name__ == '__main__':
    image = cv2.imread('../Lucas/output_0.jpg')

    red_pixels = []
    background_pixels = []
    line_pixels = []
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            
            pixel = image[y,x,:]

            if is_red_pixel(pixel):
                red_pixels.append((x,y))
            elif is_white_pixel(pixel):
                pixel = np.array([255,0,0])
                background_pixels.append((x,y))
            else:
                pixel = np.array([0,0,0])
                line_pixels.append(pixel)

            image[y,x,:] = pixel

    cv2.imwrite('x.jpg', image)