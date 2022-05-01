from pathlib import Path
import os

import cv2
import numpy as np

def is_white_pixel(pixel, threshold=175):

    return pixel[0] > threshold and pixel[1] > threshold and pixel[2] > threshold

def is_red_pixel(pixel):

    return pixel[0] < 0.75*pixel[2] and pixel[1] < 0.75*pixel[2] and pixel[2] > 90

def is_blue_pixel(pixel, ratio=(1, 0.75, 0.75), threshold=255):

    return pixel[1] < ratio[1] * pixel[0] and pixel[2] < ratio[2] * pixel[0] and pixel[0] >= threshold

def is_dark_pixel(pixel):

    return pixel[0] < 100 and pixel[1] < 100 and pixel[2] < 100

def is_black_pixel(pixel, threshold=0):

    return pixel[0] <= threshold and pixel[1] <= threshold and pixel[2] <= threshold

if __name__ == '__main__':
    image = cv2.imread('above-with-dots.png')

    red_pixels = []
    background_pixels = []
    line_pixels = []
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            
            pixel = image[y,x,:]

            if is_red_pixel(pixel):
                red_pixels.append((x,y))
            elif is_white_pixel(pixel):
                pixel = np.array([0,0,0])
                background_pixels.append((x,y))
            else:
                pixel = np.array([255,0,0])
                line_pixels.append(pixel)

            image[y,x,:] = pixel

    print(red_pixels)
    cv2.imwrite('above-black-and-red.png', image)