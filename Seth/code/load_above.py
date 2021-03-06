from functools import partial
from pathlib import Path
import os
import json
import sys

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

def main(input_image, output_image=None):

    image = cv2.imread(input_image)

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
                line_pixels.append((x,y))

            image[y,x,:] = pixel

    if output_image is not None:
        cv2.imwrite(output_image, image)
        output_image = Path(output_image)
        with open(output_image.parent / output_image.name.replace(output_image.suffix, '.json'), 'w+') as file:
            file.write(json.dumps(red_pixels, indent=4))

    return (image, red_pixels, line_pixels)

if __name__ == '__main__':

    input_image = sys.argv[1]
    if len(sys.argv) > 2:
        output_image = sys.argv[2]
        _, _, _ = main(input_image, output_image)
    else:
        _, _, _ = main(input_image)
    # input_image = above-with-dots.png
    # output_image = above-black-and-red.png