import sys
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt



def get_isolated_pitch(img, bin_buffer):
    # Convert image into HSV channels
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Get Hue channel and flatten array
    hue_flat = hsv[:,:,0].flatten()
    # Get Histogram Bins
    bins = plt.hist(hue_flat, bins=255)

    # Find largest bin and keep idx
    top_bin = 0
    top_idx = 0
    for b in range(0, len(bins[0])):
        if bins[0][b] > top_bin: 
            top_bin = bins[0][b]
            top_idx = b

    # Set range floor and cieling within limits
    floor = 0
    if top_idx - bin_buffer < 0: floor = 0
    else: floor = top_idx - bin_buffer

    ceiling = 0
    if top_idx + bin_buffer > 255 - bin_buffer: ceiling = 255
    else: ceiling = top_idx + bin_buffer

    # Create a mask based on bin range
    mask = np.where(hsv[:,:,0] >= bins[1][floor], 
                    np.where(hsv[:,:,0] < bins[1][ceiling], 1, 0), 0)

    # Apply mask to image
    for r in range(0, mask.shape[0]):
        for c in range(0, mask.shape[1]):
            if mask[r,c] != 1: 
                img[r,c] = [0, 0, 0]

    return img



def get_isolated_lines(img, percentile):
    # Convert Image to Grayscale and flatten
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_flat = gray.flatten()
    # Get Histogram Bins
    bins = plt.hist(gray_flat, bins=255)

    # Start running total
    running = 0
    # Initialize stop idx
    stop_idx = 0
    # Initialize stop condition
    stop_threshold = gray_flat.shape[0] * percentile

    # Find threshold bin
    for i in range(0, len(bins[0])):
        running += bins[0][i]
        if running > stop_threshold:
            stop_idx = i
            break

    # Create mask based on bin threshold
    mask = np.where(gray[:,:] >= bins[1][stop_idx], 1, 0)

    # Apply mask
    for r in range(0, mask.shape[0]):
        for c in range(0, mask.shape[1]):
            if mask[r,c] != 1: gray[r,c] = 0
            else: gray[r,c] = 255

    return gray



def get_hough_lines(img):
    blank_im = np.zeros(shape=img.shape)
    lines = cv2.HoughLinesP(img, rho=1, theta=np.pi/180, threshold=100, minLineLength=20, maxLineGap=10)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(blank_im, (x1, y1), (x2, y2), 255, 1)

    return blank_im, lines


def get_line_intersections(lines):
    intersections = []

    for i in range(0, len(lines)-1):
        # Get current line equation
        a1, b1, a2, b2 = lines[i][0]
        m1, b1 = get_slope_intercept(a1, b1, a2, b2)

        for j in range(i+1, len(lines)):
            # Get comparison line equation
            c1, d1, c2, d2 = lines[j][0]
            m2, b2 = get_slope_intercept(c1, d1, c2, d2)

            # Solve for intersection
            x = (b1-b2) / (m2-m1)
            y = m1 * x + b1

            # Ensure point exists and 
            # lies within bounds of both lines 
            if x and y \
                and x >= ((a1 < a2) * a1) + ((a2 < a1) * a2) \
                and x <= ((a1 > a2) * a1) + ((a2 > a1) * a2) \
                and x >= ((c1 < c2) * c1) + ((c2 < c1) * c2) \
                and x <= ((c1 > c2) * c1) + ((c2 > c1) * c2) \
                and y >= ((b1 < b2) * b1) + ((b2 < b1) * b2) \
                and y <= ((b1 > b2) * b1) + ((b2 > b1) * b2) \
                and y >= ((d1 < d2) * d1) + ((d2 < d1) * d2) \
                and y <= ((d1 > d2) * d1) + ((d2 > d1) * d2) \
                :
                intersections += [(x, y)]

    return intersections


def get_slope_intercept(x1, y1, x2, y2):
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - (x1 * slope)
    return slope, intercept



def get_line_corners(img, intersections):
    blank_im = np.zeros(shape=img.shape)

    corners = []

    for coor in intersections:
        x = int(coor[0])
        y = int(coor[1])
        if np.all((blank_im[y-10:y+10, x-10:x+10] == 0)):
            blank_im[y-10:y+10, x-10:x+10] = 255
            corners += [(x,y)]

    return blank_im, corners


if __name__ == '__main__':

    im = cv2.imread(sys.argv[1])

    hue_buffer = 35
    gray_percentile = 0.99

    im = get_isolated_pitch(im, hue_buffer)

    cv2.imwrite('PF_Step1.png', im)

    im = get_isolated_lines(im, gray_percentile)

    cv2.imwrite('PF_Step2.png', im)
    
    im, hough_lines = get_hough_lines(im)

    cv2.imwrite('PF_Step3.png', im)
    
    intersections = get_line_intersections(hough_lines)

    im, corners = get_line_corners(im, intersections)

    cv2.imwrite('PF_Step4.png', im)
    
    print(corners)