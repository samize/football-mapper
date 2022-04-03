import cv2


def main():
    img = cv2.imread('../documentation/data/original/sample_1.png')

    # Add model and detection points for bounding box

    # Bounding box code (topleft, bottomright)
    pt1, pt2 = (50, 50), (100, 100)
    cv2.rectangle(img, pt1, pt2, color=(0, 255, 0), thickness=2)
    #cv2.circle(img, pt1, radius=5, color=(255, 0, 0), thickness=-1)
    # Show detection
    cv2.imshow("Test", img)
    cv2.waitKey(0)


if __name__ == '__main__':

    main()
