import os
from pathlib import Path
from typing import List

from lxml import etree
from tqdm import tqdm

import cv2

class Box:

    def __init__(self, top, bottom, left, right):

        self._top = top
        self._bottom = bottom
        self._left = left
        self._right = right
    
    @property
    def top(self):
        return self._top

    @property
    def bottom(self):
        return self._bottom

    @property
    def left(self):
        return self._left
    
    @property
    def right(self):
        return self._right

    @property
    def top_left(self):
        return (self.left, self.top)

    @property
    def bottom_right(self):
        return (self.right, self.bottom)

    def __repr__(self):
        return f'top: {self.top} | bottom: {self.bottom} | left: {self.left} | right: {self.right}'

def parse_annotation(file_path):

    with open(file_path, 'r') as file:
        tree = etree.fromstring(file.read())

    boxes = []
    for bndbox in tree.xpath('object/bndbox'):

        left = int(bndbox.xpath('xmin/text()')[0])
        right = int(bndbox.xpath('xmax/text()')[0])
        top = int(bndbox.xpath('ymin/text()')[0])
        bottom = int(bndbox.xpath('ymax/text()')[0])
        boxes.append(Box(top, bottom, left, right))

    return boxes

if __name__ == '__main__':

    base_path: Path = Path('D:/Users/Olev/data/football-mapper/TV_soccer')
    pitch_lines_path: Path = base_path / 'pitch_lines'
    cleaned_path = base_path / 'cleaned'
    annotation_path = base_path / 'annotations/0'

    for file in tqdm(os.listdir(pitch_lines_path)):

        file = file.replace('.jpg', '.xml')

        boxes: List[Box] = parse_annotation(annotation_path / file)

        file = file.replace('.xml', '.jpg')

        image = cv2.imread(str(pitch_lines_path / file))

        for box in boxes:
            #print(box.top_left, box.bottom_right)
            image = cv2.rectangle(image, box.top_left, box.bottom_right, (0,0,0), thickness=-1)
            image = cv2.circle(image, (int((box.right + box.left) / 2), box.bottom), 3, (255,0,0))

        cv2.imwrite(str(cleaned_path / file), image)

