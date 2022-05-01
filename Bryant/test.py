import cv2 as cv
import os
#import torch
#import torchvision.models as models

# Load a model imported from Tensorflow
# img = cv2.imread('../documentation/data/original/sample_1.png')
# print(img.shape)
# img = cv2.resize(img, (1024, 1024))

# model = 'C:/Users/bpcor/PycharmProjects/football-mapper/Bryant/frozen_inference_graph.pb'
# model = 'C:/Users/bpcor/PycharmProjects/faster_rcnn_resnet50_v1_1024x1024_coco17_tpu-8/saved_model/saved_model.pb'
# config_path = 'C:/Users/bpcor/PycharmProjects/football-mapper/Bryant/TensorFlow/workspace/training_model/annotations/labels_objects.pbtxt'
# config_path = 'C:/Users/bpcor/PycharmProjects/football-mapper/Bryant/frozen_inference_graph.pbtxt'

# model = models.resnet18()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# checkpoint = torch.load(model_path)
# print(checkpoint.keys())
# model.load_state_dict(checkpoint['model'])
# optimizer.load_state_dict(checkpoint['opt'])
pb_file = 'C:/Users/bpcor/PycharmProjects/football-mapper/Bryant/frozen_inference_graph.pb'
pbtxt_file = 'C:/Users/bpcor/PycharmProjects/football-mapper/Bryant/TensorFlow/workspace/training_model/annotations/labels_objects.pbtxt'
img_file = '../documentation/data/original/sample_1.png' # also tried jpg but did not work

# net = cv2.dnn.readNetFromTensorflow(model, config_path)
# net = cv2.dnn_DetectionModel(model, config_path)
# print(type(net))
# net.setInputSize(1024, 1024)
# net.setInputScale(1.0/127.5)
# net.setInputMean((127.5,  127.5, 127.5))
# net.setInputSwapRB(True)
# something = net.detect(img, confThreshold=0.5)
# print(something)
# print(net)
# classIds, confs, bbox = net.detect(img, confThreshold=0.5)

cvNet = cv.dnn.readNet(pb_file, pbtxt_file)
print(type(cvNet))
img = cv.imread(img_file)
rows = img.shape[0]
cols = img.shape[1]

#cv.resize(img, blob, Size(50, 50));
#blob.convertTo(blob, CV_32F, 1.0/255, -0.5);
#blob = blob.reshape(1, {1, 50, 50, 3});

blob = cv.dnn.blobFromImage(img, 1.0, size=(1024, 1024), swapRB=True, crop=False)  # swapRB=True, crop=False
#blob = blob.reshape((1, 1024, 1024, 3))
print(blob.shape)
cvNet.setInput(blob)

layer_names = cvNet.getLayerNames()
print(cvNet.getUnconnectedOutLayers(), layer_names)
output_layer = [layer_names[i[0]-1] for i in cvNet.getUnconnectedOutLayers()]
cvOut = cvNet.forward(output_layer)  # <---------- location of error

for detection in cvOut[0,0,:,:]:
    score = float(detection[2])
    if score > 0.3:
        left = detection[3] * cols
        top = detection[4] * rows
        right = detection[5] * cols
        bottom = detection[6] * rows
        cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)

cv.imshow('img', img)
cv.waitKey()