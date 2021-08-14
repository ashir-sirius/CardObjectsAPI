import cv2
import numpy as np
from flask import Flask, jsonify, request
from cv2 import cv2


def detectObjects(img_path):  # put application's code here
    net = cv2.dnn.readNet('yolov3_training_last.weights', 'yolov3_testing.cfg')

    classes = []
    with open("classes.txt", "r") as f:
        classes = f.read().splitlines()

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(100, 3))

    img = cv2.imread(img_path)
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(
        img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    outputs = {}

    if len(indexes) > 0:
        outputs['detections'] = {}
        outputs['detections']['classes'] = []
        for i in indexes.flatten():
            detection = {'class': str(classes[class_ids[i]]), 'confidence': confidences[i], 'X': boxes[i][0],
                         'Y': boxes[i][1], 'Width': boxes[i][2], 'Height': boxes[i][3]}
            outputs['detections']['classes'].append(detection)
    else:
        outputs['detections'] = 'No Object Detected'

    return outputs


# detectObjects()
