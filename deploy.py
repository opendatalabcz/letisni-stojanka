import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
from PIL import Image
from os import listdir
import sys
from cv2 import VideoWriter, VideoWriter_fourcc

configPath = "data.cfg"
weightsPath = "./weights.weights"
labelsPath = "./data.names"
image_path = ""

def main():
    PATH = sys.argv[1]
    frames = os.listdir(PATH)
    

    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    LABELS = open(labelsPath).read().strip().split("\n")
    annotated_frames = []


    for frame in frames:
        image = np.asarray(Image.open(PATH + "/" + frame))

        (H, W) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
            swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)
        # Initializing for getting box coordinates, confidences, classid 
        boxes = []
        confidences = []
        classIDs = []
        threshold = 0.15

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > threshold:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")           
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))    
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        print(classIDs)           
        for i in classIDs:
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            if LABELS[classIDs[i]] == 'airplane':
                color = (0, 255, 0)
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
                text = "{}".format(LABELS[classIDs[i]])
                cv2.putText(image, text, (x + w, y + h),                     
                cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 1)
                
            if LABELS[classIDs[i]] == 'pushback_car':
                color = (0, 0, 255)
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
                text = "{}".format(LABELS[classIDs[i]])
                cv2.putText(image, text, (x + w, y + h),      
                    cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 1)

            if LABELS[classIDs[i]] == 'cargo_door':
                color = (0, 0, 255)
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
                text = "{}".format(LABELS[classIDs[i]])
                cv2.putText(image, text, (x + w, y + h),      
                    cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 1)

            if LABELS[classIDs[i]] == 'aeroplane':
                color = (0, 0, 255)
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
                text = "{}".format(LABELS[classIDs[i]])
                cv2.putText(image, text, (x + w, y + h),      
                    cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 1)

            if LABELS[classIDs[i]] == 'jet_bridge':
                color = (0, 0, 255)
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
                text = "{}".format(LABELS[classIDs[i]])
                cv2.putText(image, text, (x + w, y + h),      
                    cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 1)

            if LABELS[classIDs[i]] == 'tank_truck':
                color = (0, 0, 255)
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
                text = "{}".format(LABELS[classIDs[i]])
                cv2.putText(image, text, (x + w, y + h),      
                    cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 1)
            
            if LABELS[classIDs[i]] == 'cargo_truck':
                color = (0, 0, 255)
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
                text = "{}".format(LABELS[classIDs[i]])
                cv2.putText(image, text, (x + w, y + h),      
                    cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 1)

            if LABELS[classIDs[i]] == 'luggage_loading_truck':
                color = (0, 0, 255)
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
                text = "{}".format(LABELS[classIDs[i]])
                cv2.putText(image, text, (x + w, y + h),      
                    cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 1)

            if LABELS[classIDs[i]] == 'cargo_box':
                color = (0, 0, 255)
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
                text = "{}".format(LABELS[classIDs[i]])
                cv2.putText(image, text, (x + w, y + h),      
                    cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 1)

            if LABELS[classIDs[i]] == 'truck':
                color = (0, 0, 255)
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
                text = "{}".format(LABELS[classIDs[i]])
                cv2.putText(image, text, (x + w, y + h),      
                    cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 1)

            if LABELS[classIDs[i]] == 'passenger_bus':
                color = (0, 0, 255)
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
                text = "{}".format(LABELS[classIDs[i]])
                cv2.putText(image, text, (x + w, y + h),      
                    cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 1)

        annotated_frames.append(image)


    for af in annotated_frames:
        cv2.imshow("image", af)
        cv2.waitKey()


if __name__ == "__main__":
    main()
