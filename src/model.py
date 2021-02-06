import cv2
import numpy as np
from PIL import Image
import math
import os
import random
import time
from threading import Thread
import constants
from queue import Queue
from bounding_box import bounding_box as bb

def get_label(class_id):
    """"Returns the name of the class corresponding to the class_id."""
    return constants.CLASS_NAMES[str(class_id)]

def transform_paths(old_paths_fname, new_path):
    """Helper function that transofrms paths of images if needed"""
    new_fnames = []

    file = open(old_paths_fname, 'r') 
    lines = file.readlines() 
    for fname in lines:
        splits = fname.split("/")
        name = splits[-1]
        new_fnames.append(new_path + "/" + name[:-1])
    return new_fnames

def yolo_to_cv(x, y, w, h, img_w, img_h):
    l = int((x - w / 2) * img_w)
    r = int((x + w / 2) * img_w)
    t = int((y - h / 2) * img_h)
    b = int((y + h / 2) * img_h)
    
    if l < 0:
        l = 0
    if r > img_w - 1:
        r = img_w - 1
    if t < 0:
        t = 0
    if b > img_h - 1:
        b = img_h - 1

    return l, r, t, b

class Bbox:
    """Represents bbox - rectangle that is drawn around detected objects"""
    def __init__(self, box):
        self.x = box[0]
        self.y = box[1]
        self.w = box[2]
        self.h = box[3]
    
    def unwrap(self):
        return [int(self.x), int(self.y), int(self.w), int(self.h)]

    def get_cv2_format(self):
        return yolo_to_cv(self.x, self.y, self.w, self.h, constants.IMG_W, constants.IMG_H)

    def get_text_format(self):
        return "[{},{},{},{}]".format(self.x, self.y, self.w, self.h)


class Detection:
    """Represents detected objects in the image"""
    def __init__(self, bbox, confidence, class_id):
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.label = get_label(self.class_id)

    def get_text_format(self):
         return "label - {} | confidence - {} | bbox - {}\n".format(self.label, str(round(self.confidence, 2)), self.bbox.get_text_format())

class Inference:
    """Represents inference of the model. Contains all detection inferenced by model."""
    def __init__(self, img, style):
        if style == "path":
            self.img_path = img
            self.img = np.asarray(Image.open(self.img_path))
        else:
            self.img_path = ""
            self.img = np.asarray(img)

        self.img_h = self.img.shape[0]
        self.img_w = self.img.shape[1]
        self.detections = []

    def add_detection(self, detection):
        self.detections.append(detection)

    def get_text_format(self):
        s = "Img - {} | # of detections - {}:\n".format(self.img_path, len(self.detections))
        for d in self.detections:
            s += d.get_text_format()
        return s


    def show(self, show, ia=None):
        """Draws inferenced objects into the image, if show == True, also displays it"""
        for d in self.detections:
            color = (34,139,34)
            start_x, start_y, w, h = d.bbox.unwrap()
            
            bb.add(self.img, start_x, start_y, start_x+w, start_y+h, d.label)
            #cv2.rectangle(self.img, (start_x, start_y), (start_x+w,start_y+h), color, 2)
            #text = "{}".format(d.label)
            #cv2.putText(self.img, text, (start_x, start_y -5),                  
            #cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 1)

        #showing also grounding truths for testing purposes
        if ia != None:
            for gt in ia.gts:
                color = (128, 0, 0)

                x = int(float(gt.bbox.x) * IMG_W)
                y = int(float(gt.bbox.y) * IMG_H)

                x2 = x + int(float(gt.bbox.w) * IMG_W)
                y2 = y + int(float(gt.bbox.h) * IMG_H)

                cv2.rectangle(self.img, (x,y), (x2,y2), color, 2)
        
        if show == True:
            cv2.imshow('image',self.img)
            cv2.waitKey(0)
        

    def perform_nms(self):
        """Performs non-maximum suppression on the image"""
        boxes = []
        confidences = []
        for d in self.detections:
            boxes.append(d.bbox.unwrap())
            confidences.append(float(d.confidence))
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.3)

        new_detections = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                new_detections.append(self.detections[i])
        
        self.detections = new_detections

class Model:
    """Represents trained model for detecting objects, handles image inference"""
    def __init__(self, config_path, weights_path, labels_path):
        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        self.labels = open(labels_path).read().strip().split("\n")

    #takes image path as a param and performs object inference on it
    #returns bbox 
    def inference_img(self, img_path, style):
        In = Inference(img_path, style)

        blob = cv2.dnn.blobFromImage(In.img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_outputs = self.net.forward(self.ln)

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    box = detection[0:4] * np.array([In.img_w, In.img_h, In.img_w, In.img_h])
                    center_x, center_y, w, h = box.astype("int")
                    
                    x = int(center_x - (w / 2))
                    y = int(center_y - (h / 2))

                    b = Bbox([x,y,int(w),int(h)])
                    In.add_detection(Detection(b, confidence, class_id))
        
        In.perform_nms() 
        return In
        #print(In.get_text_format())

class GroundingTruth:
    """Represents objects present of the image"""
    def __init__(self, class_id, bbox):
        self.class_id = class_id
        self.label = get_label(self.class_id)
        self.bbox = bbox

    def get_text_format(self):
        return "label - {} | bbox - {}\n".format(self.label, self.bbox.get_text_format())

class ImageAnnotation:
    """Represents annotation of the image"""
    def __init__(self, img_path):
        self.img_path = img_path
        self.gts = []

    def add_gt(self, gt):
        self.gts.append(gt)

class Evaluator:
    """Class for model evaluation"""
    def __init__(self, model, fnames):
        self.model = model
        self.data_fnames = fnames
        self.get_gt_fnames()
        #self.load_grounding_trurths()
    
    def get_gt_fnames(self):
        """Loads ground truths filenames"""
        self.gt_fnames = []
        for f in self.data_fnames:
            splits = f.split("/")
            new_fname = constants.ANNOTATION_PATH + "/" + splits[-1][:-3] + "txt"
            self.gt_fnames.append(new_fname)

    def load_grounding_trurths(self):
        """Loads ground truths from annotation files."""
        self.gt = {}
        for fname in self.gt_fnames:
            img_fname_splits = fname.split("/")
            img_fname = constants.FRAMES_PATH + "/" + img_fname_splits[-1][:-3] + "jpg"

            Ia = ImageAnnotation(img_fname)

            file = open(fname, 'r') 
            lines = file.readlines()
            
            for gt in lines:
                splits = gt.split()
                new_gt = GroundingTruth(splits[0], Bbox([splits[1], splits[2], splits[3], splits[4]]))
                Ia.add_gt(new_gt)
            
            self.gt[img_fname] = Ia

    def demo(self):
        """Tests inference of the model, compared to grounding truth"""
        fnames = random.sample(self.data_fnames, 10)
        for fname in fnames:
            In = self.model.inference_img(fname, "path")
            In.show(True, self.gt[fname])
        


    def compute_iou(self, bboxA, bboxB):
        """Computer IOU between two bboxes"""
        xA = max(bboxA.x, bboxB.x)
        yA = max(bboxA.y, bboxB.y)
        xB = min(bboxA.w, bboxB.w)
        yB = min(bboxA.h, bboxB.h)
        

        intersection = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        bboxA_area = (bboxA.w - bboxA.x + 1) * (bboxA.h - bboxA.y + 1)
        bboxB_area = (bboxB.w - bboxB.x + 1) * (bboxB.h - bboxB.y + 1)
        
        iou = intersection / float(bboxA_area + bboxB_area - intersection)

        return iou

    