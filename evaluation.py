import cv2
import numpy as np
from PIL import Image
import math
import os
import random
import time

import constants

IMG_W = 1278
IMG_H = 720

CONFIG_PATH = "model/yolo-obj.cfg"
WEIGHTS_PATH = "model/yolo-obj_last.weights"
LABELS_PATH = "model/obj.names"
TEST_FRAMES = "/home/oliver/School/THESIS/data/hong_kong/test_frames"
TRAINING_DATA_PATH = "model/train.txt"
TEST_DATA_PATH = "model/test.txt"

#TEST_VIDEO_PATH = "/home/oliver/School/THESIS/data/test_videos/hong_kong_train.mp4"
TEST_VIDEO_PATH = "/home/oliver/School/THESIS/data/test_videos/japan_test_3.mp4"
#TEST_VIDEO_PATH = "/home/oliver/School/THESIS/data/test_videos/hong_kong_train.mp4"
#TEST_VIDEO_PATH = "/home/oliver/School/THESIS/data/test_videos/japan_test_3.mp4"
OUTPUT_TEST_VIDEO = "/home/oliver/School/THESIS/data/test_videos/japan_test3_output.mp4"

def get_label(class_id):
    return constants.CLASS_NAMES[str(class_id)]

def transform_paths(old_paths_fname, new_path):
    new_fnames = []

    file = open(old_paths_fname, 'r') 
    lines = file.readlines() 
    for fname in lines:
        splits = fname.split("/")
        name = splits[-1]
        new_fnames.append(new_path + "/" + name[:-1])
    return new_fnames


class Bbox:
    def __init__(self, box):
        self.x = box[0]
        self.y = box[1]
        self.w = box[2]
        self.h = box[3]
    
    def unwrap(self):
        return [int(self.x), int(self.y), int(self.w), int(self.h)]

    def get_cv2_format(self):
        return (self.x, self.y), (self.x + self.w, self.y + self.h)

    def get_text_format(self):
        return "[{},{},{},{}]".format(self.x, self.y, self.w, self.h)


class Detection:
    def __init__(self, bbox, confidence, class_id):
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.label = get_label(self.class_id)

    def get_text_format(self):
         return "label - {} | confidence - {} | bbox - {}\n".format(self.label, str(round(self.confidence, 2)), self.bbox.get_text_format())

class Inference:
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
        for d in self.detections:
            color = (34,139,34)
            first, sec = d.bbox.get_cv2_format()
            cv2.rectangle(self.img, first, sec, color, 2)
            text = "{}".format(d.label)
            cv2.putText(self.img, text, (d.bbox.x, d.bbox.y -5),                  
            cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)

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
        boxes = []
        confidences = []
        for d in self.detections:
            boxes.append(d.bbox.unwrap())
            confidences.append(float(d.confidence))
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

        new_detections = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                new_detections.append(self.detections[i])
        
        self.detections = new_detections




class Model:
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
                    #x = int(centerX - (width / 2))
                    #y = int(centerY - (height / 2))   
                    b = Bbox(box.astype("int"))
                    In.add_detection(Detection(b, confidence, class_id))
        
        In.perform_nms() 
        return In
        #print(In.get_text_format())

class GroundingTruth:
    def __init__(self, class_id, bbox):
        self.class_id = class_id
        self.label = get_label(self.class_id)
        self.bbox = bbox

    def get_text_format(self):
        return "label - {} | bbox - {}\n".format(self.label, self.bbox.get_text_format())

class ImageAnnotation:
    def __init__(self, img_path):
        self.img_path = img_path
        self.gts = []

    def add_gt(self, gt):
        self.gts.append(gt)

class Evaluator:
    def __init__(self, model, fnames):
        self.model = model
        self.data_fnames = fnames
        self.get_gt_fnames()
        self.load_grounding_trurths()
    
    def get_gt_fnames(self):
        self.gt_fnames = []
        for f in self.data_fnames:
            splits = f.split("/")
            new_fname = constants.ANNOTATION_PATH + "/" + splits[-1][:-3] + "txt"
            self.gt_fnames.append(new_fname)

    def load_grounding_trurths(self):
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
        fnames = random.sample(self.data_fnames, 10)
        for fname in fnames:
            In = self.model.inference_img(fname, "path")
            In.show(True, self.gt[fname])


    def video_demo(self):
        v_stream = cv2.VideoCapture(TEST_VIDEO_PATH)
        writer = None

        c = 0
        while True:
            if c == 1000:
                break
            (grabbed, frame) = v_stream.read()

            if not grabbed:
                break

            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(OUTPUT_TEST_VIDEO, fourcc, 30, (frame.shape[1], frame.shape[0]), True)
            
            #start = time.time()
            In = self.model.inference_img(frame, "frame")
            #end = time.time()
            In.show(False)
            writer.write(In.img)
            #elap = (end - start)
            #print("{:.4f}".format(elap))
            c+=1
        writer.release()
        v_stream.release()


    def compute_iou(self, bboxA, bboxB):
        xA = max(bboxA.x, bboxB.x)
        yA = max(bboxA.y, bboxB.y)
        xB = min(bboxA.w, bboxB.w)
        yB = min(bboxA.h, bboxB.h)
        

        intersection = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        bboxA_area = (bboxA.w - bboxA.x + 1) * (bboxA.h - bboxA.y + 1)
        bboxB_area = (bboxB.w - bboxB.x + 1) * (bboxB.h - bboxB.y + 1)
        
        iou = intersection / float(bboxA_area + bboxB_area - intersection)

        return iou


if __name__ == "__main__":
    m = Model(CONFIG_PATH, WEIGHTS_PATH, LABELS_PATH)

    #new_fnames = transform_paths(TRAINING_DATA_PATH, constants.FRAMES_PATH)
    new_fnames = transform_paths(TEST_DATA_PATH, constants.FRAMES_PATH)
    ev = Evaluator(m, new_fnames)
    #ev.demo()
    ev.video_demo()
    #m.inference_img(TEST_FRAMES + "/" + "frame_029256.jpg")
    