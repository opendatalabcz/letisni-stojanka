import cv2
import numpy as np
from PIL import Image
import math




import constants

CONFIG_PATH = "model/yolo-obj.cfg"
WEIGHTS_PATH = "model/yolo-obj_best.weights"
LABELS_PATH = "model/obj.names"
TEST_FRAMES = "/home/oliver/School/THESIS/data/hong_kong/test_frames"

def get_label(class_id):
    return constants.CLASS_NAMES[str(class_id)]


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
    def __init__(self, img_path):
        self.img_path = img_path
        self.img = np.asarray(Image.open(img_path))
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


    def show(self):
        for d in self.detections:
            color = (0, 0, 0)
            first, sec = d.bbox.get_cv2_format()
            cv2.rectangle(self.img, first, sec, color, 2)
            text = "{}".format(d.label)
            cv2.putText(self.img, text, (d.bbox.x + d.bbox.w, d.bbox.y + d.bbox.h),                  
            cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)

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
    def inference_img(self, img_path):
        In = Inference(img_path)

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
        print(In.get_text_format())
        #In.show()


class Evaluator:
    def __init__(self, model):
        self.model = model    


if __name__ == "__main__":
    m = Model(CONFIG_PATH, WEIGHTS_PATH, LABELS_PATH)
    ev = Evaluator(m)

    m.inference_img(TEST_FRAMES + "/" + "frame_029256.jpg")
    #print(get_label(10))
    #print(ev.model.labels)