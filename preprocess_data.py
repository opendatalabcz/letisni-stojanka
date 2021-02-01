import tensorflow as tf
import json
import numpy as np
import IPython.display as display
import cv2 
import os 
import random
import math
import time 
from matplotlib import pyplot as plt
from clodsa.augmentors.augmentorFactory import createAugmentor
from clodsa.transformers.transformerFactory import transformerGenerator
from clodsa.techniques.techniqueFactory import createTechnique
import xml.etree.ElementTree as ET 

import constants

FRAMES_PATH = './data'
FRAMES_CUT = 17500
TOTAL_FRAMES = 104422
DATA_PATH = "/home/oliver/School/SUMMER_2020/airport-apron-object-detection/data/hong_kong"

def extract_frames(video_path, frames_path):
    """Extracts frames(images) from the video source."""
    
    cam = cv2.VideoCapture(video_path)
    current_frame = TOTAL_FRAMES
    
    while(True):
        #reading from video 
        ret,frame = cam.read() 
    
        if ret: 
            #if video is still left continue creating images 
            name = frames_path + '/frame_' + str(current_frame).zfill(6) + '.jpg'
            print ('Creating...' + name) 
    
            #writing the extracted images 
            cv2.imwrite(name, frame) 
    
            current_frame += 1
            #stopping before frames that don't have annotations yet
            if current_frame == TOTAL_FRAMES + 162000:
                print("Total number of frames: {}".format(current_frame))
                break
        else: 
            break
    
    # Release all space and windows once done 
    cam.release() 
    cv2.destroyAllWindows() 

def get_img_size(img_path):
    """Returns height and width of image."""
    im = cv2.imread(img_path)
    h, w, c = im.shape
    return h, w

def map_right_cat(cat):
    CORRECT_ID = {"0": "10",\
                   "1": "9", \
                   "2": "8", \
                   "3": "7", \
                   "4": "6", \
                   "5": "5", \
                   "6": "4", \
                   "7": "3", \
                   "8": "2", \
                   "9": "1", \
                   "10": "0",}
    return CORRECT_ID[cat]

def extract_annotations(json_path, annotation_path):
    """Extracts annotations from a json file produced by CVAT tool into COCO format."""
    json_file = open(json_path)  
    
    # returns JSON object as a dictionary 
    anotation_data = json.load(json_file) 

    img_ids = {}

    # mapping image_id to corresponding filename
    for d in anotation_data['images']: 
        img_ids[d["id"]] = d["file_name"][:-3] + "jpg"
        #print(d["id"], d["file_name"]) 

    img_h, img_w = get_img_size(constants.EXAMPLE_IMG)
    
    #iterate over all bounding boxes
    for d in anotation_data['annotations']:
        x = d["bbox"][0]/img_w
        y = d["bbox"][1]/img_h
        w = d["bbox"][2]/img_w
        h = d["bbox"][3]/img_h
        
        new_id = img_ids[d["image_id"]][:-4].split("_")
        new_id[1] = int(new_id[1]) + TOTAL_FRAMES
        if new_id[1] == 197718:
            break
        new_id = new_id[0] + "_" + str(new_id[1]).zfill(6)

        img_annotation_fname = annotation_path + "/" + new_id + ".txt"
        
        if os.path.exists(img_annotation_fname):
            append_write = 'a' # append if already exists
        else:
            append_write = 'w' # make a new file if not
            
        img_annotation_file = open(img_annotation_fname, append_write)
        img_annotation_file.write( map_right_cat(str(d["category_id"]-1)) + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + "\n") 
        img_annotation_file.close() 
        
        print([x,y,w,h])
        print(d["category_id"])
        print(img_ids[d["image_id"]])

def test_annotaion(data_path):
    
    #create testing folder
    os.popen("mkdir ./test_dir")
    annotation_path = data_path + "/annotations"
    frames_path = data_path + "/frames"

    fnames = []
    for fname in os.listdir(annotation_path):
        fnames.append(fname)
    
    for i in range(10):

        fname = random.choice(fnames)
        old_fname = frames_path + "/" + fname[:-3] + "jpg" 
        os.popen('cp {} ./test_dir/'.format(old_fname))
        time.sleep(1)

        ann_fname =  annotation_path + "/" + fname
        f = open(ann_fname, "r")
        img = cv2.imread('./test_dir/' + fname[:-3] + "jpg")
        
        img_h, img_w, c = img.shape
        
        for line in f:
            splits = line.split()
            class_id = splits[0]
            bbox = [splits[1], splits[2], splits[3], splits[4]]
            print("Class id - {}, bbox[{},{},{},{}]".format(class_id, bbox[0], bbox[1], bbox[2], bbox[3]))

            x = int(float(bbox[0]) * img_w)
            y = int(float(bbox[1]) * img_h)

            x2 = x + int(float(bbox[2]) * img_w)
            y2 = y + int(float(bbox[3]) * img_h)

            img = cv2.rectangle(img, (x,y), (x2,y2), (0,0,255), 2)
            img = cv2.putText(img, constants.CLASS_NAMES[str(int(class_id))], (x+1,y+1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255)
        
        #cv2.imwrite(old_fname, img) 
        cv2.imshow('image',img)
        cv2.waitKey(0)

    #clean testing files
    os.popen("rm -rf ./test_dir")
        
def get_filenames(path):
    file = open(path, "r")
    cnt = 0
    fnames = ""
    for l in file:
        fnames += DATA_PATH + "/data/" + l[:-1] + " "
        cnt+=1
    print(cnt)
    return fnames

def zip_data():
    train_fnames = get_filenames(DATA_PATH + "/train.txt")
    test_fnames = get_filenames(DATA_PATH + "/test.txt")
    all_fnames = train_fnames + test_fnames
    print(all_fnames)
    command = os.popen('zip {}/my_data.zip {}'.format(DATA_PATH, all_fnames))

def test_mathcing_files(annotation_path, frames_path):
    annotations = []
    frames = []

    for fname in os.listdir(annotation_path):
        annotations.append(fname)

    for fname in os.listdir(frames_path):
        frames.append(fname)
    
    annotations.sort()
    frames.sort()

    for i in range(len(annotations)):
        if annotations[i][:-3] != frames[i][:-3]:
            print(annotations[i], frames[i])
            break
def get_frame_names(data_text_format_file):
    file = open(data_text_format_file, 'r') 
    lines = file.readlines()
    new_lines = []

    for i in range(len(lines)):
        splitted_l = lines[i].split("/")
        #extract only the frame name and remove the \n char at the end
        new_lines.append(splitted_l[-1][:-1])

    return new_lines

def copy_for_augment(train_data_text_file, raw_path, augmented_path, last_frame):
    frame_names = get_frame_names(train_data_text_file)
    
    for f in frame_names:
        train_frame_path = raw_path + "/" + f
        aug_frame_path = augmented_path + "/" + "frame_" + str(last_frame).zfill(6) + ".jpg"
        coppy_command = "cp {} {}".format(train_frame_path, aug_frame_path)
        os.popen(coppy_command)

        frame_label_path = constants.ANNOTATION_PATH + "/" + f[:-3] + "txt"
        new_label_path = augmented_path + "/" + "frame_" + str(last_frame).zfill(6) + ".txt"
        os.popen("cp {} {}".format(frame_label_path, new_label_path))

        last_frame += 1

def init_augment(augmented_path):
    PROBLEM = "detection"
    ANNOTATION_MODE = "yolo"
    INPUT_PATH = augmented_path + "/"
    GENERATION_MODE = "linear"
    OUTPUT_MODE = "yolo"
    OUTPUT_PATH= augmented_path + "/" + "yolo_augmentation"

    augmentor = createAugmentor(PROBLEM,ANNOTATION_MODE,OUTPUT_MODE,GENERATION_MODE,INPUT_PATH,{"outputPath":OUTPUT_PATH})
    transformer = transformerGenerator(PROBLEM)

    return augmentor, transformer

def aug_operation(operation, transformer, img, boxes):
    gen = transformer(operation)
    aug_img, aug_boxes = gen.transform(img,boxes)
    return aug_img, aug_boxes

def load_boxes(img_path, annotation_path):
    img = cv2.imread(img_path)
    (img_h, img_w) = img.shape[:2]
    lines = [line.rstrip('\n') for line in open(annotation_path)]
    boxes = []
    if lines != ['']:
        for line in lines:
            splits = line.split(" ")
            label = splits[0]
            x  = int(float(splits[1])*img_w) 
            y = int(float(splits[2])*img_h)
            h = int(float(splits[4])*img_h)
            w = int(float(splits[3])*img_w)
            boxes.append((label, (x, y, w, h)))
    return (img,boxes)

def get_frame_path(frame):
    return constants.FRAMES_PATH + "/" + frame

def get_annotation_path(frame):
    return constants.ANNOTATION_PATH + "/" + frame

def show_boxes(image, boxes):
    img = image.copy()
    img_h, img_w, c = img.shape
    color = (34,139,34)
    for box in boxes:
        if(len(box)==2):
            (label, (x, y, w, h))=box
        else:
            (label, (x, y, w, h),_)=box

        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 5)
        cv2.putText(img, constants.CLASS_NAMES[str(int(label))], (x, y -5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    cv2.imshow("img", img)
    cv2.waitKey(0)

def save_augmentation(img, boxes, augmented_path, total_frames):
    #save the augmented image
    #fname = augmented_path + "/" + "aframe_" + str(total_frames).zfill(6) + ".jpg"
    
    fname = "/home/oliver/School/THESIS/data/thesis_footage" + "/" + "aframe_" + str(total_frames).zfill(6) + ".jpg"
    cv2.imwrite(fname, img)
    return
    ann_fname = augmented_path + "/" + "aframe_" + str(total_frames).zfill(6) + ".txt"
    ann_file = open(ann_fname, "w")
    img_h, img_w, c = img.shape

    for b in boxes:
        if(len(b)==2):
            (label, (x, y, w, h)) = b
        else:
            (label, (x, y, w, h),_) = b
        
        x = float(x)/float(img_w)
        y = float(y)/float(img_h)
        w = float(w)/float(img_w)
        h = float(h)/float(img_h)

        ann_file.write( str(label) + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + "\n") 
    
    ann_file.close() 




def augment_data(train_data_text_file, raw_path, augmented_path, last_frame):
    #1. copy the images from training dataset
    #copy_for_augment(train_data_text_file, raw_path, augmented_path, last_frame)

    #2. initliaze augmentor
    augmentor, transformer = init_augment(augmented_path)

    vFlip = createTechnique("flip",{"flip":0})
    augmentor.addTransformer(transformer(vFlip))

    hFlip = createTechnique("flip",{"flip":1})
    augmentor.addTransformer(transformer(hFlip))

    hvFlip = createTechnique("flip",{"flip":-1})
    augmentor.addTransformer(transformer(hvFlip))

    rotate = createTechnique("rotate", {"angle" : 90})
    augmentor.addTransformer(transformer(rotate))
    
    avgBlur =  createTechnique("average_blurring", {"kernel" : 5})
    augmentor.addTransformer(transformer(avgBlur))

    hue = createTechnique("raise_hue", {"power" : 0.9})
    augmentor.addTransformer(transformer(hue))

    total_frames = TOTAL_FRAMES
    frames = []
    for fname in os.listdir(augmented_path):
        if fname.endswith(".jpg"):
            frames.append(fname)

    for fr in frames:   
        img, boxes = load_boxes(augmented_path + "/" + fr,augmented_path + "/" + fr[:-3] + "txt")
        
        ver_img, ver_boxes = aug_operation(vFlip, transformer, img, boxes)
        save_augmentation(ver_img, ver_boxes, augmented_path, total_frames)
        total_frames += 1

        hor_img, hor_boxes = aug_operation(hFlip, transformer, img, boxes)
        save_augmentation(hor_img, hor_boxes, augmented_path, total_frames)
        total_frames += 1
        
        blur_img, blur_boxes = aug_operation(avgBlur, transformer, img, boxes)
        save_augmentation(blur_img, blur_boxes, augmented_path, total_frames)
        total_frames += 1
        
        hv_img, hv_boxes = aug_operation(hvFlip, transformer, img, boxes)
        save_augmentation(hv_img, hv_boxes, augmented_path, total_frames)
        total_frames += 1
        
        hue_img, hue_boxes = aug_operation(hue, transformer, img, boxes)
        save_augmentation(hue_img, hue_boxes, augmented_path, total_frames)
        total_frames += 1




extract_frames("/home/blaskoli/chosen.mp4", \
               "/home/home/blaskoli/dataset/japan_batch_2/frames")


#extract_annotations("/home/oliver/School/THESIS/data/dataset/japan_2_batch/json/annotations/instances_default.json", \
#                    "/home/oliver/School/THESIS/data/dataset/japan_2_batch/dataset/annotations")

#test_annotaion("/home/oliver/School/THESIS/data/dataset/japan_2_batch/dataset")

#test_annotaion("/home/oliver/School/THESIS/data/japan_used")
#test_mathcing_files("/home/oliver/School/THESIS/data/dataset/annotations", \
                    #"/home/oliver/School/THESIS/data/dataset/frames")

#test_annotaion("/home/oliver/School/THESIS/letisni-stojanka/augmented_frames")
#test_mathcing_files("/home/oliver/School/THESIS/data/dataset/annotations", \
#                    "/home/oliver/School/THESIS/data/dataset/frames")

#extract_annotations(constants.JSON_PATH, constants.ANNOTATION_PATH)
#test_annotaion(constants.DATA_PATH)
#test_mathcing_files(constants.ANNOTATION_PATH, constants.FRAMES_PATH)

#augment_data("/home/oliver/School/THESIS/letisni-stojanka/model/train.txt", constants.FRAMES_PATH, \
#             "/home/oliver/School/THESIS/letisni-stojanka/augmented_frames", TOTAL_FRAMES)

#test_annotaion("/home/oliver/School/THESIS/letisni-stojanka/augmented_frames")

#test_mathcing_files("/home/oliver/School/THESIS/letisni-stojanka/augmented_frames/annotations", \
#                    "/home/oliver/School/THESIS/letisni-stojanka/augmented_frames/frames")
#augment_data("/home/oliver/School/THESIS/letisni-stojanka/model/train.txt", constants.FRAMES_PATH, \
#              "/home/oliver/School/THESIS/data/hong_kong/test_frames", 50423)

#show_boxes("/home/oliver/School/THESIS/data/hong_kong/frames/frame_000000.jpg", \
#           "/home/oliver/School/THESIS/data/hong_kong/annotations/frame_000000.txt")

#train_test_split("/home/blaskoli/data", 10)
#test_annotaion("/home/oliver/School/SUMMER_2020/airport-apron-object-detection/data/hong_kong/raw_annotations")
#zip_data()    


