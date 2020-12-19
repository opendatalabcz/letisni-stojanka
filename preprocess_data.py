import tensorflow as tf
import json
import numpy as np
import IPython.display as display
import cv2 
import os 
import random
import math
import time 

import constants

FRAMES_PATH = './data'
FRAMES_CUT = 17500

DATA_PATH = "/home/oliver/School/SUMMER_2020/airport-apron-object-detection/data/hong_kong"

def extract_frames(video_path, frames_path):
    cam = cv2.VideoCapture(video_path)   
    os.makedirs("frames") 
    # frame 
    current_frame = 0
    
    while(True): 
        # reading from frame 
        ret,frame = cam.read() 
    
        if ret: 
            # if video is still left continue creating images 
            name = frames_path + '/frame_' + str(current_frame).zfill(6) + '.jpg'
            #print ('Creating...' + name) 
    
            # writing the extracted images 
            cv2.imwrite(name, frame) 
    
            # increasing counter so that it will 
            # show how many frames are created 
            current_frame += 1
        else: 
            break
    
    # Release all space and windows once done 
    cam.release() 
    cv2.destroyAllWindows() 

def get_img_size(img_path):
    # img_path = './sample_video/data/frame_000125.jpg'
    im = cv2.imread(img_path)
    h, w, c = im.shape
    return h, w

RAW_ANNOTATION_PATH = "/home/oliver/School/SUMMER_2020/airport-apron-object-detection/data/hong_kong/raw_annotations"
ANNOTATIONS_PATH = "/home/oliver/School/SUMMER_2020/airport-apron-object-detection/data/hong_kong/annotations"
IMG_PATH = "/home/oliver/School/SUMMER_2020/airport-apron-object-detection/data/hong_kong/data"

def extract_annotations(json_path, annotation_path):
    # opening JSON file 
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
        
        img_annotation_fname = annotation_path + "/" + img_ids[d["image_id"]][:-3] + "txt"
        
        if os.path.exists(img_annotation_fname):
            append_write = 'a' # append if already exists
        else:
            append_write = 'w' # make a new file if not
            
        img_annotation_file = open(img_annotation_fname, append_write)
        img_annotation_file.write(str(d["category_id"]-1) + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + "\n") 
        img_annotation_file.close() 
        
        print([x,y,w,h])
        print(d["category_id"])
        print(img_ids[d["image_id"]])
        

def train_test_split(img_path, frame_rate):
    #get all annotation fnames
    fnames = []
    for fname in os.listdir(img_path):
        fnames.append("/home/blaskoli/data/" + fname[:-3] + "jpg")
    
    #shuffle them
    random.shuffle(fnames)

    #75% train split
    train_threshold = math.floor(FRAMES_CUT*(3/4))

    train_file = open("./train.txt", "w")
    test_file = open("./test.txt", "w")
    
    for i in range(0, train_threshold, frame_rate):
        train_file.write(fnames[i] + "\n")

    for i in range(train_threshold, FRAMES_CUT, frame_rate):
        test_file.write(fnames[i] + "\n")
    print(fnames)


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


#extract_annotations(constants.JSON_PATH, constants.ANNOTATION_PATH)
test_annotaion(constants.DATA_PATH)

#train_test_split("/home/blaskoli/data", 10)
#test_annotaion("/home/oliver/School/SUMMER_2020/airport-apron-object-detection/data/hong_kong/raw_annotations")
#zip_data()    


