import os
import random
import math

FRAMES_PATH = "/home/oliver/School/THESIS/data/hong_kong/frames"
OUT_PATH = "/home/blaskoli/frames"
TOTAL_FRAMES = 50422
FRAME_RATE = 30
TRAIN_SPLIT = 0.75
VALID_SPLIT = 0.15

def train_test_split(frames_path, frame_rate):
    #get all frames fnames and sort them
    fnames = []
    for fname in sorted(os.listdir(frames_path)):
        fnames.append(OUT_PATH + "/" + fname)

    #filter every n-th frame given by frame_rate parameter
    filtered_fnames = []
    for i in range(0, TOTAL_FRAMES, frame_rate):
        filtered_fnames.append(fnames[i])
 
    #shuffle them
    random.shuffle(filtered_fnames)

    total_filtered_frames = len(filtered_fnames)

    #75% train split
    train_threshold = math.floor(total_filtered_frames*TRAIN_SPLIT)
    valid_threshold = math.floor(total_filtered_frames*VALID_SPLIT) + train_threshold

    train_file = open("./model/train.txt", "w")
    valid_file = open("./model/valid.txt", "w")
    test_file = open("./model/test.txt", "w")
    
    for i in range(0, train_threshold):
        train_file.write(filtered_fnames[i] + "\n")

    for i in range(train_threshold, valid_threshold):
        valid_file.write(filtered_fnames[i] + "\n")
    
    for i in range(valid_threshold, total_filtered_frames):
        test_file.write(filtered_fnames[i] + "\n")

train_test_split(FRAMES_PATH, FRAME_RATE)