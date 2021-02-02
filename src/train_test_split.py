import os
import random
import math

FRAMES_PATH = "/home/oliver/School/THESIS/data/dataset/frames"
OUT_PATH = "/home/blaskoli/dataset/frames"

TOTAL_FRAMES = 266422

VALID_START = 92421
VALID_STOP = 104421

TEST_START = 45000
TEST_STOP = 50421

TRAIN_FRAME_RATE = 30
VALID_FRAME_RATE = 2

def transfer_path(old_path, new_path):
    """Changes paths to files from local machine to computing cluster"""
    splits = old_path.split("/")
    fname = splits[-1]
    return new_path + "/" + fname

def train_test_split(frames_path, train_frame_rate, valid_frame_rate, out_path):
    """Splits frames and annotations into three files: train, test and valid and handles filtering."""
    train_fnames = []
    valid_fnames = []
    test_fnames = []

    #get all frames fnames and sort them
    fnames = sorted(os.listdir(frames_path))
    for i in range(len(fnames)):
        new_fname = transfer_path(fnames[i], out_path)

        if VALID_START <= i <= VALID_STOP:
            valid_fnames.append(new_fname)
        elif TEST_START <= i <= TEST_STOP:
            test_fnames.append(new_fname)
        else:
            train_fnames.append(new_fname)
    
    #filter every n-th train frame given by train_frame_rate parameter
    filtered_train_fnames = []
    for i in range(0, len(train_fnames), train_frame_rate):
        filtered_train_fnames.append(train_fnames[i])
 
    #filter every n-th valid frame given by valid_frame_rate parameter
    filtered_valid_fnames = []
    for i in range(0, len(valid_fnames), valid_frame_rate):
        filtered_valid_fnames.append(valid_fnames[i])

    train_file = open("./model/train.txt", "w")
    valid_file = open("./model/valid.txt", "w")
    test_file = open("./model/test.txt", "w")
    
    for i in range(0, len(filtered_train_fnames)):
        train_file.write(filtered_train_fnames[i] + "\n")

    for i in range(0, len(filtered_valid_fnames)):
        valid_file.write(filtered_valid_fnames[i] + "\n")
    
    for i in range(0, len(test_fnames)):
        test_file.write(test_fnames[i] + "\n")


def add_augmented(frames_path, out_path):
    """Adds augmented images to the training file"""
    aug_fnames = []
    for fname in os.listdir(frames_path):
        #augmented images have prefix a"
        if fname.startswith("a"):
            new_fname = transfer_path(fname, out_path)
            aug_fnames.append(new_fname)
    
    train_file = open("./model/train.txt", "a")
    for af in aug_fnames:
        train_file.write(af + "\n")

train_test_split(FRAMES_PATH, TRAIN_FRAME_RATE, VALID_FRAME_RATE, OUT_PATH)
#add_augmented(FRAMES_PATH, OUT_PATH)