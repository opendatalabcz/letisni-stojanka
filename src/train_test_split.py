import os
import random
import math

import constants

def transfer_path(old_path, new_path):
    """Changes paths to files from local machine to computing cluster"""
    splits = old_path.split("/")
    fname = splits[-1]
    return new_path + "/" + fname

def enumerate_scenes(d_not, d_imp, fnames):
    not_scene = []
    imp_scene = []
    norm_scene = []

    for i in range(0, len(fnames)):
        done = False
        for start, stop in d_not.items():
            if int(start) <= i <= int(stop):
                not_scene.append(fnames[i])
                done = True
        if done != True:
            for start, stop in d_imp.items():
                if int(start) <= i <= int(stop):
                    imp_scene.append(fnames[i])
                    done = True
            if done != True:
                norm_scene.append(fnames[i])    
    
    return not_scene, imp_scene, norm_scene

def filter_train_frames(fnames):
    f_scenes = []
    _, imp_scene, norm_scene = enumerate_scenes(constants.NOT_SCENES, constants.IMPORTANT_SCENES, fnames)
    for i in range(0, len(norm_scene), constants.NORMAL_FR):
        f_scenes.append(norm_scene[i])
    for i in range(0, len(imp_scene), constants.IMP_FR):
        f_scenes.append(imp_scene[i])
    return sorted(f_scenes)

def save_to_file(file_path, fnames, path):
    file = open(file_path, "w")
    for i in range(0, len(fnames)):
        file.write(path + "/" + fnames[i] + "\n")
    file.close()

def filter_test_valid(fnames):
    f_fnames = []
    for i in range(0,len(fnames),constants.TEST_FR):
        f_fnames.append(fnames[i])
    return f_fnames

def train_test_split(frames_path, train_frame_rate, valid_frame_rate, out_path):
    """Splits frames and annotations into three files: train, test and valid and handles filtering."""
    train_fnames = []
    valid_fnames = []
    test_fnames = []

    #get all frames fnames and sort them
    n_fnames = []
    fnames = os.listdir(frames_path)
    for f in fnames:
        if f.endswith(".jpg"):
            n_fnames.append(f)
    fnames = sorted(n_fnames)
    
    for i in range(len(fnames)):
        if constants.VALID_START <= i <= constants.VALID_STOP:
            valid_fnames.append(fnames[i])
        elif constants.TEST_START <= i <= constants.TEST_STOP:
            test_fnames.append(fnames[i])
        else:
            train_fnames.append(fnames[i])
    
    filtered_train_fnames = filter_train_frames(train_fnames)
    filtered_test_fnames = filter_test_valid(test_fnames)
    filtered_valid_fnames = filter_test_valid(valid_fnames)      
    
    save_to_file(constants.TRAIN_FILE, filtered_train_fnames, out_path)
    save_to_file(constants.TEST_FILE, filtered_test_fnames, out_path)
    save_to_file(constants.VALID_FILE, filtered_valid_fnames, out_path)
    
def test_train_test_file(frames_path, test_path):
    fnames = sorted(os.listdir(frames_path))
    not_scene, imp_scene, norm_scene = enumerate_scenes(constants.NOT_SCENES, constants.IMPORTANT_SCENES, fnames)

    train_file = open(test_path, "w")
    for line in train_test_split: 
        print(line)

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

train_test_split(constants.FRAMES_PATH, constants.NORMAL_FR, constants.IMP_FR, constants.OUT_PATH)
#add_augmented(FRAMES_PATH, OUT_PATH)