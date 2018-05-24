import json
import os
import cv2
import numpy as np
import random
import pprint
import sys
from os.path import dirname,realpath
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

dir_path = dirname(realpath(__file__))

project_path = realpath(dir_path + "/..")

libs_dir_path = project_path + "/openpose"
sys.path.append(libs_dir_path)
print(libs_dir_path)
from openpose import poseEstimation

def get_frames(video_path, frames_per_step, segment, im_size, sess = None):

    #load video and qcquire its parameters using opencv

    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    video.set(cv2.CAP_PROP_POS_AVI_RATIO,1)
    max_len = video.get(cv2.CAP_PROP_POS_MSEC)

    # check segment consistency

    if max_len < segment[1]:

        segment[1] = max_len

    #define start frame

    central_frame = (np.linspace(segment[0], segment[1], num=3)) / 1000 *fps

    start_frame = central_frame[1] - frames_per_step / 2

    # for every frame in the clip extract frame, compute pose and insert the
    # result in the array

    frames = np.zeros(shape=(frames_per_step, im_size, im_size, 3), dtype=float)


    for z in range(frames_per_step):

        frame = start_frame + z
        video.set(1, frame)

        ret, im = video.read()

        pose_frame = poseEstimation.compute_pose_frame(im, sess)

        res = cv2.resize(pose_frame, dsize = (im_size,im_size),
                        interpolation=cv2.INTER_CUBIC)
        #res = cv2.resize(im, dsize=(im_size,im_size),
        #                  interpolation=cv2.INTER_CUBIC)
        frames[z,:,:,:] = res
    
    #TEST
    #for x in frames:
    #    cv2.imshow('image',x)
    #    cv2.waitKey(0)
    #TOP
    return frames

def read_clip_label(Batch_size,frames_per_step,entry_to_path, label_to_id, json_dict, im_size,sess):

    batch = np.zeros(shape=(Batch_size,frames_per_step,im_size,im_size,3),dtype=float)
    labels =  np.zeros(shape=(Batch_size),dtype=int)

    for s in range(Batch_size):
        entry_name = random.choice(list(json_dict.keys()))
        print(entry_name)
        training_entry = random.choice(json_dict[entry_name])

        path = entry_to_path[entry_name]

        segment = training_entry['milliseconds']
        print(segment)
        clip = get_frames(path,frames_per_step,segment,im_size,sess)
        batch[s,:,:,:,:] = clip
        labels[s] = label_to_id[training_entry['label']]

    return batch,labels


def makeDict(path=project_path):

    video_path = path + '/C3D-tensorflow/Dataset_PatternRecognition/H3.6M'

    path_dict = {}

    video2code = {}

    label2code = {}

    for category in os.listdir(video_path):
        if 'Icon' not in category:
            for video in os.listdir(video_path + "/" + category):
                if 'Icon' not in video:

                    path_dict[video] = video_path + "/"  + category + "/" + video

                    video2code[video] = len(video2code)

    dataset = json.load(open('Dataset_PatternRecognition/json/dataset_training.json'))

    list_of_labels = []

    for v in dataset.values():
        for label_dict in v:

            list_of_labels.append(label_dict['label'])

    labels = list(set(list_of_labels))

    for label in labels : label2code[label] = len(label2code)

    return path_dict, video2code, label2code


path_dict, video2code, label2code =  makeDict()

with tf.Session() as sess:

    batch,labels = read_clip_label(1,16,path_dict, label2code, json.load(open('Dataset_PatternRecognition/json/dataset_training.json')), 368,sess)
