import cv2
import numpy as np
import pandas as pd
import pickle

import os

import posenet
import tensorflow as tf

import threading
import time
from collections import deque

from keras.models import load_model

from game_controller import W,S,A,D,Shift
import game_controller

for i in range(3)[::-1]:
    print('Starting in',i+1)
    time.sleep(1)


sess = tf.Session()
model_cfg, model_outputs = posenet.load_model(50, sess)

model = load_model('_models/model.h5')
model.summary()

q = deque()

class Prefs:
    def set_running(self,running):
        self.running = running
    def set_action(self, action):
        self.action = action
    def get_running(self):
        return self.running
    def get_action(self):
        return self.action

pref = Prefs()
pref.set_running(True)
pref.set_action(None)

scale_factor = 0.7125

with open('objects/scaler.pickle','rb') as file:
    scaler = pickle.load(file)
    file.close()

with open('objects/label_encoder.pickle','rb') as file:
    encoder = pickle.load(file)
    file.close()

def run_detector():
    #cap = cv2.VideoCapture('http://56.69.41.222:8080/video')
    cap = cv2.VideoCapture('test_video.mp4')
    ret, frame = cap.read()
    while(ret == True):
        frame = cv2.resize(frame,(360,360))
        #frame = cv2.rotate(frame,cv2.ROTATE_90_CLOCKWISE)
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_image, display_image, output_scale = posenet.read_image(
                frame, scale_factor=scale_factor, output_stride=16)

        heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(model_outputs,feed_dict={'image:0': input_image})
        
        pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
            heatmaps_result.squeeze(axis=0),
            offsets_result.squeeze(axis=0),
            displacement_fwd_result.squeeze(axis=0),
            displacement_bwd_result.squeeze(axis=0),
            output_stride=16,
            max_pose_detections=10,
            min_pose_score=0.15)

        keypoint_coords *= output_scale

        # TODO this isn't particularly fast, use GL for drawing and display someday...
        '''overlay_image = posenet.draw_skel_and_kp(
            display_image, pose_scores, keypoint_scores, keypoint_coords,
            min_pose_score=0.15, min_part_score=0.1)'''

        if pose_scores[0] > 0.4:
            coordinates = keypoint_coords[0].reshape(-1)
            q.append(coordinates)
        
        '''cv2.imshow('posenet', overlay_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            pref.set_running(False)
            break'''
        ret, frame = cap.read()
    pref.set_running(False)

def run_predictor():
    while(True):
        if len(q) > 16 or len(q) == 16:
            coords = [q.popleft() for i in range(16)]
            model_input = np.array(coords)
            model_input = scaler.transform(coords)
            model_input = np.expand_dims(model_input, axis = 0)
            predict_action = model.predict_classes(model_input)
            accuracy = model.predict(model_input)
            accuracy = accuracy.reshape(-1)
            accuracy = accuracy[np.argmax(accuracy)]
            print(predict_action, encoder.inverse_transform(predict_action), accuracy)
            pref.set_action(predict_action[0])
        if pref.get_running() == False:
            break
        time.sleep(0.3)

def send_keys():
    while(pref.get_running() == True):
        if pref.get_action() == None:
            time.sleep(1)
        elif pref.get_action() == 1:
            game_controller.ReleaseKey(W)
            game_controller.ReleaseKey(Shift)
            time.sleep(1)
        elif pref.get_action() == 2:
            game_controller.PressKey(W)
            time.sleep(1)
        elif pref.get_action() == 0:
            game_controller.PressKey(W)
            game_controller.PressKey(Shift)
            time.sleep(1)
        print('Current Action',pref.get_action())
    game_controller.ReleaseKey(W)
    game_controller.ReleaseKey(Shift)

detector_thread = threading.Thread(target = run_detector, args = ())
predictor_thread = threading.Thread(target = run_predictor, args = ())
key_thread = threading.Thread(target = send_keys, args = ())
detector_thread.start()
key_thread.start()
predictor_thread.run()
