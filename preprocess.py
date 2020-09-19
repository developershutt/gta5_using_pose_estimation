import cv2
import numpy as np
import pandas as pd

import os

import posenet
import tensorflow as tf


files_dir = 'data/'
files = ['VID20200918233341.mp4','VID20200918234145.mp4','VID20200918235044.mp4']
labels = ['STANDING','WALKING','RUNNING']

parts = {
    0: 'NOSE',
    1: 'LEFT_EYE',
    2: 'RIGHT_EYE',
    3: 'LEFT_EAR',
    4: 'RIGHT_EAR',
    5: 'LEFT_SHOULDER',
    6: 'RIGHT_SHOULDER',
    7: 'LEFT_ELBOW',
    8: 'RIGHT_ELBOW',
    9: 'LEFT_WRIST',
    10: 'RIGHT_WRIST',
    11: 'LEFT_HIP',
    12: 'RIGHT_HIP',
    13: 'LEFT_KNEE',
    14: 'RIGHT_KNEE',
    15: 'LEFT_ANKLE',
    16: 'RIGHT_ANKLE'
}
print("Building Dataframe")
columns = np.array([[parts[i]+'_X', parts[i] + '_Y'] for i in range(17)]).reshape(-1).tolist()
columns.append('LABEL')
coords = []

def save_dataframe():
    df = pd.DataFrame(coords, columns = columns)
    df.to_csv('coordinates_dataset.csv',index =False)

sess = tf.Session()
model_cfg, model_outputs = posenet.load_model(100, sess)

scale_factor = 0.7125

for i in range(len(files)):
    file = files[i]
    label = labels[i]
    print('Preprocessing',file)
    count = 0
    path = files_dir + file
    print(path)
    cap = cv2.VideoCapture(path)
    ret, frame = cap.read()
    while(ret == True):
        frame = cv2.resize(frame,(360,360))
        frame = cv2.rotate(frame,cv2.ROTATE_90_CLOCKWISE)
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
        overlay_image = posenet.draw_skel_and_kp(
            display_image, pose_scores, keypoint_scores, keypoint_coords,
            min_pose_score=0.15, min_part_score=0.1)

        # update coords
        co_ordinates = keypoint_coords[0].reshape(-1).tolist()
        co_ordinates.append(label)
        coords.append(co_ordinates)
        
        cv2.imshow('posenet', overlay_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        ret, frame = cap.read()

save_dataframe()

sess.close()
cv2.destroyAllWindows()
print('finished')

