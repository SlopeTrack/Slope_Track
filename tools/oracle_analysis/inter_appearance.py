"""
Script for calculating the mean inter frame cosine distance for objects along their trajectory.

"""

from __future__ import absolute_import, division, print_function
import os
import glob as gb
import numpy as np
import cv2
import argparse
from deepsort_tracker.reid_model import Extractor
from matplotlib import pyplot as plt

class AppearanceFeature(object):
    def __init__(self, model_path, use_cuda=True):
        self.extractor = Extractor(model_path, use_cuda=use_cuda)

    def update(self, output_results, img_path):
        # Load image from disk.
        img = cv2.imread(img_path)
        #img=img_path
        if img is None:
            raise ValueError("Failed to load image: {}".format(img_path))
        self.height, self.width = img.shape[:2]


        bboxes = output_results[:, :4]  # x1, y1, x2, y2
        bbox_tlwh = self._xyxy_to_tlwh_array(bboxes)
        features = self._get_features(bbox_tlwh, img)
        return features

    @staticmethod
    def _xyxy_to_tlwh_array(bbox_xyxy):
        bbox_tlwh = bbox_xyxy.copy()
        bbox_tlwh[:, 2] = bbox_xyxy[:, 2] - bbox_xyxy[:, 0]  # width
        bbox_tlwh[:, 3] = bbox_xyxy[:, 3] - bbox_xyxy[:, 1]  # height
        return bbox_tlwh

    def _tlwh_to_xyxy(self, bbox_tlwh):

        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        y1 = max(int(y), 0)
        x2 = min(int(x + w), self.width - 1)
        y2 = min(int(y + h), self.height - 1)
        return x1, y1, x2, y2

    def _get_features(self, bbox_tlwh, img):
        im_crops = []
        for box in bbox_tlwh:
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            im_crop = img[y1:y2, x1:x2]
            im_crops.append(im_crop)
        if im_crops:
            features = self.extractor(im_crops)
            features = np.asarray(features)
            features = features / np.linalg.norm(features, axis=1, keepdims=True)
        else:
            features = np.array([])
        return features

def compute_intra_object_distances(object_feats):

    avg_intra_dists = {}
    for obj_id, feats in object_feats.items():
        #print(feats)
        cosine_dist_mat = 1. - np.dot(feats, feats.T)
        cosine_dist = cosine_dist_mat.sum() / len(feats) / len(feats)
        avg_intra_dists[obj_id] = cosine_dist
    return avg_intra_dists

if __name__ == '__main__':

    dataset = 'MOT17'
    output_folder = 'oracle_analysis/mot_val_appearance'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # MOT
    #video_list = [f for f in os.listdir(source_dir) if 'FRCNN' in f]
    video_list =  sorted(os.listdir(dataset))

    overall_video_means = []
    for video_name in video_list:
        print("Processing video:", video_name)
        object_features = {}
        det_results = {}
        gt_path = os.path.join(dataset, video_name, 'gt/gt.txt')
        #For MOT series
        #gt_path = os.path.join(dataset, video_name,'gt/gt_val_half.txt')
        with open(gt_path, 'r') as f:
            for line in f.readlines():
                linelist = line.strip().split(',')
                frame_id = int(linelist[0])
                obj_id = int(float(linelist[1]))
                bbox = [float(linelist[2]),
                        float(linelist[3]),
                        float(linelist[2]) + float(linelist[4]),
                        float(linelist[3]) + float(linelist[5]),
                        obj_id]  # bbox with object id

                if int(linelist[7]) == 1:
                    if frame_id in det_results:
                        det_results[frame_id].append(bbox)
                    else:
                        det_results[frame_id] = [bbox]
        f.close()

        tracker = AppearanceFeature(model_path='ckpt.t7')
        for frame_id in sorted(det_results.keys()):
            dets = np.array(det_results[frame_id])
            image_path = os.path.join(dataset, video_name, 'img1', '{:0>6d}.jpg'.format(frame_id))
            #For DanceTrack
            #image_path = os.path.join(dataset, video_name, 'img1', '{:0>8d}.jpg'.format(frame_id))
            appearance_feats = tracker.update(dets, image_path)
            obj_ids = dets[:, 4].astype(int)
            for i, obj_id in enumerate(obj_ids):
                feat = appearance_feats[i]
                if obj_id in object_features:
                    object_features[obj_id].append(feat)
                else:
                    object_features[obj_id] = [feat]


        for obj_id in object_features:
            object_features[obj_id] = np.vstack(object_features[obj_id])
        #print(object_features)

        # Compute per-object average intra-object distances.
        intra_object_dists = compute_intra_object_distances(object_features)
        # Compute mean intra-object distance for this video.
        video_mean = np.mean(list(intra_object_dists.values()))
        overall_video_means.append(video_mean)
        print(overall_video_means)

import matplotlib.pyplot as plt

mot = [0.17504771679353226, 0.1638105045895442,  0.17814714094476242,  0.20786567534235048, 0.2754693958725668,
       0.1794663123884326, 0.26289716678399305]
dancetrack = [0.173, 0.138, 0.157, 0.119, 0.162, 0.162, 0.190, 0.175, 0.209, 0.164, 0.201, 0.153, 
0.187, 0.163, 0.192, 0.133, 0.181, 0.174, 0.164, 0.163, 0.155, 0.194, 0.191, 0.200, 0.137]
sportsmot_val = [0.1861, 0.180, 0.185, 0.201, 0.179, 0.192, 0.153, 0.156, 0.161, 0.160, 0.152, 0.165,
0.112, 0.145, 0.195, 0.194, 0.205, 0.212, 0.208, 0.207, 0.163, 0.154, 0.165, 0.173, 0.159, 0.196]
slopetrack=[0.209, 0.237, 0.196, 0.227]

mot_x = range(len(mot))
dancetrack_x = range(len(mot), len(mot) + len(dancetrack))
sportsmot_val_x = range(len(mot)+len(dancetrack), len(mot) + len(dancetrack) + len(sportsmot_val))
slopetrack_x = range(len(mot)+len(dancetrack)+len(sportsmot_val), len(mot)+ len(dancetrack)+ len(sportsmot_val) + len(slopetrack))



fig, ax = plt.subplots(figsize=(15, 5))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.bar(x=mot_x, height=mot, alpha=0.3, color='blue', label='MOT17')
plt.bar(x=dancetrack_x, height=dancetrack, alpha=0.3, color='red', label='DanceTrack')
plt.bar(x=sportsmot_val_x, height=sportsmot_val, alpha=0.3, color='purple', label='SportsMOT')
plt.bar(x=slopetrack_x, height=slopetrack, alpha=0.3, color='green', label='Slope-Track')

plt.axhline(np.mean(mot),xmin=0.05,xmax=(0.15), color='darkblue', linestyle='dashed', linewidth=2)
plt.axhline(np.mean(dancetrack),xmin=0.15,xmax=(0.51), color='red', linestyle='dashed', linewidth=2)
plt.axhline(np.mean(sportsmot_val),xmin=0.515,xmax=(0.896), color='purple', linestyle='dashed', linewidth=2)
plt.axhline(np.mean(slopetrack), color='darkgreen', xmin=0.899,xmax=0.955, linestyle='dashed', linewidth=2)


plt.legend(fontsize=16)
#plt.xticks([len(mot_x)])
plt.ylim((0.10, 0.37))
plt.ylabel("Appearance Similarity")
plt.xlabel("Video Sequences")
#plt.title("Cosine distance of re-ID feature", fontsize=16)
plt.savefig('intra_bar.png', bbox_inches='tight', dpi=100)
plt.close()