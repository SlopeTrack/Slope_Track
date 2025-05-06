import numpy as np
import os
from filterpy.kalman import KalmanFilter
from math import sqrt
import scipy.linalg
np.random.seed(0)

"""
Script for Kalman based IoU (Intersection over Union) based on SportsMOT on the validation set of the respective dataset. 
Using the Kalman filter from the SORT algorithm.
"""


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio.
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Converts the state vector (first 4 elements: [x, y, s, r])
    into a bounding box in [x1, y1, x2, y2] format.
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


def box_area(arr):
    # arr: np.array([[x1, y1, x2, y2]])
    width = arr[:, 2] - arr[:, 0]
    height = arr[:, 3] - arr[:, 1]
    return width * height


def _box_inter_union(arr1, arr2):
    # arr1 of [N, 4]
    # arr2 of [N, 4]
    area1 = box_area(arr1)
    area2 = box_area(arr2)

    # Intersection
    top_left = np.maximum(arr1[:, :2], arr2[:, :2])  # [[x, y]]
    bottom_right = np.minimum(arr1[:, 2:], arr2[:, 2:])  # [[x, y]]
    wh = bottom_right - top_left
    # clip: if boxes not overlap then make it zero
    intersection = wh[:, 0].clip(0) * wh[:, 1].clip(0)

    # Union
    union = area1 + area2 - intersection
    return intersection, union


def box_iou(arr1, arr2):
    # arr1[N, 4]
    # arr2[N, 4]
    # N = number of bounding boxes
    assert (arr1[:, 2:] > arr1[:, :2]).all()
    assert (arr2[:, 2:] > arr2[:, :2]).all()
    inter, union = _box_inter_union(arr1, arr2)
    iou = inter / union
    return iou


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if ((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)

def consecutive_iou_kf_adjacent(annos):
    """
    Updates Kalman tracker with ground truth every frame.
    Predicts and computes IoU based on previous frame ground truth and current frame ground truth.

    'annos' format: [frame, id, x, y, w, h]
    """
    max_frame = int(annos[:, 0].max())
    min_frame = int(annos[:, 0].min())
    total_iou = 0
    total_frequency = 0

    trackers = {}  # Object ID -> KalmanBoxTracker

    for f in range(min_frame, max_frame):
        annos_curr = annos[annos[:, 0] == f]
        annos_next = annos[annos[:, 0] == f + 1]

        for row in annos_curr:
            frame_id, obj_id, x, y, w, h = row[:6]
            bbox = np.array([x, y, x + w, y + h])

            if obj_id not in trackers:
                trackers[obj_id] = KalmanBoxTracker(bbox)
                continue

            if f%1==0: #adjust prediction interval here
                pred_bbox = trackers[obj_id].predict()[0]
                pred_bbox = np.array([[pred_bbox[0], pred_bbox[1], pred_bbox[2], pred_bbox[3]]])
                gt_bbox = np.array([[x, y, x + w, y + h]])
                iou = box_iou(gt_bbox, pred_bbox).item()
                total_iou += iou
                total_frequency += 1
                trackers[obj_id].update(bbox)
            else:
                trackers[obj_id].update(bbox)

    return total_iou, total_frequency


if __name__ == "__main__":
    source_dir='Slope_Track/val'

    seqs = os.listdir(source_dir)
    # MOT
    # seqs = [f for f in os.listdir(source_dir) if 'FRCNN' in f]
    all_iou, all_freq = 0, 0
    all_switch, all_sw_freq = 0, 0
    for seq in seqs:
        if seq == ".DS_Store":
            continue
        anno_file = os.path.join(source_dir, seq, "gt/gt.txt")
        # MOT
        # anno_file = os.path.join(source_dir, seq, "gt/gt_val_half.txt")
        annos = np.loadtxt(anno_file, delimiter=",")
        seq_iou, seq_freq =  consecutive_iou_kf_adjacent(annos)
        all_iou += seq_iou
        all_freq += seq_freq

    print("Average IoU on consecutive frames = {}".format(all_iou / all_freq))

#every 10 frames
#MOT17: 0.8298065263936155
#SportsMOT:0.362
#DanceTrack:0.659
#Slope-Track:0.5648

#every 7 frames
#MOT17: 0.8606389681637334
#SportsMOT:0.442
#DanceTrack:0.707
#Slope-Track:0.643

#every 4 frames
#MOT17:0.9003
#SportsMOT:0.567
#DanceTrack: 0.765
#Slope-Track: 0.7465

#every 2 frames
#MOT17: 0.9351205085053466
#SportsMOT:0.700
#DanceTrack: 0.818
#Slope-Track: 0.8357

#every 1 frame
#MOT17:0.9552314846654183
#SportsMOT: 0.789
#DanceTrack: 0.852
#Slope-Track: 0.8884

import matplotlib.pyplot as plt

# Data
intervals = [1, 2, 4, 7, 10]
sportsmot_iou = [0.789, 0.700, 0.567, 0.442, 0.362]
dancetrack_iou = [0.852, 0.818, 0.765, 0.707, 0.659]
slopetrack_iou = [0.8884, 0.8357, 0.7465, 0.643, 0.5648]
mot17_iou = [0.9552314846654183, 0.9351205085053466,0.9003, 0.8606389681637334, 0.8298065263936155]

# Create plot
plt.figure(figsize=(10, 8))
plt.plot(intervals, mot17_iou, marker='o', label='MOT17', color='firebrick')
plt.plot(intervals, dancetrack_iou, marker='s', label='DanceTrack', color='darkgoldenrod')
plt.plot(intervals, sportsmot_iou, marker='p', label='SportsMOT', color='teal')
plt.plot(intervals, slopetrack_iou, marker='^', label='Slope-Track', color='darkviolet')

# Customize
plt.gca().invert_xaxis()
#plt.title('IoU vs Prediction Interval')
plt.xlabel('Prediction Interval (frames)', fontsize=16)
plt.ylabel('Average IoU', fontsize=16)
plt.xticks(intervals)
plt.ylim(0, 1)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=16)
plt.tight_layout()

# Show plot
plt.show()