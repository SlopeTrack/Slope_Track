import numpy as np
import os
from filterpy.kalman import KalmanFilter
import configparser     # NEW
np.random.seed(0)

def convert_bbox_to_z(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x):
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    return np.array([x[0] - w/2., x[1] - h/2., x[0] + w/2., x[1] + h/2.]).reshape((1, 4))

def box_area(arr):
    return (arr[:, 2] - arr[:, 0]) * (arr[:, 3] - arr[:, 1])

def _box_inter_union(arr1, arr2):
    inter_tl = np.maximum(arr1[:, :2], arr2[:, :2])
    inter_br = np.minimum(arr1[:, 2:], arr2[:, 2:])
    wh = (inter_br - inter_tl).clip(0)
    inter = wh[:, 0] * wh[:, 1]
    union = box_area(arr1) + box_area(arr2) - inter
    return inter, union

def box_iou(arr1, arr2):
    inter, union = _box_inter_union(arr1, arr2)
    return inter / union

class KalmanBoxTracker:

    count = 0
    def __init__(self, bbox, fps):
        dt = 1.0 / fps                       # <-- time step
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # State transition with dt
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0,  0],
            [0, 1, 0, 0, 0,  1, 0],
            [0, 0, 1, 0, 0,  0,  1],
            [0, 0, 0, 1, 0,  0,  0],
            [0, 0, 0, 0, 1,  0,  0],
            [0, 0, 0, 0, 0,  1,  0],
            [0, 0, 0, 0, 0,  0,  1]])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.
        self.kf.predict()
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

def consecutive_iou_kf_adjacent(annos, fps):

    max_frame = int(annos[:, 0].max())
    min_frame = int(annos[:, 0].min())
    total_iou, total_freq = 0.0, 0
    trackers = {}
    pred_interval = max(1, round(1 * 30 / fps))

    for f in range(min_frame, max_frame):
        curr = annos[annos[:, 0] == f]
        nxt  = annos[annos[:, 0] == f + 1]

        for row in curr:
            _, obj_id, x, y, w, h = row[:6]
            bbox = np.array([x, y, x + w, y + h])

            if obj_id not in trackers:
                trackers[obj_id] = KalmanBoxTracker(bbox, fps)
                continue
            if f % pred_interval == 0:  # adjust prediction interval here
                pred = trackers[obj_id].predict()[0]
                pred = np.array([[pred[0], pred[1], pred[2], pred[3]]])
                gt   = np.array([[x, y, x + w, y + h]])
                total_iou += box_iou(gt, pred).item()
                total_freq += 1
                trackers[obj_id].update(bbox)
            else:
                trackers[obj_id].update(bbox)

    return total_iou, total_freq

if __name__ == "__main__":
    #source_dir = "slope_track/val"
    source_dir = "MOT17_val"
    all_iou, all_freq = 0.0, 0

    for seq in os.listdir(source_dir):
        if seq.startswith(".") or seq.endswith("DPM") or seq.endswith("SDP"):
            continue

        # --- read fps from seqinfo.ini ---
        ini_path = os.path.join(source_dir, seq, "seqinfo.ini")
        config = configparser.ConfigParser()
        config.read(ini_path)
        fps = float(config["Sequence"]["frameRate"])  # key is case-insensitive
        print(f"{seq}: fps = {fps}")

        annos = np.loadtxt(os.path.join(source_dir, seq, "gt/gt.txt"), delimiter=",")
        seq_iou, seq_freq = consecutive_iou_kf_adjacent(annos, fps)
        all_iou += seq_iou
        all_freq += seq_freq

    print("Average IoU (frame-rate adjusted) = {:.4f}".format(all_iou / all_freq))



import matplotlib.pyplot as plt

# Data
intervals = [1, 2, 4, 7, 10]
sportsmot_iou   = [0.7895, 0.7001, 0.5185, 0.4121, 0.3223]
dancetrack_iou  = [0.8181, 0.7910, 0.7258, 0.6594, 0.5962]
slopetrack_iou  = [0.8870, 0.8350, 0.7481, 0.6485, 0.5723]
mot17_iou       = [0.9499, 0.9246, 0.8823, 0.8348, 0.8005]

# Create plot
plt.figure(figsize=(10, 8))
plt.plot(intervals, mot17_iou,    marker='o', label='MOT17',      color='firebrick')
plt.plot(intervals, dancetrack_iou, marker='s', label='DanceTrack', color='darkgoldenrod')
plt.plot(intervals, sportsmot_iou,  marker='p', label='SportsMoT',  color='teal')
plt.plot(intervals, slopetrack_iou, marker='^', label='SlopeTrack', color='darkviolet')

# Axis labels and larger tick fonts
plt.xlabel('Prediction Interval (frames)', fontsize=18)
plt.ylabel('Average IoU', fontsize=18)

# Larger tick labels on both axes
plt.xticks(intervals, fontsize=16)
plt.yticks(fontsize=16)

# X-axis limits (ascending from 1 to 10)
plt.xlim(0, 11)

# Grid and legend
plt.ylim(0, 1)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=16)

plt.tight_layout()
plt.show()
