"""
    Script to calculate the average IoU of the same obejct on consecutive frames from DanceTrack
"""
import numpy as np 
import os
import cv2

source_dir = "dancetrack_val"
#source_dir = "datasets/val"

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
    top_left = np.maximum(arr1[:, :2], arr2[:, :2]) # [[x, y]]
    bottom_right = np.minimum(arr1[:, 2:], arr2[:, 2:]) # [[x, y]]
    wh = bottom_right - top_left
    # clip: if boxes not overlap then make it zero
    intersection = wh[:, 0].clip(0) * wh[:, 1].clip(0)

    #union 
    union = area1 + area2 - intersection
    return intersection, union

def box_iou(arr1, arr2):
    # arr1[N, 4]
    # arr2[N, 4]
    # N = number of bounding boxes
    assert(arr1[:, 2:] > arr1[:, :2]).all()
    assert(arr2[:, 2:] > arr2[:, :2]).all()
    inter, union = _box_inter_union(arr1, arr2)
    iou = inter / union
    return iou


def consecutive_iou(annos):
    """
        calculate the IoU over bboxes on the consecutive frames
    """
    max_frame = int(annos[:, 0].max())
    min_frame = int(annos[:, 0].min())
    total_iou = 0 
    total_frequency = 0
    for find in range(min_frame, max_frame):
        anno_cur = annos[np.where(annos[:,0]==find)]
        anno_next = annos[np.where(annos[:,0]==find+1)]
        ids_cur = np.unique(anno_cur[:,1])
        ids_next = np.unique(anno_next[:,1])
        common_ids = np.intersect1d(ids_cur, ids_next)
        for tid in common_ids:
            cur_box = anno_cur[np.where(anno_cur[:,1]==tid)][:, 2:6]
            next_box = anno_next[np.where(anno_next[:,1]==tid)][:, 2:6]
            cur_box[:, 2:] += cur_box[:, :2]
            next_box[:, 2:] += next_box[:, :2]
            #print(cur_box.type, next_box)
            iou = box_iou(cur_box, next_box).item()
            total_iou += iou 
            total_frequency += 1
    return total_iou, total_frequency


if __name__ == "__main__":
    seqs = os.listdir(source_dir)
    #MOT
    #seqs = [f for f in os.listdir(source_dir) if 'FRCNN' in f]
    all_iou, all_freq = 0, 0
    all_switch, all_sw_freq = 0, 0
    for seq in seqs:
        if seq == ".DS_Store":
            continue
        anno_file = os.path.join(source_dir, seq, "gt/gt.txt")
        #MOT
        #anno_file = os.path.join(source_dir, seq, "gt/gt_val_half.txt")
        annos = np.loadtxt(anno_file, delimiter=",")
        seq_iou, seq_freq = consecutive_iou(annos)
        all_iou += seq_iou
        all_freq += seq_freq
    print("Average IoU on consecutive frames = {}".format(all_iou / all_freq))


#DanceTrack: 0.9088503662827935
#MOT17: 9460269575877778
#SportsMOT:0.80065329265497
#Slope-Track:0.8483

import matplotlib.pyplot as plt

# Example data
datasets = ['MOT17', 'DanceTrack', 'SportsMOT', 'Slope-Track']
iou_scores = [ 0.9460269575877778, 0.9088503662827935, 0.80065329265497, 0.8483]


sorted_indices = sorted(range(len(iou_scores)), key=lambda i: iou_scores[i], reverse=True)
datasets = [datasets[i] for i in sorted_indices]
iou_scores = [iou_scores[i] for i in sorted_indices]


custom_colors = ['firebrick', 'darkgoldenrod', 'darkviolet', 'teal' ]  # Example
markers = ['o', 's', '^', 'p' ]

# Create the plot
fig, ax = plt.subplots(figsize=(10, 8))

for i, (dataset, score) in enumerate(zip(datasets, iou_scores)):
    ax.hlines(y=dataset, xmin=0, xmax=score, color=custom_colors[i], linewidth=3, )
    ax.plot(score, dataset, markers[i], markersize=8, color=custom_colors[i])

ax.set_xlabel('Average IoU', fontsize=16)
#ax.set_title('Average IoU on Adjacent Frames across Datasets')
ax.set_xlim(0, 1.0)
ax.grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


