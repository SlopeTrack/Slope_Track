import os
import numpy as np
from spatial import Tracker

import argparse

"""Script to run all the spatial-based methods on Slope-Track"""

def main(args):
    for i in args.association_method:

        val_pred = os.path.join(args.save_dir, i, 'data')
        if not os.path.exists(val_pred):
            os.makedirs(val_pred)

        val_seqs = [f for f in sorted(os.listdir(args.dataset_dir)) if f.endswith('.txt')]
        for video_name in val_seqs:
            print(f'Processing association method {i} on sequence {video_name}')
            det_results = {}
            with open(os.path.join(args.dataset_dir, video_name), 'r') as f:
                for line in f.readlines():
                    linelist = line.split(',')
                    img_id = linelist[0]
                    bbox = [float(linelist[1]),
                            float(linelist[2]),
                            float(linelist[1]) + float(linelist[3]),
                            float(linelist[2]) + float(linelist[4]),
                            float(linelist[5])]
                    if int(img_id) in det_results:
                        det_results[int(img_id)].append(bbox)
                    else:
                        det_results[int(img_id)] = list()
                        det_results[int(img_id)].append(bbox)
            f.close()

            results = []
            tracker = Tracker(det_thresh=args.det_thresh, association_method=i)
            for frame_id in sorted(det_results.keys()):
                det = det_results[frame_id]
                det = np.array(det)
                online_targets = tracker.update(det)
                online_tlwhs = []
                online_ids = []
                for t in online_targets:
                    tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                    tid = t[4]
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                results.append((frame_id, online_tlwhs, online_ids))

            save_format = '{frame},{id},{x1},{y1},{w},{h},-1,-1,-1,-1\n'
            filename = os.path.join(val_pred, video_name)
            with open(filename, 'w') as f:
                for frame_id, tlwhs, track_ids in results:
                    for tlwh, track_id in zip(tlwhs, track_ids):
                        if track_id < 0:
                            continue
                        x1, y1, w, h = tlwh
                        line = save_format.format(frame=frame_id, id=int(track_id), x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1))
                        f.write(line)
            f.close()

parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('--dataset_dir', type=str, default = "test_detections", help='dataset location')
parser.add_argument('--save_dir', type=str, default = "association_analysis", help='location to save results')
parser.add_argument('--association_method', nargs='+', default=['iou','diou', 'eiou', 'ciou','giou','hmiou','byte' ], help='association method(can be multiple)')
parser.add_argument('--det_thresh', type=float, default=0.3)
args = parser.parse_args()

main(args)