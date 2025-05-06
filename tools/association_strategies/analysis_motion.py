import os
import numpy as np
from spatial import Tracker

"""Script to run all the Kalman Filter methods on Slope-Track"""


import argparse

def main(args):
    for i in args.motion_method:

        val_pred = os.path.join(args.save_dir, i, 'data')
        if not os.path.exists(val_pred):
            os.makedirs(val_pred)

        val_seqs = [f for f in sorted(os.listdir(args.dataset_dir)) if f.endswith('.txt')]
        for video_name in val_seqs:
            print(f'Processing motion method {i} on sequence {video_name}')
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
            if i =='hybrid':
                from hybrid_motion import Hybrid_Sort
                hybrid_parser = argparse.ArgumentParser(description='Process some arguments.')
                hybrid_parser.add_argument("--track_thresh", type=float, default=0.3, help="detection confidence threshold")
                hybrid_parser.add_argument("--iou_thresh", type=float, default=0.3,
                                    help="the iou threshold in Sort for matching")
                hybrid_parser.add_argument("--min_hits", type=int, default=3, help="min hits to create track in SORT")
                hybrid_parser.add_argument("--inertia", type=float, default=0.2, help="the weight of VDC term in cost matrix")
                hybrid_parser.add_argument("--deltat", type=int, default=3, help="time step difference to estimate direction")
                hybrid_parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
                hybrid_parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
                hybrid_parser.add_argument('--asso', default="iou", help="similarity function: iou/giou/diou/ciou/ctdis")

                hybrid_parser.add_argument("--TCM_first_step", default=True, action="store_true",
                                    help="use TCM in first step.")
                hybrid_parser.add_argument("--TCM_first_step_weight", type=float, default=1.5, help="TCM first step weight")
                hybrid_args = hybrid_parser.parse_args()
                tracker = Hybrid_Sort(hybrid_args, det_thresh=hybrid_args.track_thresh,
                                        iou_threshold=hybrid_args.iou_thresh,
                                        asso_func=hybrid_args.asso,
                                        delta_t=hybrid_args.deltat,
                                        inertia=hybrid_args.inertia)
            elif i=='oc':
                from oc_motion import OCSort
                oc_parser = argparse.ArgumentParser(description='Process some arguments.')
                oc_parser.add_argument("--track_thresh", type=float, default=0.3, help="detection confidence threshold")
                oc_parser.add_argument("--iou_thresh", type=float, default=0.3,
                                    help="the iou threshold in Sort for matching")
                oc_parser.add_argument("--min_hits", type=int, default=3, help="min hits to create track in SORT")
                oc_parser.add_argument("--inertia", type=float, default=0.2, help="the weight of VDC term in cost matrix")
                oc_parser.add_argument("--deltat", type=int, default=3, help="time step difference to estimate direction")
                oc_parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
                oc_parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
                oc_args = oc_parser.parse_args()
                tracker = OCSort(det_thresh=oc_args.track_thresh, iou_threshold=oc_args.iou_thresh)
            elif i=='bot':
                from bot_motion import BoTSORT
                bot_parser = argparse.ArgumentParser(description='Process some arguments.')
                bot_parser.add_argument("--track_high_thresh", type=float, default=0.3,
                                    help="tracking confidence threshold")
                bot_parser.add_argument("--track_low_thresh", default=0.1, type=float, help="lowest detection threshold")
                bot_parser.add_argument("--new_track_thresh", default=0.7, type=float, help="new track thresh")
                bot_parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
                bot_parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
                bot_args = bot_parser.parse_args()
                tracker = BoTSORT(bot_args, frame_rate=30)
            elif i =='sort':
                from sort import Sort
                tracker = Sort(det_thresh=0.3)
            for frame_id in sorted(det_results.keys()):
                det = det_results[frame_id]
                det = np.array(det)
                online_targets = tracker.update(det)
                online_tlwhs = []
                online_ids = []
                for t in online_targets:
                    try:
                        tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                        tid = t[4]
                    except:
                        tlwh = t.tlwh
                        tid = t.track_id
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
parser.add_argument('--motion_method', nargs='+', default=['oc', 'hybrid', 'bot', 'sort'], help='motion method(can be multiple)')
parser.add_argument('--det_thresh', type=float, default=0.3)
args = parser.parse_args()

main(args)