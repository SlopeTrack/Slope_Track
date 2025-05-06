import argparse
import os
import os.path as osp
import time
import cv2
import torch

from loguru import logger

from yolox.utils.visualize import plot_tracking, plot_tracking_detection
from trackers.ocsort_tracker.ocsort import OCSort
from trackers.hybrid_sort_tracker.hybrid_sort import Hybrid_Sort
from trackers.hybrid_sort_tracker.hybrid_sort_reid import Hybrid_Sort_ReID
from trackers.tracking_utils.timer import Timer
from fast_reid.fast_reid_interfece import FastReIDInterface
import copy

from sahi import AutoDetectionModel
import cv2
import os
from sahi.predict import get_sliced_prediction, predict, get_prediction
import numpy as np



IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

def get_fps_from_seqinfo(seq_path):
    seqinfo_path = osp.join(seq_path, "seqinfo.ini")
    if not osp.exists(seqinfo_path):
        raise FileNotFoundError(f"seqinfo.ini not found in {seq_path}")

def make_parser():
    parser = argparse.ArgumentParser("Hybrid-SORT parameters")
    parser.add_argument("--split", default="test", type=str, help="Choose dataset split: 'test' or 'val'")
    parser.add_argument("--dataset_root", default="slope_track", type=str, help="dataset root")
    parser.add_argument("--expn", "--experiment-name", type=str, default="slope_track")


    # distributed
    parser.add_argument( "--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--output_dir", type=str, default="evaldata/trackers/mot_challenge")
    parser.add_argument("--dist-url", default=None, type=str, help="url used to set up distributed training")
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("-d", "--devices", default=None, type=int, help="device for training")

    parser.add_argument("--local_rank", default=0, type=int, help="local rank for dist training")
    parser.add_argument( "--num_machines", default=1, type=int, help="num of node for training")
    parser.add_argument("--machine_rank", default=0, type=int, help="node rank for multi-node training")

    parser.add_argument(
        "--fp16", dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.",)
    parser.add_argument("--trt", dest="trt", default=False, action="store_true", help="Using TensorRT model for testing.",)
    parser.add_argument("--test", dest="test", default=False, action="store_true", help="Evaluating on test-dev set.",)
    parser.add_argument("--speed", dest="speed", default=False, action="store_true", help="speed test only.",)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER,)
    
    # det args
    parser.add_argument("-c", "--ckpt", default='pretrained/slopetrack_yolov11.pt', type=str, help="ckpt for eval")
    parser.add_argument("--conf", default=0.1, type=float, help="test conf")
    parser.add_argument("--nms", default=0.7, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")

    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.6, help="detection confidence threshold")
    parser.add_argument("--iou_thresh", type=float, default=0.15, help="the iou threshold in Sort for matching")
    parser.add_argument("--min_hits", type=int, default=3, help="min hits to create track in SORT")
    parser.add_argument("--inertia", type=float, default=0.05, help="the weight of VDC term in cost matrix")
    parser.add_argument("--deltat", type=int, default=3, help="time step difference to estimate direction")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
    parser.add_argument('--min-box-area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--gt-type", type=str, default="_val_half", help="suffix to find the gt annotation")
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    parser.add_argument("--public", action="store_true", help="use public detection")
    parser.add_argument('--asso', default="Height_Modulated_IoU", help="similarity function: iou/giou/diou/ciou/ctdis")
    parser.add_argument("--use_byte", dest="use_byte", default=True, action="store_true", help="use byte in tracking.")

    parser.add_argument("--TCM_first_step", default=True, action="store_true", help="use TCM in first step.")
    parser.add_argument("--TCM_byte_step", default=True, action="store_true", help="use TCM in byte step.")
    parser.add_argument("--TCM_first_step_weight", type=float, default=1.0, help="TCM first step weight")
    parser.add_argument("--TCM_byte_step_weight", type=float, default=1.0, help="TCM second step weight")
    parser.add_argument("--hybrid_sort_with_reid", default=False, action="store_true", help="use ReID model for Hybrid SORT.")

    # for fast reid
    parser.add_argument("--EG_weight_high_score", default=4.0, type=float, help="weight of appearance cost matrix when using EG")
    parser.add_argument("--EG_weight_low_score", default=4.4, type=float, help="weight of appearance cost matrix when using EG")
    parser.add_argument("--low_thresh", default=0.1, type=float, help="threshold of low score detections for BYTE")
    parser.add_argument("--high_score_matching_thresh", default=0.8, type=float, help="matching threshold for detections with high score")
    parser.add_argument("--low_score_matching_thresh", default=0.5, type=float, help="matching threshold for detections with low score")
    parser.add_argument("--alpha", default=0.8, type=float, help="momentum of embedding update")
    parser.add_argument("--with_fastreid", dest="with_fastreid", default=False, action="store_true", help="use FastReID flag.")
    parser.add_argument("--fast_reid_config", dest="fast_reid_config", default=r"fast_reid/configs/SlopeTrack/sbs_S50.yml", type=str, help="reid config file path")
    parser.add_argument("--fast_reid_weights", dest="fast_reid_weights", default=r"pretrained/slopetrack_sbs_S50.pth", type=str, help="reid weight path")
    parser.add_argument("--with_longterm_reid", dest="with_longterm_reid", default=True, action="store_true", help="use long-term reid features for association.")
    parser.add_argument("--longterm_reid_weight", default=0.20, type=float, help="weight of appearance cost matrix when using long term reid features in 1st stage association")
    parser.add_argument("--longterm_reid_weight_low", default=1.0, type=float, help="weight of appearance cost matrix when using long term reid features in 2nd stage association")
    parser.add_argument("--with_longterm_reid_correction", dest="with_longterm_reid_correction", default=True, action="store_true", help="use long-term reid features for association correction.")
    parser.add_argument("--longterm_reid_correction_thresh", default=1.0, type=float, help="threshold of correction when using long term reid features in 1st stage association")
    parser.add_argument("--longterm_reid_correction_thresh_low", default=1.0, type=float, help="threshold of correction when using long term reid features in 2nd stage association")
    parser.add_argument("--longterm_bank_length", type=int, default=30, help="max length of reid feat bank")
    parser.add_argument("--adapfs", dest="adapfs", default=False, action="store_true", help="Adaptive Feature Smoothing.")
    # ECC for CMC
    parser.add_argument("--ECC", dest="ECC", default=False, action="store_true", help="use ECC for CMC.")

    # for kitti/bdd100k inference with public detections
    parser.add_argument('--raw_results_path', type=str, default="exps/permatrack_kitti_test/",
        help="path to the raw tracking results from other tracks")
    parser.add_argument('--out_path', type=str, help="path to save output results")
    parser.add_argument("--dataset", type=str, default="mot17", help="kitti or bdd")
    parser.add_argument("--hp", action="store_true", help="use head padding to add the missing objects during \
            initializing the tracks (offline).")

    parser.add_argument(
        "--save_result",
        default=True,
        action="store_true",
        help="whether to save the inference result of image/video",
    )
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=4, help='filter out tiny boxes')
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    return parser

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class Predictor(object):
    def __init__(
        self,
        model,
        device=torch.device("cpu"),
        with_reid=False,
        fast_reid_config=None,
        fast_reid_weights=None,
    ):
        self.model = model
        self.device = device
        self.with_reid = with_reid
        if self.with_reid:
            self.fast_reid_config = fast_reid_config
            self.fast_reid_weights = fast_reid_weights
            self.encoder = FastReIDInterface(self.fast_reid_config, self.fast_reid_weights, 'cuda')

    def inference(self, img, timer):
        img_info = {"id": 0}
        #if isinstance(img, str):
        #    img_info["file_name"] = osp.basename(img)
        #   img = cv2.imread(img)
        #else:
        img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img
        
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = get_sliced_prediction(
            rgb_image,
            self.model,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.1,
            overlap_width_ratio=0.1
         )
        coco = result.to_coco_annotations()
        #print(len(coco))

        bbox_list = []
        score_list = []
        xyxy_boxes = []
        class_list = []
        key1_to_find = 'bbox'
        key2_to_find = 'score'
        key_to_find_category = 'category_name'
        for d in coco:
            if key_to_find_category in d:
                  value = d[key_to_find_category]
                  if value == 'person' and key1_to_find in d:
                      bbox_list.append(d[key1_to_find])
                  if value == 'person' and key2_to_find in d:
                      score_list.append(d[key2_to_find])

        for bbox, conf in zip(bbox_list,score_list):
            x1, y1, w, h = bbox
            x2 = x1 + w
            y2 = y1 + h
            class_ = 0
            xyxy_boxes.append([x1, y1, x2, y2, conf])
            class_list.append(class_)

        outputs = np.array(xyxy_boxes)

        #print(outputs)
        if self.with_reid:

                id_feature = self.encoder.inference(img, np.copy(outputs))  # normalization and numpy included

        #outputs = np.concatenate((bboxes, scores.reshape(-1, 1)), axis=1)
        if self.with_reid:
            return outputs, img_info, id_feature
        else:
            return outputs, img_info



def image_demo(predictor, vis_folder, current_time, args):
    if osp.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]
    files.sort()
    if not args.hybrid_sort_with_reid:
        tracker = Hybrid_Sort(args, det_thresh=args.track_thresh,
                              iou_threshold=args.iou_thresh,
                              asso_func=args.asso,
                              delta_t=args.deltat,
                              inertia=args.inertia,
                              use_byte=args.use_byte)
    else:
        tracker = Hybrid_Sort_ReID(args, det_thresh=args.track_thresh,
                                   iou_threshold=args.iou_thresh,
                                   asso_func=args.asso,
                                   delta_t=args.deltat,
                                   inertia=args.inertia)
    timer = Timer()
    results = []

    for frame_id, img_path in enumerate(files, 1):
        if args.with_fastreid:
            outputs, img_info, id_feature = predictor.inference(img_path, timer)
        else:
            outputs, img_info = predictor.inference(img_path, timer)
        if outputs[0] is not None:
            if args.with_fastreid:
                online_targets = tracker.update(outputs, [img_info['height'], img_info['width']], (img_info['height'], img_info['width']), id_feature=id_feature)
            else:
                online_targets = tracker.update(outputs, [img_info['height'], img_info['width']],
                                                (img_info['height'], img_info['width']))

            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                tid = t[4]
                score= t[5]
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    # save results
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{score:.2f},-1,-1,-1\n"
                    )
            timer.toc()
            online_im = plot_tracking(
                img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time
            )
        else:
            timer.toc()
            online_im = img_info['raw_img']

        """# result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if args.save_result:
            timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            save_folder = osp.join(vis_folder, timestamp)
            os.makedirs(save_folder, exist_ok=True)
            cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)"""

        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

    if args.save_result:
        seq_name = osp.basename(osp.dirname(args.path))
        res_file = osp.join(vis_folder, f"{seq_name}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")



def main(args):

    output_dir= '../yolo11'
    os.makedirs(output_dir, exist_ok=True)
    output_dir = osp.join(output_dir, args.expn)
    os.makedirs(output_dir, exist_ok=True)

    if args.save_result:
        if args.with_fastreid:
            vis_folder = osp.join(output_dir, "hybridsort_reid")
        else:
            vis_folder = osp.join(output_dir, "hybridsort")
        os.makedirs(vis_folder, exist_ok=True)

    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))


    ckpt_file = args.ckpt
    logger.info("loading checkpoint")
    model = AutoDetectionModel.from_pretrained(
             model_type="yolov8",
             model_path=ckpt_file,
             confidence_threshold=args.conf,
             device=args.device,  # or 'cuda:0'
             image_size=args.tsize
    )
    logger.info("loaded checkpoint done.")


    predictor = Predictor(model, args.device, with_reid=args.with_fastreid, fast_reid_config=args.fast_reid_config, fast_reid_weights=args.fast_reid_weights)
    split_path = osp.join(args.dataset_root, args.split)
    sequence_dirs = [d for d in os.listdir(split_path) if osp.isdir(osp.join(split_path, d))]

    for seq in sequence_dirs:
        seq_dir = osp.join(split_path, seq)
        seq_path = osp.join(seq_dir, "img1")

        logger.info(f"Processing sequence: {seq_path}")
        args.fps = get_fps_from_seqinfo(seq_dir)

        args.path = seq_path
        current_time = time.localtime()
        image_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()

    #args.ablation = False
    #args.mot20 = not args.fuse_score

    main(args)
