import argparse
import os
import os.path as osp
import numpy as np
import time
import cv2
import torch
import sys
sys.path.append('.')

from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracking_utils.timer import Timer

from tracker.Deep_EIoU import Deep_EIoU
from fast_reid.fast_reid_interfece import FastReIDInterface
import copy

from sahi import AutoDetectionModel
import cv2
import os
from sahi.predict import get_sliced_prediction, predict, get_prediction
import numpy as np

import configparser

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

def get_fps_from_seqinfo(seq_path):
    seqinfo_path = osp.join(seq_path, "seqinfo.ini")
    if not osp.exists(seqinfo_path):
        raise FileNotFoundError(f"seqinfo.ini not found in {seq_path}")

    config = configparser.ConfigParser()
    config.read(seqinfo_path)
    return int(config["Sequence"]["frameRate"])

def make_parser():
    parser = argparse.ArgumentParser("DeepEIoU Demo")
    parser.add_argument("--split", default="test", type=str, help="Choose dataset split: 'test' or 'val'")
    parser.add_argument("--dataset_root", default="slope_track",type=str, help="dataset root")
    parser.add_argument("--expn", "--experiment-name", type=str, default="slope_track")
    parser.add_argument(
        "--save_result",
        default=True,
        help="whether to save the inference result of image/video",
    )

    parser.add_argument("-c", "--ckpt", default= r"pretrained/slopetrack_yolov11.pt", type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.1, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=1088, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.1, type=float, help="lowest detection threshold valid for tracks")
    parser.add_argument("--new_track_thresh", default=0.7, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=60, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=1, help='filter out tiny boxes')
    parser.add_argument("--nms_thres", type=float, default=0.7, help='nms threshold')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")

    # reid args
    parser.add_argument("--with-reid", dest="with_reid", default=True, action="store_true", help="use Re-ID flag.")
    parser.add_argument("--fast_reid_config", dest="fast_reid_config", default=r"fast_reid/configs/SlopeTrack/sbs_S50.yml", type=str, help="reid config file path")
    parser.add_argument("--fast_reid_weights", dest="fast_reid_weights", default=r"pretrained/slopetrack_sbs_S50.pth", type=str, help="reid weight path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5, help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25, help='threshold for rejecting low appearance similarity reid matches')
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
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
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
        print(len(coco))

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


        id_feature = self.encoder.inference(img, np.copy(outputs))  # normalization and numpy included

        #outputs = np.concatenate((bboxes, scores.reshape(-1, 1)), axis=1)
        return outputs, img_info, id_feature




def image_demo(predictor, vis_folder, current_time, args):
    if osp.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]
    files.sort()
    tracker = Deep_EIoU(args, frame_rate=args.fps)
    timer = Timer()
    results = []

    for frame_id, img_path in enumerate(files, 1):
        outputs, img_info, id_feature = predictor.inference(img_path, timer)
        if outputs[0] is not None:
            online_targets = tracker.update(outputs, id_feature)
            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                tlwh = t.last_tlwh
                tid = t.track_id
                if tlwh[2] * tlwh[3] > args.min_box_area:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    # save results
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
            timer.toc()
            online_im = plot_tracking(
                img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time
            )
        else:
            timer.toc()
            online_im = img_info['raw_img']

        # result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        """if args.save_result:
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
        vis_folder = osp.join(output_dir, 'deepeiou')
        os.makedirs(vis_folder, exist_ok=True)

    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))


    ckpt_file = args.ckpt
    print(ckpt_file)
    logger.info("loading checkpoint")
    model = AutoDetectionModel.from_pretrained(
             model_type="yolov8",
             model_path=ckpt_file,
             confidence_threshold=args.conf,
             device=args.device,  # or 'cuda:0'
             image_size=args.tsize
    )
    logger.info("loaded checkpoint done.")


    predictor = Predictor(model, args.device, with_reid=args.with_reid, fast_reid_config=args.fast_reid_config, fast_reid_weights=args.fast_reid_weights)
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
