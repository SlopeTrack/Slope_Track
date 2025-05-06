import argparse
import os
import os.path as osp
import time
import cv2
import torch

from loguru import logger

from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer

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
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument("--split", default="test", choices=["test", "val"],
                        help="Choose dataset split: 'test' or 'val'")
    parser.add_argument("--dataset_root", type=str, default="slope_track", help="dataset root")
    parser.add_argument("--expn", "--experiment-name", default="slope_track", type=str)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--save_result",
        default=True,
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    parser.add_argument("-c", "--ckpt", default='pretrained/slopetrack_yolov11.pt', type=str, help="ckpt for eval")
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
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=1, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
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
    ):
        self.model = model
        self.device = device

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

        for bbox in bbox_list:
            x1, y1, w, h = bbox
            x2 = x1 + w
            y2 = y1 + h
            class_ = 0
            xyxy_boxes.append([x1, y1, x2, y2])
            class_list.append(class_)

        bboxes = np.array(xyxy_boxes)
        scores = np.array(score_list)
        outputs = np.concatenate((bboxes, scores.reshape(-1, 1)), axis=1)



        return outputs, img_info


def image_demo(predictor, vis_folder, current_time, args):
    if osp.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]
    files.sort()
    tracker = BYTETracker(args, frame_rate=args.fps)
    timer = Timer()
    results = []

    for frame_id, img_path in enumerate(files, 1):
        outputs, img_info = predictor.inference(img_path, timer)
        if outputs[0] is not None:
            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], [img_info['height'], img_info['width']])
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
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
        vis_folder = osp.join(output_dir, "bytetrack")
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


    predictor = Predictor(model, args.device)
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

    main(args)
