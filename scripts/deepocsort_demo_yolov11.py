import pdb
import os
import shutil
import time

import torch
import cv2
import numpy as np

#import dataset
import utils
#from external.adaptors import detector
from trackers import integrated_ocsort_embedding as tracker_module

from sahi import AutoDetectionModel
import cv2
import os
from sahi.predict import get_sliced_prediction, predict, get_prediction
import numpy as np
import os.path as osp

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names

def get_main_args():
    parser = tracker_module.args.make_parser()
    parser.add_argument("--dataset", type=str, default="slope_track")
    parser.add_argument("--result_folder", type=str, default="../yolo11")
    parser.add_argument("--split", default="test", choices=["test", "val"],
                        help="Choose dataset split: 'test' or 'val'")
    parser.add_argument("--test_dataset", action="store_true")
    parser.add_argument("--exp_name", type=str, default="slope_track")
    parser.add_argument("--min_box_area", type=float, default=1, help="filter out tiny boxes")
    parser.add_argument(
        "--aspect_ratio_thresh",
        type=float,
        default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value.",
    )
    parser.add_argument(
        "--post",
        action="store_true",
        help="run post-processing linear interpolation.",
    )
    parser.add_argument("--device", default="gpu", type=str, help="device to run our model, can either be cpu or gpu")
    parser.add_argument("--w_assoc_emb", type=float, default=0.75, help="Combine weight for emb cost")
    parser.add_argument(
        "--alpha_fixed_emb",
        type=float,
        default=0.95,
        help="Alpha fixed for EMA embedding",
    )
    parser.add_argument("--emb_off", action="store_true")
    parser.add_argument("--cmc_off", action="store_true")
    parser.add_argument("--aw_off", action="store_true")
    parser.add_argument("--aw_param", type=float, default=0.5)
    parser.add_argument("--new_kf_off", action="store_true")
    parser.add_argument("--grid_off", action="store_true")
    args = parser.parse_args()


    """if args.dataset == "slope_track":
        args.result_folder = os.path.join(args.result_folder, "SlopeTrack-val")
    else:
        print('Dataset not defined in this file')
    if args.test_dataset:
        args.result_folder.replace("-val", "-test")"""

    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")
    return args


def main():
    np.set_printoptions(suppress=True, precision=5)
    # Set dataset and detector
    args = get_main_args()

    if args.dataset == "slope_track":
        # Same model for test and validation
        detector_path= 'pretrained/slopetrack_yolov11.pt'
    else:
        raise RuntimeError("Need to update paths for detector for extra datasets.")
    model = AutoDetectionModel.from_pretrained(
             model_type="yolov8",
             model_path=detector_path,
             confidence_threshold=0.1,
             device=args.device,  # or 'cuda:0'
             image_size=1088)



    # Set up tracker
    oc_sort_args = dict(
        args=args,
        det_thresh=args.track_thresh,
        iou_threshold=args.iou_thresh,
        asso_func=args.asso,
        delta_t=args.deltat,
        inertia=args.inertia,
        w_association_emb=args.w_assoc_emb,
        alpha_fixed_emb=args.alpha_fixed_emb,
        embedding_off=args.emb_off,
        cmc_off=args.cmc_off,
        aw_off=args.aw_off,
        aw_param=args.aw_param,
        new_kf_off=args.new_kf_off,
        grid_off=args.grid_off,
    )
    tracker = tracker_module.ocsort.OCSort(**oc_sort_args)

    split_path = osp.join(args.dataset, args.split)
    sequence_dirs = [d for d in os.listdir(split_path) if osp.isdir(osp.join(split_path, d))]

    for seq in sequence_dirs:
        seq_dir = osp.join(split_path, seq)
        seq_path = osp.join(seq_dir, "img1")

        args.path = seq_path

        if os.path.isdir(args.path):
            files = get_image_list(args.path)
        else:
            files = [args.path]
        files.sort()

        seq_name = os.path.basename(os.path.dirname(args.path))

        results = {}
        frame_count = 0
        total_time = 0



        # See __getitem__ of dataset.MOTDataset
        for frame_id, img_path in enumerate(files, 1):
            # Frame info
            video_name = seq_name
            tag = f"{video_name}:{frame_id}"
            if video_name not in results:
                results[video_name] = []
            img = cv2.imread(img_path)

            # Initialize tracker on first frame of a new video
            print(f"Processing {video_name}:{frame_id}\r")
            if frame_id == 1:
                print(f"Initializing tracker for {video_name}")
                print(f"Time spent: {total_time:.3f}, FPS {frame_count / (total_time + 1e-9):.2f}")
                tracker.dump_cache()
                tracker = tracker_module.ocsort.OCSort(**oc_sort_args)

            start_time = time.time()


            # Nx5 of (x1, y1, x2, y2, conf), pass in tag for caching
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = get_sliced_prediction(
                             rgb_image,
                             model,
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

            pred = np.array(xyxy_boxes)
            if pred is None:
                continue
                    # Nx5 of (x1, y1, x2, y2, ID)
            targets = tracker.update(pred, img, img, tag)
            tlwhs, ids = utils.filter_targets(targets, args.aspect_ratio_thresh, args.min_box_area)

            total_time += time.time() - start_time
            frame_id += 1
            frame_count += 1

            results[video_name].append((frame_id, tlwhs, ids))



    print(f"Time spent: {total_time:.3f}, FPS {frame_count / (total_time + 1e-9):.2f}")
    # Save detector results
    #det.dump_cache()
    tracker.dump_cache()

    # Save for all sequences
    folder = os.path.join(args.result_folder, args.exp_name, "deepocsort")
    os.makedirs(folder, exist_ok=True)
    for name, res in results.items():
        result_filename = os.path.join(folder, f"{name}.txt")
        utils.write_results_no_score(result_filename, res)
    print(f"Finished, results saved to {folder}")
    if args.post:
        post_folder = os.path.join(args.result_folder, args.exp_name , "deepocsort_post")
        pre_folder = os.path.join(args.result_folder, args.exp_name, "deepocsort")
        if os.path.exists(post_folder):
            print(f"Overwriting previous results in {post_folder}")
            shutil.rmtree(post_folder)
        shutil.copytree(pre_folder, post_folder)
        post_folder_data = os.path.join(post_folder, "data")
        utils.dti(post_folder_data, post_folder_data)
        print(f"Linear interpolation post-processing applied, saved to {post_folder_data}.")


def draw(name, pred, i):
    pred = pred.cpu().numpy()
    name = os.path.join("data/mot/train", name)
    img = cv2.imread(name)
    for s in pred:
        p = np.round(s[:4]).astype(np.int32)
        cv2.rectangle(img, (p[0], p[1]), (p[2], p[3]), (255, 0, 0), 3)
    for s in pred:
        p = np.round(s[:4]).astype(np.int32)
        cv2.putText(
            img,
            str(int(round(s[4], 2) * 100)),
            (p[0] + 20, p[1] + 20),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (0, 0, 255),
            thickness=3,
        )
    cv2.imwrite(f"debug/{i}.png", img)


if __name__ == "__main__":
    main()
