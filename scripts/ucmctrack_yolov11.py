from ultralytics import YOLO
import os,cv2
import argparse

from tracker.ucmc import UCMCTrack
from detector.mapper import Mapper
import numpy as np

from sahi import AutoDetectionModel
import cv2
import os
from sahi.predict import get_sliced_prediction, predict, get_prediction


# 定义一个Detection类，包含id,bb_left,bb_top,bb_width,bb_height,conf,det_class
class Detection:

    def __init__(self, id, bb_left = 0, bb_top = 0, bb_width = 0, bb_height = 0, conf = 0, det_class = 0):
        self.id = id
        self.bb_left = bb_left
        self.bb_top = bb_top
        self.bb_width = bb_width
        self.bb_height = bb_height
        self.conf = conf
        self.det_class = det_class
        self.track_id = 0
        self.y = np.zeros((2, 1))
        self.R = np.eye(4)


    def __str__(self):
        return 'd{}, bb_box:[{},{},{},{}], conf={:.2f}, class{}, uv:[{:.0f},{:.0f}], mapped to:[{:.1f},{:.1f}]'.format(
            self.id, self.bb_left, self.bb_top, self.bb_width, self.bb_height, self.conf, self.det_class,
            self.bb_left+self.bb_width/2,self.bb_top+self.bb_height,self.y[0,0],self.y[1,0])

    def __repr__(self):
        return self.__str__()


# Detector类，用于从Yolo检测器获取目标检测的结果
class Detector:
    def __init__(self):
        self.seq_length = 0
        self.gmc = None

    def load(self,cam_para_file):
        self.mapper = Mapper(cam_para_file,"SlopeTrack")
        
        model_path = "'pretrained/slopetrack_yolov11.pt'"
        self.model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=model_path,
            confidence_threshold=0.1,
            device="cuda:0",  # or 'cuda:0'
            ) 
        #self.model = YOLO('pretrained/yolov8x.pt')

    def get_dets(self, img):
        
        dets = []

        # 将帧从 BGR 转换为 RGB（因为 OpenCV 使用 BGR 格式）  
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  

        # 使用 RTDETR 进行推理  
        #results = self.model(frame,imgsz = 1088)

        
        
        results = get_sliced_prediction(
                  frame,
                  self.model,
                  slice_height=640,
                  slice_width=640,
                  overlap_height_ratio=0.1,
                  overlap_width_ratio=0.1,
            )

        coco = results.to_coco_annotations()

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

        det_id = 0
        for bbox, conf in zip(bbox_list, score_list):
            x1, y1, w, h = bbox
            det = Detection(det_id)
            det.bb_left = x1
            det.bb_top = y1
            det.bb_width = w
            det.bb_height = h
            det.conf = conf
            det.det_class = 0.0
            det.y,det.R = self.mapper.mapto([det.bb_left,det.bb_top,det.bb_width,det.bb_height])
            det_id += 1

            dets.append(det)

        return dets

    def cmc(self,x,y,w,h,frame_id):
        u,v = self.mapper.xy2uv(x,y)
        affine = self.gmc.get_affine(frame_id)
        M = affine[:,:2]
        T = np.zeros((2,1))
        T[0,0] = affine[0,2]
        T[1,0] = affine[1,2]

        p_center = np.array([[u],[v-h/2]])
        p_wh = np.array([[w],[h]])
        p_center = np.dot(M,p_center) + T
        p_wh = np.dot(M,p_wh)

        u = p_center[0,0]
        v = p_center[1,0]+p_wh[1,0]/2

        xy,_ = self.mapper.uv2xy(np.array([[u],[v]]),np.eye(2))

        return xy[0,0],xy[1,0] 

def main(args):
    dataset_path = os.path.join(args.dataset_root, args.split)
    sequences = sorted(os.listdir(dataset_path))

    for seq in sequences:
        seq_path = os.path.join(dataset_path, seq)
        img_dir = os.path.join(seq_path, "img1")
        seq_info_path = os.path.join(seq_path, "seqinfo.ini")



        cam_para_file = os.path.join("cam_para", f"{seq}.txt")

        # Parse seqinfo.ini for fps
        with open(seq_info_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if line.startswith("frameRate"):
                fps = float(line.strip().split('=')[1])

        # Get frame file list
        frame_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')])
        if not frame_files:
            print(f"No images found in {img_dir}")
            continue

        detector = Detector()
        detector.load(cam_para_file)

        tracker = UCMCTrack(args.a, args.a, args.wx, args.wy, args.vmax, args.cdt, fps, "SlopETrack", args.high_score,
                            False, None)

        results = []
        frame_id = 1
        for frame_name in frame_files:
            frame_path = os.path.join(img_dir, frame_name)
            frame_img = cv2.imread(frame_path)
            if frame_img is None:
                print(f"Warning: failed to read {frame_path}")
                continue

            dets = detector.get_dets(frame_img, args.conf_thresh, [1])
            tracker.update(dets, frame_id)

            for det in dets:
                if det.track_id > 0:
                    results.append(
                        f"{frame_id},{det.track_id},{det.bb_left:.2f},{det.bb_top:.2f},{det.bb_width:.2f},{det.bb_height:.2f},{det.conf:.2f},-1,-1,-1\n"
                    )
            frame_id += 1

        # Save results
        output_dir = '../yolo11'
        os.makedirs(output_dir, exist_ok=True)
        output_dir = os.path.join(output_dir, args.expn)
        os.makedirs(output_dir, exist_ok=True)

        vis_folder = os.path.join(output_dir, "ucmctrack")
        os.makedirs(vis_folder, exist_ok=True)

        res_file = os.path.join(vis_folder, f'{seq}.txt')
        with open(res_file, 'w') as f:
            f.writelines(results)
        print(f"Saved results to {res_file}")


parser = argparse.ArgumentParser(description='UCMCTrack parameters')
parser.add_argument("--split", default="test", type=str, help="Choose dataset split: 'test' or 'val'")
parser.add_argument("--dataset_root", default="slope_track", type=str, help="dataset root")
parser.add_argument("--expn", "--experiment-name", type=str, default="slope_track")
parser.add_argument('--wx', type=float, default=5, help='wx')
parser.add_argument('--wy', type=float, default=5, help='wy')
parser.add_argument('--vmax', type=float, default=10, help='vmax')
parser.add_argument('--a', type=float, default=100.0, help='assignment threshold')
parser.add_argument('--cdt', type=float, default=10.0, help='coasted deletion time')
parser.add_argument('--high_score', type=float, default=0.5, help='high score threshold')
parser.add_argument('--conf_thresh', type=float, default=0.1, help='detection confidence threshold')
args = parser.parse_args()

main(args)
